"""An additive, explicitly gated SL extension of the original VL-DistMult.

The author DistMult scorer, 768-D fusion, and relation representation are
unchanged. The residual has its own shared entity projection and separate
relation projection. Its signed zero-initialized gate is allowed to reject
or reverse the proposed geometric compatibility; this is not pure SL KGE.

Calibration MUST receive training-only positive and corrupted triples from
the caller, never validation/test triples. Mean/std are frozen buffers after
one calibration. SL uses the shared Gregory-12 symmetric local discrepancy,
not an exact global geodesic distance. The Euclidean residual is its matched
projection/capacity control. No model or calibration code samples data.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from sl_manifold.core import (
    algebra_exp,
    coordinates_to_algebra,
    orthonormal_sl_basis,
    summarize_sl_diagnostics,
    symmetric_distance,
)
from vlkge.models.distmult import DistMult


class DistMultSL(DistMult):
    """Original score plus a bounded train-calibrated geometric residual.

    ``enabled=False`` directly delegates to the original forward, even when
    uncalibrated. The original entity/relation representation APIs still
    return 768 dimensions. Call ``calibrate(triples)`` exactly once before
    an enabled forward; the supplied tensor has shape ``[N,3]`` (h,r,t).
    """

    CALIBRATION_SCALE_FLOOR_MULTIPLIER = 4.0
    CALIBRATION_PROBE_CAPACITY = 256

    def __init__(self, *, geometry="sl8", enabled=True, residual_weight=1.0,
                 coordinate_scale=0.1, chart_radius=0.5, score_chunk=1024,
                 checkpoint_blocks=True, **author_distmult_kwargs):
        if geometry not in ("sl8", "euclidean"):
            raise ValueError("geometry must be 'sl8' or 'euclidean'")
        for name, value in (("coordinate_scale", coordinate_scale),
                            ("chart_radius", chart_radius)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(residual_weight) or residual_weight < 0:
            raise ValueError("residual_weight must be finite and nonnegative")
        if isinstance(score_chunk, bool) or not isinstance(score_chunk, int) or score_chunk < 1:
            raise ValueError("score_chunk must be a positive integer")
        if not isinstance(enabled, bool) or not isinstance(checkpoint_blocks, bool):
            raise ValueError("enabled and checkpoint_blocks must be bool")
        scale_floor = max(self.CALIBRATION_SCALE_FLOOR_MULTIPLIER * chart_radius * chart_radius, 1e-6)
        if not math.isfinite(scale_floor) or scale_floor > torch.finfo(torch.float32).max:
            raise ValueError("chart_radius produces a nonfinite float32 calibration floor")

        # Do not override reset_parameters: this invokes the original
        # DistMult constructor and its original initialization unchanged.
        super().__init__(**author_distmult_kwargs)
        if self.embedding_dim != 768 or self.fusion_mode != "average":
            raise ValueError("controlled residual requires author 768-D average fusion")
        self.geometry = geometry
        self.enabled = enabled
        self.residual_weight = float(residual_weight)
        self.coordinate_scale = float(coordinate_scale)
        self.chart_radius = float(chart_radius)
        self.score_chunk = score_chunk
        self.checkpoint_blocks = checkpoint_blocks
        # Added CPU initialization neither modifies the original weights nor
        # advances the caller's RNG stream (including subsequent negatives).
        with torch.random.fork_rng(devices=[]):
            self.entity_sl_projection = nn.Linear(768, 63, bias=False)
            self.relation_sl_projection = nn.Linear(768, 63, bias=False)
            nn.init.xavier_uniform_(self.entity_sl_projection.weight)
            nn.init.xavier_uniform_(self.relation_sl_projection.weight)
        self.raw_gate = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("calibration_center", torch.tensor(0.0))
        self.register_buffer("calibration_scale", torch.tensor(1.0))
        self.register_buffer("calibration_empirical_std", torch.tensor(0.0))
        self.register_buffer("calibration_scale_floor", torch.tensor(scale_floor))
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("calibration_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibration_probe_triples", torch.zeros(
            self.CALIBRATION_PROBE_CAPACITY, 3, dtype=torch.long))
        self.register_buffer("calibration_probe_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("sl_basis", orthonormal_sl_basis(8), persistent=False)
        self._evaluation_cache = None
        self._evaluation_cache_signature = None

    @property
    def gate(self):
        """Signed learned multiplier before the fixed residual_weight."""
        return self.raw_gate.tanh()

    def clear_evaluation_cache(self):
        self._evaluation_cache = None
        self._evaluation_cache_signature = None

    def train(self, mode=True):
        self.clear_evaluation_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_evaluation_cache()
        result = super()._apply(fn)
        # These upstream lookup tensors are ordinary attributes, not buffers.
        for name in ("visual_id_lookup", "textual_id_lookup"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, fn(value))
        self.device = self.entity_sl_projection.weight.device
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_evaluation_cache()
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.device = self.entity_sl_projection.weight.device
        return result

    @staticmethod
    def _require_finite(value, description):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite {description} in DistMultSL")

    def _bounded(self, coordinates):
        scaled = coordinates * self.coordinate_scale
        return scaled / torch.sqrt(1 + scaled.square().sum(dim=-1, keepdim=True)
                                   / self.chart_radius ** 2)

    def _map(self, coordinates):
        if self.geometry == "sl8":
            return algebra_exp(coordinates_to_algebra(coordinates, 8, basis=self.sl_basis))
        return coordinates

    def get_geometry_entity_representations(self, entity_ids):
        fused = self.get_entity_representations(entity_ids)
        return self._map(self._bounded(self.entity_sl_projection(fused)))

    def get_geometry_relation_representations(self, relation_ids):
        relation = self.get_relation_representations(relation_ids)
        return self._map(self._bounded(self.relation_sl_projection(relation)))

    def _cache_signature(self):
        return (self.geometry, self.coordinate_scale, self.chart_radius,
                tuple((id(p), p._version) for p in self.parameters()),
                tuple((id(b), b._version) for b in self.buffers()))

    def _cached_entities(self):
        signature = self._cache_signature()
        if self._evaluation_cache is None or signature != self._evaluation_cache_signature:
            blocks = []
            for start in range(0, self.num_entities, self.score_chunk):
                ids = torch.arange(start, min(start + self.score_chunk, self.num_entities),
                                   device=self.entity_sl_projection.weight.device)
                blocks.append(self.get_geometry_entity_representations(ids))
            self._evaluation_cache = torch.cat(blocks)
            self._evaluation_cache_signature = signature
        return self._evaluation_cache

    def _distance_block(self, head, relation, tail):
        if self.geometry == "sl8":
            return symmetric_distance(relation @ head, tail, terms=12,
                                      jitter=1e-7, trace_project=True).square()
        return (head + relation - tail).square().sum(dim=-1)

    def geometric_discrepancy(self, head, relation, tail):
        """Differentiable residual discrepancy, without the author score."""
        if head.ndim != 1 or head.shape != relation.shape or head.shape != tail.shape:
            raise ValueError("head, relation and tail must be matching 1-D ID tensors")
        if isinstance(self.score_chunk, bool) or not isinstance(self.score_chunk, int) or self.score_chunk < 1:
            raise ValueError("score_chunk must be a positive integer")
        count = head.numel()
        if count == 0:
            return self.entity_sl_projection.weight.new_empty((0,))
        if not self.training and not torch.is_grad_enabled():
            entity_repr = self._cached_entities()
            head_map, tail_map = head, tail
        else:
            entity_ids, entity_map = torch.unique(torch.cat((head, tail)), return_inverse=True)
            entity_repr = self.get_geometry_entity_representations(entity_ids)
            head_map, tail_map = entity_map[:count], entity_map[count:]
        relation_ids, relation_map = torch.unique(relation, return_inverse=True)
        relation_repr = self.get_geometry_relation_representations(relation_ids)
        values = []
        for start in range(0, count, self.score_chunk):
            stop = start + self.score_chunk
            inputs = (entity_repr[head_map[start:stop]],
                      relation_repr[relation_map[start:stop]],
                      entity_repr[tail_map[start:stop]])
            if self.checkpoint_blocks and torch.is_grad_enabled():
                distance = checkpoint(self._distance_block, *inputs, use_reentrant=False,
                                      preserve_rng_state=False)
            else:
                distance = self._distance_block(*inputs)
            values.append(distance)
        result = torch.cat(values)
        self._require_finite(result, "geometric discrepancy")
        return result

    @torch.no_grad()
    def calibrate(self, triples):
        """Freeze mean/std from supplied TRAIN-only positive/corrupted triples.

        No random draws, mode changes or gradients occur here. The caller is
        responsible for split provenance and balanced positive/negative data.
        Calling twice is forbidden so a run cannot silently recalibrate.
        """
        if bool(self.calibrated):
            raise RuntimeError("DistMultSL is already calibrated; reuse its frozen buffers")
        if not isinstance(triples, torch.Tensor) or triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("calibration triples must be a Tensor[N,3]")
        if triples.shape[0] == 0 or triples.dtype != torch.long:
            raise ValueError("calibration needs nonempty int64 triples")
        triples = triples.to(device=self.entity_sl_projection.weight.device)
        delta = self.geometric_discrepancy(triples[:, 0], triples[:, 1], triples[:, 2])
        center = delta.mean()
        empirical_std = delta.std(unbiased=False)
        self._require_finite(empirical_std, "calibration empirical std")
        if not bool(empirical_std > 0):
            raise ValueError("calibration discrepancies have zero empirical variance")
        # Frozen initial std alone is too small after the author's lr=.1
        # Adagrad updates. The fixed radius-based floor prevents this avoidable
        # local normalization from immediately saturating the bounded score.
        scale = torch.maximum(empirical_std, self.calibration_scale_floor)
        self._require_finite(center, "calibration center")
        self._require_finite(scale, "calibration scale")
        self.calibration_center.copy_(center)
        self.calibration_scale.copy_(scale)
        self.calibration_empirical_std.copy_(empirical_std)
        self.calibration_count.fill_(triples.shape[0])
        probe_count = min(self.CALIBRATION_PROBE_CAPACITY, triples.shape[0])
        probe_indices = torch.linspace(0, triples.shape[0] - 1, probe_count,
                                       device=triples.device).round().long()
        self.calibration_probe_triples.zero_()
        self.calibration_probe_triples[:probe_count].copy_(triples[probe_indices])
        self.calibration_probe_count.fill_(probe_count)
        self.calibrated.fill_(True)
        self.clear_evaluation_cache()
        return {
            "calibrated": True, "count": int(triples.shape[0]),
            "center": float(center), "scale": float(scale),
            "empirical_std": float(empirical_std),
            "scale_floor": float(self.calibration_scale_floor),
            "scale_floor_multiplier": self.CALIBRATION_SCALE_FLOOR_MULTIPLIER,
            "probe_count": probe_count,
            "delta_min": float(delta.min()), "delta_max": float(delta.max()),
            "source_contract": "caller_supplied_train_only_positive_and_corrupted_triples",
            "frozen": True,
        }

    def forward(self, head, relation, tail):
        # The hard-disable path intentionally does no extra encoding, checks,
        # cache work, calibration, or arithmetic on the author score.
        if not self.enabled:
            return super().forward(head, relation, tail)
        if not bool(self.calibrated):
            raise RuntimeError("calibrate DistMultSL with training-only triples before enabling it")
        self._require_finite(self.raw_gate, "raw gate")
        self._require_finite(self.calibration_center, "calibration center")
        if not bool(torch.isfinite(self.calibration_scale)) or not bool(self.calibration_scale > 0):
            raise FloatingPointError("DistMultSL calibration scale must be finite and positive")
        author_score = super().forward(head, relation, tail)
        self._require_finite(author_score, "author score")
        delta = self.geometric_discrepancy(head, relation, tail)
        compatibility = torch.tanh((self.calibration_center - delta) / self.calibration_scale)
        residual = self.residual_weight * self.gate * compatibility
        self._require_finite(residual, "geometric residual")
        result = author_score + residual
        self._require_finite(result, "combined output")
        return result

    @torch.no_grad()
    def compatibility_diagnostics(self):
        """Monitor the frozen, caller-supplied TRAIN-only calibration subset."""
        if not bool(self.calibrated) or int(self.calibration_probe_count) < 1:
            return {"available": False, "source": "train_only_calibration_probe"}
        triples = self.calibration_probe_triples[:int(self.calibration_probe_count)]
        delta = self.geometric_discrepancy(triples[:, 0], triples[:, 1], triples[:, 2])
        q = torch.tanh((self.calibration_center - delta) / self.calibration_scale)
        self._require_finite(q, "training probe compatibility")
        return {
            "available": True, "source": "train_only_calibration_probe",
            "count": len(triples), "q_mean": float(q.mean()),
            "q_std": float(q.std(unbiased=False)),
            "saturation_abs_q_gt_0_99": float((q.abs() > 0.99).float().mean()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std(unbiased=False)),
            "delta_min": float(delta.min()), "delta_max": float(delta.max()),
        }

    @torch.no_grad()
    def diagnostics(self, sample_size=256):
        """Deterministic spread sample; not certification of all triples."""
        count = min(max(int(sample_size), 1), self.num_entities)
        ids = torch.linspace(0, self.num_entities - 1, count,
                             device=self.entity_sl_projection.weight.device).round().long()
        tails = (ids * 17 + 1) % self.num_entities
        relations = ids % self.num_relations
        head = self.get_geometry_entity_representations(ids)
        tail = self.get_geometry_entity_representations(tails)
        relation = self.get_geometry_relation_representations(relations)
        self._require_finite(self.raw_gate, "diagnostic raw gate")
        self._require_finite(head, "diagnostic entity representation")
        self._require_finite(tail, "diagnostic tail representation")
        self._require_finite(relation, "diagnostic relation representation")
        output = {
            "geometry": self.geometry, "enabled": self.enabled,
            "raw_gate": float(self.raw_gate), "gate": float(self.gate),
            "residual_weight": self.residual_weight,
            "effective_gate": float(self.gate) * self.residual_weight,
            "coordinate_scale": self.coordinate_scale, "chart_radius": self.chart_radius,
            "calibrated": bool(self.calibrated),
            "calibration_center": float(self.calibration_center),
            "calibration_scale": float(self.calibration_scale),
            "calibration_empirical_std": float(self.calibration_empirical_std),
            "calibration_scale_floor": float(self.calibration_scale_floor),
            "calibration_scale_floor_multiplier": self.CALIBRATION_SCALE_FLOOR_MULTIPLIER,
            "calibration_count": int(self.calibration_count),
            "compatibility_on_train_probe": self.compatibility_diagnostics(),
            "sample_size": count,
            "diagnostic_scope": "deterministic_spread_entity_sample_not_all_triples",
            "parameters_finite": all(bool(torch.isfinite(p).all()) for p in self.parameters()),
        }
        if self.geometry == "sl8":
            transformed = relation @ head
            relatives = torch.cat((torch.linalg.solve(transformed, tail),
                                   torch.linalg.solve(tail, transformed)))
            output["entities"] = summarize_sl_diagnostics(head, gregory_terms=12)
            output["relations"] = summarize_sl_diagnostics(relation, gregory_terms=12)
            output["relative_score_matrices"] = summarize_sl_diagnostics(relatives, gregory_terms=12)
        return output
