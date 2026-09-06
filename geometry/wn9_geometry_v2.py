"""WN9 matched geometry heads: offset - exp(log_alpha) * LINEAR distance.

Author frozen visual/textual features and 768-D fusion -> common 63-D
projection. No DistMult residual, per-entity biases, relation diagonals or
tanh score compression. Relation-left-action Poincare d_H/2 matches the
Euclidean/SL tangent-coordinate limit. SL uses a checked principal-relative-
log Frobenius discrepancy, not a globally defined Riemannian geodesic.
"""

import math
import torch

from sl_manifold.core import algebra_exp, coordinates_to_algebra, symmetric_distance
from .models import VLGeometry, mobius_add, poincare_exp0
from .matrix_log import quadrature_rule, checked_symmetric_log_distance, log_pair_diagnostics
from .gregory_diagnostics import gregory_pair_diagnostics


class WN9GeometryV2(VLGeometry):
    GEOMETRIES = ("euclidean", "hyperbolic", "sl8")

    def __init__(self, *, geometry="sl8", coordinate_scale=1.0, chart_radius=1.5,
                 initial_logit_scale=1.0, initial_offset=0.0, score_chunk=1024,
                 checkpoint_blocks=True, log_order=16, log_backend="gregory12", **author_base_kwargs):
        if geometry not in self.GEOMETRIES or log_order not in (16, 32):
            raise ValueError("unknown geometry or quadrature order (must be16/32)")
        if log_backend not in ("gregory12", "gauss_legendre"):
            raise ValueError("log_backend must be gregory12 or gauss_legendre")
        if any(not math.isfinite(v) or v <= 0 for v in (coordinate_scale, chart_radius)):
            raise ValueError("coordinate_scale/chart_radius must be finite positive")
        if not math.isfinite(initial_offset):
            raise ValueError("initial_offset must be finite")
        # Identical no-diagonal parameter allocation/RNG stream for all three.
        super().__init__(geometry="euclidean", coordinate_scale=coordinate_scale,
                         chart_radius=chart_radius, initial_logit_scale=initial_logit_scale,
                         initial_offset=initial_offset, score_chunk=score_chunk,
                         checkpoint_blocks=checkpoint_blocks, **author_base_kwargs)
        self.geometry, self.log_order = geometry, log_order
        self.log_backend = log_backend
        device, dtype = self.projection.weight.device, self.projection.weight.dtype
        nodes, weights = quadrature_rule(log_order, dtype=dtype, device=device)
        self.register_buffer("quadrature_nodes", nodes, persistent=False)
        self.register_buffer("quadrature_weights", weights, persistent=False)
        self.register_buffer("geometry_initialized", torch.tensor(False, device=device))
        self.register_buffer("initialization_target_norm", torch.tensor(0.0, device=device))
        self.register_buffer("initialization_train_entity_count", torch.tensor(0, dtype=torch.long, device=device))
        self.clear_evaluation_cache()

    def _manifold_map(self, coordinates):
        if self.geometry == "sl8":
            return algebra_exp(coordinates_to_algebra(coordinates, 8, basis=self.sl_basis))
        if self.geometry == "hyperbolic":
            return poincare_exp0(coordinates)
        return coordinates

    @torch.no_grad()
    def initialize_geometry(self, train_entity_ids, target_norm=0.5):
        """One train-only rescaling: entity median=every relation norm=target.

        Caller supplies training entity IDs; random relation directions are
        retained. No random draws/held-out calibration. chart_radius is an
        upper bound, never the requested initial magnitude.
        """
        if bool(self.geometry_initialized):
            raise RuntimeError("geometry already initialized")
        if not math.isfinite(target_norm) or not 0 < target_norm < self.chart_radius:
            raise ValueError("target_norm must be finite positive and below radius")
        if not isinstance(train_entity_ids, torch.Tensor) or train_entity_ids.ndim != 1 or train_entity_ids.dtype != torch.long or len(train_entity_ids) == 0:
            raise ValueError("nonempty 1-D int64 training IDs required")
        ids = train_entity_ids.to(self.projection.weight.device).unique()
        if bool((ids < 0).any() or (ids >= self.num_entities).any()):
            raise ValueError("training entity ID outside table")
        raw_norms = [(self.projection(self.get_entity_representations(block)) * self.coordinate_scale).norm(dim=-1)
                     for block in ids.split(self.score_chunk)]
        median = torch.cat(raw_norms).median()
        relations = self.relation_embeddings.weight
        relation_norms = relations.norm(dim=-1, keepdim=True)
        if not bool(torch.isfinite(median)) or not bool(median > 0) or not bool(torch.isfinite(relation_norms).all()) or bool((relation_norms == 0).any()):
            raise ValueError("zero/nonfinite initialization coordinate norms")
        target_raw = target_norm / math.sqrt(1 - (target_norm / self.chart_radius) ** 2)
        multiplier = target_raw / median
        self.projection.weight.mul_(multiplier)
        relations.mul_((target_raw / self.coordinate_scale) / relation_norms)
        self.geometry_initialized.fill_(True)
        self.initialization_target_norm.fill_(target_norm)
        self.initialization_train_entity_count.fill_(len(ids))
        self.clear_evaluation_cache()
        entity_norms = [self.bound_coordinates(self.projection(self.get_entity_representations(block))).norm(dim=-1)
                        for block in ids.split(self.score_chunk)]
        relation_norms = self.bound_coordinates(relations).norm(dim=-1)
        return {"initialized": True, "source": "caller_supplied_training_entity_ids_only",
                "training_entity_count": len(ids), "target_bounded_coordinate_norm": target_norm,
                "train_entity_bounded_norm_median": float(torch.cat(entity_norms).median()),
                "relation_bounded_norm_mean": float(relation_norms.mean()),
                "relation_bounded_norm_min": float(relation_norms.min()),
                "relation_bounded_norm_max": float(relation_norms.max()),
                "projection_multiplier": float(multiplier), "chart_radius": self.chart_radius,
                "coordinate_scale": self.coordinate_scale, "rng_consumed": False}

    def _distance_block(self, head_coord, head, relation, diagonal, tail):
        if self.geometry == "sl8":
            if self.log_backend == "gregory12":
                return symmetric_distance(relation @ head, tail, terms=12,
                                          jitter=1e-7, trace_project=True)
            return checked_symmetric_log_distance(relation @ head, tail, order=self.log_order,
                                                   nodes=self.quadrature_nodes, weights=self.quadrature_weights)
        if self.geometry == "hyperbolic":
            transformed = mobius_add(relation, head)
            delta = mobius_add(-transformed, tail)
            norm = torch.linalg.vector_norm(delta, dim=-1)
            return torch.atanh(norm.clamp(max=1 - 8 * torch.finfo(norm.dtype).eps))
        return torch.linalg.vector_norm(head + relation - tail, dim=-1)

    def forward(self, head, relation, tail):
        if not bool(self.geometry_initialized):
            raise RuntimeError("initialize_geometry(training_entity_ids) before scoring")
        # Inherited score is offset-alpha*kernel output. Kernel returns D.
        score = super().forward(head, relation, tail)
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError("nonfinite WN9 geometry v2 score")
        return score

    @torch.no_grad()
    def diagnostics(self, train_head=None, train_relation=None, train_tail=None, *,
                    sample_size=32, scipy_reference=False, raise_on_failure=True):
        if not bool(self.geometry_initialized):
            raise RuntimeError("initialize geometry before diagnostics")
        if self.geometry == "sl8" and self.log_backend == "gregory12" and (
                train_head is None or train_relation is None or train_tail is None):
            raise ValueError("Gregory accuracy diagnostics require a caller-supplied fixed training triple probe")
        device = self.projection.weight.device
        if train_head is None and train_relation is None and train_tail is None:
            size = min(sample_size, self.num_entities)
            head = torch.linspace(0, self.num_entities - 1, size, device=device).round().long()
            relation, tail = head % self.num_relations, (head * 17 + 1) % self.num_entities
            scope = "deterministic_synthetic_numerical_probe"
        else:
            if train_head is None or train_relation is None or train_tail is None or train_head.shape != train_relation.shape or train_head.shape != train_tail.shape or len(train_head) == 0:
                raise ValueError("three matching nonempty training probe ID tensors required")
            indices = torch.linspace(0, len(train_head) - 1, min(sample_size, len(train_head)), device=device).round().long()
            head, relation, tail = (x.to(device)[indices] for x in (train_head, train_relation, train_tail))
            scope = "caller_supplied_training_triple_probe"
        hc, h = self.get_geometry_representations(head)
        tc, t = self.get_geometry_representations(tail)
        rc = self.bound_coordinates(self.relation_embeddings(relation))
        r = self._manifold_map(rc)
        delta = self._distance_block(hc, h, r, rc.new_empty((len(head), 0)), t)
        if not all(bool(torch.isfinite(p).all()) for p in self.parameters()) or not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("nonfinite WN9 geometry parameter/distance")
        result = {
            "geometry": self.geometry, "sample_scope": scope, "sample_size": len(head),
            "log_backend": self.log_backend,
            "model_contract": {"family": "wn9_geometry_v2", "dimension": 63,
                               "score": "offset-alpha*distance", "distance_power": 1,
                               "entity_bias": False, "relation_diagonal": False,
                               "relation_action": "left", "hyperbolic_metric_multiplier": 0.5,
                               "log_backend": self.log_backend,
                               "log_terms": 12 if self.log_backend == "gregory12" else self.log_order},
            "chart_radius": self.chart_radius, "coordinate_scale": self.coordinate_scale,
            "initialization_target_norm": float(self.initialization_target_norm),
            "initialization_train_entity_count": int(self.initialization_train_entity_count),
            "logit_scale": float(self.logit_scale), "score_offset": float(self.score_offset),
            "entity_coordinate_norm_mean": float(torch.cat((hc, tc)).norm(dim=-1).mean()),
            "entity_coordinate_norm_max": float(torch.cat((hc, tc)).norm(dim=-1).max()),
            "relation_coordinate_norm_mean": float(rc.norm(dim=-1).mean()),
            "relation_coordinate_norm_max": float(rc.norm(dim=-1).max()),
            "distance_mean": float(delta.mean()), "distance_min": float(delta.min()),
            "distance_max": float(delta.max()), "distance_std": float(delta.std(unbiased=False)),
            "parameters_finite": True, "sampled_health_passed": True}
        if self.geometry == "sl8":
            if self.log_backend == "gregory12":
                result["principal_log"] = gregory_pair_diagnostics(r @ h, t)
                result["sampled_reference_accuracy_passed"] = result["principal_log"]["reference_accuracy_passed"]
                result["sampled_health_scope"] = "finite_values_and_solves; reference_accuracy_reported_separately"
                result["training_probe_triples"] = torch.stack((head, relation, tail), dim=1).cpu().tolist()
            else:
                result["principal_log"] = log_pair_diagnostics(r @ h, t, order=self.log_order,
                                                                scipy_reference=scipy_reference,
                                                                raise_on_failure=raise_on_failure)
                result["principal_log"]["backend"] = "gauss_legendre"
            result["sampled_health_passed"] = result["principal_log"]["passed"]
            sign, logdet = torch.linalg.slogdet(torch.cat((h, r, t)))
            result["max_sample_group_abs_logdet"] = float(logdet.abs().max())
            result["nonpositive_sample_group_determinant_count"] = int((sign <= 0).sum())
        elif self.geometry == "hyperbolic":
            result["max_sample_ball_norm"] = float(torch.cat((h, r, t, mobius_add(r, h))).norm(dim=-1).max())
            result["curvature"] = 1.0
        return result
