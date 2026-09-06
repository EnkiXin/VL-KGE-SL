"""Matched-capacity structural KGE: Euclidean, Poincare and SL(8).

No text/image encoder or DistMult residual is used. All geometries have the
same trainable coordinates and only a scalar scale/offset, without entity
biases. Scoring uses LINEAR distance. SL defaults to the unchanged shared
Gregory-12 discrepancy with sampled SciPy/reconstruction accuracy auditing,
not a globally defined geodesic. The Poincare model
is a relation-left-action adaptation, NOT MuRP. Its metric is d_H/2 so the
origin tangent-coordinate limit matches Euclidean/SL distances.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from sl_manifold.core import algebra_exp, coordinates_to_algebra, orthonormal_sl_basis, symmetric_distance
from geometry.matrix_log import log_pair_diagnostics, quadrature_rule, checked_symmetric_log_distance
from geometry.gregory_diagnostics import gregory_pair_diagnostics


def poincare_exp0(coordinates):
    norm = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True).clamp_min(1e-8)
    point = coordinates * (torch.tanh(norm) / norm)
    maximum = 1 - 8 * torch.finfo(point.dtype).eps
    point_norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    return point * (maximum / point_norm.clamp_min(maximum))


def mobius_add(left, right):
    left_sq, right_sq = left.square().sum(-1, keepdim=True), right.square().sum(-1, keepdim=True)
    dot = (left * right).sum(-1, keepdim=True)
    denominator = (1 + 2 * dot + left_sq * right_sq).clamp_min(torch.finfo(left.dtype).tiny)
    point = ((1 + 2 * dot + right_sq) * left + (1 - left_sq) * right) / denominator
    maximum = 1 - 8 * torch.finfo(point.dtype).eps
    norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    return point * (maximum / norm.clamp_min(maximum))


def poincare_half_squared_distance(left, right):
    """(d_H(left,right)/2)^2, curvature -1 and exp0(v)=tanh(||v||)v/||v||."""
    delta = mobius_add(-left, right)
    norm = torch.linalg.vector_norm(delta, dim=-1)
    safe = norm.clamp(min=1e-8, max=1 - 8 * torch.finfo(norm.dtype).eps)
    ratio = torch.atanh(safe) / safe
    return delta.square().sum(-1) * ratio.square()


def poincare_half_distance(left, right):
    """Linear d_H/2 with finite PyTorch zero-norm subgradient."""
    delta = mobius_add(-left, right)
    norm = torch.linalg.vector_norm(delta, dim=-1)
    return torch.atanh(norm.clamp(max=1 - 8 * torch.finfo(norm.dtype).eps))


class StructuralKGE(nn.Module):
    GEOMETRIES = ("euclidean", "hyperbolic", "sl8")

    def __init__(self, num_entities, num_relations, geometry="euclidean", dim=63,
                 init_scale=0.5, coordinate_radius=1.5, score_scale=1.0,
                 score_chunk=1024, checkpoint_blocks=True, log_order=16,
                 log_backend="gregory12", initial_offset=0.0):
        super().__init__()
        if geometry not in self.GEOMETRIES:
            raise ValueError("unknown structural geometry")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (num_entities, num_relations, dim, score_chunk)):
            raise ValueError("entity/relation counts, dimension and chunk must be positive integers")
        if geometry == "sl8" and dim != 63:
            raise ValueError("SL(8) requires exactly 63 orthonormal algebra coordinates")
        if log_order not in (16, 32):
            raise ValueError("log_order must be 16 or 32")
        if log_backend not in ("gregory12", "gauss_legendre"):
            raise ValueError("log_backend must be gregory12 or gauss_legendre")
        if any(not math.isfinite(value) or value <= 0 for value in (init_scale, score_scale)):
            raise ValueError("init_scale and score_scale must be finite and positive")
        if not math.isfinite(initial_offset):
            raise ValueError("initial_offset must be finite")
        if coordinate_radius is not None and (not math.isfinite(coordinate_radius) or coordinate_radius <= 0):
            raise ValueError("coordinate_radius must be None or finite positive")
        if coordinate_radius is not None and init_scale >= coordinate_radius:
            raise ValueError("initial norm must be below coordinate_radius")
        self.num_entities, self.num_relations, self.dim = num_entities, num_relations, dim
        self.geometry, self.coordinate_radius = geometry, coordinate_radius
        self.init_scale, self.score_chunk = float(init_scale), score_chunk
        self.checkpoint_blocks, self.log_order = bool(checkpoint_blocks), log_order
        self.log_backend = log_backend
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.relation_embeddings = nn.Embedding(num_relations, dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(score_scale)))
        self.score_offset = nn.Parameter(torch.tensor(float(initial_offset)))
        # Deterministic nontrainable geometry buffers do not alter RNG parity.
        self.register_buffer("sl_basis", orthonormal_sl_basis(8), persistent=False)
        nodes, weights = quadrature_rule(log_order)
        self.register_buffer("quadrature_nodes", nodes, persistent=False)
        self.register_buffer("quadrature_weights", weights, persistent=False)
        self._eval_cache = None
        self._eval_cache_signature = None
        self.reset_parameters()

    @property
    def score_scale(self):
        return self.log_scale.exp()

    def reset_parameters(self):
        # Both tables use random directions but EXACTLY the same initial
        # bounded coordinate norm. No data/features/calibration is required.
        raw_target = (self.init_scale if self.coordinate_radius is None else
                      self.init_scale / math.sqrt(1 - (self.init_scale / self.coordinate_radius) ** 2))
        with torch.no_grad():
            for table in (self.entity_embeddings, self.relation_embeddings):
                nn.init.normal_(table.weight)
                table.weight.mul_(raw_target / table.weight.norm(dim=-1, keepdim=True))
        self.clear_eval_cache()

    def clear_eval_cache(self):
        self._eval_cache, self._eval_cache_signature = None, None

    def clear_evaluation_cache(self):
        self.clear_eval_cache()

    def train(self, mode=True):
        self.clear_eval_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_eval_cache()
        return super()._apply(fn)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_eval_cache()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _bound(self, coordinates):
        if self.coordinate_radius is None:
            return coordinates
        return coordinates / torch.sqrt(1 + coordinates.square().sum(-1, keepdim=True)
                                         / self.coordinate_radius ** 2)

    def _map(self, coordinates):
        coordinates = self._bound(coordinates)
        if self.geometry == "sl8":
            return algebra_exp(coordinates_to_algebra(coordinates, 8, basis=self.sl_basis))
        if self.geometry == "hyperbolic":
            return poincare_exp0(coordinates)
        return coordinates

    def _cache_signature(self):
        return (self.geometry, self.coordinate_radius,
                tuple((id(p), p._version) for p in self.parameters()))

    @torch.no_grad()
    def prepare_eval_cache(self):
        """Encode every entity/relation once; caller must use eval+no_grad."""
        if self.training:
            raise RuntimeError("call eval() before prepare_eval_cache()")
        signature = self._cache_signature()
        if self._eval_cache is None or signature != self._eval_cache_signature:
            entities = torch.cat([self._map(block) for block in
                                  self.entity_embeddings.weight.split(self.score_chunk)])
            relations = torch.cat([self._map(block) for block in
                                   self.relation_embeddings.weight.split(self.score_chunk)])
            self._eval_cache = (entities, relations)
            self._eval_cache_signature = signature
        return self._eval_cache

    def _distance_block(self, head, relation, tail):
        if self.geometry == "sl8":
            if self.log_backend == "gregory12":
                return symmetric_distance(relation @ head, tail, terms=12,
                                          jitter=1e-7, trace_project=True)
            return checked_symmetric_log_distance(relation @ head, tail, order=self.log_order,
                                                   nodes=self.quadrature_nodes, weights=self.quadrature_weights)
        if self.geometry == "hyperbolic":
            return poincare_half_distance(mobius_add(relation, head), tail)
        return torch.linalg.vector_norm(head + relation - tail, dim=-1)

    def forward(self, head, relation, tail):
        if head.ndim != 1 or head.shape != relation.shape or head.shape != tail.shape:
            raise ValueError("head/relation/tail IDs must be matching 1-D tensors")
        if self.score_chunk < 1:
            raise ValueError("score_chunk must be positive")
        if head.numel() == 0:
            return self.entity_embeddings.weight.new_empty((0,))
        count = head.numel()
        if not self.training and not torch.is_grad_enabled():
            entities, relations = self.prepare_eval_cache()
            head_map, tail_map, relation_map = head, tail, relation
        else:
            ids, entity_map = torch.unique(torch.cat((head, tail)), return_inverse=True)
            relation_ids, relation_map = torch.unique(relation, return_inverse=True)
            entities = self._map(self.entity_embeddings(ids))
            relations = self._map(self.relation_embeddings(relation_ids))
            head_map, tail_map = entity_map[:count], entity_map[count:]
        distances = []
        for start in range(0, count, self.score_chunk):
            stop = start + self.score_chunk
            inputs = (entities[head_map[start:stop]], relations[relation_map[start:stop]], entities[tail_map[start:stop]])
            if self.checkpoint_blocks and torch.is_grad_enabled():
                delta = checkpoint(self._distance_block, *inputs, use_reentrant=False, preserve_rng_state=False)
            else:
                delta = self._distance_block(*inputs)
            distances.append(delta)
        distance = torch.cat(distances)
        return self.score_offset - self.score_scale * distance

    def regularization(self, head, relation, tail):
        """Common raw-coordinate mean squared norm, averaged over h/r/t roles."""
        return ((self.entity_embeddings(head).square().sum(-1)
                 + self.relation_embeddings(relation).square().sum(-1)
                 + self.entity_embeddings(tail).square().sum(-1)) / 3).mean()

    @torch.no_grad()
    def diagnostics(self, head=None, relation=None, tail=None, *, sample_size=32,
                    scipy_reference=False, raise_on_failure=True,
                    reconstruction_tolerance=1e-3, order_tolerance=1e-3):
        """Sample actual supplied training pairs without consuming RNG.

        With no IDs, use a deterministic synthetic numerical probe, not any
        dataset ranking metric. Scoring has no per-candidate host sync;
        callers must separately check loss/gradient finiteness each update.
        """
        if any(not bool(torch.isfinite(p).all()) for p in self.parameters()):
            raise FloatingPointError("nonfinite structural KGE parameter")
        if self.geometry == "sl8" and self.log_backend == "gregory12" and (
                head is None or relation is None or tail is None):
            raise ValueError("Gregory diagnostics require a fixed caller-supplied training triple probe")
        device = self.entity_embeddings.weight.device
        if head is None and relation is None and tail is None:
            size = min(sample_size, self.num_entities)
            head = torch.linspace(0, self.num_entities - 1, size, device=device).round().long()
            relation, tail = head % self.num_relations, (head * 17 + 1) % self.num_entities
            scope = "deterministic_synthetic_numerical_probe"
        elif head is None or relation is None or tail is None or head.shape != relation.shape or head.shape != tail.shape:
            raise ValueError("provide all three matching probe ID tensors")
        else:
            indices = torch.linspace(0, len(head) - 1, min(sample_size, len(head)), device=device).round().long()
            head, relation, tail = head.to(device)[indices], relation.to(device)[indices], tail.to(device)[indices]
            scope = "caller_supplied_training_probe"
        hcoord, rcoord, tcoord = self.entity_embeddings(head), self.relation_embeddings(relation), self.entity_embeddings(tail)
        h, r, t = self._map(hcoord), self._map(rcoord), self._map(tcoord)
        delta = self._distance_block(h, r, t)
        if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(self.score_scale)):
            raise FloatingPointError("nonfinite sampled structural score geometry")
        result = {
            "geometry": self.geometry, "scope": scope, "sample_size": len(head),
            "log_backend": self.log_backend,
            "model_contract": {"family": "structural_kge", "dimension": self.dim,
                               "score": "offset-alpha*distance", "distance_power": 1,
                               "entity_bias": False, "relation_diagonal": False,
                               "log_backend": self.log_backend,
                               "log_terms": 12 if self.log_backend == "gregory12" else self.log_order,
                               "features": "id_only", "hyperbolic_metric_multiplier": 0.5},
            "coordinate_radius": self.coordinate_radius, "parameters_finite": True,
            "score_scale": float(self.score_scale), "score_offset": float(self.score_offset),
            "initialization_target_bounded_norm": self.init_scale,
            "max_sample_entity_coordinate_norm": float(torch.cat((hcoord, tcoord)).norm(dim=-1).max()),
            "max_sample_relation_coordinate_norm": float(rcoord.norm(dim=-1).max()),
            "sample_distance_mean": float(delta.mean()),
            "sample_distance_min": float(delta.min()),
            "sample_distance_max": float(delta.max()),
            "max_sample_bounded_entity_coordinate_norm": float(self._bound(torch.cat((hcoord, tcoord))).norm(dim=-1).max()),
            "max_sample_bounded_relation_coordinate_norm": float(self._bound(rcoord).norm(dim=-1).max()),
            "sampled_health_passed": True,
        }
        if self.geometry == "sl8":
            if self.log_backend == "gregory12":
                result["principal_log"] = gregory_pair_diagnostics(r @ h, t)
                result["sampled_reference_accuracy_passed"] = result["principal_log"]["reference_accuracy_passed"]
                result["sampled_health_scope"] = "finite_values_and_solves; reference_accuracy_reported_separately"
                result["training_probe_triples"] = torch.stack((head, relation, tail), dim=1).cpu().tolist()
            else:
                result["principal_log"] = log_pair_diagnostics(
                    r @ h, t, order=self.log_order, scipy_reference=scipy_reference,
                    reconstruction_tolerance=reconstruction_tolerance, order_tolerance=order_tolerance,
                    raise_on_failure=raise_on_failure)
                result["principal_log"]["backend"] = "gauss_legendre"
            result["sampled_health_passed"] = result["principal_log"]["passed"]
            sign, logdet = torch.linalg.slogdet(torch.cat((h, r, t)))
            result["max_sample_group_abs_logdet"] = float(logdet.abs().max())
            result["nonpositive_sample_group_determinant_count"] = int((sign <= 0).sum())
        elif self.geometry == "hyperbolic":
            result.update(curvature=1.0, metric_multiplier=0.5,
                          max_sample_ball_norm=float(torch.cat((h, r, t, mobius_add(r, h))).norm(dim=-1).max()))
        return result
