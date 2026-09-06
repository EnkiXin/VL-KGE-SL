"""Matched 63-coordinate VL-KGE geometry adapters.

These are adapted VL models, not reproductions of the original MuRP paper.
MuRE/MuRP follow its relation diagonal and tail-translation equations:
https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py
They share the VL-KGE loss/encoder, tangent parameterization, bounded input
chart and scalar calibration here; original per-entity biases are omitted.

SL uses the shared sl-manifold-core implementation, not a copied logarithm.
Its symmetric Gregory-12 local-log discrepancy is NOT an exact global
Riemannian distance. Diagnostics inspect relative matrices used in scoring.
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
from vlkge.models.vlkge import VLKGEBase


def _ball_project(point):
    """Numerical guard only; the ordinary exp0 maps are already in the ball."""
    radius = 1.0 - 8.0 * torch.finfo(point.dtype).eps
    norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    return point * (radius / norm.clamp_min(radius)).clamp_max(1.0)


def poincare_exp0(tangent):
    """Unit-curvature exp0 with finite derivatives at the origin."""
    norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(1e-8)
    ratio = torch.tanh(safe_norm) / safe_norm
    return _ball_project(tangent * ratio)


def mobius_add(left, right):
    """Ordered unit-curvature Poincare addition, left ⊕ right."""
    left_sq = left.square().sum(dim=-1, keepdim=True)
    right_sq = right.square().sum(dim=-1, keepdim=True)
    dot = (left * right).sum(dim=-1, keepdim=True)
    numerator = (1 + 2 * dot + right_sq) * left + (1 - left_sq) * right
    denominator = 1 + 2 * dot + left_sq * right_sq
    return _ball_project(numerator / denominator.clamp_min(torch.finfo(left.dtype).tiny))


def poincare_squared_distance(left, right):
    """Squared distance with a continuous zero-distance gradient."""
    delta = mobius_add(-left, right)
    norm = torch.linalg.vector_norm(delta, dim=-1)
    safe_norm = norm.clamp(min=1e-8, max=1 - 8 * torch.finfo(norm.dtype).eps)
    # Express d² as ||delta||² * (2 atanh(r)/r)²; unlike acosh(1),
    # the factor is smooth and finite at a coincident pair.
    ratio = 2 * torch.atanh(safe_norm) / safe_norm
    return delta.square().sum(dim=-1) * ratio.square()


class VLGeometry(VLKGEBase):
    """Author multimodal fusion followed by one common 63-D projection.

    ``get_entity_representations`` deliberately retains the author's fused
    768-D output; ``get_geometry_representations`` returns chart/manifold data.
    Public ``score_chunk`` can be changed between calls. Frozen evaluation
    caching is used only when both eval mode AND no_grad are active, and is
    invalidated by parameter versions, train(), load_state_dict(), and to().
    """

    GEOMETRIES = ("euclidean", "mure", "murp", "sl8")

    def __init__(self, *, geometry="sl8", coordinate_scale=0.1,
                 chart_radius=0.5, score_chunk=1024, checkpoint_blocks=True,
                 initial_logit_scale=100.0, initial_offset=0.0,
                 relation_features=None, **author_base_kwargs):
        if geometry not in self.GEOMETRIES:
            raise ValueError(f"unknown geometry: {geometry}")
        if coordinate_scale <= 0 or chart_radius <= 0 or score_chunk < 1:
            raise ValueError("coordinate_scale, chart_radius and score_chunk must be positive")
        if not math.isfinite(initial_logit_scale) or initial_logit_scale <= 0:
            raise ValueError("initial_logit_scale must be positive and finite")
        if relation_features is not None:
            raise ValueError("this controlled WN9 adapter does not use relation features")
        super().__init__(**author_base_kwargs)
        if self.embedding_dim != 768 or self.fusion_mode != "average":
            raise ValueError("controlled geometry adapter requires author 768-D average fusion")
        self.geometry = geometry
        self.coordinate_dim = 63
        self.coordinate_scale = float(coordinate_scale)
        self.chart_radius = float(chart_radius)
        self.score_chunk = int(score_chunk)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.projection = nn.Linear(self.embedding_dim, 63, bias=False)
        self.relation_embeddings = nn.Embedding(self.num_relations, 63)
        # Its default random initialization is immediately overwritten with
        # ones. Avoid advancing the CPU RNG so a common seed initializes the
        # shared front end identically across all four geometry choices.
        with torch.random.fork_rng(devices=[]):
            self.relation_diagonal = (
                nn.Embedding(self.num_relations, 63) if geometry in ("mure", "murp") else None
            )
        self.score_offset = nn.Parameter(torch.tensor(float(initial_offset)))
        # Optimize log(alpha), so a common Adagrad step changes the scale
        # proportionately instead of barely changing a large softplus input.
        self.raw_logit_scale = nn.Parameter(torch.tensor(math.log(initial_logit_scale)))
        self.register_buffer("sl_basis", orthonormal_sl_basis(8), persistent=False)
        self._evaluation_cache = None
        self._evaluation_cache_signature = None
        self.reset_parameters()
        self.to(device=self.device)

    @property
    def logit_scale(self):
        return self.raw_logit_scale.exp()

    def reset_parameters(self):
        if self.entity_embeddings is not None:
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.normal_(self.relation_embeddings.weight, std=1e-3)
        if self.relation_diagonal is not None:
            nn.init.ones_(self.relation_diagonal.weight)
        # The author WN9 input has identity feature projections. Preserve
        # Xavier initialization if the base creates a dimension adapter.
        for name in ("visual_linear", "textual_linear", "unified_projection"):
            layer = getattr(self, name, None)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.clear_evaluation_cache()

    def clear_evaluation_cache(self):
        self._evaluation_cache = None
        self._evaluation_cache_signature = None

    def train(self, mode=True):
        self.clear_evaluation_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_evaluation_cache()
        result = super()._apply(fn)
        # Upstream stores lookup tensors and the device outside buffers.
        for name in ("visual_id_lookup", "textual_id_lookup"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, fn(value))
        self.device = self.projection.weight.device
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_evaluation_cache()
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.device = self.projection.weight.device
        return result

    def bound_coordinates(self, coordinates):
        scaled = coordinates * self.coordinate_scale
        # Smooth radial bound, with identity derivative at zero before scale.
        denominator = torch.sqrt(1 + scaled.square().sum(dim=-1, keepdim=True)
                                 / self.chart_radius ** 2)
        return scaled / denominator

    def _manifold_map(self, coordinates):
        if self.geometry == "sl8":
            return algebra_exp(coordinates_to_algebra(coordinates, 8, basis=self.sl_basis))
        if self.geometry == "murp":
            return poincare_exp0(coordinates)
        return coordinates

    def get_geometry_representations(self, entity_ids):
        coordinates = self.bound_coordinates(self.projection(
            super().get_entity_representations(entity_ids)))
        return coordinates, self._manifold_map(coordinates)

    def get_relation_representations(self, relation_ids):
        coordinates = self.bound_coordinates(self.relation_embeddings(relation_ids))
        return self._manifold_map(coordinates)

    def _cache_signature(self):
        # Optimizer updates in eval/no_grad must not silently reuse stale data.
        return (self.coordinate_scale, self.chart_radius,
                tuple((id(p), p._version) for p in self.parameters()),
                tuple((id(b), b._version) for b in self.buffers()))

    def _cached_entities(self):
        signature = self._cache_signature()
        if self._evaluation_cache is None or signature != self._evaluation_cache_signature:
            coordinates, representations = [], []
            for start in range(0, self.num_entities, self.score_chunk):
                ids = torch.arange(start, min(start + self.score_chunk, self.num_entities),
                                   device=self.projection.weight.device)
                coord, manifold = self.get_geometry_representations(ids)
                coordinates.append(coord)
                representations.append(manifold)
            self._evaluation_cache = (torch.cat(coordinates), torch.cat(representations))
            self._evaluation_cache_signature = signature
        return self._evaluation_cache

    def _distance_block(self, head_coord, head, relation, diagonal, tail):
        if self.geometry == "sl8":
            return symmetric_distance(relation @ head, tail, terms=12,
                                      jitter=1e-7, trace_project=True).square()
        if self.geometry == "murp":
            transformed_head = poincare_exp0(head_coord * diagonal)
            translated_tail = mobius_add(tail, relation)
            return poincare_squared_distance(transformed_head, translated_tail)
        if self.geometry == "mure":
            return (head * diagonal - (tail + relation)).square().sum(dim=-1)
        return (head + relation - tail).square().sum(dim=-1)

    def forward(self, head, relation, tail):
        if head.ndim != 1 or head.shape != relation.shape or head.shape != tail.shape:
            raise ValueError("head, relation and tail must be matching 1-D ID tensors")
        if self.score_chunk < 1:
            raise ValueError("score_chunk must be positive")
        count = head.numel()
        if count == 0:
            return self.projection.weight.new_empty((0,))
        if not self.training and not torch.is_grad_enabled():
            entity_coord, entity_repr = self._cached_entities()
            head_map, tail_map = head, tail
        else:
            entity_ids, entity_map = torch.unique(torch.cat((head, tail)), return_inverse=True)
            entity_coord, entity_repr = self.get_geometry_representations(entity_ids)
            head_map, tail_map = entity_map[:count], entity_map[count:]
        relation_ids, relation_map = torch.unique(relation, return_inverse=True)
        relation_repr = self.get_relation_representations(relation_ids)
        diagonal_repr = (self.relation_diagonal(relation_ids) if self.relation_diagonal is not None
                         else relation_repr.new_empty((len(relation_ids), 0)))
        distances = []
        for start in range(0, count, self.score_chunk):
            stop = start + self.score_chunk
            hm, tm, rm = head_map[start:stop], tail_map[start:stop], relation_map[start:stop]
            values = (entity_coord[hm], entity_repr[hm], relation_repr[rm],
                      diagonal_repr[rm], entity_repr[tm])
            if self.checkpoint_blocks and torch.is_grad_enabled():
                distance = checkpoint(self._distance_block, *values, use_reentrant=False,
                                      preserve_rng_state=False)
            else:
                distance = self._distance_block(*values)
            distances.append(distance)
        return self.score_offset - self.logit_scale * torch.cat(distances)

    @torch.no_grad()
    def diagnostics(self, sample_size=256):
        """RNG-free spread sample, not exhaustive certification of all triples.

        Sample entity IDs span the complete ID range, with deterministic
        counterpart/relation choices. This never changes the train mode.
        """
        count = min(max(int(sample_size), 1), self.num_entities)
        ids = torch.linspace(0, self.num_entities - 1, steps=count,
                             device=self.projection.weight.device).round().long()
        tails = (ids * 17 + 1) % self.num_entities
        relations = ids % self.num_relations
        head_coord, head = self.get_geometry_representations(ids)
        _, tail = self.get_geometry_representations(tails)
        relation = self.get_relation_representations(relations)
        output = {
            "geometry": self.geometry, "sample_size": count,
            "diagnostic_scope": "deterministic_spread_entity_sample_not_all_triples",
            "coordinate_scale": self.coordinate_scale, "chart_radius": self.chart_radius,
            "max_entity_coordinate_norm": float(head_coord.norm(dim=-1).max()),
            "logit_scale": float(self.logit_scale), "score_offset": float(self.score_offset),
            "parameters_finite": all(bool(torch.isfinite(p).all()) for p in self.parameters()),
        }
        if self.geometry == "sl8":
            transformed = relation @ head
            # Both directions actually used by the symmetric score.
            forward = torch.linalg.solve(transformed, tail)
            reverse = torch.linalg.solve(tail, transformed)
            output["entities"] = summarize_sl_diagnostics(head, gregory_terms=12)
            output["relations"] = summarize_sl_diagnostics(relation, gregory_terms=12)
            output["relative_score_matrices"] = summarize_sl_diagnostics(
                torch.cat((forward, reverse)), gregory_terms=12)
        elif self.geometry == "murp":
            transformed = poincare_exp0(head_coord * self.relation_diagonal(relations))
            translated = mobius_add(tail, relation)
            output["max_ball_norm"] = float(torch.cat((head, transformed, translated)).norm(dim=-1).max())
            output["curvature"] = 1.0
        return output
