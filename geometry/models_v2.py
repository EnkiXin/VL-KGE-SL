"""Geometry v2: SL(n) and Euclidean VL-KGE scorers with the v1 defects removed.

Differences from ``geometry/models.py`` (v1), each motivated by the v1 analysis:

* **Chart.** v1 squashed coordinates radially into a ball of radius 0.5, where
  ``exp`` is indistinguishable from the identity map and the SL distance is
  Euclidean to within 1%.  v2 only *clips* the Frobenius norm at a much larger
  radius (default 2.0), so the geometry is actually used and the radial
  gradient is not attenuated inside the chart.
* **Logarithm.** v1 used a 12-term Gregory series whose accuracy collapses
  once relative matrices leave a small neighbourhood of the identity.  v2 uses
  inverse scaling and squaring: ``sqrt_steps`` Denman--Beavers square roots
  bring the relative matrix close to the identity, the Gregory series is
  applied there, and the result is multiplied back by ``2**sqrt_steps``.  The
  domain is the full principal-logarithm domain (no eigenvalue on the negative
  real axis), which is monitored by the diagnostics instead of a conservative
  spectral-norm bound.
* **Score.** v1 used ``offset - 100 * D**2``, which saturates the logistic
  loss immediately.  v2 uses ``offset - exp(log_scale) * D`` with modest,
  learnable initial values (the form used by the SL(n) paper and by VL-TransE).
* **Relations.** v1 initialised relations at 1e-4 of the entity scale, so the
  model started as a symmetric entity-distance and the head entity was always
  ranked first (validation Hits@1 was exactly 0 for the first epochs).  v2
  initialises relation coordinates at a fixed expected norm (default 0.5) and
  keeps them in their own clip radius.
* **Dimension.** ``matrix_dim`` is free (SL(8) = 63 coordinates, SL(28) = 783),
  so capacity can be matched to the 768-D author models.  The Euclidean
  control uses the same coordinate count, projection, clipping and score.
  ``entity_mapping='fixed_pad'`` instead appends zero coordinates to the
  author's fused 768-D vector, without any learned entity projection. For
  SL(28) this appends 15 zeros; scaling and radial clipping are unchanged.
  ``entity_mapping='direct'`` instead starts with embedding_dim=n**2-1 and
  uses the fused vector directly. For SL(28), the author base aligns the
  frozen 768-D features to 783 before fusion, with no post-fusion projection.

Everything else (author fusion, entity table, loss, sampler, evaluator) is
identical to v1.  The one-sided ``||log(A^{-1}B)||`` is used because the
symmetrised v1 form is mathematically identical for a principal logarithm.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from sl_manifold.core import (
    algebra_exp,
    coordinates_to_algebra,
    orthonormal_sl_basis,
    project_to_sl_algebra,
)
from vlkge.models.vlkge import VLKGEBase


# ----------------------------------------------------------------------------
# Principal logarithm by inverse scaling and squaring
# ----------------------------------------------------------------------------

def _eye_like(matrix):
    n = matrix.shape[-1]
    return torch.eye(n, dtype=matrix.dtype, device=matrix.device).expand(matrix.shape)


def denman_beavers_sqrt(matrix, iterations=6):
    """Principal square root by the Denman--Beavers iteration.

    Converges quadratically for every matrix without eigenvalues on the closed
    negative real axis, i.e. exactly the principal-logarithm domain.  Each
    iteration costs two batched inverses.
    """
    if iterations < 1:
        raise ValueError("Denman-Beavers needs at least one iteration")
    y, z = matrix, _eye_like(matrix)
    for _ in range(iterations):
        y_next = 0.5 * (y + torch.linalg.inv(z))
        z = 0.5 * (z + torch.linalg.inv(y))
        y = y_next
    return y


def _gregory(cayley, terms):
    """``2 * atanh(C)`` as a truncated odd power series in ``C``."""
    if terms < 1:
        raise ValueError("Gregory terms must be positive")
    c2 = cayley @ cayley
    power, series = cayley, cayley
    for index in range(1, terms):
        power = power @ c2
        series = series + power / float(2 * index + 1)
    return 2.0 * series


def principal_log_with_flags(matrix, *, sqrt_steps=1, db_iterations=6, terms=12, jitter=0.0,
                             residual_tolerance=1e-4):
    """Principal matrix logarithm for batches of small square matrices.

    ``log(M) = 2**s * log(M**(1/2**s))`` with the inner logarithm evaluated by
    the Gregory series in Cayley coordinates.  With ``s = 1`` and 12 terms the
    result matches an eigen-decomposition logarithm to about 1e-6 relative
    error for relative matrices whose eigenvalues satisfy ``|log|lambda|| <= 3``
    and stay away from the negative real axis (the principal-logarithm domain).

    Matrices for which the square-root iteration does not converge (no real
    principal logarithm, or a numerically broken iteration) are flagged and
    scored with the plain Gregory series on the original matrix instead: that
    value is finite, differentiable and *large*, so such pairs are ranked as
    far apart rather than producing NaN.  The fraction of flagged matrices is
    reported by the diagnostics; it should be zero in a healthy run.
    Returns ``(logarithm, flagged)``.
    """
    if sqrt_steps < 0:
        raise ValueError("sqrt_steps must be non-negative")
    identity = _eye_like(matrix)
    finite_input = torch.isfinite(matrix).all(dim=(-2, -1))
    safe_matrix = torch.where(finite_input[..., None, None], matrix, identity)
    root = safe_matrix
    for _ in range(sqrt_steps):
        root = denman_beavers_sqrt(root, db_iterations)
    if sqrt_steps > 0:
        power = root
        for _ in range(sqrt_steps):
            power = power @ power
        residual = torch.linalg.matrix_norm(power - safe_matrix, dim=(-2, -1))
        scale = torch.linalg.matrix_norm(safe_matrix, dim=(-2, -1)).clamp_min(1e-12)
        flagged = ~torch.isfinite(root).all(dim=(-2, -1)) | ~torch.isfinite(residual) \
            | (residual > residual_tolerance * scale) | ~finite_input
    else:
        flagged = ~finite_input
    # Sanitise the scaled branch so its gradient is finite everywhere; torch.where
    # would otherwise propagate NaN gradients from the unselected branch.
    root = torch.where(flagged[..., None, None], identity, root)
    cayley = torch.linalg.solve(root + (1.0 + float(jitter)) * identity, root - identity)
    logarithm = _gregory(cayley, terms) * float(2 ** sqrt_steps)
    if bool(flagged.any()):
        fallback_cayley = torch.linalg.solve(safe_matrix + (1.0 + max(float(jitter), 1e-6)) * identity,
                                             safe_matrix - identity)
        fallback = _gregory(fallback_cayley, terms)
        logarithm = torch.where(flagged[..., None, None], fallback, logarithm)
    return logarithm, flagged


def principal_log(matrix, **kwargs):
    """See :func:`principal_log_with_flags`; returns only the logarithm."""
    return principal_log_with_flags(matrix, **kwargs)[0]


def relative_matrix(left, right):
    """``left^{-1} right`` without forming the inverse explicitly."""
    return torch.linalg.solve(left, right)


def sl_distance(left, right, *, p=2.0, sqrt_steps=1, db_iterations=6, terms=12, jitter=0.0):
    """``||Pi_sl log(left^{-1} right)||_{S_p}`` (one-sided; equals the symmetrised form)."""
    logarithm = project_to_sl_algebra(
        principal_log(relative_matrix(left, right), sqrt_steps=sqrt_steps,
                      db_iterations=db_iterations, terms=terms, jitter=jitter))
    if p == 2:
        return torch.linalg.matrix_norm(logarithm, ord="fro", dim=(-2, -1))
    singular = torch.linalg.svdvals(logarithm)
    if math.isinf(p):
        return singular.amax(dim=-1)
    return torch.linalg.vector_norm(singular, ord=p, dim=-1)


@torch.no_grad()
def spectrum_diagnostics(matrix):
    """Principal-domain margin ``min(pi - |arg lambda|)`` and log-radius ``max|log|lambda||``."""
    work = matrix.double() if matrix.dtype != torch.float64 else matrix
    finite = torch.isfinite(work).all(dim=(-2, -1))
    safe = torch.where(finite[..., None, None], work, _eye_like(work))
    eigenvalues = torch.linalg.eigvals(safe)
    margin = (math.pi - eigenvalues.angle().abs()).amin(dim=-1)
    log_radius = eigenvalues.abs().clamp_min(1e-300).log().abs().amax(dim=-1)
    return {
        "num_matrices": int(finite.numel()),
        "nonfinite_fraction": float((~finite).double().mean()),
        "principal_domain_min_margin_rad": float(margin.min()),
        "principal_domain_margin_below_0p1_fraction": float((margin < 0.1).double().mean()),
        "max_abs_log_eigenvalue_modulus": float(log_radius.max()),
        "mean_abs_log_eigenvalue_modulus": float(log_radius.mean()),
    }


@torch.no_grad()
def log_accuracy_check(matrix, **log_kwargs):
    """Relative Frobenius error of :func:`principal_log` against an eigen-decomposition log."""
    work = matrix.double()
    finite = torch.isfinite(work).all(dim=(-2, -1))
    safe = torch.where(finite[..., None, None], work, _eye_like(work))
    values, vectors = torch.linalg.eig(safe)
    reference = (vectors @ torch.diag_embed(values.log()) @ torch.linalg.inv(vectors)).real
    approximate, flagged = principal_log_with_flags(safe, **log_kwargs)
    error = torch.linalg.matrix_norm(approximate - reference, dim=(-2, -1))
    scale = torch.linalg.matrix_norm(reference, dim=(-2, -1)).clamp_min(1e-8)
    relative = error / scale
    return {"log_max_rel_error_vs_eig": float(relative.max()),
            "log_mean_rel_error_vs_eig": float(relative.mean()),
            "log_fallback_fraction": float(flagged.double().mean())}


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def radial_clip(x, radius):
    """Scale vectors whose norm exceeds ``radius`` back onto the sphere; identity inside."""
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * torch.clamp(radius / norm.clamp_min(1e-12), max=1.0)


class VLGeometryV2(VLKGEBase):
    """Author 768-D fusion -> coordinate mapping -> SL(n) or Euclidean scorer.

    ``geometry='sl'`` maps coordinates to ``exp(X)`` in SL(matrix_dim);
    ``geometry='euclidean'`` uses the raw coordinates with the same dimension,
    clipping, relation handling and score form, so the pair isolates the
    effect of the geometry alone.
    ``entity_mapping='fixed_pad'`` preserves the fused coordinates and appends
    zeros, without a learned projection. It requires n**2 - 1 >= 768 for SL.
    ``entity_mapping='direct'`` requires embedding_dim == coordinate_dim and
    uses the author's fused representation directly, without padding. Any
    input-modality dimension alignment remains in the unchanged author base.
    Existing coordinate scaling and radial clipping still apply afterward.
    """

    GEOMETRIES = ("sl", "euclidean")
    RELATION_MODES = ("left", "sandwich")
    ENTITY_MAPPINGS = ("linear", "fixed_pad", "direct")

    def __init__(self, *, geometry="sl", matrix_dim=8, coordinate_dim=None,
                 entity_mapping="linear",
                 coordinate_scale=0.1, chart_radius=2.0, relation_radius=1.5,
                 relation_init_norm=0.5, relation_mode="left",
                 score_form="linear", initial_logit_scale=3.0, initial_offset=3.0,
                 schatten_p=2.0, log_sqrt_steps=1, log_db_iterations=6, log_terms=12,
                 log_jitter=0.0, score_chunk=1024, checkpoint_blocks=True,
                 relation_features=None, **author_base_kwargs):
        if geometry not in self.GEOMETRIES:
            raise ValueError(f"unknown geometry: {geometry}")
        if entity_mapping not in self.ENTITY_MAPPINGS:
            raise ValueError(f"unknown entity_mapping: {entity_mapping}")
        if relation_mode not in self.RELATION_MODES:
            raise ValueError(f"unknown relation_mode: {relation_mode}")
        if score_form not in ("linear", "squared"):
            raise ValueError("score_form must be 'linear' or 'squared'")
        if matrix_dim < 2:
            raise ValueError("matrix_dim must be at least 2")
        for name, value in (("coordinate_scale", coordinate_scale), ("chart_radius", chart_radius),
                            ("relation_radius", relation_radius), ("initial_logit_scale", initial_logit_scale)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if relation_init_norm < 0 or not math.isfinite(relation_init_norm):
            raise ValueError("relation_init_norm must be finite and non-negative")
        if not math.isfinite(initial_offset):
            raise ValueError("initial_offset must be finite")
        if schatten_p < 1:
            raise ValueError("schatten_p must be at least 1")
        if score_chunk < 1 or log_sqrt_steps < 0 or log_db_iterations < 1 or log_terms < 1:
            raise ValueError("invalid chunk or logarithm settings")
        if relation_features is not None:
            raise ValueError("this controlled WN9 adapter does not use relation features")
        super().__init__(**author_base_kwargs)
        if self.fusion_mode != "average":
            raise ValueError("controlled geometry adapter requires author average fusion")
        if entity_mapping != "direct" and self.embedding_dim != 768:
            raise ValueError("linear/fixed_pad adapters require author 768-D fusion")
        self.geometry = geometry
        self.matrix_dim = int(matrix_dim)
        self.coordinate_dim = int(coordinate_dim) if coordinate_dim else self.matrix_dim ** 2 - 1
        if geometry == "sl" and self.coordinate_dim != self.matrix_dim ** 2 - 1:
            raise ValueError("SL coordinates must number matrix_dim**2 - 1")
        self.entity_mapping = entity_mapping
        if entity_mapping == "direct" and self.coordinate_dim != self.embedding_dim:
            raise ValueError("direct mapping requires embedding_dim == coordinate_dim")
        if entity_mapping == "fixed_pad" and self.coordinate_dim < self.embedding_dim:
            raise ValueError("fixed_pad cannot reduce the author 768-D representation")
        self.coordinate_scale = float(coordinate_scale)
        self.chart_radius = float(chart_radius)
        self.relation_radius = float(relation_radius)
        self.relation_init_norm = float(relation_init_norm)
        self.relation_mode = relation_mode
        self.score_form = score_form
        self.schatten_p = float(schatten_p)
        self.log_kwargs = dict(sqrt_steps=int(log_sqrt_steps), db_iterations=int(log_db_iterations),
                               terms=int(log_terms), jitter=float(log_jitter))
        self.score_chunk = int(score_chunk)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.projection = (nn.Linear(self.embedding_dim, self.coordinate_dim, bias=False)
                           if entity_mapping == "linear" else None)
        relation_slots = 2 if relation_mode == "sandwich" else 1
        self.relation_embeddings = nn.Embedding(self.num_relations, relation_slots * self.coordinate_dim)
        self.score_offset = nn.Parameter(torch.tensor(float(initial_offset)))
        self.raw_logit_scale = nn.Parameter(torch.tensor(math.log(initial_logit_scale)))
        if geometry == "sl":
            self.register_buffer("sl_basis", orthonormal_sl_basis(self.matrix_dim), persistent=False)
        else:
            self.sl_basis = None
        self._evaluation_cache = None
        self._evaluation_cache_signature = None
        self.reset_parameters()
        self.to(device=self.device)

    # ---------------------------------------------------------------- setup
    @property
    def logit_scale(self):
        return self.raw_logit_scale.exp()

    def reset_parameters(self):
        if self.entity_embeddings is not None:
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
        if self.projection is not None:
            nn.init.xavier_uniform_(self.projection.weight)
        # Expected Euclidean norm of a Gaussian vector with per-coordinate std s
        # over d coordinates is about s*sqrt(d): choose s so that norm ~ init_norm.
        std = self.relation_init_norm / math.sqrt(self.coordinate_dim)
        nn.init.normal_(self.relation_embeddings.weight, std=max(std, 1e-8))
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
        for name in ("visual_id_lookup", "textual_id_lookup"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, fn(value))
        self.device = self.score_offset.device
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_evaluation_cache()
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.device = self.score_offset.device
        return result

    # ------------------------------------------------------ representations
    def _to_manifold(self, coordinates):
        if self.geometry == "sl":
            return algebra_exp(coordinates_to_algebra(coordinates, self.matrix_dim, basis=self.sl_basis))
        return coordinates

    def entity_coordinates(self, entity_ids):
        fused = super().get_entity_representations(entity_ids)
        if self.projection is not None:
            coordinates = self.projection(fused)
        elif self.entity_mapping == "fixed_pad":
            coordinates = F.pad(fused, (0, self.coordinate_dim - self.embedding_dim))
        else:
            coordinates = fused
        return radial_clip(coordinates * self.coordinate_scale, self.chart_radius)

    def get_geometry_representations(self, entity_ids):
        coordinates = self.entity_coordinates(entity_ids)
        return coordinates, self._to_manifold(coordinates)

    def relation_coordinates(self, relation_ids):
        raw = self.relation_embeddings(relation_ids)
        parts = raw.split(self.coordinate_dim, dim=-1)
        return tuple(radial_clip(part, self.relation_radius) for part in parts)

    def get_relation_representations(self, relation_ids):
        return tuple(self._to_manifold(part) for part in self.relation_coordinates(relation_ids))

    def _cache_signature(self):
        return (self.geometry, self.entity_mapping, self.coordinate_scale, self.chart_radius, self.relation_radius,
                tuple((id(p), p._version) for p in self.parameters()),
                tuple((id(b), b._version) for b in self.buffers()))

    def _cached_entities(self):
        signature = self._cache_signature()
        if self._evaluation_cache is None or signature != self._evaluation_cache_signature:
            blocks = []
            for start in range(0, self.num_entities, self.score_chunk):
                ids = torch.arange(start, min(start + self.score_chunk, self.num_entities),
                                   device=self.score_offset.device)
                blocks.append(self.get_geometry_representations(ids)[1])
            self._evaluation_cache = torch.cat(blocks)
            self._evaluation_cache_signature = signature
        return self._evaluation_cache

    # ---------------------------------------------------------------- score
    def transform_head(self, head, relation_parts):
        if self.geometry == "sl":
            if self.relation_mode == "sandwich":
                return relation_parts[0] @ head @ relation_parts[1]
            return relation_parts[0] @ head
        if self.relation_mode == "sandwich":
            return head + relation_parts[0] + relation_parts[1]
        return head + relation_parts[0]

    def discrepancy(self, transformed_head, tail):
        if self.geometry == "sl":
            distance = sl_distance(transformed_head, tail, p=self.schatten_p, **self.log_kwargs)
        else:
            distance = torch.linalg.vector_norm(transformed_head - tail, dim=-1)
        return distance.square() if self.score_form == "squared" else distance

    def _distance_block(self, head, tail, *relation_parts):
        return self.discrepancy(self.transform_head(head, relation_parts), tail)

    def forward(self, head, relation, tail):
        if head.ndim != 1 or head.shape != relation.shape or head.shape != tail.shape:
            raise ValueError("head, relation and tail must be matching 1-D ID tensors")
        count = head.numel()
        if count == 0:
            return self.score_offset.new_empty((0,))
        if not self.training and not torch.is_grad_enabled():
            entity_repr = self._cached_entities()
            head_map, tail_map = head, tail
        else:
            entity_ids, entity_map = torch.unique(torch.cat((head, tail)), return_inverse=True)
            entity_repr = self.get_geometry_representations(entity_ids)[1]
            head_map, tail_map = entity_map[:count], entity_map[count:]
        relation_ids, relation_map = torch.unique(relation, return_inverse=True)
        relation_parts = self.get_relation_representations(relation_ids)
        distances = []
        for start in range(0, count, self.score_chunk):
            stop = start + self.score_chunk
            hm, tm, rm = head_map[start:stop], tail_map[start:stop], relation_map[start:stop]
            values = (entity_repr[hm], entity_repr[tm]) + tuple(part[rm] for part in relation_parts)
            if self.checkpoint_blocks and torch.is_grad_enabled():
                distance = checkpoint(self._distance_block, *values, use_reentrant=False,
                                      preserve_rng_state=False)
            else:
                distance = self._distance_block(*values)
            distances.append(distance)
        return self.score_offset - self.logit_scale * torch.cat(distances)

    # ---------------------------------------------------------- diagnostics
    @torch.no_grad()
    def diagnostics(self, sample_size=256):
        count = min(max(int(sample_size), 1), self.num_entities)
        device = self.score_offset.device
        ids = torch.linspace(0, self.num_entities - 1, steps=count, device=device).round().long()
        tails = (ids * 17 + 1) % self.num_entities
        relations = ids % self.num_relations
        head_coord, head = self.get_geometry_representations(ids)
        tail_coord, tail = self.get_geometry_representations(tails)
        relation_coord = self.relation_coordinates(relations)
        relation_parts = tuple(self._to_manifold(part) for part in relation_coord)
        entity_norm = head_coord.norm(dim=-1)
        relation_norm = relation_coord[0].norm(dim=-1)
        transformed = self.transform_head(head, relation_parts)
        output = {
            "geometry": self.geometry, "matrix_dim": self.matrix_dim, "coordinate_dim": self.coordinate_dim,
            "entity_mapping": self.entity_mapping, "fused_dimension": self.embedding_dim,
            "learned_entity_projection": self.projection is not None,
            "sample_size": count, "diagnostic_scope": "deterministic_spread_entity_sample_not_all_triples",
            "coordinate_scale": self.coordinate_scale, "chart_radius": self.chart_radius,
            "relation_radius": self.relation_radius, "relation_mode": self.relation_mode,
            "score_form": self.score_form, "schatten_p": self.schatten_p, "log": dict(self.log_kwargs),
            "entity_coordinate_norm": {"max": float(entity_norm.max()), "mean": float(entity_norm.mean()),
                                       "at_clip_fraction": float((entity_norm >= self.chart_radius - 1e-6).float().mean())},
            "relation_coordinate_norm": {"max": float(relation_norm.max()), "mean": float(relation_norm.mean()),
                                         "at_clip_fraction": float((relation_norm >= self.relation_radius - 1e-6).float().mean())},
            "logit_scale": float(self.logit_scale), "score_offset": float(self.score_offset),
            "sample_discrepancy": {"mean": float(self.discrepancy(transformed, tail).mean()),
                                   "self_candidate_mean": float(self.discrepancy(transformed, head).mean())},
            "parameters_finite": all(bool(torch.isfinite(p).all()) for p in self.parameters()),
        }
        if self.geometry == "sl":
            relatives = relative_matrix(transformed, tail)
            output["entities"] = spectrum_diagnostics(head)
            output["relations"] = spectrum_diagnostics(relation_parts[0])
            output["relative_score_matrices"] = spectrum_diagnostics(relatives)
            output["relative_score_matrices"].update(log_accuracy_check(relatives, **self.log_kwargs))
            output["relative_score_matrices"]["max_abs_det_minus_one"] = float(
                (torch.linalg.det(relatives.double()) - 1).abs().max())
        else:
            output["nonfinite_fraction"] = float((~torch.isfinite(transformed).all(dim=-1)).float().mean())
        return output
