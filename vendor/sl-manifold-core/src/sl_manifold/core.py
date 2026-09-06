"""Small, differentiable building blocks for representation learning on ``SL(n)``.

The logarithm used by the accompanying SL(n) work is a truncated Gregory
series in Cayley coordinates.  It is reliable in a neighbourhood of the
identity, which is where scaled projection heads and Lie--Euler increments
operate.  It is deliberately exposed as :func:`cayley_log`: not every real
special-linear matrix has a real logarithm, so this module does not pretend to
provide a globally valid real principal logarithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

Trivialization = Literal["body", "spatial"]
MultiplicationSide = Literal["left", "right"]


@dataclass(frozen=True)
class SLDiagnostics:
    """Per-matrix numerical diagnostics, retaining arbitrary batch shape."""

    log_abs_det: Tensor
    determinant_sign: Tensor
    condition_number: Tensor
    cayley_spectral_norm: Tensor
    branch_risk: Tensor
    nonfinite: Tensor


def sl_dimension(matrix_dim: int) -> int:
    """Return the intrinsic dimension ``n^2 - 1`` of ``SL(n)``."""

    if matrix_dim < 2:
        raise ValueError("SL matrix dimension must be at least two")
    return matrix_dim * matrix_dim - 1


def _validate_square(matrix: Tensor, name: str = "matrix") -> int:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"{name} must end in a square matrix shape")
    if matrix.shape[-1] < 2:
        raise ValueError("SL matrix dimension must be at least two")
    return matrix.shape[-1]


def orthonormal_sl_basis(
    matrix_dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    r"""Construct a Frobenius-orthonormal basis of ``sl(n)``.

    The first ``n(n-1)`` basis vectors are row-major off-diagonal matrix units.
    The final ``n-1`` vectors form the standard orthonormal diagonal Cartan
    basis

    .. math::
       H_k = \operatorname{diag}(1,\ldots,1,-k,0,\ldots,0)/\sqrt{k(k+1)}.

    This convention makes coordinate Euclidean norm exactly equal to the
    matrix Frobenius norm.  It intentionally differs from the older ProCLIP
    helper, whose ``E_ii-E_nn`` diagonal coordinates are correlated.
    """

    dimension = sl_dimension(matrix_dim)
    dtype = dtype or torch.get_default_dtype()
    basis = torch.zeros(dimension, matrix_dim, matrix_dim, dtype=dtype, device=device)

    coordinate = 0
    for row in range(matrix_dim):
        for column in range(matrix_dim):
            if row == column:
                continue
            basis[coordinate, row, column] = 1.0
            coordinate += 1

    for k in range(1, matrix_dim):
        normalizer = math.sqrt(float(k * (k + 1)))
        basis[coordinate, torch.arange(k, device=basis.device), torch.arange(k, device=basis.device)] = (
            1.0 / normalizer
        )
        basis[coordinate, k, k] = -float(k) / normalizer
        coordinate += 1

    return basis


def coordinates_to_algebra(
    coordinates: Tensor,
    matrix_dim: int,
    *,
    basis: Optional[Tensor] = None,
) -> Tensor:
    """Map orthonormal coordinates to a trace-free ``n x n`` matrix."""

    expected = sl_dimension(matrix_dim)
    if coordinates.shape[-1] != expected:
        raise ValueError(
            f"expected {expected} SL({matrix_dim}) coordinates; got {coordinates.shape[-1]}"
        )
    if basis is None:
        basis = orthonormal_sl_basis(
            matrix_dim, dtype=coordinates.dtype, device=coordinates.device
        )
    expected_basis_shape = (expected, matrix_dim, matrix_dim)
    if basis.shape != expected_basis_shape:
        raise ValueError(f"basis must have shape {expected_basis_shape}; got {tuple(basis.shape)}")
    if basis.dtype != coordinates.dtype or basis.device != coordinates.device:
        basis = basis.to(dtype=coordinates.dtype, device=coordinates.device)
    return torch.einsum("...a,aij->...ij", coordinates, basis)


def algebra_to_coordinates(
    algebra: Tensor,
    *,
    basis: Optional[Tensor] = None,
) -> Tensor:
    """Return coordinates of the trace-free orthogonal projection of a matrix.

    For a trace-free input this is the exact inverse of
    :func:`coordinates_to_algebra`.  For a general square matrix, the identity
    component is discarded automatically because every basis vector is
    trace-free.
    """

    matrix_dim = _validate_square(algebra, "algebra")
    if basis is None:
        basis = orthonormal_sl_basis(
            matrix_dim, dtype=algebra.dtype, device=algebra.device
        )
    expected_basis_shape = (sl_dimension(matrix_dim), matrix_dim, matrix_dim)
    if basis.shape != expected_basis_shape:
        raise ValueError(f"basis must have shape {expected_basis_shape}; got {tuple(basis.shape)}")
    if basis.dtype != algebra.dtype or basis.device != algebra.device:
        basis = basis.to(dtype=algebra.dtype, device=algebra.device)
    return torch.einsum("...ij,aij->...a", algebra, basis)


def project_to_sl_algebra(matrix: Tensor) -> Tensor:
    """Orthogonally remove the scalar-identity component from a square matrix."""

    matrix_dim = _validate_square(matrix)
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    identity = torch.eye(matrix_dim, dtype=matrix.dtype, device=matrix.device)
    return matrix - (trace / float(matrix_dim))[..., None, None] * identity


def algebra_exp(algebra: Tensor, scale: Union[float, Tensor] = 1.0) -> Tensor:
    r"""Exponentiate a trace-free matrix into ``SL(n)``.

    Half and bfloat16 inputs are evaluated in float32 because matrix
    exponentials in low precision are not sufficiently stable.  The caller is
    responsible for passing a trace-free matrix; use
    :func:`project_to_sl_algebra` for unconstrained network outputs.
    """

    _validate_square(algebra, "algebra")
    if isinstance(scale, Tensor) and scale.ndim < algebra.ndim:
        # A per-example ODE step normally has the leading batch shape, e.g.
        # [B].  Append singleton matrix axes instead of relying on trailing
        # broadcasting, which would accidentally compare B with n.
        scale = scale.reshape(scale.shape + (1,) * (algebra.ndim - scale.ndim))
    scaled = algebra * scale
    work = scaled.float() if scaled.dtype in (torch.float16, torch.bfloat16) else scaled
    group = torch.matrix_exp(work)
    return group.to(dtype=scaled.dtype) if group.dtype != scaled.dtype else group


def cayley_transform(matrix: Tensor, jitter: float = 0.0) -> Tensor:
    r"""Return ``(A + (1+jitter)I)^{-1}(A-I)``.

    The Gregory series converges when the spectral radius of this transform is
    below one.  Its spectral norm is a convenient conservative diagnostic.
    """

    matrix_dim = _validate_square(matrix)
    if jitter < 0:
        raise ValueError("log jitter must be non-negative")
    identity = torch.eye(matrix_dim, dtype=matrix.dtype, device=matrix.device)
    return torch.linalg.solve(
        matrix + (1.0 + float(jitter)) * identity,
        matrix - identity,
    )


@torch.no_grad()
def sl_diagnostics(
    matrix: Tensor,
    *,
    jitter: float = 0.0,
    branch_norm_threshold: float = 1.0,
) -> SLDiagnostics:
    r"""Measure determinant, conditioning, and local-log chart safety.

    ``branch_risk`` is a conservative warning.  It is true when an input is
    non-finite, the Cayley solve fails, the determinant sign is non-positive,
    or ``||Cayley(A)||_2`` reaches ``branch_norm_threshold``.  The Gregory
    series is guaranteed by the stronger sufficient condition
    ``||Cayley(A)||_2 < 1``; a warning does not prove that no logarithm exists.

    This function never raises merely because a matrix is singular.  Invalid
    solves are represented by infinite condition/chart norms and a true risk
    flag so failed trajectories can still write diagnostics to a manifest.
    """

    matrix_dim = _validate_square(matrix)
    if matrix.numel() == 0:
        raise ValueError("diagnostics require at least one matrix")
    if jitter < 0:
        raise ValueError("log jitter must be non-negative")
    if not 0 < branch_norm_threshold <= 1:
        raise ValueError("branch norm threshold must lie in (0, 1]")

    work = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
    nonfinite = ~torch.isfinite(work).all(dim=(-2, -1))
    safe_matrix = torch.where(
        nonfinite[..., None, None],
        torch.zeros_like(work),
        work,
    )

    determinant_sign, log_abs_det = torch.linalg.slogdet(safe_matrix)
    singular_values = torch.linalg.svdvals(safe_matrix)
    smallest = singular_values[..., -1]
    condition_number = singular_values[..., 0] / smallest
    infinity = torch.full_like(condition_number, float("inf"))
    condition_number = torch.where(
        nonfinite | (smallest == 0), infinity, condition_number
    )

    identity = torch.eye(matrix_dim, dtype=work.dtype, device=work.device)
    cayley, solve_info = torch.linalg.solve_ex(
        safe_matrix + (1.0 + float(jitter)) * identity,
        safe_matrix - identity,
        check_errors=False,
    )
    cayley_nonfinite = ~torch.isfinite(cayley).all(dim=(-2, -1))
    safe_cayley = torch.where(
        cayley_nonfinite[..., None, None],
        torch.zeros_like(cayley),
        cayley,
    )
    cayley_spectral_norm = torch.linalg.matrix_norm(
        safe_cayley, ord=2, dim=(-2, -1)
    )
    solve_failed = solve_info != 0
    cayley_spectral_norm = torch.where(
        nonfinite | cayley_nonfinite | solve_failed,
        torch.full_like(cayley_spectral_norm, float("inf")),
        cayley_spectral_norm,
    )

    # Retain NaNs for determinant fields of non-finite input, while singular
    # but finite matrices naturally report sign=0 and log|det|=-inf.
    nan = torch.full_like(log_abs_det, float("nan"))
    log_abs_det = torch.where(nonfinite, nan, log_abs_det)
    determinant_sign = torch.where(nonfinite, nan, determinant_sign)
    branch_risk = (
        nonfinite
        | solve_failed
        | cayley_nonfinite
        | (determinant_sign <= 0)
        | (cayley_spectral_norm >= float(branch_norm_threshold))
    )
    return SLDiagnostics(
        log_abs_det=log_abs_det,
        determinant_sign=determinant_sign,
        condition_number=condition_number,
        cayley_spectral_norm=cayley_spectral_norm,
        branch_risk=branch_risk,
        nonfinite=nonfinite,
    )


@torch.no_grad()
def summarize_sl_diagnostics(
    matrix: Tensor,
    *,
    jitter: float = 0.0,
    branch_norm_threshold: float = 1.0,
    gregory_terms: Optional[int] = None,
) -> dict:
    """Return JSON-serializable aggregate diagnostics for an experiment manifest."""

    diagnostics = sl_diagnostics(
        matrix,
        jitter=jitter,
        branch_norm_threshold=branch_norm_threshold,
    )
    absolute_log_det = diagnostics.log_abs_det.abs().reshape(-1)
    condition_number = diagnostics.condition_number.reshape(-1)
    cayley_norm = diagnostics.cayley_spectral_norm.reshape(-1)
    result = {
        "num_matrices": int(absolute_log_det.numel()),
        "max_abs_log_det": float(absolute_log_det.max().item()),
        "mean_abs_log_det": float(absolute_log_det.mean().item()),
        "max_condition_number": float(condition_number.max().item()),
        "median_condition_number": float(condition_number.median().item()),
        "max_cayley_spectral_norm": float(cayley_norm.max().item()),
        "mean_cayley_spectral_norm": float(cayley_norm.mean().item()),
        "branch_norm_threshold": float(branch_norm_threshold),
        "branch_risk_fraction": float(
            diagnostics.branch_risk.float().mean().item()
        ),
        "nonfinite_fraction": float(diagnostics.nonfinite.float().mean().item()),
    }
    if gregory_terms is not None:
        if gregory_terms < 1:
            raise ValueError("Gregory log terms must be positive")
        # If r=||Cayley(A)||_2<1, the omitted atanh series has induced-2-norm
        # bound 2*r^(2K+1)/((2K+1)*(1-r^2)). Multiplying by sqrt(n) gives a
        # conservative Frobenius bound, matching the default Schatten-2 loss.
        inside = cayley_norm < 1.0
        safe = cayley_norm.clamp(max=1.0 - torch.finfo(cayley_norm.dtype).eps)
        bound = (
            2.0
            * math.sqrt(float(matrix.shape[-1]))
            * safe.pow(2 * int(gregory_terms) + 1)
            / (
                float(2 * int(gregory_terms) + 1)
                * (1.0 - safe.square())
            )
        )
        bound = torch.where(inside, bound, torch.full_like(bound, float("inf")))
        result.update(
            {
                "gregory_terms": int(gregory_terms),
                "log_jitter": float(jitter),
                "max_gregory_frobenius_remainder_bound": float(bound.max().item()),
                "mean_gregory_frobenius_remainder_bound": float(bound.mean().item()),
            }
        )
    return result


def _gregory_series(cayley: Tensor, terms: int) -> Tensor:
    if terms < 1:
        raise ValueError("Gregory log terms must be positive")
    if terms == 12:
        # Paterson--Stockmeyer-style grouping of the existing ProCLIP helper's
        # 12-term local approximation: seven matrix products instead of the
        # direct recurrence's twelve. This is not the manuscript's path-
        # refinement K, which denotes 1/2/4/8 piecewise-exponential segments.
        identity = torch.eye(
            cayley.shape[-1], dtype=cayley.dtype, device=cayley.device
        )
        z2 = cayley @ cayley
        z4 = z2 @ z2
        z6 = z4 @ z2
        block0 = identity + z2 / 3.0 + z4 / 5.0
        block1 = identity / 7.0 + z2 / 9.0 + z4 / 11.0
        block2 = identity / 13.0 + z2 / 15.0 + z4 / 17.0
        block3 = identity / 19.0 + z2 / 21.0 + z4 / 23.0
        polynomial = block2 + z6 @ block3
        polynomial = block1 + z6 @ polynomial
        polynomial = block0 + z6 @ polynomial
        return 2.0 * (cayley @ polynomial)

    cayley_squared = cayley @ cayley
    power = cayley
    series = cayley
    for index in range(1, terms):
        power = power @ cayley_squared
        series = series + power / float(2 * index + 1)
    return 2.0 * series


def cayley_log(matrix: Tensor, *, terms: int = 12, jitter: float = 0.0) -> Tensor:
    r"""Approximate a local real matrix logarithm with the Gregory series.

    ``terms=12`` matches the approximation used in the existing ProCLIP SL(n)
    experiments.  This routine is differentiable, but it is a local chart: for
    long compositions callers should monitor ``||cayley_transform(A)||_2`` or
    keep updates local instead of treating it as a global logarithm.
    """

    return _gregory_series(cayley_transform(matrix, jitter=jitter), terms)


def relative_log(
    left: Tensor,
    right: Tensor,
    *,
    terms: int = 12,
    jitter: float = 0.0,
    trace_project: bool = False,
) -> Tensor:
    r"""Approximate ``log(left^{-1} right)`` without materialising an inverse."""

    left_dim = _validate_square(left, "left")
    right_dim = _validate_square(right, "right")
    if left_dim != right_dim:
        raise ValueError("left and right must end in the same square matrix shape")
    if jitter < 0:
        raise ValueError("log jitter must be non-negative")

    # If R=left^{-1}right, then
    # (R+(1+j)I)^{-1}(R-I)=(right+(1+j)left)^{-1}(right-left).
    cayley = torch.linalg.solve(
        right + (1.0 + float(jitter)) * left,
        right - left,
    )
    logarithm = _gregory_series(cayley, terms)
    return project_to_sl_algebra(logarithm) if trace_project else logarithm


def schatten_norm(matrix: Tensor, p: float = 2.0) -> Tensor:
    """Compute the Schatten-p norm over the final two matrix dimensions."""

    _validate_square(matrix)
    if p < 1:
        raise ValueError("Schatten p must be at least one")
    if p == 2:
        return torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1))
    singular_values = torch.linalg.svdvals(matrix)
    if math.isinf(p):
        return singular_values.amax(dim=-1)
    return torch.linalg.vector_norm(singular_values, ord=p, dim=-1)


def directed_distance(
    left: Tensor,
    right: Tensor,
    *,
    p: float = 2.0,
    terms: int = 12,
    jitter: float = 1e-7,
    trace_project: bool = True,
) -> Tensor:
    r"""Return ``||log(left^{-1} right)||_{S_p}`` in the local Cayley chart."""

    return schatten_norm(
        relative_log(
            left,
            right,
            terms=terms,
            jitter=jitter,
            trace_project=trace_project,
        ),
        p=p,
    )


def symmetric_distance(
    left: Tensor,
    right: Tensor,
    *,
    p: float = 2.0,
    terms: int = 12,
    jitter: float = 1e-7,
    trace_project: bool = True,
) -> Tensor:
    """Symmetrize the two local directed distances for broadcastable inputs."""

    forward = directed_distance(
        left, right, p=p, terms=terms, jitter=jitter, trace_project=trace_project
    )
    reverse = directed_distance(
        right, left, p=p, terms=terms, jitter=jitter, trace_project=trace_project
    )
    return 0.5 * (forward + reverse)


def _symmetric_distance_block(
    left: Tensor,
    right: Tensor,
    p: float,
    terms: int,
    jitter: float,
    trace_project: bool,
) -> Tensor:
    return symmetric_distance(
        left[:, None],
        right[None, :],
        p=p,
        terms=terms,
        jitter=jitter,
        trace_project=trace_project,
    )


def pairwise_distance(
    left: Tensor,
    right: Tensor,
    *,
    p: float = 2.0,
    terms: int = 12,
    jitter: float = 1e-7,
    left_chunk: int = 0,
    right_chunk: int = 0,
    checkpoint_blocks: bool = False,
    trace_project: bool = True,
) -> Tensor:
    """Compute all symmetric SL distances with optional two-axis chunking.

    ``left`` and ``right`` must have shapes ``[N,n,n]`` and ``[M,n,n]``.
    Chunking both axes bounds each matrix-log block by
    ``left_chunk * right_chunk``.  Activation checkpointing additionally
    trades backward compute for memory.
    """

    left_dim = _validate_square(left, "left")
    right_dim = _validate_square(right, "right")
    if left.ndim != 3 or right.ndim != 3 or left_dim != right_dim:
        raise ValueError("left and right must have shapes [N,n,n] and [M,n,n] with the same n")
    if left_chunk < 0 or right_chunk < 0:
        raise ValueError("pairwise chunk sizes must be non-negative")

    left_chunk = left.shape[0] if left_chunk == 0 else left_chunk
    right_chunk = right.shape[0] if right_chunk == 0 else right_chunk
    if left_chunk < 1 or right_chunk < 1:
        raise ValueError("pairwise inputs must be non-empty")

    def kernel(left_block: Tensor, right_block: Tensor) -> Tensor:
        return _symmetric_distance_block(
            left_block, right_block, p, terms, jitter, trace_project
        )

    rows = []
    for left_start in range(0, left.shape[0], left_chunk):
        left_block = left[left_start : left_start + left_chunk]
        columns = []
        for right_start in range(0, right.shape[0], right_chunk):
            right_block = right[right_start : right_start + right_chunk]
            needs_grad = left_block.requires_grad or right_block.requires_grad
            if checkpoint_blocks and torch.is_grad_enabled() and needs_grad:
                block = activation_checkpoint(
                    kernel, left_block, right_block, use_reentrant=False
                )
            else:
                block = kernel(left_block, right_block)
            columns.append(block)
        rows.append(torch.cat(columns, dim=1))
    return torch.cat(rows, dim=0)


def pairwise_squared_distance(left: Tensor, right: Tensor, **kwargs: object) -> Tensor:
    """Squared version of :func:`pairwise_distance`."""

    return pairwise_distance(left, right, **kwargs).square()


def ordered_compose(
    elements: Tensor,
    *,
    initial: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    side: MultiplicationSide = "left",
    return_prefixes: bool = False,
) -> Tensor:
    r"""Compose a temporal sequence of group elements without losing order.

    The time axis is ``-3``.  With ``side='left'`` this implements the paper's
    recurrence ``G_t = g_t G_{t-1}``, hence the final state is
    ``g_T ... g_2 g_1 initial``.  ``side='right'`` implements
    ``G_t = G_{t-1} g_t``.  A false mask entry inserts the identity.
    """

    matrix_dim = _validate_square(elements, "elements")
    if elements.ndim < 3:
        raise ValueError("elements must include a time axis before the matrix axes")
    time_steps = elements.shape[-3]
    if time_steps < 1:
        raise ValueError("ordered composition requires at least one element")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    batch_shape = elements.shape[:-3]
    identity = torch.eye(matrix_dim, dtype=elements.dtype, device=elements.device)
    identity = identity.expand(batch_shape + (matrix_dim, matrix_dim))
    state = identity if initial is None else initial
    if state.shape[-2:] != (matrix_dim, matrix_dim):
        raise ValueError("initial must end in the same matrix shape as elements")

    if mask is not None:
        expected_mask_shape = batch_shape + (time_steps,)
        if mask.shape != expected_mask_shape:
            raise ValueError(f"mask must have shape {expected_mask_shape}; got {tuple(mask.shape)}")
        mask = mask.to(device=elements.device, dtype=torch.bool)

    prefixes = []
    for time_index in range(time_steps):
        element = elements[..., time_index, :, :]
        if mask is not None:
            element = torch.where(mask[..., time_index, None, None], element, identity)
        state = element @ state if side == "left" else state @ element
        if return_prefixes:
            prefixes.append(state)
    return torch.stack(prefixes, dim=-3) if return_prefixes else state


def lie_euler_step(
    group: Tensor,
    algebra_velocity: Tensor,
    step_size: Union[float, Tensor],
    *,
    trivialization: Trivialization = "body",
    project_velocity: bool = True,
) -> Tensor:
    r"""Take one Lie--Euler step while remaining on ``SL(n)``.

    ``trivialization='body'`` represents ``dot(G)=G Omega`` and updates with
    ``G exp(h Omega)``.  ``'spatial'`` represents ``dot(G)=Omega G`` and
    updates with ``exp(h Omega) G``.  By default unconstrained network output
    is projected to a trace-free velocity before exponentiation.
    """

    group_dim = _validate_square(group, "group")
    velocity_dim = _validate_square(algebra_velocity, "algebra_velocity")
    if group_dim != velocity_dim:
        raise ValueError("group and algebra_velocity must use the same matrix dimension")
    if trivialization not in ("body", "spatial"):
        raise ValueError("trivialization must be 'body' or 'spatial'")
    velocity = (
        project_to_sl_algebra(algebra_velocity)
        if project_velocity
        else algebra_velocity
    )
    increment = algebra_exp(velocity, scale=step_size)
    return group @ increment if trivialization == "body" else increment @ group


class SLManifold(nn.Module):
    """Reusable ``nn.Module`` wrapper that stores the orthonormal basis once."""

    def __init__(
        self,
        matrix_dim: int,
        *,
        coordinate_scale: float = 0.1,
        log_terms: int = 12,
        log_jitter: float = 1e-7,
    ) -> None:
        super().__init__()
        if coordinate_scale <= 0:
            raise ValueError("coordinate scale must be positive")
        if log_terms < 1:
            raise ValueError("Gregory log terms must be positive")
        if log_jitter < 0:
            raise ValueError("log jitter must be non-negative")
        self.matrix_dim = matrix_dim
        self.intrinsic_dim = sl_dimension(matrix_dim)
        self.coordinate_scale = float(coordinate_scale)
        self.log_terms = int(log_terms)
        self.log_jitter = float(log_jitter)
        self.register_buffer("basis", orthonormal_sl_basis(matrix_dim))

    def coordinates_to_algebra(self, coordinates: Tensor) -> Tensor:
        return coordinates_to_algebra(
            coordinates, self.matrix_dim, basis=self.basis
        )

    def algebra_to_coordinates(self, algebra: Tensor) -> Tensor:
        return algebra_to_coordinates(algebra, basis=self.basis)

    def exp(self, coordinates: Tensor, *, scale: Optional[float] = None) -> Tensor:
        algebra = self.coordinates_to_algebra(coordinates)
        return algebra_exp(
            algebra,
            scale=self.coordinate_scale if scale is None else scale,
        )

    def relative_log(
        self, left: Tensor, right: Tensor, *, trace_project: bool = False
    ) -> Tensor:
        return relative_log(
            left,
            right,
            terms=self.log_terms,
            jitter=self.log_jitter,
            trace_project=trace_project,
        )

    def directed_distance(self, left: Tensor, right: Tensor, *, p: float = 2.0) -> Tensor:
        return directed_distance(
            left,
            right,
            p=p,
            terms=self.log_terms,
            jitter=self.log_jitter,
        )

    def distance(self, left: Tensor, right: Tensor, *, p: float = 2.0) -> Tensor:
        return symmetric_distance(
            left,
            right,
            p=p,
            terms=self.log_terms,
            jitter=self.log_jitter,
        )

    def pairwise_distance(self, left: Tensor, right: Tensor, **kwargs: object) -> Tensor:
        return pairwise_distance(
            left,
            right,
            terms=self.log_terms,
            jitter=self.log_jitter,
            **kwargs,
        )

    def lie_euler_step_from_coordinates(
        self,
        group: Tensor,
        velocity_coordinates: Tensor,
        step_size: Union[float, Tensor],
        *,
        trivialization: Trivialization = "body",
    ) -> Tensor:
        velocity = self.coordinates_to_algebra(velocity_coordinates)
        return lie_euler_step(
            group,
            velocity,
            step_size,
            trivialization=trivialization,
            project_velocity=False,
        )
