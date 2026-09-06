"""Differentiable Gauss--Legendre matrix logarithm quadrature.

For matrices whose spectrum avoids the nonpositive real axis,
log(M) = integral_0^1 [I+t(M-I)]^-1(M-I) dt.
Reference: https://nhigham.com/2020/11/17/what-is-the-matrix-logarithm/

This is fixed-degree integral quadrature, NOT the complete adaptive Higham
inverse-scaling-and-squaring/Schur algorithm. Orders 16 and 32 are supported.
Neither membership in SL(n) nor agreement of two quadrature orders proves
principal-log validity. Sample diagnostics additionally inspect eigenvalues,
actual exponential reconstruction and optional independent scipy.logm.
"""

from functools import lru_cache
import math

import numpy as np
import torch


@lru_cache(maxsize=2)
def _rule_numpy(order):
    if order not in (16, 32):
        raise ValueError("quadrature order must be 16 or 32")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return ((nodes + 1) / 2).copy(), (weights / 2).copy()


def quadrature_rule(order=16, *, dtype=torch.float32, device=None):
    nodes, weights = _rule_numpy(order)
    return (torch.tensor(nodes, dtype=dtype, device=device),
            torch.tensor(weights, dtype=dtype, device=device))


def relative_log_quadrature(left, right, *, order=16, nodes=None, weights=None,
                            check_errors=False):
    """Approximate log(left^-1 right) without forming an explicit inverse.

    Broadcastable inputs end in [n,n]. The equivalent integrand is
    solve((1-t)*left+t*right, right-left). Batched solve_ex avoids one CPU
    synchronization per candidate block. Singular solves are marked NaN;
    training must check loss/gradients and invoke sampled diagnostics.
    ``check_errors=True`` additionally asks PyTorch to raise on failed solves.
    """
    if left.ndim < 2 or right.ndim < 2 or left.shape[-1] != left.shape[-2] or right.shape[-2:] != left.shape[-2:]:
        raise ValueError("left/right must end in matching square matrices")
    if left.dtype not in (torch.float32, torch.float64) or right.dtype != left.dtype:
        raise TypeError("matrix-log quadrature requires matching float32/float64 inputs")
    if left.device != right.device:
        raise ValueError("matrix-log inputs must share a device")
    if order not in (16, 32):
        raise ValueError("quadrature order must be 16 or 32")
    if nodes is None and weights is None:
        nodes, weights = quadrature_rule(order, dtype=left.dtype, device=left.device)
    elif nodes is None or weights is None or nodes.shape != (order,) or weights.shape != (order,):
        raise ValueError("provide both quadrature node/weight vectors with the requested order")
    if nodes.dtype != left.dtype or nodes.device != left.device or weights.dtype != left.dtype or weights.device != left.device:
        raise ValueError("quadrature rule must match input dtype/device")
    difference = right - left
    coefficients = left.unsqueeze(-3) + nodes[:, None, None] * difference.unsqueeze(-3)
    rhs = difference.unsqueeze(-3).expand_as(coefficients)
    integrand, info = torch.linalg.solve_ex(coefficients, rhs, check_errors=check_errors)
    integrand = torch.where(info[..., None, None] == 0, integrand,
                            torch.full_like(integrand, float("nan")))
    return (integrand * weights[:, None, None]).sum(dim=-3)


def matrix_log_quadrature(matrix, *, order=16, **kwargs):
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    return relative_log_quadrature(identity, matrix, order=order, **kwargs)


def symmetric_squared_log_distance(left, right, *, order=16, nodes=None, weights=None):
    """Square of the mean forward/reverse principal-log Frobenius norms.

    On the exact principal domain log(M^-1)=-log(M), so both norms coincide.
    Evaluating both preserves the previous symmetric discrepancy convention.
    This is not claimed to be a global Riemannian geodesic distance.
    """
    forward = relative_log_quadrature(left, right, order=order, nodes=nodes, weights=weights)
    reverse = relative_log_quadrature(right, left, order=order, nodes=nodes, weights=weights)
    distance = 0.5 * (torch.linalg.vector_norm(forward.flatten(-2), dim=-1)
                      + torch.linalg.vector_norm(reverse.flatten(-2), dim=-1))
    return distance.square()


def checked_symmetric_log_distance(left, right, *, order=16, nodes=None, weights=None,
                                   reconstruction_tolerance=1e-3, branch_tolerance=1e-6):
    """Linear discrepancy with detached spectrum/reconstruction checks per block.

    Near-cut failures invalidate this numerical principal-log chart, NOT the
    SL group element itself. No Cayley-norm sufficient condition is used.
    """
    forward = relative_log_quadrature(left, right, order=order, nodes=nodes, weights=weights)
    reverse = relative_log_quadrature(right, left, order=order, nodes=nodes, weights=weights)
    with torch.no_grad():
        relative = torch.linalg.solve(left.detach(), right.detach())
        inverse = torch.linalg.solve(right.detach(), left.detach())
        if not bool(torch.isfinite(relative).all() and torch.isfinite(inverse).all()
                    and torch.isfinite(forward).all() and torch.isfinite(reverse).all()):
            raise FloatingPointError("nonfinite SL relative matrix/logarithm")
        eigenvalues = torch.linalg.eigvals(relative)
        near_cut = ((eigenvalues.real <= 0)
                    & (eigenvalues.imag.abs() <= branch_tolerance * eigenvalues.abs().clamp_min(1)))
        if bool(near_cut.any() or (eigenvalues.abs() <= branch_tolerance).any()):
            raise FloatingPointError("SL relative matrix is on/within numerical tolerance of principal-log branch cut or zero spectrum; not invalid SL membership")
        error1 = (torch.matrix_exp(forward.detach()) - relative).norm(dim=(-2, -1)) / relative.norm(dim=(-2, -1)).clamp_min(1e-12)
        error2 = (torch.matrix_exp(reverse.detach()) - inverse).norm(dim=(-2, -1)) / inverse.norm(dim=(-2, -1)).clamp_min(1e-12)
        error = torch.maximum(error1, error2)
        if not bool(torch.isfinite(error).all()) or bool((error > reconstruction_tolerance).any()):
            raise FloatingPointError(f"SL log reconstruction exceeds {reconstruction_tolerance}: max={float(error.max())}")
    return 0.5 * (torch.linalg.vector_norm(forward.flatten(-2), dim=-1)
                  + torch.linalg.vector_norm(reverse.flatten(-2), dim=-1))


@torch.no_grad()
def log_pair_diagnostics(left, right, *, order=16, scipy_reference=False,
                         reconstruction_tolerance=1e-3, order_tolerance=1e-3,
                         raise_on_failure=True):
    """Diagnose the supplied actual relative matrices, not a Cayley bound.

    Actual scoring precision is used for the selected-order logarithms;
    reconstruction, eigenspectrum and independent order-32 comparison use
    float64. This is a sampled check, never proof for unobserved triples.
    """
    if left.numel() == 0:
        raise ValueError("diagnostics require a nonempty matrix sample")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise FloatingPointError("nonfinite SL matrices in log diagnostics")
    left, right = torch.broadcast_tensors(left, right)
    dimension = left.shape[-1]
    left, right = left.reshape(-1, dimension, dimension), right.reshape(-1, dimension, dimension)
    a, b = left.double(), right.double()
    relative = torch.linalg.solve(a, b)
    inverse = torch.linalg.solve(b, a)
    log_forward = relative_log_quadrature(left, right, order=order).double()
    log_reverse = relative_log_quadrature(right, left, order=order).double()
    reference32 = relative_log_quadrature(a, b, order=32)
    reference16 = relative_log_quadrature(a, b, order=16)
    logs_finite = bool(torch.isfinite(log_forward).all() and torch.isfinite(log_reverse).all()
                       and torch.isfinite(reference32).all())
    if not logs_finite:
        raise FloatingPointError("nonfinite sampled principal-log quadrature")
    denominator = relative.norm(dim=(-2, -1)).clamp_min(1e-15)
    inverse_denominator = inverse.norm(dim=(-2, -1)).clamp_min(1e-15)
    reconstruction = (torch.matrix_exp(log_forward) - relative).norm(dim=(-2, -1)) / denominator
    reconstruction_reverse = (torch.matrix_exp(log_reverse) - inverse).norm(dim=(-2, -1)) / inverse_denominator
    # Use a unit floor so near-identity roundoff is not mislabeled a huge
    # relative error merely because the exact logarithm is zero.
    log_denominator = reference32.norm(dim=(-2, -1)).clamp_min(1.0)
    order_gap = (reference16 - reference32).norm(dim=(-2, -1)) / log_denominator
    selected_gap = (log_forward - reference32).norm(dim=(-2, -1)) / log_denominator
    inverse_gap = (log_forward + log_reverse).norm(dim=(-2, -1)) / log_denominator
    eigenvalues = torch.linalg.eigvals(relative)
    on_cut = (eigenvalues.real <= 0) & (eigenvalues.imag == 0)
    min_angle_to_cut = (math.pi - torch.angle(eigenvalues).abs()).min()
    sign, logdet = torch.linalg.slogdet(relative)
    # Informative legacy-chart diagnostic only. A large Cayley norm is a
    # failed sufficient convergence condition, not a principal-log failure.
    # Failed/singular solves remain JSON-safe unavailable entries and never
    # change ``passed`` or the actual spectrum/reconstruction checks below.
    identity = torch.eye(dimension, dtype=relative.dtype, device=relative.device)
    cayley, cayley_info = torch.linalg.solve_ex(relative + identity, relative - identity,
                                               check_errors=False)
    cayley_available = (cayley_info == 0) & torch.isfinite(cayley).all(dim=(-2, -1))
    safe_cayley = torch.where(cayley_available[:, None, None], cayley, torch.zeros_like(cayley))
    cayley_norms = torch.linalg.matrix_norm(safe_cayley, ord=2, dim=(-2, -1))
    available_count = int(cayley_available.sum())
    unavailable_count = len(relative) - available_count
    cayley_max = float(cayley_norms[cayley_available].max()) if available_count else None
    result = {
        "sample_matrix_pairs": len(relative), "log_order": order,
        "definition": "fixed_order_Gauss_Legendre_principal_relative_log_not_global_geodesic",
        "scope": "supplied_sample_not_all_scored_pairs", "logs_finite": logs_finite,
        "principal_branch_cut_eigenvalue_count": int(on_cut.sum()),
        "minimum_eigenvalue_absolute_value": float(eigenvalues.abs().min()),
        "minimum_eigenvalue_angle_to_negative_real_axis": float(min_angle_to_cut),
        "max_relative_condition_number": float(torch.linalg.cond(relative).max()),
        "max_relative_abs_logdet": float(logdet.abs().max()),
        "nonpositive_relative_determinant_count": int((sign <= 0).sum()),
        "max_forward_reconstruction_relative_error": float(reconstruction.max()),
        "max_reverse_reconstruction_relative_error": float(reconstruction_reverse.max()),
        "max_float64_order16_vs32_scaled_error": float(order_gap.max()),
        "max_selected_vs_float64_order32_scaled_error": float(selected_gap.max()),
        "max_inverse_log_consistency_scaled_error": float(inverse_gap.max()),
        "max_absolute_log_trace": float(log_forward.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()),
        "cayley_diagnostic_status": ("ok" if unavailable_count == 0 else
                                     "unavailable" if available_count == 0 else "partially_unavailable"),
        "max_cayley_spectral_norm": cayley_max,
        "cayley_spectral_norm_ge_one_count": int((cayley_available & (cayley_norms >= 1)).sum()),
        "cayley_unavailable_count": unavailable_count,
        "cayley_warning_is_fatal": False,
    }
    passed = (not bool(on_cut.any()) and bool((sign > 0).all())
              and float(torch.maximum(reconstruction.max(), reconstruction_reverse.max())) <= reconstruction_tolerance
              and float(selected_gap.max()) <= order_tolerance
              and (order == 32 or float(order_gap.max()) <= order_tolerance))
    if scipy_reference:
        from scipy.linalg import logm
        relative_numpy = relative.cpu().numpy()
        errors, imaginary = [], []
        for index, matrix in enumerate(relative_numpy):
            exact = logm(matrix)
            imaginary.append(float(np.linalg.norm(np.imag(exact), ord="fro")))
            errors.append(float(np.linalg.norm(log_forward[index].cpu().numpy() - exact, ord="fro")
                                / max(1.0, np.linalg.norm(exact, ord="fro"))))
        result.update(scipy_reference_used=True, max_scipy_logm_scaled_error=max(errors),
                      max_scipy_logm_imaginary_frobenius_norm=max(imaginary))
        passed = passed and max(errors) <= order_tolerance and max(imaginary) <= order_tolerance
    else:
        result["scipy_reference_used"] = False
    result["passed"] = bool(passed)
    if raise_on_failure and not passed:
        raise FloatingPointError("sampled SL principal-log validation failed: " + str(result))
    return result
