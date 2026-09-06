"""Sample-only accuracy audit of the unchanged shared Gregory-12 kernel.

Finite values and successful linear solves are distinct from approximation
accuracy. SciPy reference/reconstruction failures of tolerance are recorded,
not silently promoted to global validity or used as Cayley hard rejection.
There is deliberately no infinite Gregory-remainder bound in this output.
"""

import math
import numpy as np
import torch
from scipy.linalg import logm

from sl_manifold.core import relative_log


@torch.no_grad()
def gregory_pair_diagnostics(left, right, *, reference_tolerance=1e-3,
                             reconstruction_tolerance=1e-3):
    """Audit actual scoring-precision logs on caller-supplied training pairs."""
    if left.numel() == 0 or right.numel() == 0:
        raise ValueError("Gregory audit requires nonempty training pairs")
    left, right = torch.broadcast_tensors(left, right)
    dimension = left.shape[-1]
    left, right = left.reshape(-1, dimension, dimension), right.reshape(-1, dimension, dimension)
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise FloatingPointError("nonfinite Gregory audit input matrices")
    # These are the exact parameters and trace projection used in scoring.
    forward = relative_log(left, right, terms=12, jitter=1e-7, trace_project=True)
    reverse = relative_log(right, left, terms=12, jitter=1e-7, trace_project=True)
    if not bool(torch.isfinite(forward).all() and torch.isfinite(reverse).all()):
        raise FloatingPointError("nonfinite actual Gregory-12 logarithm")
    a, b = left.double(), right.double()
    relative, inverse = torch.linalg.solve(a, b), torch.linalg.solve(b, a)
    f, r = forward.double(), reverse.double()
    reconstructed_f, reconstructed_r = torch.matrix_exp(f), torch.matrix_exp(r)
    if not bool(torch.isfinite(relative).all() and torch.isfinite(inverse).all()
                and torch.isfinite(reconstructed_f).all() and torch.isfinite(reconstructed_r).all()):
        raise FloatingPointError("nonfinite actual Gregory reconstruction or relative solve")
    reconstruction_f = (reconstructed_f - relative).norm(dim=(-2, -1)) / relative.norm(dim=(-2, -1)).clamp_min(1e-15)
    reconstruction_r = (reconstructed_r - inverse).norm(dim=(-2, -1)) / inverse.norm(dim=(-2, -1)).clamp_min(1e-15)
    eigenvalues = torch.linalg.eigvals(relative)
    on_cut = (eigenvalues.real <= 0) & (eigenvalues.imag == 0)
    identity = torch.eye(dimension, dtype=relative.dtype, device=relative.device)
    # Unavailable legacy chart statistics remain informative, JSON-safe and
    # never participate in the finite/accuracy pass conditions below.
    cayley, info = torch.linalg.solve_ex(relative + identity, relative - identity, check_errors=False)
    available = (info == 0) & torch.isfinite(cayley).all(dim=(-2, -1))
    safe_cayley = torch.where(available[:, None, None], cayley, torch.zeros_like(cayley))
    cayley_norm = torch.linalg.matrix_norm(safe_cayley, ord=2, dim=(-2, -1))
    availability_count = int(available.sum())
    scipy_errors_f, scipy_errors_r, scipy_imaginary = [], [], []
    pair_records = []
    for index in range(len(relative)):
        reference_f = logm(relative[index].cpu().numpy())
        reference_r = logm(inverse[index].cpu().numpy())
        if not np.isfinite(reference_f).all() or not np.isfinite(reference_r).all():
            raise FloatingPointError("nonfinite independent SciPy logm reference")
        error_f = float(np.linalg.norm(f[index].cpu().numpy() - reference_f, ord="fro")
                        / max(1.0, np.linalg.norm(reference_f, ord="fro")))
        error_r = float(np.linalg.norm(r[index].cpu().numpy() - reference_r, ord="fro")
                        / max(1.0, np.linalg.norm(reference_r, ord="fro")))
        imaginary = float(max(np.linalg.norm(np.imag(reference_f), ord="fro"),
                              np.linalg.norm(np.imag(reference_r), ord="fro")))
        scipy_errors_f.append(error_f); scipy_errors_r.append(error_r); scipy_imaginary.append(imaginary)
        pair_records.append({"sample_index": index,
            "forward_reconstruction_relative_error": float(reconstruction_f[index]),
            "reverse_reconstruction_relative_error": float(reconstruction_r[index]),
            "forward_scipy_logm_scaled_error": error_f, "reverse_scipy_logm_scaled_error": error_r,
            "scipy_logm_imaginary_frobenius_norm": imaginary,
            "cayley_spectral_norm": float(cayley_norm[index]) if bool(available[index]) else None,
            "principal_branch_cut_eigenvalue_count": int(on_cut[index].sum())})
    reference_ok = (max(scipy_errors_f + scipy_errors_r) <= reference_tolerance
                    and max(scipy_imaginary) <= reference_tolerance)
    reconstruction_ok = float(torch.maximum(reconstruction_f, reconstruction_r).max()) <= reconstruction_tolerance
    return {
        "backend": "gregory12", "log_order": 12, "terms": 12,
        "jitter": 1e-7, "trace_project": True, "sample_matrix_pairs": len(relative),
        "scope": "caller_supplied_fixed_training_probe_only_not_all_scored_pairs",
        "scipy_reference_used": True, "reference_tolerance": reference_tolerance,
        "reconstruction_tolerance": reconstruction_tolerance,
        "finite_and_solve_checks_passed": True,
        "passed": True, "passed_scope": "finite_values_and_linear_solves_only_not_approximation_accuracy",
        "reference_accuracy_passed": bool(reference_ok and reconstruction_ok),
        "reference_accuracy_failure_is_fatal": False,
        "max_forward_reconstruction_relative_error": float(reconstruction_f.max()),
        "max_reverse_reconstruction_relative_error": float(reconstruction_r.max()),
        "max_scipy_logm_scaled_error": max(scipy_errors_f + scipy_errors_r),
        "max_scipy_logm_imaginary_frobenius_norm": max(scipy_imaginary),
        "max_absolute_log_trace": float(f.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()),
        "max_inverse_log_consistency_frobenius_norm": float((f + r).norm(dim=(-2, -1)).max()),
        "max_relative_condition_number": float(torch.linalg.cond(relative).max()),
        "principal_branch_cut_eigenvalue_count": int(on_cut.sum()),
        "cayley_diagnostic_status": "ok" if availability_count == len(relative) else "unavailable" if availability_count == 0 else "partially_unavailable",
        "max_cayley_spectral_norm": float(cayley_norm[available].max()) if availability_count else None,
        "cayley_spectral_norm_ge_one_count": int((available & (cayley_norm >= 1)).sum()),
        "cayley_unavailable_count": len(relative) - availability_count,
        "cayley_warning_is_fatal": False,
        "per_pair_accuracy": pair_records,
    }
