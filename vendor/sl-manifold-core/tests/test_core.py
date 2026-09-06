import math

import pytest
import torch

from sl_manifold import (
    SLManifold,
    algebra_exp,
    algebra_to_coordinates,
    cayley_log,
    coordinates_to_algebra,
    lie_euler_step,
    ordered_compose,
    orthonormal_sl_basis,
    pairwise_distance,
    project_to_sl_algebra,
    relative_log,
    schatten_norm,
    sl_diagnostics,
    summarize_sl_diagnostics,
    symmetric_distance,
)


@pytest.mark.parametrize("matrix_dim", [2, 3, 4, 8])
def test_basis_is_trace_free_and_frobenius_orthonormal(matrix_dim):
    basis = orthonormal_sl_basis(matrix_dim, dtype=torch.float64)
    gram = torch.einsum("aij,bij->ab", basis, basis)
    torch.testing.assert_close(
        gram,
        torch.eye(matrix_dim * matrix_dim - 1, dtype=torch.float64),
        rtol=2e-14,
        atol=2e-14,
    )
    torch.testing.assert_close(
        basis.diagonal(dim1=-2, dim2=-1).sum(dim=-1),
        torch.zeros(matrix_dim * matrix_dim - 1, dtype=torch.float64),
        rtol=0.0,
        atol=2e-15,
    )


def test_coordinate_matrix_roundtrip_and_isometry():
    torch.manual_seed(1)
    coordinates = torch.randn(2, 5, 15, dtype=torch.float64)
    algebra = coordinates_to_algebra(coordinates, matrix_dim=4)
    recovered = algebra_to_coordinates(algebra)

    torch.testing.assert_close(recovered, coordinates, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(
        torch.linalg.vector_norm(coordinates, dim=-1),
        torch.linalg.matrix_norm(algebra, ord="fro", dim=(-2, -1)),
        rtol=2e-14,
        atol=2e-14,
    )


def test_projection_is_trace_free_and_discards_only_identity_component():
    torch.manual_seed(2)
    matrix = torch.randn(7, 3, 3, dtype=torch.float64)
    projected = project_to_sl_algebra(matrix)
    trace = projected.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    torch.testing.assert_close(trace, torch.zeros_like(trace), rtol=0.0, atol=2e-15)
    expected_difference = (
        matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    )[..., None, None] * torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(matrix - projected, expected_difference)


@pytest.mark.parametrize("matrix_dim", [2, 4, 8])
def test_exp_of_coordinates_has_unit_determinant(matrix_dim):
    torch.manual_seed(matrix_dim)
    manifold = SLManifold(matrix_dim, coordinate_scale=0.04).double()
    coordinates = torch.randn(6, matrix_dim * matrix_dim - 1, dtype=torch.float64)
    group = manifold.exp(coordinates)
    torch.testing.assert_close(
        torch.linalg.det(group),
        torch.ones(6, dtype=torch.float64),
        # torch.matrix_exp uses different optimized kernels across CPU/GPU;
        # n=8 on Apple Accelerate shows determinant drift around 4e-9.
        rtol=1e-8,
        atol=1e-8,
    )


def test_cayley_log_matches_exact_commuting_diagonal_log():
    value = 0.27
    matrix = torch.diag(
        torch.tensor([math.exp(value), math.exp(-value)], dtype=torch.float64)
    )
    actual = cayley_log(matrix, terms=12)
    expected = torch.diag(torch.tensor([value, -value], dtype=torch.float64))
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_relative_log_is_left_invariant_in_local_chart():
    torch.manual_seed(4)
    manifold = SLManifold(3, coordinate_scale=0.03, log_jitter=0.0).double()
    left = manifold.exp(torch.randn(5, 8, dtype=torch.float64))
    right = manifold.exp(torch.randn(5, 8, dtype=torch.float64))
    factor = manifold.exp(torch.randn(5, 8, dtype=torch.float64))
    torch.testing.assert_close(
        relative_log(factor @ left, factor @ right),
        relative_log(left, right),
        rtol=4e-12,
        atol=4e-12,
    )


def test_schatten_two_is_frobenius_not_spectral_norm():
    matrix = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(schatten_norm(matrix, p=2), torch.tensor(5.0, dtype=torch.float64))
    torch.testing.assert_close(
        schatten_norm(matrix, p=float("inf")), torch.tensor(4.0, dtype=torch.float64)
    )


def test_symmetric_distance_swaps_inputs_and_supports_paired_batches():
    torch.manual_seed(9)
    manifold = SLManifold(3, coordinate_scale=0.03).double()
    left = manifold.exp(torch.randn(5, 8, dtype=torch.float64))
    right = manifold.exp(torch.randn(5, 8, dtype=torch.float64))
    torch.testing.assert_close(
        symmetric_distance(left, right),
        symmetric_distance(right, left),
        rtol=2e-13,
        atol=2e-13,
    )


def test_distance_projects_jittered_local_log_back_to_sl_algebra():
    matrix = torch.diag(
        torch.tensor([math.exp(0.4), math.exp(-0.4)], dtype=torch.float64)
    )
    identity = torch.eye(2, dtype=torch.float64)
    forward = relative_log(identity, matrix, jitter=1e-3, trace_project=False)
    reverse = relative_log(matrix, identity, jitter=1e-3, trace_project=False)
    assert forward.trace().abs().item() > 1e-8
    expected = 0.5 * (
        schatten_norm(project_to_sl_algebra(forward))
        + schatten_norm(project_to_sl_algebra(reverse))
    )
    torch.testing.assert_close(
        symmetric_distance(identity, matrix, jitter=1e-3), expected
    )


def test_diagnostics_are_exact_for_identity_and_manifest_serializable():
    identity = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    diagnostics = sl_diagnostics(identity)
    torch.testing.assert_close(
        diagnostics.log_abs_det, torch.zeros(4, dtype=torch.float64)
    )
    torch.testing.assert_close(
        diagnostics.determinant_sign, torch.ones(4, dtype=torch.float64)
    )
    torch.testing.assert_close(
        diagnostics.condition_number, torch.ones(4, dtype=torch.float64)
    )
    torch.testing.assert_close(
        diagnostics.cayley_spectral_norm, torch.zeros(4, dtype=torch.float64)
    )
    assert not diagnostics.branch_risk.any()
    assert not diagnostics.nonfinite.any()

    summary = summarize_sl_diagnostics(identity)
    assert summary["num_matrices"] == 4
    assert summary["max_abs_log_det"] == 0.0
    assert summary["max_condition_number"] == 1.0
    assert summary["branch_risk_fraction"] == 0.0
    assert all(isinstance(value, (int, float)) for value in summary.values())


def test_gregory_remainder_bound_is_recorded_and_grows_toward_chart_boundary():
    near = torch.diag(torch.tensor([math.exp(0.1), math.exp(-0.1)], dtype=torch.float64))
    farther = torch.diag(torch.tensor([math.exp(1.5), math.exp(-1.5)], dtype=torch.float64))
    summary_near = summarize_sl_diagnostics(near, gregory_terms=12)
    summary_farther = summarize_sl_diagnostics(farther, gregory_terms=12)
    assert summary_near["gregory_terms"] == 12
    assert summary_near["max_gregory_frobenius_remainder_bound"] >= 0.0
    assert (
        summary_farther["max_gregory_frobenius_remainder_bound"]
        > summary_near["max_gregory_frobenius_remainder_bound"]
    )


def test_diagnostics_flag_cayley_branch_boundary_without_crashing():
    # -I belongs to SL(2), but A+I is singular and the local Cayley chart
    # cannot represent it.  Diagnostics must record the failure, not raise.
    negative_identity = -torch.eye(2, dtype=torch.float64)
    diagnostics = sl_diagnostics(negative_identity)
    assert diagnostics.determinant_sign.item() == 1.0
    assert math.isinf(diagnostics.cayley_spectral_norm.item())
    assert diagnostics.branch_risk.item()


def test_diagnostics_report_ill_conditioned_but_unit_determinant_point():
    matrix = torch.diag(torch.tensor([1e5, 1e-5], dtype=torch.float64))
    diagnostics = sl_diagnostics(matrix, branch_norm_threshold=0.99)
    torch.testing.assert_close(
        diagnostics.log_abs_det, torch.tensor(0.0, dtype=torch.float64)
    )
    assert diagnostics.condition_number.item() == pytest.approx(1e10)
    assert diagnostics.cayley_spectral_norm.item() > 0.99
    assert diagnostics.branch_risk.item()


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_two_axis_chunked_pairwise_values_and_gradients_match_dense(checkpoint_blocks):
    torch.manual_seed(5)
    manifold = SLManifold(3, coordinate_scale=0.025).double()
    left_coordinates = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    right_coordinates = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    left = manifold.exp(left_coordinates)
    right = manifold.exp(right_coordinates)

    dense = pairwise_distance(left, right)
    chunked = pairwise_distance(
        left,
        right,
        left_chunk=2,
        right_chunk=3,
        checkpoint_blocks=checkpoint_blocks,
    )
    torch.testing.assert_close(chunked, dense, rtol=2e-13, atol=2e-13)

    dense.square().sum().backward(retain_graph=True)
    dense_left_gradient = left_coordinates.grad.detach().clone()
    dense_right_gradient = right_coordinates.grad.detach().clone()
    left_coordinates.grad.zero_()
    right_coordinates.grad.zero_()
    chunked.square().sum().backward()
    torch.testing.assert_close(left_coordinates.grad, dense_left_gradient, rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(right_coordinates.grad, dense_right_gradient, rtol=2e-11, atol=2e-12)


def test_ordered_compose_respects_noncommutativity_mask_and_prefixes():
    first = torch.tensor([[1.0, 0.4], [0.0, 1.0]], dtype=torch.float64)
    second = torch.tensor([[1.0, 0.0], [0.7, 1.0]], dtype=torch.float64)
    elements = torch.stack((first, second))

    left = ordered_compose(elements, side="left")
    right = ordered_compose(elements, side="right")
    torch.testing.assert_close(left, second @ first)
    torch.testing.assert_close(right, first @ second)
    assert not torch.allclose(left, right)

    prefixes = ordered_compose(elements, side="left", return_prefixes=True)
    torch.testing.assert_close(prefixes[0], first)
    torch.testing.assert_close(prefixes[1], second @ first)
    masked = ordered_compose(elements, side="left", mask=torch.tensor([True, False]))
    torch.testing.assert_close(masked, first)


@pytest.mark.parametrize("trivialization", ["body", "spatial"])
def test_lie_euler_stays_in_sl_and_matches_defined_update(trivialization):
    torch.manual_seed(6)
    raw_group_algebra = project_to_sl_algebra(torch.randn(4, 3, 3, dtype=torch.float64))
    group = algebra_exp(raw_group_algebra, scale=0.03)
    raw_velocity = torch.randn(4, 3, 3, dtype=torch.float64, requires_grad=True)
    step_size = 0.07

    actual = lie_euler_step(
        group,
        raw_velocity,
        step_size,
        trivialization=trivialization,
    )
    velocity = project_to_sl_algebra(raw_velocity)
    increment = algebra_exp(velocity, scale=step_size)
    expected = group @ increment if trivialization == "body" else increment @ group
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.linalg.det(actual),
        torch.ones(4, dtype=torch.float64),
        rtol=2e-11,
        atol=2e-11,
    )
    actual.square().sum().backward()
    assert raw_velocity.grad is not None
    assert torch.isfinite(raw_velocity.grad).all()


def test_lie_euler_accepts_per_example_step_sizes():
    torch.manual_seed(7)
    group = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    velocity = project_to_sl_algebra(torch.randn(4, 3, 3, dtype=torch.float64))
    step_sizes = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
    actual = lie_euler_step(group, velocity, step_sizes)
    expected = torch.stack(
        [algebra_exp(velocity[index], step_sizes[index]) for index in range(4)]
    )
    torch.testing.assert_close(actual, expected)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="at least two"):
        orthonormal_sl_basis(1)
    with pytest.raises(ValueError, match="expected 8"):
        coordinates_to_algebra(torch.zeros(2, 7), 3)
    with pytest.raises(ValueError, match="non-negative"):
        cayley_log(torch.eye(2), jitter=-1e-3)
    with pytest.raises(ValueError, match="non-negative"):
        pairwise_distance(torch.eye(2)[None], torch.eye(2)[None], left_chunk=-1)
    with pytest.raises(ValueError, match="at least one"):
        ordered_compose(torch.empty(0, 2, 2))
