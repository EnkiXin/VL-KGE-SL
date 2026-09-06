# Shared SL(n) geometry core

This package is the common geometry layer for the proposed MAI/CIR and FMA
experiments.  It is independent of `ProCLIP-reproduction`; no historical code,
checkpoints, manifests, or results are modified.

## Install during development

```bash
python -m pip install -e vendor/sl-manifold-core
```

Both downstream projects can then use the same API:

```python
from sl_manifold import SLManifold, ordered_compose

sl = SLManifold(matrix_dim=8, coordinate_scale=0.1).to(device)
token_points = sl.exp(token_head(token_features))       # [B,T,8,8]
query_point = ordered_compose(token_points, side="left")
distances = sl.pairwise_distance(query_point, image_points,
                                 left_chunk=32, right_chunk=128)
```

For a group-valued FMA state:

```python
omega_coordinates = velocity_network(state, time)       # [B, n^2-1]
state = sl.lie_euler_step_from_coordinates(
    state, omega_coordinates, step_size=dt,
    trivialization="body",
)
```

Both projects should write the same numerical health fields to their result
manifests:

```python
from sl_manifold import summarize_sl_diagnostics

manifest["sl_diagnostics"] = summarize_sl_diagnostics(
    validation_points.detach(), branch_norm_threshold=1.0
)
```

This records maximum/mean `abs(log|det|)`, maximum/median condition number,
maximum/mean Cayley spectral norm, branch-risk fraction, and non-finite
fraction.  `branch_risk` is a conservative warning for the local Gregory chart,
not a proof that a global real logarithm does or does not exist.
Pass `gregory_terms=12` to additionally record a conservative Frobenius-norm
bound on the omitted local series terms whenever the Cayley norm is below one.
This bounds truncation error only; it does not remove branch-domain constraints
or the small bias introduced by a nonzero solve jitter.

Distance functions trace-project the finite-series logarithm back into
`sl(n)` by default. This removes the scalar-identity component introduced by a
nonzero solve jitter; the exact local logarithm of an `SL(n)` relative state is
trace-free. Orthogonal projection cannot enlarge the Frobenius truncation
error, so the reported remainder bound stays conservative.

## Conventions that experiments must record

- Coordinates use a **Frobenius-orthonormal** basis.  Therefore
  `||coordinates||_2 == ||algebra||_F`.  They are not checkpoint-compatible
  with the older row-major-plus-last-diagonal ProCLIP coordinates unless the
  output head is converted.
- `SL(n)` has `n^2-1` intrinsic dimensions.  Capacity-matched baselines must
  use this number, rather than comparing `SL(8)` to an 8-dimensional vector.
- The mapping scale belongs to the exponential map.  It must not silently be
  counted a second time in the loss temperature.  Save both
  `coordinate_scale` and the learned/fixed logit scale in every manifest.
- Ordered composition defaults to the paper's left recurrence
  `G_t = g_t G_{t-1}`.  Reversing it is a required order-control ablation.
- The K-term Gregory/Cayley logarithm is a **local approximation**.  For long
  products monitor the Cayley-transform norm, regularize states near the
  identity, or use local Lie--Euler increments.  A global real logarithm does
  not exist for every matrix in `SL(n,R)`.
- The supplied ICLR 2027 manuscript defines its semidistance with the principal
  matrix logarithm. Its path-refinement `K in {1,2,4,8}` counts path segments;
  it is unrelated to this implementation's default 12 Gregory terms. Therefore
  downstream results must be labeled `Gregory-12 local-log approximation`
  unless an exact principal-log implementation and an error check are used.
- Pairwise distance supports independent left/right chunks.  This limits the
  largest `[left_chunk,right_chunk,n,n]` intermediate; checkpointing controls
  backward activation memory as a separate option.

## Tests

```bash
python -m pytest -q
```

The tests cover basis orthonormality, coordinate round trips, determinant-one
exponentials, local logarithms, left invariance, two-axis chunking with gradient
agreement, noncommutative ordered composition, masks/prefixes, and Lie--Euler
updates with finite gradients, plus manifest-ready determinant/condition/Cayley
diagnostics and branch-boundary handling.
