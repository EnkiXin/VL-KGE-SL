# WN9 geometry comparison v1

Separate extension: the locked author source and completed DistMult/ComplEx
reproduction are unchanged. These new methods are experimental adapters, not
claimed reproductions of the original MuRP/MuRE paper.

## Common front end and capacity

All methods retain the author's trainable 768-dimensional entity ID table,
frozen and unnormalized CLIP image/text inputs, and average fusion. A shared
bias-free 768-to-63 projection produces entity coordinates. All four variants
start with identical common weights at seed 42. Smooth radial bounding is
`scale*x / sqrt(1 + ||scale*x||^2 / radius^2)`, initially scale=0.1, radius=0.5.
There are no entity-specific score biases in any new variant.

| Variant | Relation action and squared discrepancy |
|---|---|
| euclidean | `||a_h + a_r - a_t||^2` |
| mure | `||diag_r * a_h - (a_t + a_r)||^2` |
| murp | `d_P(exp0(diag_r * a_h), exp0(a_t) (+)_P exp0(a_r))^2` |
| sl8 | `D_SL(exp(A_r) exp(A_h), exp(A_t))^2` |

MuRE/MuRP are adapted from the relation equations in
https://github.com/ibalazevic/multirelational-poincare ; we use the common VL
fusion, loss, tangent parameterization and scalar calibration, and omit the
original per-entity biases. Curvature is fixed at -1 in this first search.
MuRE/MuRP have an additional 63-value diagonal per relation; report exact
parameter counts rather than calling their parameter budgets identical.

SL uses the shared Frobenius-orthonormal trace-free basis and symmetric
Gregory-12 local logarithm approximation from `sl-manifold-core`. It is not a
global exact geodesic metric. It tests relation-left-action SL, not bilateral
actions, graph propagation, Mamba, or manifold-space multimodal fusion.

Scores are `offset - exp(log_scale) * discrepancy_squared`. Scale and offset
are learned. Base initial scale is 100; MuRP starts at 25 because its standard
Poincare squared distance in these exp0 coordinates has a local factor of 4.
This fixed local-metric calibration is not selected using test performance.

## Training and evaluation

- Same pinned WN9 inputs: 6555 entities, 9 relations, 11741/1337/1319 splits.
- Adagrad, batch512,100 unique negatives/positive, seed42,200 max epochs,
  patience50,full validation every epoch, strict `>` checkpoint selection.
- New models all use gradient-norm clipping at5, common score calibration,
  and the bounded chart. These are documented changes from the original
  DistMult/ComplEx training, not a claim that all computations are unchanged.
- Author uniform sampler and logistic loss are imported unchanged; shuffle and
  sampling share the explicit CPU RNG. Sampled-training evaluation is omitted
  because it does not select checkpoints. No warm start from author models.
- All entities are evaluated in BOTH directions. Filtering intentionally uses
  the author's combined-direction/all-split map for this first comparison.
  The known author filtering issues remain; a later strict-protocol track must
  retrain every comparison model under corrected rules.
- Stable tie ranking uses ascending entity IDs. The new chunked evaluator was
  checked on the completed original DistMult full test against both the saved
  paper-reproduction result and the live original evaluator, matching all four
  metrics exactly (MRR 0.9350893979294906).
- Pilots use full validation but NEVER test. For each model, select one LR by
  best pilot validation MRR, restart from the common initialization, and only
  test its best-validation checkpoint once after the formal run ends.

## Bounded first queue and safeguards

Default queue: 4 models x LR {0.01,0.03} x10-epoch pilots; then one200-epoch
formal run per model. Single GPU lock, at most6 hours wall clock including
pilots, failure stops rather than retries, and completed runs are copied to
the supplied persistent backup directory. This is a first narrow search, not
exhaustive tuning or multi-seed evidence. Check `queue.json` for actual order,
settings and whether the wall limit was reached.

Mapping unique IDs, chunked scoring, activation checkpointing, and no-grad
entity caching bound memory. Training/evaluation reject nonfinite scores;
gradients reject nonfinite values. SL diagnostics sample distributed entity
IDs and both relative-score directions, checking determinants, conditioning,
Cayley norms and Gregory remainder bounds; they are sampled diagnostics, not
a proof covering every candidate. A sampled branch risk or remainder bound
over1e-3 stops the run. Files capture source/input hashes, exact configuration,
per-epoch validation/loss/timing/memory, and model/optimizer/RNG checkpoints.

Implementation: `geometry/models.py`, `geometry/evaluation.py`,
`scripts/run_geometry.py`, `scripts/run_geometry_queue.py`.
