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

## Geometry v2 (2026-09-06)

`geometry/models_v2.py` re-implements the SL and Euclidean scorers after the v1
analysis showed that v1's SL(8) was numerically indistinguishable from a 63-D
Euclidean model: its radial bound (radius 0.5) kept every matrix within 1% of
`I + A`, its relation init (1e-4) made the head entity the nearest candidate
(validation Hits@1 was exactly 0 in the first epochs), and `offset - 100*D^2`
saturated the logistic loss from the first batch.

Changes: radial *clipping* at a larger radius (entities 2.0, relations 1.5),
a principal logarithm by inverse scaling and squaring (Denman--Beavers square
root + Gregory series; flagged fallback for matrices without a real principal
log), the linear score `offset - exp(log_scale) * D` with learnable modest
initial values (3, 3), relations initialised at norm 0.5, a free matrix size
`n` (SL(8) = 63 coordinates, SL(28) = 783), and a Euclidean control that shares
every other choice.  Runner models: `slv2`, `euclidv2`; queue specs `slv2`,
`euclidv2`, `slv2n28`.  Diagnostics report the principal-domain margin of
sampled relative matrices, the logarithm error against an eigen-decomposition
reference, and the fallback fraction; the runner stops if the sampled error
exceeds 1e-2 or more than 5% of sampled pairs fell back.

The original v2 launch ran SL(8)/Euclidean pilots and formal runs before a
separate SL(28) queue. Those results remain a distinct historical experiment.

## SL(28) without a learned entity projection (2026-09-06)

The initially requested **slv2n28 with entity_mapping=fixed_pad** was
implemented and GPU-smoke-tested, but its long-running queue was cancelled
before launch when the user clarified that the initial embedding should
instead be 783-D. The fixed-pad implementation remains available explicitly:

1. Keep the author's trainable 768-D entity table, frozen 768-D CLIP visual
   and textual features, and original average fusion.
2. Append exactly 15 zero coordinates to that fused vector: 768 -> 783.
   There is **no Linear layer or learned entity projection** in this mode.
3. Apply the existing coordinate scale and radial clipping; map the 783
   coordinates into the fixed orthonormal trace-free 28x28 basis and exponentiate.
4. Keep the existing learned 783-D relation coordinates, left matrix action,
   matrix-log implementation and `b - alpha * D` score unchanged.

The padding is injective, but the overall representation is still subject to
the already configured radial clipping and nonlinear exponential map. This
is not a claim that the entire model is globally lossless or equivalent to
DistMult. The author WN9 configuration has 5,041,152 trainable parameters;
this no-projection SL(28) has 5,041,289 (only 137 more). The old projected
SL(28) would have an additional 601,344 projection weights.

CLI default `--entity-mapping linear` preserves historical runner behavior
and checkpoints; fixed padding is available only when explicitly selected.
Fixed padding refuses dimensions below 768 instead of truncating features.
Tests: `python -m unittest tests.test_geometry_v2 tests.test_geometry_queue_v2
tests.test_geometry_v2_fixed_pad`.

## Cancelled proposal: start at embedding_dim=783 and map directly to SL(28)

This alternative was implemented and unit-tested, but its server launch was
cancelled before execution after the user selected the 768 -> 783 projection
experiment instead. It remains explicitly available as slv2n28 with
embedding_dim=783 and entity_mapping=direct, but is not the launch default.

- The trainable entity table starts at 783 dimensions, as does the relation table.
- Frozen CLIP image/text features remain 768-D. The unchanged author base
  automatically creates its original bias-free `visual_linear` and
  `textual_linear` layers (768 -> 783) to align these modalities before fusion.
- Author average fusion produces 783 coordinates, used directly as SL(28)
  algebra coordinates. There is no padding and no post-fusion Linear layer.
- Existing coordinate scale, clipping radii, relation initialization, matrix
  exponential/logarithm, relation-left-action and linear-distance scoring are
  unchanged. No DistMult residual is added.
- WN9 trainable parameters: **6,342,302**, including 1,202,688 parameters in
  the two author modality-alignment layers. This is not equal to the original
  VL-DistMult's 5,041,152, nor to the cancelled fixed-pad variant's 5,041,289.

The script retains LR {0.01,0.03,0.1}, 10 pilot epochs, at most 200 formal
epochs, validation every 5 epochs, and a validation-selected learning rate.
Only the final formal checkpoint selected by validation is evaluated on test.
When replacing a paid-server run, pass `MAX_HOURS` for the remaining already
authorized budget; changing this model does not authorize a new 24-hour window.
Additional tests: `python -m unittest tests.test_geometry_v2_direct`.

## Selected architecture: 768 -> Linear(768,783) -> SL(28)

The user's final explicit instruction was to replace the old **768 -> 63 ->
SL(8)** path with **768 -> 783 -> SL(28)**. `scripts/run_v2_wn9.sh` therefore
launches only `slv2n28`, `--embedding-dim 768`, `--entity-mapping linear`.

- Keep the author's 768-D trainable entity table, frozen 768-D CLIP features,
  and unchanged 768-D average fusion. Modality alignment layers are Identity.
- Use a single learned bias-free `Linear(768,783)` after fusion, followed by
  the unchanged scale/radial clip and SL(28) algebra/exponential mapping.
- The relation table has 783 coordinates per relation. Relation action,
  initialization norm, matrix logarithm and distance score are unchanged.
- There is no 63-D compression, no fixed padding, no initial 783-D entity
  table, and no extra modality-projection layers in this selected experiment.
- WN9 trainable parameters: **5,642,633** (the original VL-DistMult has
  5,041,152). The post-fusion projection contains 601,344 weights.
- Only this new model is trained. Old SL(8), old Euclidean, fixed-pad and
  direct-783 trials will not be resumed. Retained tests for those code paths
  are CPU compatibility checks, not additional GPU experiments.

The same LR grid, pilot/formal epoch limits, five-epoch validation schedule,
remaining authorized time limit, backups and test-only-at-the-end protocol
apply. Architecture-specific tests: `tests.test_geometry_v2_projection783`.

## Latest controller: all three author datasets

`scripts/run_sl28_three_datasets.sh` supersedes the single-dataset launch above
for the September 6 three-dataset experiment. It retains exactly the selected
768 -> Linear(768,783) -> SL(28) architecture and extends data handling to
WN9-IMG, WikiArt-MKG-v1 and WikiArt-MKG-v2. The WikiArt protocols retain the
author's inductive masks, available-modality fusion and relation-specific
candidate pools.

Unlike the historical validation-selected launch above, this controller defaults
to explicit **test-set** checkpoint and learning-rate selection, labeled
`test_tuned_not_held_out`. Its six learning rates are 0.03, 0.1, 0.01, 0.05,
0.003 and 0.2. All pilot searches precede the formal runs under one shared
absolute deadline. See [the configuration](../experiments/sl28_three_datasets_test_tuned.json)
and [the launch record](../experiments/SL28_THREE_DATASETS_RUN.md). The older
single-dataset launch and cancelled variants are retained for provenance, not
silently reused as results for this experiment.
