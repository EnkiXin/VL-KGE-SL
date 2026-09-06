# VL-KGE-SL

Reproduction of [VL-KGE](https://github.com/thefth/vl-kge) and controlled
Euclidean, hyperbolic and SL(n) geometry extensions, including SL(8) and SL(28).

Paper: [VL-KGE: Vision–Language Models Meet Knowledge Graph Embeddings](https://arxiv.org/abs/2603.02435).
Author commit: `c78994e14cf2dfda251b701c2803215d9d5fe254`.

## What is included

- Author-release reproduction wrappers and input integrity checks.
- Geometry v1: pure SL(8), adapted MuRP/MuRE, and Euclidean translation scorers.
- **VL-DistMult + SL(8) residual**, with a parameter-matched Euclidean residual (paused).
- **WN9 geometry v2**: equal-capacity 63-coordinate Euclidean, Poincare and
  SL(8) heads with linear distance scores and matched initialization.
- **Structural KG controls** on pinned WN18RR and FB15k-237: direct 63-D
  ID tables, matched parameter counts, and the original shared Gregory-12.
- Unit/integration tests, validation-only HPO manifests and a bounded queue executor.
- A vendored source snapshot of the shared SL manifold core.
- **Latest SL(28) experiment**: author 768-D fusion -> learned 783-D projection
  -> SL(28), with a bounded six-learning-rate search on all three author datasets.
- [Results and interpretation limits](RESULTS.md), [HPO plan](experiments/SL_HPO_PLAN.md),
  and [third-party provenance](THIRD_PARTY.md).

The new residual model retains the author's 768-D multimodal fusion, relation
embeddings, original DistMult score, loss, sampler, training loop and ranking.
A pair of 768-to-63 projections maps entities/relations to SL(8). A gated
geometric compatibility is added to the author score. This is a **hybrid model**,
not a claim that all computations occur on SL.

Hard-disabling the branch reproduces the original scores, gradients and one
Adagrad update exactly in tests. The branch has 96,769 added trainable
parameters (+1.92% on WN9); the Euclidean residual has exactly the same count.
The gate is signed, so a negative learned gate must not be interpreted as
positive-weight distance-based compatibility.

## Latest experiment: SL(28) on three author datasets

The September 6, 2026 experiment retains the author's frozen 768-D CLIP features,
trainable 768-D entity table and average fusion, then applies one bias-free
`Linear(768,783)`, the trace-free 28x28 matrix map and matrix exponential.
Relations act by left multiplication; scoring is `b - alpha * D`, without a
DistMult residual. The original matrix-log implementation in
`geometry/models_v2.py` is retained, not the historical GL16 quadrature track.

The controller searches learning rates `{0.03, 0.1, 0.01, 0.05, 0.003, 0.2}`
with Adagrad, batch size 512 and seed 42. All three pilot searches precede
the formal runs, with only one GPU job at a time and one shared absolute deadline.

| Dataset | Pilot epochs per rate | Formal maximum epochs |
| --- | ---: | ---: |
| WN9-IMG | 10 | 200 |
| WikiArt-MKG-v1 | 5 | 50 |
| WikiArt-MKG-v2 | 2 | 20 |

**This experiment explicitly selects both learning rates and checkpoints using
test MRR.** Results are labeled `test_tuned_not_held_out`: they are exploratory
test-tuned maxima, not independent held-out estimates or a protocol-matched
reproduction of the paper. The older validation-selected tracks below remain
separate. See the [configuration](experiments/sl28_three_datasets_test_tuned.json)
and [launch record](experiments/SL28_THREE_DATASETS_RUN.md). A configuration or
launch record does not establish that every stage has completed.

## Recorded evidence from earlier experiments

The original WN9 author models were reproduced: test MRR 0.9350893979
(VL-DistMult) and 0.9272085161 (VL-ComplEx), matching the paper's rounding.
Geometry v1 has also completed; SL test MRR is 0.8899170687, still below
the original DistMult. See [RESULTS.md](RESULTS.md).

The **new DistMult + SL residual passes CPU integration tests and a short
RTX 4090 preflight** (three optimizer updates and 32 full-candidate validation
queries). No out-of-memory or sampled numerical-health failures occurred in
that preflight; this is not evidence of full-run accuracy or universal safety.
The residual queue was stopped and backed up on September 6, 2026. Its
30-epoch anchor validation MRRs were 0.88957125 (original DistMult) and
0.88967874 (Euclidean residual). The interrupted SL residual reached
0.80284538 at epoch 18, versus 0.81420922 for original DistMult at that
same epoch. No final test scoring was performed in this queue.

The previous study was [WN9 geometry v2](experiments/WN9_GEOMETRY_V2.md):
`b - alpha * D`, radius 1.5/2.0, 63-coordinate heads and equal trainable
parameter counts. The initial entity median and every relation norm are
0.5. Frozen visual/text features are retained; the upstream 768-D structural
entity table and shared 63-D projection remain trainable. V2 uses a new
strict directional filtering protocol and therefore cannot be compared
directly against the earlier author-release 0.93509 result. Unit tests and
an executable six-job smoke plan are included; a plan is not a training result.

On September 6 the WN9 GL16 queue was stopped and its completed/partial
checkpoints backed up. The subsequent historical study was the
[structural geometry experiment](experiments/STRUCTURAL_GEOMETRY.md) on
WN18RR and FB15k-237. It restores the unchanged shared Gregory-12 scorer,
retains radius 1.5/2, linear distance and same-scale initialization, and removes
multimodal features. Small-sample reference errors remain separate from
finite-value health checks. GPU timing must be read before starting a
matched screen; no new validation/test performance is claimed here.

## Setup

Use Python 3.11 for the recorded author environment. Git and network access
are needed to retrieve the pinned author source and, optionally, its inputs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pip install --no-deps -e vendor/sl-manifold-core

# Fetch the pinned author code without downloading all Git LFS data.
python scripts/setup_upstream.py

# Download only the three WN9 CLIP inputs and verify their Git LFS hashes.
python scripts/fetch_wn9.py --repo upstream/vl-kge --manifest artifacts/wn9-inputs.json

# For the latest three-dataset experiment, retrieve each dataset's triples
# and precomputed CLIP inputs (not raw images or encoder weights).
python scripts/fetch_wn9.py --repo upstream/vl-kge \
  --datasets wn9_img wikiart_mkg_v1 wikiart_mkg_v2 --manifest-dir artifacts
```

`setup_upstream.py` applies only the recorded package-qualified `utils`
checkpoint-import correction. It does not change model mathematics or author
YAML. It refuses incompatible existing checkouts; it never overwrites the
user's unrelated edits. WN9 input retrieval downloads roughly 216 MB;
the WikiArt inputs are additional downloads.
Dataset files, checkpoints, environment folders, private deployment artifacts
and credentials are deliberately excluded from this repository.

The author framework uses frozen image/text features; these experiments do
not train a CLIP encoder or call a paid model API.

## Checks

```bash
python scripts/setup_upstream.py --check-only
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
python -m pytest -q vendor/sl-manifold-core/tests
```

Run from the repository root after setup. Pure HPO-manifest tests require
only the standard library. Model tests need PyTorch and the pinned author
source, but use tiny CPU data and do not need a GPU.

## Training entry points

These commands start training only when deliberately invoked. Run serially
on a free GPU and choose an explicit compute budget for any search.

For the latest three-dataset SL(28) study, use
`scripts/run_sl28_three_datasets.sh`. It requires an explicit `DEADLINE_UTC`
(an authorized future ISO-8601 UTC timestamp), `BACKUP_ROOT` (a persistent
directory outside the source root), and a fresh `CONTROLLER_ID`. Its default
`SELECTION_SPLIT=test` reproduces the exploratory test-tuned protocol; choose
`validation` explicitly for validation-based selection. The historical dates
in the recorded JSON do not grant a new compute budget. Existing run directories
are never overwritten; interrupted runs are not automatically resumed.

For the historical structural study, use `scripts/run_structural_geometry.py`
and `scripts/run_structural_geometry_queue.py`, with the smoke/profile plans
and compute gates in [STRUCTURAL_GEOMETRY.md](experiments/STRUCTURAL_GEOMETRY.md).
The queue only accepts Gregory-12, uses one fixed UTC deadline, and backs up
each trial. Validation-only screen trials use complete datasets, not subsets.

For the previous WN9 study, use `scripts/run_wn9_geometry_v2.py`
and `scripts/run_wn9_geometry_v2_queue.py`. Its historical smoke manifest is
`experiments/wn9_geometry_v2_smoke.json`; its fixed September 6 UTC deadline
must match the queue argument. It intentionally expires rather than
silently granting another budget. Only smoke permits truncated training
or validation; pilots use all training batches and all 1,337 validation
triples. Neither mode evaluates test. Historical residual commands follow.

```bash
# Original author model, unchanged training/evaluation.
python scripts/run_author.py --repo upstream/vl-kge \
  --config vlkge/configs/wn9_img/distmult_clip.yaml \
  --run-dir runs/author-distmult

# New hybrid: short GPU profile with full validation, but NO final test scoring.
python scripts/run_distmult_sl_author.py --repo upstream/vl-kge \
  --method sl8 --run-dir runs/sl-residual-profile --validation-only \
  --author-overrides '{"epochs":2}'

# Prepare 18 matched anchor experiments. This DOES NOT start training.
python scripts/plan_distmult_sl_hpo.py --stage anchors \
  --out experiments/prepared/anchors-v1

# Prepare a three-method, two-epoch profile, then deliberately launch both
# stages under ONE eight-hour deadline. Use a new queue ID and a separate
# persistent backup directory. This command consumes GPU time.
python scripts/plan_distmult_sl_hpo.py --stage profile \
  --out experiments/prepared/profile-v1
python scripts/run_distmult_sl_queue.py --root "$PWD" \
  --manifest experiments/prepared/profile-v1/manifest.json \
             experiments/prepared/anchors-v1/manifest.json \
  --queue-id distmult-sl-anchors-v1 --max-hours 8 \
  --backup-root /path/to/persistent/results
```

The new launcher accepts `--method baseline`, `euclidean`, or `sl8`.
HPO overrides are restricted to the author's `lr`, `batch_size`,
`num_neg_samples`, `epochs`, `patience`, and `seed` fields.
In validation-only mode it intercepts the final test call after the author
reloads its best checkpoint, before test scores are computed. It does not
invent substitute test metrics.

The executor runs the three profiles first; only successful, test-free
completion permits the 18 anchor trials to follow. It uses one GPU lock,
stops on failed trials/backups or the shared deadline, and refuses existing
queue directories. Each completed trial's checkpoint and metadata are copied
to the separate backup directory. A finite budget does not guarantee that all
trials finish, and stopping this program does not stop rental billing.

## Important limitations

- Reproduction retains the author's all-split, mixed-direction filtering,
  including its use during negative sampling. A stricter study must apply
  corrections to all models and retrain them consistently.
- Geometry v1/residual SL scoring uses Gregory-12. The historical
  `geometry/wn9_geometry_v2.py` track uses Gauss-Legendre quadrature; the latest
  `geometry/models_v2.py` uses inverse scaling/squaring and a Gregory series
  with sampled diagnostics and a flagged fallback. These are distinct tracks.
  None is claimed to provide an exact global geodesic distance or a complete
  global matrix-log algorithm.
- Residual calibration uses training positives and train-filtered negatives
  only. Its center and scale are frozen; a radius-based scale floor guards
  against early saturation. Saturation statistics are recorded each epoch.
- Original checkpoints lack complete sampler/RNG state. HPO stages restart
  from scratch rather than claiming bitwise-equivalent resumed training.
- Earlier tracks select checkpoints and hyperparameters using validation.
  The latest three-dataset SL(28) run instead explicitly uses test selection
  and must be labeled accordingly. Report independent held-out evaluations,
  multiple seeds and matched-capacity controls before making superiority claims.
- Original author results, old geometry v1, the hybrid and geometry v2 are separate
  tracks; do not mix their metrics or protocol labels.
