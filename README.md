# VL-KGE-SL

Reproduction of [VL-KGE](https://github.com/thefth/vl-kge) and controlled
Euclidean, hyperbolic and SL(8) geometry extensions.

Paper: [VL-KGE: Vision–Language Models Meet Knowledge Graph Embeddings](https://arxiv.org/abs/2603.02435).
Author commit: `c78994e14cf2dfda251b701c2803215d9d5fe254`.

## What is included

- Author-release reproduction wrappers and input integrity checks.
- Geometry v1: pure SL(8), adapted MuRP/MuRE, and Euclidean translation scorers.
- New **VL-DistMult + SL(8) residual**, with a parameter-matched Euclidean residual.
- Unit/integration tests and a reproducible, validation-only HPO plan generator.
- A vendored source snapshot of the shared SL manifold core.
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

## Current evidence

The original WN9 author models were reproduced: test MRR 0.9350893979
(VL-DistMult) and 0.9272085161 (VL-ComplEx), matching the paper's rounding.
Geometry v1 has also completed; SL test MRR is 0.8899170687, still below
the original DistMult. See [RESULTS.md](RESULTS.md).

The **new DistMult + SL residual has only been checked locally on CPU**,
including integration with the actual author training/evaluation functions.
It has no formal GPU ranking results yet. The prepared HPO plan is not a
running queue and does not authorize or launch paid compute.

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
```

`setup_upstream.py` applies only the recorded package-qualified `utils`
checkpoint-import correction. It does not change model mathematics or author
YAML. It refuses incompatible existing checkouts; it never overwrites the
user's unrelated edits. Input retrieval downloads roughly 216 MB.
Dataset files, checkpoints, environment folders, machine paths, deployment
notes and credentials are deliberately excluded from this repository.

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
```

The new launcher accepts `--method baseline`, `euclidean`, or `sl8`.
HPO overrides are restricted to the author's `lr`, `batch_size`,
`num_neg_samples`, `epochs`, `patience`, and `seed` fields.
In validation-only mode it intercepts the final test call after the author
reloads its best checkpoint, before test scores are computed. It does not
invent substitute test metrics.

## Important limitations

- Reproduction retains the author's all-split, mixed-direction filtering,
  including its use during negative sampling. A stricter study must apply
  corrections to all models and retrain them consistently.
- SL scoring uses a Gregory-12 **local matrix-log approximation**, not an
  exact global geodesic distance. Sampled diagnostics are not a proof of
  safety for every entity pair.
- Residual calibration uses training positives and train-filtered negatives
  only. Its center and scale are frozen; a radius-based scale floor guards
  against early saturation. Saturation statistics are recorded each epoch.
- Original checkpoints lack complete sampler/RNG state. HPO stages restart
  from scratch rather than claiming bitwise-equivalent resumed training.
- Best-checkpoint selection and HPO use validation, not test scores. Report
  multiple seeds and matched-capacity controls before making superiority claims.
- Original author results, old geometry v1, and the new hybrid are separate
  tracks; do not mix their metrics or protocol labels.
