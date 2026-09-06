# Results and protocol boundaries (snapshot: 2026-09-06)

All entries in this first table are final **test** metrics from the best-validation
checkpoint, with seed 42 and 200 completed epochs. These are single-seed
results, not estimates of statistical significance.

| Method | Test MRR | Hits@1 | Hits@3 | Hits@10 | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Author VL-DistMult | 0.9350893979 | 0.9245640637 | 0.9397270660 | 0.9571645186 | 192 |
| Author VL-ComplEx | 0.9272085161 | 0.9200151630 | 0.9287338893 | 0.9412433662 | 196 |
| Pure SL(8) geometry v1 | 0.8899170687 | 0.8631539045 | 0.9105382866 | 0.9275966641 | 175 |
| Adapted MuRE geometry v1 | 0.8881952292 | 0.8586050038 | 0.9105382866 | 0.9340409401 | 175 |
| Euclidean translation v1 | 0.8852110631 | 0.8559514784 | 0.9094010614 | 0.9264594390 | 177 |
| Adapted MuRP geometry v1 | 0.8802126805 | 0.8453373768 | 0.9082638362 | 0.9340409401 | 200 |

Author configurations: Adagrad lr=0.1, batch=512, 100 negatives, frozen CLIP,
768-D average fusion, patience=50. Author DistMult and ComplEx match the paper's
three-decimal reporting; rounding is not evidence of exceeding the paper.

Geometry v1: same data release and filtering convention, lr=0.03 selected from
two 10-epoch validation-only pilots (0.01, 0.03), batch=512, 100 negatives,
768-to-63 projection, bounded chart radius=0.5, coordinate scale=0.1,
gradient clipping=5. Initial score scales are 100 except MuRP=25. These extra
choices mean it is not a literal scorer-only replacement in the author loop.

SL v1 has the highest MRR among these four geometry adaptations, but remains
below author VL-DistMult. The small difference from MuRE is not a demonstrated
robust advantage. MuRP/MuRE here are adaptations, not original-paper reproductions.

## New DistMult + SL residual extension

`geometry/distmult_sl.py` retains the author's original DistMult scorer and
adds a relation-aware geometric residual; a matched Euclidean residual is
provided. This is a different model from geometry v1. At this snapshot:

- Local CPU correctness and author-trainer integration tests have passed.
- Initial calibration saturation was found and corrected using train-only
  diagnostics. This is not a ranking result.
- The residual GPU queue was subsequently run and stopped/backed up on
  September 6. Its 30-epoch anchors reached validation MRR 0.88957125
  (original DistMult) and 0.88967874 (Euclidean residual). SL was interrupted
  at epoch 18 with validation MRR 0.80284538; original DistMult at that epoch
  was 0.81420922. These are validation-only, not final test results.
- The HPO plan/configuration generator prepares experiments; it does not launch
  a scheduler or authorize GPU expenditure.

## Protocol limitations

The author release combines train/validation/test positives in its filtering
map and merges head/tail direction keys; this also affects negative sampling.
These conventions are preserved for reproduction. A strict-protocol study
must change them consistently and retrain every comparator. No test metric is
used to select the new HPO configurations. Different batch sizes imply
different optimizer-step counts even at equal epochs.

## Stopped WN9 geometry v2 — validation only

The GL16 queue was stopped at 2026-09-06 07:07:21 UTC at the user's request.
All three completed checkpoints and the interrupted checkpoint were backed up.
These runs use a different strict directional protocol and **must not be
compared directly with the first table's test MRR**.

| Geometry | Radius | Epochs completed | Best epoch | Best full-validation MRR |
| --- | ---: | ---: | ---: | ---: |
| SL(8), GL16 | 1.5 | 15 | 15 | 0.0723665207 |
| Euclidean | 1.5 | 15 | 15 | 0.0635534088 |
| Hyperbolic | 1.5 | 15 | 15 | 0.0062108812 |
| SL(8), GL16, interrupted | 2.0 | 10 | 10 | 0.1388788395 |

Common settings: seed 42, lr 0.01, 512 positives and 100 negatives, linear
distance, initial bounded entity/relation norm 0.5, initial alpha 1, offset 0.
These short runs are not convergence results or evidence of a robust SL gain.
The GL16 candidate implementation was substantially slower than the previous
shared Gregory-12; it has now been removed from the active experiment path.

## WN18RR / FB15k-237 structural study — smoke only so far

The [structural study](experiments/STRUCTURAL_GEOMETRY.md) uses new direct
63-D ID tables and the original Gregory-12 scorer, without multimodal features.
Six GPU integration smokes (3 batches, 32 validation triples) all completed and
were backed up. No OOM occurred. The SL fixed training-probe maximum scaled
SciPy logm errors after 3 batches were 3.79e-7 (WN18RR) and 4.26e-7 (FB15k-237).
These are small-sample numerical checks, not global validity guarantees.

Do not use smoke ranking metrics as benchmark results. Full-epoch GPU timing
is being measured before setting the matched-screen scope; this section does
not yet report a full structural experiment or any test result.
