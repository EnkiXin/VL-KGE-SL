# WN9-IMG results (snapshot: 2026-09-06)

All entries below are final **test** metrics from the best-validation
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
- The new extension has **not run formal GPU training** and has no reported
  validation or test performance.
- The HPO plan/configuration generator prepares experiments; it does not launch
  a scheduler or authorize GPU expenditure.

## Protocol limitations

The author release combines train/validation/test positives in its filtering
map and merges head/tail direction keys; this also affects negative sampling.
These conventions are preserved for reproduction. A strict-protocol study
must change them consistently and retrain every comparator. No test metric is
used to select the new HPO configurations. Different batch sizes imply
different optimizer-step counts even at equal epochs.
