# WN9 geometry v2: geometry-only comparison before capacity expansion

## Research question and scope

Does the SL(8) relative-log scorer outperform matched 63-coordinate Euclidean
and hyperbolic scorers on WN9-IMG? This is NOT the DistMult residual experiment,
NOT a reproduction of MuRP/AttH author results, and NOT a comparison with
the earlier 0.93509 author-release test MRR under a different training/filtering
protocol. WN18RR and FB15k-237 were downloaded and verified while that alternative
was being considered; no structural-KGE training was launched.

## Matched architecture

- Retain the pinned author's WN9 entity IDs, train/validation/test splits,
  frozen image/text features and 768-dimensional average fusion.
- All three models use the same 768-to-63 entity projection and a 63-coordinate
  relation table, with equal trainable parameter counts. This is equal
  **scoring dimension**, not a claim that upstream entity tables are 63-D.
- Train-only initialization matches entity and relation coordinate scales.
  The initial coordinate norm target is 0.5; chart radii 1.5 and 2.0 are
  exploration limits, not a demand to start all points at the boundary.
- The score is **b - alpha * D**, not squared distance. `alpha > 0` and `b`
  are learned scalars; initial alpha=1, b=0. No DistMult score, bounded residual
  gate, tanh score squashing, or frozen calibration scale floor is used.
- Euclidean relation action: h+r; hyperbolic: ordered Mobius action of the
  relation on the head; SL: exp(A_r) exp(A_h). All are compared with the tail.
- Poincare distance is divided by two under the standard exp0 convention so
  its near-origin units agree with the Euclidean and SL linearized controls.
- SL uses a relative principal matrix-log discrepancy, **not an exact global
  Riemannian geodesic**. The exponential parametrization is not all of SL(8).

## Numerical safeguards

The previous Cayley spectral-norm warning is recorded but is not itself a
reason to reject a trial. Replace fixed Gregory-12 scoring with a differentiable
matrix-log approximation verified against reference principal logarithms,
matrix-exponential reconstruction, and gradient checks. A quadrature agreement
check alone is insufficient: even-order quadratures can agree incorrectly
outside the principal-log domain.

Actual nonfinite values, failed solves and demonstrably inaccurate/undefined
principal-log computations must not be accepted as valid scores. A numeric
failure is recorded explicitly; the queue does not secretly shrink the radius,
replace the distance, or reduce batch size. A finite set of diagnostic samples
is not a certificate for every possible group element.

References for numerical definitions (not claims of implementing the complete
Higham algorithm):

- https://nhigham.com/2020/11/17/what-is-the-matrix-logarithm/
- https://eprints.maths.manchester.ac.uk/1799/1/paper13.pdf

## Training and evaluation

Use one shared trainer, optimizer, batch size, negative count, clipping and
initialization policy. Negative filtering uses training triples only, with
head and tail directions separated. Evaluation filters all known triples,
again directionally, and documents tie handling. No test scoring during
profiling or hyperparameter selection.

GPU smoke checks passed for both radii and all three models without OOM.
Measured throughput sets the first-stage screen to 15 epochs for each of the
18 matched trials: radii {1.5, 2.0}, learning rates {0.003, 0.01, 0.03}, and
three methods, with complete validation. Thirty epochs across all 18 trials
would exceed the inherited budget. This pilot is not a final-convergence or
final-test result; the ready plan still requires an explicit queue command.

The inherited compute deadline remains **2026-09-06 12:17:36 UTC** (21:17:36 JST).
Setup, profiling and the new queue do not silently reset the earlier eight-hour
budget. A bounded queue may leave trials unstarted. It stops its own processes,
not the rented instance or its billing, and backs up completed checkpoints.

## Decision rule

A single-seed validation lead is screening evidence, not proof of generalization
or statistical superiority. Before claiming a clear SL advantage, lock the
selected settings, run matched multi-seed confirmation and evaluate test only
after selection. Report gaps against both controls, uncertainty, capacity and
runtime, including unsuccessful trials.

Only after a credible matched-geometry gain should the study expand capacity
(e.g. SL(28), with 783 degrees of freedom, or a product of SL(8) blocks). Neither
capacity expansion nor a high-dimensional DistMult hybrid is launched by this
v2 plan.
