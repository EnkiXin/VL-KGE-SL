# Audit of the existing ProCLIP SL(n) helper

Audited file:
`ProCLIP-reproduction/proclip_repro/sl_geometry.py`

## Findings

1. **Basis:** the old coordinate map fills all entries except the final
   diagonal and sets that diagonal to minus the preceding trace.  It is
   trace-free and injective, but its diagonal directions are neither normalized
   nor mutually orthogonal.  Consequently coordinate norm is not Frobenius
   tangent norm and diagonal directions receive correlated scaling.
2. **Exponential:** `torch.matrix_exp` is appropriate and low precision is
   promoted to float32.  Determinant one follows mathematically from zero trace;
   finite-precision drift should be monitored, not repaired by dividing by a
   potentially sign-ambiguous determinant root.
3. **Logarithm:** the optimized 12-term Gregory polynomial correctly matches
   the existing ProCLIP helper's local Cayley approximation. It is not the
   principal matrix logarithm defined in the supplied ICLR 2027 manuscript.
   The manuscript's `K in {1,2,4,8}` denotes piecewise-exponential path
   segments, not Gregory terms. The series requires its Cayley transform to
   remain inside the convergence region, which matters more after many ordered
   products or long ODE paths.
4. **Relative calculation:** the one-solve formula
   `(B+(1+jitter)A)^{-1}(B-A)` is algebraically equivalent to taking the Cayley
   transform of `A^{-1}B`, and avoids explicitly forming an inverse.
5. **Distance:** Schatten-2 is correctly implemented as Frobenius norm.  The
   average of both directed quantities is symmetric, but it should be described
   as a semidistance/local dissimilarity rather than assumed to be a global
   geodesic metric.
6. **Memory:** only the query axis is chunked.  Large candidate banks still
   create `[query_chunk, all_candidates, n, n]` intermediates.  The shared core
   chunks both axes and optionally checkpoints each block.
7. **Missing operations:** the old helper has no inverse coordinate map,
   ordered sequence composition, mask/prefix support, or structure-preserving
   Lie--Euler update.  These are required by MAI/CIR and FMA respectively.
8. **Scale calibration:** with points `exp(sX)`, local squared distance scales
   approximately as `s^2`.  A fixed contrastive temperature therefore changes
   effective logit temperature by `s^2`; mapping scale and softmax/logit scale
   must be logged and calibrated separately.

The new package retains the ProCLIP-compatible 12-term local approximation
while making the coordinate convention, composition direction, integration
convention, and memory controls explicit. Results must label this as a
numerical approximation to the manuscript's principal-log semidistance, not as
an exact implementation of that logarithm.

The shared diagnostics contract records `abs(log|det|)`, condition number,
Cayley spectral norm and the fraction outside the conservative local-chart
threshold.  MAI/CIR and FMA should use the same keys so numerical failures are
comparable rather than silently discarded.
