# Structural geometry experiment — 2026-09-06

This replaces the WN9-IMG GL16 queue at the user's request. WN9 results and checkpoints remain preserved; they are not pooled with this experiment.

## Requested changes

- Use the existing shared **Gregory-12**, jitter 1e-7, trace projection, arithmetic mean of forward/reverse Frobenius log norms. No GL16, eigendecomposition, or reconstruction check in candidate scoring.
- Coordinate radius **1.5 or 2.0**. Record conservative Cayley/branch diagnostics without rejecting a run for that sufficient-condition warning. Finite-score, finite-gradient, and linear-solve failures remain fatal.
- Score **b − exp(log_alpha) · D**, not squared. Initial alpha 1, offset 0. Entity and relation bounded-coordinate norms are both initialized to 0.5.
- Compare direct **63-dimensional entity/relation ID tables** in Euclidean space, the Poincare ball, and SL(8). Exactly `(entities + relations) * 63 + 2` trainable parameters. No CLIP, 768-dimensional projection, entity bias, diagonal relation transform, or residual branch.
- Euclidean relation addition, relation-left Mobius addition with d_H/2, and SL relation-left multiplication are controlled adaptations, **not original MuRP or AttH reproductions**. The shared SL discrepancy is not claimed to be a globally defined geodesic.

## Datasets and scope

Use original fixed WN18RR and FB15k-237 splits from the [official RotatE repository](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/2e440e0f9c687314d5ff67ead68ce985dc446e3a/data). Every raw file is pinned by Git blob SHA-1, independently SHA-256 recorded, and verified before loading.

| Dataset | Entities | Relations | Train / validation / test | Parameters per geometry |
| --- | ---: | ---: | --- | ---: |
| WN18RR | 40,943 | 11 | 86,835 / 3,034 / 3,134 | 2,580,104 |
| FB15k-237 | 14,541 | 237 | 272,115 / 17,535 / 20,466 | 931,016 |

WN18RR probes lexical hierarchical structure; FB15k-237 provides a contrasting multi-relation KG. Both are established low-dimensional geometry benchmarks, including [AttH (ACL 2020)](https://aclanthology.org/2020.acl-main.617/). Dataset choice does not imply SL or hyperbolic geometry will win.

## Fairness and compute gates

Same seed 42, batch 512, 100 negatives, Adagrad learning rate 0.01, gradient clip 5, initialization and loss across geometry controls. The loss retains the author's logistic mean over one positive and N negatives. Negative sampling is train-filtered, directional, uniform without replacement per triple. Full-entity head and tail validation uses all-splits directional filtering and realistic average ties. Test triples are used only for filtered-ranking masks; test metrics are never scored in this screen.

1. Six GPU integration smokes: three training batches and 32 validation triples. These are not benchmark scores.
2. One full SL epoch and full validation per dataset: measure actual time and memory, inspect fixed training-positive/negative sample against SciPy logm. Approximation accuracy is reported separately from finiteness; a conservative chart warning is not an accuracy guarantee.
3. Only after reading those timings, create a bounded matched screen. Tentative scope is both datasets × both radii × three geometries, 15 epochs each, validation every 5 epochs and at the last epoch. Reduce scope if measured cost cannot fit; do not silently extend the budget or automatically enlarge HPO.

All stages retain the existing **2026-09-06 12:17:36 UTC** deadline. Save last checkpoint each complete epoch, best validation checkpoint at evaluation points, provenance, optimizer/RNG state, timing, diagnostics, and persistent backups. No new queue resumes historical WN9 checkpoints. Source hashes freeze each launched queue.
