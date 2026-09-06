# SL(28), three author datasets — launch record

Launched 2026-09-06 around 13:40 UTC (22:40 JST), on a rented GPU instance.
Verified GPU: NVIDIA RTX 3090, 24 GiB. Connection details and credentials are
not included in this public record.

## Architecture and selection policy

The author's trainable entity table and frozen CLIP features remain 768-dimensional.
Their average fusion is followed by **one trainable bias-free Linear(768,783)**,
then the existing SL(28) relation action and distance score. The user-supplied
geometry mathematics is unchanged. This is not fixed padding, initial 783-D
modality projection, an SL(8) run, or a DistMult residual branch.

The user explicitly requested **test-set MRR for both hyperparameter selection and
checkpoint selection**. This run uses `--selection-split test` and labels every
result `test_tuned_not_held_out`. These are exploratory test-tuned maxima, not
independent held-out estimates or a protocol-matched paper reproduction.

Learning-rate execution order: **0.03, 0.1, 0.01, 0.05, 0.003, 0.2**. Optimizer:
Adagrad; batch 512; seed 42. Other geometry settings are fixed in the accompanying
JSON manifest, not additionally searched in this queue.

| Dataset | Pilot epochs per LR | Formal maximum | Test selection interval | Negatives |
| --- | ---: | ---: | ---: | ---: |
| WN9-IMG | 10 | 200 | Every 5 epochs, plus final epoch | 100 |
| WikiArt-MKG-v1 | 5 | 50 | Every epoch | 1 |
| WikiArt-MKG-v2 | 2 | 20 | Every epoch | 1 |

WN9 uses the author's full-entity, two-direction ranking. WikiArt uses the author's
tail-only relation candidate pools, inductive entity mask, and missing-modality
fusion. WikiArt-v2 retains split-specific artist pools, evaluation exclusions, and
per-epoch downsampling of `isRelatedToArtwork` to 0.001. Its released YAML has a
malformed inverse-edge mapping that actually adds no edges; that literal behavior
is preserved and recorded, rather than silently correcting the author checkout.

## Execution and budget

The controller runs the three pilot searches first, then three formal runs from
scratch at the selected rates. Only one GPU training process runs at once.
Phase time is apportioned from the remaining absolute budget with weights
`1,1,1,2,2,2`. Completed pilots remain eligible if a search phase times out;
incomplete/failed searches and unstarted stages are explicitly reported.

**Original authorization ends 2026-09-07 11:37:14.613583 UTC (20:37:14 JST).**
The controller and separate verified-process guard stop by
**2026-09-07 11:36:44.613583 UTC**. This is not a new 24 hours for each dataset.
Completed and interrupted checkpoints/logs are copied to persistent storage.

## Locations

- Local launch script: `scripts/run_sl28_three_datasets.sh`
- Configuration record: `experiments/sl28_three_datasets_test_tuned.json`
- Remote source/run directory name: `vl-kge-sl28-three-datasets-20260906`
- Controller ID: `sl28-three-datasets-test-tuned-20260906`
- Controller state: `runs/sl28-three-datasets-test-tuned-20260906/state.json`
- Controller launch log: `artifacts/sl28-three-datasets-test-tuned-20260906.log`
- Deployment-specific guard log: `artifacts/guard-sl28-three-datasets.log`
  (the additional fixed-instance guard is not part of the portable launcher;
  the controller itself enforces the required absolute deadline).
- Persistent parent: the separately supplied `BACKUP_ROOT`
- Persistent trial backups: `<persistent parent>/<controller ID>/queues/<queue ID>/<trial>`

Author commit: `c78994e14cf2dfda251b701c2803215d9d5fe254`.
Source archive: `artifacts/sl28-three-datasets-source-20260906.tar.gz`.
Archive SHA256: `3b4216c567bd8df5bb5ff09bca3b8bfb6054b47b616e177c15718318995353b9`.
The same archive has been verified and saved on persistent remote storage.

## Checks at launch

- Local full suite: **274 tests passed**.
- Remote launch checks: **47 tests passed**; all nine author input files verified.
- Both WikiArt CPU input audits passed. V1 has 76,758 entities, four relations,
  and 299,968/34,020/19,695 train/validation/test triples. V2 has 224,166 entities
  and 22 relations; after the author's evaluation exclusions its split sizes are
  7,877,220/145,379/145,045. Its first downsampled training epoch uses 1,140,056
  triples. Peak CPU audit RSS was about 4.62 GiB for V2, with no GPU use.
- WN9 model GPU smoke passed with **5,642,633 trainable parameters**.
- New controller PID at launch: **13813**; first actual training PID: **13851**.
  These are historical identifiers, not authority to signal future processes.
- The first test-tuned WN9 pilot was verified training at LR 0.03; epoch 1 took
  28.1 s and epoch 2 took 26.6 s. Process GPU allocation was about 2.9 GiB.
  These are training-only times; complete ranking evaluations add time.

The preceding validation-selected projection run was stopped on the selection
policy change, retaining its checkpoint and logs at
`vl-kge-sl28-proj783-20260906/session-final` under the prior persistent backup root.
It is historical data only and is not reused or represented as a completed
test-tuned trial. The prior SL8 experiments are also preserved separately.
