#!/usr/bin/env bash
# Geometry v2 on WN9-IMG, launched as ONE background job on ONE GPU:
#   Author embedding/fusion width stays 768; a learned post-fusion Linear maps
#   768->783 coordinates, then maps those coordinates to SL(28).
#   Only this projection experiment runs: no SL(8), Euclidean, direct or padding stage.
# Pilots over LRS (PILOT_EPOCHS each, validation only) -> one FORMAL_EPOCHS run
# at the best pilot lr -> one test evaluation of the best-validation checkpoint.
# Completed runs are copied to BACKUP_ROOT.  Nothing here touches the author checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
BACKUP_ROOT="${BACKUP_ROOT:-$ROOT/../vl-kge-sl-results}"
PYTHON="${PYTHON:-python}"
LRS="${LRS:-0.003 0.01 0.03 0.05 0.1 0.2}"
PILOT_EPOCHS="${PILOT_EPOCHS:-10}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-200}"
MAX_HOURS="${MAX_HOURS:-24}"
VALIDATE_EVERY_HIGH="${VALIDATE_EVERY_HIGH:-5}"
COMMON=(--coordinate-scale "${COORDINATE_SCALE:-0.1}" --chart-radius "${CHART_RADIUS:-2.0}"
        --relation-radius "${RELATION_RADIUS:-1.5}" --relation-init-norm "${RELATION_INIT_NORM:-0.5}"
        --initial-logit-scale "${LOGIT_SCALE:-3}" --initial-offset "${OFFSET:-3}"
        --score-form linear --log-sqrt-steps "${LOG_SQRT_STEPS:-1}" --patience "${PATIENCE:-50}"
        --pilot-epochs "$PILOT_EPOCHS" --formal-epochs "$FORMAL_EPOCHS")

if [[ ! "$STAMP" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "STAMP must be a simple run identifier" >&2
  exit 1
fi
QUEUE_ID="geometry-v2-$STAMP-sl28-proj783"
LOG="runs/$QUEUE_ID.log"
for target in "$LOG" "runs/$QUEUE_ID" "$BACKUP_ROOT/$QUEUE_ID"; do
  if [[ -e "$target" || -L "$target" ]]; then
    echo "Refusing to overwrite existing output: $target" >&2
    exit 1
  fi
done
"$PYTHON" -c 'import math, sys; h = float(sys.argv[1]); sys.exit(0 if math.isfinite(h) and 0 < h <= 24 else "MAX_HOURS must be in (0, 24]")' "$MAX_HOURS"
"$PYTHON" scripts/setup_upstream.py --check-only
"$PYTHON" -m unittest tests.test_geometry_v2 tests.test_geometry_v2_fixed_pad tests.test_geometry_v2_direct tests.test_geometry_v2_projection783 tests.test_geometry_queue_v2 -q
mkdir -p runs "$BACKUP_ROOT"
read -r -a LR_VALUES <<< "$LRS"
# Noclobber also protects the log against a concurrent launch with the same stamp.
set -o noclobber
exec 3> "$LOG"
nohup "$PYTHON" -u scripts/run_geometry_queue.py --root "$ROOT" --queue-id "$QUEUE_ID" \
  --backup-root "$BACKUP_ROOT" --models slv2n28 --matrix-dim 28 --embedding-dim 768 --entity-mapping linear \
  --learning-rates "${LR_VALUES[@]}" --max-hours "$MAX_HOURS" \
  --validate-every "$VALIDATE_EVERY_HIGH" "${COMMON[@]}" >&3 2>&1 &
exec 3>&-
echo "started pid $! ; log $LOG ; queue runs/$QUEUE_ID ; backups in $BACKUP_ROOT"
