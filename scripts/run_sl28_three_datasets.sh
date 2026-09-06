#!/usr/bin/env bash
# One-GPU, six-learning-rate SL(28) search across all three author datasets.
# An absolute shared deadline is mandatory: never grant each dataset a new 24h.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
: "${DEADLINE_UTC:?Set the original authorized absolute UTC deadline}"
: "${BACKUP_ROOT:?Set a persistent directory outside the source root}"
CONTROLLER_ID="${CONTROLLER_ID:-sl28-three-datasets-test-tuned-20260906}"
SELECTION_SPLIT="${SELECTION_SPLIT:-test}"
if [[ ! "$CONTROLLER_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "CONTROLLER_ID must be a simple name" >&2
  exit 1
fi
ARGS=(--root "$ROOT" --backup-root "$BACKUP_ROOT" --deadline-utc "$DEADLINE_UTC"
      --controller-id "$CONTROLLER_ID" --selection-split "$SELECTION_SPLIT")
if [[ -n "${PRECEDING_ROOT:-}" || -n "${PRECEDING_QUEUE:-}" ]]; then
  : "${PRECEDING_ROOT:?Both preceding root and queue are required}"
  : "${PRECEDING_QUEUE:?Both preceding root and queue are required}"
  ARGS+=(--preceding-root "$PRECEDING_ROOT" --preceding-queue "$PRECEDING_QUEUE")
fi
for target in "runs/$CONTROLLER_ID" "artifacts/$CONTROLLER_ID.log" "$BACKUP_ROOT/$CONTROLLER_ID"; do
  if [[ -e "$target" || -L "$target" ]]; then
    echo "Refusing to overwrite prior experiment: $target" >&2
    exit 1
  fi
done
"$PYTHON" -c 'from scripts.run_sl28_three_datasets import parse_args; parse_args()' "${ARGS[@]}"
"$PYTHON" scripts/setup_upstream.py --check-only
"$PYTHON" -m unittest tests.test_geometry_datasets tests.test_geometry_queue_datasets tests.test_sl28_three_datasets -q
mkdir -p artifacts runs "$BACKUP_ROOT"
set -o noclobber
exec 3> "artifacts/$CONTROLLER_ID.log"
nohup "$PYTHON" -u scripts/run_sl28_three_datasets.py "${ARGS[@]}" >&3 2>&1 </dev/null &
CONTROLLER_PID=$!
exec 3>&-
echo "started controller pid $CONTROLLER_PID; status runs/$CONTROLLER_ID/state.json"
