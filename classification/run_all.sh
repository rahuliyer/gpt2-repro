#!/usr/bin/env bash
# Run all four classification setups, two at a time: one queue per GPU.
#
# Usage: classification/run_all.sh [checkpoint-dir]
#
# The full fine-tune is the longest run, so it gets GPU 0 to itself while the
# other three run back to back on GPU 1. Each run's output goes to
# <checkpoint-dir>/logs/<script>.log.
set -uo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT_DIR="${1:-checkpoints/classification}"
LOG_DIR="$CHECKPOINT_DIR/logs"
mkdir -p "$LOG_DIR"

run_queue() {
    local device="$1"
    shift
    local failed=0
    for script in "$@"; do
        local log="$LOG_DIR/$script.log"
        echo "[gpu $device] starting $script, logging to $log"
        if PYTHONUNBUFFERED=1 uv run python "classification/$script.py" \
            --checkpoint-dir "$CHECKPOINT_DIR" \
            --device-id "$device" > "$log" 2>&1; then
            echo "[gpu $device] finished $script: $(grep '^Test over' "$log")"
        else
            echo "[gpu $device] FAILED $script, see $log"
            failed=1
        fi
    done
    return "$failed"
}

# A script's background jobs ignore ^C, so on its own an interrupt would stop
# this script and leave the runs going. Signal the whole process group instead.
trap 'echo "stopping all runs"; trap - TERM; kill 0' INT TERM

run_queue 0 train_linear_head_full &
gpu0=$!
run_queue 1 train_mlp_head train_mlp_head_last_block train_linear_head_last_block &
gpu1=$!

status=0
wait "$gpu0" || status=1
wait "$gpu1" || status=1
exit "$status"
