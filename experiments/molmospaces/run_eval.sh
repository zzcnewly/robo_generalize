#!/bin/bash
#
# Run DreamZero evaluation on MolmoSpaces benchmark.
#
# This script is meant to run INSIDE the mlspaces conda environment.
# The DreamZero inference server must already be running in a separate
# terminal/process (in the dreamzero conda env).
#
# Usage:
#   # Interactive (server already running):
#   bash experiments/molmospaces/run_eval.sh --port 5000
#
#   # With all options:
#   bash experiments/molmospaces/run_eval.sh \
#       --port 5000 \
#       --host localhost \
#       --benchmark_dir /path/to/benchmark \
#       --task_horizon 500 \
#       --output_dir eval_output/molmospaces
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOLMOSPACES_DIR="${REPO_ROOT}/external/molmospaces"
ASSETS_DIR="${REPO_ROOT}/external/molmospaces_assets"

# Defaults
PORT=8000
HOST="localhost"
BENCHMARK_DIR=""
TASK_HORIZON=500
OUTPUT_DIR="${REPO_ROOT}/eval_output/molmospaces"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --benchmark_dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --task_horizon) TASK_HORIZON="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --idx) EXTRA_ARGS="$EXTRA_ARGS --idx $2"; shift 2 ;;
        --no_wandb) EXTRA_ARGS="$EXTRA_ARGS --no_wandb"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-discover benchmark directory if not specified
if [ -z "$BENCHMARK_DIR" ]; then
    CACHE="${MLSPACES_CACHE_DIR:-${REPO_ROOT}/external/molmospaces_cache/mujoco}"
    BENCH_JSON=$(find "$CACHE/benchmarks" -path "*/ithor/FrankaPickHardBench/*/benchmark.json" 2>/dev/null | head -1)
    if [ -n "$BENCH_JSON" ]; then
        BENCHMARK_DIR="$(dirname "$BENCH_JSON")"
    else
        echo "ERROR: No benchmark directory found. Please specify --benchmark_dir or download assets first."
        echo "  See: bash experiments/molmospaces/setup.sh"
        exit 1
    fi
fi

echo "============================================"
echo "DreamZero MolmoSpaces Evaluation"
echo "============================================"
echo "Server:     ${HOST}:${PORT}"
echo "Benchmark:  ${BENCHMARK_DIR}"
echo "Horizon:    ${TASK_HORIZON} steps"
echo "Output:     ${OUTPUT_DIR}"
echo "============================================"
echo ""

# Set environment
export MLSPACES_ASSETS_DIR="$ASSETS_DIR"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-${REPO_ROOT}/external/molmospaces_cache/mujoco}"
export DREAMZERO_HOST="$HOST"
export DREAMZERO_PORT="$PORT"
export PYTHONPATH="${REPO_ROOT}:${MOLMOSPACES_DIR}:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DISPLAY="${DISPLAY:-:0}"
export _IN_MULTIPROCESSING_CHILD=1

python -m molmo_spaces.evaluation.eval_main \
    "experiments.molmospaces.eval_config:DreamZeroBenchmarkEvalConfig" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --task_horizon_steps "$TASK_HORIZON" \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb \
    $EXTRA_ARGS

EVAL_EXIT=$?
echo ""
echo "[$(date)] Evaluation finished with exit code: $EVAL_EXIT"
