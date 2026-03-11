#!/bin/bash
#SBATCH --job-name=dreamzero-mlspaces
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint="gpu80"
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --output=logs/mlspaces_eval_%j.out
#SBATCH --error=logs/mlspaces_eval_%j.err

# ============ Configuration ============
REPO_ROOT="/scratch/gpfs/HENDERSON/zs7353/robo_generalize"
MODEL_PATH="/scratch/gpfs/HENDERSON/zs7353/dreamzero/checkpoints/dreamzero_droid"
MOLMOSPACES_DIR="${REPO_ROOT}/external/molmospaces"
ASSETS_DIR="${REPO_ROOT}/external/molmospaces_assets"
CACHE_DIR="${REPO_ROOT}/external/molmospaces_cache/mujoco"

NUM_GPUS=2
SERVER_PORT=8000
TASK_HORIZON=500
EVAL_IDX=""
# =======================================

cd "$REPO_ROOT"
mkdir -p logs

# Load modules and init conda for non-interactive shell
module purge
module load anaconda3/2024.10
module load cudatoolkit/12.9
eval "$(conda shell.bash hook)"

# Start virtual X framebuffer (needed by pynput, a transitive MolmoSpaces dependency)
Xvfb :99 -screen 0 1024x768x24 &
XVFB_PID=$!
export DISPLAY=:99
echo "Started Xvfb on :99 (PID $XVFB_PID)"

echo "========================================"
echo "DreamZero MolmoSpaces Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL_PATH"
echo "Cache: $CACHE_DIR"
echo "========================================"
echo ""
nvidia-smi

# ── Step 0: Ensure MolmoSpaces asset symlinks are correct ──
echo ""
echo "[$(date)] Step 0: Setting up MolmoSpaces asset symlinks..."
conda activate mlspaces

export MLSPACES_ASSETS_DIR="$ASSETS_DIR"
export MLSPACES_CACHE_DIR="$CACHE_DIR"
export MLSPACES_FORCE_INSTALL=True
export MLSPACES_AUTO_INSTALL=True
export MLSPACES_CACHE_LOCK=False
export PYTHONPATH="${REPO_ROOT}:${MOLMOSPACES_DIR}:${PYTHONPATH}"

python << 'PYEOF'
import os, shutil
from pathlib import Path

assets = Path(os.environ["MLSPACES_ASSETS_DIR"]).resolve()
cache  = Path(os.environ["MLSPACES_CACHE_DIR"]).resolve()

SOURCES_WE_HAVE = {
    "objects":    {"thor": "20251117", "objathor_metadata": "20260129"},
    "grasps":     {"droid": "20251116", "droid_objaverse": "20251218"},
    "benchmarks": {"molmospaces-bench-v1": "20260210"},
    "robots":     {"franka_droid": "20260127"},
    "scenes":     {"ithor": "20251217", "refs": "20250923"},
}

# Scene XMLs contain relative paths like ../../objects/thor/ that MuJoCo
# resolves from the physical directory of the file. If scenes/<source> is a
# directory symlink, ../../ goes through the symlink target (cache tree)
# instead of the assets tree, breaking cross-type references. So scenes
# must be real directories with per-item symlinks inside.
NEEDS_REAL_DIR = {"scenes"}

os.makedirs(assets, exist_ok=True)

for data_type, source_to_version in SOURCES_WE_HAVE.items():
    for source, version in source_to_version.items():
        cache_path = cache / data_type / source / version
        link_path  = assets / data_type / source

        if not cache_path.exists():
            print(f"  SKIP {data_type}/{source} (not in cache)")
            continue

        if data_type in NEEDS_REAL_DIR:
            if link_path.is_symlink():
                link_path.unlink()
            link_path.mkdir(parents=True, exist_ok=True)
            linked = 0
            for item in sorted(cache_path.iterdir()):
                dst = link_path / item.name
                if dst.exists() or dst.is_symlink():
                    if dst.is_symlink() and dst.resolve() == item.resolve():
                        continue
                    if dst.is_symlink():
                        dst.unlink()
                    elif dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                dst.symlink_to(item.resolve(), target_is_directory=item.is_dir())
                linked += 1
            total = sum(1 for _ in cache_path.iterdir())
            print(f"  LINK {data_type}/{source}: real dir, {linked} new + {total - linked} existing")
        else:
            if link_path.is_symlink():
                if link_path.resolve() == cache_path.resolve():
                    print(f"  OK   {data_type}/{source} (already correct)")
                    continue
                link_path.unlink()
            elif link_path.is_dir():
                shutil.rmtree(link_path)

            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(cache_path.resolve(), target_is_directory=True)
            print(f"  LINK {link_path} -> {cache_path.resolve()}")

print("Asset symlinks setup complete.")
PYEOF
echo "" 2>&1
SETUP_EXIT=$?
if [ $SETUP_EXIT -ne 0 ]; then
    echo "[$(date)] WARNING: Asset setup had errors (exit $SETUP_EXIT), continuing..."
fi

# ── Step 1: Start DreamZero inference server (dreamzero env) ──
echo ""
echo "[$(date)] Step 1: Starting DreamZero inference server on port $SERVER_PORT..."
conda activate dreamzero

export CUDA_VISIBLE_DEVICES=0,1
export ATTENTION_BACKEND=FA2

torchrun --standalone --nproc_per_node=$NUM_GPUS \
    socket_test_optimized_AR.py \
    --port $SERVER_PORT \
    --enable-dit-cache \
    --model-path "$MODEL_PATH" \
    --timeout-seconds 3600 &

SERVER_PID=$!
echo "[$(date)] Server PID: $SERVER_PID"

# Wait for server to be ready by checking if port is listening
echo "[$(date)] Waiting for server to start listening on port $SERVER_PORT..."
MAX_WAIT=900
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost', $SERVER_PORT)); s.close()" 2>/dev/null; then
        echo "[$(date)] Server is listening! (waited ${WAITED}s)"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 60)) -eq 0 ]; then
        echo "[$(date)] Still waiting... (${WAITED}s elapsed)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[$(date)] ERROR: Server did not start within ${MAX_WAIT}s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Give the server a few more seconds to fully initialize after port is open
sleep 10

# ── Step 2: Warm up the server ──
echo ""
echo "[$(date)] Step 2: Warming up server..."
python test_client_AR.py \
    --host localhost \
    --port $SERVER_PORT \
    --num-chunks 1 \
    --prompt "pick up the object" \
    --use-zero-images
echo "[$(date)] Warmup complete."

# ── Step 3: Run MolmoSpaces evaluation (mlspaces env) ──
echo ""
echo "[$(date)] Step 3: Switching to mlspaces env and running evaluation..."
conda activate mlspaces

export MLSPACES_ASSETS_DIR="$ASSETS_DIR"
export MLSPACES_CACHE_DIR="$CACHE_DIR"
export DREAMZERO_HOST="localhost"
export DREAMZERO_PORT="$SERVER_PORT"
export PYTHONPATH="${REPO_ROOT}:${MOLMOSPACES_DIR}:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export _IN_MULTIPROCESSING_CHILD=1

# Find a Franka benchmark directory in ithor scenes
BENCHMARK_DIR=$(find "$CACHE_DIR/benchmarks" -path "*/ithor/FrankaPickHardBench/*/benchmark.json" 2>/dev/null | head -1)
if [ -n "$BENCHMARK_DIR" ]; then
    BENCHMARK_DIR="$(dirname "$BENCHMARK_DIR")"
    echo "[$(date)] Found benchmark: $BENCHMARK_DIR"
else
    echo "[$(date)] ERROR: No benchmark.json found in cache."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

EVAL_ARGS="--port $SERVER_PORT --benchmark_dir $BENCHMARK_DIR --task_horizon $TASK_HORIZON --no_wandb"
if [ -n "$EVAL_IDX" ]; then
    EVAL_ARGS="$EVAL_ARGS --idx $EVAL_IDX"
fi
bash experiments/molmospaces/run_eval.sh $EVAL_ARGS

EVAL_EXIT=$?
echo "[$(date)] Evaluation exited with code: $EVAL_EXIT"

# ── Cleanup ──
echo ""
echo "[$(date)] Shutting down server..."
kill $SERVER_PID 2>/dev/null
timeout 30 bash -c "wait $SERVER_PID" 2>/dev/null
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "[$(date)] Server still alive after SIGTERM, sending SIGKILL..."
    kill -9 $SERVER_PID 2>/dev/null
fi
kill $XVFB_PID 2>/dev/null
echo "[$(date)] Done. Eval exit code: $EVAL_EXIT"
