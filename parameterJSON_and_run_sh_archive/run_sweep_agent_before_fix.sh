#!/bin/bash
#SBATCH --job-name=saluki_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --partition=student_project
#SBATCH --time=07:50:00
#SBATCH --requeue

# wandb sweep agent for the REAL Saluki hyperparameter sweep.
#
# Run with `sbatch run_sweep_agent.sh` (no args needed). Each job pulls trials
# from the shared sweep controller and runs them back-to-back until a ~7h40m soft
# deadline, then exits. Launch as many as you like -- they coordinate via one
# shared sweep and never run duplicate configs:
#   sbatch run_sweep_agent.sh                       # first job: creates + joins the sweep
#   sbatch --array=1-4 run_sweep_agent.sh           # scale up: more agents on the SAME sweep
# (You can also pin an id explicitly: `sbatch run_sweep_agent.sh <ENTITY/PROJECT/SWEEP_ID>`.)

set -uo pipefail

echo "Starting sweep agent on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"

# Source conda initialization
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate saluki_pytorch

# --- Resolve the sweep id ---------------------------------------------------
# Priority: (1) CLI arg, (2) $SWEEP_ID env var, (3) sweep_id.txt, (4) create a new
# sweep from sweep_saluki_13sp.yaml and record it in sweep_id.txt so later jobs
# reuse it. Creation is flock-guarded so an array submitted before the file
# exists still makes only ONE sweep.
SWEEP_FILE="sweep_id.txt"
YAML="sweep_saluki_13sp.yaml"

if [ -n "${1:-}" ]; then
    SWEEP_SPEC="$1"
elif [ -n "${SWEEP_ID:-}" ]; then
    SWEEP_SPEC="$SWEEP_ID"
else
    exec 9>".${SWEEP_FILE}.lock"
    flock 9
    if [ ! -s "$SWEEP_FILE" ]; then
        echo "No $SWEEP_FILE yet; creating the sweep from $YAML ..."
        sweep_out="$(wandb sweep --project saluki_sweep --entity project-half-life "$YAML" 2>&1)"
        echo "$sweep_out"
        echo "$sweep_out" | awk '/Run sweep agent with/{print $NF}' | tail -1 > "$SWEEP_FILE"
    fi
    flock -u 9
    SWEEP_SPEC="$(cat "$SWEEP_FILE")"
fi

if [ -z "$SWEEP_SPEC" ]; then
    echo "ERROR: no sweep id (arg, \$SWEEP_ID, and $SWEEP_FILE are all empty)." >&2
    echo "       (Are you logged in? Run 'wandb login' once on the login node.)" >&2
    exit 1
fi
echo "Sweep: $SWEEP_SPEC"

# --- Run trials until the soft deadline -------------------------------------
# SALUKI_JOB_DEADLINE (~7h40m) is read by train.py's wall-clock guard, which
# stops an in-flight trial cleanly at an epoch boundary before the 8h wall.
export SALUKI_JOB_DEADLINE=$(( $(date +%s) + 27600 ))
while [ "$(date +%s)" -lt "$SALUKI_JOB_DEADLINE" ]; do
    # --count 1: one trial, then re-check the clock. Non-zero exit (e.g. sweep
    # finished / run_cap reached) breaks the loop.
    wandb agent --count 1 "$SWEEP_SPEC" || break
done

echo "Sweep agent finished at $(date)"
