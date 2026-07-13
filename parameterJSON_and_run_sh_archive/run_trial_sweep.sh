#!/bin/bash
#SBATCH --job-name=saluki_trial_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=52G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --partition=student_project
#SBATCH --time=00:15:00

# Self-contained SMOKE TEST: run with `sbatch run_trial_sweep.sh` (no args).
# It creates a one-trial sweep (sweep_saluki_trial.yaml has run_cap: 1 and points
# --base_params at example_params_sweep_trial_13sp.json = full 13 species, 2 epochs)
# and runs that single trial, to validate the whole wandb wiring end-to-end.

set -uo pipefail

echo "Starting job on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"

# Source conda initialization. conda's activation scripts reference unset env
# vars (e.g. MKL_INTERFACE_LAYER), which trip `set -u`; disable nounset just
# around activation, then re-enable it.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate saluki_pytorch
set -u

# 1) Create the trial sweep and parse its "entity/project/sweep_id" spec.
echo "Creating trial sweep from sweep_saluki_trial.yaml ..."
sweep_out="$(wandb sweep --project saluki_sweep_trial --entity project-half-life sweep_saluki_trial.yaml 2>&1)"
echo "$sweep_out"
SWEEP_SPEC="$(echo "$sweep_out" | awk '/Run sweep agent with/{print $NF}' | tail -1)"

if [ -z "$SWEEP_SPEC" ]; then
    echo "ERROR: could not parse the sweep id from the 'wandb sweep' output above." >&2
    echo "       (Are you logged in? Run 'wandb login' once on the login node.)" >&2
    exit 1
fi
echo "Trial sweep: $SWEEP_SPEC"

# 2) Run the single trial. run_cap:1 makes the agent exit after one trial.
wandb agent "$SWEEP_SPEC"

echo "Trial sweep finished at $(date)"
