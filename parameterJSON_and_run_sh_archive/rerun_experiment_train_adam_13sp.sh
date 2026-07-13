#!/bin/bash
#SBATCH --job-name=saluki_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --partition=student_project

echo "Starting job on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Source conda initialization
source $(conda info --base)/etc/profile.d/conda.sh
conda activate saluki_pytorch

python3 main.py \
    params_rerun_experiment_train_adam_13sp.json \
    -o outputs/out_setup_2 \
    --wandb_project saluki_pytorch \
    --run_name test_run_F_adam_l2Reg_13sp
