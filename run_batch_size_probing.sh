#!/bin/bash
#SBATCH --job-name=probe_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe_%j.log
#SBATCH --error=logs/probe_%j.err
#SBATCH --partition=student_project

echo "Starting job on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Source conda initialization
source $(conda info --base)/etc/profile.d/conda.sh
conda activate saluki_pytorch

# Run the batch size probe
python probe_batch_size.py