#!/usr/bin/bash
#SBATCH --job-name=verl-sft
#SBATCH --partition=general          # Use the general partition
#SBATCH --time=47:00:00              # <= 48h limit on general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 1 task per GPU
#SBATCH --gres=gpu:A6000:4           # Request 4 A6000 GPUs
#SBATCH --cpus-per-task=16           # For dataloaders / preprocessing
#SBATCH --mem=120GB                  # Adjust as needed, safe for 4 GPUs
#SBATCH --output=/home/jamesdin/logs/verl-sft-%j.out
#SBATCH --error=/home/jamesdin/logs/verl-sft-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jamesdin@andrew.cmu.edu

# Choose a scratch location; adjust to your cluster layout
export TMPDIR=/scratch/job_tmp
echo "Using TMPDIR=$TMPDIR"

echo "==== Job started on $(hostname) at $(date) ===="

# ------------------------------
# Conda environment
# ------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Make sure log dir exists (matches SBATCH paths)
mkdir -p /home/jamesdin/logs

export PYTHONUNBUFFERED=1

# ------------------------------
# NCCL / distributed env
# ------------------------------
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1     # Babel often needs this
export NCCL_P2P_DISABLE=1

# Let Slurm drive world size
export nnodes=${SLURM_NNODES:-1}
export n_gpus_per_node=${SLURM_GPUS_ON_NODE:-4}
export WORLD_SIZE=$(( nnodes * n_gpus_per_node ))

echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"

# Run Training Script
bash test_sft_babel.sh

echo "==== Job finished at $(date) ===="
