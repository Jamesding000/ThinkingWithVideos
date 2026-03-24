#!/usr/bin/bash
#SBATCH --job-name=verl-rl-qwen3
#SBATCH --partition=general          # general partition
#SBATCH --time=47:00:00              # <= 48h limit on general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 1 task per GPU
#SBATCH --gres=gpu:L40S:8           # 4 GPUs
#SBATCH --cpus-per-task=8            # 8 * 4 = 32 CPUs total
#SBATCH --mem=1250GB                  # safe for 4 GPUs + dataloaders
#SBATCH --output=/home/jamesdin/logs/verl-rl-%j.out
#SBATCH --error=/home/jamesdin/logs/verl-rl-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jamesdin@andrew.cmu.edu

export TMPDIR=/scratch/job_tmp
echo "Using TMPDIR=$TMPDIR"
echo "==== Job started on $(hostname) at $(date) ===="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

mkdir -p /home/jamesdin/logs
export PYTHONUNBUFFERED=1

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export nnodes=${SLURM_NNODES:-1}
export n_gpus_per_node=${SLURM_GPUS_ON_NODE:-8}
export WORLD_SIZE=$(( nnodes * n_gpus_per_node ))

echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"

# Launch your RL script (which sets n_gpus_per_node=4, n_cpus=16, etc.)
bash test_rl_tool_babel.sh

echo "==== Job finished at $(date) ===="