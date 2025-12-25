#!/usr/bin/bash
#SBATCH --job-name=verl-sft-qwen3
#SBATCH --partition=general          # Use the general partition
#SBATCH --time=47:00:00              # <= 48h limit on general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # 1 task per GPU → 8 GPUs total
#SBATCH --gres=gpu:L40S:8            # request 8 × L40S GPUs
#SBATCH --cpus-per-task=8            # 8 * 8 = 64 CPUs total
#SBATCH --mem=360GB                  # Adjust as needed, safe for 8 GPUs
#SBATCH --output=/home/jamesdin/logs/verl-sft-qwen3-%j.out
#SBATCH --error=/home/jamesdin/logs/verl-sft-qwen3-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jamesdin@andrew.cmu.edu

# Choose a scratch location; adjust to your cluster layout
SCRATCH_BASE=/scratch/$USER
export TMPDIR=${SCRATCH_BASE}/job_${SLURM_JOB_ID}

mkdir -p "${TMPDIR}"

# Ray will put sessions/logs here instead of /tmp
export RAY_TEMP_DIR="${TMPDIR}/ray"
mkdir -p "${RAY_TEMP_DIR}"

echo "Using TMPDIR=${TMPDIR}"
echo "Using RAY_TEMP_DIR=${RAY_TEMP_DIR}"
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
export n_gpus_per_node=${SLURM_GPUS_ON_NODE:-8}
export WORLD_SIZE=$(( nnodes * n_gpus_per_node ))

echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"

######## Run Training Script #######
# qwen2.5
bash scripts/qwen2_5/test_sft_babel.sh
# qwen3
bash scripts/qwen3/test_sft_babel.sh