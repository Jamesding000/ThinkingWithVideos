#!/usr/bin/bash
#SBATCH --job-name=rextime-data
#SBATCH --partition=cpu          # general partition, 48h limit
#SBATCH --time=47:00:00              # max 47h
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # single process driving everything
#SBATCH --cpus-per-task=16           # plenty of cores for ffmpeg / yt-dlp
#SBATCH --mem=64GB                   # generous but not crazy
#SBATCH --output=/home/jamesdin/logs/rextime-data-%j.out
#SBATCH --error=/home/jamesdin/logs/rextime-data-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jamesdin@andrew.cmu.edu

echo "==== Job started on $(hostname) at $(date) ===="

# ------------------------------
# Scratch / TMPDIR
# ------------------------------
export TMP_DIR=/scratch/$USER
    export REXTIME_TMPDIR=${TMP_DIR}/job_${SLURM_JOB_ID}/rextime
mkdir -p "$REXTIME_TMPDIR"
echo "Using REXTIME_TMPDIR=$REXTIME_TMPDIR"

# ------------------------------
# Conda environment
# ------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate videollava

# Make sure log dir exists (matches SBATCH paths)
mkdir -p /home/jamesdin/logs

export PYTHONUNBUFFERED=1

# ------------------------------
# Go to project directory
# ------------------------------
cd /home/jamesdin/James/ThinkingWithVideos

echo "Current directory: $(pwd)"
echo "Starting data download + frame extraction at $(date)"

# ------------------------------
# Run your data script
# ------------------------------
data_dir=/data/user_data/jamesdin/data/rextime

python download_and_extract_frames.py \
    --annotation-json-path /data/user_data/jamesdin/data/rextime/rextime_video_ids.txt \
    --output-dir /data/user_data/jamesdin/data/rextime \
    --tmp-video-dir /scratch/$USER/tmp \
    --fps 2 \
    --max-frames 14400 \
    --size 224

echo "==== Job finished at $(date) ===="