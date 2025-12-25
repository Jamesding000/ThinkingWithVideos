#!/usr/bin/env bash
set -euo pipefail

# Datasets to process
DATASETS=("vidchapters" "longvideo-reason")
INPUT_BASE_DIR="/data/user_data/jamesdin/data"
OUTPUT_BASE_DIR="/data/user_data/jamesdin/data"
FPS=2
MAX_FRAMES=14400
NUM_WORKERS=8   # 4, 8, 16

for dataset in "${DATASETS[@]}"; do
    INPUT_DIR="${INPUT_BASE_DIR}/${dataset}/raw_videos"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${dataset}/video_${MAX_FRAMES}frames_fps${FPS}"

    python extract_frames.py \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --dataset "${dataset}" \
        --fps "${FPS}" \
        --max-frames "${MAX_FRAMES}" \
        --num-workers "${NUM_WORKERS}"
done
