# /bin/bash!

set -x

# Download data for VSI-Bench
DATA_DIR="/data/user_data/jamesdin/data/vsibench"

if [ -d "$DIRECTORY" ]; then
    echo "Directory '$DIRECTORY' exists."
else
echo "Directory '$DIRECTORY' does not exist, downloading and processing..."
mkdir ${DATA_DIR}
mkdir ${DATA_DIR}/raw_videos
git clone https://huggingface.co/datasets/nyu-visionx/VSI-Bench
unzip VSI-Bench/arkitscenes.zip -d ${DATA_DIR}/raw_videos
unzip VSI-Bench/scannet.zip -d ${DATA_DIR}/raw_videos
unzip VSI-Bench/scannetpp.zip -d ${DATA_DIR}/raw_videos
rm -rf VSI-Bench
gdown https://drive.google.com/uc?id=1aHoEEHl63ErXDuV2K_pHNnshVUiIzDaj -O ${DATA_DIR}/test_set_grpo.json
python extract_frames.py \
    --input-dir ${DATA_DIR}/raw_videos \
    --output-dir ${DATA_DIR}/video_14400frames_fps2 \
    --dataset vsibench \
    --fps 2.0 \
    --max-frames 14400 \
    --num-workers 8
echo "Data downloaded and processed!"
fi

