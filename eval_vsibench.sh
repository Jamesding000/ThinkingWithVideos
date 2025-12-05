#!/bin/bash
set -x

model="Qwen3-VL-2B-Thinking"

# Download model
DIRECTORY="models"

if [ -d "$DIRECTORY" ]; then
    echo "Directory '$DIRECTORY' exists."
else
    mkdir $DIRECTORY
fi

DIRECTORY="models/${model}"

if [ -d "$DIRECTORY" ]; then
    echo "'$model' exists."
else
    mkdir $DIRECTORY
    cd $DIRECTORY
    git clone https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking
    cd ../..
fi

# Download data for VSI-Bench
DIRECTORY="data/vsibench"

if [ -d "$DIRECTORY" ]; then
    echo "Directory '$DIRECTORY' exists."
else
    echo "Directory '$DIRECTORY' does not exist, downloading and processing..."
    mkdir data/vsibench
    mkdir data/vsibench/raw_videos
    git clone https://huggingface.co/datasets/nyu-visionx/VSI-Bench
    unzip VSI-Bench/arkitscenes.zip -d data/vsibench/raw_videos
    unzip VSI-Bench/scannet.zip -d data/vsibench/raw_videos
    unzip VSI-Bench/scannetpp.zip -d data/vsibench/raw_videos
    rm -rf VSI-Bench
    gdown https://drive.google.com/uc?id=1aHoEEHl63ErXDuV2K_pHNnshVUiIzDaj -O data/vsibench/test_set_grpo.json
    python extract_frames.py \
        --input-dir data/vsibench/raw_videos \
        --output-dir data/vsibench/video_14400frames_fps2 \
        --dataset vsibench \
        --fps 2.0 \
        --max-frames 14400 \
        --num-workers 8
    echo "Data downloaded and processed!"
fi


# VSI-Bench Evaluation Script
# Based on test_eval.sh, configured specifically for VSI-Bench dataset

# Model configuration
export model="Qwen3-VL-2B-Thinking"
export model_path="models/${model}"

# GPU configuration
export n_gpus_per_node=1

# Evaluation parameters
max_pixels_expr='384*384'
max_pixels=$((${max_pixels_expr}))
max_frames=256

# Dataset configuration
dataset=vsibench
step=100  # Set to your checkpoint step or 0 if using base model

# Output path configuration
export project_name=sft
export exp_suffix=thinking_lr1e_5
export batch_size=128
export dataset_train=data_mtvr_cot
export EXP_NAME=${model}_${exp_suffix}_${n_gpus_per_node}g_sft_${dataset_train}_bs${batch_size}
export SAVE_PATH=outputs/${project_name}/${EXP_NAME}


# Logging configuration
export DATE="$(TZ='Asia/Shanghai' date +%m%d_%H%M%S)"
export HYDRA_FULL_ERROR=1 
export PYTHONUNBUFFERED=1 
export VERL_LOGGING_LEVEL=DEBUG

# VLLM configuration
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Create output directory if it doesn't exist
mkdir -p ${SAVE_PATH}

echo "=========================================="
echo "Starting VSI-Bench evaluation"
echo "Dataset: ${dataset}"
echo "Model: ${model}"
echo "Step: ${step}"
echo "Max pixels: ${max_pixels_expr}"
echo "Max frames: ${max_frames}"
echo "Output directory: ${SAVE_PATH}/global_step_${step}"
echo "=========================================="

# Run evaluation
python eval_VQA.py \
    --inference_file inference_vllm_origin_number.py \
    --model-path ${model_path} \
    --dataset ${dataset} \
    --output_dir ${SAVE_PATH}/global_step_${step} \
    --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
    --num_chunks ${n_gpus_per_node} \
    --max_pixels ${max_pixels} \
    --max_frames ${max_frames} \
    --prompt_type TG_TRAIN_TEMPLATE \
    --no_cache \
    2>&1 | tee "${SAVE_PATH}/${DATE}_eval_vsibench_step${step}.log"

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${SAVE_PATH}/global_step_${step}/evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number/${dataset}"
echo "Log file: ${SAVE_PATH}/${DATE}_eval_vsibench_step${step}.log"
echo "=========================================="

# Optional: Clear data
# rm -rf data/vsibench