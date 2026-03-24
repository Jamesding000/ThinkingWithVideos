#!/bin/bash
set -x

# VSI-Bench Evaluation Script
# Based on test_eval.sh, configured specifically for VSI-Bench dataset

# Model configuration
export model="Qwen3-VL-2B-Thinking"
# export model_path=/data/user_data/jamesdin/models/Qwen3-VL-2B-Thinking
expor



# GPU configuration
export n_gpus_per_node=2

# Evaluation parameters
max_pixels_expr='384*384'
max_pixels=$((${max_pixels_expr}))
max_frames=256

# Dataset configuration
dataset=vsibench
step=200  # Set to your checkpoint step or 0 if using base model

# Output path configuration
export project_name=eval
export exp_suffix=thinking_lr1e_5
export batch_size=128
export dataset_train=data_mtvr_cot
export EXP_NAME=${model}_${exp_suffix}_${n_gpus_per_node}g_sft_${dataset_train}_bs${batch_size}
# export SAVE_PATH=/data/user_data/jamesdin/outputs/${project_name}/${EXP_NAME}
export SAVE_PATH=/data/user_data/jamesdin/outputs/${project_name}

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
    --inference_file inference_vllm_origin_number_qwen3.py \
    --model-path ${model_path} \
    --dataset ${dataset} \
    --output_dir ${SAVE_PATH}/${model}/global_step_${step} \
    --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
    --num_chunks ${n_gpus_per_node} \
    --max_pixels ${max_pixels} \
    --max_frames ${max_frames} \
    --prompt_type TG_TRAIN_TEMPLATE \
    --no_cache

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${SAVE_PATH}/global_step_${step}/evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number/${dataset}"
echo "Log file: ${SAVE_PATH}/${DATE}_eval_vsibench_step${step}.log"
echo "=========================================="

# Optional: Clear data
# rm -rf data/vsibench
