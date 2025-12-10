#!/bin/bash

set -x

# training
export nnodes=1
export n_gpus_per_node=1
export batch_size=128
export micro_batch_size_per_gpu=1
# data
export dataset=data_mtvr_cot
# model
export model_path=models/Qwen2.5-VL-7B-Instruct
# export dataset_json_path=[data/charades/train_set_sft_new_10k6_src_diff.json,data/actnet/train_set_sft_new_9k5_src_diff.json,data/vidchapters/train_set_sft_diff_exist_4k3.json,data/nextgqa/train_set_sft_src_diff_3k.json,data/rextime/train_set_sft_exist_src_diff_3k7.json,data/Video-R1-data/train_set_sft_005_video.json,data/Video-R1-data/train_set_sft_005_image.json,data/longvideo-reason/train_set_sft_src_exist_8k.json]
export dataset_json_path=[data/processed_data/rextime/rextime_validation.json]
export dataset_json_path_val=[]
export dataset_video_base=[/data/user_data/jamesdin/data/rextime/video_14400frames_fps2]
export max_prompt_length=4096
# name
# export project_name=sft
# export exp_suffix=thinking_lr1e_5
export PYTHONPATH=/home/yiqunh/ThinkingWithVideos/verl:$PYTHONPATH

# auto config
export EXP_NAME=qwen2_5_vl_7b_${exp_suffix}_${n_gpus_per_node}g_sft_${dataset}_bs${batch_size}
export SAVE_PATH=/home/yiqunh/ThinkingWithVideos/processed_data/rextime

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_MODE=offline  # TODO: setitng to online cause failed to login issue
export WANDB_DIR=${SAVE_PATH}
export WANDB_CONFIG_DIR=${SAVE_PATH}
export DATE="$(TZ='Asia/Shanghai' date +%m%d_%H%M%S)"
# print, logging, and debug
export HYDRA_FULL_ERROR=1 
export PYTHONUNBUFFERED=1 
export VERL_LOGGING_LEVEL=DEBUG
export CONSOLE_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_training.log
export LOGGER_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_logging.log


export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export n_gpus_per_node=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

max_pixels_expr='384*384'
max_pixels=$((${max_pixels_expr}))
max_frames=256

dataset=rextime_val
step=100
echo start eval ${dataset} at step ${step}
suffix=step${step}
python eval_GQA.py \
    --inference_file inference_vllm_multiturn_number.py \
    --model-path ${model_path} \
    --dataset ${dataset} \
    --output_dir ${SAVE_PATH}/global_step_${step} \
    --evaluation_name evaluation_maxpix${max_pixels_expr}_number \
    --num_chunks ${n_gpus_per_node} \
    --max_pixels ${max_pixels} \
    --prompt_type TG_TRAIN_TEMPLATE \
    --no_cache
    >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
