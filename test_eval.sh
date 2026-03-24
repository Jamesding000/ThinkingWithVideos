set -x

# Command
# python3 inference_vllm_origin_number.py --model-path outputs/sft/qwen2_5_vl_7b_thinking_lr1e_5_1g_sft_data_mtvr_cot_bs128/global_step_100 --video_dir data/nextgqa/video_14400frames_fps2 --gt_file data/nextgqa/test_set_grpo_src.json --output_dir outputs/sft/qwen2_5_vl_7b_thinking_lr1e_5_1g_sft_data_mtvr_cot_bs128/global_step_100/evaluation_maxpix384*384_number/next_gqa --output_name pred --num-chunks 1 --chunk-idx 0 --max_pixels 147456 --no-cache

# training
export nnodes=1
export n_gpus_per_node=1
export batch_size=128
export micro_batch_size_per_gpu=1
# data
export dataset=data_mtvr_cot
# model
# export model_path=models/Qwen2.5-VL-7B-Instruct
export model_path=models/Qwen3-VL-2B-Instruct
# export dataset_json_path=[data/charades/train_set_sft_new_10k6_src_diff.json,data/actnet/train_set_sft_new_9k5_src_diff.json,data/vidchapters/train_set_sft_diff_exist_4k3.json,data/nextgqa/train_set_sft_src_diff_3k.json,data/rextime/train_set_sft_exist_src_diff_3k7.json,data/Video-R1-data/train_set_sft_005_video.json,data/Video-R1-data/train_set_sft_005_image.json,data/longvideo-reason/train_set_sft_src_exist_8k.json]
export dataset_json_path=[data/MultiTaskVideoReasoning/MTVR_CoT/actnet.json,data/MultiTaskVideoReasoning/MTVR_CoT/charades.json,data/MultiTaskVideoReasoning/MTVR_CoT/vidchapters.json,data/MultiTaskVideoReasoning/MTVR_CoT/nextgqa.json,data/MultiTaskVideoReasoning/MTVR_CoT/rextime.json,data/MultiTaskVideoReasoning/MTVR_CoT/longvideo-reason.json,data/MultiTaskVideoReasoning/MTVR_CoT/Video-R1-data-image.json,data/MultiTaskVideoReasoning/MTVR_CoT/Video-R1-data-video.json]
export dataset_json_path_val=[]
export dataset_video_base=[data/charades/video_14400frames_fps2,data/actnet/video_14400frames_fps2,data/vidchapters/video_14400frames_fps2,data/nextgqa/video_14400frames_fps2,data/rextime/video_14400frames_fps2,data/Video-R1-data/video_14400frames_fps2,data/Video-R1-data,data/longvideo-reason/train_video_14400frames_fps2]
export max_prompt_length=4096
# name
export project_name=sft
export exp_suffix=thinking_lr1e_5

# auto config
export EXP_NAME=qwen2_5_vl_7b_${exp_suffix}_${n_gpus_per_node}g_sft_${dataset}_bs${batch_size}
export SAVE_PATH=/data/user_data/jamesdin/outputs/${project_name}/${EXP_NAME}

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

max_pixels_expr='384*384'
max_pixels=$((${max_pixels_expr}))
max_frames=256

dataset=next_gqa
step=100
echo start eval ${dataset} at step ${step}
suffix=step${step}
python eval_GQA.py \
    --inference_file inference_vllm_origin_number.py \
    --model-path ${model_path} \
    --dataset ${dataset} \
    --output_dir ${SAVE_PATH}/global_step_${step} \
    --evaluation_name evaluation_maxpix${max_pixels_expr}_number \
    --num_chunks ${n_gpus_per_node} \
    --max_pixels ${max_pixels} \
    --prompt_type TG_TRAIN_TEMPLATE \
    --no_cache
    # >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 


# for dataset in charades_sta_src actnet_tg_src
# do
#     for step in 100 200
#     do
#         echo start eval ${dataset} at step ${step}
#         suffix=step${step}
#         python eval_temporal_grounding.py \
#             --inference_file inference_vllm_origin_number.py \
#             --model-path ${SAVE_PATH}/global_step_${step} \
#             --dataset ${dataset} \
#             --output_dir ${SAVE_PATH}/global_step_${step} \
#             --evaluation_name evaluation_maxpix${max_pixels_expr}_number \
#             --num_chunks ${n_gpus_per_node} \
#             --max_pixels ${max_pixels} \
#             --prompt_type TG_TRAIN_TEMPLATE \
#             --no_cache \
#             >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
#     done
# done
# for dataset in next_gqa rextime_val
# do
#     for step in 100 200
#     do
#         echo start eval ${dataset} at step ${step}
#         suffix=step${step}
#         python eval_GQA.py \
#             --inference_file inference_vllm_origin_number.py \
#             --model-path ${SAVE_PATH}/global_step_${step} \
#             --dataset ${dataset} \
#             --output_dir ${SAVE_PATH}/global_step_${step} \
#             --evaluation_name evaluation_maxpix${max_pixels_expr}_number \
#             --num_chunks ${n_gpus_per_node} \
#             --max_pixels ${max_pixels} \
#             --prompt_type TG_TRAIN_TEMPLATE \
#             --no_cache \
#             >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
#     done
# done
# for dataset in mmvu videommmu vsibench
# do
#     for step in 100 200
#     do
#         echo start eval ${dataset} at step ${step}
#         suffix=step${step}
#         python eval_VQA.py \
#             --inference_file inference_vllm_origin_number.py \
#             --model-path ${SAVE_PATH}/global_step_${step} \
#             --dataset ${dataset} \
#             --output_dir ${SAVE_PATH}/global_step_${step} \
#             --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
#             --num_chunks ${n_gpus_per_node} \
#             --max_pixels ${max_pixels} \
#             --max_frames ${max_frames} \
#             --prompt_type TG_TRAIN_TEMPLATE \
#             --no_cache \
#             >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
#     done
# done


# max_pixels_expr='224*224'
# max_pixels=$((${max_pixels_expr}))
# max_frames=1024
# for dataset in vidchapter_src
# do
#     for step in 100 200
#     do
#         echo start eval ${dataset} at step ${step}
#         suffix=step${step}
#         python eval_temporal_grounding.py \
#             --inference_file inference_vllm_origin_number.py \
#             --model-path ${SAVE_PATH}/global_step_${step} \
#             --dataset ${dataset} \
#             --output_dir ${SAVE_PATH}/global_step_${step} \
#             --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
#             --num_chunks ${n_gpus_per_node} \
#             --max_pixels ${max_pixels} \
#             --max_frames ${max_frames} \
#             --prompt_type TG_TRAIN_TEMPLATE \
#             --no_cache \
#             >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
#     done
# done
# for dataset in vidi_src
# do
#     for step in 100 200
#     do
#         echo start eval ${dataset} at step ${step}
#         suffix=step${step}
#         python eval_temporal_grounding_multi.py \
#             --inference_file inference_vllm_origin_number.py \
#             --model-path ${SAVE_PATH}/global_step_${step}/Qwen2.5-VL-7B-Instruct \
#             --dataset ${dataset} \
#             --output_dir ${SAVE_PATH}/global_step_${step} \
#             --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
#             --num_chunks ${n_gpus_per_node} \
#             --max_pixels ${max_pixels} \
#             --max_frames ${max_frames} \
#             --prompt_type TG_TRAIN_TEMPLATE \
#             --no_cache \
#             >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
#     done
# done
# for dataset in longvideo-reason videomme
# do
#     for step in 100 200
#     do
#         echo start eval ${dataset} 
#         suffix=step${step}
#         python eval_VQA.py \
#             --inference_file inference_vllm_origin_number.py \
#             --model-path ${SAVE_PATH}/global_step_${step} \
#             --dataset ${dataset} \
#             --output_dir ${SAVE_PATH}/global_step_${step} \
#             --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
#             --num_chunks ${n_gpus_per_node} \
#             --max_pixels ${max_pixels} \
#             --max_frames ${max_frames} \
#             --prompt_type TG_TRAIN_TEMPLATE \
#             --no_cache \
#             >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
#     done
# done
