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
export n_gpus_per_node=8

max_pixels_expr='384*384'
max_pixels=$((${max_pixels_expr}))
max_frames=256

for dataset in charades_sta_src actnet_tg_src
do
    for step in 50 100
    do
        echo start eval ${dataset} at step ${step}
        suffix=step${step}
        python eval_temporal_grounding.py \
            --inference_file inference_vllm_multiturn_number.py \
            --model-path ${SAVE_PATH}/global_step_${step} \
            --dataset ${dataset} \
            --output_dir ${SAVE_PATH}/global_step_${step} \
            --evaluation_name evaluation_maxpix${max_pixels_expr}_number \
            --num_chunks ${n_gpus_per_node} \
            --max_pixels ${max_pixels} \
            --prompt_type TG_TRAIN_TEMPLATE \
            --no_cache \
            >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
    done
done
for dataset in next_gqa rextime_val
do
    for step in 50 100
    do
        echo start eval ${dataset} at step ${step}
        suffix=step${step}
        python eval_GQA.py \
            --inference_file inference_vllm_multiturn_number.py \
            --model-path ${SAVE_PATH}/global_step_${step} \
            --dataset ${dataset} \
            --output_dir ${SAVE_PATH}/global_step_${step} \
            --evaluation_name evaluation_maxpix${max_pixels_expr}_number \
            --num_chunks ${n_gpus_per_node} \
            --max_pixels ${max_pixels} \
            --prompt_type TG_TRAIN_TEMPLATE \
            --no_cache \
            >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
    done
done
for dataset in mmvu videommmu vsibench
do
    for step in 50 100
    do
        echo start eval ${dataset} at step ${step}
        suffix=step${step}
        python eval_VQA.py \
            --inference_file inference_vllm_multiturn_number.py \
            --model-path ${SAVE_PATH}/global_step_${step} \
            --dataset ${dataset} \
            --output_dir ${SAVE_PATH}/global_step_${step} \
            --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
            --num_chunks ${n_gpus_per_node} \
            --max_pixels ${max_pixels} \
            --max_frames ${max_frames} \
            --prompt_type TG_TRAIN_TEMPLATE \
            --no_cache \
            >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
    done
done


max_pixels_expr='224*224'
max_pixels=$((${max_pixels_expr}))
max_frames=1024
for dataset in vidchapter_src
do
    for step in 50 100
    do
        echo start eval ${dataset} at step ${step}
        suffix=step${step}
        python eval_temporal_grounding.py \
            --inference_file inference_vllm_multiturn_number.py \
            --model-path ${SAVE_PATH}/global_step_${step} \
            --dataset ${dataset} \
            --output_dir ${SAVE_PATH}/global_step_${step} \
            --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
            --num_chunks ${n_gpus_per_node} \
            --max_pixels ${max_pixels} \
            --max_frames ${max_frames} \
            --prompt_type TG_TRAIN_TEMPLATE \
            --no_cache \
            >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
    done
done
for dataset in vidi_src
do
    for step in 50 100
    do
        echo start eval ${dataset} at step ${step}
        suffix=step${step}
        python eval_temporal_grounding_multi.py \
            --inference_file inference_vllm_multiturn_number.py \
            --model-path ${SAVE_PATH}/global_step_${step}/Qwen2.5-VL-7B-Instruct \
            --dataset ${dataset} \
            --output_dir ${SAVE_PATH}/global_step_${step} \
            --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
            --num_chunks ${n_gpus_per_node} \
            --max_pixels ${max_pixels} \
            --max_frames ${max_frames} \
            --prompt_type TG_TRAIN_TEMPLATE \
            --no_cache \
            >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
    done
done
for dataset in longvideo-reason videomme
do
    for step in 50 100
    do
        echo start eval ${dataset} 
        suffix=step${step}
        python eval_VQA.py \
            --inference_file inference_vllm_multiturn_number.py \
            --model-path ${SAVE_PATH}/global_step_${step} \
            --dataset ${dataset} \
            --output_dir ${SAVE_PATH}/global_step_${step} \
            --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
            --num_chunks ${n_gpus_per_node} \
            --max_pixels ${max_pixels} \
            --max_frames ${max_frames} \
            --prompt_type TG_TRAIN_TEMPLATE \
            --no_cache \
            >> "${SAVE_PATH}/${DATE}_eval_qwen2vl7b_${dataset}_${suffix}.log" 2>&1 
    done
done