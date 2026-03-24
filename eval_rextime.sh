

step=200
SAVE_PATH=/data/user_data/jamesdin/outputs/sft_then_grpo_then_sft_then_grpo/qwen3_vl_2b_thinking_lr1e_6_8g_grpo_data_mtvr_cot_tool_rl_gs4_bs8
dataset=rextime


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