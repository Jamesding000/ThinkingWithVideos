# example
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /data/user_data/jamesdin/outputs/sft_then_grpo_then_sft_then_grpo/qwen3_vl_2b_thinking_lr1e_6_8g_grpo_data_mtvr_cot_tool_rl_gs4_bs8/global_step_400/actor \
    --target_dir /data/user_data/jamesdin/exports/qwen3_vl_2b_thinking_tool_rl_step400_hf

# example
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /data/user_data/jamesdin/outputs/sft/qwen2_5_vl_7b_thinking_lr1e_5_4g_sft_data_mtvr_cot_bs128/global_step_41/actor \
    --target_dir /data/user_data/jamesdin/exports/qwen2_5_vl_7b_instruct_step41_hf

# example
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /data/user_data/jamesdin/outputs/sft/qwen3_vl_2b_thinking_thinking_lr1e_5_4g_sft_data_mtvr_cot_bs128/global_step_41/actor \
    --target_dir /data/user_data/jamesdin/exports/qwen3_vl_2b_thinking_step41_hf

# example
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /data/user_data/jamesdin/outputs/sft_then_grpo_then_sft_then_grpo/qwen3_vl_2b_thinking_lr1e_6_8g_grpo_data_mtvr_cot_tool_rl_gs4_bs8/global_step_577/actor \
    --target_dir /data/user_data/jamesdin/exports/qwen3_vl_2b_thinking_tool_rl_step577_hf
