set -x

# # path to your inner verl repo
# export VERL_REPO="/home/jamesdin/James/ThinkingWithVideos/verl"

# # make sure Python imports the right 'verl' package
# export PYTHONPATH="$VERL_REPO:$PYTHONPATH"

export user_prompt_template=THINK_GENERAL_TOOL
export MY_PROMPT_TEMPLATE="This is a video with duration {duration} seconds.
You should first think the user's question step-by-step and then provides the user with the answer. 
Note that there must be an answer for each question and you must answer it.
# Instruction
1. Output your thought process within the <think> </think> tags, and output your answer within the <answer> </answer> tags.
2. You can call the provided tools ONCE to get more visual information within the <tool_call> </tool_call> tags. 
3. When you get the tool result, you need to integrate your initial reasoning with the new visual evidence from the tool, think step-by-step again and provide the final answer. 
# Output Format
<think> ... </think> <tool_call> ... </tool_call> <think> ... </think> <answer> ... </answer>
User Question: 
{input_text}"

# training
export nnodes=1
export n_gpus_per_node=4
export batch_size=128
export micro_batch_size_per_gpu=1
# data
export dataset=data_mtvr_cot_tool
# model
# export model_path=models/Qwen2.5-VL-7B-Instruct
export model_path=/data/user_data/jamesdin/models/Qwen2.5-VL-3B-Instruct
export max_turns=0
export tool_config_path=verl/verl/tools/config/zoom_tool_config_new.yaml

# export dataset_json_path=[data/charades/train_set_sft_new_10k6_src_diff.json,data/actnet/train_set_sft_new_9k5_src_diff.json,data/vidchapters/train_set_sft_diff_exist_4k3.json,data/nextgqa/train_set_sft_src_diff_3k.json,data/rextime/train_set_sft_exist_src_diff_3k7.json,data/Video-R1-data/train_set_sft_005_video.json,data/Video-R1-data/train_set_sft_005_image.json,data/longvideo-reason/train_set_sft_src_exist_8k.json]
# export dataset_json_path=[data/MultiTaskVideoReasoning/MTVR_CoT/actnet.json,data/MultiTaskVideoReasoning/MTVR_CoT/charades.json,data/MultiTaskVideoReasoning/MTVR_CoT/vidchapters.json,data/MultiTaskVideoReasoning/MTVR_CoT/nextgqa.json,data/MultiTaskVideoReasoning/MTVR_CoT/rextime.json,data/MultiTaskVideoReasoning/MTVR_CoT/longvideo-reason.json,data/MultiTaskVideoReasoning/MTVR_CoT/Video-R1-data-image.json,data/MultiTaskVideoReasoning/MTVR_CoT/Video-R1-data-video.json]
# export dataset_json_path=[data/MultiTaskVideoReasoning/MTVR_CoT/charades.json,data/MultiTaskVideoReasoning/MTVR_CoT/nextgqa.json]
export dataset_json_path=[data/MultiTaskVideoReasoning/MTVR_Tool_CoT/longvideo-reason.json,data/MultiTaskVideoReasoning/MTVR_Tool_CoT/vidchapters.json]
export dataset_json_path_val=[]
# export dataset_video_base=[data/charades/video_14400frames_fps2,data/actnet/video_14400frames_fps2,data/vidchapters/video_14400frames_fps2,data/nextgqa/video_14400frames_fps2,data/rextime/video_14400frames_fps2,data/Video-R1-data/video_14400frames_fps2,data/Video-R1-data,data/longvideo-reason/train_video_14400frames_fps2]
# export dataset_video_base=[/data/user_data/jamesdin/data/charades/video_14400frames_fps2,/data/user_data/jamesdin/data/nextqa/video_14400frames_fps2]
export dataset_video_base=[/data/user_data/jamesdin/data/longvideo-reason/video_14400frames_fps2,/data/user_data/jamesdin/data/vidchapters/video_14400frames_fps2]

export max_prompt_length=10240  # 10240
# name
export project_name=sft_tool
export exp_suffix=thinking_tool_lr1e_5

# auto config
export EXP_NAME=qwen2.5_vl_3b_instruct_${exp_suffix}_${n_gpus_per_node}g_sft_${dataset}_bs${batch_size}
export SAVE_PATH=/data/user_data/jamesdin/outputs/${project_name}/${EXP_NAME}

export WANDB_API_KEY=$(jq -r '.WANDB_API_KEY' secret.json)
export WANDB_MODE=online  # TODO: setitng to online cause failed to login issue
export WANDB_DIR=${SAVE_PATH}
export WANDB_CONFIG_DIR=${SAVE_PATH}
export DATE="$(TZ='Asia/Shanghai' date +%m%d_%H%M%S)"
# print, logging, and debug
export HYDRA_FULL_ERROR=1 
export PYTHONUNBUFFERED=1 
export VERL_LOGGING_LEVEL=INFO
export CONSOLE_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_training.log
export LOGGER_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_logging.log



mkdir -p $SAVE_PATH
touch $CONSOLE_OUTPUT_FILE $LOGGER_OUTPUT_FILE
chown -R tiger $SAVE_PATH 

echo "start sft, write to ${SAVE_PATH}"
echo "MY_PROMPT_TEMPLATE=$MY_PROMPT_TEMPLATE"

torchrun \
    --nproc_per_node=$n_gpus_per_node \
    --standalone \
    -m verl.trainer.fsdp_sft_trainer \
    optim.lr=1e-5 \
    +data.max_prompt_length=$max_prompt_length \
    data.train_files=$dataset_json_path \
    data.val_files=$dataset_json_path_val \
    data.train_batch_size=$batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.prompt_key=prompt \
    +data.image_key=images \
    +data.video_key=videos \
    data.custom_cls.path=verl/verl/utils/dataset/sft_dataset_multi_turn_qwen2_5.py \
    data.custom_cls.name=SFTDatasetMultiTurn \
    +data.user_prompt_template=$user_prompt_template \
    +data.response_dict_keys=['answer'] \
    +data.multi_turn.max_turns=$max_turns \
    +data.multi_turn.tool_config_path=$tool_config_path \
    +data.multi_turn.video_base=$dataset_video_base \
    +data.multi_turn.video_kwargs.draw_number=true \
    +data.multi_turn.video_kwargs.parallel=true \
    +data.multi_turn.video_kwargs.max_frames=128 \
    model.partial_pretrain=$model_path \
    model.fsdp_config.model_dtype=bf16 \
    model.use_liger=True \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$EXP_NAME \
    trainer.logger=['console','wandb'] \
    trainer.save_freq=50 \
    trainer.test_freq=0 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=null \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    use_remove_padding=true \