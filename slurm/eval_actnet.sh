#!/usr/bin/bash
#SBATCH --job-name=eval-actnet
#SBATCH --partition=general          # Use the general partition
#SBATCH --time=47:00:00              # <= 48h limit on general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 1 task per GPU
#SBATCH --gres=gpu:A6000:4           # Request 4 A6000 GPUs
#SBATCH --cpus-per-task=16           # For dataloaders / preprocessing
#SBATCH --mem=250GB                  # Adjust as needed, safe for 4 GPUs
#SBATCH --output=/home/jamesdin/logs/eval_actnet-%j.out
#SBATCH --error=/home/jamesdin/logs/eval_actnet-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jamesdin@andrew.cmu.edu

# Choose a scratch location; adjust to your cluster layout
SCRATCH_BASE=/scratch/$USER
export TMPDIR=${SCRATCH_BASE}/job_${SLURM_JOB_ID}

mkdir -p "${TMPDIR}"

# Ray will put sessions/logs here instead of /tmp
export RAY_TEMP_DIR="${TMPDIR}/ray"
mkdir -p "${RAY_TEMP_DIR}"

echo "Using TMPDIR=${TMPDIR}"
echo "Using RAY_TEMP_DIR=${RAY_TEMP_DIR}"
echo "==== Job started on $(hostname) at $(date) ===="

# ------------------------------
# Conda environment
# ------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Make sure log dir exists (matches SBATCH paths)
mkdir -p /home/jamesdin/logs

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

# Model configuration
# export model_paths=(/data/user_data/jamesdin/models/Qwen3-VL-2B-Thinking /data/user_data/jamesdin/exports/qwen3_vl_2b_thinking_step41_hf /data/user_data/jamesdin/exports/qwen3_vl_2b_thinking_tool_rl_step577_hf)
# export model_paths=(/data/user_data/jamesdin/exports/qwen3_vl_2b_thinking_step41_hf)
export model_paths=(/data/user_data/jamesdin/models/Qwen2.5-VL-3B-Instruct)

# GPU configuration
export n_gpus_per_node=2

# Evaluation parameters
max_pixels_expr='224*224'
max_pixels=$((${max_pixels_expr}))
max_frames=256

# Dataset configuration
dataset=actnet
step=200  # Set to your checkpoint step or 0 if using base model

# Output path configuration
export project_name=eval
export exp_suffix=thinking_lr1e_5
export batch_size=128
export dataset_train=data_mtvr_cot
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

# Iterate through model paths
for model_path in "${model_paths[@]}"; do
    # Extract model name from base dirname
    model=$(basename ${model_path})
    
    echo "=========================================="
    echo "Starting ActNet evaluation"
    echo "Dataset: ${dataset}"
    echo "Model: ${model}"
    echo "Step: ${step}"
    echo "Max pixels: ${max_pixels_expr}"
    echo "Max frames: ${max_frames}"
    echo "Inference file: ${inference_file}"
    echo "Output directory: ${SAVE_PATH}/${model}"
    echo "=========================================="
    
    # Run evaluation
    python eval_temporal_grounding.py \
        --inference_file inference_vllm_multiturn_number.py \
        --model-path ${model_path} \
        --dataset ${dataset} \
        --output_dir ${SAVE_PATH}/${model} \
        --evaluation_name evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number \
        --num_chunks ${n_gpus_per_node} \
        --max_pixels ${max_pixels} \
        --max_frames ${max_frames} \
        --prompt_type TG_TRAIN_TEMPLATE \
        --no_cache
    echo "=========================================="
    echo "Evaluation complete!"
    echo "Results saved to: ${SAVE_PATH}/${model}/evaluation_maxpix${max_pixels_expr}_maxfrm${max_frames}_number/${dataset}"
    echo "Log file: ${SAVE_PATH}/${DATE}_eval_actnet_step${step}.log"
    echo "=========================================="
done
