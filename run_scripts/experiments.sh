#!/usr/bin/env bash
#SBATCH --job-name=cxr_grpo
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=14:00:00

# =============================================================================
# CXR GRPO Training with SLURM
# =============================================================================

# Load environment
source ~/.bashrc
conda activate rebecka

# Configuration
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"

# FIXED: Use HOME directory for writable outputs
export OUTPUT_BASE="${HOME}/vlm_experiments"
export RUNS_BASE="${HOME}/runs"

# Your CXR data
data_paths="${HOME}/train_scxr2.jsonl"
image_folders="/cluster/dataset/medinfmk/public_radiology_repo"
model_path="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"
is_reward_customized_from_vlm_module=True
TASK_TYPE="rec"

# Experiment configuration - CHANGE THESE FOR DIFFERENT EXPERIMENTS
# Options: 
# - "accuracy format" (IoU + Format)
# - "iou_only" (IoU Only)
# - "format_only" (Format Only)
# - "combined" (50% IoU + 50% Format)
# - "map_only" (mAP Only)
# - "combined_map" (50% mAP + 50% Format)
REWARD_FUNCS="accuracy format"
EXP_NAME="CXR_iou_format_test"

# Training settings
export DEBUG_MODE="true"
export WANDB_DISABLED=true

# Setup directories in writable locations
mkdir -p ${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME}
mkdir -p ${RUNS_BASE}/${EXP_NAME}/log

# Setup directories in writable locations
mkdir -p ${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME}
mkdir -p ${RUNS_BASE}/${EXP_NAME}/log

export LOG_PATH="${RUNS_BASE}/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

echo "=== CXR GRPO Training Start ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Experiment: ${EXP_NAME}"
echo "Reward functions: ${REWARD_FUNCS}"
echo "Data: ${data_paths}"
echo "Images: ${image_folders}"
echo "Model: ${model_path}"
echo "Logs: ${LOG_PATH}"
echo "=== Training Starting ==="

# Check GPU availability
nvidia-smi

# Change to source directory
cd ${REPO_HOME}/src/open-r1-multimodal

# Launch training
torchrun \
  --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12349 \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 3 \
    --max_completion_length 2048 \
    --reward_funcs ${REWARD_FUNCS} \
    --beta 0.04 \
    --report_to none \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
  2>&1 | tee "${LOG_PATH}"

echo "=== Training Completed ==="
echo "Model saved to: ${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME}"
echo "Logs saved to: ${LOG_PATH}"
echo "Job finished at: $(date)"
