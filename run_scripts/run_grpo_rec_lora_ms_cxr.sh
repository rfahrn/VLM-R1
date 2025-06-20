#!/usr/bin/env bash
# =============================================================================
# LoRA-based GRPO Rec Training Script for VLM-R1 (Single-GPU)
# =============================================================================
#SBATCH --job-name=grpo_lora
#SBATCH --output=%x_%j_grpo_rewardcombined.out
#SBATCH --error=%x_%j_grpo_rewardcombined.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=14:00:00

# Load environment
source ~/.bashrc
conda activate rebecka


export TASK_TYPE="rec"
export DEBUG_MODE="true"

# wandb (disabled)
export WANDB_PROJECT="vlm-r1"
export WANDB_NAME="Qwen2.5-VL-3B-Instruct-rec-lora"
export WANDB_API_KEY=15b5344c70fad59908246ded2a98fdef6a4e9eda
export WANDB_MODE=offline
export WANDB_DISABLED=true

# Repository and data paths
export REPO_HOME="/cluster/customapps/medinfmk/fahrnr/VLM-R1"
export DATA_FILES="${HOME}/train_scxr2.jsonl"
export IMAGE_FOLDERS="/cluster/dataset/medinfmk/public_radiology_repo"
export MODEL_PATH="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"

# Experiment naming and logs
export EXP_NAME="grpo_lora_combined_gpu"
# Set writable output path
export RUN_ROOT="/cluster/home/fahrnr/runs"
export LOG_PATH="${RUN_ROOT}/${EXP_NAME}/logs/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
mkdir -p "$(dirname "$LOG_PATH")"
echo "Writing logs to: $LOG_PATH"
echo "Launching with $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPUs"



echo "Starting GRPO LoRA run: ${EXP_NAME}"
echo "Data files: ${DATA_FILES}"
echo "Image folders: ${IMAGE_FOLDERS}"
echo "Model path: ${MODEL_PATH}"
echo "Logs: ${LOG_PATH}"

# Move to training directory
cd "${REPO_HOME}/src/open-r1-multimodal"

export MAX_STEPS=500


# Launch training
torchrun \
  --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --dataset_name this_is_not_used \
    --data_file_paths "${DATA_FILES}" \
    --image_folders "${IMAGE_FOLDERS}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${RUN_ROOT}/${EXP_NAME}/output" \
    --resume_from_checkpoint True \
    --is_reward_customized_from_vlm_module True \
    --task_type rec \
    --reward_funcs combined \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --gradient_checkpointing True \
    --num_generations 3 \
    --max_completion_length 2048 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --deepspeed local_scripts/zero2.json \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules True \
    --report_to none \
    --dataset_name this_is_not_used
  2>&1 | tee "${LOG_PATH}"

echo "GRPO LoRA training completed for ${EXP_NAME} (Single GPU)"
