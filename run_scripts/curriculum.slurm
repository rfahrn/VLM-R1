#!/usr/bin/env bash
#SBATCH --job-name=cxr_grpo_improved
#SBATCH --output=/cluster/home/fahrnr/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/fahrnr/slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=14:00:00

# Project setup
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
cd "${REPO_HOME}/src/open-r1-multimodal"
echo "REPO_HOME: $REPO_HOME"

# Environment
source ~/.bashrc
conda activate rebecka

# Configuration
export EXP_NAME="CXR_anti_reward_hack"
export DEBUG_MODE="true"
data_paths="/cluster/dataset/medinfmk/public_radiology_repo/train_scxr2.jsonl"
image_folders="/cluster/dataset/medinfmk/public_radiology_repo"
model_path="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"

mkdir -p "${REPO_HOME}/checkpoints/rl/${EXP_NAME}"
mkdir -p "${REPO_HOME}/runs/${EXP_NAME}/log"
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%F-%H-%M-%S).txt"

echo "=== Training Configuration ==="
echo "Experiment: $EXP_NAME"
echo "Data: $data_paths"
echo "Model: $model_path"
echo "Log: $LOG_PATH"

# Auto-detect checkpoint
CKPT_DIR="${REPO_HOME}/checkpoints/rl/${EXP_NAME}"
LATEST_CKPT="$(ls -td ${CKPT_DIR}/checkpoint-* 2>/dev/null | head -n1 || true)"

if [[ -n "$LATEST_CKPT" && -d "$LATEST_CKPT" ]]; then
  echo "Resuming from: $LATEST_CKPT"
  RESUME_FLAGS="--resume_from_checkpoint $LATEST_CKPT"
else
  echo "Starting fresh training"
  RESUME_FLAGS=""
fi

# Build the training command
TRAINING_CMD="torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12349 src/open_r1/grpo_jsonl.py"
TRAINING_CMD="$TRAINING_CMD --use_vllm False"
TRAINING_CMD="$TRAINING_CMD --output_dir $CKPT_DIR"
TRAINING_CMD="$TRAINING_CMD $RESUME_FLAGS"
TRAINING_CMD="$TRAINING_CMD --model_name_or_path $model_path"
TRAINING_CMD="$TRAINING_CMD --data_file_paths $data_paths"
TRAINING_CMD="$TRAINING_CMD --image_folders $image_folders"
TRAINING_CMD="$TRAINING_CMD --is_reward_customized_from_vlm_module True"
TRAINING_CMD="$TRAINING_CMD --task_type rec"
TRAINING_CMD="$TRAINING_CMD --per_device_train_batch_size 1"
TRAINING_CMD="$TRAINING_CMD --gradient_accumulation_steps 2"
TRAINING_CMD="$TRAINING_CMD --gradient_checkpointing True"
TRAINING_CMD="$TRAINING_CMD --logging_steps 1"
TRAINING_CMD="$TRAINING_CMD --num_train_epochs 3"
TRAINING_CMD="$TRAINING_CMD --bf16"
TRAINING_CMD="$TRAINING_CMD --attn_implementation flash_attention_2"
TRAINING_CMD="$TRAINING_CMD --run_name $EXP_NAME"
TRAINING_CMD="$TRAINING_CMD --data_seed 42"
TRAINING_CMD="$TRAINING_CMD --save_steps 50"
TRAINING_CMD="$TRAINING_CMD --num_generations 4"
TRAINING_CMD="$TRAINING_CMD --max_completion_length 1024"
TRAINING_CMD="$TRAINING_CMD --reward_funcs curriculum_combined"
TRAINING_CMD="$TRAINING_CMD --beta 0.02"
TRAINING_CMD="$TRAINING_CMD --learning_rate 3e-5"
TRAINING_CMD="$TRAINING_CMD --warmup_steps 100"
TRAINING_CMD="$TRAINING_CMD --use_peft True"
TRAINING_CMD="$TRAINING_CMD --lora_r 16"
TRAINING_CMD="$TRAINING_CMD --lora_alpha 32"
TRAINING_CMD="$TRAINING_CMD --lora_dropout 0.1"
TRAINING_CMD="$TRAINING_CMD --lora_task_type CAUSAL_LM"
TRAINING_CMD="$TRAINING_CMD --freeze_vision_modules True"
TRAINING_CMD="$TRAINING_CMD --weight_decay 0.01"
TRAINING_CMD="$TRAINING_CMD --max_grad_norm 1.0"

echo "=== Starting Training ==="
echo "Command: $TRAINING_CMD"

# Execute the training
eval "$TRAINING_CMD" 2>&1 | tee "${LOG_PATH}"

EXIT_CODE=$?
echo "=== Training Completed ==="
echo "Exit code: $EXIT_CODE"
echo "Log saved to: $LOG_PATH"
