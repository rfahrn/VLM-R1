#!/usr/bin/env bash
#SBATCH --job-name=cxr_grpo_improved
#SBATCH --output=/cluster/home/fahrnr/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/fahrnr/slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=14:00:00

set -euo pipefail

#----------------------------------------
# Project setup
#----------------------------------------
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="$PROJECT_ROOT"
cd "$REPO_HOME/src/open-r1-multimodal"

#----------------------------------------
# Environment
#----------------------------------------
source ~/.bashrc
conda activate rebecka

#----------------------------------------
# Paths & experiment name
#----------------------------------------
export EXP_NAME="CXR_anti_reward_hack"
export DEBUG_MODE="true"

DATA_FILE="${HOME}/train_scxr2.jsonl"
IMAGE_FOLDERS="/cluster/dataset/medinfmk/public_radiology_repo"
BASE_MODEL="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"

export OUTPUT_BASE="${HOME}/vlm_experiments"
export RUNS_BASE="${HOME}/runs"
CKPT_DIR="${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME}"
LOG_DIR="${RUNS_BASE}/${EXP_NAME}/log"
LOG_PATH="${LOG_DIR}/debug_log.$(date +%F-%H-%M-%S).txt"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

echo "=== Configuration ==="
echo "Experiment:    $EXP_NAME"
echo "Data file:     $DATA_FILE"
echo "Image folders: $IMAGE_FOLDERS"
echo "Base model:    $BASE_MODEL"
echo "Checkpoint dir:$CKPT_DIR"
echo "Log file:      $LOG_PATH"

# sanity checks
[[ -f "$DATA_FILE" ]]      || { echo "❌ Missing data: $DATA_FILE"; exit 1; }
[[ -d "$BASE_MODEL" ]]     || { echo "❌ Missing model dir: $BASE_MODEL"; exit 1; }

#----------------------------------------
# Resume logic
#----------------------------------------
echo "Looking for checkpoints in: $CKPT_DIR"
LATEST="$(ls -td ${CKPT_DIR}/checkpoint-* 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST" ]]; then
  echo "✅ Resuming from $LATEST"
  MODEL_TO_LOAD="$LATEST"
  RESUME_FLAG="--resume_from_checkpoint True"
else
  echo "ℹ️  No checkpoint found, starting from base model"
  MODEL_TO_LOAD="$BASE_MODEL"
  RESUME_FLAG=""
fi

#----------------------------------------
# Launch
#----------------------------------------
echo "=== Starting Training ==="
torchrun \
  --nproc_per_node=3 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12349 \
  src/open_r1/grpo_jsonl.py \
    --dataset-name this_is_not_used \
    --use_vllm False \
    --output_dir "$CKPT_DIR" \
    $RESUME_FLAG \
    --model_name_or_path "$MODEL_TO_LOAD" \
    --data_file_paths "$DATA_FILE" \
    --image_folders "$IMAGE_FOLDERS" \
    --is_reward_customized_from_vlm_module True \
    --task_type rec \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name "$EXP_NAME" \
    --data_seed 42 \
    --save_steps 50 \
    --num_generations 4 \
    --max_completion_length 1024 \
    --reward_funcs curriculum_combined \
    --beta 0.02 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    --use_peft True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules True \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
  2>&1 | tee "$LOG_PATH"

EXIT_CODE=$?
echo "=== Training Completed (exit $EXIT_CODE) ==="
echo "Checkpoints in: $CKPT_DIR"
echo "Logs in:        $LOG_PATH"
echo "=== Final Checkpoints ==="
ls -1 "$CKPT_DIR"
