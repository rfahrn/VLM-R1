#!/usr/bin/env bash
#SBATCH --job-name=cxr_grpo_improved
#SBATCH --output=/cluster/home/fahrnr/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/fahrnr/slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=14:00:00

#----------------------------------------
# Project setup
#----------------------------------------
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
cd "${REPO_HOME}/src/open-r1-multimodal"
echo "REPO_HOME: $REPO_HOME"

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

# FIXED: Use the correct paths where your data and checkpoints actually are
data_paths="${HOME}/train_scxr2.jsonl"  # This is where your data file is
image_folders="/cluster/dataset/medinfmk/public_radiology_repo"
model_path="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"

# FIXED: Use the correct checkpoint directory path
export OUTPUT_BASE="${HOME}/vlm_experiments"
export RUNS_BASE="${HOME}/runs"
CKPT_DIR="${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME}"

mkdir -p "${CKPT_DIR}"
mkdir -p "${RUNS_BASE}/${EXP_NAME}/log"
export LOG_PATH="${RUNS_BASE}/${EXP_NAME}/log/debug_log.$(date +%F-%H-%M-%S).txt"

echo "=== Configuration ==="
echo "Experiment:    $EXP_NAME"
echo "Data paths:    $data_paths"
echo "Image folders: $image_folders"
echo "Model path:    $model_path"
echo "Checkpoint dir: $CKPT_DIR"
echo "Log file:      $LOG_PATH"

# Check if data file exists
if [[ ! -f "$data_paths" ]]; then
    echo "❌ ERROR: Data file not found: $data_paths"
    ls -la "${HOME}/"*jsonl 2>/dev/null || echo "No .jsonl files found in HOME"
    exit 1
fi

# Check if model path exists
if [[ ! -d "$model_path" ]]; then
    echo "❌ ERROR: Model path not found: $model_path"
    exit 1
fi

echo "✅ Data file exists: $data_paths"
echo "✅ Model path exists: $model_path"

#----------------------------------------
# Auto-detect last checkpoint
#----------------------------------------
echo "Looking for checkpoints in: $CKPT_DIR"
LATEST_CKPT="$(ls -td ${CKPT_DIR}/checkpoint-* 2>/dev/null | head -n1 || true)"

if [[ -n "$LATEST_CKPT" && -d "$LATEST_CKPT" ]]; then
    echo "✅ Found existing checkpoint: $LATEST_CKPT"
    # Use the checkpoint directory as the model path for resuming
    model_path="$LATEST_CKPT"
    RESUME_FLAGS="--resume_from_checkpoint True"
else
    echo "ℹ️  No checkpoint found; starting fresh from: $model_path"
    RESUME_FLAGS=""
fi

#----------------------------------------
# Launch training
#----------------------------------------
echo "=== Starting Training ==="
echo "Using model: $model_path"
echo "Reward function: curriculum_combined"
echo "Command starting..."

torchrun \
  --nproc_per_node=3 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=12349 \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir "${CKPT_DIR}" \
    $RESUME_FLAGS \
    --model_name_or_path "$model_path" \
    --data_file_paths "$data_paths" \
    --image_folders "$image_folders" \
    --is_reward_customized_from_vlm_module True \
    --task_type "rec" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name "${EXP_NAME}" \
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
  2>&1 | tee "${LOG_PATH}"

EXIT_CODE=$?
echo "=== Training Completed ==="
echo "Exit code: $EXIT_CODE"
echo "Checkpoints saved to: $CKPT_DIR"
echo "Log saved to: $LOG_PATH"

# List the checkpoints created
echo "=== Final Checkpoint Status ==="
ls -la "${CKPT_DIR}/" 2>/dev/null || echo "No checkpoints found"
