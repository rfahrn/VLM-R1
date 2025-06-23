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

data_paths="/cluster/dataset/medinfmk/public_radiology_repo/train_scxr2.jsonl"
image_folders="/cluster/dataset/medinfmk/public_radiology_repo"
model_path="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"

mkdir -p "${REPO_HOME}/checkpoints/rl/${EXP_NAME}"
mkdir -p "${REPO_HOME}/runs/${EXP_NAME}/log"

export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%F-%H-%M-%S).txt"

echo "Experiment:    $EXP_NAME"
echo "Data paths:    $data_paths"
echo "Image folders: $image_folders"
echo "Model path:    $model_path"
echo "Log file:      $LOG_PATH"

#----------------------------------------
# Auto-detect last checkpoint (if any)
#----------------------------------------
CKPT_DIR="${REPO_HOME}/checkpoints/rl/${EXP_NAME}"
LATEST_CKPT="$(ls -td ${CKPT_DIR}/checkpoint-* 2>/dev/null | head -n1 || true)"

if [[ -n "$LATEST_CKPT" && -d "$LATEST_CKPT" ]]; then
  echo "Found existing checkpoint: $LATEST_CKPT"
  RESUME_FLAGS="--resume_from_checkpoint $LATEST_CKPT"
else
  echo "No checkpoint found; starting fresh"
  RESUME_FLAGS=""
fi

#----------------------------------------
# Launch training
#----------------------------------------
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

echo "Training completed for ${EXP_NAME}"

