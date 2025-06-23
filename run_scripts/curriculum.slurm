#!/usr/bin/env bash
#SBATCH --job-name=cxr_grpo_improved
#SBATCH --output=/cluster/home/fahrnr/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/fahrnr/slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=14:00:00

cd /cluster/customapps/medinfmk/fahrnr/VLM-R1

# Load conda environment
source ~/.bashrc
conda activate rebecka

# Base paths
export REPO_HOME="/cluster/customapps/medinfmk/fahrnr/VLM-R1"
export OUTPUT_BASE="${HOME}/vlm_experiments"
export RUNS_BASE="${HOME}/runs"

# Data and model
data_paths="${HOME}/train_scxr2.jsonl"
image_folders="/cluster/dataset/medinfmk/public_radiology_repo"
model_path="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"

# Choose exactly one combined reward
REWARD_FUNCS="curriculum_combined"   # options: combined, combined_map, curriculum_combined, threshold_gated
EXP_NAME="CXR_anti_reward_hack"

# Create output dirs
mkdir -p "${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME}"
mkdir -p "${RUNS_BASE}/${EXP_NAME}/log"

export DEBUG_MODE="true"
export LOG_PATH="${RUNS_BASE}/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

echo "=== Anti-Reward-Hacking GRPO Training ==="
echo "Reward function: ${REWARD_FUNCS}"
echo "Strategy: Curriculum learning with progressive weighting"

torchrun \
  --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12349 \
  src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir "${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME}" \
    --model_name_or_path "$model_path" \
    --data_file_paths "$data_paths" \
    --image_folders "$image_folders" \
    --is_reward_customized_from_vlm_module True \
    --task_type "rec" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --bf16 \
    --run_name "${EXP_NAME}" \
    --max_steps 1000 \
    --save_steps 50 \
    --num_generations 4 \
    --max_completion_length 1024 \
    --reward_funcs "${REWARD_FUNCS}" \
    \
    # Anti-reward-hacking hyperparams
    --beta 0.02 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    \
    # LoRA & PEFT
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    \
    # Freeze vision, checkpointing, regularization
    --freeze_vision_modules true \
    --gradient_checkpointing true \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
  2>&1 | tee "${LOG_PATH}"

echo "=== Training Completed ==="

