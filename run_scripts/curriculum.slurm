#!/usr/bin/env bash
#SBATCH --job-name=cxr_single_gpu
#SBATCH --output=/cluster/home/fahrnr/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/fahrnr/slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
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
echo "PWD: $(pwd)"

#----------------------------------------
# Environment
#----------------------------------------
source ~/.bashrc
conda activate rebecka

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

#----------------------------------------
# Paths & experiment name
#----------------------------------------
export EXP_NAME="CXR_single_gpu_curriculum"
export DEBUG_MODE="true"

# Use correct paths
data_paths="${HOME}/train_scxr2.jsonl"
image_folders="/cluster/dataset/medinfmk/public_radiology_repo"
model_path="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"

# Output directories
export OUTPUT_BASE="${HOME}/vlm_experiments"
export RUNS_BASE="${HOME}/runs"
CKPT_DIR="${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME}"

mkdir -p "${CKPT_DIR}"
mkdir -p "${RUNS_BASE}/${EXP_NAME}/log"
export LOG_PATH="${RUNS_BASE}/${EXP_NAME}/log/debug_log.$(date +%F-%H-%M-%S).txt"

echo "=== Configuration ==="
echo "Experiment: $EXP_NAME"
echo "Data: $data_paths"
echo "Images: $image_folders"
echo "Model: $model_path"
echo "Output: $CKPT_DIR"
echo "Log: $LOG_PATH"

# Validate paths
if [[ ! -f "$data_paths" ]]; then
    echo "❌ ERROR: Data file not found: $data_paths"
    exit 1
fi

if [[ ! -d "$model_path" ]]; then
    echo "❌ ERROR: Model path not found: $model_path"
    exit 1
fi

if [[ ! -d "$image_folders" ]]; then
    echo "❌ ERROR: Image folder not found: $image_folders"
    exit 1
fi

echo "✅ All paths validated"

# Check for existing checkpoints from previous experiments
OLD_CKPT_DIR="${OUTPUT_BASE}/checkpoints/rl/CXR_anti_reward_hack"
if [[ -d "$OLD_CKPT_DIR" ]]; then
    LATEST_CKPT="$(ls -td ${OLD_CKPT_DIR}/checkpoint-* 2>/dev/null | head -n1 || true)"
    if [[ -n "$LATEST_CKPT" && -d "$LATEST_CKPT" ]]; then
        echo "✅ Found previous checkpoint: $LATEST_CKPT"
        echo "Using it as starting point"
        model_path="$LATEST_CKPT"
        RESUME_FLAGS="--resume_from_checkpoint True"
    else
        echo "ℹ️  No previous checkpoints found, starting fresh"
        RESUME_FLAGS=""
    fi
else
    echo "ℹ️  No previous experiment found, starting fresh"
    RESUME_FLAGS=""
fi

#----------------------------------------
# Test the reward function first
#----------------------------------------
echo "=== Testing Reward Function ==="
python -c "
import sys
sys.path.append('src')
try:
    from open_r1.vlm_modules.qwen_module import Qwen2VLModule
    func = Qwen2VLModule.select_reward_func('curriculum_combined', 'rec')
    print('✅ curriculum_combined reward function works!')
except Exception as e:
    print(f'❌ Reward function error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Reward function test failed!"
    exit 1
fi

#----------------------------------------
# Launch training (SINGLE GPU - NO TORCHRUN)
#----------------------------------------
echo "=== Starting Single GPU Training ==="
echo "Using model: $model_path"
echo "Reward: curriculum_combined (0.854 avg reward in tests)"

# Use direct python instead of torchrun for single GPU
python src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir "${CKPT_DIR}" \
    $RESUME_FLAGS \
    --model_name_or_path "$model_path" \
    --data_file_paths "$data_paths" \
    --image_folders "$image_folders" \
    --is_reward_customized_from_vlm_module True \
    --task_type "rec" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 6 \
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
    --dataset_name this_is_not_used \
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

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training successful!"
    echo "Checkpoints: $CKPT_DIR"
    ls -la "${CKPT_DIR}/"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check log: $LOG_PATH"
fi
