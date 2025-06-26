#!/bin/bash
#SBATCH --job-name=rec_grpo
#SBATCH --output=/cluster/home/fahrnr/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/fahrnr/slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=14:00:00
#SBATCH --chdir=/cluster/customapps/medinfmk/fahrnr/VLM-R1

# or, if you don't use --chdir, add explicitly:
cd /cluster/customapps/medinfmk/fahrnr/VLM-R1

# Ensure slurm_logs directory exists
mkdir -p /cluster/home/fahrnr/slurm_logs

# Load environment
source ~/.bashrc
conda activate rebecka

# Configuration
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="/cluster/customapps/medinfmk/fahrnr/VLM-R1"
echo "REPO_HOME: $REPO_HOME"

# Use HOME directory for writable outputs
export OUTPUT_BASE="${HOME}/vlm_experiments"
DATA_DIR="/cluster/dataset/medinfmk/public_radiology_repo"
TRAIN_JSON="$HOME/rec_jsonl/train_rec_grpo.jsonl"
VAL_JSON="$HOME/rec_jsonl/test_rec_grpo.jsonl"
data_paths="${TRAIN_JSON}:${VAL_JSON}"
# image_folders="/cluster/dataset/medinfmk/public_radiology_repo"
image_folders="${DATA_DIR}:${DATA_DIR}"
model_path="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"

is_reward_customized_from_vlm_module=True
TASK_TYPE="rec"
export EXP_NAME="Qwen2.5-VL-3B-REC-grpo"
TASK_TYPE="rec"

echo "data_paths:    $data_paths"
echo "image_folders: $image_folders"

#### 3) Logging dirs
cd $REPO_HOME/src/open-r1-multimodal
export DEBUG_MODE="true"
mkdir -p $HOME/runs/${EXP_NAME}/log
mkdir -p $HOME/checkpoints/rl/${EXP_NAME}
export LOG_PATH="$HOME/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

#### 4) Launch training
torchrun \
  --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12349 \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir      ${OUTPUT_BASE}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path              $model_path \
    --data_file_paths                 $data_paths \
    --image_folders                   $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type                       $TASK_TYPE \
    --per_device_train_batch_size     8 \
    --gradient_accumulation_steps     2 \
    --gradient_checkpointing          true \
    --logging_steps                   1 \
    --num_train_epochs                2 \
    --bf16                            \
    --attn_implementation             flash_attention_2 \
    --run_name                        $EXP_NAME \
    --data_seed                       42 \
    --save_steps                      100 \
    --num_generations                 8 \
    --max_completion_length           2048 \
    --reward_funcs                    iou map format partial_iou iou_fbeta distance_based shaped_reward \
    --beta                            0.04 \
    --report_to                       none \
    --dataset-name                    this_is_not_used \
    --deepspeed                       $REPO_HOME/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate                   7e-5 \
    --use_peft                        true \
    --lora_r                          64 \
    --lora_alpha                      128 \
    --lora_dropout                    0.05 \
    --lora_task_type                  CAUSAL_LM \
    --freeze_vision_modules           true

echo "✅ Training completed for ${EXP_NAME}"


















