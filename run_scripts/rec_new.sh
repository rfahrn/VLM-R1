#!/usr/bin/env bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "🛠  REPO_HOME = $REPO_HOME"

data_paths="/capstor/scratch/cscs/rfahrni/train_rec_grpo.jsonl:/capstor/scratch/cscs/rfahrni/test_rec_grpo.jsonl"
echo "📑 data_paths = $data_paths"

# ─── Where the images live (MS-CXR PNGs) ───────────────────────────────────────
image_root="/capstor/store/cscs/swissai/a135/RadVLM_project/data/"
image_folders="$image_root:$image_root"
echo "🖼  image_folders = $image_folders"

# ─── Which model you want to fine-tune ─────────────────────────────────────────
model_path="/capstor/store/cscs/swissai/a135/RadVLM_project//models/Qwen2.5-VL-7B-CS"
echo "🤖 model_path = $model_path"

# ─── Experiment name & task settings ──────────────────────────────────────────
export EXP_NAME="Qwen2.5-VL-7B-CS-rec"
TASK_TYPE="rec"
is_reward_customized_from_vlm_module=True

# ─── Prepare logs & checkpoints ────────────────────────────────────────────────
export WANDB_API_KEY="15b5344c70fad59908246ded2a98fdef6a4e9eda"
export WANDB_PROJECT="GRPO"


cd "${REPO_HOME}/src/open-r1-multimodal"
export DEBUG_MODE="true"
mkdir -p "${REPO_HOME}/runs/${EXP_NAME}/log"
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_$(date +%Y-%m-%d-%H-%M-%S).txt"
echo "📝 LOG_PATH = $LOG_PATH"

# ─── Launch distributed training ───────────────────────────────────────────────
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=12349 \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir "${REPO_HOME}/checkpoints/rl/${EXP_NAME}" \
    --resume_from_checkpoint True \
    --model_name_or_path "${model_path}" \
    --data_file_paths "${data_paths}" \
    --image_folders "${image_folders}" \
    --is_reward_customized_from_vlm_module "${is_reward_customized_from_vlm_module}" \
    --task_type "${TASK_TYPE}" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name "${EXP_NAME}" \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \

echo "✅ Training completed for ${EXP_NAME}"
