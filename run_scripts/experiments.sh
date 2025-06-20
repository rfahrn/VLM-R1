#!/usr/bin/env bash
# =============================================================================
# Experimental GRPO Training Scripts for Different Reward Combinations
# =============================================================================

# Common settings
export TASK_TYPE="rec"
export DEBUG_MODE="true"
export REPO_HOME="/cluster/customapps/medinfmk/fahrnr/VLM-R1"
export DATA_FILES="${HOME}/train_scxr2.jsonl"
export IMAGE_FOLDERS="/cluster/dataset/medinfmk/public_radiology_repo"
export MODEL_PATH="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"
export RUN_ROOT="/cluster/home/fahrnr/runs"
export MAX_STEPS=500

# =============================================================================
# Experiment 1: Combined Reward (50% IoU + 50% Format)
# =============================================================================
run_combined_experiment() {
    export EXP_NAME="grpo_combined_5050"
    export LOG_PATH="${RUN_ROOT}/${EXP_NAME}/logs/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
    mkdir -p "$(dirname "$LOG_PATH")"
    
    echo "=== Running Combined Reward Experiment (50% IoU + 50% Format) ==="
    
    cd "${REPO_HOME}/src/open-r1-multimodal"
    
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
        --dataset_name this_is_not_used \
      2>&1 | tee "${LOG_PATH}"
}

# =============================================================================
# Experiment 2: IoU Only (Check for Format Reward Hacking)
# =============================================================================
run_iou_only_experiment() {
    export EXP_NAME="grpo_iou_only"
    export LOG_PATH="${RUN_ROOT}/${EXP_NAME}/logs/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
    mkdir -p "$(dirname "$LOG_PATH")"
    
    echo "=== Running IoU Only Experiment ==="
    
    cd "${REPO_HOME}/src/open-r1-multimodal"
    
    torchrun \
      --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --master_port=12346 \
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
        --reward_funcs iou_only \
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
        --dataset_name this_is_not_used \
      2>&1 | tee "${LOG_PATH}"
}

# =============================================================================
# Experiment 3: Format Only (Check for IoU Reward Hacking)
# =============================================================================
run_format_only_experiment() {
    export EXP_NAME="grpo_format_only"
    export LOG_PATH="${RUN_ROOT}/${EXP_NAME}/logs/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
    mkdir -p "$(dirname "$LOG_PATH")"
    
    echo "=== Running Format Only Experiment ==="
    
    cd "${REPO_HOME}/src/open-r1-multimodal"
    
    torchrun \
      --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --master_port=12347 \
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
        --reward_funcs format_only \
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
        --dataset_name this_is_not_used \
      2>&1 | tee "${LOG_PATH}"
}

# =============================================================================
# Experiment 4: Different Weight Combinations
# =============================================================================
run_weighted_experiments() {
    # You can modify the combined_reward function to accept weight parameters
    # For now, you would need to create different versions with different weights
    
    # 70% IoU + 30% Format
    export EXP_NAME="grpo_iou70_format30"
    echo "=== Running 70% IoU + 30% Format Experiment ==="
    # (Would require modifying the combined_reward function)
    
    # 30% IoU + 70% Format  
    export EXP_NAME="grpo_iou30_format70"
    echo "=== Running 30% IoU + 70% Format Experiment ==="
    # (Would require modifying the combined_reward function)
}

# =============================================================================
# Multi-Reward Function Experiment (Most Comprehensive)
# =============================================================================
run_multi_reward_experiment() {
    export EXP_NAME="grpo_multi_rewards"
    export LOG_PATH="${RUN_ROOT}/${EXP_NAME}/logs/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
    mkdir -p "$(dirname "$LOG_PATH")"
    
    echo "=== Running Multi-Reward Experiment (IoU + Format separately tracked) ==="
    
    cd "${REPO_HOME}/src/open-r1-multimodal"
    
    # This tracks both IoU and format as separate reward functions
    torchrun \
      --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --master_port=12348 \
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
        --reward_funcs mean_iou format \
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
        --dataset_name this_is_not_used \
      2>&1 | tee "${LOG_PATH}"
}

# =============================================================================
# Run experiments sequentially or choose one
# =============================================================================

echo "Choose which experiment to run:"
echo "1. Combined (50% IoU + 50% Format)"
echo "2. IoU Only" 
echo "3. Format Only"
echo "4. Multi-Reward (IoU + Format tracked separately)"
echo "5. Run all experiments sequentially"

read -p "Enter choice (1-5): " choice

case $choice in
    1) run_combined_experiment ;;
    2) run_iou_only_experiment ;;
    3) run_format_only_experiment ;;
    4) run_multi_reward_experiment ;;
    5) 
        run_iou_only_experiment
        run_format_only_experiment  
        run_combined_experiment
        run_multi_reward_experiment
        ;;
    *) echo "Invalid choice" ;;
esac

echo "Experiment(s) completed!"
