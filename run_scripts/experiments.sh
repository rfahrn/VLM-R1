#!/usr/bin/env bash
# =============================================================================
# Comprehensive CXR GRPO Experiments - All Reward Combinations
# =============================================================================

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
source ~/.bashrc
conda activate rebecka

# Your CXR data
data_paths="${HOME}/train_scxr2.jsonl"
image_folders="/cluster/dataset/medinfmk/public_radiology_repo"
model_path="${REPO_HOME}/Qwen2.5-VL-3B-Instruct"
is_reward_customized_from_vlm_module=True
TASK_TYPE="rec"

cd ${REPO_HOME}/src/open-r1-multimodal
export DEBUG_MODE="true"

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local reward_funcs=$2
    local description=$3
    local port=$4
    
    echo "=== ${description} ==="
    echo "Experiment: ${exp_name}"
    echo "Reward functions: ${reward_funcs}"
    
    mkdir -p ${REPO_HOME}/runs/${exp_name}/log
    export LOG_PATH="${REPO_HOME}/runs/${exp_name}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
    
    echo "Starting training..."
    echo "Logs: ${LOG_PATH}"
    
    torchrun --nproc_per_node="3" \
        --nnodes="1" \
        --node_rank="0" \
        --master_addr="127.0.0.1" \
        --master_port="${port}" \
      src/open_r1/grpo_jsonl.py \
        --use_vllm False \
        --output_dir ${REPO_HOME}/checkpoints/rl/${exp_name} \
        --resume_from_checkpoint True \
        --model_name_or_path $model_path \
        --data_file_paths $data_paths \
        --image_folders $image_folders \
        --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
        --task_type $TASK_TYPE \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 3 \
        --gradient_checkpointing true \
        --logging_steps 1 \
        --num_train_epochs 2 \
        --bf16 \
        --attn_implementation flash_attention_2 \
        --run_name ${exp_name} \
        --data_seed 42 \
        --save_steps 100 \
        --num_generations 3 \
        --max_completion_length 2048 \
        --reward_funcs ${reward_funcs} \
        --beta 0.04 \
        --report_to none \
        --dataset-name this_is_not_used \
        --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
        --learning_rate 1e-5 \
        --use_peft true \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --lora_task_type CAUSAL_LM \
        --freeze_vision_modules true \
      2>&1 | tee "${LOG_PATH}"
    
    echo "Completed: ${description}"
    echo "Model saved to: ${REPO_HOME}/checkpoints/rl/${exp_name}"
    echo ""
}

echo "=== CXR Bbox Detection GRPO Experiments ==="
echo "Choose experiment to run:"
echo ""
echo "=== STANDARD EXPERIMENTS ==="
echo "1.  IoU + Format (Standard)"
echo "2.  IoU Only"
echo "3.  Format Only"
echo ""
echo "=== MAP EXPERIMENTS ==="
echo "4.  mAP + Format"
echo "5.  mAP Only"
echo ""
echo "=== COMBINED EXPERIMENTS ==="
echo "6.  Combined IoU (50% IoU + 50% Format)"
echo "7.  Combined mAP (50% mAP + 50% Format)"
echo ""
echo "=== COMPARISON EXPERIMENTS ==="
echo "8.  IoU vs mAP (Both accuracy methods)"
echo "9.  Multi-reward (IoU + mAP + Format separately)"
echo ""
echo "=== BATCH EXPERIMENTS ==="
echo "10. All Standard (IoU-based experiments)"
echo "11. All MAP (mAP-based experiments)"
echo "12. All Combined (50/50 combinations)"
echo "13. All Individual (single reward functions)"
echo "14. ALL EXPERIMENTS (complete comparison)"
echo ""

read -p "Enter choice (1-14): " choice

case $choice in
    1)
        run_experiment \
            "CXR_iou_format" \
            "accuracy format" \
            "IoU + Format (Standard)" \
            "12349"
        ;;
    2)
        run_experiment \
            "CXR_iou_only" \
            "iou_only" \
            "IoU Only" \
            "12350"
        ;;
    3)
        run_experiment \
            "CXR_format_only" \
            "format_only" \
            "Format Only" \
            "12351"
        ;;
    4)
        run_experiment \
            "CXR_map_format" \
            "map format" \
            "mAP + Format" \
            "12352"
        ;;
    5)
        run_experiment \
            "CXR_map_only" \
            "map_only" \
            "mAP Only" \
            "12353"
        ;;
    6)
        run_experiment \
            "CXR_combined_iou" \
            "combined" \
            "Combined IoU (50% IoU + 50% Format)" \
            "12354"
        ;;
    7)
        run_experiment \
            "CXR_combined_map" \
            "combined_map" \
            "Combined mAP (50% mAP + 50% Format)" \
            "12355"
        ;;
    8)
        run_experiment \
            "CXR_iou_vs_map" \
            "iou map" \
            "IoU vs mAP Comparison" \
            "12356"
        ;;
    9)
        run_experiment \
            "CXR_multi_reward" \
            "iou map format" \
            "Multi-reward (IoU + mAP + Format)" \
            "12357"
        ;;
    10)
        echo "Running all standard IoU-based experiments..."
        run_experiment "CXR_iou_only" "iou_only" "IoU Only" "12350"
        run_experiment "CXR_format_only" "format_only" "Format Only" "12351"
        run_experiment "CXR_iou_format" "accuracy format" "IoU + Format" "12349"
        ;;
    11)
        echo "Running all mAP-based experiments..."
        run_experiment "CXR_map_only" "map_only" "mAP Only" "12353"
        run_experiment "CXR_map_format" "map format" "mAP + Format" "12352"
        ;;
    12)
        echo "Running all combined reward experiments..."
        run_experiment "CXR_combined_iou" "combined" "Combined IoU" "12354"
        run_experiment "CXR_combined_map" "combined_map" "Combined mAP" "12355"
        ;;
    13)
        echo "Running all individual reward experiments..."
        run_experiment "CXR_iou_only" "iou_only" "IoU Only" "12350"
        run_experiment "CXR_map_only" "map_only" "mAP Only" "12353"
        run_experiment "CXR_format_only" "format_only" "Format Only" "12351"
        ;;
    14)
        echo "Running ALL experiments for complete comparison..."
        echo "This will take a very long time!"
        read -p "Are you sure? (y/N): " confirm
        if [[ $confirm == "y" || $confirm == "Y" ]]; then
            # Individual rewards
            run_experiment "CXR_iou_only" "iou_only" "IoU Only" "12350"
            run_experiment "CXR_map_only" "map_only" "mAP Only" "12353"
            run_experiment "CXR_format_only" "format_only" "Format Only" "12351"
            
            # Standard combinations
            run_experiment "CXR_iou_format" "accuracy format" "IoU + Format" "12349"
            run_experiment "CXR_map_format" "map format" "mAP + Format" "12352"
            
            # Combined rewards
            run_experiment "CXR_combined_iou" "combined" "Combined IoU" "12354"
            run_experiment "CXR_combined_map" "combined_map" "Combined mAP" "12355"
            
            # Multi-reward
            run_experiment "CXR_multi_reward" "iou map format" "Multi-reward" "12357"
        else
            echo "Cancelled."
        fi
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

echo "=== Experiment(s) completed! ==="
echo ""
echo "Results saved in: ${REPO_HOME}/checkpoints/rl/"
echo "Logs saved in: ${REPO_HOME}/runs/"
echo ""
echo "To analyze results:"
echo "1. Check training logs for convergence"
echo "2. Compare reward curves between experiments"
echo "3. Look for reward hacking patterns"
echo "4. Evaluate final models on test set"
