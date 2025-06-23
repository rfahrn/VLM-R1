#!/usr/bin/env python3
"""
Quick Test for Specific Reward Functions

Use this for rapid testing of individual reward functions during development.
"""

import sys
import os

# Add path to modules
sys.path.append('/cluster/customapps/medinfmk/fahrnr/VLM-R1/src/open-r1-multimodal/src')

def quick_test_reward_function(func_name, test_detailed=False):
    """Quick test of a specific reward function."""
    
    # Setup test environment
    os.environ["DEBUG_MODE"] = "true"
    os.environ["LOG_PATH"] = "/tmp/quick_test_debug.txt"
    
    try:
        from open_r1.vlm_modules.qwen_module import Qwen2VLModule
        print(f"✅ Imported Qwen2VLModule")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Get the reward function
    try:
        reward_func = Qwen2VLModule.select_reward_func(func_name, "rec")
        print(f"✅ Got reward function: {func_name}")
    except Exception as e:
        print(f"❌ Failed to get reward function '{func_name}': {e}")
        return False
    
    # Test cases from your original setup
    test_cases = [
        # Case 1: Perfect format and match
        "<think>reasoning</think><answer>It is displayed at [0.19, 0.5, 0.48, 0.84] and [0.63, 0.48, 0.98, 0.87] in the radiograph.</answer>",
        
        # Case 2: Model generates text without coordinates  
        "<think>reasoning</think><answer>The patchy bibasilar opacities can be seen in the lower regions of both lungs.</answer>",
        
        # Case 3: Model generates coordinates in different format
        "<think>reasoning</think><answer>The area is located at coordinates (0.19, 0.5, 0.48, 0.84) and (0.63, 0.48, 0.98, 0.87).</answer>",
        
        # Case 4: Model generates partial coordinates
        "<think>reasoning</think><answer>It is at [0.19, 0.5] in the image.</answer>",
        
        # Case 5: Close but not exact match
        "<think>reasoning</think><answer>[0.20, 0.51, 0.47, 0.83] and [0.64, 0.49, 0.97, 0.86]</answer>",
    ]
    
    ground_truth = "It is displayed at [0.19, 0.5, 0.48, 0.84] and [0.63, 0.48, 0.98, 0.87] in the radiograph."
    
    print(f"\n🧪 Testing {func_name} with {len(test_cases)} cases")
    print(f"Ground truth: {ground_truth}")
    
    all_rewards = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        if test_detailed:
            print(f"Input: {test_case[:100]}{'...' if len(test_case) > 100 else ''}")
        
        try:
            # Prepare inputs as GRPO would
            completions = [[{"content": test_case}]]
            solution = [ground_truth]
            
            # Add kwargs for advanced functions
            kwargs = {
                'current_step': 150,  # For curriculum learning
                'consecutive_perfect_format': 3,  # For momentum
                'format_mask_prob': 0.2,  # For adversarial
            }
            
            # Call reward function
            rewards = reward_func(completions, solution, **kwargs)
            reward = rewards[0] if rewards else 0.0
            all_rewards.append(reward)
            
            # Extract details
            pred_answer = Qwen2VLModule.extract_answer_content(test_case)
            gt_answer = Qwen2VLModule.extract_answer_content(ground_truth)
            
            pred_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(pred_answer)
            gt_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(gt_answer)
            
            # Check format
            import re
            has_format = bool(re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", test_case, re.DOTALL))
            
            print(f"Reward: {reward:.3f}")
            if test_detailed:
                print(f"Extracted answer: {pred_answer}")
                print(f"Pred bboxes: {pred_bboxes}")
                print(f"GT bboxes: {gt_bboxes}")
                print(f"Has format: {has_format}")
            else:
                print(f"Pred bboxes: {len(pred_bboxes)}, GT bboxes: {len(gt_bboxes)}, Format: {has_format}")
            
            # Status
            if reward > 0.5:
                status = "✅ GOOD"
            elif reward > 0.1:
                status = "⚠️  OKAY"
            elif reward > 0:
                status = "🔶 LOW"
            else:
                status = "❌ ZERO"
            
            print(f"Status: {status}")
            
        except Exception as e:
            print(f"❌ Error in test case {i}: {e}")
            import traceback
            traceback.print_exc()
            all_rewards.append(-1.0)  # Error indicator
    
    # Summary
    print(f"\n📊 SUMMARY for {func_name}")
    print("="*50)
    
    valid_rewards = [r for r in all_rewards if r >= 0]
    error_count = sum(1 for r in all_rewards if r < 0)
    
    if valid_rewards:
        print(f"Valid tests: {len(valid_rewards)}/{len(all_rewards)}")
        print(f"Average reward: {sum(valid_rewards)/len(valid_rewards):.3f}")
        print(f"Max reward: {max(valid_rewards):.3f}")
        print(f"Min reward: {min(valid_rewards):.3f}")
        print(f"Non-zero rewards: {sum(1 for r in valid_rewards if r > 0)}/{len(valid_rewards)}")
        
        # Specific insights
        if func_name in ["combined", "combined_map"]:
            print(f"ℹ️  Combined function - expect rewards around 0.5 for format-only cases")
        elif func_name in ["curriculum_combined"]:
            print(f"ℹ️  Curriculum function - weights change based on training step")
        elif func_name in ["hierarchical"]:
            print(f"ℹ️  Hierarchical function - format is prerequisite for IoU bonus")
        elif func_name in ["threshold_gated"]:
            print(f"ℹ️  Threshold gated - requires minimum IoU for format credit")
    
    if error_count > 0:
        print(f"❌ Errors: {error_count}")
        return False
    else:
        print(f"✅ No errors - function works correctly!")
        return True

def test_all_basic_functions():
    """Test all basic reward functions quickly."""
    basic_functions = [
        "partial_iou",
        "iou", 
        "map",
        "format_only",
        "combined",
        "combined_map"
    ]
    
    print("🚀 Quick test of all basic functions")
    results = {}
    
    for func in basic_functions:
        print(f"\n{'='*60}")
        success = quick_test_reward_function(func, test_detailed=False)
        results[func] = success
    
    print(f"\n🎉 FINAL RESULTS")
    print("="*50)
    for func, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {func}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✅ All basic functions work! Ready for training!")
    else:
        print(f"\n❌ Some functions failed. Fix before training.")
    
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific function
        func_name = sys.argv[1]
        detailed = len(sys.argv) > 2 and sys.argv[2] == "--detailed"
        success = quick_test_reward_function(func_name, test_detailed=detailed)
        sys.exit(0 if success else 1)
    else:
        # Test all basic functions
        success = test_all_basic_functions()
        sys.exit(0 if success else 1)

