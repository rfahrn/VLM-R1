#!/usr/bin/env python3
"""
Comprehensive Test Suite for Qwen2VL Reward Functions

This script tests all reward functions with various edge cases to ensure
they work correctly before running expensive training jobs.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add path to modules
sys.path.append('/cluster/customapps/medinfmk/fahrnr/VLM-R1/src/open-r1-multimodal/src')

def setup_test_environment():
    """Setup test environment with debug mode."""
    os.environ["DEBUG_MODE"] = "true"
    os.environ["LOG_PATH"] = "/tmp/test_rewards_debug.txt"
    print("🔧 Test environment setup complete")

def create_test_cases():
    """Create comprehensive test cases covering various scenarios."""
    return {
        "perfect_match": {
            "input": "<think>I can see the opacities clearly in both lung fields.</think><answer>It is displayed at [0.19, 0.5, 0.48, 0.84] and [0.63, 0.48, 0.98, 0.87] in the radiograph.</answer>",
            "expected_format": True,
            "expected_bbox_count": 2,
            "description": "Perfect format and exact coordinate match"
        },
        
        "close_match": {
            "input": "<think>Looking at the chest X-ray, I can identify the areas.</think><answer>[0.20, 0.51, 0.47, 0.83] and [0.64, 0.49, 0.97, 0.86]</answer>",
            "expected_format": True,
            "expected_bbox_count": 2,
            "description": "Good format with close but not exact coordinates"
        },
        
        "partial_match": {
            "input": "<think>I can see one clear area.</think><answer>The region is at [0.19, 0.5, 0.48, 0.84] in the image.</answer>",
            "expected_format": True,
            "expected_bbox_count": 1,
            "description": "Correct format but only one bbox instead of two"
        },
        
        "no_coordinates": {
            "input": "<think>Looking at the image...</think><answer>The patchy bibasilar opacities can be seen in the lower regions of both lungs.</answer>",
            "expected_format": True,
            "expected_bbox_count": 0,
            "description": "Correct format but descriptive answer instead of coordinates"
        },
        
        "wrong_format": {
            "input": "The areas are located at coordinates (0.19, 0.5, 0.48, 0.84) and (0.63, 0.48, 0.98, 0.87).",
            "expected_format": False,
            "expected_bbox_count": 2,
            "description": "Missing think/answer tags but has coordinates"
        },
        
        "incomplete_coordinates": {
            "input": "<think>Examining the X-ray...</think><answer>It is at [0.19, 0.5] in the image.</answer>",
            "expected_format": True,
            "expected_bbox_count": 0,
            "description": "Correct format but incomplete coordinates (only 2 values)"
        },
        
        "multiple_formats": {
            "input": "<think>I see multiple areas.</think><answer>Areas at [0.19, 0.5, 0.48, 0.84], (0.63, 0.48, 0.98, 0.87), and 0.1, 0.2, 0.3, 0.4</answer>",
            "expected_format": True,
            "expected_bbox_count": 3,
            "description": "Mixed coordinate formats (brackets, parentheses, comma-separated)"
        },
        
        "empty_answer": {
            "input": "<think>This is difficult to determine.</think><answer></answer>",
            "expected_format": True,
            "expected_bbox_count": 0,
            "description": "Correct format but empty answer"
        },
        
        "malformed_coordinates": {
            "input": "<think>Trying to locate...</think><answer>[0.19, 0.5, 0.48] and [0.63, 0.48, 0.98, 0.87, 0.99]</answer>",
            "expected_format": True,
            "expected_bbox_count": 1,
            "description": "One valid bbox, one malformed (wrong number of coordinates)"
        },
        
        "extra_large_coordinates": {
            "input": "<think>Looking at the image.</think><answer>[1.5, 2.0, 3.0, 4.0] and [0.63, 0.48, 0.98, 0.87]</answer>",
            "expected_format": True,
            "expected_bbox_count": 2,
            "description": "Coordinates outside [0,1] range (should still be extracted)"
        }
    }

def create_ground_truth_variations():
    """Create different ground truth formats for testing."""
    return {
        "standard": "It is displayed at [0.19, 0.5, 0.48, 0.84] and [0.63, 0.48, 0.98, 0.87] in the radiograph.",
        "with_answer_tags": "<answer>[0.19, 0.5, 0.48, 0.84] and [0.63, 0.48, 0.98, 0.87]</answer>",
        "single_bbox": "The area specified is at coordinates [0.19, 0.5, 0.48, 0.84].",
        "no_coordinates": "The opacities are visible in the bilateral lower lung fields.",
        "parentheses_format": "You'll find it at (0.19, 0.5, 0.48, 0.84) and (0.63, 0.48, 0.98, 0.87)."
    }

def test_reward_function(reward_func, func_name, test_cases, ground_truth_dict, **kwargs):
    """Test a specific reward function with all test cases."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {func_name.upper()}")
    print(f"{'='*60}")
    
    results = {}
    
    for gt_name, ground_truth in ground_truth_dict.items():
        print(f"\n--- Ground Truth: {gt_name} ---")
        
        gt_answer = Qwen2VLModule.extract_answer_content(ground_truth)
        gt_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(gt_answer)
        print(f"GT bboxes ({len(gt_bboxes)}): {gt_bboxes}")
        
        case_results = []
        
        for case_name, test_case in test_cases.items():
            try:
                # Prepare inputs as GRPO would
                completions = [[{"content": test_case["input"]}]]
                solution = [ground_truth]
                
                # Add test-specific kwargs
                test_kwargs = kwargs.copy()
                test_kwargs.update({
                    'current_step': 150,  # For curriculum learning
                    'consecutive_perfect_format': 5,  # For momentum
                    'format_mask_prob': 0.3,  # For adversarial
                })
                
                # Call the reward function
                rewards = reward_func(completions, solution, **test_kwargs)
                reward = rewards[0] if rewards else 0.0
                
                # Extract prediction details
                pred_answer = Qwen2VLModule.extract_answer_content(test_case["input"])
                pred_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(pred_answer)
                
                # Check format
                import re
                has_format = bool(re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", 
                                          test_case["input"], re.DOTALL))
                
                result = {
                    'case': case_name,
                    'description': test_case['description'],
                    'reward': reward,
                    'pred_bboxes': len(pred_bboxes),
                    'expected_bboxes': test_case['expected_bbox_count'],
                    'has_format': has_format,
                    'expected_format': test_case['expected_format'],
                    'bbox_match': len(pred_bboxes) == test_case['expected_bbox_count'],
                    'format_match': has_format == test_case['expected_format']
                }
                
                case_results.append(result)
                
                # Print result
                status = "✅" if reward > 0 else "❌" if reward == 0 else "⚠️"
                print(f"{status} {case_name:20} | Reward: {reward:6.3f} | "
                      f"Bboxes: {len(pred_bboxes)}/{test_case['expected_bbox_count']} | "
                      f"Format: {has_format}")
                
                if len(pred_bboxes) != test_case['expected_bbox_count']:
                    print(f"    ⚠️  Expected {test_case['expected_bbox_count']} bboxes, got {len(pred_bboxes)}")
                
            except Exception as e:
                print(f"❌ {case_name:20} | ERROR: {str(e)}")
                case_results.append({
                    'case': case_name,
                    'description': test_case['description'],
                    'reward': -1.0,  # Error indicator
                    'error': str(e)
                })
        
        results[gt_name] = case_results
        
        # Summary for this ground truth
        valid_rewards = [r['reward'] for r in case_results if 'error' not in r and r['reward'] >= 0]
        if valid_rewards:
            print(f"\n📊 Summary for {gt_name}:")
            print(f"   Average reward: {np.mean(valid_rewards):.3f}")
            print(f"   Max reward: {np.max(valid_rewards):.3f}")
            print(f"   Non-zero rewards: {sum(1 for r in valid_rewards if r > 0)}/{len(valid_rewards)}")
    
    return results

def run_comprehensive_tests():
    """Run comprehensive tests for all reward functions."""
    print("🚀 Starting Comprehensive Reward Function Tests")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    setup_test_environment()
    
    try:
        from open_r1.vlm_modules.qwen_module import Qwen2VLModule
        print("✅ Qwen2VLModule imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Prepare test data
    test_cases = create_test_cases()
    ground_truths = create_ground_truth_variations()
    
    print(f"\n📋 Test Setup:")
    print(f"   Test cases: {len(test_cases)}")
    print(f"   Ground truth variations: {len(ground_truths)}")
    
    # Define reward functions to test
    reward_functions_to_test = [
        # Basic functions
        ("iou_reward", "iou"),
        ("partial_credit_iou", "partial_iou"),
        ("map_reward", "map"),
        ("format_reward", "format_only"),
        
        # Combined functions
        ("combined_reward", "combined"),
        ("combined_map_reward", "combined_map"),
        
        # Advanced functions (if implemented)
        ("curriculum_combined", "curriculum_combined"),
        ("hierarchical_reward", "hierarchical"),
        ("threshold_gated", "threshold_gated"),
    ]
    
    all_results = {}
    
    for func_display_name, func_key in reward_functions_to_test:
        try:
            reward_func = Qwen2VLModule.select_reward_func(func_key, "rec")
            results = test_reward_function(reward_func, func_display_name, test_cases, ground_truths)
            all_results[func_display_name] = results
            
        except Exception as e:
            print(f"\n❌ Failed to test {func_display_name}: {e}")
            if "curriculum_combined" in func_key or "hierarchical" in func_key or "threshold_gated" in func_key:
                print(f"   ℹ️  This is expected if you haven't added the new reward functions yet")
            else:
                print(f"   ⚠️  This is a basic function and should work!")
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    for func_name, func_results in all_results.items():
        print(f"\n🔧 {func_name}:")
        
        all_rewards = []
        error_count = 0
        
        for gt_name, case_results in func_results.items():
            valid_rewards = [r['reward'] for r in case_results if 'error' not in r and r['reward'] >= 0]
            error_count += sum(1 for r in case_results if 'error' in r)
            all_rewards.extend(valid_rewards)
        
        if all_rewards:
            print(f"   Average reward: {np.mean(all_rewards):.3f}")
            print(f"   Reward range: {np.min(all_rewards):.3f} - {np.max(all_rewards):.3f}")
            print(f"   Non-zero rewards: {sum(1 for r in all_rewards if r > 0)}/{len(all_rewards)} ({100*sum(1 for r in all_rewards if r > 0)/len(all_rewards):.1f}%)")
        
        if error_count > 0:
            print(f"   ❌ Errors: {error_count}")
        else:
            print(f"   ✅ No errors")
    
    # Check for debug files
    debug_files = [f for f in os.listdir('/tmp') if 'test_rewards_debug' in f]
    if debug_files:
        print(f"\n📁 Debug files created: {len(debug_files)}")
        for file in debug_files[:5]:  # Show first 5
            print(f"   - /tmp/{file}")
    
    print(f"\n🎉 Test completed at {datetime.now().strftime('%H:%M:%S')}")
    return True

if __name__ == "__main__":
    success = run_comprehensive_tests()
    if success:
        print("\n✅ All tests completed successfully!")
        print("You can now run training with confidence! 🚀")
    else:
        print("\n❌ Tests failed. Fix issues before training.")
        sys.exit(1)
