from qwen_module import Qwen2VLModule
test_cases = [
    # Case 1: Perfect format
    "<think>reasoning</think><answer>It is displayed at [0.19, 0.5, 0.48, 0.84] and [0.63, 0.48, 0.98, 0.87] in the radiograph.</answer>",
    
    # Case 2: Model generates text without coordinates
    "<think>reasoning</think><answer>The patchy bibasilar opacities can be seen in the lower regions of both lungs.</answer>",
    
    # Case 3: Model generates coordinates in different format
    "<think>reasoning</think><answer>The area is located at coordinates (0.19, 0.5, 0.48, 0.84) and (0.63, 0.48, 0.98, 0.87).</answer>",
    
    # Case 4: Model generates partial coordinates
    "<think>reasoning</think><answer>It is at [0.19, 0.5] in the image.</answer>",
]

ground_truth = "It is displayed at [0.19, 0.5, 0.48, 0.84] and [0.63, 0.48, 0.98, 0.87] in the radiograph."

for i, test_case in enumerate(test_cases, 1):
    print(f"\n=== Test Case {i} ===")
    print(f"Input: {test_case}")
    
    pred_answer = Qwen2VLModule.extract_answer_content(test_case)
    gt_answer = Qwen2VLModule.extract_answer_content(ground_truth)
    
    pred_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(pred_answer)
    gt_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(gt_answer)
    
    print(f"Extracted answer: {pred_answer}")
    print(f"Pred bboxes: {pred_bboxes}")
    print(f"GT bboxes: {gt_bboxes}")
    
    if pred_bboxes and gt_bboxes:
        reward = Qwen2VLModule.calculate_multi_bbox_score(pred_bboxes, gt_bboxes)
    else:
        reward = 0.0
    
    print(f"IoU reward: {reward}")
