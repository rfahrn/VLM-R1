from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union, List, Tuple
from trl.data_utils import maybe_apply_chat_template
import torch
import re, json
import os
from datetime import datetime
import numpy as np
from open_r1.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec": # specific for bbox task adapted 
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Format your answer as coordinate lists like [x1, y1, x2, y2] where coordinates are between 0 and 1. For multiple regions, list all coordinates separated by 'and'. Example: <answer>[0.19, 0.5, 0.48, 0.84] and [0.63, 0.48, 0.98, 0.87]</answer>"""
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    @staticmethod
    def extract_all_bboxes_from_text(text: str) -> List[List[float]]:
        """Extract ALL bounding box coordinates from text."""
        text = text.strip()
        bboxes = []
        
        # Patterns to match coordinates in brackets or parentheses
        patterns = [
            r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',
            r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                coords = [float(x) for x in match]
                bboxes.append(coords)
        
        # If no brackets/parentheses, try to find comma-separated coordinates
        if not bboxes:
            number_pattern = r'(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)'
            matches = re.findall(number_pattern, text)
            for match in matches:
                coords = [float(x) for x in match]
                bboxes.append(coords)
        
        return bboxes

    @staticmethod
    def extract_answer_content(text: str) -> str:
        """Extract content from <answer></answer> tags."""
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, text, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        else:
            return text.strip()

    @staticmethod
    def calculate_single_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter_area = 0
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return float(inter_area) / float(union_area)

    @staticmethod
    def calculate_multi_bbox_score(pred_bboxes: List[List[float]], gt_bboxes: List[List[float]]) -> float:
        """Calculate IoU score for multiple bounding boxes using optimal matching."""
        if not pred_bboxes and not gt_bboxes:
            return 1.0
        
        if not pred_bboxes or not gt_bboxes:
            return 0.0
        
        if len(pred_bboxes) == len(gt_bboxes) == 1:
            return Qwen2VLModule.calculate_single_iou(pred_bboxes[0], gt_bboxes[0])
        
        # Create IoU matrix
        iou_matrix = []
        for pred_box in pred_bboxes:
            row = []
            for gt_box in gt_bboxes:
                iou_val = Qwen2VLModule.calculate_single_iou(pred_box, gt_box)
                row.append(iou_val)
            iou_matrix.append(row)
        
        # Greedy matching
        used_gt = set()
        used_pred = set()
        total_iou = 0.0
        matches = 0
        
        while len(used_pred) < len(pred_bboxes) and len(used_gt) < len(gt_bboxes):
            best_iou = 0.0
            best_pred = -1
            best_gt = -1
            
            for p in range(len(pred_bboxes)):
                if p in used_pred:
                    continue
                for g in range(len(gt_bboxes)):
                    if g in used_gt:
                        continue
                    if iou_matrix[p][g] > best_iou:
                        best_iou = iou_matrix[p][g]
                        best_pred = p
                        best_gt = g
            
            if best_iou > 0.0:
                total_iou += best_iou
                matches += 1
                used_pred.add(best_pred)
                used_gt.add(best_gt)
            else:
                break
        
        if matches == 0:
            return 0.0
        
        avg_iou = total_iou / matches
        total_boxes = max(len(pred_bboxes), len(gt_bboxes))
        coverage_penalty = matches / total_boxes
        
        return avg_iou * coverage_penalty

    @staticmethod
    def calculate_map_score(pred_bboxes: List[List[float]], gt_bboxes: List[List[float]], iou_threshold: float = 0.5) -> float:
        """
        Calculate Mean Average Precision (mAP) score.
        Simplified version for bbox detection.
        """
        if not pred_bboxes and not gt_bboxes:
            return 1.0
        
        if not pred_bboxes or not gt_bboxes:
            return 0.0
        
        # Calculate IoU for all pred-gt pairs
        ious = []
        for pred_box in pred_bboxes:
            for gt_box in gt_bboxes:
                iou_val = Qwen2VLModule.calculate_single_iou(pred_box, gt_box)
                ious.append(iou_val)
        
        # Count true positives (IoU > threshold)
        true_positives = sum(1 for iou in ious if iou >= iou_threshold)
        
        # Calculate precision and recall
        precision = true_positives / len(pred_bboxes) if pred_bboxes else 0.0
        recall = true_positives / len(gt_bboxes) if gt_bboxes else 0.0
        
        # F1-score as a proxy for mAP (simplified)
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    # ========================================================================
    # REWARD FUNCTIONS FOR ALL EXPERIMENTS
    # ========================================================================

    @staticmethod
    def _iou(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        inter  = max(0, x1-x0) * max(0, y1-y0)
        if inter == 0: return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _match_boxes(pred, gt):
        """Greedy one-to-one matching with highest IoU."""
        pairs, used = [], set()
        for i, p in enumerate(pred):
            best, best_iou = None, 0.0
            for j, g in enumerate(gt):
                if j in used: continue
                iou = Qwen2VLModule._iou(p, g)
                if iou > best_iou:
                    best_iou, best = iou, j
            if best is not None:
                pairs.append((i, best, best_iou))
                used.add(best)
        return pairs

    @staticmethod
    def continuous_iou_fbeta_reward(pred_bboxes, gt_bboxes, alpha=0.5):
        """
        Continuous reward  ∈ [0,1]
            R = mean(IoU) × Fβ  with  β = alpha / (1-alpha)
        mean(IoU) – localisation quality  
        Fβ        – balance precision & recall (β=1 when alpha=0.5)
        """
        if not pred_bboxes and not gt_bboxes:
            return 1.0
        if not pred_bboxes or not gt_bboxes:
            return 0.0
        matches   = Qwen2VLModule._match_boxes(pred_bboxes, gt_bboxes)
        if not matches:
            return 0.0
        mean_iou  = sum(m[2] for m in matches) / len(matches)
        tp        = len(matches)
        precision = tp / len(pred_bboxes)
        recall    = tp / len(gt_bboxes)
        beta_sq   = (alpha / (1-alpha))**2
        f_beta    = (1+beta_sq)*precision*recall / (beta_sq*precision + recall + 1e-9)
        return mean_iou * f_beta

    @staticmethod
    def iou_fbeta_reward_batch(completions, solution, **kwargs):
        # alpha can be set as desired, or passed via kwargs
        alpha = kwargs.get("alpha", 0.5)
        rewards = []
        for content, sol in zip([c[0]["content"] for c in completions], solution):
            pred_answer = Qwen2VLModule.extract_answer_content(content)
            gt_answer   = Qwen2VLModule.extract_answer_content(sol)
            pred_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(pred_answer)
            gt_bboxes   = Qwen2VLModule.extract_all_bboxes_from_text(gt_answer)
            reward = Qwen2VLModule.continuous_iou_fbeta_reward(pred_bboxes, gt_bboxes, alpha=alpha)
            rewards.append(reward)
        return rewards
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """IoU-based accuracy reward (your main metric)."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            reward = 0.0
            
            try:
                # Extract ground truth - handle both formats
                gt_answer = Qwen2VLModule.extract_answer_content(sol)
                gt_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(gt_answer)
                
                # Extract prediction - handle both formats  
                pred_answer = Qwen2VLModule.extract_answer_content(content)
                pred_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(pred_answer)
                
                # Debug logging
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH", "debug.log")
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    with open(log_path.replace(".txt", "_iou_debug.txt"), "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} IoU Debug -------------\n")
                        f.write(f"Raw content: {content}\n")
                        f.write(f"Raw solution: {sol}\n")
                        f.write(f"Extracted pred_answer: {pred_answer}\n")
                        f.write(f"Extracted gt_answer: {gt_answer}\n")
                        f.write(f"Pred bboxes: {pred_bboxes}\n")
                        f.write(f"GT bboxes: {gt_bboxes}\n")
                
                reward = Qwen2VLModule.calculate_multi_bbox_score(pred_bboxes, gt_bboxes)
                
                # More debug logging
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH", "debug.log")
                    with open(log_path.replace(".txt", "_iou_debug.txt"), "a", encoding='utf-8') as f:
                        f.write(f"Final IoU reward: {reward}\n\n")
                        
            except Exception as e:
                print(f"Error calculating IoU reward: {e}")
                reward = 0.0
                
                # Log errors too
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH", "debug.log")
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    with open(log_path.replace(".txt", "_iou_errors.txt"), "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Error -------------\n")
                        f.write(f"Error: {e}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n\n")
            
            rewards.append(reward)
        
        return rewards

    @staticmethod
    def map_reward(completions, solution, **kwargs):
        """mAP-based accuracy reward."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            reward = 0.0
            
            try:
                gt_answer = Qwen2VLModule.extract_answer_content(sol)
                gt_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(gt_answer)
                
                pred_answer = Qwen2VLModule.extract_answer_content(content)
                pred_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(pred_answer)
                
                reward = Qwen2VLModule.calculate_map_score(pred_bboxes, gt_bboxes)
                    
            except Exception as e:
                print(f"Error calculating mAP reward: {e}")
                reward = 0.0
            
            rewards.append(reward)
            
            # Debug logging for mAP
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH", "debug.log")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path.replace(".txt", "_map_rewards.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} mAP Reward: {reward:.3f} -------------\n")
                    f.write(f"RAW MODEL OUTPUT: {content}\n")  # ← NEW: See full model output
                    f.write(f"RAW GROUND TRUTH: {sol}\n")     # ← NEW: See full ground truth
                    f.write(f"EXTRACTED PRED ANSWER: {pred_answer}\n")  # ← NEW: See extracted answer
                    f.write(f"EXTRACTED GT ANSWER: {gt_answer}\n")      # ← NEW: See extracted GT
                    f.write(f"Predicted Bboxes ({len(pred_bboxes)}): {pred_bboxes}\n")
                    f.write(f"GT Bboxes ({len(gt_bboxes)}): {gt_bboxes}\n\n")
        
        return rewards

    @staticmethod
    def partial_credit_iou_reward(completions, solution, **kwargs):
        """IoU reward with partial credit for close predictions."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            reward = 0.0
            
            try:
                gt_answer = Qwen2VLModule.extract_answer_content(sol)
                gt_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(gt_answer)
                
                pred_answer = Qwen2VLModule.extract_answer_content(content)
                pred_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(pred_answer)
                
                if not pred_bboxes and not gt_bboxes:
                    reward = 1.0
                elif not pred_bboxes or not gt_bboxes:
                    reward = 0.0
                else:
                    # Calculate IoU matrix
                    iou_matrix = []
                    for pred_box in pred_bboxes:
                        row = []
                        for gt_box in gt_bboxes:
                            iou = Qwen2VLModule.calculate_single_iou(pred_box, gt_box)
                            row.append(iou)
                        iou_matrix.append(row)
                    
                    # Greedy matching but with partial credit
                    used_gt = set()
                    used_pred = set()
                    total_score = 0.0
                    matches = 0
                    
                    while len(used_pred) < len(pred_bboxes) and len(used_gt) < len(gt_bboxes):
                        best_iou = 0.0
                        best_pred = -1
                        best_gt = -1
                        
                        for p in range(len(pred_bboxes)):
                            if p in used_pred:
                                continue
                            for g in range(len(gt_bboxes)):
                                if g in used_gt:
                                    continue
                                if iou_matrix[p][g] > best_iou:
                                    best_iou = iou_matrix[p][g]
                                    best_pred = p
                                    best_gt = g
                        
                        if best_iou > 0.05:  # Lower threshold - give credit for IoU > 5%
                            # Progressive reward: IoU < 0.1 = 25%, IoU < 0.3 = 50%, IoU >= 0.3 = 100%
                            if best_iou >= 0.3:
                                box_score = best_iou
                            elif best_iou >= 0.1:
                                box_score = 0.5 * best_iou  # Partial credit
                            else:
                                box_score = 0.25 * best_iou  # Small credit for being close
                            
                            total_score += box_score
                            matches += 1
                            used_pred.add(best_pred)
                            used_gt.add(best_gt)
                        else:
                            break
                    
                    if matches == 0:
                        reward = 0.0
                    else:
                        avg_score = total_score / matches
                        total_boxes = max(len(pred_bboxes), len(gt_bboxes))
                        coverage_bonus = matches / total_boxes
                        reward = avg_score * coverage_bonus
                        
            except Exception as e:
                print(f"Error calculating partial credit IoU reward: {e}")
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards
    
    @staticmethod 
    def distance_based_reward(completions, solution, **kwargs):
        """Reward based on center distance + IoU."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            reward = 0.0
            
            try:
                gt_answer = Qwen2VLModule.extract_answer_content(sol)
                gt_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(gt_answer)
                
                pred_answer = Qwen2VLModule.extract_answer_content(content)
                pred_bboxes = Qwen2VLModule.extract_all_bboxes_from_text(pred_answer)
                
                if not pred_bboxes and not gt_bboxes:
                    reward = 1.0
                elif not pred_bboxes or not gt_bboxes:
                    reward = 0.0
                else:
                    def calculate_center_distance(box1, box2):
                        center1 = [(box1[0] + box1[2])/2, (box1[1] + box1[3])/2]
                        center2 = [(box2[0] + box2[2])/2, (box2[1] + box2[3])/2]
                        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                    
                    # Find best matches based on combination of IoU and distance
                    total_reward = 0.0
                    used_gt = set()
                    
                    for pred_box in pred_bboxes:
                        best_score = 0.0
                        best_gt = -1
                        
                        for g, gt_box in enumerate(gt_bboxes):
                            if g in used_gt:
                                continue
                                
                            iou = Qwen2VLModule.calculate_single_iou(pred_box, gt_box)
                            distance = calculate_center_distance(pred_box, gt_box)
                            
                            # Distance reward: closer = better (max distance ~1.4, so 1-distance/1.4)
                            distance_reward = max(0, 1 - distance/1.4)
                            
                            # Combined score: 70% IoU + 30% distance
                            combined_score = 0.7 * iou + 0.3 * distance_reward
                            
                            if combined_score > best_score:
                                best_score = combined_score
                                best_gt = g
                        
                        if best_gt != -1:
                            total_reward += best_score
                            used_gt.add(best_gt)
                    
                    # Average over max number of boxes
                    total_boxes = max(len(pred_bboxes), len(gt_bboxes))
                    reward = total_reward / total_boxes if total_boxes > 0 else 0.0
                        
            except Exception as e:
                print(f"Error calculating distance-based reward: {e}")
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards



    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Format reward - checks <think></think><answer></answer> structure."""
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def combined_reward(completions, solution, **kwargs):
        """Combined reward: 50% IoU + 50% Format."""
        iou_rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
        format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        
        combined = [0.5 * iou + 0.5 * fmt for iou, fmt in zip(iou_rewards, format_rewards)]
        return combined

    @staticmethod
    def combined_map_reward(completions, solution, **kwargs):
        """Combined reward: 50% mAP + 50% Format."""
        map_rewards = Qwen2VLModule.map_reward(completions, solution, **kwargs)
        format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        
        combined = [0.5 * map_r + 0.5 * fmt for map_r, fmt in zip(map_rewards, format_rewards)]
        return combined

    @staticmethod
    def iou_only_reward(completions, solution, **kwargs):
        """IoU only (for individual analysis)."""
        return Qwen2VLModule.iou_reward(completions, solution, **kwargs)

    @staticmethod
    def map_only_reward(completions, solution, **kwargs):
        """mAP only (for individual analysis)."""
        return Qwen2VLModule.map_reward(completions, solution, **kwargs)

    @staticmethod
    def format_only(completions, solution, **kwargs):
        """Format only (for individual analysis)."""
        return Qwen2VLModule.format_reward_rec(completions, **kwargs)


    @staticmethod
    def curriculum_combined_reward(completions, solution, **kwargs):
        """
        Curriculum learning: Start with high format weight, gradually shift to IoU.
        
        Training phases:
        - Phase 1 (steps 0-200): 90% Format + 10% IoU (learn structure)
        - Phase 2 (steps 200-500): 70% Format + 30% IoU (transition)  
        - Phase 3 (steps 500+): 30% Format + 70% IoU (focus on accuracy)
        """
        current_step = kwargs.get('current_step', 0)
        
        # Define curriculum phases
        if current_step < 200:
            format_weight, iou_weight = 0.9, 0.1
            phase = "Structure Learning"
        elif current_step < 500:
            format_weight, iou_weight = 0.7, 0.3  
            phase = "Transition"
        else:
            format_weight, iou_weight = 0.3, 0.7
            phase = "Accuracy Focus"
        
        # Calculate individual rewards
        iou_rewards = Qwen2VLModule.partial_credit_iou_reward(completions, solution, **kwargs)
        format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        
        combined = [format_weight * fmt + iou_weight * iou 
                    for fmt, iou in zip(format_rewards, iou_rewards)]
        
        # Enhanced debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "debug.log")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            
            with open(log_path.replace(".txt", "_curriculum_rewards.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- Step {current_step} - {phase} -------------\n")
                f.write(f"Weights: Format={format_weight:.1f}, IoU={iou_weight:.1f}\n")
                f.write(f"Avg Format: {sum(format_rewards)/len(format_rewards):.3f}\n")
                f.write(f"Avg IoU: {sum(iou_rewards)/len(iou_rewards):.3f}\n")
                f.write(f"Avg Combined: {sum(combined)/len(combined):.3f}\n\n")
        
        return combined

    @staticmethod 
    def momentum_reward(completions, solution, **kwargs):
        """
        Apply momentum/discount to format rewards to prevent over-reliance.
        
        If format reward has been 1.0 for N consecutive steps, start reducing it.
        This forces the model to focus more on IoU improvement.
        """
        format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        iou_rewards = Qwen2VLModule.partial_credit_iou_reward(completions, solution, **kwargs)
        
        # Track consecutive perfect format scores (simplified - in practice use global state)
        avg_format = sum(format_rewards) / len(format_rewards)
        momentum_factor = kwargs.get('momentum_factor', 0.95)
        
        # If format is consistently perfect, apply decay
        if avg_format >= 0.95:  # Nearly perfect format
            format_decay = momentum_factor ** kwargs.get('consecutive_perfect_format', 0)
            format_rewards = [r * format_decay for r in format_rewards]
        
        combined = [0.5 * fmt + 0.5 * iou for fmt, iou in zip(format_rewards, iou_rewards)]
        
        return combined

    @staticmethod
    def shaped_reward_with_bonus(completions, solution, **kwargs):
        """
        Reward shaping: Give bonus for improvements, penalty for stagnation.
        - Bonus: +0.2 if IoU improves significantly 
        - Penalty: -0.1 if only format is correct but IoU is 0
        - Progressive bonus: Extra reward for reaching IoU milestones
        """
        format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        iou_rewards = Qwen2VLModule.partial_credit_iou_reward(completions, solution, **kwargs)
        
        shaped_rewards = []
        
        for fmt, iou in zip(format_rewards, iou_rewards):
            base_reward = 0.5 * fmt + 0.5 * iou
            
            # Penalty for format-only success
            if fmt == 1.0 and iou < 0.05:
                penalty = -0.1
                base_reward += penalty
            
            # Progressive IoU bonuses
            if iou >= 0.5:
                bonus = 0.3  # High accuracy bonus
            elif iou >= 0.3:
                bonus = 0.2  # Good accuracy bonus  
            elif iou >= 0.1:
                bonus = 0.1  # Improvement bonus
            else:
                bonus = 0.0
                
            final_reward = base_reward + bonus
            shaped_rewards.append(max(0.0, final_reward))  # Ensure non-negative
        
        return shaped_rewards
    
    @staticmethod
    def adversarial_reward(completions, solution, **kwargs):
        """
        Adversarial approach: Randomly mask format reward to force IoU learning.
        With probability p, set format reward to 0 even if format is correct.
        This prevents the model from relying solely on format.
        """
        import random
        format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        iou_rewards = Qwen2VLModule.partial_credit_iou_reward(completions, solution, **kwargs)
        mask_probability = kwargs.get('format_mask_prob', 0.3)  # 30% chance to mask
        masked_format_rewards = []
        for fmt in format_rewards:
            if random.random() < mask_probability:
                masked_format_rewards.append(0.0)  # Force focus on IoU
            else:
                masked_format_rewards.append(fmt)
        combined = [0.5 * fmt + 0.5 * iou 
                    for fmt, iou in zip(masked_format_rewards, iou_rewards)]
        
        return combined

    @staticmethod
    def hierarchical_reward(completions, solution, **kwargs):
        """
        Hierarchical rewards: Format is prerequisite, IoU multiplies the reward.
        
        Structure: R = Format × (1 + IoU_bonus)
        - No format = 0 reward regardless of IoU
        - Perfect format + good IoU = exponential bonus
        - This creates strong incentive for both components
        """
        format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        iou_rewards = Qwen2VLModule.partial_credit_iou_reward(completions, solution, **kwargs)
        
        hierarchical_rewards = []
        
        for fmt, iou in zip(format_rewards, iou_rewards):
            if fmt == 0.0:
                # No format = no reward at all
                reward = 0.0
            else:
                # Format is prerequisite, IoU provides multiplicative bonus
                base_reward = 0.3  # Base reward for correct format
                iou_multiplier = 1 + 2 * iou  # IoU can triple the reward
                reward = base_reward * fmt * iou_multiplier
            
            hierarchical_rewards.append(reward)
        
        return hierarchical_rewards
    
    @staticmethod 
    def threshold_gated_reward(completions, solution, **kwargs):
        """
        Threshold-gated rewards: Only give format reward if IoU > threshold.
        
        Forces the model to achieve minimum IoU before getting format credit.
        Gradually lower the threshold as training progresses.
        """
        current_step = kwargs.get('current_step', 0)
        
        # Dynamic threshold: Start high, gradually lower
        if current_step < 300:
            iou_threshold = 0.15  # Require 15% IoU minimum
        elif current_step < 600:
            iou_threshold = 0.10  # Lower to 10%
        else:
            iou_threshold = 0.05  # Final threshold 5%
        
        format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        iou_rewards = Qwen2VLModule.partial_credit_iou_reward(completions, solution, **kwargs)
        
        gated_rewards = []
        
        for fmt, iou in zip(format_rewards, iou_rewards):
            if iou >= iou_threshold:
                # IoU meets threshold - give full combined reward
                reward = 0.4 * fmt + 0.6 * iou
            else:
                # IoU too low - only partial format credit
                reward = 0.1 * fmt + 0.6 * iou  # Reduced format weight
            
            gated_rewards.append(reward)
        
        return gated_rewards

    # ========================================================================
    # REWARD FUNCTION SELECTION
    # ========================================================================

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        """Enhanced reward function selection with anti-reward-hacking functions."""
        if task_type != "rec":
            raise ValueError(f"Unsupported task type: {task_type}")
        
        function_registry = {
            # Existing functions...
            "accuracy": Qwen2VLModule.iou_reward,
            "format": Qwen2VLModule.format_reward_rec,
            "format_only": Qwen2VLModule.format_reward_rec,  
            "iou": Qwen2VLModule.iou_reward,
            "partial_iou": Qwen2VLModule.partial_credit_iou_reward,
            "combined": Qwen2VLModule.combined_reward,
            
            # NEW: Anti-reward-hacking functions
            "curriculum_combined": Qwen2VLModule.curriculum_combined_reward,
            "momentum_reward": Qwen2VLModule.momentum_reward,
            "shaped_reward": Qwen2VLModule.shaped_reward_with_bonus,
            "adversarial_reward": Qwen2VLModule.adversarial_reward,
            "hierarchical": Qwen2VLModule.hierarchical_reward,
            "threshold_gated": Qwen2VLModule.threshold_gated_reward,
            
            # Existing functions...
            "map": Qwen2VLModule.map_reward,
            "combined_map": Qwen2VLModule.combined_map_reward,
            "iou_fbeta": Qwen2VLModule.iou_fbeta_reward_batch,
            "distance_based": Qwen2VLModule.distance_based_reward,
        }
        
        if func not in function_registry:
            available_funcs = list(function_registry.keys())
            raise ValueError(f"Unsupported reward function: '{func}'. Available: {available_funcs}")
        
        return function_registry[func]
