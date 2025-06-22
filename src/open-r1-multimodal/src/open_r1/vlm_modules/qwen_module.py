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
    def format_only_reward(completions, solution, **kwargs):
        """Format only (for individual analysis)."""
        return Qwen2VLModule.format_reward_rec(completions, **kwargs)

    # ========================================================================
    # REWARD FUNCTION SELECTION
    # ========================================================================

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        """Select the appropriate reward function for ALL your experiments."""
        if task_type != "rec":
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Standard reward functions (work with --is_reward_customized_from_vlm_module=True)
        if func == "accuracy":
            return Qwen2VLModule.iou_reward
        elif func == "format":
            return Qwen2VLModule.format_reward_rec
        
        # Custom reward functions for your experiments
        elif func == "iou":
            return Qwen2VLModule.iou_reward
        elif func == "map":
            return Qwen2VLModule.map_reward
        elif func == "combined":
            return Qwen2VLModule.combined_reward
        elif func == "combined_map":
            return Qwen2VLModule.combined_map_reward
        elif func == "iou_only":
            return Qwen2VLModule.iou_only_reward
        elif func == "map_only":
            return Qwen2VLModule.map_only_reward
        elif func == "format_only":
            return Qwen2VLModule.format_only_reward
        elif func == "iou_fbeta":
            return Qwen2VLModule.iou_fbeta_reward_batch
        else:
            raise ValueError(f"Unsupported reward function: {func}")
