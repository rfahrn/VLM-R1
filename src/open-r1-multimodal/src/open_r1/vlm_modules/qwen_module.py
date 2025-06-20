from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import re, json
from datetime import datetime
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
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
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
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    @staticmethod
    def format_reward_rec(completions, **kwargs):
        import re, json
        answer_pat = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        rewards = []
        for comp in completions:
            content = comp[0]["content"]
            m = answer_pat.search(content)
            if not m:
                rewards.append(0.0)
                continue
            txt = m.group(1).strip()
            # Try JSON parse
            try:
                data = json.loads(txt)
                # Accept either a list of 4 numbers or a dict with "boxes"
                if (isinstance(data, list) and len(data) == 4) or \
                   (isinstance(data, dict) and "boxes" in data and len(data["boxes"][0]) == 4):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
        return rewards


    """
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        Check if the Qwen model output matches a specific format.
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]"""
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards
        
    
    @staticmethod
    def iou_reward_new(completions, solution, **kwargs):
        import re, json

        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2] - 1, box2[2] - 1)
            inter_y2 = min(box1[3] - 1, box2[3] - 1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
            else:
                inter = 0
            union = ((box1[2] - box1[0]) * (box1[3] - box1[1]) +
                     (box2[2] - box2[0]) * (box2[3] - box2[1]) -
                     inter)
            return float(inter) / union if union > 0 else 0.0

        answer_pat = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        rewards = []

        for completion, sol in zip(completions, solution):
            # -- ground truth --
            gt_box = None
            m = answer_pat.search(sol or "")
            if m:
                txt = m.group(1).strip()
                if txt:
                    try:
                        data = json.loads(txt)
                        if isinstance(data, dict) and data.get("boxes"):
                            gt_box = data["boxes"][0]
                        elif isinstance(data, list) and len(data) == 4:
                            gt_box = data
                    except (ValueError, TypeError):
                        pass

            # -- prediction --
            pred_box = None
            content = completion[0].get("content", "")
            m2 = answer_pat.search(content)
            if m2:
                txt2 = m2.group(1).strip()
                if txt2:
                    try:
                        data2 = json.loads(txt2)
                        if isinstance(data2, dict) and data2.get("boxes"):
                            pred_box = data2["boxes"][0]
                        elif isinstance(data2, list) and len(data2) == 4:
                            pred_box = data2
                    except (ValueError, TypeError):
                        pass

            # -- compute IoU or zero --
            if gt_box is not None and pred_box is not None:
                rewards.append(iou(pred_box, gt_box))
            else:
                rewards.append(0.0)

        return rewards


    

    #@staticmethod 
    #def combined_reward(completions, solution, **kwargs):
     #   ious = Qwen2VLModule.iou_reward_new(completions, solution, **kwargs)
      #  fmts = Qwen2VLModule.format_reward_rec(completions, **kwargs)
       # return [0.5*i + 0.5*f for i,f in zip(ious, fmts)]
        
    
        """
        @staticmethod
        def select_reward_func(func: str, task_type: str):
            if func == "accuracy":
                match task_type:
                    case "rec":
                        return Qwen2VLModule.iou_reward
                    case _:
                        raise ValueError(f"Unsupported reward function: {func}")
            elif func == "format":
                match task_type:
                    case "rec":
                        return Qwen2VLModule.format_reward_rec
                    case _:
                        raise ValueError(f"Unsupported reward function: {func}")
            else:
                raise ValueError(f"Unsupported reward function: {func}")"""

    @staticmethod
    def parse_boxes_from_text(text):
        """
        Parses a string containing multiple boxes in either JSON or 
        simple list-of-lists format.
        """
        import json, re
        # Try JSON
        try:
            data = json.loads(text)
            if isinstance(data, list):
                # list of boxes
                return [list(map(float, box)) for box in data if isinstance(box, (list, tuple)) and len(box) == 4]
            elif isinstance(data, dict) and 'boxes' in data:
                return [list(map(float, box)) for box in data['boxes']]
        except Exception:
            # Fallback: match all [n, n, n, n] patterns
            matches = re.findall(r"\[([\d\.eE+-]+),\s*([\d\.eE+-]+),\s*([\d\.eE+-]+),\s*([\d\.eE+-]+)\]", text)
            return [list(map(float, m)) for m in matches]
        return []
    
    @staticmethod
    def mean_max_iou_reward(completions, solution, **kwargs):
        # Returns: for each sample, mean(best IoU GT->Pred), mean(best IoU Pred->GT), mean of both, and list of all IoUs
        import numpy as np
        import re
    
        answer_pat = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        results = []
    
        for comp, sol in zip(completions, solution):
            comp_text = comp[0].get("content", "")
            pred_match = answer_pat.search(comp_text)
            sol_match = answer_pat.search(sol or "")
    
            pred_boxes = Qwen2VLModule.parse_boxes_from_text(pred_match.group(1).strip()) if pred_match else []
            gt_boxes   = Qwen2VLModule.parse_boxes_from_text(sol_match.group(1).strip()) if sol_match else []
    
            if not pred_boxes or not gt_boxes:
                results.append({
                    "mean_iou_gt_to_pred": 0.0,
                    "mean_iou_pred_to_gt": 0.0,
                    "mean_iou": 0.0,
                    "all_ious": [],
                    "n_pred": len(pred_boxes),
                    "n_gt": len(gt_boxes),
                })
                continue
    
            # Compute pairwise IoU matrix
            def iou(box1, box2):
                inter_x1 = max(box1[0], box2[0])
                inter_y1 = max(box1[1], box2[1])
                inter_x2 = min(box1[2], box2[2])
                inter_y2 = min(box1[3], box2[3])
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = box1_area + box2_area - inter_area
                return inter_area / union if union > 0 else 0.0
    
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt in enumerate(gt_boxes):
                for j, pred in enumerate(pred_boxes):
                    iou_matrix[i, j] = iou(gt, pred)
    
            # mean of best IoU for each GT (coverage), mean of best IoU for each pred (precision)
            mean_iou_gt_to_pred = np.mean(iou_matrix.max(axis=1))
            mean_iou_pred_to_gt = np.mean(iou_matrix.max(axis=0))
            mean_iou = (mean_iou_gt_to_pred + mean_iou_pred_to_gt) / 2
            results.append({
                "mean_iou_gt_to_pred": float(mean_iou_gt_to_pred),
                "mean_iou_pred_to_gt": float(mean_iou_pred_to_gt),
                "mean_iou": float(mean_iou),
                "all_ious": iou_matrix.flatten().tolist(),
                "n_pred": len(pred_boxes),
                "n_gt": len(gt_boxes),
            })
        return results

    @staticmethod
    def combined_reward(completions, solution, **kwargs):
        iou_stats = Qwen2VLModule.mean_max_iou_reward(completions, solution, **kwargs)
        fmts = Qwen2VLModule.format_reward_rec(completions, **kwargs)
        
        # Return only the combined scalar values - the trainer expects a list of numbers
        combined = [0.5*stat['mean_iou'] + 0.5*f for stat, f in zip(iou_stats, fmts)]
        
        # Optional: Log the detailed stats for debugging (if DEBUG_MODE is enabled)
        if os.getenv("DEBUG_MODE") == "true":
            import os
            from datetime import datetime
            log_path = os.getenv("LOG_PATH", "debug.log")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            
            with open(log_path.replace(".txt", "_combined_rewards.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Combined Reward Details -------------\n")
                for i, (stat, fmt, comb) in enumerate(zip(iou_stats, fmts, combined)):
                    f.write(f"Sample {i}: mean_iou={stat['mean_iou']:.3f}, format={fmt:.3f}, combined={comb:.3f}\n")
                    f.write(f"  IoU details: gt_to_pred={stat['mean_iou_gt_to_pred']:.3f}, pred_to_gt={stat['mean_iou_pred_to_gt']:.3f}\n")
                    f.write(f"  Counts: n_pred={stat['n_pred']}, n_gt={stat['n_gt']}\n")
                f.write("\n")
        
        return combined


    # Alternative approach: Create separate reward functions for individual metrics
    @staticmethod
    def iou_only_reward(completions, solution, **kwargs):
        """Return only IoU rewards for individual analysis"""
        iou_stats = Qwen2VLModule.mean_max_iou_reward(completions, solution, **kwargs)
        return [stat['mean_iou'] for stat in iou_stats]
    
    @staticmethod
    def format_only_reward(completions, solution, **kwargs):
        """Return only format rewards for individual analysis"""
        return Qwen2VLModule.format_reward_rec(completions, **kwargs)
    
    
    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if task_type != "rec":
            raise ValueError(f"Unsupported reward function for task type: {task_type}")
    
        # Main single-metric reward functions
        if func == "accuracy":
            return Qwen2VLModule.iou_reward
        elif func == "format":
            return Qwen2VLModule.format_reward_rec
        elif func == "combined":
            return Qwen2VLModule.combined_reward
        elif func == "iou_only":  # New: for individual IoU analysis
            return Qwen2VLModule.iou_only_reward
        elif func == "format_only":  # New: for individual format analysis
            return Qwen2VLModule.format_only_reward
    
        # Multi-metric accessors (all use mean_max_iou_reward internally)
        elif func == "mean_iou":
                return lambda completions, solution, **kwargs: [
                    stat["mean_iou"] for stat in Qwen2VLModule.mean_max_iou_reward(completions, solution, **kwargs)
                ]
            elif func == "mean_iou_gt_to_pred":
                return lambda completions, solution, **kwargs: [
                    stat["mean_iou_gt_to_pred"] for stat in Qwen2VLModule.mean_max_iou_reward(completions, solution, **kwargs)
                ]
            elif func == "mean_iou_pred_to_gt":
                return lambda completions, solution, **kwargs: [
                    stat["mean_iou_pred_to_gt"] for stat in Qwen2VLModule.mean_max_iou_reward(completions, solution, **kwargs)
                ]
            elif func == "n_pred":
                return lambda completions, solution, **kwargs: [
                    stat["n_pred"] for stat in Qwen2VLModule.mean_max_iou_reward(completions, solution, **kwargs)
                ]
            elif func == "n_gt":
                return lambda completions, solution, **kwargs: [
                    stat["n_gt"] for stat in Qwen2VLModule.mean_max_iou_reward(completions, solution, **kwargs)
                ]
            else:
                raise ValueError(f"Unsupported reward function: {func}")


