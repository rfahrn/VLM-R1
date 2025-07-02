# ────────────────────────────────────────────────────────────────────────────
# open_r1/vlm_modules/llava_module.py
# ────────────────────────────────────────────────────────────────────────────
"""
Adapter module so VLM-R1 can drive **LLaVA-OneVision** checkpoints
(e.g. llava-hf/llava-onevision-qwen2-7b-ov-hf).
"""

from typing import Any, Dict, Union, List

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, SiglipVisionModel
from trl.data_utils import maybe_apply_chat_template

from open_r1.vlm_modules.vlm_module import VLMBaseModule
from open_r1.vlm_modules.qwen_module import Qwen2VLModule  # we reuse its prompt template


def _patch_init():
    orig_init = LlavaOnevisionForConditionalGeneration.__init__

    def wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)

        # alias expected by the trainer
        if not hasattr(self, "vision_model"):
            self.vision_model = self.vision_tower

        # ------- NO self-cycle anymore -------------------------
        vt = self.vision_tower
        vt.gradient_checkpointing = False           # default
        # provide a *dummy* encoder with the attribute, but not a Module
        if not hasattr(vt, "encoder"):
            class _EncoderStub:                      # lightweight shell
                gradient_checkpointing: bool = False
            vt.encoder = _EncoderStub()
    LlavaOnevisionForConditionalGeneration.__init__ = wrapped_init

_patch_init()



class LlavaOneVisionModule(VLMBaseModule):
    # ───────────────────────────── identifiers ──────────────────────────────
    def get_vlm_key(self) -> str:
        return "llava"

    # ─────────────────────── model / processor classes ──────────────────────
    def get_model_class(
        self, model_id: str, model_init_kwargs: Dict[str, Any]
    ):
        # VLM-R1 sometimes injects generic extras like `use_cache=False`
        # which One-Vision’s from_pretrained() does *not* accept → drop them.
        model_init_kwargs.pop("use_cache", None)
        return LlavaOnevisionForConditionalGeneration

    def get_processing_class(self):
        return AutoProcessor

    # ───────────────────────── post-load patching ───────────────────────────
    def post_model_init(self, model, processing_class):
        # ------------------------------------------------------------------
        # 1) trainer expects .vision_model  ---------------------------------
        if not hasattr(model, "vision_model"):
            model.vision_model = model.vision_tower

        # ------------------------------------------------------------------
        # 2) Qwen-2 backend MUST use left padding --------------------------
        try:
            tok = self.processor.tokenizer        # AutoProcessor ➜ tokenizer
        except AttributeError:
            tok = None

        if tok is not None:
            tok.padding_side = "left"             # <- critical line!
            # make sure a pad-token exists
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model.config.pad_token_id = tok.pad_token_id


    # ─────────────────────── keyword helpers (unchanged) ────────────────────
    def get_vision_modules_keywords(self) -> List[str]:
        # both names are now discoverable
        return ["vision_model", "vision_tower"]

    def get_custom_multimodal_keywords(self):
        # keep BOTH fields that the processor returns and the model needs
        return ["pixel_values", "image_sizes"]

    def get_non_generate_params(self):
        return []

    def get_custom_processing_keywords(self):
        # propagate max/min-pixels settings to the image processor
        return [("image_processor", "max_pixels"),
                ("image_processor", "min_pixels")]

    # ───────────────────────── prompt helpers ───────────────────────────────
    def prepare_prompt(
        self,
        processing_class,
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ):
        # same mechanism the Qwen module uses
        return [
            maybe_apply_chat_template(example, processing_class)["prompt"]
            for example in inputs
        ]

    def prepare_model_inputs(
            self,
            processing_class,
            prompts_text,
            images,
            *,
            padding_side: str = "left",   # <-- accept it for compatibility
            **other,                      # <-- swallow any future extras
    ):
        # we still force-left internally
        kwargs = dict(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",          # <- hard-coded
            add_special_tokens=False,
        )
        if images:
            kwargs["images"] = images
        return processing_class(**kwargs)


    # ─────────────────────── prompt template (reuse Qwen) ───────────────────
    @staticmethod
    def get_question_template(task_type: str):
        return Qwen2VLModule.get_question_template(task_type)
    
    @classmethod
    def load_pretrained(cls, model_id: str, **kwargs):
        """
        Mirror the helper present in Qwen’s adapter so external utilities
        (like our checker) can instantiate the module in one line.
        """
        self = cls()
        # 1) strip unsupported kwargs  (use_cache, device_map, …)
        kwargs.pop("use_cache", None)

        # 2) model + processor
        model_cls = self.get_model_class(model_id, kwargs)
        self.model      = model_cls.from_pretrained(model_id, **kwargs)
        proc_cls        = self.get_processing_class()
        self.processor  = proc_cls.from_pretrained(model_id)

        # 3) post-init tweaks (aliases, etc.)
        self.post_model_init(self.model, proc_cls)

        return self
