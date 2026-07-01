from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from mmengine.model import BaseModel
from transformers import AutoModel
from xtuner.model.utils import guess_load_checkpoint, make_inputs_require_grad


class Qwen3VLForMotionAtlas(BaseModel):
    """Thin MMEngine wrapper for Qwen3-VL supervised fine-tuning."""

    def __init__(
        self,
        mllm_name_or_path: str,
        trust_remote_code: bool = True,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze_llm: bool = False,
        freeze_visual_encoder: bool = False,
        use_activation_checkpointing: bool = True,
        pretrained_pth: str | None = None,
        **_: Any,
    ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.mllm = AutoModel.from_pretrained(
            mllm_name_or_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        if hasattr(self.mllm, "config"):
            self.config = self.mllm.config
        if hasattr(self.mllm, "model") and hasattr(self.mllm.model, "config"):
            self.mllm.model.config.use_cache = False

        if freeze_llm:
            language_model = self._language_model()
            if language_model is not None:
                language_model.requires_grad_(False)
        if freeze_visual_encoder:
            visual_model = self._visual_model()
            if visual_model is not None:
                visual_model.requires_grad_(False)

        if use_activation_checkpointing:
            if hasattr(self.mllm, "enable_input_require_grads"):
                self.mllm.enable_input_require_grads()
            elif hasattr(self.mllm, "get_input_embeddings"):
                self.mllm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if hasattr(self.mllm, "gradient_checkpointing_enable"):
                self.mllm.gradient_checkpointing_enable()
            elif hasattr(self.mllm, "model") and hasattr(self.mllm.model, "gradient_checkpointing_enable"):
                self.mllm.model.gradient_checkpointing_enable()

        if pretrained_pth:
            state_dict = guess_load_checkpoint(pretrained_pth)
            msg = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained checkpoint from {pretrained_pth}: {msg}", flush=True)

    def _language_model(self) -> Any | None:
        candidates = [
            ("model", "language_model"),
            ("language_model",),
            ("model", "model"),
        ]
        return self._first_attr(candidates)

    def _visual_model(self) -> Any | None:
        candidates = [
            ("model", "visual"),
            ("visual",),
            ("vision_model",),
        ]
        return self._first_attr(candidates)

    def _first_attr(self, candidates: list[tuple[str, ...]]) -> Any | None:
        for path in candidates:
            value: Any = self.mllm
            ok = True
            for name in path:
                if not hasattr(value, name):
                    ok = False
                    break
                value = getattr(value, name)
            if ok:
                return value
        return None

    def _visual_dtype(self) -> torch.dtype:
        visual = self._visual_model()
        if visual is not None:
            for param in visual.parameters():
                return param.dtype
        for param in self.mllm.parameters():
            return param.dtype
        return torch.bfloat16

    def state_dict(self, *args: Any, **kwargs: Any) -> OrderedDict:
        return super().state_dict(*args, **kwargs)

    def init_weights(self) -> None:
        pass

    def forward(self, data: dict[str, Any], data_samples: Any = None, mode: str = "loss") -> dict[str, torch.Tensor]:
        if mode != "loss":
            raise NotImplementedError(f"Unsupported mode: {mode}")

        visual_dtype = self._visual_dtype()
        kwargs = {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
            "position_ids": data.get("position_ids"),
            "labels": data["labels"],
            "use_cache": False,
        }
        if data.get("pixel_values") is not None:
            kwargs["pixel_values"] = data["pixel_values"].to(visual_dtype)
            kwargs["image_grid_thw"] = data.get("image_grid_thw")
        if data.get("pixel_values_videos") is not None:
            kwargs["pixel_values_videos"] = data["pixel_values_videos"].to(visual_dtype)
            kwargs["video_grid_thw"] = data.get("video_grid_thw")
        if data.get("second_per_grid_ts") is not None:
            kwargs["second_per_grid_ts"] = data["second_per_grid_ts"]

        outputs = self.mllm(**kwargs)
        return {"loss": outputs.loss}

