from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .torch_utils import require_torch


torch, nn = require_torch()


def require_transformers():
    try:
        from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Qwen-VL backbone requires Hugging Face Transformers. "
            "Install optional Qwen deps with `pip install transformers qwen-vl-utils`."
        ) from exc
    return AutoProcessor, AutoModelForImageTextToText, AutoModel


@dataclass
class QwenVLInputs:
    messages: list[dict[str, Any]]
    image_paths: list[str] | None = None
    video_paths: list[str] | None = None


class QwenVLProcessorAdapter:
    """Small wrapper around Qwen-VL chat formatting and vision preprocessing."""

    def __init__(self, model_name: str, trust_remote_code: bool = True):
        AutoProcessor, _, _ = require_transformers()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    @staticmethod
    def build_messages(instruction: str, image_path: str | None = None) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]

    def __call__(
        self,
        instructions: list[str],
        image_paths: list[str | None] | None = None,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        image_paths = image_paths or [None] * len(instructions)
        messages = [self.build_messages(text, image_path) for text, image_path in zip(instructions, image_paths)]
        text = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in messages
        ]
        image_inputs = None
        video_inputs = None
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        except ModuleNotFoundError:
            image_inputs = [path for path in image_paths if path]
        processor_kwargs: dict[str, Any] = {
            "text": text,
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        inputs = self.processor(**processor_kwargs)
        if device is not None:
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        return inputs


class QwenVLBackbone(nn.Module):
    """Qwen-VL/Qwen2.5-VL feature backbone projected into OARL-VLA hidden space.

    The module is intentionally optional: it is only constructed when
    `OARLVLAConfig.vlm_backbone == "qwen_vl"`, so lightweight tests and symbolic
    training do not download large model weights.
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        *,
        freeze: bool = True,
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        device_map: str | None = None,
    ):
        super().__init__()
        _, AutoModelForImageTextToText, AutoModel = require_transformers()
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        elif torch_dtype == "auto":
            load_kwargs["torch_dtype"] = "auto"
        if device_map:
            load_kwargs["device_map"] = device_map
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        except Exception:
            self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        source_dim = self._hidden_size()
        self.proj = nn.Linear(source_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, qwen_inputs: dict[str, Any]):
        outputs = self.model(**qwen_inputs, output_hidden_states=True, return_dict=True)
        if not getattr(outputs, "hidden_states", None):
            raise RuntimeError("Qwen-VL model did not return hidden states; pass output_hidden_states=True support is required.")
        hidden = outputs.hidden_states[-1]
        attention_mask = qwen_inputs.get("attention_mask")
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.to(hidden.device).float().unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(self.proj(pooled.float()))

    def _hidden_size(self) -> int:
        config = self.model.config
        for attr in ("hidden_size", "text_hidden_size"):
            value = getattr(config, attr, None)
            if value:
                return int(value)
        text_config = getattr(config, "text_config", None)
        if text_config is not None and getattr(text_config, "hidden_size", None):
            return int(text_config.hidden_size)
        raise RuntimeError("Could not infer Qwen-VL hidden size from model config.")

