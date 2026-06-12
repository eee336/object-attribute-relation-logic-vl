from __future__ import annotations

from pathlib import Path
from typing import Any

from .encoders import SimpleTokenizer
from .torch_utils import require_torch
from .vla_model import OARLVLAConfig, OARLVLAModel


torch, _ = require_torch()


def save_checkpoint(
    path: str | Path,
    model: OARLVLAModel,
    tokenizer: SimpleTokenizer,
    feature_metadata: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "feature_metadata": feature_metadata,
            "extra": _json_safe(extra or {}),
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[OARLVLAModel, SimpleTokenizer, dict[str, Any], dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    config = OARLVLAConfig.from_dict(payload["config"])
    model = OARLVLAModel(config)
    model.load_state_dict(payload["model_state"])
    tokenizer = SimpleTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload.get("feature_metadata", {}), payload.get("extra", {})


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
