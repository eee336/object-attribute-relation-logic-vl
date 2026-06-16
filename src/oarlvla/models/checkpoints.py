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
    missing, unexpected = model.load_state_dict(
        _remap_legacy_oarl_adapter_keys(payload["model_state"]),
        strict=False,
    )
    allowed_missing = {"oarl_core.object_region_norm.weight", "oarl_core.object_region_norm.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
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


def _remap_legacy_oarl_adapter_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Load checkpoints saved before the OARL reasoning core was named explicitly."""
    legacy_prefixes = {
        "object_encoder.": "oarl_core.object_encoder.",
        "graph_encoder.": "oarl_core.graph_encoder.",
        "fusion.": "oarl_core.fusion.",
        "target_head.": "oarl_core.target_head.",
        "global_norm.": "oarl_core.global_norm.",
    }
    if any(key.startswith("oarl_core.") for key in state_dict):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("oarl_adapter."):
            new_key = "oarl_core." + key[len("oarl_adapter.") :]
            remapped[new_key] = value
            continue
        for old_prefix, new_prefix in legacy_prefixes.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                break
        remapped[new_key] = value
    return remapped
