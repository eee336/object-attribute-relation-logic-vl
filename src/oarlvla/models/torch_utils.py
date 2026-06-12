from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for the trainable OARL-VLA model. "
            "Install it with `pip install torch` or `pip install -r requirements.txt`."
        ) from exc
    return torch, nn

