from __future__ import annotations

from .oarl_core import OARLReasoningCore


# Backward-compatible import name for older scripts/checkpoints.
OARLAdapter = OARLReasoningCore

__all__ = ["OARLAdapter"]
