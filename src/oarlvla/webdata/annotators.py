from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_manual_annotation(annotation_path: str | Path, updates: dict[str, Any]) -> Path:
    path = Path(annotation_path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault("manual_updates", []).append(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

