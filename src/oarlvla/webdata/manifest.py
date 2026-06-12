from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import WebImageRecord


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path: str | Path, records: Iterable[WebImageRecord]) -> None:
    write_jsonl(path, (record.to_dict() for record in records))


def read_manifest(path: str | Path) -> list[WebImageRecord]:
    return [WebImageRecord(**row) for row in read_jsonl(path)]

