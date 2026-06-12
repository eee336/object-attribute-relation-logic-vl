from __future__ import annotations

from pathlib import Path

from .schemas import WebImageRecord


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_records(records: list[WebImageRecord]) -> tuple[list[WebImageRecord], list[WebImageRecord]]:
    seen: set[str] = set()
    kept: list[WebImageRecord] = []
    duplicates: list[WebImageRecord] = []
    for record in records:
        if record.sha256 in seen:
            duplicates.append(record)
            continue
        seen.add(record.sha256)
        kept.append(record)
    return kept, duplicates

