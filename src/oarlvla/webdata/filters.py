from __future__ import annotations

from pathlib import Path

from .image_utils import read_image_size
from .schemas import ImageAnnotationBundle, WebImageRecord


class QualityFilter:
    def score(self, record: WebImageRecord, annotations: ImageAnnotationBundle | dict | None = None) -> float:
        score = 0.0
        path = Path(record.local_path)
        width, height = read_image_size(path)
        if width <= 0 or height <= 0:
            return 0.0
        score += 0.2
        if width >= 224 and height >= 224:
            score += 0.2
        elif width >= 32 and height >= 32:
            score += 0.1
        score += min(0.15, (width * height) / (512 * 512) * 0.15)
        score += 0.05
        if record.license:
            score += 0.1
        query = record.query.lower()
        ann_text = str(annotations.to_dict() if isinstance(annotations, ImageAnnotationBundle) else annotations or {}).lower()
        if any(token in ann_text for token in query.split()[:4]):
            score += 0.1
        if annotations:
            candidate_tasks = annotations.candidate_tasks if isinstance(annotations, ImageAnnotationBundle) else annotations.get("candidate_tasks", [])
            if candidate_tasks:
                score += 0.1
        sensitive_terms = ["face", "child", "passport", "id card", "license plate", "nsfw", "violence"]
        if any(term in query for term in sensitive_terms):
            score -= 0.4
        return max(0.0, min(1.0, score))
