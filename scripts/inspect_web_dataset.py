from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.webdata.dataset_builder import export_review_html, summarize_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/web_manifest.jsonl"))
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--export-review-html", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize_manifest(args.manifest, args.annotations_dir)
    print(f"Total images: {summary['total_images']}")
    print(f"Valid images: {summary['valid_images']}")
    print(f"Rejected images: {summary['rejected_images']}")
    print(f"Average quality score: {summary['average_quality_score']:.3f}")
    print(f"By source: {json.dumps(summary['by_source'], ensure_ascii=False)}")
    print(f"By task type: {json.dumps(summary['by_task_type'], ensure_ascii=False)}")
    if args.export_review_html:
        path = export_review_html(args.manifest, args.export_review_html, args.annotations_dir)
        print(f"Review HTML: {path}")


if __name__ == "__main__":
    main()

