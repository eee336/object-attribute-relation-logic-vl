from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.webdata.annotators import save_manual_annotation


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a manual annotation update to an image annotation JSON.")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--target-id", default=None)
    parser.add_argument("--target-type", default="unknown")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    path = save_manual_annotation(
        args.annotation,
        {
            "category": args.category,
            "target_id": args.target_id,
            "target_type": args.target_type,
            "notes": args.notes,
            "source": "manual",
        },
    )
    print(json.dumps({"updated": str(path)}, indent=2))


if __name__ == "__main__":
    main()

