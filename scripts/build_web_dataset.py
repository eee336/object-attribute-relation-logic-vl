from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.webdata.dataset_builder import build_web_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local", "wikimedia"], required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=Path("configs/web_queries.yaml"))
    parser.add_argument("--max-per-query", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("data/web_dataset"))
    parser.add_argument("--mode", choices=["metadata_only", "heuristic", "model_assisted"], default="metadata_only")
    args = parser.parse_args()
    report = build_web_dataset(
        source_name=args.source,
        input_dir=args.input_dir,
        queries_path=args.queries,
        max_per_query=args.max_per_query,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

