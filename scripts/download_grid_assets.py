from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.gridworld.web_assets import download_grid_web_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wikimedia Commons object photos and convert them into transparent grid cutouts.")
    parser.add_argument("--asset-dir", type=Path, default=Path("data/grid_assets"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/grid_asset_raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/grid_assets_manifest.json"))
    parser.add_argument("--candidates-per-query", type=int, default=8)
    parser.add_argument("--sprite-size", type=int, default=192)
    parser.add_argument("--early-stop-score", type=float, default=9.5)
    parser.add_argument("--max-usable-candidates", type=int, default=14)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    report = download_grid_web_assets(
        asset_dir=args.asset_dir,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        candidates_per_query=args.candidates_per_query,
        sprite_size=args.sprite_size,
        force=args.force,
        early_stop_score=args.early_stop_score,
        max_usable_candidates=args.max_usable_candidates,
        verbose=args.verbose,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
