from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


RULES = {
    "state_filtering": {
        "blackened_banana": [
            "banana with black spots on table",
            "overripe banana next to fresh banana",
            "fresh and rotten bananas side by side",
        ]
    },
    "group_grounding": {
        "pair_of_shoes": [
            "multiple pairs of shoes by the door",
            "two pairs of sneakers on floor",
            "clean and dirty pairs of shoes on floor",
        ]
    },
    "affordance": {
        "coffee_suitable_cup": [
            "ceramic coffee mug on table",
            "paper coffee cup on desk",
            "mug and glass cup on kitchen counter",
        ]
    },
    "attribute_comparison": {
        "largest_drink": [
            "small and large drink bottles side by side",
            "assorted beverages different sizes on table",
        ]
    },
}


def _simple_yaml_dump(data) -> str:
    lines = []
    for top_key, mapping in data.items():
        lines.append(f"{top_key}:")
        for key, values in mapping.items():
            lines.append(f"  {key}:")
            for value in values:
                lines.append(f"    - \"{value}\"")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-report", type=Path, default=Path("outputs/benchmark_results.json"))
    parser.add_argument("--existing-queries", type=Path, default=Path("configs/web_queries.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/suggested_queries.yaml"))
    args = parser.parse_args()
    report = json.loads(args.benchmark_report.read_text(encoding="utf-8")) if args.benchmark_report.exists() else {}
    if yaml is not None and args.existing_queries.exists():
        existing = yaml.safe_load(args.existing_queries.read_text(encoding="utf-8"))
    else:
        existing = {}
    method = report.get("methods", {}).get("OARL-VLA Logic Reasoner", {})
    by_task = method.get("by_task", {})
    suggestions = {"suggested_queries": {}}
    for task_type, intents in RULES.items():
        accuracy = by_task.get(task_type, {}).get("accuracy", 0.0)
        sample_count = by_task.get(task_type, {}).get("n", 0)
        if accuracy < 0.85 or sample_count < 20:
            for intent, queries in intents.items():
                existing_queries = str(existing.get(task_type, {}).get(intent, {}))
                novel = [q for q in queries if q not in existing_queries]
                if novel:
                    suggestions["suggested_queries"][intent] = novel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        text = yaml.safe_dump(suggestions, sort_keys=False)
    else:
        text = _simple_yaml_dump(suggestions)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote query suggestions to {args.output}")


if __name__ == "__main__":
    main()
