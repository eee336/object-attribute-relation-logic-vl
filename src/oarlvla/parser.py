from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .taxonomy import normalize_term


@dataclass
class ProgramStep:
    op: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        if not self.args:
            return self.op
        arg_text = ", ".join(f"{key}={_format_value(value)}" for key, value in self.args.items())
        return f"{self.op}({arg_text})"


@dataclass
class Program:
    steps: list[ProgramStep]
    task_type: str = "unknown"

    def to_string(self) -> str:
        return "->".join(step.to_string() for step in self.steps)


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


def parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw in {"True", "False"}:
        return raw == "True"
    if raw in {"None", "null"}:
        return None
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_program(program: str, task_type: str = "unknown") -> Program:
    steps: list[ProgramStep] = []
    for token in filter(None, (part.strip() for part in program.split("->"))):
        match = re.fullmatch(r"([a-zA-Z_]+)(?:\((.*)\))?", token)
        if not match:
            raise ValueError(f"Invalid program step: {token}")
        op, raw_args = match.group(1), match.group(2)
        args: dict[str, Any] = {}
        if raw_args:
            for part in re.split(r",\s*", raw_args):
                if not part:
                    continue
                key, value = part.split("=", 1)
                args[key.strip()] = parse_value(value)
        steps.append(ProgramStep(op, args))
    return Program(steps, task_type=task_type)


def ordinal_to_int(text: str) -> int | None:
    text = text.lower().strip()
    words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
    }
    if text in words:
        return words[text]
    match = re.fullmatch(r"(\d+)(st|nd|rd|th)?", text)
    return int(match.group(1)) if match else None


def parse_instruction(instruction: str) -> Program:
    text = re.sub(r"\s+", " ", instruction.lower().strip().rstrip("."))
    text = text.replace("trash bin", "trash_bin").replace("water bottle", "water_bottle")
    text = text.replace("soda can", "soda_can").replace("juice box", "juice_box")

    if "just put down" in text:
        return Program(
            [ProgramStep("select_from_history", {"event_type": "put_down", "most_recent": True})],
            task_type="history_reference",
        )
    if "moved most recently" in text:
        return Program([ProgramStep("select_from_history", {"most_recent": True})], task_type="history_reference")

    if "pair of shoes" in text:
        steps = [ProgramStep("filter_group", {"group_type": "pair_of_shoes"})]
        if "farthest" in text or "furthest" in text:
            steps.append(ProgramStep("argmax", {"attribute": "distance_to_origin"}))
        elif "nearest" in text or "closest" in text:
            steps.append(ProgramStep("argmin", {"attribute": "distance_to_origin"}))
        elif "cleanest" in text or "clean" in text:
            steps.append(ProgramStep("argmax", {"attribute": "cleanliness"}))
        else:
            steps.append(ProgramStep("select_best"))
        return Program(steps, task_type="group_grounding")

    if "suitable for drinking coffee" in text or "suitable for coffee" in text:
        return Program(
            [
                ProgramStep("filter", {"super_category": "drinkware"}),
                ProgramStep("filter_affordance", {"name": "coffee_suitable"}),
                ProgramStep("argmax", {"attribute": "cleanliness"}),
            ],
            task_type="affordance",
        )
    if "edible fruit" in text or "can eat" in text:
        return Program(
            [
                ProgramStep("filter", {"super_category": "fruit"}),
                ProgramStep("filter_state", {"key": "is_edible", "value": True}),
                ProgramStep("argmax", {"attribute": "ripeness"}),
            ],
            task_type="category_taxonomy",
        )

    nth = re.search(r"(?:pick the )?(?P<n>\d+(?:st|nd|rd|th)|first|second|third|fourth) (?P<cat>[a-z_]+) from (?P<direction>left to right|right to left)", text)
    if nth:
        n = ordinal_to_int(nth.group("n")) or 1
        direction = "left_to_right" if nth.group("direction") == "left to right" else "right_to_left"
        return Program(
            [
                ProgramStep("filter", {"category": normalize_term(nth.group("cat"))}),
                ProgramStep("nth", {"n": n, "direction": direction}),
            ],
            task_type="ordinal_relation",
        )

    left_right = re.search(r"pick the (?P<a>[a-z_]+) (?P<rel>left|right) of the (?P<b>[a-z_]+)", text)
    if left_right:
        op = "left_of" if left_right.group("rel") == "left" else "right_of"
        return Program(
            [
                ProgramStep("filter", {"category": normalize_term(left_right.group("a"))}),
                ProgramStep("relation", {"op": op, "ref_category": normalize_term(left_right.group("b"))}),
                ProgramStep("select_best"),
            ],
            task_type="spatial_relation",
        )

    near_far = re.search(r"pick the (?P<a>[a-z_]+) (?P<rel>nearest|closest|farthest|furthest) (?:to|from) the (?P<b>[a-z_]+)", text)
    if near_far:
        op = "nearest_to" if near_far.group("rel") in {"nearest", "closest"} else "farthest_from"
        category = normalize_term(near_far.group("a"))
        filter_key = "super_category" if category in {"fruit", "drink", "drinkware", "footwear"} else "category"
        return Program(
            [
                ProgramStep("filter", {filter_key: category}),
                ProgramStep(op, {"ref_category": normalize_term(near_far.group("b"))}),
            ],
            task_type="spatial_relation",
        )

    between_match = re.search(r"pick the (?P<a>[a-z_]+) between the (?P<b>[a-z_]+) and the (?P<c>[a-z_]+)", text)
    if between_match:
        return Program(
            [
                ProgramStep("filter", {"category": normalize_term(between_match.group("a"))}),
                ProgramStep(
                    "between",
                    {
                        "ref_category_a": normalize_term(between_match.group("b")),
                        "ref_category_b": normalize_term(between_match.group("c")),
                    },
                ),
                ProgramStep("select_best"),
            ],
            task_type="spatial_relation",
        )

    if "leftmost" in text or "rightmost" in text:
        direction = "left_to_right" if "leftmost" in text else "right_to_left"
        cat = _last_known_term(text)
        return Program(
            [
                ProgramStep("filter", {"category": normalize_term(cat)}),
                ProgramStep("nth", {"n": 1, "direction": direction}),
            ],
            task_type="spatial_relation",
        )

    if "not near" in text:
        category = "fruit" if "fruit" in text else _last_known_term(text)
        filter_key = "super_category" if category in {"fruit", "drink", "drinkware"} else "category"
        return Program(
            [
                ProgramStep("filter", {filter_key: normalize_term(category)}),
                ProgramStep("exclude_near", {"ref_category": "trash_bin"}),
                ProgramStep("select_best"),
            ],
            task_type="negation",
        )

    if "not turned black" in text or "not blackened" in text or "has not turned black" in text:
        return Program(
            [
                ProgramStep("filter", {"category": "banana"}),
                ProgramStep("filter_state", {"key": "is_blackened", "value": False}),
                ProgramStep("argmin", {"attribute": "black_spot_ratio"}),
            ],
            task_type="state_filtering",
        )
    if "blackened banana" in text:
        return Program(
            [
                ProgramStep("filter", {"category": "banana"}),
                ProgramStep("filter_state", {"key": "is_blackened", "value": True}),
                ProgramStep("argmax", {"attribute": "black_spot_ratio"}),
            ],
            task_type="state_filtering",
        )
    if "not rotten" in text:
        return Program(
            [
                ProgramStep("filter", {"category": "banana"}),
                ProgramStep("filter_state", {"key": "is_rotten", "value": False}),
                ProgramStep("argmin", {"attribute": "black_spot_ratio"}),
            ],
            task_type="state_filtering",
        )
    if "unopened" in text or "not opened" in text:
        category = "drink" if "drink" in text or "water" in text else _last_known_term(text)
        filter_key = "super_category" if category == "drink" else "category"
        return Program(
            [
                ProgramStep("filter", {filter_key: normalize_term(category)}),
                ProgramStep("filter_state", {"key": "is_opened", "value": False}),
                ProgramStep("argmax", {"attribute": "fill_level"}),
            ],
            task_type="state_filtering" if "not opened" not in text else "negation",
        )
    if "not empty" in text:
        category = "bottle" if "bottle" in text else "drink"
        filter_key = "category" if category == "bottle" else "super_category"
        return Program(
            [
                ProgramStep("filter", {filter_key: category}),
                ProgramStep("filter_state", {"key": "is_empty", "value": False}),
                ProgramStep("argmax", {"attribute": "fill_level"}),
            ],
            task_type="negation" if "not empty" in text else "state_filtering",
        )

    comparisons = {
        "largest": ("argmax", "volume_ml", "size"),
        "smallest": ("argmin", "volume_ml", "size"),
        "cleanest": ("argmax", "cleanliness", None),
        "dirtiest": ("argmin", "cleanliness", None),
        "fullest": ("argmax", "fill_level", None),
        "emptiest": ("argmin", "fill_level", None),
        "brightest": ("argmax", "brightness", None),
        "darkest": ("argmin", "brightness", None),
    }
    for word, (op, attribute, fallback) in comparisons.items():
        if word in text:
            category = _last_known_term(text)
            category = "drink" if category in {"drink", "drinks"} else category
            category = "drinkware" if category in {"cup", "cups", "drinkware"} and attribute == "cleanliness" else category
            filter_key = "super_category" if category in {"drink", "fruit", "drinkware"} else "category"
            args: dict[str, Any] = {"attribute": attribute}
            if fallback:
                args["fallback"] = fallback
            return Program(
                [ProgramStep("filter", {filter_key: normalize_term(category)}), ProgramStep(op, args)],
                task_type="attribute_comparison" if word not in {"fullest", "emptiest"} else "state_filtering",
            )

    if "clean cup" in text:
        return Program(
            [
                ProgramStep("filter", {"super_category": "drinkware"}),
                ProgramStep("filter_threshold", {"attribute": "cleanliness", "op": ">=", "value": 0.6}),
                ProgramStep("argmax", {"attribute": "cleanliness"}),
            ],
            task_type="state_filtering",
        )
    if "dirty cup" in text:
        return Program(
            [
                ProgramStep("filter", {"super_category": "drinkware"}),
                ProgramStep("filter_threshold", {"attribute": "cleanliness", "op": "<", "value": 0.4}),
                ProgramStep("argmin", {"attribute": "cleanliness"}),
            ],
            task_type="state_filtering",
        )

    cat = _last_known_term(text)
    return Program([ProgramStep("filter", {"category": normalize_term(cat)}), ProgramStep("select_best")])


def _last_known_term(text: str) -> str:
    terms = [
        "water_bottle",
        "soda_can",
        "juice_box",
        "trash_bin",
        "banana",
        "apple",
        "orange",
        "bottle",
        "drink",
        "drinkware",
        "fruit",
        "cup",
        "mug",
        "shoe",
        "spoon",
        "bowl",
        "book",
        "remote",
        "object",
    ]
    for term in terms:
        if re.search(rf"\b{re.escape(term)}s?\b", text):
            return term
    return "object"
