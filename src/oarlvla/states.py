from __future__ import annotations

from typing import Any

from .objects import ObjectInstance


def banana_states(ripeness: float, black_spot_ratio: float) -> dict[str, Any]:
    is_blackened = black_spot_ratio > 0.35
    is_rotten = black_spot_ratio > 0.65 or ripeness > 0.95
    return {
        "is_blackened": is_blackened,
        "is_rotten": is_rotten,
        "is_edible": not is_rotten,
    }


def drink_states(is_opened: bool, fill_level: float) -> dict[str, Any]:
    return {
        "is_opened": is_opened,
        "fill_level": fill_level,
        "is_empty": fill_level <= 0.05,
    }


def cup_states(is_broken: bool = False) -> dict[str, Any]:
    return {
        "is_broken": is_broken,
        "is_usable": not is_broken,
    }


def shoe_states(is_wearable: bool = True) -> dict[str, Any]:
    return {"is_wearable": is_wearable}


def is_state(obj: ObjectInstance, key: str, expected: Any) -> bool:
    if key in obj.states:
        return obj.states[key] == expected
    if key in obj.attributes:
        return obj.attributes[key] == expected
    return False


def coffee_suitable(obj: ObjectInstance) -> bool:
    return obj.category in {"mug", "cup"} and not obj.states.get("is_broken", False)


def edible(obj: ObjectInstance) -> bool:
    return obj.states.get("is_edible", False)

