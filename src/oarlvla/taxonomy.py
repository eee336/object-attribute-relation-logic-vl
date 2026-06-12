from __future__ import annotations

CATEGORY_TAXONOMY: dict[str, list[str]] = {
    "apple": ["fruit", "food"],
    "banana": ["fruit", "food"],
    "orange": ["fruit", "food"],
    "bottle": ["drink", "container"],
    "water_bottle": ["drink", "container"],
    "can": ["drink", "container"],
    "soda_can": ["drink", "container"],
    "juice_box": ["drink", "container"],
    "cup": ["container", "drinkware"],
    "mug": ["container", "drinkware"],
    "bowl": ["container"],
    "shoe": ["footwear"],
    "spoon": ["utensil"],
    "trash_bin": ["container", "waste_related"],
    "book": ["readable_object"],
    "remote": ["electronics"],
}

ALIASES: dict[str, str] = {
    "beverage": "drink",
    "beverages": "drink",
    "drinks": "drink",
    "fruit": "fruit",
    "fruits": "fruit",
    "shoe": "shoe",
    "shoes": "shoe",
    "pair of shoes": "pair_of_shoes",
    "cup": "drinkware",
    "cups": "drinkware",
    "drinkware": "drinkware",
    "coffee cup": "drinkware",
    "coffee mug": "mug",
    "water": "water_bottle",
}


def get_super_categories(category: str) -> list[str]:
    return CATEGORY_TAXONOMY.get(category, [])


def normalize_term(term: str) -> str:
    cleaned = term.lower().strip().replace("-", "_").replace(" ", "_")
    phrase = term.lower().strip().replace("-", " ")
    if phrase in ALIASES:
        return ALIASES[phrase]
    return ALIASES.get(cleaned, cleaned)


def is_category(category: str, query: str) -> bool:
    query_norm = normalize_term(query)
    if category == query_norm:
        return True
    return query_norm in CATEGORY_TAXONOMY.get(category, [])


def categories_for_super_category(super_category: str) -> list[str]:
    super_category = normalize_term(super_category)
    return [
        category
        for category, supers in CATEGORY_TAXONOMY.items()
        if super_category in supers or category == super_category
    ]


def is_drink(category: str, super_categories: list[str] | None = None) -> bool:
    return "drink" in (super_categories or get_super_categories(category))


def is_fruit(category: str, super_categories: list[str] | None = None) -> bool:
    return "fruit" in (super_categories or get_super_categories(category))

