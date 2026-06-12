from oarlvla.taxonomy import categories_for_super_category, get_super_categories, is_category


def test_taxonomy_maps_open_vocab_to_categories():
    assert "fruit" in get_super_categories("banana")
    assert "bottle" in categories_for_super_category("drink")
    assert "juice_box" in categories_for_super_category("drink")
    assert is_category("mug", "drinkware")

