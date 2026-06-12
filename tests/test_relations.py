from oarlvla.relations import filter_objects, select_nth
from oarlvla.reasoning import LogicAwareReasoner
from oarlvla.scene import generate_scene


def test_nth_left_right_relation_helpers():
    scene = generate_scene(seed=5)
    bananas = filter_objects(scene, category="banana")
    assert select_nth(bananas, 1, "left_to_right").id == "banana_1"
    assert select_nth(bananas, 1, "right_to_left").id == "banana_3"


def test_spoon_left_of_cup_relation_reasoning():
    scene = generate_scene(seed=5)
    result = LogicAwareReasoner().reason(scene, "Pick the spoon left of the cup.")
    assert result.target_id == "spoon_1"

