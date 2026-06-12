from oarlvla.reasoning import LogicAwareReasoner
from oarlvla.scene import generate_scene


def test_farthest_pair_of_shoes_selects_group():
    scene = generate_scene(seed=2)
    result = LogicAwareReasoner().reason(scene, "Pick the farthest pair of shoes.")
    assert result.target_type == "group"
    assert result.target_id == "shoe_pair_1"

