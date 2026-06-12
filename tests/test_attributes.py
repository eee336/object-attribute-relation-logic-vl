from oarlvla.reasoning import LogicAwareReasoner
from oarlvla.scene import generate_scene


def test_largest_drink_and_cleanest_cup_are_selected():
    scene = generate_scene(seed=3)
    reasoner = LogicAwareReasoner()
    largest = reasoner.reason(scene, "Pick the largest drink.")
    cleanest = reasoner.reason(scene, "Pick the cleanest cup.")
    assert largest.target_id == "juice_box_1"
    assert cleanest.target_id == "cup_1"

