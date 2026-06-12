from oarlvla.reasoning import LogicAwareReasoner
from oarlvla.scene import generate_scene
from oarlvla.states import banana_states


def test_banana_blackened_rotten_and_edible_rules():
    fresh = banana_states(ripeness=0.7, black_spot_ratio=0.08)
    rotten = banana_states(ripeness=0.96, black_spot_ratio=0.2)
    blackened = banana_states(ripeness=0.8, black_spot_ratio=0.5)
    assert fresh["is_blackened"] is False
    assert rotten["is_rotten"] is True
    assert rotten["is_edible"] is False
    assert blackened["is_blackened"] is True


def test_unopened_drink_and_not_empty_bottle_reasoning():
    scene = generate_scene(seed=4)
    reasoner = LogicAwareReasoner()
    unopened = reasoner.reason(scene, "Pick the drink that is unopened.")
    not_empty = reasoner.reason(scene, "Pick the bottle that is not empty.")
    assert unopened.target_id == "soda_can_1"
    assert not_empty.target_id == "bottle_1"

