from oarlvla.parser import parse_program
from oarlvla.reasoning import LogicAwareReasoner, ProgramExecutor
from oarlvla.scene import generate_scene


def test_negation_and_history_reference():
    scene = generate_scene(seed=6)
    reasoner = LogicAwareReasoner()
    negation = reasoner.reason(scene, "Pick the banana that is not blackened.")
    history = reasoner.reason(scene, "Pick the object I just put down.")
    assert negation.target_id == "banana_1"
    assert history.target_id == "remote_1"


def test_program_executor_runs_parsed_program():
    scene = generate_scene(seed=7)
    program = parse_program("filter(category='banana')->filter_state(key='is_blackened', value=False)->argmin(attribute='black_spot_ratio')")
    result = ProgramExecutor().execute(scene, program)
    assert result.target.id == "banana_1"

