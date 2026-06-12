from oarlvla.instruction import TASK_TYPES, generate_instruction
from oarlvla.scene import generate_scene


def test_instruction_generator_produces_valid_targets_for_each_task_type():
    scene = generate_scene(seed=8)
    for idx, task_type in enumerate(TASK_TYPES):
        example = generate_instruction(scene, task_type, seed=idx)
        assert example.instruction
        assert example.program
        assert example.target_id is not None

