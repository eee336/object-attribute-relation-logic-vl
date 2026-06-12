from oarlvla.webdata.pseudo_labeler import PseudoLabeler, infer_task_from_query
from oarlvla.webdata.schemas import WebImageRecord


def test_pseudo_labeler_generates_weak_label_from_query():
    spec = infer_task_from_query("yellow banana no black spots on table")
    assert spec.task_type == "state_filtering"
    assert spec.states["is_blackened"] is False
    record = WebImageRecord(
        image_id="img",
        local_path="tests/fixtures/images/fresh_banana.ppm",
        source_name="local",
        source_url="tests/fixtures/images/fresh_banana.ppm",
        license="user-provided",
        author=None,
        query="yellow banana no black spots on table",
        downloaded_at="now",
        width=8,
        height=8,
        sha256="abc",
        split="train",
        raw_metadata={},
    )
    bundle = PseudoLabeler().label(record, mode="heuristic")
    assert bundle.candidate_tasks[0].target_id is None
    assert bundle.pseudo_labels[0]["label_quality"] == "weak"

