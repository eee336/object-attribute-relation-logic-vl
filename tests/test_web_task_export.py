from pathlib import Path

from oarlvla.webdata.dataset_builder import build_web_dataset, export_review_html, summarize_manifest


def test_web_dataset_builder_exports_tasks_sft_preferences_and_review(tmp_path: Path):
    output_dir = tmp_path / "web_dataset"
    report = build_web_dataset(
        source_name="local",
        input_dir="tests/fixtures/images",
        queries_path="configs/web_queries.yaml",
        max_per_query=2,
        output_dir=output_dir,
        mode="metadata_only",
    )
    assert report["records"] == 1
    assert report["weak_grounding_tasks"] == 1
    assert (tmp_path / "web_manifest.jsonl").exists()
    assert (tmp_path / "web_tasks.jsonl").exists()
    assert (tmp_path / "oarlvla_web_sft.jsonl").exists()
    assert (tmp_path / "oarlvla_web_preferences.jsonl").exists()
    summary = summarize_manifest(tmp_path / "web_manifest.jsonl", tmp_path / "annotations")
    assert summary["total_images"] == 1
    html = export_review_html(tmp_path / "web_manifest.jsonl", tmp_path / "review.html", tmp_path / "annotations")
    assert html.exists()

