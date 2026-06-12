from oarlvla.webdata.manifest import read_manifest, write_manifest
from oarlvla.webdata.schemas import WebImageRecord


def test_web_manifest_schema_roundtrip(tmp_path):
    record = WebImageRecord(
        image_id="img1",
        local_path="image.ppm",
        source_name="local",
        source_url="/tmp/image.ppm",
        license="user-provided",
        author=None,
        query="fresh banana",
        downloaded_at="2026-01-01T00:00:00Z",
        width=8,
        height=8,
        sha256="abc",
        split="train",
        raw_metadata={"filename": "image.ppm"},
    )
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [record])
    loaded = read_manifest(path)
    assert loaded[0].image_id == "img1"
    assert loaded[0].license == "user-provided"

