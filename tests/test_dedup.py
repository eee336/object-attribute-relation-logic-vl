from oarlvla.webdata.dedup import deduplicate_records
from oarlvla.webdata.schemas import WebImageRecord


def _record(image_id: str, sha: str) -> WebImageRecord:
    return WebImageRecord(image_id, "", "local", "", None, None, "", "", 1, 1, sha, "train", {})


def test_sha256_dedup_keeps_first_record():
    kept, duplicates = deduplicate_records([_record("a", "same"), _record("b", "same"), _record("c", "other")])
    assert [r.image_id for r in kept] == ["a", "c"]
    assert [r.image_id for r in duplicates] == ["b"]

