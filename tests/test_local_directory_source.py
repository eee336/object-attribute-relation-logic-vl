from pathlib import Path

from oarlvla.webdata.sources import LocalDirectorySource


def test_local_directory_source_imports_fixture(tmp_path: Path):
    source = LocalDirectorySource("tests/fixtures/images")
    results = source.search("fresh banana", max_results=10)
    assert results
    record = source.download(results[0], tmp_path / "web_dataset")
    assert record.width == 8
    assert record.height == 8
    assert Path(record.local_path).exists()

