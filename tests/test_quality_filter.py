from oarlvla.webdata.filters import QualityFilter
from oarlvla.webdata.sources import LocalDirectorySource


def test_quality_filter_scores_openable_image(tmp_path):
    source = LocalDirectorySource("tests/fixtures/images")
    record = source.download(source.search("fresh banana", 1)[0], tmp_path / "web_dataset")
    score = QualityFilter().score(record)
    assert 0.0 < score <= 1.0

