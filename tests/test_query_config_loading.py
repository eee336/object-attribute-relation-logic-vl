from pathlib import Path

from oarlvla.webdata.dataset_builder import flatten_queries, load_query_plan


def test_query_config_loading():
    path = Path("configs/web_queries.yaml")
    plan = load_query_plan(path)
    queries = flatten_queries(plan)
    assert "state_filtering" in plan
    assert any("banana" in query for _, _, query in queries)

