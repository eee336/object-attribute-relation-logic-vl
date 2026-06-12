from __future__ import annotations

from pathlib import Path

try:
    import requests
except ModuleNotFoundError:
    requests = None


def download_url(url: str, output_path: str | Path, timeout: float = 15.0) -> Path:
    if requests is None:
        raise RuntimeError("requests is not installed; network image download is unavailable")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "OARL-VLA research prototype/0.1"})
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path
