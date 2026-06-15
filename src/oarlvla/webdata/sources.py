from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ModuleNotFoundError:
    requests = None

from .dedup import sha256_file
from .downloader import HTTP_USER_AGENT, download_url
from .image_utils import read_image_size
from .schemas import WebImageRecord


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".ppm"}
WIKIMEDIA_RASTER_MIMES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@dataclass
class ImageSearchResult:
    result_id: str
    url: str
    source_url: str
    query: str
    license: str | None = None
    author: str | None = None
    local_path: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class ImageSource:
    name: str = "base"

    def search(self, query: str, max_results: int) -> list[ImageSearchResult]:
        raise NotImplementedError

    def download(self, result: ImageSearchResult, output_dir: Path) -> WebImageRecord:
        raise NotImplementedError


class LocalDirectorySource(ImageSource):
    name = "local"

    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)

    def search(self, query: str, max_results: int) -> list[ImageSearchResult]:
        if not self.input_dir.exists():
            return []
        files = sorted(path for path in self.input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if max_results > 0:
            files = files[:max_results]
        return [
            ImageSearchResult(
                result_id=path.stem,
                url=str(path),
                source_url=str(path.resolve()),
                query=query,
                license="user-provided",
                author=None,
                local_path=str(path),
                raw_metadata={"filename": path.name},
            )
            for path in files
        ]

    def download(self, result: ImageSearchResult, output_dir: Path) -> WebImageRecord:
        source_path = Path(result.local_path or result.url)
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(source_path)
        suffix = source_path.suffix.lower() or ".jpg"
        image_id = f"local_{digest[:16]}"
        target_path = image_dir / f"{image_id}{suffix}"
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        width, height = _image_size(target_path)
        return WebImageRecord(
            image_id=image_id,
            local_path=str(target_path),
            source_name=self.name,
            source_url=result.source_url,
            license=result.license,
            author=result.author,
            query=result.query,
            downloaded_at=_now(),
            width=width,
            height=height,
            sha256=digest,
            split="train",
            raw_metadata=result.raw_metadata,
        )


class WikimediaCommonsSource(ImageSource):
    name = "wikimedia"

    def __init__(self, api_url: str = "https://commons.wikimedia.org/w/api.php"):
        self.api_url = api_url

    def search(self, query: str, max_results: int) -> list[ImageSearchResult]:
        if requests is None:
            return []
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": max_results,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 512,
        }
        try:
            response = requests.get(
                self.api_url,
                params=params,
                timeout=15,
                headers={"User-Agent": HTTP_USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []
        pages = data.get("query", {}).get("pages", {})
        results: list[ImageSearchResult] = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime")
            if mime not in WIKIMEDIA_RASTER_MIMES:
                continue
            original_url = info.get("url")
            url = info.get("thumburl") or original_url
            if not url:
                continue
            meta = info.get("extmetadata", {})
            license_name = _meta_value(meta, "LicenseShortName") or _meta_value(meta, "UsageTerms")
            author = _meta_value(meta, "Artist")
            source_url = info.get("descriptionurl") or url
            result_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
            results.append(
                ImageSearchResult(
                    result_id=result_id,
                    url=url,
                    source_url=source_url,
                    query=query,
                    license=license_name,
                    author=author,
                    raw_metadata={"title": page.get("title"), "imageinfo": info, "original_url": original_url},
                )
            )
        return results

    def download(self, result: ImageSearchResult, output_dir: Path) -> WebImageRecord:
        suffix = Path(result.url.split("?")[0]).suffix.lower()
        suffix = suffix if suffix in IMAGE_SUFFIXES else ".jpg"
        image_id = f"wikimedia_{result.result_id}"
        target_path = output_dir / "images" / f"{image_id}{suffix}"
        time.sleep(0.75)
        download_url(result.url, target_path)
        digest = sha256_file(target_path)
        width, height = _image_size(target_path)
        return WebImageRecord(
            image_id=image_id,
            local_path=str(target_path),
            source_name=self.name,
            source_url=result.source_url,
            license=result.license,
            author=result.author,
            query=result.query,
            downloaded_at=_now(),
            width=width,
            height=height,
            sha256=digest,
            split="train",
            raw_metadata=result.raw_metadata,
        )


def make_source(name: str, input_dir: str | Path | None = None) -> ImageSource:
    if name == "local":
        if input_dir is None:
            raise ValueError("--input-dir is required for local source")
        return LocalDirectorySource(input_dir)
    if name == "wikimedia":
        return WikimediaCommonsSource()
    raise ValueError(f"Unsupported source: {name}")


def _image_size(path: Path) -> tuple[int, int]:
    return read_image_size(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _meta_value(meta: dict[str, Any], key: str) -> str | None:
    value = meta.get(key, {})
    if isinstance(value, dict):
        return value.get("value")
    if isinstance(value, str):
        return value
    return None
