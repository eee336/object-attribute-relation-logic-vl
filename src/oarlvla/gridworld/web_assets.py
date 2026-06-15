from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oarlvla.webdata.downloader import download_url
from oarlvla.webdata.sources import ImageSearchResult, WikimediaCommonsSource

from .sprites import SPRITE_CATEGORIES


GRID_ASSET_QUERIES: dict[str, list[str]] = {
    "apple": ['intitle:"Red Apple" fruit', 'intitle:"Picture of Red Apple"', "red apple fruit"],
    "banana": ['intitle:"Banana on whitebackground"', "banana whitebackground", "banana fruit"],
    "orange": ['intitle:"Orange Fruit Close-up"', "orange fruit", "mandarin orange fruit"],
    "bottle": ['intitle:"Plastic bottle.jpg"', 'intitle:"300ml square cosmetic plastic bottle"', 'intitle:"Empty Plastic Bottle"'],
    "water_bottle": ['intitle:"Metal water bottle"', 'intitle:"Blue Waters bottle no label"', 'intitle:"Bottle of Water"', "water bottle"],
    "soda_can": ['intitle:"Can of Diet Sierra Mist"', "soda can", "beverage can"],
    "juice_box": ['intitle:"Beep drink carton"', "soft drink carton", "juice box drink"],
    "cup": ['intitle:"Cup of tea isolated on white background"', "white cup isolated", "cup isolated"],
    "mug": ['intitle:"IBM merchandising coffee mug"', 'intitle:"WSMR Coffee Mug"', 'intitle:"Coffee mug" white'],
    "shoe": ['intitle:"Reebok Royal Glide Ripple Clip shoe"', "shoe isolated", "black shoe"],
    "spoon": ['intitle:"Spoon (21619616885)"', 'intitle:"One Metal Spoon"', 'intitle:"Metal Chinese spoon - 01"', "metal spoon"],
    "bowl": ['intitle:"Light green ceramic bowl"', 'intitle:"Ceramic bowl"', "ceramic bowl"],
    "trash_bin": ['intitle:"Trash bin in Paris"', 'intitle:"Trash bin at Viborg"', "trash bin", "waste bin"],
    "book": ['intitle:"Bamboo book - closed"', "closed book", "book isolated"],
    "remote": ['intitle:"AverMedia RM-RH Remote control"', "remote control white background", "tv remote"],
}


TITLE_POSITIVES: dict[str, set[str]] = {
    "apple": {"apple", "fruit", "red"},
    "banana": {"banana", "fruit", "whitebackground"},
    "orange": {"orange", "mandarin", "citrus", "fruit"},
    "bottle": {"bottle", "plastic", "green"},
    "water_bottle": {"water", "bottle"},
    "soda_can": {"soda", "can", "beverage", "drink"},
    "juice_box": {"juice", "box", "tetra", "pak", "carton", "package"},
    "cup": {"cup", "tea", "isolated"},
    "mug": {"mug", "coffee"},
    "shoe": {"shoe", "sneaker", "reebok"},
    "spoon": {"spoon", "cutlery"},
    "bowl": {"bowl", "ceramic"},
    "trash_bin": {"trash", "bin", "waste", "garbage", "container"},
    "book": {"book", "closed"},
    "remote": {"remote", "control"},
}


TITLE_NEGATIVES: dict[str, set[str]] = {
    "apple": {"computer", "apple ii", "shop", "tree", "basket"},
    "banana": {"sale", "plantation", "tree"},
    "orange": {"refrigerator", "peaches", "apricots", "pieces", "tree"},
    "bottle": {"bales", "dead plant", "flowers", "booth", "station", "icon", "fly", "insect", "fish"},
    "water_bottle": {"metal bottles", "group", "people", "vortex", "draining", "tornado"},
    "soda_can": {"garden", "flattened", "cans"},
    "juice_box": {"factory", "school", "inventing", "processing", "equipment", "portfolio"},
    "cup": {"measuring"},
    "mug": {"desk", "table", "mugs", "echo", "creamer"},
    "shoe": {"cobbler", "repairing", "workshop", "roundabout"},
    "spoon": {"family", "food", "koshary", "rawon", "chopsticks", "painting", "scissors", "ruler"},
    "bowl": {"museum", "met", "chickpeas"},
    "trash_bin": {"handles"},
    "book": {"shops", "sculpture", "cartoon", "muscle", "radiographic"},
    "remote": {"lights", "piloty"},
}


TITLE_PREFERENCES: dict[str, set[str]] = {
    "apple": {"red apple.jpg"},
    "banana": {"banana on whitebackground"},
    "orange": {"orange fruit close-up"},
    "bottle": {"plastic bottle.jpg", "300ml square cosmetic plastic bottle"},
    "water_bottle": {"metal water bottle", "bottle of water", "blue waters bottle no label"},
    "soda_can": {"can of diet sierra mist"},
    "juice_box": {"beep drink carton"},
    "cup": {"cup of tea isolated on white background"},
    "mug": {"wsmr coffee mug", "ibm merchandising coffee mug", "ceramic white and teal coffee mug"},
    "shoe": {"reebok royal glide ripple clip shoe"},
    "spoon": {"spoon (21619616885)", "one metal spoon"},
    "bowl": {"ceramic bowl", "light green ceramic bowl"},
    "trash_bin": {"trash bin in paris", "trash bin at viborg"},
    "book": {"bamboo book - closed"},
    "remote": {"avermedia rm-rh remote control"},
}


@dataclass
class AssetCandidateReport:
    category: str
    title: str
    query: str
    source_url: str
    image_url: str
    license: str | None
    author: str | None
    raw_path: str
    candidate_path: str
    score: float
    width: int
    height: int
    mask_area_ratio: float
    bbox_area_ratio: float
    edge_foreground_ratio: float
    border_std: float
    used_existing_alpha: bool


def download_grid_web_assets(
    *,
    asset_dir: str | Path,
    raw_dir: str | Path,
    manifest_path: str | Path,
    categories: list[str] | None = None,
    candidates_per_query: int = 8,
    sprite_size: int = 192,
    force: bool = False,
    sleep_seconds: float = 0.35,
    early_stop_score: float = 9.5,
    max_usable_candidates: int = 14,
    verbose: bool = False,
) -> dict[str, Any]:
    asset_dir = Path(asset_dir)
    raw_dir = Path(raw_dir)
    manifest_path = Path(manifest_path)
    asset_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    categories = categories or list(SPRITE_CATEGORIES)
    source = WikimediaCommonsSource()
    selected: list[dict[str, Any]] = []
    failures: dict[str, str] = {}

    for category in categories:
        target_path = asset_dir / f"{category}.png"
        if target_path.exists() and not force:
            selected.append({"category": category, "asset_path": str(target_path), "status": "kept_existing"})
            continue
        best: tuple[float, AssetCandidateReport] | None = None
        seen_urls: set[str] = set()
        usable_candidates = 0
        if verbose:
            print(f"[grid-assets] searching {category}", flush=True)
        for query in GRID_ASSET_QUERIES.get(category, [category]):
            if usable_candidates >= max_usable_candidates:
                break
            for result in source.search(query, candidates_per_query):
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                try:
                    report = _download_and_prepare_candidate(
                        category=category,
                        result=result,
                        raw_dir=raw_dir,
                        sprite_size=sprite_size,
                    )
                except Exception:
                    continue
                usable_candidates += 1
                if best is None or report.score > best[0]:
                    best = (report.score, report)
                    if verbose:
                        print(f"[grid-assets] {category}: best {report.score:.2f} {report.title}", flush=True)
                if report.score >= early_stop_score:
                    break
                time.sleep(sleep_seconds)
            if best is not None and best[0] >= early_stop_score:
                break
        if best is None:
            failures[category] = "No usable Wikimedia Commons candidate found."
            if verbose:
                print(f"[grid-assets] {category}: failed", flush=True)
            continue
        _, report = best
        candidate_path = Path(report.candidate_path)
        target_path.write_bytes(candidate_path.read_bytes())
        selected.append({**asdict(report), "asset_path": str(target_path), "status": "downloaded"})
        if verbose:
            print(f"[grid-assets] {category}: selected {report.title} ({report.score:.2f})", flush=True)

    manifest = {
        "source": "wikimedia_commons",
        "asset_dir": str(asset_dir),
        "raw_dir": str(raw_dir),
        "sprite_size": sprite_size,
        "selected": selected,
        "failures": failures,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def remove_background_and_crop(image, output_size: int = 192, keep_largest_component: bool = True):
    from PIL import Image, ImageFilter, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGBA")
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    used_existing_alpha = bool((alpha < 245).mean() > 0.02)

    if used_existing_alpha:
        mask = alpha > 16
        border_std = _border_std(rgb)
    else:
        mask, border_std = _foreground_mask_from_border(rgb)
    if keep_largest_component:
        mask = _keep_largest_component(mask)

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.1))
    solid_mask = mask_img.point(lambda value: 255 if value > 18 else 0)
    bbox = solid_mask.getbbox()
    if bbox is None:
        raise ValueError("No foreground detected")

    rgba = image.copy()
    rgba.putalpha(mask_img)
    cropped = rgba.crop(bbox)
    cropped_mask = mask_img.crop(bbox)
    cropped.putalpha(cropped_mask)

    canvas = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 0))
    max_side = int(output_size * 0.84)
    cropped.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    x = (output_size - cropped.width) // 2
    y = (output_size - cropped.height) // 2
    canvas.alpha_composite(cropped, (x, y))

    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    mask_area_ratio = float(mask.mean())
    bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, w * h)
    edge_foreground_ratio = _edge_foreground_ratio(mask)
    metrics = {
        "width": int(image.width),
        "height": int(image.height),
        "mask_area_ratio": mask_area_ratio,
        "bbox_area_ratio": float(bbox_area_ratio),
        "edge_foreground_ratio": edge_foreground_ratio,
        "border_std": border_std,
        "used_existing_alpha": used_existing_alpha,
    }
    return canvas, metrics


def _download_and_prepare_candidate(
    *,
    category: str,
    result: ImageSearchResult,
    raw_dir: Path,
    sprite_size: int,
) -> AssetCandidateReport:
    from PIL import Image

    title = str(result.raw_metadata.get("title") or "")
    digest = hashlib.sha1(result.url.encode("utf-8")).hexdigest()[:16]
    suffix = _safe_suffix(result.url)
    category_dir = raw_dir / category
    raw_path = category_dir / f"{digest}{suffix}"
    candidate_path = category_dir / f"{digest}_cutout.png"
    if not raw_path.exists():
        download_url(result.url, raw_path)
    with Image.open(raw_path) as image:
        cutout, metrics = remove_background_and_crop(
            image,
            output_size=sprite_size,
            keep_largest_component=category in {"juice_box", "mug", "trash_bin", "remote"},
        )
    score = _score_candidate(category, title, metrics)
    if not _is_usable_candidate(metrics, score, category):
        raise ValueError(f"Rejected candidate: {title}")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    cutout.save(candidate_path)
    return AssetCandidateReport(
        category=category,
        title=title,
        query=result.query,
        source_url=result.source_url,
        image_url=result.url,
        license=result.license,
        author=result.author,
        raw_path=str(raw_path),
        candidate_path=str(candidate_path),
        score=score,
        width=metrics["width"],
        height=metrics["height"],
        mask_area_ratio=metrics["mask_area_ratio"],
        bbox_area_ratio=metrics["bbox_area_ratio"],
        edge_foreground_ratio=metrics["edge_foreground_ratio"],
        border_std=metrics["border_std"],
        used_existing_alpha=metrics["used_existing_alpha"],
    )


def _foreground_mask_from_border(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    h, w, _ = rgb.shape
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    bg = np.median(border, axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    border_dist = np.sqrt(((border - bg) ** 2).sum(axis=1))
    threshold = float(max(22.0, min(70.0, np.percentile(border_dist, 82) + 18.0)))
    background_like = dist <= threshold
    connected_bg = _flood_connected_background(background_like)
    mask = ~connected_bg
    return mask, _border_std(rgb)


def _flood_connected_background(background_like: np.ndarray) -> np.ndarray:
    h, w = background_like.shape
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if background_like[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if background_like[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and background_like[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))
    return visited


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    best: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            pixels: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                pixels.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            if len(pixels) > len(best):
                best = pixels
    if not best:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for x, y in best:
        out[y, x] = True
    return out


def _score_candidate(category: str, title: str, metrics: dict[str, Any]) -> float:
    title_l = title.lower().replace("_", " ")
    score = 0.0
    for word in TITLE_POSITIVES.get(category, set()):
        if word in title_l:
            score += 2.0
    for word in TITLE_NEGATIVES.get(category, set()):
        if word in title_l:
            score -= 8.0
    for phrase in TITLE_PREFERENCES.get(category, set()):
        if phrase in title_l:
            score += 8.0
    for word in ("isolated", "white background", "whitebackground", "transparent", "clip", "edit"):
        if word in title_l:
            score += 1.5
    if metrics["used_existing_alpha"]:
        score += 2.0
    score += min(2.0, math.log(max(metrics["width"], metrics["height"]), 2) / 6)
    score -= abs(metrics["mask_area_ratio"] - 0.24) * 5.0
    score -= max(0.0, metrics["edge_foreground_ratio"] - 0.04) * 16.0
    score -= max(0.0, metrics["border_std"] - 55.0) / 18.0
    if 0.08 <= metrics["bbox_area_ratio"] <= 0.86:
        score += 1.2
    return score


def _is_usable_candidate(metrics: dict[str, Any], score: float, category: str | None = None) -> bool:
    if metrics["width"] < 120 or metrics["height"] < 120:
        return False
    if not 0.015 <= metrics["mask_area_ratio"] <= 0.88:
        return False
    if not 0.035 <= metrics["bbox_area_ratio"] <= 0.96:
        if not (category in {"water_bottle", "soda_can", "juice_box", "trash_bin", "mug"} and score > 2.0 and metrics["bbox_area_ratio"] <= 1.0):
            return False
    if metrics["edge_foreground_ratio"] > 0.42:
        if not (category in {"water_bottle", "soda_can", "juice_box", "trash_bin", "mug"} and score > 2.0 and metrics["edge_foreground_ratio"] <= 0.72):
            return False
    return score > -4.5


def _safe_suffix(url: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"


def _border_std(rgb: np.ndarray) -> float:
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    return float(border.std())


def _edge_foreground_ratio(mask: np.ndarray) -> float:
    edge = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]], axis=0)
    return float(edge.mean())
