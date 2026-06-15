from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


SPRITE_CATEGORIES = [
    "apple",
    "banana",
    "orange",
    "bottle",
    "water_bottle",
    "soda_can",
    "juice_box",
    "cup",
    "mug",
    "shoe",
    "spoon",
    "bowl",
    "trash_bin",
    "book",
    "remote",
]

SPRITE_COLORS: dict[str, str] = {
    "apple": "#d1495b",
    "banana": "#f2c84b",
    "orange": "#f28e2b",
    "bottle": "#59a14f",
    "water_bottle": "#76b7b2",
    "soda_can": "#e15759",
    "juice_box": "#4e79a7",
    "cup": "#f7f7f7",
    "mug": "#c44e52",
    "shoe": "#242424",
    "spoon": "#b8b8b8",
    "bowl": "#f1ce63",
    "trash_bin": "#7f7f7f",
    "book": "#4e79a7",
    "remote": "#111111",
}


def ensure_sprite_assets(asset_dir: str | Path, sprite_size: int = 128) -> dict[str, Path]:
    """Generate transparent household-object cutouts for stage-1 grid scenes.

    These are intentionally generated assets rather than text labels. They are
    lightweight, license-clean, and visually closer to real object silhouettes.
    """

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ModuleNotFoundError:
        return paths
    for category in SPRITE_CATEGORIES:
        path = asset_dir / f"{category}.png"
        if not path.exists():
            image = _make_asset(category, sprite_size, Image, ImageDraw, ImageFilter)
            image.save(path)
        paths[category] = path
    return paths


def _make_asset(category: str, size: int, Image, ImageDraw, ImageFilter):
    scale = 4
    canvas_size = size * scale
    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    painter = _PAINTERS.get(category, _draw_generic)
    painter(draw, canvas_size)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.15 * scale))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _xy(points, s: int):
    return [(int(x * s), int(y * s)) for x, y in points]


def _bbox(x1: float, y1: float, x2: float, y2: float, s: int):
    return (int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s))


def _draw_apple(draw, s: int) -> None:
    draw.ellipse(_bbox(0.22, 0.25, 0.78, 0.82, s), fill="#d8344f", outline="#7d1123", width=s // 42)
    draw.ellipse(_bbox(0.30, 0.18, 0.52, 0.55, s), fill="#d8344f", outline="#7d1123", width=s // 46)
    draw.ellipse(_bbox(0.46, 0.18, 0.70, 0.55, s), fill="#d8344f", outline="#7d1123", width=s // 46)
    draw.line(_xy([(0.50, 0.24), (0.55, 0.10)], s), fill="#5b351b", width=s // 24)
    draw.ellipse(_bbox(0.55, 0.10, 0.76, 0.22, s), fill="#3f8f39", outline="#1f5c21", width=s // 56)
    draw.arc(_bbox(0.30, 0.30, 0.62, 0.70, s), 210, 285, fill="#ff9aa7", width=s // 42)


def _draw_banana(draw, s: int) -> None:
    outer = _xy([(0.16, 0.62), (0.27, 0.75), (0.47, 0.79), (0.70, 0.68), (0.86, 0.43)], s)
    inner = _xy([(0.18, 0.52), (0.34, 0.59), (0.52, 0.59), (0.70, 0.49), (0.82, 0.32)], s)
    draw.line(outer, fill="#b47a12", width=s // 9, joint="curve")
    draw.line(outer, fill="#f2c84b", width=s // 12, joint="curve")
    draw.line(inner, fill="#fff2a0", width=s // 28, joint="curve")
    draw.ellipse(_bbox(0.11, 0.57, 0.21, 0.67, s), fill="#6b3d12")
    draw.ellipse(_bbox(0.80, 0.25, 0.90, 0.36, s), fill="#6b3d12")


def _draw_orange(draw, s: int) -> None:
    draw.ellipse(_bbox(0.20, 0.20, 0.82, 0.82, s), fill="#f28e2b", outline="#944f08", width=s // 38)
    for x, y in [(0.38, 0.36), (0.62, 0.44), (0.48, 0.62), (0.32, 0.55), (0.65, 0.66)]:
        draw.ellipse(_bbox(x - 0.018, y - 0.018, x + 0.018, y + 0.018, s), fill="#ffbf66")
    draw.ellipse(_bbox(0.48, 0.15, 0.58, 0.25, s), fill="#417e2b")


def _draw_bottle(draw, s: int) -> None:
    draw.rounded_rectangle(_bbox(0.34, 0.24, 0.66, 0.84, s), radius=s // 13, fill="#5fac5b", outline="#225a2d", width=s // 40)
    draw.rectangle(_bbox(0.42, 0.12, 0.58, 0.28, s), fill="#4a9347", outline="#225a2d", width=s // 48)
    draw.rectangle(_bbox(0.39, 0.08, 0.61, 0.13, s), fill="#1f6d34", outline="#14431f", width=s // 56)
    draw.rounded_rectangle(_bbox(0.37, 0.48, 0.63, 0.66, s), radius=s // 34, fill="#f5f0d0", outline="#d5c58e", width=s // 64)
    draw.line(_xy([(0.42, 0.30), (0.42, 0.80)], s), fill="#b8e1b8", width=s // 64)


def _draw_water_bottle(draw, s: int) -> None:
    draw.rounded_rectangle(_bbox(0.35, 0.20, 0.65, 0.86, s), radius=s // 12, fill="#bfe8ef", outline="#478a98", width=s // 40)
    draw.rectangle(_bbox(0.43, 0.11, 0.57, 0.23, s), fill="#9fd4df", outline="#478a98", width=s // 52)
    draw.rectangle(_bbox(0.40, 0.07, 0.60, 0.12, s), fill="#2f8fa8", outline="#1d6072", width=s // 58)
    draw.rectangle(_bbox(0.37, 0.56, 0.63, 0.82, s), fill="#6bbfd1")
    draw.rounded_rectangle(_bbox(0.38, 0.36, 0.62, 0.50, s), radius=s // 42, fill="#ffffff", outline="#84cbd8", width=s // 70)


def _draw_soda_can(draw, s: int) -> None:
    draw.rounded_rectangle(_bbox(0.31, 0.18, 0.69, 0.84, s), radius=s // 9, fill="#e15759", outline="#8a1e24", width=s // 38)
    draw.ellipse(_bbox(0.31, 0.14, 0.69, 0.28, s), fill="#f07375", outline="#8a1e24", width=s // 48)
    draw.ellipse(_bbox(0.31, 0.74, 0.69, 0.88, s), fill="#bc3339", outline="#8a1e24", width=s // 48)
    draw.rectangle(_bbox(0.34, 0.43, 0.66, 0.57, s), fill="#ffffff")
    draw.ellipse(_bbox(0.45, 0.17, 0.56, 0.21, s), fill="#5e5e5e")


def _draw_juice_box(draw, s: int) -> None:
    draw.polygon(_xy([(0.28, 0.22), (0.58, 0.14), (0.74, 0.28), (0.44, 0.36)], s), fill="#6da1dc", outline="#1e4c83")
    draw.polygon(_xy([(0.28, 0.22), (0.44, 0.36), (0.44, 0.84), (0.28, 0.70)], s), fill="#2d6cae", outline="#1e4c83")
    draw.polygon(_xy([(0.44, 0.36), (0.74, 0.28), (0.74, 0.76), (0.44, 0.84)], s), fill="#4e79a7", outline="#1e4c83")
    draw.rectangle(_bbox(0.50, 0.48, 0.68, 0.66, s), fill="#fff0b3", outline="#d8b34c")
    draw.line(_xy([(0.62, 0.17), (0.78, 0.06)], s), fill="#d8d8d8", width=s // 28)


def _draw_cup(draw, s: int) -> None:
    draw.rounded_rectangle(_bbox(0.30, 0.30, 0.66, 0.78, s), radius=s // 14, fill="#f7f7f7", outline="#5e5e5e", width=s // 36)
    draw.arc(_bbox(0.58, 0.40, 0.82, 0.66, s), -80, 95, fill="#5e5e5e", width=s // 28)
    draw.ellipse(_bbox(0.30, 0.24, 0.66, 0.38, s), fill="#ffffff", outline="#5e5e5e", width=s // 42)
    draw.rectangle(_bbox(0.35, 0.42, 0.62, 0.70, s), fill="#eeeeee")


def _draw_mug(draw, s: int) -> None:
    draw.rounded_rectangle(_bbox(0.25, 0.28, 0.64, 0.80, s), radius=s // 12, fill="#c44e52", outline="#6e1d22", width=s // 34)
    draw.ellipse(_bbox(0.25, 0.21, 0.64, 0.36, s), fill="#dc6a70", outline="#6e1d22", width=s // 44)
    draw.arc(_bbox(0.55, 0.38, 0.88, 0.70, s), -85, 95, fill="#6e1d22", width=s // 20)
    draw.arc(_bbox(0.61, 0.44, 0.80, 0.64, s), -85, 95, fill="#f8f7f2", width=s // 30)


def _draw_shoe(draw, s: int) -> None:
    draw.polygon(_xy([(0.15, 0.70), (0.26, 0.48), (0.48, 0.52), (0.62, 0.62), (0.86, 0.66), (0.90, 0.77), (0.20, 0.79)], s), fill="#242424", outline="#050505")
    draw.polygon(_xy([(0.26, 0.48), (0.44, 0.35), (0.56, 0.53), (0.48, 0.52)], s), fill="#3d3d3d", outline="#050505")
    draw.line(_xy([(0.38, 0.55), (0.60, 0.64)], s), fill="#cfcfcf", width=s // 50)
    draw.line(_xy([(0.42, 0.51), (0.64, 0.62)], s), fill="#cfcfcf", width=s // 50)
    draw.rectangle(_bbox(0.18, 0.78, 0.88, 0.84, s), fill="#8c8c8c", outline="#555555")


def _draw_spoon(draw, s: int) -> None:
    draw.ellipse(_bbox(0.36, 0.10, 0.64, 0.38, s), fill="#d9d9d9", outline="#777777", width=s // 42)
    draw.rounded_rectangle(_bbox(0.47, 0.34, 0.54, 0.90, s), radius=s // 30, fill="#b8b8b8", outline="#777777", width=s // 56)
    draw.line(_xy([(0.43, 0.18), (0.57, 0.30)], s), fill="#ffffff", width=s // 64)


def _draw_bowl(draw, s: int) -> None:
    draw.pieslice(_bbox(0.18, 0.28, 0.82, 0.92, s), 0, 180, fill="#fff2a8", outline="#8e762a", width=s // 38)
    draw.arc(_bbox(0.18, 0.25, 0.82, 0.48, s), 0, 360, fill="#8e762a", width=s // 28)
    draw.ellipse(_bbox(0.24, 0.27, 0.76, 0.43, s), fill="#fff8cf", outline="#8e762a", width=s // 44)


def _draw_trash_bin(draw, s: int) -> None:
    draw.polygon(_xy([(0.28, 0.25), (0.72, 0.25), (0.64, 0.85), (0.36, 0.85)], s), fill="#7f7f7f", outline="#3f3f3f")
    draw.rounded_rectangle(_bbox(0.22, 0.16, 0.78, 0.27, s), radius=s // 28, fill="#686868", outline="#3f3f3f", width=s // 42)
    for x in (0.40, 0.50, 0.60):
        draw.line(_xy([(x, 0.32), (x - 0.03, 0.78)], s), fill="#555555", width=s // 60)


def _draw_book(draw, s: int) -> None:
    draw.polygon(_xy([(0.22, 0.25), (0.62, 0.18), (0.78, 0.72), (0.36, 0.82)], s), fill="#4e79a7", outline="#1e3e5d")
    draw.polygon(_xy([(0.36, 0.82), (0.78, 0.72), (0.80, 0.78), (0.39, 0.89)], s), fill="#f7f4e8", outline="#1e3e5d")
    draw.line(_xy([(0.32, 0.30), (0.48, 0.84)], s), fill="#ffffff", width=s // 56)


def _draw_remote(draw, s: int) -> None:
    draw.rounded_rectangle(_bbox(0.33, 0.12, 0.67, 0.90, s), radius=s // 18, fill="#151515", outline="#000000", width=s // 38)
    draw.ellipse(_bbox(0.44, 0.20, 0.56, 0.32, s), fill="#d94141")
    for i, y in enumerate([0.42, 0.52, 0.62, 0.72]):
        for x in [0.42, 0.50, 0.58]:
            draw.ellipse(_bbox(x - 0.025, y - 0.025, x + 0.025, y + 0.025, s), fill="#6d6d6d")


def _draw_generic(draw, s: int) -> None:
    draw.rounded_rectangle(_bbox(0.24, 0.24, 0.76, 0.76, s), radius=s // 12, fill="#dddddd", outline="#444444", width=s // 40)


_PAINTERS: dict[str, Callable[[Any, int], None]] = {
    "apple": _draw_apple,
    "banana": _draw_banana,
    "orange": _draw_orange,
    "bottle": _draw_bottle,
    "water_bottle": _draw_water_bottle,
    "soda_can": _draw_soda_can,
    "juice_box": _draw_juice_box,
    "cup": _draw_cup,
    "mug": _draw_mug,
    "shoe": _draw_shoe,
    "spoon": _draw_spoon,
    "bowl": _draw_bowl,
    "trash_bin": _draw_trash_bin,
    "book": _draw_book,
    "remote": _draw_remote,
}

