from __future__ import annotations

from pathlib import Path

from oarlvla.scene import Scene
from oarlvla.webdata.image_utils import write_simple_scene_png

from .sprites import SPRITE_COLORS, ensure_sprite_assets


def render_grid_scene(
    scene: Scene,
    output_path: str | Path,
    grid_size: int,
    cell_size: int,
    asset_dir: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw

        return _render_with_pillow(scene, output_path, grid_size, cell_size, asset_dir, Image, ImageDraw)
    except ModuleNotFoundError:
        rectangles = [
            {
                "bbox": obj.bbox,
                "fill": SPRITE_COLORS.get(obj.category, "#dddddd"),
                "outline": "#202124",
                "width": 2,
            }
            for obj in scene.objects
        ]
        return write_simple_scene_png(output_path, scene.width, scene.height, rectangles)


def _render_with_pillow(scene: Scene, output_path: Path, grid_size: int, cell_size: int, asset_dir, Image, ImageDraw) -> Path:
    image = Image.new("RGB", (scene.width, scene.height), "#f8f7f2")
    draw = ImageDraw.Draw(image)
    assets = ensure_sprite_assets(asset_dir or output_path.parent.parent / "grid_assets", sprite_size=max(96, int(cell_size * 1.8)))
    for i in range(grid_size + 1):
        x = i * cell_size
        y = i * cell_size
        draw.line((x, 0, x, scene.height), fill="#d6d2c4", width=1)
        draw.line((0, y, scene.width, y), fill="#d6d2c4", width=1)
    for group in scene.groups:
        if group.bbox:
            x1, y1, x2, y2 = group.bbox
            draw.rounded_rectangle((x1 - 4, y1 - 4, x2 + 4, y2 + 4), radius=6, outline="#7b2cbf", width=3)
    for obj in scene.objects:
        _paste_sprite(image, draw, obj, assets)
    image.save(output_path)
    return output_path


def _paste_sprite(image, draw, obj, assets: dict[str, Path]) -> None:
    from PIL import Image

    x1, y1, x2, y2 = obj.bbox
    pad = max(2, int((x2 - x1) * 0.15))
    box = (int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad))
    asset_path = assets.get(obj.category)
    if not asset_path or not asset_path.exists():
        draw.rounded_rectangle(box, radius=6, fill=SPRITE_COLORS.get(obj.category, "#dddddd"), outline="#202124", width=2)
        return
    sprite = Image.open(asset_path).convert("RGBA")
    sprite = _apply_state_overlays(sprite, obj)
    sprite = sprite.resize((max(1, box[2] - box[0]), max(1, box[3] - box[1])), Image.Resampling.LANCZOS)
    image.paste(sprite, box[:2], sprite)


def _apply_state_overlays(sprite, obj):
    if obj.category == "banana" and obj.attributes.get("black_spot_ratio", 0) > 0.35:
        from PIL import ImageDraw

        sprite = sprite.copy()
        draw = ImageDraw.Draw(sprite)
        w, h = sprite.size
        for x, y in [(0.40, 0.60), (0.50, 0.66), (0.62, 0.58), (0.70, 0.48)]:
            r = max(2, w // 45)
            draw.ellipse((int(x * w) - r, int(y * h) - r, int(x * w) + r, int(y * h) + r), fill="#4a2d12")
    if obj.category in {"bottle", "water_bottle", "soda_can", "juice_box"} and obj.states.get("is_opened"):
        from PIL import ImageDraw

        sprite = sprite.copy()
        draw = ImageDraw.Draw(sprite)
        w, h = sprite.size
        draw.line((int(0.25 * w), int(0.18 * h), int(0.75 * w), int(0.82 * h)), fill="#d00000", width=max(2, w // 24))
    return sprite
