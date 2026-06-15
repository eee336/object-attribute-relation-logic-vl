from __future__ import annotations

from pathlib import Path

from oarlvla.scene import Scene
from oarlvla.webdata.image_utils import write_simple_scene_png

from .sprites import SPRITE_COLORS, SPRITE_LABELS


def render_grid_scene(scene: Scene, output_path: str | Path, grid_size: int, cell_size: int) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont

        return _render_with_pillow(scene, output_path, grid_size, cell_size, Image, ImageDraw, ImageFont)
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


def _render_with_pillow(scene: Scene, output_path: Path, grid_size: int, cell_size: int, Image, ImageDraw, ImageFont) -> Path:
    image = Image.new("RGB", (scene.width, scene.height), "#f8f7f2")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
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
        _draw_sprite(draw, font, obj)
    image.save(output_path)
    return output_path


def _draw_sprite(draw, font, obj) -> None:
    x1, y1, x2, y2 = obj.bbox
    fill = SPRITE_COLORS.get(obj.category, "#dddddd")
    outline = "#202124"
    if obj.category in {"apple", "orange", "banana"}:
        draw.ellipse((x1, y1, x2, y2), fill=fill, outline=outline, width=2)
    elif obj.category in {"bottle", "water_bottle", "soda_can", "cup", "mug"}:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=fill, outline=outline, width=2)
        if obj.states.get("is_opened"):
            draw.line((x1 + 6, y1 + 6, x2 - 6, y2 - 6), fill="#d00000", width=2)
    elif obj.category == "shoe":
        draw.polygon([(x1, y2), (x1 + 8, y1 + 8), (x2 - 6, y1 + 14), (x2, y2)], fill=fill, outline=outline)
    elif obj.category == "spoon":
        cx, cy = obj.center
        draw.ellipse((cx - 8, y1, cx + 8, y1 + 14), fill=fill, outline=outline, width=2)
        draw.line((cx, y1 + 14, cx, y2), fill=outline, width=3)
    elif obj.category == "trash_bin":
        draw.rectangle((x1, y1 + 6, x2, y2), fill=fill, outline=outline, width=2)
        draw.rectangle((x1 - 3, y1, x2 + 3, y1 + 6), fill="#6c6c6c", outline=outline, width=1)
    elif obj.category == "book":
        draw.rectangle((x1, y1, x2, y2), fill=fill, outline=outline, width=2)
        draw.line((x1 + 7, y1, x1 + 7, y2), fill="#ffffff", width=1)
    else:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=5, fill=fill, outline=outline, width=2)
    label = SPRITE_LABELS.get(obj.category, obj.category[:2])
    draw.text((obj.center[0] - 8, obj.center[1] - 5), label, fill="#000000", font=font)
    if obj.category == "banana" and obj.attributes.get("black_spot_ratio", 0) > 0.35:
        for dx in (-8, 0, 8):
            draw.ellipse((obj.center[0] + dx - 2, obj.center[1] - 2, obj.center[0] + dx + 2, obj.center[1] + 2), fill="#4a2d12")

