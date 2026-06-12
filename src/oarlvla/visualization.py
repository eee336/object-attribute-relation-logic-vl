from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

from .scene import Scene
from .webdata.image_utils import write_simple_scene_png


PALETTE = {
    "banana": "#f2c84b",
    "apple": "#d1495b",
    "orange": "#f28e2b",
    "bottle": "#59a14f",
    "water_bottle": "#76b7b2",
    "soda_can": "#e15759",
    "juice_box": "#4e79a7",
    "cup": "#bab0ac",
    "mug": "#b07aa1",
    "shoe": "#4a4a4a",
    "trash_bin": "#7f7f7f",
    "book": "#59a14f",
    "spoon": "#9c9c9c",
    "bowl": "#f1ce63",
    "remote": "#333333",
}


def visualize_scene(
    scene: Scene,
    output_path: str | Path,
    *,
    ground_truth_id: str | None = None,
    predicted_id: str | None = None,
    title: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not HAS_MATPLOTLIB:
        return _visualize_scene_pillow(scene, output_path, ground_truth_id, predicted_id, title)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, scene.width)
    ax.set_ylim(scene.height, 0)
    ax.set_facecolor("#f7f4ed")
    ax.set_title(title or scene.id)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for group in scene.groups:
        if group.bbox:
            _draw_rect(ax, group.bbox, "purple", 2.0, linestyle="--")
            ax.text(group.center[0], group.center[1] - 28, group.id, fontsize=8, color="purple", ha="center")

    for obj in scene.objects:
        face = PALETTE.get(obj.category, "#d0d0d0")
        edge = "#444444"
        width = 1.0
        if obj.id == ground_truth_id:
            edge = "#198754"
            width = 3.0
        if obj.id == predicted_id and predicted_id != ground_truth_id:
            edge = "#dc3545"
            width = 3.0
        elif obj.id == predicted_id:
            edge = "#0d6efd"
            width = 3.0
        _draw_rect(ax, obj.bbox, edge, width, facecolor=face)
        label = f"{obj.id}\n{obj.category}"
        key_attrs = []
        for key in ("size", "black_spot_ratio", "fill_level", "cleanliness"):
            if key == "size":
                value = obj.size
            else:
                value = obj.attributes.get(key, obj.states.get(key))
            if isinstance(value, (int, float)):
                key_attrs.append(f"{key[:5]}={value:.2f}")
        if key_attrs:
            label += "\n" + " ".join(key_attrs[:2])
        ax.text(obj.center[0], obj.center[1], label, fontsize=7, ha="center", va="center", color="black")

    if predicted_id and predicted_id in {group.id for group in scene.groups}:
        group = scene.group_by_id(predicted_id)
        if group and group.bbox:
            color = "#0d6efd" if predicted_id == ground_truth_id else "#dc3545"
            _draw_rect(ax, group.bbox, color, 3.0, linestyle="-.")
    if ground_truth_id and ground_truth_id in {group.id for group in scene.groups}:
        group = scene.group_by_id(ground_truth_id)
        if group and group.bbox:
            _draw_rect(ax, group.bbox, "#198754", 3.0, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _draw_rect(ax, bbox, edgecolor, linewidth, facecolor="none", linestyle="-") -> None:
    x1, y1, x2, y2 = bbox
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=0.82 if facecolor != "none" else 1.0,
            linestyle=linestyle,
        )
    )


def _visualize_scene_pillow(
    scene: Scene,
    output_path: Path,
    ground_truth_id: str | None,
    predicted_id: str | None,
    title: str | None,
) -> Path:
    scale = 2
    margin = 36
    rectangles = []

    for group in scene.groups:
        if group.bbox:
            x1, y1, x2, y2 = group.bbox
            rectangles.append(
                {
                    "bbox": (x1 * scale, y1 * scale + margin, x2 * scale, y2 * scale + margin),
                    "fill": "#f7f4ed",
                    "outline": "#7b2cbf",
                    "width": 3,
                }
            )

    for obj in scene.objects:
        x1, y1, x2, y2 = obj.bbox
        fill = PALETTE.get(obj.category, "#d0d0d0")
        edge = "#444444"
        width = 2
        if obj.id == ground_truth_id:
            edge, width = "#198754", 6
        if obj.id == predicted_id and predicted_id != ground_truth_id:
            edge, width = "#dc3545", 6
        elif obj.id == predicted_id:
            edge, width = "#0d6efd", 6
        rectangles.append(
            {
                "bbox": (x1 * scale, y1 * scale + margin, x2 * scale, y2 * scale + margin),
                "fill": fill,
                "outline": edge,
                "width": width,
            }
        )

    return write_simple_scene_png(output_path, scene.width * scale, scene.height * scale + margin, rectangles)
