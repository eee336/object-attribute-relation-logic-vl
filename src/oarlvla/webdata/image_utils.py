from __future__ import annotations

import struct
import zlib
from pathlib import Path


def read_image_size(path: str | Path) -> tuple[int, int]:
    path = Path(path)
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"P3") or data.startswith(b"P6"):
        tokens = _ppm_tokens(data)
        if len(tokens) >= 3:
            return int(tokens[1]), int(tokens[2])
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


def mean_rgb(path: str | Path) -> tuple[float, float, float] | None:
    path = Path(path)
    data = path.read_bytes()
    if data.startswith(b"P3"):
        tokens = _ppm_tokens(data)
        if len(tokens) < 7:
            return None
        width, height, max_value = int(tokens[1]), int(tokens[2]), int(tokens[3])
        values = [int(token) for token in tokens[4 : 4 + width * height * 3]]
        if not values:
            return None
        scale = 255 / max(max_value, 1)
        channels = [value * scale for value in values]
        pixels = max(1, len(channels) // 3)
        return (
            sum(channels[0::3]) / pixels,
            sum(channels[1::3]) / pixels,
            sum(channels[2::3]) / pixels,
        )
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as im:
            im = im.convert("RGB").resize((32, 32))
            stat = ImageStat.Stat(im)
            return tuple(float(v) for v in stat.mean)  # type: ignore[return-value]
    except Exception:
        return None


def write_simple_scene_png(path: str | Path, width: int, height: int, rectangles: list[dict]) -> Path:
    path = Path(path)
    canvas = bytearray([247, 244, 237] * width * height)
    for rect in rectangles:
        x1, y1, x2, y2 = [int(v) for v in rect["bbox"]]
        fill = _hex_to_rgb(rect.get("fill", "#d0d0d0"))
        outline = _hex_to_rgb(rect.get("outline", "#333333"))
        line_width = int(rect.get("width", 2))
        _fill_rect(canvas, width, height, x1, y1, x2, y2, fill)
        for offset in range(line_width):
            _outline_rect(canvas, width, height, x1 - offset, y1 - offset, x2 + offset, y2 + offset, outline)
    _write_png(path, width, height, bytes(canvas))
    return path


def _write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(b"\x00" + rgb[y * stride : (y + 1) * stride])
    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _fill_rect(canvas: bytearray, width: int, height: int, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
    x1, x2 = max(0, min(x1, x2)), min(width - 1, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(height - 1, max(y1, y2))
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            i = (y * width + x) * 3
            canvas[i : i + 3] = bytes(color)


def _outline_rect(canvas: bytearray, width: int, height: int, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
    x1, x2 = max(0, min(x1, x2)), min(width - 1, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(height - 1, max(y1, y2))
    for x in range(x1, x2 + 1):
        for y in (y1, y2):
            i = (y * width + x) * 3
            canvas[i : i + 3] = bytes(color)
    for y in range(y1, y2 + 1):
        for x in (x1, x2):
            i = (y * width + x) * 3
            canvas[i : i + 3] = bytes(color)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _ppm_tokens(data: bytes) -> list[str]:
    text = data.decode("ascii", errors="ignore")
    lines = [line.split("#", 1)[0] for line in text.splitlines()]
    return " ".join(lines).split()


def _jpeg_size(data: bytes) -> tuple[int, int]:
    idx = 2
    while idx + 9 < len(data):
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        idx += 2
        if marker in {0xD8, 0xD9}:
            continue
        if idx + 2 > len(data):
            break
        length = struct.unpack(">H", data[idx : idx + 2])[0]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height, width = struct.unpack(">HH", data[idx + 3 : idx + 7])
            return width, height
        idx += length
    return (0, 0)

