from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SIZE = 900
TITLE = "Data Cleansing"
TITLE_SECOND_LINE = "with SGD"
TITLE_SINGLE_LINE = "Data Cleansing with SGD"
SUBTITLE = "NeurIPS 2019 · Reproduction Study"
TITLE_FILL = "#F5F2EA"
SUBTITLE_FILL = "#E07B39"
SHADOW_FILL = "#0D1428"
BAND_FILL = "#0D1428"
SHADOW_OFFSET = 3


def load_generate_n() -> ModuleType:
    source = ROOT / "generate-N.py"
    spec = importlib.util.spec_from_file_location("generate_n", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def draw_text(
    draw: ImageDraw.ImageDraw,
    value: str,
    position: tuple[int, int],
    face: ImageFont.FreeTypeFont,
    fill: str,
    *,
    centered: bool,
) -> tuple[int, int, int, int]:
    x, y = position
    if centered:
        local_box = draw.textbbox((0, 0), value, font=face)
        x = round((SIZE - (local_box[2] - local_box[0])) / 2 - local_box[0])
    draw.text((x + SHADOW_OFFSET, y + SHADOW_OFFSET), value, font=face, fill=SHADOW_FILL)
    draw.text((x, y), value, font=face, fill=fill)
    box = draw.textbbox((x, y), value, font=face)
    return box[0], box[1], box[2] + SHADOW_OFFSET, box[3] + SHADOW_OFFSET


def assert_text_bounds(name: str, boxes: list[tuple[int, int, int, int]], *, bottom_limit: int = SIZE) -> None:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    assert 0 <= left < right <= SIZE, f"{name}: text exceeds horizontal bounds: {(left, right)}"
    assert 0 <= top < bottom <= SIZE, f"{name}: text exceeds vertical bounds: {(top, bottom)}"
    assert bottom <= bottom_limit, f"{name}: text ends at {bottom}px, limit is {bottom_limit}px"


def two_line_layout(
    raw: Image.Image,
    name: str,
    title_face: ImageFont.FreeTypeFont,
    subtitle_face: ImageFont.FreeTypeFont,
    *,
    title_y: int,
    subtitle_y: int,
    centered: bool,
    band: bool = False,
    bottom_limit: int = SIZE,
) -> Image.Image:
    result = raw.copy()
    draw = ImageDraw.Draw(result)
    if band:
        draw.rectangle((0, 0, SIZE - 1, 252), fill=BAND_FILL)
    title_x = 0 if centered else 56
    subtitle_x = 0 if centered else 58
    boxes = [
        draw_text(draw, TITLE, (title_x, title_y), title_face, TITLE_FILL, centered=centered),
        draw_text(
            draw,
            TITLE_SECOND_LINE,
            (title_x, title_y + 90),
            title_face,
            TITLE_FILL,
            centered=centered,
        ),
        draw_text(draw, SUBTITLE, (subtitle_x, subtitle_y), subtitle_face, SUBTITLE_FILL, centered=centered),
    ]
    assert_text_bounds(name, boxes, bottom_limit=bottom_limit)
    if band:
        assert max(box[3] for box in boxes) <= 252, f"{name}: text falls outside top band"
    return result


def minimal_layout(
    raw: Image.Image,
    title_face: ImageFont.FreeTypeFont,
    subtitle_face: ImageFont.FreeTypeFont,
) -> Image.Image:
    result = raw.copy()
    draw = ImageDraw.Draw(result)
    boxes = [
        draw_text(draw, TITLE_SINGLE_LINE, (0, 40), title_face, TITLE_FILL, centered=True),
        draw_text(draw, SUBTITLE, (0, 120), subtitle_face, SUBTITLE_FILL, centered=True),
    ]
    assert_text_bounds("P6", boxes)
    return result


def save_candidate(name: str, image: Image.Image) -> int:
    assert image.size == (SIZE, SIZE), f"{name}: unexpected in-memory size {image.size}"
    path = ROOT / f"{name}.png"
    image.save(path, format="PNG", compress_level=6, dpi=(144, 144))
    with Image.open(path) as check:
        assert check.size == (SIZE, SIZE), f"{name}: unexpected saved size {check.size}"
        assert check.format == "PNG", f"{name}: unexpected saved format {check.format}"
    byte_size = path.stat().st_size
    assert byte_size > 30 * 1024, f"{name}: file is below 30 KB ({byte_size} bytes)"
    return byte_size


def main() -> None:
    generate_n = load_generate_n()
    raw_n1 = generate_n.build_surface(generate_n.DARK, generate_n.simulate_sgd("N1"))
    assert raw_n1.size == (SIZE, SIZE), f"raw N1: unexpected size {raw_n1.size}"

    title_96 = font(generate_n.GEORGIA_BOLD, 96)
    subtitle_30 = font(generate_n.GEORGIA, 30)
    title_58 = font(generate_n.GEORGIA_BOLD, 58)
    subtitle_24 = font(generate_n.GEORGIA, 24)

    candidates = {
        "P1": two_line_layout(
            raw_n1, "P1", title_96, subtitle_30, title_y=22, subtitle_y=218, centered=True
        ),
        "P2": two_line_layout(
            raw_n1,
            "P2",
            title_96,
            subtitle_30,
            title_y=600,
            subtitle_y=796,
            centered=True,
            bottom_limit=870,
        ),
        "P3": two_line_layout(
            raw_n1, "P3", title_96, subtitle_30, title_y=22, subtitle_y=218, centered=False
        ),
        "P4": two_line_layout(
            raw_n1,
            "P4",
            title_96,
            subtitle_30,
            title_y=600,
            subtitle_y=796,
            centered=False,
            bottom_limit=870,
        ),
        "P5": two_line_layout(
            raw_n1,
            "P5",
            title_96,
            subtitle_30,
            title_y=0,
            subtitle_y=196,
            centered=True,
            band=True,
        ),
        "P6": minimal_layout(raw_n1, title_58, subtitle_24),
    }

    for name, image in candidates.items():
        byte_size = save_candidate(name, image)
        print(f"{name}.png: {SIZE}x{SIZE}, {byte_size} bytes")


if __name__ == "__main__":
    main()
