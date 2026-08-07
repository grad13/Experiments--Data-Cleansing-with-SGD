from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SIZE = 900

NAVY = "#16213E"
DARKER = "#0D1428"
CREAM = "#F5F2EA"
ORANGE = "#E07B39"

TITLE = "Data Cleansing"
TITLE_SECOND = "with SGD"
TITLE_SINGLE = "Data Cleansing with SGD"
SUBTITLE = "NeurIPS 2019 · Reproduction Study"


def load_generate_n() -> ModuleType:
    source = ROOT / "generate-N.py"
    spec = importlib.util.spec_from_file_location("generate_n_for_q", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def face(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def text_size(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), value, font=font, anchor="lt")
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    value: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill: str | tuple[int, int, int, int],
    *,
    centered: bool = False,
) -> tuple[int, int, int, int]:
    width, height = text_size(draw, value, font)
    x, y = position
    if centered:
        x = round((SIZE - width) / 2)
    draw.text((x, y), value, font=font, fill=fill, anchor="lt")
    return x, y, x + width, y + height


def draw_fitted_text(
    image: Image.Image,
    value: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
) -> tuple[int, int, int, int]:
    probe = ImageDraw.Draw(image)
    width, height = text_size(probe, value, font)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((0, 0), value, font=font, fill=fill, anchor="lt")
    output_width = min(width, max_width)
    if output_width != width:
        layer = layer.resize((output_width, height), Image.Resampling.LANCZOS)
    x, y = position
    image.paste(layer, (x, y), layer)
    return x, y, x + output_width, y + height


def assert_text_bounds(
    name: str,
    boxes: list[tuple[int, int, int, int]],
    *,
    container: tuple[int, int, int, int] = (0, 0, SIZE, SIZE),
) -> None:
    container_left, container_top, container_right, container_bottom = container
    for box in boxes:
        left, top, right, bottom = box
        assert container_left <= left < right <= container_right, f"{name}: horizontal text overflow: {box}"
        assert container_top <= top < bottom <= container_bottom, f"{name}: vertical text overflow: {box}"


def q1_poster_tilt(raw: Image.Image, bold: Path, regular: Path) -> Image.Image:
    result = raw.copy()
    title_font = face(bold, 120)
    subtitle_font = face(regular, 24)

    title_layer = Image.new("RGBA", (970, 260), (0, 0, 0, 0))
    title_draw = ImageDraw.Draw(title_layer)
    title_draw.text((6, 0), TITLE, font=title_font, fill=CREAM, anchor="lt")
    title_draw.text((6, 128), TITLE_SECOND, font=title_font, fill=CREAM, anchor="lt")
    content_box = title_layer.getbbox()
    assert content_box is not None
    title_layer = title_layer.crop(content_box).rotate(-8, resample=Image.Resampling.BICUBIC, expand=True)
    result.paste(title_layer, (-58, 18), title_layer)

    draw = ImageDraw.Draw(result)
    subtitle_width, subtitle_height = text_size(draw, SUBTITLE, subtitle_font)
    subtitle_box = draw_text(
        draw,
        SUBTITLE,
        (SIZE - subtitle_width - 28, SIZE - subtitle_height - 24),
        subtitle_font,
        ORANGE,
    )
    assert_text_bounds("Q1 subtitle", [subtitle_box])
    return result


def q2_split_panel(raw: Image.Image, bold: Path, regular: Path) -> Image.Image:
    panel_width = round(SIZE * 0.38)
    figure_width = SIZE - panel_width
    crop_left = 300
    cropped = raw.crop((crop_left, 0, crop_left + figure_width, SIZE))

    result = Image.new("RGB", (SIZE, SIZE), DARKER)
    result.paste(cropped, (panel_width, 0))
    title_font = face(bold, 88)
    subtitle_font = face(regular, 24)
    x = 34
    max_width = panel_width - x - 28
    boxes = [
        draw_fitted_text(result, "Data", (x, 120), title_font, CREAM, max_width),
        draw_fitted_text(result, "Cleansing", (x, 224), title_font, CREAM, max_width),
        draw_fitted_text(result, "with SGD", (x, 328), title_font, CREAM, max_width),
        draw_fitted_text(result, SUBTITLE, (x, 830), subtitle_font, ORANGE, max_width),
    ]
    assert_text_bounds("Q2", boxes, container=(0, 0, panel_width, SIZE))
    return result


def q3_backdrop_type(raw: Image.Image, bold: Path, regular: Path) -> Image.Image:
    result = raw.convert("RGBA")
    giant = Image.new("RGBA", result.size, (0, 0, 0, 0))
    giant_draw = ImageDraw.Draw(giant)
    giant_box = draw_text(giant_draw, "SGD", (0, 315), face(bold, 340), (245, 242, 234, 60), centered=True)
    result = Image.alpha_composite(result, giant)

    draw = ImageDraw.Draw(result)
    boxes = [
        draw_text(draw, "Data Cleansing with", (0, 238), face(bold, 44), CREAM, centered=True),
        giant_box,
        draw_text(draw, SUBTITLE, (0, 665), face(regular, 26), ORANGE, centered=True),
    ]
    assert_text_bounds("Q3", boxes)
    return result.convert("RGB")


def q4_museum_frame(raw: Image.Image, bold: Path, regular: Path) -> Image.Image:
    result = Image.new("RGB", (SIZE, SIZE), DARKER)
    figure = raw.resize((760, 760), Image.Resampling.LANCZOS)
    result.paste(figure, (70, 40))
    draw = ImageDraw.Draw(result)
    draw.rectangle((68, 38, 831, 801), outline=ORANGE, width=3)
    boxes = [
        draw_text(draw, TITLE_SINGLE, (0, 816), face(bold, 54), CREAM, centered=True),
        draw_text(draw, SUBTITLE, (0, 872), face(regular, 22), ORANGE, centered=True),
    ]
    assert_text_bounds("Q4", boxes)
    return result


def q5_magazine_cover(raw: Image.Image, bold: Path, regular: Path) -> Image.Image:
    result = raw.convert("RGBA")
    gradient = Image.new("RGBA", (SIZE, SIZE), (13, 20, 40, 0))
    gradient_draw = ImageDraw.Draw(gradient)
    for y in range(430, SIZE):
        alpha = round(255 * 0.88 * (y - 430) / (SIZE - 1 - 430))
        gradient_draw.line((0, y, SIZE, y), fill=(13, 20, 40, alpha))
    result = Image.alpha_composite(result, gradient)

    draw = ImageDraw.Draw(result)
    title_font = face(bold, 110)
    boxes = [
        draw_text(draw, TITLE, (0, 555), title_font, CREAM, centered=True),
        draw_text(draw, TITLE_SECOND, (0, 680), title_font, CREAM, centered=True),
        draw_text(draw, SUBTITLE, (0, 800), face(regular, 28), ORANGE, centered=True),
    ]
    assert_text_bounds("Q5", boxes)
    return result.convert("RGB")


def q6_orange_plate(raw: Image.Image, bold: Path, regular: Path) -> Image.Image:
    result = raw.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 560, 620, SIZE), fill=ORANGE)
    draw.line((0, 560, 620, 560), fill=CREAM, width=3)

    x = 48
    max_width = 620 - x - 24
    title_font = face(bold, 76)
    boxes = [
        draw_fitted_text(result, TITLE, (x, 602), title_font, NAVY, max_width),
        draw_fitted_text(result, TITLE_SECOND, (x, 700), title_font, NAVY, max_width),
        draw_fitted_text(result, SUBTITLE, (x, 814), face(regular, 22), DARKER, max_width),
    ]
    assert_text_bounds("Q6", boxes, container=(0, 560, 620, SIZE))
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
    raw = generate_n.build_surface(generate_n.DARK, generate_n.simulate_sgd("N1"))
    assert raw.size == (SIZE, SIZE), f"raw N1: unexpected size {raw.size}"

    bold = generate_n.GEORGIA_BOLD
    regular = generate_n.GEORGIA
    candidates = {
        "Q1": q1_poster_tilt(raw, bold, regular),
        "Q2": q2_split_panel(raw, bold, regular),
        "Q3": q3_backdrop_type(raw, bold, regular),
        "Q4": q4_museum_frame(raw, bold, regular),
        "Q5": q5_magazine_cover(raw, bold, regular),
        "Q6": q6_orange_plate(raw, bold, regular),
    }
    for name, image in candidates.items():
        byte_size = save_candidate(name, image)
        print(f"{name}.png: {SIZE}x{SIZE}, {byte_size} bytes")


if __name__ == "__main__":
    main()
