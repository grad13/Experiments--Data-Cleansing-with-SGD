from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
SIZE = 900

NAVY = "#16213E"
DARKER = "#0D1428"
CREAM = "#F5F2EA"
ORANGE = "#E07B39"

TITLE = "Data Cleansing with SGD"
SUBTITLE = "NeurIPS 2019 · Reproduction Study"

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
GEORGIA = FONT_DIR / "Georgia.ttf"
GEORGIA_BOLD = FONT_DIR / "Georgia Bold.ttf"
MENLO = Path("/System/Library/Fonts/Menlo.ttc")
COURIER_NEW = FONT_DIR / "Courier New.ttf"

Box = tuple[int, int, int, int]


def load_generate_n() -> ModuleType:
    source = ROOT / "generate-N.py"
    spec = importlib.util.spec_from_file_location("generate_n_for_r", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def serif(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(GEORGIA_BOLD if bold else GEORGIA), size)


def mono(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(MENLO), size, index=0)
    except (OSError, ValueError):
        return ImageFont.truetype(str(COURIER_NEW), size)


def add_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    font: ImageFont.FreeTypeFont,
    fill: str | tuple[int, int, int, int],
    anchor: str = "lt",
    **kwargs: object,
) -> Box:
    draw.text(position, value, font=font, fill=fill, anchor=anchor, **kwargs)
    box = draw.textbbox(position, value, font=font, anchor=anchor, **kwargs)
    return tuple(round(value) for value in box)


def assert_boxes(name: str, boxes: list[Box]) -> None:
    for box in boxes:
        left, top, right, bottom = box
        assert 0 <= left < right <= SIZE, f"{name}: horizontal text overflow: {box}"
        assert 0 <= top < bottom <= SIZE, f"{name}: vertical text overflow: {box}"


def paste_layer(image: Image.Image, layer: Image.Image, position: tuple[int, int], name: str) -> Box:
    x, y = position
    box = (x, y, x + layer.width, y + layer.height)
    assert_boxes(name, [box])
    image.paste(layer, position, layer)
    return box


def projected_path(generate_n: ModuleType, points: object) -> list[tuple[float, float]]:
    path_z = generate_n.loss(points[:, 0], points[:, 1]) + 0.075
    path_x, path_y, _ = generate_n.project(points[:, 0], points[:, 1], path_z)
    scale = SIZE / generate_n.RENDER_SIZE
    return [(float(x * scale), float(y * scale)) for x, y in zip(path_x, path_y)]


def r1_annotation(raw: Image.Image, path: list[tuple[float, float]]) -> Image.Image:
    result = raw.convert("RGBA")
    draw = ImageDraw.Draw(result, "RGBA")
    end_x, end_y = path[-1]
    label = (500, 66, 870, 158)

    draw.line(
        ((round(end_x + 10), round(end_y - 9)), (474, label[3]), (label[0], label[3])),
        fill=CREAM,
        width=1,
    )
    draw.rectangle(label, fill=(13, 20, 40, 205), outline=(245, 242, 234, 115), width=1)
    draw.line((label[0], label[1], label[0], label[3]), fill=ORANGE, width=4)
    boxes = [
        add_text(draw, (518, 82), TITLE, serif(30), CREAM),
        add_text(draw, (518, 123), SUBTITLE, serif(20), ORANGE),
        add_text(draw, (round(end_x + 18), round(end_y - 34)), "θ*", serif(26), CREAM),
    ]
    assert_boxes("R1", boxes)
    return result.convert("RGB")


def r2_badge(raw: Image.Image) -> Image.Image:
    result = raw.convert("RGBA")
    draw = ImageDraw.Draw(result, "RGBA")
    boxes = [
        add_text(draw, (35, 658), "Data Cleansing", serif(92, bold=True), CREAM),
        add_text(draw, (35, 752), "with SGD", serif(92, bold=True), CREAM),
    ]
    assert_boxes("R2 title", boxes)

    badge = Image.new("RGBA", (250, 250), (0, 0, 0, 0))
    badge_draw = ImageDraw.Draw(badge)
    badge_draw.ellipse((15, 15, 235, 235), outline=ORANGE, width=4)
    badge_font = serif(34, bold=True)
    badge_boxes = [
        add_text(badge_draw, (125, 83), "NeurIPS", badge_font, CREAM, anchor="mm"),
        add_text(badge_draw, (125, 137), "2019", badge_font, CREAM, anchor="mm"),
    ]
    for box in badge_boxes:
        assert 0 <= box[0] < box[2] <= badge.width and 0 <= box[1] < box[3] <= badge.height
    badge = badge.rotate(-12, resample=Image.Resampling.BICUBIC, expand=True)
    paste_layer(result, badge, (740 - badge.width // 2, 150 - badge.height // 2), "R2 badge")
    return result.convert("RGB")


def r3_terminal(raw: Image.Image) -> Image.Image:
    result = Image.new("RGB", (SIZE, SIZE), DARKER)
    figure = ImageOps.fit(raw, (SIZE, SIZE - 64), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    result.paste(figure, (0, 64))
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, SIZE, 63), fill=DARKER)
    for x, color in ((31, "#E0604F"), (70, "#E0A63F"), (109, "#57B26B")):
        draw.ellipse((x - 14, 32 - 14, x + 14, 32 + 14), fill=color)

    chrome_font = mono(22)
    chrome_box = draw.textbbox((0, 0), "sgd-influence — reproduction", font=chrome_font, anchor="lt")
    chrome_width = chrome_box[2] - chrome_box[0]
    boxes = [
        add_text(draw, ((SIZE - chrome_width) // 2, 21), "sgd-influence — reproduction", chrome_font, CREAM),
        add_text(draw, (38, 700), "$ python run.py --clean", mono(34), CREAM),
        add_text(draw, (38, 754), TITLE, mono(40), CREAM),
    ]
    cursor_x = boxes[-1][2] + 10
    cursor = (cursor_x, 758, cursor_x + 23, 798)
    draw.rectangle(cursor, fill=CREAM)
    assert_boxes("R3", boxes + [cursor])
    return result


def r4_outline_type(raw: Image.Image) -> Image.Image:
    result = raw.convert("RGBA")
    outline = Image.new("RGBA", result.size, (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(outline)
    outline_draw.text(
        (145, 255),
        "SGD",
        font=serif(380, bold=True),
        fill=(0, 0, 0, 0),
        stroke_width=6,
        stroke_fill=ORANGE,
        anchor="lt",
    )
    result = Image.alpha_composite(result, outline)
    draw = ImageDraw.Draw(result)
    boxes = [
        add_text(draw, (36, 32), "Data Cleansing with", serif(40), CREAM),
        add_text(draw, (36, 838), SUBTITLE, serif(24), ORANGE),
    ]
    assert_boxes("R4", boxes)
    outline_box = outline_draw.textbbox(
        (145, 255), "SGD", font=serif(380, bold=True), stroke_width=6, anchor="lt"
    )
    assert outline_box[2] > SIZE, f"R4: D should intentionally clip at right edge: {outline_box}"
    return result.convert("RGB")


def r5_blueprint(raw: Image.Image, path: list[tuple[float, float]]) -> Image.Image:
    result = raw.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for coordinate in range(0, SIZE + 1, 75):
        draw.line((coordinate, 0, coordinate, SIZE), fill=(245, 242, 234, 28), width=1)
        draw.line((0, coordinate, SIZE, coordinate), fill=(245, 242, 234, 28), width=1)

    tick = 23
    inset = 18
    for x, y in ((inset, inset), (SIZE - inset, inset), (inset, SIZE - inset), (SIZE - inset, SIZE - inset)):
        draw.line((x - tick, y, x + tick, y), fill=(245, 242, 234, 135), width=1)
        draw.line((x, y - tick, x, y + tick), fill=(245, 242, 234, 135), width=1)

    start_x, start_y = path[0]
    end_x, end_y = path[-1]
    start_label_x = min(max(round(start_x - 58), 8), SIZE - 52)
    start_label_y = min(max(round(start_y - 34), 8), SIZE - 38)
    boxes = [
        add_text(draw, (start_label_x, start_label_y), "θ0", mono(24), CREAM),
        add_text(draw, (round(end_x + 18), round(end_y - 34)), "θ*", mono(24), CREAM),
    ]
    # N1 begins just beyond the right crop; this tick connects its computed projected
    # start to the in-canvas label at the exact entry edge.
    draw.line((SIZE - 1, round(start_y), start_label_x + 12, start_label_y + 30), fill=CREAM, width=1)

    panel = (426, 785, 876, 875)
    draw.rectangle(panel, fill=(13, 20, 40, 185), outline=(245, 242, 234, 175), width=1)
    boxes.extend(
        [
            add_text(draw, (444, 799), TITLE, mono(28), CREAM),
            add_text(draw, (444, 841), SUBTITLE, mono(18), ORANGE),
        ]
    )
    assert_boxes("R5", boxes)
    return Image.alpha_composite(result, overlay).convert("RGB")


def horizontal_text_layer(
    value: str, font: ImageFont.FreeTypeFont, fill: str, padding: int = 8
) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.textbbox((0, 0), value, font=font, anchor="lt")
    width, height = box[2] - box[0], box[3] - box[1]
    layer = Image.new("RGBA", (width + 2 * padding, height + 2 * padding), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((padding, padding), value, font=font, fill=fill, anchor="lt")
    return layer


def r6_vertical(raw: Image.Image) -> Image.Image:
    crop = raw.crop((100, 150, 800, 850)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    result = crop.convert("RGBA")

    title = horizontal_text_layer(TITLE, serif(64, bold=True), CREAM).rotate(
        90, resample=Image.Resampling.BICUBIC, expand=True
    )
    subtitle = horizontal_text_layer(SUBTITLE, serif(22), ORANGE).rotate(
        90, resample=Image.Resampling.BICUBIC, expand=True
    )
    title_position = (SIZE - title.width - 12, (SIZE - title.height) // 2)
    subtitle_position = (title_position[0] - subtitle.width - 3, (SIZE - subtitle.height) // 2)
    paste_layer(result, title, title_position, "R6 title")
    paste_layer(result, subtitle, subtitle_position, "R6 subtitle")
    return result.convert("RGB")


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
    points = generate_n.simulate_sgd("N1")
    raw = generate_n.build_surface(generate_n.DARK, generate_n.simulate_sgd("N1"))
    assert raw.size == (SIZE, SIZE), f"raw N1: unexpected size {raw.size}"
    path = projected_path(generate_n, points)

    candidates = {
        "R1": r1_annotation(raw, path),
        "R2": r2_badge(raw),
        "R3": r3_terminal(raw),
        "R4": r4_outline_type(raw),
        "R5": r5_blueprint(raw, path),
        "R6": r6_vertical(raw),
    }
    for name, image in candidates.items():
        byte_size = save_candidate(name, image)
        print(f"{name}.png: {SIZE}x{SIZE}, {byte_size} bytes")


if __name__ == "__main__":
    main()
