from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thumbnail-prototypes"
PAPER = ROOT / "site" / "assets" / "paper"

SIZE = 900
SAFE = 90
WHITE = "#FFFFFF"
DARK = "#1A1A1A"
LIGHT = "#F2F2F2"
GRAY = "#777777"
BLUE = "#3E6C99"
ORANGE = "#B0521D"

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
REGULAR_PATH = FONT_DIR / "Georgia.ttf"
BOLD_PATH = FONT_DIR / "Georgia Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = BOLD_PATH if bold else REGULAR_PATH
    return ImageFont.truetype(str(path), size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    value: str,
    face: ImageFont.FreeTypeFont,
    fill: str,
) -> tuple[int, int, int, int]:
    box = draw.textbbox((0, 0), value, font=face)
    width = box[2] - box[0]
    x = (SIZE - width) // 2 - box[0]
    draw.text((x, y), value, font=face, fill=fill)
    return draw.textbbox((x, y), value, font=face)


def mark_box(mask: Image.Image, box: tuple[int, int, int, int]) -> None:
    ImageDraw.Draw(mask).rectangle(box, fill=255)


def mark_text(mask: Image.Image, box: tuple[int, int, int, int]) -> None:
    ImageDraw.Draw(mask).rectangle(box, fill=255)


def load_rgb(name: str) -> Image.Image:
    with Image.open(PAPER / name) as source:
        return source.convert("RGB")


def contain(source: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int]]:
    left, top, right, bottom = box
    max_w, max_h = right - left, bottom - top
    scale = min(max_w / source.width, max_h / source.height)
    new_size = (round(source.width * scale), round(source.height * scale))
    fitted = source.resize(new_size, Image.Resampling.LANCZOS)
    return fitted, (left + (max_w - new_size[0]) // 2, top + (max_h - new_size[1]) // 2)


def gradient_background() -> Image.Image:
    small = Image.new("RGB", (256, 256))
    pixels = small.load()
    colors = ((26, 26, 46), (22, 33, 62), (15, 52, 96))
    for y in range(256):
        for x in range(256):
            t = (x + y) / 510
            if t <= 0.5:
                local = t * 2
                start, end = colors[0], colors[1]
            else:
                local = (t - 0.5) * 2
                start, end = colors[1], colors[2]
            pixels[x, y] = tuple(round(a + (b - a) * local) for a, b in zip(start, end))
    return small.resize((SIZE, SIZE), Image.Resampling.BICUBIC)


def candidate_s1() -> tuple[Image.Image, Image.Image, Image.Image]:
    background = Image.new("RGB", (SIZE, SIZE), "#16213E")
    image = background.copy()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)

    mark_text(mask, centered_text(draw, 235, "Data Cleansing", font(82, True), LIGHT))
    mark_text(mask, centered_text(draw, 337, "with SGD", font(82, True), LIGHT))
    underline = (370, 465, 530, 474)
    draw.rounded_rectangle(underline, radius=4, fill=ORANGE)
    mark_box(mask, underline)
    mark_text(
        mask,
        centered_text(draw, 535, "NeurIPS 2019 · Reproduction Study", font(27), "#C9CDD5"),
    )
    return image, background, mask


def candidate_s2() -> tuple[Image.Image, Image.Image, Image.Image]:
    background = gradient_background()
    image = background.copy()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)

    mark_text(mask, centered_text(draw, 250, "Data Cleansing", font(82, True), WHITE))
    mark_text(mask, centered_text(draw, 352, "with SGD", font(82, True), WHITE))
    rule = (270, 491, 630, 496)
    draw.rectangle(rule, fill=BLUE)
    mark_box(mask, rule)
    return image, background, mask


def candidate_s3() -> tuple[Image.Image, Image.Image, Image.Image]:
    background = Image.new("RGB", (SIZE, SIZE), WHITE)
    image = background.copy()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)

    mark_text(mask, centered_text(draw, 105, "Data Cleansing with SGD", font(42, True), DARK))
    fitted, (x, y) = contain(load_rgb("fig2.png"), (110, 235, 790, 720))
    image.paste(fitted, (x, y))
    frame = (x - 1, y - 1, x + fitted.width, y + fitted.height)
    draw.rectangle(frame, outline="#DDDDDD", width=2)
    mark_box(mask, frame)
    return image, background, mask


def candidate_s4() -> tuple[Image.Image, Image.Image, Image.Image]:
    background = Image.new("RGB", (SIZE, SIZE), WHITE)
    image = background.copy()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)

    mark_text(mask, centered_text(draw, 150, "SGD-INFLUENCE", font(29, True), BLUE))
    fitted, (x, y) = contain(load_rgb("eq2.png"), (100, 285, 800, 610))
    image.paste(fitted, (x, y))
    mark_box(mask, (x, y, x + fitted.width, y + fitted.height))
    mark_text(mask, centered_text(draw, 665, "NeurIPS 2019", font(25), GRAY))
    return image, background, mask


def candidate_s5() -> tuple[Image.Image, Image.Image, Image.Image]:
    background = Image.new("RGB", (SIZE, SIZE), WHITE)
    image = background.copy()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)

    mark_text(mask, centered_text(draw, 245, "Data Cleansing", font(78, True), DARK))
    mark_text(mask, centered_text(draw, 344, "with SGD", font(78, True), DARK))
    draw.rectangle((0, 702, SIZE, SIZE), fill=BLUE)
    chip = (108, 758, 154, 804)
    draw.rectangle(chip, fill=ORANGE)
    mark_box(mask, chip)
    mark_text(
        mask,
        centered_text(draw, 758, "Reproduction & Bug-Fix Study", font(29), WHITE),
    )
    return image, background, mask


def candidate_s6() -> tuple[Image.Image, Image.Image, Image.Image]:
    background = Image.new("RGB", (SIZE, SIZE), WHITE)
    image = background.copy()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)

    mark_text(mask, centered_text(draw, 105, "Data Cleansing with SGD", font(43, True), DARK))
    margin_line = (162, 255, 164, 650)
    draw.rectangle(margin_line, fill="#D9A0A0")
    mark_box(mask, margin_line)

    widths = (490, 565, 430, 525, 560, 470, 545)
    start_y = 286
    for index, width in enumerate(widths):
        y = start_y + index * 48
        x = 205 if index in (0, 3, 6) else 230
        if index == 3:
            bar = (x, y, x + width, y + 15)
            draw.rounded_rectangle(bar, radius=7, fill="#555555")
            underline = (x, y + 22, x + width, y + 27)
            draw.rounded_rectangle(underline, radius=2, fill=ORANGE)
            mark_box(mask, (bar[0], bar[1], underline[2], underline[3]))
        else:
            bar = (x, y, x + width, y + 15)
            draw.rounded_rectangle(bar, radius=7, fill="#E8E8E8")
            mark_box(mask, bar)

    mark_text(
        mask,
        centered_text(draw, 730, "sgd-influence · reproduction · extra-1", font(22), BLUE),
    )
    return image, background, mask


def validate_and_save(
    name: str,
    image: Image.Image,
    background: Image.Image,
    semantic_mask: Image.Image,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    assert image.size == (SIZE, SIZE)
    actual_bbox = ImageChops.difference(image, background).getbbox()
    semantic_bbox = semantic_mask.getbbox()
    assert actual_bbox is not None
    assert semantic_bbox is not None
    left, top, right, bottom = semantic_bbox
    assert left >= SAFE and top >= SAFE and right <= SIZE - SAFE and bottom <= SIZE - SAFE

    path = OUT / name
    image.save(path, format="PNG", compress_level=6, dpi=(144, 144))
    file_size = path.stat().st_size
    assert file_size > 15 * 1024
    return actual_bbox, semantic_bbox, file_size


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = (
        ("S1.png", candidate_s1),
        ("S2.png", candidate_s2),
        ("S3.png", candidate_s3),
        ("S4.png", candidate_s4),
        ("S5.png", candidate_s5),
        ("S6.png", candidate_s6),
    )
    for name, build in candidates:
        actual_bbox, semantic_bbox, file_size = validate_and_save(name, *build())
        print(
            f"{name}: {file_size} bytes; actual bbox={actual_bbox}; "
            f"safe-content bbox={semantic_bbox}"
        )


if __name__ == "__main__":
    main()
