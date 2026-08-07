from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thumbnail-prototypes"
PAPER = ROOT / "site" / "assets" / "paper"

WIDTH, HEIGHT = 1200, 800
WHITE = "#FFFFFF"
TEXT = "#1A1A1A"
GRAY = "#666666"
BLUE = "#3E6C99"
ORANGE = "#B0521D"

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
REGULAR_PATH = FONT_DIR / "Georgia.ttf"
BOLD_PATH = FONT_DIR / "Georgia Bold.ttf"
FALLBACK_PATH = FONT_DIR / "Times New Roman.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    preferred = BOLD_PATH if bold else REGULAR_PATH
    return ImageFont.truetype(str(preferred if preferred.exists() else FALLBACK_PATH), size)


def canvas() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), WHITE)


def text_width(draw: ImageDraw.ImageDraw, value: str, face: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), value, font=face)
    return box[2] - box[0]


def draw_centered(draw: ImageDraw.ImageDraw, y: int, value: str, face, fill: str) -> None:
    x = (WIDTH - text_width(draw, value, face)) // 2
    draw.text((x, y), value, font=face, fill=fill)


def load_rgb(name: str) -> Image.Image:
    with Image.open(PAPER / name) as source:
        return source.convert("RGB")


def contain(source: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int]]:
    left, top, right, bottom = box
    max_w, max_h = right - left, bottom - top
    scale = min(max_w / source.width, max_h / source.height)
    size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    fitted = source.resize(size, Image.Resampling.LANCZOS)
    return fitted, (left + (max_w - size[0]) // 2, top + (max_h - size[1]) // 2)


def save(image: Image.Image, name: str) -> None:
    image.save(OUT / name, format="PNG", compress_level=6, dpi=(144, 144))


def candidate_a() -> None:
    image = canvas()
    draw = ImageDraw.Draw(image)
    title = font(112, bold=True)
    subtitle = font(34)
    footer = font(23, bold=True)

    draw.text((104, 118), "Data Cleansing", font=title, fill=TEXT)
    draw.text((104, 238), "with SGD", font=title, fill=TEXT)
    draw.rectangle((106, 382, 166, 386), fill=ORANGE)
    draw.text((106, 436), "Reproduction & Bug-Fix Study — NeurIPS 2019", font=subtitle, fill=GRAY)
    draw.text((106, 700), "sgd-influence · reproduction · extra-1", font=footer, fill=BLUE)
    save(image, "A.png")


def candidate_a2() -> None:
    width = height = 900
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    title = font(86, bold=True)
    subtitle = font(31)
    year = font(27)
    footer = font(21)

    def centered(y: int, value: str, face, fill: str) -> None:
        box = draw.textbbox((0, 0), value, font=face)
        text_w = box[2] - box[0]
        x = (width - text_w) // 2 - box[0]
        draw.text((x, y), value, font=face, fill=fill)

    centered(184, "Data Cleansing", title, TEXT)
    centered(290, "with SGD", title, TEXT)
    draw.rounded_rectangle((390, 423, 510, 430), radius=3, fill=ORANGE)
    centered(476, "Reproduction & Bug-Fix Study", subtitle, GRAY)
    centered(526, "NeurIPS 2019", year, GRAY)
    centered(752, "sgd-influence · reproduction · extra-1", footer, BLUE)
    save(image, "A2.png")


def candidate_b() -> None:
    image = canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 60, 1140, 64), fill=BLUE)
    source = load_rgb("fig2.png")
    fitted, position = contain(source, (60, 116, 1140, 655))
    image.paste(fitted, position)
    draw_centered(draw, 704, "Paper Figure 2 — cleansing results", font(27), GRAY)
    save(image, "B.png")


def candidate_c() -> None:
    image = canvas()
    draw = ImageDraw.Draw(image)
    draw.text((74, 62), "The estimator", font=font(34), fill=BLUE)
    source = load_rgb("eq2.png")
    fitted, position = contain(source, (74, 190, 1126, 660))
    image.paste(fitted, position)
    save(image, "C.png")


def candidate_d() -> None:
    image = canvas()
    draw = ImageDraw.Draw(image)
    draw.text((74, 54), "Algorithm 1", font=font(34), fill=BLUE)
    source = load_rgb("alg1.png")
    fitted, position = contain(source, (74, 130, 1126, 744))
    image.paste(fitted, position)
    save(image, "D.png")


def candidate_e() -> None:
    image = canvas()
    draw = ImageDraw.Draw(image)
    draw.text((74, 62), "Counterfactual SGD", font=font(34), fill=BLUE)
    source = load_rgb("def2.png")
    fitted, position = contain(source, (74, 168, 1126, 684))
    image.paste(fitted, position)
    save(image, "E.png")


def candidate_f() -> None:
    image = canvas()
    draw = ImageDraw.Draw(image)
    title = font(57, bold=True)
    draw.text((74, 51), "Data Cleansing with SGD", font=title, fill=TEXT)
    draw.rectangle((74, 150, 1126, 154), fill=ORANGE)
    source = load_rgb("fig2.png")
    fitted, position = contain(source, (74, 224, 1126, 750))
    image.paste(fitted, position)
    save(image, "F.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_a()
    candidate_b()
    candidate_c()
    candidate_d()
    candidate_e()
    candidate_f()


if __name__ == "__main__":
    main()
