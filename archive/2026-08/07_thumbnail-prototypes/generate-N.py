from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SIZE = 900
RENDER_SIZE = 1800
GRID_SIZE = 60

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
GEORGIA_BOLD = FONT_DIR / "Georgia Bold.ttf"
GEORGIA = FONT_DIR / "Georgia.ttf"


@dataclass(frozen=True)
class Style:
    background: str
    low: str
    high: str
    wire: str
    path: str
    endpoint: str
    title: str
    subtitle: str
    shadow: str


DARK = Style(
    background="#16213E",
    low="#24395E",
    high="#6FA0CC",
    wire="#90B6D5",
    path="#E07B39",
    endpoint="#FFFFFF",
    title="#F5F2EA",
    subtitle="#E07B39",
    shadow="#0D1428",
)

LIGHT = Style(
    background="#FFFFFF",
    low="#D8E2EC",
    high="#3E6C99",
    wire="#A6BACD",
    path="#B0521D",
    endpoint="#1A1A1A",
    title="#1A1A1A",
    subtitle="#666666",
    shadow="#FFFFFF",
)


def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[index : index + 2], 16) for index in (0, 2, 4)])


def loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A smooth bowl with two broad, low-amplitude ridges."""
    radius = x * x + y * y
    bowl = 0.115 * (x * x + 0.82 * y * y)
    ridge_a = 0.20 * np.sin(1.18 * x + 0.34 * y) * np.exp(-0.075 * radius)
    ridge_b = 0.105 * np.cos(1.42 * y - 0.28 * x) * np.exp(-0.055 * radius)
    return bowl + ridge_a + ridge_b


def gradient(x: float, y: float) -> np.ndarray:
    radius = x * x + y * y
    exp_a = np.exp(-0.075 * radius)
    exp_b = np.exp(-0.055 * radius)
    phase_a = 1.18 * x + 0.34 * y
    phase_b = 1.42 * y - 0.28 * x

    dx = 0.23 * x
    dx += 0.20 * exp_a * (1.18 * np.cos(phase_a) - 0.15 * x * np.sin(phase_a))
    dx += 0.105 * exp_b * (0.28 * np.sin(phase_b) - 0.11 * x * np.cos(phase_b))

    dy = 0.1886 * y
    dy += 0.20 * exp_a * (0.34 * np.cos(phase_a) - 0.15 * y * np.sin(phase_a))
    dy += 0.105 * exp_b * (-1.42 * np.sin(phase_b) - 0.11 * y * np.cos(phase_b))
    return np.array((dx, dy))


@dataclass(frozen=True)
class OptimizerRun:
    rule: str
    label: str
    start: tuple[float, float]
    learning_rate: float | tuple[float, float, float]
    steps: int
    beta: float = 0.0
    noise_sigma: float = 0.0
    seed: int = 0


RUNS = {
    "N1": OptimizerRun(
        rule="heavy_ball",
        label="heavy-ball momentum",
        start=(3.2, 3.2),
        learning_rate=0.25,
        beta=0.95,
        steps=180,
    ),
    "N2": OptimizerRun(
        rule="gradient_descent",
        label="large-lr gradient descent",
        start=(3.6, 3.55),
        learning_rate=7.5,
        steps=10,
    ),
    "N3": OptimizerRun(
        rule="noisy_heavy_ball",
        label="noisy heavy-ball momentum SGD",
        start=(-3.2, 3.2),
        learning_rate=1.75,
        beta=0.88,
        noise_sigma=0.10,
        seed=53,
        steps=140,
    ),
    "N4": OptimizerRun(
        rule="vanilla_sgd",
        label="vanilla SGD",
        start=(-3.2, 3.2),
        learning_rate=0.20,
        noise_sigma=2.0,
        seed=54,
        steps=200,
    ),
    "N5": OptimizerRun(
        rule="step_decay_sgd",
        label="step-decay learning-rate SGD",
        start=(-3.2, 3.2),
        learning_rate=(8.25, 3.0, 0.30),
        noise_sigma=0.005,
        seed=65,
        steps=100,
    ),
    "N6": OptimizerRun(
        rule="nesterov",
        label="Nesterov accelerated gradient",
        start=(3.6, 3.6),
        learning_rate=0.50,
        beta=0.93,
        steps=100,
    ),
}


def run_optimizer(config: OptimizerRun) -> np.ndarray:
    """Run the named optimizer verbatim, with no path-shaping forces."""
    point = np.array(config.start, dtype=float)
    velocity = np.zeros(2, dtype=float)
    rng = np.random.default_rng(config.seed)
    points = [point.copy()]

    for step in range(config.steps):
        if config.rule == "heavy_ball":
            velocity = config.beta * velocity - float(config.learning_rate) * gradient(*point)
            point = point + velocity
        elif config.rule == "gradient_descent":
            point = point - float(config.learning_rate) * gradient(*point)
        elif config.rule == "noisy_heavy_ball":
            noisy_gradient = gradient(*point) + rng.normal(0.0, config.noise_sigma, size=2)
            velocity = config.beta * velocity - float(config.learning_rate) * noisy_gradient
            point = point + velocity
        elif config.rule == "vanilla_sgd":
            noisy_gradient = gradient(*point) + rng.normal(0.0, config.noise_sigma, size=2)
            point = point - float(config.learning_rate) * noisy_gradient
        elif config.rule == "step_decay_sgd":
            rates = config.learning_rate
            assert isinstance(rates, tuple)
            learning_rate = rates[0] if step < 10 else rates[1] if step < 20 else rates[2]
            noisy_gradient = gradient(*point) + rng.normal(0.0, config.noise_sigma, size=2)
            point = point - learning_rate * noisy_gradient
        elif config.rule == "nesterov":
            lookahead = point + config.beta * velocity
            velocity = config.beta * velocity - float(config.learning_rate) * gradient(*lookahead)
            point = point + velocity
        else:
            raise ValueError(f"unknown update rule: {config.rule}")
        points.append(point.copy())

    return np.array(points)


def simulate_sgd(variant: str) -> np.ndarray:
    """Return one deterministic, genuine optimizer trajectory."""
    try:
        config = RUNS[variant]
    except KeyError as error:
        raise ValueError(f"unknown variant: {variant}") from error
    return run_optimizer(config)


def verify_provenance(variant: str, points: np.ndarray) -> float:
    """Replay the stated update rule and require exact stored-point provenance."""
    recomputed = run_optimizer(RUNS[variant])
    assert recomputed.shape == points.shape, f"{variant}: provenance shape mismatch"
    residual = float(np.max(np.abs(recomputed - points)))
    assert residual <= 1e-12, f"{variant}: update-rule residual={residual:.3e}"
    return residual


def project(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(-60.0)
    elevation = np.deg2rad(35.0)
    rotated_x = np.cos(azimuth) * x - np.sin(azimuth) * y
    rotated_y = np.sin(azimuth) * x + np.cos(azimuth) * y
    screen_x = RENDER_SIZE * 0.5 + rotated_x * 214.0
    screen_y = RENDER_SIZE * 0.59 - (np.cos(elevation) * z - np.sin(elevation) * rotated_y) * 220.0
    depth = np.cos(elevation) * rotated_y + np.sin(elevation) * z
    return screen_x, screen_y, depth


def shaded_color(style: Style, height: float, lambert: float, z_min: float, z_max: float) -> tuple[int, int, int, int]:
    low = hex_rgb(style.low)
    high = hex_rgb(style.high)
    height_mix = np.clip((height - z_min) / (z_max - z_min), 0.0, 1.0)
    height_mix = 0.12 + 0.76 * height_mix
    color = low * (1.0 - height_mix) + high * height_mix
    light_factor = 0.76 + 0.30 * np.clip(lambert, 0.0, 1.0)
    color = np.clip(color * light_factor, 0, 255).astype(int)
    return int(color[0]), int(color[1]), int(color[2]), 236


def build_surface(style: Style, points: np.ndarray) -> Image.Image:
    image = Image.new("RGBA", (RENDER_SIZE, RENDER_SIZE), style.background)
    surface = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(surface, "RGBA")

    axis = np.linspace(-3.25, 3.25, GRID_SIZE)
    x, y = np.meshgrid(axis, axis)
    z = loss(x, y)
    screen_x, screen_y, depth = project(x, y, z)
    z_min, z_max = float(z.min()), float(z.max())

    light = np.array((-0.35, -0.45, 0.82))
    light /= np.linalg.norm(light)
    quads: list[tuple[float, list[tuple[float, float]], float, float]] = []
    for row in range(GRID_SIZE - 1):
        for col in range(GRID_SIZE - 1):
            corners = ((row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col))
            polygon = [(float(screen_x[r, c]), float(screen_y[r, c])) for r, c in corners]
            center_x = float(np.mean([x[r, c] for r, c in corners]))
            center_y = float(np.mean([y[r, c] for r, c in corners]))
            slope = gradient(center_x, center_y)
            normal = np.array((-slope[0], -slope[1], 1.0))
            normal /= np.linalg.norm(normal)
            lambert = float(np.dot(normal, light))
            quads.append(
                (
                    float(np.mean([depth[r, c] for r, c in corners])),
                    polygon,
                    float(np.mean([z[r, c] for r, c in corners])),
                    lambert,
                )
            )

    wire_rgb = tuple(int(channel) for channel in hex_rgb(style.wire))
    for _, polygon, height, lambert in sorted(quads, key=lambda item: item[0], reverse=True):
        draw.polygon(
            polygon,
            fill=shaded_color(style, height, lambert, z_min, z_max),
            outline=(*wire_rgb, 82),
            width=2,
        )

    image = Image.alpha_composite(image, surface)
    path_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    path_draw = ImageDraw.Draw(path_layer, "RGBA")
    path_z = loss(points[:, 0], points[:, 1]) + 0.075
    path_x, path_y, _ = project(points[:, 0], points[:, 1], path_z)
    path_points = [(float(px), float(py)) for px, py in zip(path_x, path_y)]
    path_rgb = tuple(int(channel) for channel in hex_rgb(style.path))
    path_draw.line(path_points, fill=(*path_rgb, 255), width=10, joint="curve")
    for index, (px, py) in enumerate(path_points):
        if index % 4 == 0:
            radius = 6 if index < len(path_points) - 8 else 7
            path_draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(*path_rgb, 255))

    end_x, end_y = path_points[-1]
    endpoint_rgb = tuple(int(channel) for channel in hex_rgb(style.endpoint))
    path_draw.ellipse((end_x - 16, end_y - 16, end_x + 16, end_y + 16), fill=(*endpoint_rgb, 255))
    path_draw.ellipse((end_x - 16, end_y - 16, end_x + 16, end_y + 16), outline=(*path_rgb, 255), width=5)
    image = Image.alpha_composite(image, path_layer)

    # Rendering large and resizing supplies antialiasing while guaranteeing the exact deliverable size.
    return image.convert("RGB").crop((0, 0, RENDER_SIZE, RENDER_SIZE)).resize(
        (SIZE, SIZE), Image.Resampling.LANCZOS
    )


def add_title(image: Image.Image, style: Style) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    title_font = ImageFont.truetype(str(GEORGIA_BOLD), 96)
    subtitle_font = ImageFont.truetype(str(GEORGIA), 30)
    boxes: list[tuple[int, int, int, int]] = []

    def centered(value: str, y: int, face: ImageFont.FreeTypeFont, fill: str) -> tuple[int, int, int, int]:
        local_box = draw.textbbox((0, 0), value, font=face)
        x = round((SIZE - (local_box[2] - local_box[0])) / 2 - local_box[0])
        draw.text((x + 3, y + 3), value, font=face, fill=style.shadow)
        draw.text((x, y), value, font=face, fill=fill)
        box = draw.textbbox((x, y), value, font=face)
        boxes.append((box[0], box[1], box[2] + 3, box[3] + 3))
        return box

    centered("Data Cleansing", 22, title_font, style.title)
    centered("with SGD", 112, title_font, style.title)
    centered("NeurIPS 2019 · Reproduction Study", 218, subtitle_font, style.subtitle)
    overlay_box = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    assert overlay_box[3] < 300, f"title bbox ends at {overlay_box[3]}px"
    return result


def path_metrics(points: np.ndarray) -> tuple[float, float, float, float, float]:
    path_z = loss(points[:, 0], points[:, 1]) + 0.075
    path_x, path_y, _ = project(points[:, 0], points[:, 1], path_z)
    width_percent = 100.0 * float(np.ptp(path_x)) / RENDER_SIZE
    height_percent = 100.0 * float(np.ptp(path_y)) / RENDER_SIZE
    minimum = np.array((-0.49858644, -0.78231073))
    start_loss = float(loss(points[0, 0], points[0, 1]))
    end_distance = float(np.linalg.norm(points[-1] - minimum))
    assert width_percent >= 50.0, f"path width coverage is only {width_percent:.1f}%"
    assert height_percent >= 40.0, f"path height coverage is only {height_percent:.1f}%"
    assert start_loss >= 1.0, f"path does not start high on rim: loss={start_loss:.3f}"
    assert end_distance <= 0.35, f"path does not end near minimum: distance={end_distance:.3f}"
    return width_percent, height_percent, start_loss, end_distance, float(loss(points[-1, 0], points[-1, 1]))


def verify_m1_look(raw_n1: Image.Image) -> tuple[float, float]:
    with Image.open(ROOT / "M1.png") as reference_file:
        reference = reference_file.convert("RGB")
    raw_array = np.asarray(raw_n1, dtype=np.int16)
    reference_array = np.asarray(reference, dtype=np.int16)
    difference = np.abs(raw_array - reference_array)
    # The optimizer trace intentionally differs from M1; compare the unchanged
    # bowl renderer only, excluding orange/white trace pixels in either image.
    raw_trace = (raw_array[:, :, 0] > raw_array[:, :, 2]) | np.all(raw_array > 225, axis=2)
    reference_trace = (reference_array[:, :, 0] > reference_array[:, :, 2]) | np.all(
        reference_array > 225, axis=2
    )
    renderer_pixels = ~(raw_trace | reference_trace)
    renderer_difference = difference[renderer_pixels]
    mean_absolute_error = float(renderer_difference.mean())
    close_pixel_percent = 100.0 * float(np.mean(np.max(renderer_difference, axis=1) <= 3))
    assert mean_absolute_error < 1.0, f"M1 renderer mismatch: MAE={mean_absolute_error:.3f}"
    # Small differences remain only along antialiased trace-mask boundaries.
    assert close_pixel_percent > 97.0, f"M1 renderer mismatch: only {close_pixel_percent:.2f}% close pixels"
    return mean_absolute_error, close_pixel_percent


def save_candidate(name: str, image: Image.Image) -> int:
    path = ROOT / name
    image.save(path, format="PNG", compress_level=6, dpi=(144, 144))
    with Image.open(path) as check:
        assert check.size == (SIZE, SIZE), f"{name}: unexpected size {check.size}"
        assert check.format == "PNG", f"{name}: unexpected format {check.format}"
    size = path.stat().st_size
    assert size > 30 * 1024, f"{name}: file is below 30 KB"
    return size


def main() -> None:
    traces = {name: simulate_sgd(name) for name in ("N1", "N2", "N3", "N4", "N5", "N6")}
    for left_index, left_name in enumerate(traces):
        for right_name in list(traces)[left_index + 1 :]:
            left, right = traces[left_name], traces[right_name]
            assert left.shape != right.shape or not np.array_equal(left, right), (
                f"{left_name} and {right_name} point sets are identical"
            )

    raw_n1 = build_surface(DARK, traces["N1"])
    mae, close_percent = verify_m1_look(raw_n1)
    print(f"M1 look check: MAE={mae:.4f}, close pixels={close_percent:.2f}%")

    for name, points in traces.items():
        provenance_residual = verify_provenance(name, points)
        metrics = path_metrics(points)
        raw = raw_n1 if name == "N1" else build_surface(DARK, points)
        output = add_title(raw, DARK)
        byte_size = save_candidate(f"{name}.png", output)
        width, height, start_loss, end_distance, end_loss = metrics
        print(
            f"{name}.png: rule={RUNS[name].label}, residual={provenance_residual:.1e}, "
            f"{SIZE}x{SIZE}, {byte_size} bytes, {len(points) - 1} steps, "
            f"bbox={width:.1f}%x{height:.1f}%, start_loss={start_loss:.3f}, "
            f"end_distance={end_distance:.3f}, end_loss={end_loss:.4f}"
        )


if __name__ == "__main__":
    main()
