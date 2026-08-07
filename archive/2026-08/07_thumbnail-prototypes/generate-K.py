from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "generate-manifold.py"
PYTHON = "/Users/takuo-h/.local/share/pyenv/versions/3.14.5/bin/python3"


def load_manifold_module():
    spec = importlib.util.spec_from_file_location("thumbnail_generate_manifold", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses resolve forward annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gm = load_manifold_module()


RICH_JET_STOPS = (
    (0.00, "#0000C8"),
    (0.01, "#00C8FF"),
    (0.04, "#46DC64"),
    (0.32, "#FFE100"),
    (0.65, "#FF7800"),
    (1.00, "#D71E00"),
)


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    azimuth: float = -50.0
    valley_angle: float = 104.5
    start_along: float = -8.5
    start_across: float = 3.5
    target_minimum: tuple[float, float] = (0.55, 0.68)
    pixels_per_world: float = 72.0
    relief_scale: float = 1.0
    stops: tuple[tuple[float, str], ...] = gm.JET_STOPS
    white_blend: float = 0.12
    path_style: str = "black"
    title_position: str = "top"


VARIANTS = (
    Variant(
        "K1.png",
        "Mirrored +50° camera view with the true-GD trace re-framed in the lower right.",
        azimuth=50.0,
        valley_angle=204.5,
        start_along=-6.0,
        start_across=-3.0,
    ),
    Variant(
        "K2.png",
        "Richer classic-jet terrain colors with no white blending.",
        stops=RICH_JET_STOPS,
        white_blend=0.0,
    ),
    Variant(
        "K3.png",
        "White true-GD zigzag with a thin black outline and ringed endpoint.",
        path_style="white-outline",
    ),
    Variant(
        "K4.png",
        "A 1.5× deeper basin with taller valley walls and a re-run true-GD trace.",
        relief_scale=1.5,
    ),
    Variant(
        "K5.png",
        "Bottom-centered title treatment with the true-GD trace moved into the upper field.",
        target_minimum=(0.55, 0.42),
        title_position="bottom",
    ),
    Variant(
        "K6.png",
        "A 1.35× closer crop that lets the true-GD zigzag fill more of the frame.",
        target_minimum=(0.55, 0.81),
        pixels_per_world=72.0 * 1.35,
    ),
)


def configure(variant: Variant) -> None:
    gm.AZIMUTH_DEG = variant.azimuth
    gm.VALLEY_ANGLE_DEG = variant.valley_angle
    gm.PIXELS_PER_WORLD = variant.pixels_per_world
    gm.TARGET_MINIMUM = np.array(variant.target_minimum, dtype=float) * gm.SIZE
    gm.STEEP_CURVATURE = 0.030 * variant.relief_scale
    gm.SHALLOW_CURVATURE = gm.STEEP_CURVATURE / 8.0
    gm.HEIGHT_SATURATION = 2.5 * variant.relief_scale
    gm.COLOR_WHITE_BLEND = variant.white_blend


def build_trace(variant: Variant):
    """Run exact GD; the two start coefficients only vary for K1 as requested."""
    along_axis, across_axis = gm.valley_axes()
    start_camera = variant.start_along * along_axis + variant.start_across * across_axis
    start_x, start_y = gm.world_coordinates(start_camera[0], start_camera[1])
    learning_rate = gm.GD_DAMPING / gm.STEEP_CURVATURE
    points = [np.array((float(start_x), float(start_y)), dtype=float)]
    gradients: list[np.ndarray] = []
    for _ in range(gm.GD_STEPS):
        gradient = gm.loss_gradient(points[-1])
        gradients.append(gradient)
        points.append(points[-1] - learning_rate * gradient)
    trace = gm.GDTrace(
        points=np.stack(points),
        gradients=np.stack(gradients),
        learning_rate=learning_rate,
    )
    residual = np.diff(trace.points, axis=0) + trace.learning_rate * trace.gradients
    assert float(np.max(np.linalg.norm(residual, axis=1))) <= 1e-12
    return trace


def interpolate(stops: tuple[tuple[float, str], ...], normalized: float, white_blend: float):
    positions = np.array([stop[0] for stop in stops], dtype=float)
    colors = np.array([gm.hex_rgb(stop[1]) for stop in stops])
    value = float(np.clip(normalized, 0.0, 1.0))
    upper = min(max(int(np.searchsorted(positions, value, side="right")), 1), len(stops) - 1)
    lower = upper - 1
    fraction = (value - positions[lower]) / (positions[upper] - positions[lower])
    color = colors[lower] * (1.0 - fraction) + colors[upper] * fraction
    color = color * (1.0 - white_blend) + 255.0 * white_blend
    rgb = np.clip(color, 0, 255).astype(int)
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), 255


def draw_trace(image: Image.Image, projection, points: np.ndarray, path_style: str):
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(layer, "RGBA")
    mask_draw = ImageDraw.Draw(mask)
    z = gm.loss(points[:, 0], points[:, 1])
    path_x, path_y, _ = gm.project(projection, points[:, 0], points[:, 1], z)
    screen_points = [
        (float(x * gm.SUPERSAMPLE), float(y * gm.SUPERSAMPLE))
        for x, y in zip(path_x, path_y)
    ]
    black = (0, 0, 0, 255)
    white = (255, 255, 255, 255)
    inner_width = 10 * gm.SUPERSAMPLE
    outline_width = 14 * gm.SUPERSAMPLE

    for start, end in zip(screen_points[:-1], screen_points[1:]):
        triangle = gm.arrowhead_polygon(start, end)
        if path_style == "white-outline":
            draw.line((start, end), fill=black, width=outline_width)
            draw.line((start, end), fill=white, width=inner_width)
            draw.polygon(
                triangle,
                fill=white,
                outline=black,
                width=2 * gm.SUPERSAMPLE,
            )
            mask_draw.line((start, end), fill=255, width=outline_width)
        else:
            draw.line((start, end), fill=black, width=inner_width)
            draw.polygon(triangle, fill=black)
            mask_draw.line((start, end), fill=255, width=inner_width)
        mask_draw.polygon(triangle, fill=255)

    end_x, end_y = screen_points[-1]
    inner_radius = 9 * gm.SUPERSAMPLE
    if path_style == "white-outline":
        outer_radius = 11 * gm.SUPERSAMPLE
        draw.ellipse(
            (end_x - outer_radius, end_y - outer_radius, end_x + outer_radius, end_y + outer_radius),
            fill=black,
        )
        draw.ellipse(
            (end_x - inner_radius, end_y - inner_radius, end_x + inner_radius, end_y + inner_radius),
            fill=white,
        )
        mask_draw.ellipse(
            (end_x - outer_radius, end_y - outer_radius, end_x + outer_radius, end_y + outer_radius),
            fill=255,
        )
    else:
        draw.ellipse(
            (end_x - inner_radius, end_y - inner_radius, end_x + inner_radius, end_y + inner_radius),
            fill=black,
        )
        mask_draw.ellipse(
            (end_x - inner_radius, end_y - inner_radius, end_x + inner_radius, end_y + inner_radius),
            fill=255,
        )

    composited = Image.alpha_composite(image, layer)
    output_mask = gm.crop_and_downsample(mask, resample=Image.Resampling.NEAREST)
    return composited, gm.path_bbox_from_mask(output_mask), len(points) - 1


def render_surface(variant: Variant, projection, points: np.ndarray):
    axis = np.linspace(-gm.DOMAIN_RADIUS, gm.DOMAIN_RADIUS, gm.GRID_SIZE)
    x, y = np.meshgrid(axis, axis)
    z = gm.loss(x, y)
    screen_x, screen_y, depth = gm.project(projection, x, y, z)
    center_x = 0.25 * (screen_x[:-1, :-1] + screen_x[:-1, 1:] + screen_x[1:, 1:] + screen_x[1:, :-1])
    center_y = 0.25 * (screen_y[:-1, :-1] + screen_y[:-1, 1:] + screen_y[1:, 1:] + screen_y[1:, :-1])
    center_z = 0.25 * (z[:-1, :-1] + z[:-1, 1:] + z[1:, 1:] + z[1:, :-1])
    visible = (
        (center_x >= gm.CROP_MARGIN - 30)
        & (center_x <= gm.CROP_MARGIN + gm.SIZE + 30)
        & (center_y >= gm.CROP_MARGIN - 30)
        & (center_y <= gm.CROP_MARGIN + gm.SIZE + 30)
    )
    quantile_range = np.quantile(center_z[visible], (0.02, 0.90))

    terrain = Image.new("RGBA", (gm.RENDER_SIZE, gm.RENDER_SIZE), (255, 255, 255, 255))
    coverage = Image.new("L", terrain.size, 0)
    draw = ImageDraw.Draw(terrain, "RGBA")
    coverage_draw = ImageDraw.Draw(coverage)

    quad_order: list[tuple[float, int, int]] = []
    for row in range(gm.GRID_SIZE - 1):
        for col in range(gm.GRID_SIZE - 1):
            quad_depth = float(np.mean((depth[row, col], depth[row, col + 1], depth[row + 1, col + 1], depth[row + 1, col])))
            quad_order.append((quad_depth, row, col))
    for _, row, col in sorted(quad_order, reverse=True):
        polygon = [
            (float(screen_x[r, c] * gm.SUPERSAMPLE), float(screen_y[r, c] * gm.SUPERSAMPLE))
            for r, c in ((row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col))
        ]
        normalized = float(gm.smooth_height_norm(np.array(center_z[row, col]), quantile_range))
        draw.polygon(polygon, fill=interpolate(variant.stops, normalized, variant.white_blend))
        coverage_draw.polygon(polygon, fill=255)

    wire = tuple(int(channel) for channel in gm.hex_rgb(gm.JET.wire)) + (52,)
    indices = list(range(0, gm.GRID_SIZE, gm.WIREFRAME_STRIDE))
    if indices[-1] != gm.GRID_SIZE - 1:
        indices.append(gm.GRID_SIZE - 1)
    for row in indices:
        draw.line(
            [(float(screen_x[row, col] * gm.SUPERSAMPLE), float(screen_y[row, col] * gm.SUPERSAMPLE)) for col in range(gm.GRID_SIZE)],
            fill=wire,
            width=gm.SUPERSAMPLE,
        )
    for col in indices:
        draw.line(
            [(float(screen_x[row, col] * gm.SUPERSAMPLE), float(screen_y[row, col] * gm.SUPERSAMPLE)) for row in range(gm.GRID_SIZE)],
            fill=wire,
            width=gm.SUPERSAMPLE,
        )

    terrain, path_box, arrow_count = draw_trace(terrain, projection, points, variant.path_style)
    final = gm.crop_and_downsample(terrain, resample=Image.Resampling.LANCZOS).convert("RGB")
    coverage = gm.crop_and_downsample(coverage, resample=Image.Resampling.NEAREST)
    coverage_ratio = float(np.count_nonzero(np.asarray(coverage))) / float(gm.SIZE * gm.SIZE)
    return final, coverage_ratio, path_box, arrow_count, float(np.ptp(z))


def add_bottom_title(image: Image.Image, path_box: tuple[int, int, int, int]):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    title_font = ImageFont.truetype(str(gm.GEORGIA_BOLD), 96)
    subtitle_font = ImageFont.truetype(str(gm.GEORGIA), 30)
    boxes: list[tuple[int, int, int, int]] = []

    def centered(text: str, y: int, font, fill: str):
        local = draw.textbbox((0, 0), text, font=font, anchor="lt")
        x = round((gm.SIZE - (local[2] - local[0])) / 2 - local[0])
        draw.text((x + 3, y + 3), text, font=font, fill="#FFFFFF", anchor="lt")
        draw.text((x, y), text, font=font, fill=fill, anchor="lt")
        box = draw.textbbox((x, y), text, font=font, anchor="lt")
        boxes.append((box[0], box[1], box[2] + 3, box[3] + 3))
        return box

    first = centered("Data Cleansing", 622, title_font, "#171717")
    second = centered("with SGD", first[3] - 2, title_font, "#171717")
    centered("NeurIPS 2019 · Reproduction Study", second[3] + 11, subtitle_font, "#4B4B4B")
    title_box = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    assert path_box[3] < title_box[1]
    assert not gm.boxes_overlap(title_box, path_box)
    return result, title_box


def title_image(variant: Variant, image: Image.Image, path_box: tuple[int, int, int, int]):
    if variant.title_position == "bottom":
        return add_bottom_title(image, path_box)
    return gm.add_title(image, path_box)


def validate_variant(variant: Variant, projection, trace, coverage: float, path_box, arrows: int, relief: float, baseline_relief: float, title_box):
    width_fraction = (path_box[2] - path_box[0]) / gm.SIZE
    residuals = np.diff(trace.points, axis=0) + trace.learning_rate * trace.gradients
    max_residual = float(np.max(np.linalg.norm(residuals, axis=1)))
    minimum_z = gm.loss(np.array(0.0), np.array(0.0))
    min_x, min_y, _ = gm.project(projection, np.array(0.0), np.array(0.0), np.array(minimum_z))
    projected_min = (float(min_x) - gm.CROP_MARGIN, float(min_y) - gm.CROP_MARGIN)
    assert coverage >= 0.97
    assert arrows == gm.GD_STEPS == 11
    assert max_residual <= 1e-12
    assert not gm.boxes_overlap(title_box, path_box)
    assert abs(projected_min[0] - variant.target_minimum[0] * gm.SIZE) <= 1.0
    assert abs(projected_min[1] - variant.target_minimum[1] * gm.SIZE) <= 1.0
    if variant.name == "K1.png":
        assert variant.azimuth == 50.0 and 0.25 <= width_fraction <= 0.45
        assert path_box[0] > gm.SIZE * 0.5 and path_box[1] > gm.SIZE * 0.5
    if variant.name == "K2.png":
        assert variant.white_blend == 0.0 and variant.stops == RICH_JET_STOPS
    if variant.name == "K3.png":
        assert variant.path_style == "white-outline"
    if variant.name == "K4.png":
        assert 1.49 <= relief / baseline_relief <= 1.51
    if variant.name == "K5.png":
        z = gm.loss(trace.points[:, 0], trace.points[:, 1])
        px, py, _ = gm.project(projection, trace.points[:, 0], trace.points[:, 1], z)
        assert float(px[0] - gm.CROP_MARGIN) < gm.SIZE / 2
        assert float(py[0] - gm.CROP_MARGIN) < gm.SIZE / 2
        assert abs(projected_min[0] - 0.55 * gm.SIZE) <= 1.0
        assert abs(projected_min[1] - 0.42 * gm.SIZE) <= 1.0
    if variant.name == "K6.png":
        assert 0.40 <= width_fraction <= 0.60
        assert abs(variant.pixels_per_world / 72.0 - 1.35) <= 1e-12
    return width_fraction, max_residual, projected_min


def save(name: str, image: Image.Image) -> int:
    path = ROOT / name
    image.save(path, format="PNG", compress_level=6, dpi=(144, 144))
    with Image.open(path) as check:
        assert check.size == (900, 900)
        assert check.format == "PNG"
        assert check.mode == "RGB"
    size = path.stat().st_size
    assert size > 30 * 1024
    return size


def main() -> None:
    assert sys.executable == PYTHON, f"Use the requested interpreter: {PYTHON}"
    assert gm.GEORGIA_BOLD.exists() and gm.GEORGIA.exists()
    baseline_relief = 2.5
    for variant in VARIANTS:
        configure(variant)
        minimum_xy, minimum_z = gm.find_global_minimum()
        projection = gm.make_projection(minimum_xy, minimum_z)
        trace = build_trace(variant)
        surface, coverage, path_box, arrows, relief = render_surface(variant, projection, trace.points)
        titled, title_box = title_image(variant, surface, path_box)
        width_fraction, max_residual, projected_min = validate_variant(
            variant,
            projection,
            trace,
            coverage,
            path_box,
            arrows,
            relief,
            baseline_relief,
            title_box,
        )
        size = save(variant.name, titled)
        print(
            f"{variant.name}: 900x900 RGB PNG; bytes={size}; surface_coverage={coverage:.3%}; "
            f"path_bbox={path_box}; path_width={width_fraction:.3%}; title_bbox={title_box}; "
            f"overlap=false; arrows={arrows}; max_gd_residual={max_residual:.2e}; "
            f"projected_min=({projected_min[0]:.1f},{projected_min[1]:.1f}); relief={relief:.4f}"
        )


if __name__ == "__main__":
    main()
