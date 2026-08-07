from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SIZE = 900
LOGICAL_RENDER_SIZE = 1400
SUPERSAMPLE = 2
RENDER_SIZE = LOGICAL_RENDER_SIZE * SUPERSAMPLE
CROP_MARGIN = (LOGICAL_RENDER_SIZE - SIZE) // 2
GRID_SIZE = 385
DOMAIN_RADIUS = 22.0

AZIMUTH_DEG = -50.0
ELEVATION_DEG = 30.0
PIXELS_PER_WORLD = 72.0
TARGET_MINIMUM = np.array((0.55 * SIZE, 0.68 * SIZE), dtype=float)
VALLEY_ANGLE_DEG = 104.5
STEEP_CURVATURE = 0.030
SHALLOW_CURVATURE = STEEP_CURVATURE / 8.0
HEIGHT_SATURATION = 2.5
GD_DAMPING = 1.60
GD_STEPS = 11
COLOR_WHITE_BLEND = 0.12
WIREFRAME_STRIDE = 4

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
GEORGIA_BOLD = FONT_DIR / "Georgia Bold.ttf"
GEORGIA = FONT_DIR / "Georgia.ttf"

JET_STOPS = (
    (0.00, "#3B4CC0"),
    (0.01, "#7FB8D8"),
    (0.04, "#CBE8C4"),
    (0.32, "#F5E27A"),
    (0.65, "#E8944A"),
    (1.00, "#C0392B"),
)
BLUE_STOPS = (
    (0.00, "#D8E2EC"),
    (0.35, "#B5CBDE"),
    (0.68, "#7FA1C0"),
    (1.00, "#3E6C99"),
)


@dataclass(frozen=True)
class Style:
    name: str
    stops: tuple[tuple[float, str], ...]
    wire: str
    path: str


@dataclass(frozen=True)
class Projection:
    shift_x: float
    shift_y: float
    minimum_xy: np.ndarray
    minimum_z: float


@dataclass(frozen=True)
class GDTrace:
    points: np.ndarray
    gradients: np.ndarray
    learning_rate: float


JET = Style("jet", JET_STOPS, "#252A30", "#000000")
SITE = Style("blue", BLUE_STOPS, "#2F465B", "#B0521D")


def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=float)


def camera_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(AZIMUTH_DEG)
    p = np.cos(azimuth) * x - np.sin(azimuth) * y
    q = np.sin(azimuth) * x + np.cos(azimuth) * y
    return p, q


def world_coordinates(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(AZIMUTH_DEG)
    x = np.cos(azimuth) * p + np.sin(azimuth) * q
    y = -np.sin(azimuth) * p + np.cos(azimuth) * q
    return x, y


def valley_axes() -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal unit vectors along and across the narrow valley."""
    angle = np.deg2rad(VALLEY_ANGLE_DEG)
    along = np.array((np.cos(angle), np.sin(angle)), dtype=float)
    across = np.array((-along[1], along[0]), dtype=float)
    assert abs(float(np.dot(along, across))) < 1e-12
    return along, across


def valley_coordinates(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    along_axis, across_axis = valley_axes()
    along = along_axis[0] * p + along_axis[1] * q
    across = across_axis[0] * p + across_axis[1] * q
    return along, across


def loss_from_camera(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """A smooth, diagonally oriented 8:1 narrow valley with bounded outer ridges."""
    along, across = valley_coordinates(p, q)
    quadratic = 0.5 * (
        SHALLOW_CURVATURE * along**2 + STEEP_CURVATURE * across**2
    )
    return HEIGHT_SATURATION * np.tanh(quadratic / HEIGHT_SATURATION)


def loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p, q = camera_coordinates(x, y)
    return loss_from_camera(p, q)


def loss_gradient(xy: np.ndarray) -> np.ndarray:
    """Analytic world-space gradient of ``loss`` at one point."""
    p, q = camera_coordinates(xy[0], xy[1])
    along_axis, across_axis = valley_axes()
    along, across = valley_coordinates(p, q)
    quadratic = 0.5 * (
        SHALLOW_CURVATURE * along**2 + STEEP_CURVATURE * across**2
    )
    saturation_derivative = 1.0 / np.cosh(quadratic / HEIGHT_SATURATION) ** 2
    gradient_camera = saturation_derivative * (
        SHALLOW_CURVATURE * along * along_axis
        + STEEP_CURVATURE * across * across_axis
    )
    gradient_x, gradient_y = world_coordinates(gradient_camera[0], gradient_camera[1])
    return np.array((float(gradient_x), float(gradient_y)), dtype=float)


def find_global_minimum() -> tuple[np.ndarray, float]:
    # The analytic minimum is the axes' intersection; retain a dense global check.
    axis = np.linspace(-DOMAIN_RADIUS, DOMAIN_RADIUS, 501)
    p, q = np.meshgrid(axis, axis)
    z = loss_from_camera(p, q)
    row, col = np.unravel_index(int(np.argmin(z)), z.shape)
    dense_best = np.array((p[row, col], q[row, col]), dtype=float)
    assert np.linalg.norm(dense_best) < 1e-12, "dense scan must locate the analytic minimum"
    assert float(z[row, col]) == 0.0
    x, y = world_coordinates(0.0, 0.0)
    return np.array((float(x), float(y))), 0.0


def make_projection(minimum_xy: np.ndarray, minimum_z: float) -> Projection:
    p_min, q_min = camera_coordinates(minimum_xy[0], minimum_xy[1])
    elevation = np.deg2rad(ELEVATION_DEG)
    target = TARGET_MINIMUM + CROP_MARGIN
    raw_min_x = float(p_min) * PIXELS_PER_WORLD
    raw_min_y = (
        float(q_min) * PIXELS_PER_WORLD * np.sin(elevation)
        - minimum_z * PIXELS_PER_WORLD * np.cos(elevation)
    )
    return Projection(
        shift_x=float(target[0] - raw_min_x),
        shift_y=float(target[1] - raw_min_y),
        minimum_xy=minimum_xy,
        minimum_z=minimum_z,
    )


def project(
    projection: Projection,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, q = camera_coordinates(x, y)
    elevation = np.deg2rad(ELEVATION_DEG)
    screen_x = projection.shift_x + p * PIXELS_PER_WORLD
    screen_y = projection.shift_y + (
        q * PIXELS_PER_WORLD * np.sin(elevation)
        - z * PIXELS_PER_WORLD * np.cos(elevation)
    )
    depth = q * np.cos(elevation) + z * np.sin(elevation)
    return screen_x, screen_y, depth


def build_path() -> GDTrace:
    """Run genuine underdamped GD from the upper-left wall of the valley."""
    along_axis, across_axis = valley_axes()
    start_camera = -8.5 * along_axis + 3.5 * across_axis
    start_x, start_y = world_coordinates(start_camera[0], start_camera[1])
    learning_rate = GD_DAMPING / STEEP_CURVATURE

    points = [np.array((float(start_x), float(start_y)), dtype=float)]
    gradients: list[np.ndarray] = []
    for _ in range(GD_STEPS):
        gradient = loss_gradient(points[-1])
        gradients.append(gradient)
        points.append(points[-1] - learning_rate * gradient)

    trace = GDTrace(
        points=np.stack(points),
        gradients=np.stack(gradients),
        learning_rate=learning_rate,
    )
    # This identity is the core provenance assertion: no point is hand-adjusted.
    assert np.allclose(
        np.diff(trace.points, axis=0),
        -trace.learning_rate * trace.gradients,
        rtol=1e-13,
        atol=1e-13,
    ), "every displayed segment must be the exact -eta*gradient update"
    return trace


def smooth_height_norm(values: np.ndarray, quantile_range: np.ndarray) -> np.ndarray:
    """Quantile-normalize then apply one smooth, monotone smoothstep curve."""
    low, high = (float(value) for value in quantile_range)
    linear = np.clip((values - low) / (high - low), 0.0, 1.0)
    return linear * linear * (3.0 - 2.0 * linear)


def interpolate_colormap(style: Style, normalized: float) -> tuple[int, int, int, int]:
    positions = np.array([stop[0] for stop in style.stops], dtype=float)
    colors = np.array([hex_rgb(stop[1]) for stop in style.stops])
    normalized = float(np.clip(normalized, 0.0, 1.0))
    upper = int(np.searchsorted(positions, normalized, side="right"))
    upper = min(max(upper, 1), len(positions) - 1)
    lower = upper - 1
    span = positions[upper] - positions[lower]
    fraction = (normalized - positions[lower]) / span
    color = colors[lower] * (1.0 - fraction) + colors[upper] * fraction
    if style is JET:
        color = color * (1.0 - COLOR_WHITE_BLEND) + 255.0 * COLOR_WHITE_BLEND
    color = np.clip(color, 0, 255).astype(int)
    return int(color[0]), int(color[1]), int(color[2]), 255


def connected_component_sizes(mask: np.ndarray) -> list[int]:
    seen = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    rows, cols = mask.shape
    for start_row, start_col in zip(*np.nonzero(mask & ~seen)):
        if seen[start_row, start_col]:
            continue
        stack = [(int(start_row), int(start_col))]
        seen[start_row, start_col] = True
        size = 0
        while stack:
            row, col = stack.pop()
            size += 1
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= next_row < rows and 0 <= next_col < cols:
                    if mask[next_row, next_col] and not seen[next_row, next_col]:
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))
        sizes.append(size)
    return sizes


def arrowhead_polygon(
    start: tuple[float, float], end: tuple[float, float]
) -> list[tuple[float, float]]:
    direction = np.array((end[0] - start[0], end[1] - start[1]), dtype=float)
    length = float(np.linalg.norm(direction))
    assert length > 0.0, "path segments must have nonzero length"
    unit = direction / length
    perpendicular = np.array((-unit[1], unit[0]))
    tip = np.array(end, dtype=float)
    base = tip - unit * (18.0 * SUPERSAMPLE)
    half_width = 8.0 * SUPERSAMPLE
    return [tuple(tip), tuple(base + perpendicular * half_width), tuple(base - perpendicular * half_width)]


def crop_and_downsample(image: Image.Image, *, resample: Image.Resampling) -> Image.Image:
    margin = CROP_MARGIN * SUPERSAMPLE
    side = SIZE * SUPERSAMPLE
    cropped = image.crop((margin, margin, margin + side, margin + side))
    return cropped.resize((SIZE, SIZE), resample)


def path_bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int]:
    box = mask.getbbox()
    assert box is not None, "path artwork mask is empty"
    return tuple(int(value) for value in box)


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def draw_path(
    image: Image.Image,
    projection: Projection,
    points: np.ndarray,
    color_hex: str,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int], int]:
    path_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    path_mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(path_layer, "RGBA")
    mask_draw = ImageDraw.Draw(path_mask)
    z = loss(points[:, 0], points[:, 1])
    path_x, path_y, _ = project(projection, points[:, 0], points[:, 1], z)
    screen_points = [
        (float(px * SUPERSAMPLE), float(py * SUPERSAMPLE)) for px, py in zip(path_x, path_y)
    ]
    color = (*tuple(int(channel) for channel in hex_rgb(color_hex)), 255)
    width = 10 * SUPERSAMPLE
    arrow_count = 0
    for start, end in zip(screen_points[:-1], screen_points[1:]):
        draw.line((start, end), fill=color, width=width)
        mask_draw.line((start, end), fill=255, width=width)
        triangle = arrowhead_polygon(start, end)
        draw.polygon(triangle, fill=color)
        mask_draw.polygon(triangle, fill=255)
        arrow_count += 1

    end_x, end_y = screen_points[-1]
    radius = 9 * SUPERSAMPLE
    dot_box = (end_x - radius, end_y - radius, end_x + radius, end_y + radius)
    draw.ellipse(dot_box, fill=color)
    mask_draw.ellipse(dot_box, fill=255)
    composited = Image.alpha_composite(image, path_layer)
    path_pixels = np.asarray(path_mask) > 0
    assert np.array_equal(
        np.asarray(composited)[path_pixels], np.asarray(path_layer)[path_pixels]
    ), "C4 path must be composited opaquely after the terrain"
    output_mask = crop_and_downsample(path_mask, resample=Image.Resampling.NEAREST)
    path_box = path_bbox_from_mask(output_mask)
    return composited, output_mask, path_box, arrow_count


def render_surface(
    style: Style,
    projection: Projection,
    points: np.ndarray,
) -> tuple[Image.Image, float, tuple[int, int, int, int], int, dict[str, float]]:
    axis = np.linspace(-DOMAIN_RADIUS, DOMAIN_RADIUS, GRID_SIZE)
    x, y = np.meshgrid(axis, axis)
    z = loss(x, y)
    screen_x, screen_y, depth = project(projection, x, y, z)

    center_x = 0.25 * (screen_x[:-1, :-1] + screen_x[:-1, 1:] + screen_x[1:, 1:] + screen_x[1:, :-1])
    center_y = 0.25 * (screen_y[:-1, :-1] + screen_y[:-1, 1:] + screen_y[1:, 1:] + screen_y[1:, :-1])
    center_z = 0.25 * (z[:-1, :-1] + z[:-1, 1:] + z[1:, 1:] + z[1:, :-1])
    visible = (
        (center_x >= CROP_MARGIN - 30)
        & (center_x <= CROP_MARGIN + SIZE + 30)
        & (center_y >= CROP_MARGIN - 30)
        & (center_y <= CROP_MARGIN + SIZE + 30)
    )
    visible_heights = center_z[visible]
    assert visible_heights.size > 1000, "too few visible terrain cells"
    quantile_range = np.quantile(visible_heights, (0.02, 0.90))
    assert quantile_range[1] > quantile_range[0], "height normalization must have positive range"

    terrain = Image.new("RGBA", (RENDER_SIZE, RENDER_SIZE), (255, 255, 255, 255))
    coverage = Image.new("L", terrain.size, 0)
    draw = ImageDraw.Draw(terrain, "RGBA")
    coverage_draw = ImageDraw.Draw(coverage)

    quad_order: list[tuple[float, int, int]] = []
    for row in range(GRID_SIZE - 1):
        for col in range(GRID_SIZE - 1):
            quad_depth = float(
                np.mean((depth[row, col], depth[row, col + 1], depth[row + 1, col + 1], depth[row + 1, col]))
            )
            quad_order.append((quad_depth, row, col))

    for _, row, col in sorted(quad_order, reverse=True):
        polygon = [
            (float(screen_x[r, c] * SUPERSAMPLE), float(screen_y[r, c] * SUPERSAMPLE))
            for r, c in ((row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col))
        ]
        height = float(center_z[row, col])
        normalized = float(smooth_height_norm(np.array(height), quantile_range))
        draw.polygon(polygon, fill=interpolate_colormap(style, normalized))
        coverage_draw.polygon(polygon, fill=255)

    # Preserve the mesh cue without letting it dominate the smoothly shaded surface.
    wire_rgb = tuple(int(channel) for channel in hex_rgb(style.wire))
    wire_color = (*wire_rgb, 52)
    wire_width = 1 * SUPERSAMPLE
    wire_indices = list(range(0, GRID_SIZE, WIREFRAME_STRIDE))
    if wire_indices[-1] != GRID_SIZE - 1:
        wire_indices.append(GRID_SIZE - 1)
    for row in wire_indices:
        draw.line(
            [
                (float(screen_x[row, col] * SUPERSAMPLE), float(screen_y[row, col] * SUPERSAMPLE))
                for col in range(GRID_SIZE)
            ],
            fill=wire_color,
            width=wire_width,
        )
    for col in wire_indices:
        draw.line(
            [
                (float(screen_x[row, col] * SUPERSAMPLE), float(screen_y[row, col] * SUPERSAMPLE))
                for row in range(GRID_SIZE)
            ],
            fill=wire_color,
            width=wire_width,
        )

    terrain, _, path_box, arrow_count = draw_path(terrain, projection, points, style.path)
    final_image = crop_and_downsample(terrain, resample=Image.Resampling.LANCZOS).convert("RGB")
    final_coverage = crop_and_downsample(coverage, resample=Image.Resampling.NEAREST)
    coverage_ratio = float(np.count_nonzero(np.asarray(final_coverage))) / float(SIZE * SIZE)

    normalized_visible = smooth_height_norm(visible_heights, quantile_range)
    jet_colors = np.array([interpolate_colormap(JET, float(value))[:3] for value in normalized_visible])
    height_quantiles = np.quantile(visible_heights, (0.25, 0.60))
    top = visible_heights >= height_quantiles[1]
    middle = (visible_heights >= height_quantiles[0]) & (visible_heights < height_quantiles[1])
    top_red_orange = (
        (jet_colors[:, 0] >= 180)
        & (jet_colors[:, 0] > jet_colors[:, 2] + 35)
    )
    middle_green_yellow = (
        (jet_colors[:, 0] >= 165) & (jet_colors[:, 1] >= 165)
    )
    blue_pocket = (
        (jet_colors[:, 2] > jet_colors[:, 0] + 35)
        & (jet_colors[:, 2] > jet_colors[:, 1] + 15)
    )
    color_metrics = {
        "top_red_orange": float(np.mean(top_red_orange[top])),
        "middle_green_yellow": float(np.mean(middle_green_yellow[middle])),
        "basin_fraction": float(np.mean(blue_pocket)),
        "wire_lines": len(wire_indices) * 2,
    }

    # C1: the projected surface, not a background fill, covers practically every output pixel.
    assert coverage_ratio >= 0.97, f"C1 surface coverage {coverage_ratio:.4%} is below 97%"
    # C3: smooth normalization makes the ridges warm and confines deep blue to one pocket.
    assert color_metrics["top_red_orange"] >= 0.85, "C3 upper 40% is not predominantly red/orange"
    assert color_metrics["middle_green_yellow"] >= 0.70, "C3 middle is not predominantly green/yellow"
    assert 0.03 <= color_metrics["basin_fraction"] <= 0.15, "C3 blue basin is not a compact pocket"
    assert GRID_SIZE >= 257, "C3 surface grid is too coarse"
    assert wire_width == 2 and WIREFRAME_STRIDE == 4, "C3 wire must be faint, thin, and every fourth line"
    assert COLOR_WHITE_BLEND == 0.12, "C3 jet colors must be softened 12% toward white"
    if style is SITE:
        assert style.stops[0][1] == "#D8E2EC" and style.stops[-1][1] == "#3E6C99"

    return final_image, coverage_ratio, path_box, arrow_count, color_metrics


def add_title(
    image: Image.Image, path_box: tuple[int, int, int, int]
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    title_font = ImageFont.truetype(str(GEORGIA_BOLD), 96)
    subtitle_font = ImageFont.truetype(str(GEORGIA), 30)
    shadow_offset = 3
    boxes: list[tuple[int, int, int, int]] = []

    def centered(value: str, y: int, font: ImageFont.FreeTypeFont, fill: str) -> tuple[int, int, int, int]:
        local_box = draw.textbbox((0, 0), value, font=font, anchor="lt")
        width = local_box[2] - local_box[0]
        x = round((SIZE - width) / 2 - local_box[0])
        draw.text(
            (x + shadow_offset, y + shadow_offset),
            value,
            font=font,
            fill="#FFFFFF",
            anchor="lt",
        )
        draw.text((x, y), value, font=font, fill=fill, anchor="lt")
        box = draw.textbbox((x, y), value, font=font, anchor="lt")
        expanded = (box[0], box[1], box[2] + shadow_offset, box[3] + shadow_offset)
        boxes.append(expanded)
        return box

    first_box = centered("Data Cleansing", 34, title_font, "#171717")
    second_box = centered("with SGD", first_box[3] - 2, title_font, "#171717")
    centered("NeurIPS 2019 · Reproduction Study", second_box[3] + 11, subtitle_font, "#4B4B4B")
    title_box = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    # C5: exact typography/shadow and strict separation from the optimization trace.
    assert title_font.size == 96 and shadow_offset == 3, "C5 title typography or shadow changed"
    assert title_box[3] < path_box[1], f"C5 title {title_box} reaches the path {path_box}"
    assert not boxes_overlap(title_box, path_box), f"C5 title {title_box} overlaps path {path_box}"
    return result, title_box


def validate_scene(
    projection: Projection,
    trace: GDTrace,
    path_box: tuple[int, int, int, int],
    arrow_count: int,
    grid_z: np.ndarray,
    visible_low_mask: np.ndarray,
) -> dict[str, float | int | tuple[int, int, int, int]]:
    points = trace.points
    z_range = float(np.ptp(grid_z))
    relative_relief = z_range / (2.0 * DOMAIN_RADIUS)
    low_components = connected_component_sizes(visible_low_mask)
    significant_components = [size for size in low_components if size >= 4]

    min_z = loss(np.array(projection.minimum_xy[0]), np.array(projection.minimum_xy[1]))
    min_x, min_y, _ = project(
        projection,
        np.array(projection.minimum_xy[0]),
        np.array(projection.minimum_xy[1]),
        np.array(min_z),
    )
    projected_min = np.array((float(min_x) - CROP_MARGIN, float(min_y) - CROP_MARGIN))

    path_width = path_box[2] - path_box[0]
    path_height = path_box[3] - path_box[1]
    width_fraction = path_width / SIZE
    height_fraction = path_height / SIZE
    path_z = loss(points[:, 0], points[:, 1])
    path_p, path_q = camera_coordinates(points[:, 0], points[:, 1])
    path_along, path_across = valley_coordinates(path_p, path_q)
    px, py, _ = project(projection, points[:, 0], points[:, 1], path_z)
    px -= CROP_MARGIN
    py -= CROP_MARGIN
    dx = np.diff(px)
    first_six_dx = dx[:6]
    alternating_dx = np.sign(first_six_dx[:-1]) * np.sign(first_six_dx[1:]) < 0
    vectors = np.column_stack((np.diff(px), np.diff(py)))
    turn_angles: list[float] = []
    for first, second in zip(vectors[:-1], vectors[1:]):
        cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
        turn_angles.append(float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    horizontal_ratios = np.abs(first_six_dx) / np.maximum(np.abs(np.diff(py)[:6]), 1e-12)
    cross_alternation = path_across[:6] * path_across[1:7] < 0.0
    loss_changes = np.diff(path_z)
    nonmonotonic_bumps = int(np.count_nonzero(loss_changes >= 0.0))
    update_residuals = np.diff(points, axis=0) + trace.learning_rate * trace.gradients
    max_update_residual = float(np.max(np.linalg.norm(update_residuals, axis=1)))
    endpoint_distance = float(np.linalg.norm(points[-1] - projection.minimum_xy))

    # C2: low-relief rolling terrain, one connected basin pocket, and correctly placed minimum.
    assert relative_relief <= 0.16, f"C2 relief/extent {relative_relief:.4f} is too steep"
    assert len(significant_components) == 1, f"C2 expected one basin pocket, found {significant_components}"
    assert np.all(np.abs(projected_min - TARGET_MINIMUM) <= 1.0), (
        f"C2 minimum projects to {projected_min}, expected {TARGET_MINIMUM}"
    )
    outer_p, outer_q = camera_coordinates(points[0, 0], points[0, 1])
    assert float(loss_from_camera(outer_p, outer_q) - projection.minimum_z) >= 0.25, "C2 basin is not clear"
    assert abs(STEEP_CURVATURE / SHALLOW_CURVATURE - 8.0) < 1e-12, "C2 valley anisotropy changed"

    # C4: every arrow is a true GD update in an underdamped narrow valley.
    segment_count = len(points) - 1
    assert 10 <= segment_count <= 12, f"C4 segment count is {segment_count}"
    assert arrow_count == segment_count, f"C4 arrowhead count {arrow_count} != {segment_count}"
    assert 0.25 <= width_fraction <= 0.45, f"C4 path width {width_fraction:.2%}"
    assert 0.28 <= height_fraction <= 0.50, f"C4 path height {height_fraction:.2%}"
    assert bool(np.all(cross_alternation)), "C4 first six updates do not bounce across the valley"
    assert bool(np.all(alternating_dx)), "C4 first six projected directions do not alternate"
    assert float(np.min(horizontal_ratios)) >= 1.10, "C4 first six updates are not horizontal-ish"
    assert min(turn_angles[:5]) >= 110.0, f"C4 early turns are not sharp enough: {min(turn_angles[:5]):.1f} degrees"
    assert px[0] < SIZE / 2 and py[0] < SIZE / 2, "C4 path must begin upper-left of center"
    assert float(path_along[0]) < 0.0 and float(path_across[0]) > 0.0, "C4 start is not on the upper wall"
    assert float(path_z[-1]) < float(path_z[0]), "C4 loss does not decrease overall"
    assert nonmonotonic_bumps <= 2, f"C4 has {nonmonotonic_bumps} non-monotonic loss bumps"
    assert float(path_z[-1]) == float(np.min(path_z)), "C4 endpoint is not the lowest path point"
    assert max_update_residual <= 1e-12, "C4 a displayed segment is not its actual GD update"
    assert endpoint_distance <= 2.0, f"C4 endpoint is {endpoint_distance:.3f} world units from minimum"

    return {
        "relative_relief": relative_relief,
        "basin_components": len(significant_components),
        "minimum_x": float(projected_min[0]),
        "minimum_y": float(projected_min[1]),
        "segments": segment_count,
        "arrows": arrow_count,
        "path_bbox": path_box,
        "path_width_fraction": width_fraction,
        "path_height_fraction": height_fraction,
        "minimum_turn_angle": min(turn_angles),
        "early_min_turn_angle": min(turn_angles[:5]),
        "min_horizontal_ratio": float(np.min(horizontal_ratios)),
        "loss_start": float(path_z[0]),
        "loss_end": float(path_z[-1]),
        "nonmonotonic_bumps": nonmonotonic_bumps,
        "endpoint_distance": endpoint_distance,
        "max_gd_residual": max_update_residual,
        "learning_rate": trace.learning_rate,
        "overlay_verified": True,
    }


def save_candidate(name: str, image: Image.Image) -> int:
    path = ROOT / name
    image.save(path, format="PNG", compress_level=6, dpi=(144, 144))
    with Image.open(path) as check:
        assert check.size == (SIZE, SIZE), f"{name}: unexpected size {check.size}"
        assert check.format == "PNG", f"{name}: unexpected format {check.format}"
        assert check.mode == "RGB", f"{name}: unexpected mode {check.mode}"
    size_bytes = path.stat().st_size
    assert size_bytes > 30 * 1024, f"{name}: file is below 30 KB"
    return size_bytes


def main() -> None:
    assert SIZE == 900 and LOGICAL_RENDER_SIZE > SIZE and CROP_MARGIN == 250
    assert 28.0 <= ELEVATION_DEG <= 34.0 and AZIMUTH_DEG == -50.0
    assert GEORGIA_BOLD.exists() and GEORGIA.exists(), "required Georgia fonts are missing"

    minimum_xy, minimum_z = find_global_minimum()
    projection = make_projection(minimum_xy, minimum_z)
    trace = build_path()
    points = trace.points

    axis = np.linspace(-DOMAIN_RADIUS, DOMAIN_RADIUS, GRID_SIZE)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid_z = loss(grid_x, grid_y)
    grid_screen_x, grid_screen_y, _ = project(projection, grid_x, grid_y, grid_z)
    visible_nodes = (
        (grid_screen_x >= CROP_MARGIN)
        & (grid_screen_x <= CROP_MARGIN + SIZE)
        & (grid_screen_y >= CROP_MARGIN)
        & (grid_screen_y <= CROP_MARGIN + SIZE)
    )
    visible_values = grid_z[visible_nodes]
    low_threshold = float(np.quantile(visible_values, 0.05))
    visible_low_mask = visible_nodes & (grid_z <= low_threshold)

    jet, jet_coverage, path_box, jet_arrows, jet_colors = render_surface(JET, projection, points)
    blue, blue_coverage, blue_path_box, blue_arrows, blue_colors = render_surface(SITE, projection, points)
    assert blue_path_box == path_box and blue_arrows == jet_arrows
    metrics = validate_scene(
        projection,
        trace,
        path_box,
        jet_arrows,
        grid_z,
        visible_low_mask,
    )

    jet_titled, jet_title_box = add_title(jet, path_box)
    blue_titled, blue_title_box = add_title(blue, blue_path_box)
    assert jet_title_box == blue_title_box

    outputs = {
        "J1.png": (jet, jet_coverage, None),
        "J2.png": (jet_titled, jet_coverage, jet_title_box),
        "J3.png": (blue, blue_coverage, None),
        "J4.png": (blue_titled, blue_coverage, blue_title_box),
    }
    sizes: dict[str, int] = {}
    for name, (image, coverage, title_box) in outputs.items():
        sizes[name] = save_candidate(name, image)
        title_text = "none" if title_box is None else str(title_box)
        print(
            f"{name}: 900x900 RGB PNG; bytes={sizes[name]}; "
            f"surface_coverage={coverage:.3%}; path_bbox={path_box}; title_bbox={title_text}"
        )

    print(
        "C1 ASSERTED: "
        f"jet_coverage={jet_coverage:.3%}, blue_coverage={blue_coverage:.3%} (required >=97%)"
    )
    print(
        "C2 ASSERTED: "
        f"relief/extent={metrics['relative_relief']:.4f}, basin_components={metrics['basin_components']}, "
        f"projected_min=({metrics['minimum_x']:.1f}, {metrics['minimum_y']:.1f}), "
        f"elevation={ELEVATION_DEG:.1f}deg, azimuth={AZIMUTH_DEG:.1f}deg, "
        f"anisotropy={STEEP_CURVATURE / SHALLOW_CURVATURE:.1f}:1, "
        f"valley_axis={VALLEY_ANGLE_DEG:.1f}deg, domain_radius={DOMAIN_RADIUS:.1f}"
    )
    print(
        "C3 ASSERTED: "
        f"top_red_orange={jet_colors['top_red_orange']:.3%}, "
        f"middle_green_yellow={jet_colors['middle_green_yellow']:.3%}, "
        f"blue_pocket={jet_colors['basin_fraction']:.3%}, grid={GRID_SIZE}x{GRID_SIZE}, "
        f"wire_stride={WIREFRAME_STRIDE}, wire_lines={int(jet_colors['wire_lines'])}, "
        f"white_blend={COLOR_WHITE_BLEND:.0%}, normalization=smoothstep"
    )
    print(
        "C4 ASSERTED: "
        f"segments={metrics['segments']}, arrows={metrics['arrows']}, bbox={metrics['path_bbox']}, "
        f"width={metrics['path_width_fraction']:.3%}, height={metrics['path_height_fraction']:.3%}, "
        f"early_min_turn={metrics['early_min_turn_angle']:.1f}deg, "
        f"min_horizontal_ratio={metrics['min_horizontal_ratio']:.2f}, eta={metrics['learning_rate']:.4f}, "
        f"loss={metrics['loss_start']:.6f}->{metrics['loss_end']:.6f}, "
        f"nonmonotonic_bumps={metrics['nonmonotonic_bumps']}, "
        f"endpoint_distance={metrics['endpoint_distance']:.4f}, "
        f"max_gd_residual={metrics['max_gd_residual']:.2e}, "
        f"overlay_verified={str(metrics['overlay_verified']).lower()}"
    )
    print(
        "C5 ASSERTED: "
        f"Georgia Bold=96px, shadow=3px, title_bbox={jet_title_box}, path_bbox={path_box}, overlap=false"
    )


if __name__ == "__main__":
    main()
