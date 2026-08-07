from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "generate-K.py"
PYTHON = "/Users/takuo-h/.local/share/pyenv/versions/3.14.5/bin/python3"


def load_k_module():
    spec = importlib.util.spec_from_file_location("thumbnail_generate_k", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


k = load_k_module()
gm = k.gm


def evenly_spaced_stops(*colors: str) -> tuple[tuple[float, str], ...]:
    return tuple((index / (len(colors) - 1), color) for index, color in enumerate(colors))


@dataclass(frozen=True)
class Variant(k.Variant):
    path_color: str = "#000000"
    height_mapping: str = "dark-in-valley"


VARIANTS = (
    Variant(
        "L1.png",
        "Muted viridis: deep violet valley through blue-green to soft yellow ridges.",
        stops=evenly_spaced_stops("#46327E", "#3B6C8C", "#4C9A78", "#9CC163", "#E8DD64"),
        white_blend=0.08,
        path_color="#000000",
        height_mapping="dark-in-valley",
    ),
    Variant(
        "L2.png",
        "Earth-map tints: olive lowlands, sand and tan slopes, brown highlands.",
        stops=evenly_spaced_stops("#8AA37B", "#C9BC8F", "#C89A6B", "#A9714B", "#8C5B45"),
        white_blend=0.10,
        path_color="#2B2B2B",
        height_mapping="olive-in-valley/brown-on-ridges",
    ),
    Variant(
        "L3.png",
        "Cool blue-gray: a light valley grading to dark slate ridges with an orange trace.",
        stops=evenly_spaced_stops("#E9EEF3", "#C4D2DE", "#93AEC4", "#5D82A3", "#37536F"),
        white_blend=0.0,
        path_color="#B0521D",
        height_mapping="light-in-valley/dark-on-ridges",
    ),
    Variant(
        "L4.png",
        "Pastel warm gray: paper-light valley to softly shaded taupe ridges.",
        stops=evenly_spaced_stops("#F3F1ED", "#DDD5CB", "#C3B4A6", "#A38D7D", "#7E6A5C"),
        white_blend=0.0,
        path_color="#C0392B",
        height_mapping="light-in-valley/dark-on-ridges",
    ),
    Variant(
        "L5.png",
        "Deep ocean: navy valley, teal slopes, and pale foam ridges with a warm-orange trace.",
        stops=evenly_spaced_stops("#0E2A47", "#175E73", "#4E9C8F", "#CBE3D6"),
        white_blend=0.0,
        path_color="#E8944A",
        height_mapping="dark-in-valley/light-on-ridges",
    ),
    Variant(
        "L6.png",
        "Grayscale paper: white valley to graphite ridges, with the zigzag as the only color accent.",
        stops=evenly_spaced_stops("#FAFAFA", "#E0E0E0", "#BDBDBD", "#8F8F8F", "#5F5F5F"),
        white_blend=0.0,
        path_color="#B0521D",
        height_mapping="light-in-valley/dark-on-ridges",
    ),
)


def configure(variant: Variant) -> None:
    """Use generate-K's J2-default scene configuration unchanged."""
    k.configure(variant)
    assert gm.AZIMUTH_DEG == -50.0
    assert gm.VALLEY_ANGLE_DEG == 104.5
    assert gm.PIXELS_PER_WORLD == 72.0
    assert np.array_equal(gm.TARGET_MINIMUM, np.array((0.55, 0.68)) * gm.SIZE)
    assert gm.STEEP_CURVATURE == 0.030
    assert gm.SHALLOW_CURVATURE == gm.STEEP_CURVATURE / 8.0
    assert gm.HEIGHT_SATURATION == 2.5


def draw_trace(
    image: Image.Image,
    projection,
    points: np.ndarray,
    path_style: str,
    path_color: str,
):
    """Draw generate-K's exact trace geometry with a palette-specific solid color."""
    assert path_style == "black"
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
    color = tuple(int(channel) for channel in gm.hex_rgb(path_color)) + (255,)
    inner_width = 10 * gm.SUPERSAMPLE

    for start, end in zip(screen_points[:-1], screen_points[1:]):
        triangle = gm.arrowhead_polygon(start, end)
        draw.line((start, end), fill=color, width=inner_width)
        draw.polygon(triangle, fill=color)
        mask_draw.line((start, end), fill=255, width=inner_width)
        mask_draw.polygon(triangle, fill=255)

    end_x, end_y = screen_points[-1]
    radius = 9 * gm.SUPERSAMPLE
    draw.ellipse(
        (end_x - radius, end_y - radius, end_x + radius, end_y + radius),
        fill=color,
    )
    mask_draw.ellipse(
        (end_x - radius, end_y - radius, end_x + radius, end_y + radius),
        fill=255,
    )

    composited = Image.alpha_composite(image, layer)
    output_mask = gm.crop_and_downsample(mask, resample=Image.Resampling.NEAREST)
    return composited, gm.path_bbox_from_mask(output_mask), len(points) - 1


def render(variant: Variant, projection, points: np.ndarray):
    """Reuse generate-K's surface renderer while injecting only the path color."""
    original_draw_trace = k.draw_trace
    try:
        k.draw_trace = lambda image, proj, trace_points, style: draw_trace(
            image, proj, trace_points, style, variant.path_color
        )
        return k.render_surface(variant, projection, points)
    finally:
        k.draw_trace = original_draw_trace


def main() -> None:
    assert sys.executable == PYTHON, f"Use the requested interpreter: {PYTHON}"
    assert gm.GEORGIA_BOLD.exists() and gm.GEORGIA.exists()
    reference_points: np.ndarray | None = None
    reference_path_box: tuple[int, int, int, int] | None = None
    reference_title_box: tuple[int, int, int, int] | None = None

    for variant in VARIANTS:
        configure(variant)
        minimum_xy, minimum_z = gm.find_global_minimum()
        projection = gm.make_projection(minimum_xy, minimum_z)
        trace = k.build_trace(variant)
        surface, coverage, path_box, arrows, relief = render(variant, projection, trace.points)
        titled, title_box = gm.add_title(surface, path_box)

        if reference_points is None:
            reference_points = trace.points.copy()
            reference_path_box = path_box
            reference_title_box = title_box
        else:
            assert np.array_equal(trace.points, reference_points)
            assert path_box == reference_path_box
            assert title_box == reference_title_box

        residuals = np.diff(trace.points, axis=0) + trace.learning_rate * trace.gradients
        max_residual = float(np.max(np.linalg.norm(residuals, axis=1)))
        assert coverage >= 0.97
        assert arrows == gm.GD_STEPS == 11
        assert max_residual <= 1e-12
        assert not gm.boxes_overlap(title_box, path_box)
        assert variant.title_position == "top"
        assert variant.azimuth == -50.0 and variant.valley_angle == 104.5
        assert variant.start_along == -8.5 and variant.start_across == 3.5
        assert variant.target_minimum == (0.55, 0.68)
        assert variant.pixels_per_world == 72.0 and variant.relief_scale == 1.0

        size = k.save(variant.name, titled)
        print(
            f"{variant.name}: 900x900 RGB PNG; bytes={size}; surface_coverage={coverage:.3%}; "
            f"path_bbox={path_box}; title_bbox={title_box}; overlap=false; arrows={arrows}; "
            f"max_gd_residual={max_residual:.2e}; azimuth={variant.azimuth:.1f}deg; "
            f"valley_axis={variant.valley_angle:.1f}deg; normalization=smoothstep; "
            f"white_blend={variant.white_blend:.0%}; mapping={variant.height_mapping}; "
            f"path={variant.path_color}; wire=rgba(37,42,48,52); relief={relief:.4f}"
        )


if __name__ == "__main__":
    main()
