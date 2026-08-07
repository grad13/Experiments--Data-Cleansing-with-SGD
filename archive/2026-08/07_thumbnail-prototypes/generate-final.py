from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
OUTPUT = REPOSITORY_ROOT / "site" / ".face" / "thumbnail.png"


def load_generate_q():
    source = ROOT / "generate-Q.py"
    spec = importlib.util.spec_from_file_location("generate_q_for_final", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    generate_q = load_generate_q()
    generate_n = generate_q.load_generate_n()
    raw = generate_n.build_surface(generate_n.DARK, generate_n.simulate_sgd("N1"))

    generate_q.SUBTITLE = "Reproduction Study"
    final = generate_q.q5_magazine_cover(
        raw,
        generate_n.GEORGIA_BOLD,
        generate_n.GEORGIA,
    )
    assert final.size == (900, 900)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUTPUT, format="PNG", compress_level=6, dpi=(144, 144))
    with Image.open(OUTPUT) as check:
        assert check.size == (900, 900)
        assert check.format == "PNG"
    print(f"{OUTPUT}: {OUTPUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
