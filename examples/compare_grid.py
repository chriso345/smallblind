"""Build a side-by-side CVD comparison grid and save it as an image.

Run with:

    python examples/compare_grid.py

Pass a real image path as the first argument to use that instead of
the generated test swatch:

    python examples/compare_grid.py photo.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import smallblind as sb

OUTPUT_DIR = Path(__file__).parent / "output"


def make_swatch_image(size: int = 200) -> np.ndarray:
    bands = [
        (230, 25, 25),  # red
        (25, 200, 25),  # green
        (25, 25, 230),  # blue
        (230, 230, 25),  # yellow
    ]
    band_height = size // len(bands)
    image = np.zeros((band_height * len(bands), size, 3), dtype=np.uint8)
    for i, color in enumerate(bands):
        image[i * band_height : (i + 1) * band_height, :] = color
    return image


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else make_swatch_image()

    OUTPUT_DIR.mkdir(exist_ok=True)

    grid_common = sb.compare(source, cvd="common")
    common_path = OUTPUT_DIR / "compare_common.png"
    grid_common.save(common_path)
    print(f"saved {common_path}")

    grid_all = sb.compare(source, cvd="all", columns=3)
    all_path = OUTPUT_DIR / "compare_all.png"
    grid_all.save(all_path)
    print(f"saved {all_path}")


if __name__ == "__main__":
    main()
