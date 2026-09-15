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
    source: str | np.ndarray
    if len(sys.argv) > 1:
        source = sys.argv[1]
        label = Path(source).stem
    else:
        source = make_swatch_image()
        label = "swatch"

    OUTPUT_DIR.mkdir(exist_ok=True)

    for cvd_type in sb.CVDType:
        result = sb.simulate(source, cvd_type)
        out_path = OUTPUT_DIR / f"{label}_{cvd_type.value}.png"
        result.save(out_path)
        print(f"saved {out_path}")

    # Also demonstrate partial severity for an anomalous type.
    mild = sb.simulate(source, "deuteranomaly", severity=0.3)
    mild_path = OUTPUT_DIR / f"{label}_deuteranomaly_mild.png"
    mild.save(mild_path)
    print(f"saved {mild_path}")


if __name__ == "__main__":
    main()
