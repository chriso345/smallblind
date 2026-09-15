from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .cvd import CVDLiteral, CVDType
from .image import ImageInput, _load_as_array, simulate_image

_ALL_CVD_TYPES: tuple = tuple(CVDType)
_COMMON_CVD_TYPES: tuple = (
    CVDType.PROTANOPIA,
    CVDType.DEUTERANOPIA,
    CVDType.TRITANOPIA,
)

_LABEL_HEIGHT = 20


def compare(
    image: ImageInput,
    cvd: Sequence[CVDType | CVDLiteral] | str = "common",
    severity: float | None = None,
    columns: int | None = None,
    labels: bool = True,
) -> Image.Image:
    """Build a grid comparing the original image against CVD simulations."""
    if cvd == "all":
        cvd_types: tuple = _ALL_CVD_TYPES
    elif cvd == "common":
        cvd_types = _COMMON_CVD_TYPES
    else:
        cvd_types = tuple(CVDType(c) for c in cvd)  # type: ignore[union-attr]

    base_arr, has_alpha = _load_as_array(image)
    base_uint8 = np.clip(base_arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    base_img = Image.fromarray(base_uint8, mode="RGBA" if has_alpha else "RGB")

    panels: list[tuple[str, Image.Image]] = [("original", base_img)]
    for cvd_type in cvd_types:
        panels.append((cvd_type.value, simulate_image(image, cvd_type, severity)))

    n = len(panels)
    cols = columns or n
    rows = (n + cols - 1) // cols

    w, h = base_img.size
    label_h = _LABEL_HEIGHT if labels else 0
    cell_h = h + label_h
    grid = Image.new("RGB", (w * cols, cell_h * rows), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid) if labels else None

    for i, (name, panel) in enumerate(panels):
        row, col = divmod(i, cols)
        x, y = col * w, row * cell_h
        paste_target = panel.convert("RGB") if panel.mode == "RGBA" else panel
        grid.paste(paste_target, (x, y + label_h))
        if draw is not None:
            draw.text((x + 4, y + 4), name, fill=(0, 0, 0))

    return grid
