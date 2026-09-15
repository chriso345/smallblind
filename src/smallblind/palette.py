from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .color import apply_matrix
from .cvd import CVDLiteral, CVDType, get_matrix

ColorTuple = tuple[float, ...]


def _normalize_colors(colors: Sequence[ColorTuple]) -> tuple[np.ndarray, bool]:
    """Convert a sequence of RGB or RGBA colors (0-1 floats) to an array."""
    arr = np.asarray(colors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError("colors must be a sequence of RGB or RGBA tuples")
    has_alpha = arr.shape[1] == 4
    return arr[:, :3], has_alpha


def simulate_palette(
    colors: Sequence[ColorTuple],
    cvd: CVDType | CVDLiteral,
    severity: float | None = None,
) -> list[ColorTuple]:
    """Simulate a CVD type on a list of RGB or RGBA colors."""
    cvd_type = CVDType(cvd)
    rgb, has_alpha = _normalize_colors(colors)
    matrix = get_matrix(cvd_type, severity)
    simulated = apply_matrix(rgb, matrix)

    if has_alpha:
        alpha = np.asarray(colors, dtype=np.float64)[:, 3:4]
        simulated = np.concatenate([simulated, alpha], axis=1)

    return [tuple(row) for row in simulated.tolist()]
