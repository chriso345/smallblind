from __future__ import annotations

from typing import TYPE_CHECKING

from .compare import compare
from .cvd import CVDLiteral, CVDType
from .image import ImageInput, simulate_array, simulate_image
from .palette import ColorTuple, simulate_palette
from .safety import (
    DEFAULT_THRESHOLD,
    ConfusionPair,
    SafetyReport,
    check_palette_safety,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from PIL.Image import Image as PILImage

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_THRESHOLD",
    "CVDLiteral",
    "CVDType",
    "ColorTuple",
    "ConfusionPair",
    "ImageInput",
    "SafetyReport",
    "check_palette_safety",
    "compare",
    "simulate",
    "simulate_array",
    "simulate_figure",
    "simulate_image",
    "simulate_palette",
]


def simulate(
    image: ImageInput,
    cvd: CVDType | CVDLiteral,
    severity: float | None = None,
) -> PILImage:
    """Simulate a CVD type on an image. Alias for `simulate_image`."""
    return simulate_image(image, cvd, severity)


def simulate_figure(
    fig: Figure,
    cvd: CVDType | CVDLiteral,
    severity: float | None = None,
    dpi: float | None = None,
) -> PILImage:
    """Simulate a CVD type on a matplotlib Figure."""
    from .mpl import simulate_figure as _simulate_figure

    return _simulate_figure(fig, cvd, severity, dpi)
