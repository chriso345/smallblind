from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PIL import Image

from .cvd import CVDLiteral, CVDType
from .image import simulate_image

if TYPE_CHECKING:
    from matplotlib.figure import Figure

try:
    import matplotlib.figure  # noqa: F401, pyrefly: ignore[unused-import]
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for smallblind.mpl; install it with 'pip install smallblind[mpl]'"
    ) from exc


def simulate_figure(
    fig: Figure,
    cvd: CVDType | CVDLiteral,
    severity: float | None = None,
    dpi: float | None = None,
) -> Image.Image:
    """Rasterize a matplotlib Figure and simulate a CVD type on it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    with Image.open(buf) as rendered:
        rendered.load()
        return simulate_image(rendered, cvd, severity)
