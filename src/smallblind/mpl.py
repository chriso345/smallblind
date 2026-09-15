from __future__ import annotations

import io

from matplotlib import figure
from PIL import Image

from .cvd import CVDLiteral, CVDType
from .image import simulate_image


def simulate_figure(
    fig: figure.Figure,
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
