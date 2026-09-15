from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .color import apply_matrix
from .cvd import CVDLiteral, CVDType, get_matrix

ImageInput = str | Path | Image.Image | np.ndarray


def _load_as_array(image: ImageInput) -> tuple[np.ndarray, bool]:
    """Load `image` into a float array of shape (h, w, 3) or (h, w, 4)."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            mode = "RGBA" if opened.mode in ("RGBA", "LA", "PA") else "RGB"
            arr = np.asarray(opened.convert(mode), dtype=np.float64) / 255.0
    elif isinstance(image, Image.Image):
        mode = "RGBA" if image.mode in ("RGBA", "LA", "PA") else "RGB"
        arr = np.asarray(image.convert(mode), dtype=np.float64) / 255.0
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
    else:
        raise TypeError(f"unsupported image input type: {type(image)!r}")

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("image array must have shape (h, w, 3) or (h, w, 4)")

    return arr, arr.shape[2] == 4


def simulate_image(
    image: ImageInput,
    cvd: CVDType | CVDLiteral,
    severity: float | None = None,
) -> Image.Image:
    """Simulate a CVD type on an image and return a new PIL Image."""
    cvd_type = CVDType(cvd)
    arr, has_alpha = _load_as_array(image)
    matrix = get_matrix(cvd_type, severity)

    simulated_rgb = apply_matrix(arr[:, :, :3], matrix)

    if has_alpha:
        out = np.concatenate([simulated_rgb, arr[:, :, 3:4]], axis=2)
        mode = "RGBA"
    else:
        out = simulated_rgb
        mode = "RGB"

    out_uint8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out_uint8, mode=mode)


def simulate_array(
    array: np.ndarray,
    cvd: CVDType | CVDLiteral,
    severity: float | None = None,
) -> np.ndarray:
    """Like `simulate_image`, but returns a float numpy array in [0, 1]."""
    cvd_type = CVDType(cvd)
    arr, has_alpha = _load_as_array(array)
    matrix = get_matrix(cvd_type, severity)

    simulated_rgb = apply_matrix(arr[:, :, :3], matrix)

    if has_alpha:
        return np.concatenate([simulated_rgb, arr[:, :, 3:4]], axis=2)
    return simulated_rgb
