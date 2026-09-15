from __future__ import annotations

import numpy as np


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert gamma-encoded sRGB values in [0, 1] to linear-light RGB."""
    a = 0.055
    low = srgb / 12.92
    high = ((srgb + a) / (1 + a)) ** 2.4
    return np.where(srgb <= 0.04045, low, high)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light RGB values in [0, 1] back to gamma-encoded sRGB."""
    a = 0.055
    linear = np.clip(linear, 0.0, 1.0)
    low = linear * 12.92
    high = (1 + a) * np.power(linear, 1 / 2.4) - a
    return np.where(linear <= 0.0031308, low, high)


def apply_matrix(rgb01: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 linear-RGB transform matrix to an array of sRGB colors."""
    linear = srgb_to_linear(rgb01)
    transformed = linear @ matrix.T
    srgb = linear_to_srgb(transformed)
    return np.clip(srgb, 0.0, 1.0)
