from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np

# Literal alias mirroring CVDType's values, for callers who prefer plain
# strings over the enum (e.g. `simulate(img, "deuteranopia")`).
CVDLiteral = Literal[
    "protanopia",
    "protanomaly",
    "deuteranopia",
    "deuteranomaly",
    "tritanopia",
    "tritanomaly",
    "achromatopsia",
    "achromatomaly",
]


class CVDType(str, Enum):
    """Supported color vision deficiency types.

    The "-opia" types are complete dichromacies while
    the "-anomaly" types are partial.
    """

    PROTANOPIA = "protanopia"
    PROTANOMALY = "protanomaly"
    DEUTERANOPIA = "deuteranopia"
    DEUTERANOMALY = "deuteranomaly"
    TRITANOPIA = "tritanopia"
    TRITANOMALY = "tritanomaly"
    ACHROMATOPSIA = "achromatopsia"
    ACHROMATOMALY = "achromatomaly"


_BASE_TYPE: dict[CVDType, CVDType] = {
    CVDType.PROTANOPIA: CVDType.PROTANOPIA,
    CVDType.PROTANOMALY: CVDType.PROTANOPIA,
    CVDType.DEUTERANOPIA: CVDType.DEUTERANOPIA,
    CVDType.DEUTERANOMALY: CVDType.DEUTERANOPIA,
    CVDType.TRITANOPIA: CVDType.TRITANOPIA,
    CVDType.TRITANOMALY: CVDType.TRITANOPIA,
    CVDType.ACHROMATOPSIA: CVDType.ACHROMATOPSIA,
    CVDType.ACHROMATOMALY: CVDType.ACHROMATOPSIA,
}

DEFAULT_SEVERITY: dict[CVDType, float] = {
    CVDType.PROTANOPIA: 1.0,
    CVDType.PROTANOMALY: 0.6,
    CVDType.DEUTERANOPIA: 1.0,
    CVDType.DEUTERANOMALY: 0.6,
    CVDType.TRITANOPIA: 1.0,
    CVDType.TRITANOMALY: 0.6,
    CVDType.ACHROMATOPSIA: 1.0,
    CVDType.ACHROMATOMALY: 0.6,
}

# Values derived from Machado et al. (2009) and Guijosa (2020). The matrices
# are for linear-light RGB, so they should be applied after converting from
# gamma-encoded sRGB to linear-light RGB, and the result should be converted
# back to sRGB afterward.
#
# See: Guijosa, A. R. (2020). ALGORITMOS PARA MEJORAR LA EXPERIENCIA VISUAL DE
#  PERSONAS CON DALTONISMO.
_FULL_MATRIX: dict[CVDType, np.ndarray] = {
    CVDType.PROTANOPIA: np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    CVDType.DEUTERANOPIA: np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    CVDType.TRITANOPIA: np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}

# Rec. 709 luma weights, used to build the full achromatopsia (total
# color blindness) matrix.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
_ACHROMATOPSIA_MATRIX = np.tile(_LUMA_WEIGHTS, (3, 1))

_IDENTITY = np.eye(3)


def get_severity(cvd: CVDType, severity: float | None) -> float:
    """Resolve the severity to use for `cvd`, falling back to a default."""
    if severity is None:
        return DEFAULT_SEVERITY[cvd]

    # TODO: Add a typing annotated check for severity to ensure it's a float between 0.0 and 1.0
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be between 0.0 and 1.0")
    return severity


def get_matrix(cvd: CVDType, severity: float | None = None) -> np.ndarray:
    """Return the 3x3 linear-RGB transform matrix for a CVD type."""
    base = _BASE_TYPE[cvd]
    s = get_severity(cvd, severity)
    full = (
        _ACHROMATOPSIA_MATRIX if base is CVDType.ACHROMATOPSIA else _FULL_MATRIX[base]
    )
    if s >= 1.0:
        return full
    if s <= 0.0:
        return _IDENTITY
    return _IDENTITY + s * (full - _IDENTITY)
