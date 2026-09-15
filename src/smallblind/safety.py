from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .cvd import DEFAULT_SEVERITY, CVDLiteral, CVDType
from .palette import ColorTuple, simulate_palette

# The three dichromacies, considered by default when no explicit list of
# CVD types is given. They are the most common and most severe forms of
# color vision deficiency.
_DEFAULT_TYPES: tuple = (
    CVDType.PROTANOPIA,
    CVDType.DEUTERANOPIA,
    CVDType.TRITANOPIA,
)

# Below this "redmean" color distance (on a 0-255 scale), two colors are
# considered hard to tell apart. This is a rough heuristic threshold,
# not a rigorously calibrated perceptual value.
DEFAULT_THRESHOLD = 35.0


def _redmean_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Low-cost perceptual color distance (the "redmean" formula)."""
    r1, g1, b1 = c1[:3] * 255.0
    r2, g2, b2 = c2[:3] * 255.0
    r_bar = (r1 + r2) / 2.0
    d_r, d_g, d_b = r1 - r2, g1 - g2, b1 - b2
    weight_r = 2 + r_bar / 256.0
    weight_g = 4.0
    weight_b = 2 + (255 - r_bar) / 256.0
    return float(np.sqrt(weight_r * d_r**2 + weight_g * d_g**2 + weight_b * d_b**2))


@dataclass
class ConfusionPair:
    """A pair of palette colors that are hard to distinguish under a CVD type."""

    index_a: int
    index_b: int
    cvd: CVDType
    distance: float


@dataclass
class SafetyReport:
    """Result of checking a palette for CVD safety."""

    confusions: list = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """True if no confusable color pairs were found."""
        return len(self.confusions) == 0

    @property
    def warnings(self) -> list:
        """Human readable descriptions of each confusable pair found."""
        return [
            f"colors {c.index_a} and {c.index_b} are hard to distinguish "
            f"under {c.cvd.value} (distance={c.distance:.1f})"
            for c in self.confusions
        ]


def check_palette_safety(
    colors: Sequence[ColorTuple],
    cvd_types: Sequence[CVDType | CVDLiteral] | None = None,
    severity: float | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> SafetyReport:
    """Check a palette for colors that become confusable under CVD."""
    types = (
        _DEFAULT_TYPES if cvd_types is None else tuple(CVDType(c) for c in cvd_types)
    )

    report = SafetyReport()
    n = len(colors)
    for cvd_type in types:
        s = severity if severity is not None else DEFAULT_SEVERITY[cvd_type]
        simulated = simulate_palette(colors, cvd_type, s)
        sim_arr = [np.asarray(c, dtype=np.float64) for c in simulated]
        for i, j in combinations(range(n), 2):
            dist = _redmean_distance(sim_arr[i], sim_arr[j])
            if dist < threshold:
                report.confusions.append(ConfusionPair(i, j, cvd_type, dist))

    return report
