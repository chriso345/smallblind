from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def sample_image_array(rng: np.random.Generator) -> np.ndarray:
    return (rng.random((16, 16, 3)) * 255).astype(np.uint8)


@pytest.fixture
def sample_palette() -> list[tuple[float, float, float]]:
    return [
        (0.90, 0.10, 0.10),
        (0.10, 0.90, 0.10),
        (0.10, 0.10, 0.90),
        (0.90, 0.90, 0.10),
    ]
