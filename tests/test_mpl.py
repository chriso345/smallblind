from __future__ import annotations

import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from smallblind import simulate_figure


@pytest.fixture
def simple_figure():
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1, 2], [0, 1, 0], color=(0.9, 0.1, 0.1))
    ax.plot([0, 1, 2], [1, 0, 1], color=(0.1, 0.9, 0.1))
    yield fig
    plt.close(fig)


def test_simulate_figure_returns_pil_image(simple_figure) -> None:
    result = simulate_figure(simple_figure, "protanopia")
    assert isinstance(result, Image.Image)


def test_simulate_figure_accepts_dpi(simple_figure) -> None:
    low_dpi = simulate_figure(simple_figure, "deuteranopia", dpi=50)
    high_dpi = simulate_figure(simple_figure, "deuteranopia", dpi=150)
    assert high_dpi.size[0] > low_dpi.size[0]


def test_simulate_figure_does_not_modify_original(simple_figure) -> None:
    original_lines = len(simple_figure.axes[0].lines)
    simulate_figure(simple_figure, "tritanopia")
    assert len(simple_figure.axes[0].lines) == original_lines
