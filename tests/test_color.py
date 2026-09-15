from __future__ import annotations

import numpy as np

from smallblind.color import apply_matrix, linear_to_srgb, srgb_to_linear


def test_srgb_linear_round_trip() -> None:
    values = np.linspace(0.0, 1.0, 101)
    linear = srgb_to_linear(values)
    back = linear_to_srgb(linear)
    assert np.allclose(back, values, atol=1e-8)


def test_srgb_to_linear_endpoints() -> None:
    assert np.isclose(srgb_to_linear(np.array(0.0)), 0.0)
    assert np.isclose(srgb_to_linear(np.array(1.0)), 1.0)


def test_srgb_to_linear_is_monotonic() -> None:
    values = np.linspace(0.0, 1.0, 50)
    linear = srgb_to_linear(values)
    assert np.all(np.diff(linear) >= 0)


def test_apply_matrix_identity_preserves_color() -> None:
    colors = np.array([[0.9, 0.1, 0.1], [0.2, 0.8, 0.4]])
    result = apply_matrix(colors, np.eye(3))
    assert np.allclose(result, colors, atol=1e-6)


def test_apply_matrix_clips_to_valid_range() -> None:
    # An exaggerated matrix that would push values out of [0, 1].
    exaggerate = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    colors = np.array([[0.9, 0.9, 0.9]])
    result = apply_matrix(colors, exaggerate)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_apply_matrix_supports_image_shaped_input() -> None:
    image = np.random.default_rng(0).random((4, 5, 3))
    result = apply_matrix(image, np.eye(3))
    assert result.shape == image.shape
    assert np.allclose(result, image, atol=1e-6)
