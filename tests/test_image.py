from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from smallblind import simulate_array, simulate_image
from smallblind.image import _load_as_array


def test_simulate_image_from_numpy_array(sample_image_array: np.ndarray) -> None:
    result = simulate_image(sample_image_array, "deuteranopia")
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (16, 16)


def test_simulate_image_from_pil_image(sample_image_array: np.ndarray) -> None:
    pil_image = Image.fromarray(sample_image_array)
    result = simulate_image(pil_image, "protanopia")
    assert result.size == pil_image.size


def test_simulate_image_from_file_path(
    sample_image_array: np.ndarray, tmp_path: Path
) -> None:
    path = tmp_path / "sample.png"
    Image.fromarray(sample_image_array).save(path)
    result = simulate_image(path, "tritanopia")
    assert result.size == (16, 16)


def test_simulate_image_from_file_path_as_str(
    sample_image_array: np.ndarray, tmp_path: Path
) -> None:
    path = tmp_path / "sample.png"
    Image.fromarray(sample_image_array).save(path)
    result = simulate_image(str(path), "tritanopia")
    assert result.size == (16, 16)


def test_simulate_image_preserves_alpha(sample_image_array: np.ndarray) -> None:
    rgba = np.dstack([sample_image_array, np.full((16, 16), 128, dtype=np.uint8)])
    result = simulate_image(rgba, "deuteranopia")
    assert result.mode == "RGBA"
    original_alpha = rgba[:, :, 3]
    result_alpha = np.asarray(result)[:, :, 3]
    assert np.array_equal(original_alpha, result_alpha)


def test_simulate_image_severity_zero_is_unchanged(
    sample_image_array: np.ndarray,
) -> None:
    result = simulate_image(sample_image_array, "protanomaly", severity=0.0)
    result_arr = np.asarray(result)
    assert np.allclose(result_arr, sample_image_array, atol=2)


def test_simulate_image_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        simulate_image(12345, "protanopia")  # type: ignore[arg-type]


def test_load_as_array_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        _load_as_array(np.zeros((4, 4)))


def test_simulate_array_returns_float_in_unit_range(
    sample_image_array: np.ndarray,
) -> None:
    result = simulate_array(sample_image_array, "tritanomaly", severity=0.5)
    assert result.dtype == np.float64
    assert result.shape == (16, 16, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_simulate_array_accepts_float_input_in_unit_range() -> None:
    float_image = np.random.default_rng(2).random((8, 8, 3))
    result = simulate_array(float_image, "deuteranopia")
    assert result.shape == float_image.shape


def test_simulate_image_and_simulate_array_agree(
    sample_image_array: np.ndarray,
) -> None:
    from_image = np.asarray(simulate_image(sample_image_array, "deuteranopia")) / 255.0
    from_array = simulate_array(sample_image_array, "deuteranopia")
    assert np.allclose(from_image, from_array, atol=1 / 255.0)


def test_simulate_image_aliases_to_simulate(
    sample_image_array: np.ndarray,
) -> None:
    from smallblind import simulate

    result1 = simulate(sample_image_array, "deuteranopia")
    result2 = simulate_image(sample_image_array, "deuteranopia")
    assert np.allclose(np.asarray(result1), np.asarray(result2))


def test_simulate_array_with_alpha_channel(sample_image_array: np.ndarray) -> None:
    rgba = np.dstack([sample_image_array, np.full((16, 16), 128, dtype=np.uint8)])
    result = simulate_array(rgba, "deuteranopia")
    assert result.shape == (16, 16, 4)
    original_alpha = rgba[:, :, 3]
    result_alpha = result[:, :, 3]
    assert np.allclose(original_alpha / 255.0, result_alpha, atol=1e-6)
