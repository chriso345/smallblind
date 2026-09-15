from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from smallblind import compare
from smallblind.cvd import CVDType


def test_compare_common_has_four_panels(sample_image_array: np.ndarray) -> None:
    grid = compare(sample_image_array, cvd="common", labels=False)
    w, h = sample_image_array.shape[1], sample_image_array.shape[0]
    assert isinstance(grid, Image.Image)
    # original + protanopia + deuteranopia + tritanopia = 4 panels, one row.
    assert grid.size == (w * 4, h)


def test_compare_all_has_one_panel_per_cvd_type_plus_original(
    sample_image_array: np.ndarray,
) -> None:
    grid = compare(sample_image_array, cvd="all", labels=False)
    w = sample_image_array.shape[1]
    expected_panels = len(list(CVDType)) + 1
    assert grid.size[0] == w * expected_panels


def test_compare_accepts_explicit_type_list(sample_image_array: np.ndarray) -> None:
    grid = compare(sample_image_array, cvd=["protanopia", "tritanopia"], labels=False)
    w = sample_image_array.shape[1]
    assert grid.size[0] == w * 3  # original + 2 requested types


def test_compare_columns_wraps_into_rows(sample_image_array: np.ndarray) -> None:
    grid = compare(sample_image_array, cvd="common", columns=2, labels=False)
    w, h = sample_image_array.shape[1], sample_image_array.shape[0]
    # 4 panels at 2 columns -> 2 rows.
    assert grid.size == (w * 2, h * 2)


def test_compare_with_labels_adds_height(sample_image_array: np.ndarray) -> None:
    no_labels = compare(sample_image_array, cvd="common", labels=False)
    with_labels = compare(sample_image_array, cvd="common", labels=True)
    assert with_labels.size[1] > no_labels.size[1]
    assert with_labels.size[0] == no_labels.size[0]


def test_compare_rejects_unknown_cvd_type(sample_image_array: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compare(sample_image_array, cvd=["not-a-real-cvd-type"])  # type: ignore[list-item]
