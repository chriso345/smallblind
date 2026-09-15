from __future__ import annotations

import numpy as np
import pytest

from smallblind.cvd import (
    DEFAULT_SEVERITY,
    CVDType,
    get_matrix,
    get_severity,
)


def test_cvd_type_is_str_enum() -> None:
    assert isinstance(CVDType.PROTANOPIA, str)
    assert CVDType.PROTANOPIA == "protanopia"
    assert CVDType("deuteranomaly") is CVDType.DEUTERANOMALY


def test_cvd_type_accepts_all_literal_values() -> None:
    values = [
        "protanopia",
        "protanomaly",
        "deuteranopia",
        "deuteranomaly",
        "tritanopia",
        "tritanomaly",
        "achromatopsia",
        "achromatomaly",
    ]
    for value in values:
        assert CVDType(value).value == value


def test_default_severity_covers_all_types() -> None:
    for cvd_type in CVDType:
        s = DEFAULT_SEVERITY[cvd_type]
        assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("cvd_type", list(CVDType))
def test_get_severity_falls_back_to_default(cvd_type: CVDType) -> None:
    assert get_severity(cvd_type, None) == DEFAULT_SEVERITY[cvd_type]


@pytest.mark.parametrize("cvd_type", list(CVDType))
def test_get_severity_passes_through_explicit_value(cvd_type: CVDType) -> None:
    assert get_severity(cvd_type, 0.42) == 0.42


@pytest.mark.parametrize("bad_severity", [-0.1, 1.1, 2.0, -5.0])
def test_get_severity_rejects_out_of_range(bad_severity: float) -> None:
    with pytest.raises(ValueError):
        get_severity(CVDType.PROTANOPIA, bad_severity)


@pytest.mark.parametrize("cvd_type", list(CVDType))
def test_get_matrix_severity_zero_is_identity(cvd_type: CVDType) -> None:
    matrix = get_matrix(cvd_type, severity=0.0)
    assert np.allclose(matrix, np.eye(3))


def test_get_matrix_severity_one_is_full_dichromacy() -> None:
    matrix = get_matrix(CVDType.PROTANOPIA, severity=1.0)
    # Should not be the identity, and should match the full-severity matrix
    # used directly (protanomaly at severity 1.0 uses the same base matrix).
    matrix_via_anomaly = get_matrix(CVDType.PROTANOMALY, severity=1.0)
    assert not np.allclose(matrix, np.eye(3))
    assert np.allclose(matrix, matrix_via_anomaly)


def test_get_matrix_interpolates_linearly() -> None:
    full = get_matrix(CVDType.DEUTERANOPIA, severity=1.0)
    half = get_matrix(CVDType.DEUTERANOMALY, severity=0.5)
    expected = np.eye(3) + 0.5 * (full - np.eye(3))
    assert np.allclose(half, expected)


def test_achromatopsia_matrix_collapses_to_gray() -> None:
    matrix = get_matrix(CVDType.ACHROMATOPSIA, severity=1.0)
    color = np.array([0.9, 0.1, 0.3])
    result = matrix @ color
    assert np.allclose(result[0], result[1])
    assert np.allclose(result[1], result[2])


def test_get_matrix_default_severity_used_when_none() -> None:
    for cvd_type in CVDType:
        explicit = get_matrix(cvd_type, DEFAULT_SEVERITY[cvd_type])
        implicit = get_matrix(cvd_type, None)
        assert np.allclose(explicit, implicit)
