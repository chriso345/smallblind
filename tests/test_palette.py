from __future__ import annotations

import pytest

from smallblind import CVDType, simulate_palette


def test_simulate_palette_preserves_length(sample_palette) -> None:
    result = simulate_palette(sample_palette, "deuteranopia")
    assert len(result) == len(sample_palette)


def test_simulate_palette_returns_rgb_tuples_in_range(sample_palette) -> None:
    result = simulate_palette(sample_palette, "protanopia")
    for color in result:
        assert len(color) == 3
        for channel in color:
            assert 0.0 <= channel <= 1.0


def test_simulate_palette_accepts_enum_and_string(sample_palette) -> None:
    from_enum = simulate_palette(sample_palette, CVDType.TRITANOPIA)
    from_str = simulate_palette(sample_palette, "tritanopia")
    assert from_enum == from_str


def test_simulate_palette_preserves_alpha() -> None:
    colors = [(0.9, 0.1, 0.1, 0.5), (0.1, 0.9, 0.1, 1.0)]
    result = simulate_palette(colors, "deuteranopia")
    assert len(result) == 2
    for original, simulated in zip(colors, result, strict=True):
        assert len(simulated) == 4
        assert simulated[3] == original[3]


def test_simulate_palette_severity_zero_is_identity(sample_palette) -> None:
    result = simulate_palette(sample_palette, "protanomaly", severity=0.0)
    for original, simulated in zip(sample_palette, result, strict=True):
        assert simulated == pytest.approx(original, abs=1e-6)


def test_simulate_palette_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        simulate_palette([(0.1, 0.2)], "protanopia")


def test_simulate_palette_achromatopsia_removes_hue() -> None:
    colors = [(0.9, 0.1, 0.1)]
    result = simulate_palette(colors, "achromatopsia")
    r, g, b = result[0]
    assert r == pytest.approx(g, abs=1e-6)
    assert g == pytest.approx(b, abs=1e-6)
