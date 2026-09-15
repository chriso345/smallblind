from __future__ import annotations

from smallblind import CVDType, check_palette_safety
from smallblind.safety import ConfusionPair, SafetyReport


def test_distinct_palette_is_reported_safe(sample_palette) -> None:
    report = check_palette_safety(sample_palette)
    assert isinstance(report, SafetyReport)
    assert report.is_safe
    assert report.warnings == []


def test_identical_colors_are_never_safe() -> None:
    colors = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    report = check_palette_safety(colors, cvd_types=["deuteranopia"])
    assert not report.is_safe
    assert len(report.confusions) == 1
    pair = report.confusions[0]
    assert isinstance(pair, ConfusionPair)
    assert pair.distance == 0.0
    assert pair.cvd == CVDType.DEUTERANOPIA


def test_warnings_are_human_readable_strings() -> None:
    colors = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    report = check_palette_safety(colors, cvd_types=["deuteranopia"])
    assert len(report.warnings) == 1
    message = report.warnings[0]
    assert "0" in message and "1" in message
    assert "deuteranopia" in message


def test_check_palette_safety_defaults_to_three_dichromacies() -> None:
    colors = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    report = check_palette_safety(colors)
    found_types = {pair.cvd for pair in report.confusions}
    assert found_types == {
        CVDType.PROTANOPIA,
        CVDType.DEUTERANOPIA,
        CVDType.TRITANOPIA,
    }


def test_higher_threshold_flags_more_pairs() -> None:
    colors = [(0.9, 0.1, 0.1), (0.8, 0.2, 0.15)]
    lenient = check_palette_safety(colors, cvd_types=["protanopia"], threshold=1.0)
    strict = check_palette_safety(colors, cvd_types=["protanopia"], threshold=200.0)
    assert len(lenient.confusions) <= len(strict.confusions)


def test_single_color_palette_has_no_pairs() -> None:
    report = check_palette_safety([(0.1, 0.2, 0.3)])
    assert report.is_safe
