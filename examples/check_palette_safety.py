from __future__ import annotations

import smallblind as sb

RISKY_PALETTE = [
    (0.85, 0.10, 0.10),  # red
    (0.10, 0.75, 0.10),  # green
    (0.80, 0.60, 0.10),  # olive, close to both under some CVD types
]

SAFER_PALETTE = [
    (0.90, 0.10, 0.10),  # red
    (0.10, 0.40, 0.90),  # blue
    (1.00, 0.85, 0.10),  # yellow
]


def report_on(name: str, palette: list[tuple[float, float, float]]) -> None:
    report = sb.check_palette_safety(palette)
    print(f"\n{name}: {'safe' if report.is_safe else 'has confusable colors'}")
    for warning in report.warnings:
        print(f"  - {warning}")


def main() -> None:
    report_on("risky palette", RISKY_PALETTE)
    report_on("safer palette", SAFER_PALETTE)


if __name__ == "__main__":
    main()
