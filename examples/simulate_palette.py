from __future__ import annotations

import smallblind as sb


def get_palette() -> list[tuple[float, float, float]]:
    try:
        # Use the `smallblind[sns]` extra to get seaborn for examples.
        import seaborn as sns

        return sns.color_palette("Set1")
    except ImportError:
        # A reasonable stand-in for seaborn's "Set1" palette.
        return [
            (0.894, 0.102, 0.110),
            (0.216, 0.494, 0.722),
            (0.302, 0.686, 0.290),
            (0.596, 0.306, 0.639),
            (1.000, 0.498, 0.000),
        ]


def main() -> None:
    palette = get_palette()
    print("original palette:")
    for color in palette:
        print(f"  {tuple(round(c, 3) for c in color)}")

    for cvd_type in (
        sb.CVDType.PROTANOPIA,
        sb.CVDType.DEUTERANOPIA,
        sb.CVDType.TRITANOPIA,
    ):
        simulated = sb.simulate_palette(palette, cvd_type)
        print(f"\nunder {cvd_type.value}:")
        for color in simulated:
            print(f"  {tuple(round(c, 3) for c in color)}")


if __name__ == "__main__":
    main()
