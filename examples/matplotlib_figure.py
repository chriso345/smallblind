from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import smallblind as sb

OUTPUT_DIR = Path(__file__).parent / "output"


def make_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    x = [0, 1, 2, 3, 4]
    ax.plot(x, [0, 1, 0, 1, 0], color=(0.90, 0.10, 0.10), label="red line", linewidth=3)
    ax.plot(
        x, [1, 0, 1, 0, 1], color=(0.10, 0.75, 0.10), label="green line", linewidth=3
    )
    ax.plot(
        x,
        [0.5, 0.5, 1.5, 1.5, 0.5],
        color=(0.10, 0.40, 0.90),
        label="blue line",
        linewidth=3,
    )
    ax.legend()
    ax.set_title("Example plot with a red/green line pair")
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig = make_figure()

    original_path = OUTPUT_DIR / "figure_original.png"
    fig.savefig(original_path)
    print(f"saved {original_path}")

    for cvd_type in (sb.CVDType.PROTANOPIA, sb.CVDType.DEUTERANOPIA):
        simulated = sb.simulate_figure(fig, cvd_type)
        out_path = OUTPUT_DIR / f"figure_{cvd_type.value}.png"
        simulated.save(out_path)
        print(f"saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
