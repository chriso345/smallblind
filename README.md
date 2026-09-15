# smallblind

**smallblind** is a Python library for simulating color vision deficiency (CVD) on images, matplotlib/seaborn plots, and raw color palettes.

---

## Installation

```bash
pip install git+https://github.com/chriso345/smallblind.git 
```

---

## Usage

```python
import smallblind as sb

# Simulate a CVD type on an image (path, PIL.Image, or numpy array all work)
result = sb.simulate("chart.png", "deuteranopia")
result.save("chart_deuteranopia.png")

# Simulate on a matplotlib/seaborn palette
import seaborn as sns

palette = sns.color_palette("Set1")
safe_view = sb.simulate_palette(palette, sb.CVDType.PROTANOPIA)

# Check whether the palette holds up under common CVD types
report = sb.check_palette_safety(palette)
if not report.is_safe:
    for warning in report.warnings:
        print(warning)
```

### Matplotlib Figures

`sb.simulate_figure` rasterizes a `Figure` and returns a new `PIL.Image`, leaving the original figure untouched. This works for anything matplotlib can draw, including seaborn plots.

```python
import matplotlib.pyplot as plt
import smallblind as sb

fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 0], color="red")
ax.plot([0, 1, 2], [1, 0, 1], color="green")

sb.simulate_figure(fig, "deuteranopia").save("figure_deuteranopia.png")
```

### Comparison Grids

`sb.compare` builds a single labeled grid comparing the original against one or more CVD types.

```python
grid = sb.compare("chart.png", cvd="all", columns=3)
grid.save("chart_comparison.png")
```

### Palette Safety Checking

`sb.check_palette_safety` flags palette colors that become hard to tell apart under CVD, returning a `SafetyReport` with `is_safe`, `warnings`, and the underlying `confusions` list.

```python
report = sb.check_palette_safety(
    [(0.85, 0.10, 0.10), (0.10, 0.75, 0.10), (0.10, 0.40, 0.90)],
    cvd_types=["protanopia", "deuteranopia"],
)

report.is_safe      # False
report.warnings     # ["colors 0 and 1 are hard to distinguish under deuteranopia (distance=..)"]
```

### Jupyter Notebooks

Every image-producing function in smallblind (`sb.simulate`, `sb.simulate_image`, `sb.simulate_figure`, `sb.compare`) returns a plain `PIL.Image`, which Jupyter renders inline automatically.

```python
import smallblind as sb

sb.compare("chart.png", cvd="common")  # renders inline, no extra imports needed
```

---

## License

Licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
