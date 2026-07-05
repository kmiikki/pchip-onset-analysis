# xy-pchip

**xy-pchip** is a command-line tool for transforming noisy `x–y` profile data using robust binning followed by shape-preserving **PCHIP interpolation**.

It is useful for analyzing datasets where signal intensity or brightness changes with a continuous variable such as temperature, time, or concentration. Typical applications include imaging-based crystallization studies, sensor calibration data, and general one-dimensional signal analysis.

---

## ⚙️ Features

* Automatic recognition of positional CSV filenames (`python xy-pchip.py data.csv`)
* Column selection by name (`--xcol`, `--ycol`) or by numeric index (`--xcoln`, `--ycoln`)
* Robust x-binning using quantiles before interpolation
* Shape-preserving **PCHIP** interpolation (no overshoot)
* Optional grid and x-axis reversal for cooling-type data
* Publication-quality 300 dpi plots (`.png`)
* Optional export of the PCHIP-interpolated curve as CSV

---

## 🧩 Basic Usage

```bash
python xy-pchip.py data.csv
```

or, explicitly:

```bash
python xy-pchip.py --csv data.csv --xcol "Tr (°C)" --ycol BW --grid --save-csv
```

### Output files

* `<stem>-pchip.png` — PCHIP plot
* `<stem>-pchip.csv` — optional CSV export

The output *stem* is automatically derived from the input filename (e.g. `data.csv → data-pchip.*`).

---

## 🔧 Command-line options

| Option                        | Description                                                             |
| ----------------------------- | ----------------------------------------------------------------------- |
| `--csv`                       | Input CSV file (or give filename positionally).                         |
| `--outstem`                   | Base name for output files. Default: `<input_stem>-pchip`.              |
| `--xcol`, `--ycol`            | Column names for X and Y.                                               |
| `--xcoln`, `--ycoln`          | Column indices (1-based). Mutually exclusive with name-based selection. |
| `--bins`                      | Number of quantile bins before interpolation (default: 200).            |
| `--dense`                     | Number of interpolation points for the PCHIP curve (default: 2000).     |
| `--grid`                      | Show grid lines on the plot.                                            |
| `--reverse-x`                 | Reverse x-axis (useful for cooling plots).                              |
| `--no-raw`, `--no-binned`     | Hide raw or binned points in the plot.                                  |
| `--pchip-lw`, `--pchip-color` | Control PCHIP line width and color.                                     |
| `--xlabel`, `--ylabel`        | Axis labels (default: column names).                                    |
| `--title`                     | Custom plot title.                                                      |
| `--save-csv`                  | Also export the PCHIP-interpolated curve to CSV.                                  |

---

## 🧠 Algorithm

The full mathematical and implementation details are described in [xy-pchip-algorithm.md](xy-pchip-algorithm.md).

---

## 🧪 Example

```bash
python xy-pchip.py ex4_data.csv --xcol "Tr (°C)" --ycol BW --grid \
  --reverse-x --pchip-color k --pchip-lw 1.0 --save-csv
```

Result:

* `ex4_data-pchip.png` — 300 dpi plot with black PCHIP line and grid
* `ex4_data-pchip.csv` — PCHIP-interpolated curve with same column names as input

---

## 🧰 Requirements

```bash
pip install numpy pandas matplotlib scipy
```

Python ≥ 3.9 is recommended.

---

## 📄 License

MIT License © 2025
Author: [Kim Miikki](https://github.com/kmiikki)
