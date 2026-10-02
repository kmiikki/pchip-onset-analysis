# Algorithm: Robust Binning + Shape-Preserving PCHIP Interpolation

This document describes the algorithm used in **`xy-pchip.py`** to transform noisy `x–y` profile data into a shape-preserving PCHIP-interpolated curve suitable for quantitative analysis and visualization. Downstream onset selection
is documented separately in [mgi-onsets.md](mgi-onsets.md).

---

## 1. Input

The input is a CSV file containing two numeric columns:

* **X**: independent variable (e.g., temperature, time, position)
* **Y**: dependent variable (e.g., brightness, signal intensity)

The columns are selected either by name (`--xcol`, `--ycol`) or by numeric index (`--xcoln`, `--ycoln`).

---

## 2. Pre-processing

### 2.1 Numeric conversion and filtering

All selected columns are coerced to numeric values.
Rows containing `NaN` in either column are dropped.

### 2.2 Sorting

The data are sorted by ascending x:

$$
(x, y) \leftarrow \mathrm{sort_{by\_x}}(x, y)
$$

### 2.3 Duplicate collapsing

If multiple rows share the same x, their y values are averaged:

$$
\tilde{y}(x) = \frac{1}{|{i : x_i = x}|} \sum_{i : x_i = x} y_i
$$

This ensures that x is strictly increasing — a requirement for monotone interpolation.

---

## 3. Robust Binning

### 3.1 Quantile bin edges

The x range is divided into quantile-based bins, typically 150–400, depending on data density.

$$
\mathrm{edges} = \mathrm{quantile}(x, k / B), ; k = 0, 1, \dots, B
$$

where **B** is the number of bins (at least 5).
Duplicate edges are removed to prevent degenerate bins. Assignment uses
`np.digitize(x, bins=edges[1:-1], right=True)`: an interior boundary belongs
to the bin on its left. With unique edges indexed 0 through K, bin 0 includes
`edges[0] <= x <= edges[1]`; later bins b include
`edges[b] < x <= edges[b+1]`. Both outer extrema are retained.

### 3.2 Aggregation per bin

For each non-empty bin **b**:

$$
x_b = \mathrm{median}(x_i \in \mathrm{bin}\ b), ;
y_b = \mathrm{mean}(y_i \in \mathrm{bin}\ b)
$$

These (x_b, y_b) points form the *binned series*.

---

## 4. Monotone PCHIP Interpolation

### 4.1 Concept

The **Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)** creates an interpolated curve that passes through all (x_b, y_b) points while preserving local monotonicity and avoiding interpolation overshoot. A
nonmonotonic input signal is not made globally monotonic.

Given ascending x_b values, PCHIP constructs cubic polynomials p_k(x) on each interval [x_k, x_{k+1}] such that:

* p_k(x_k) = y_k
* p_k(x_{k+1}) = y_{k+1}
* First derivatives are continuous across intervals

The interpolated dense grid is defined as:

$$
x^* = \mathrm{linspace}(\min(x_b), \max(x_b), N_{\mathrm{dense}})
$$

and the interpolated output:

$$
y^* = \mathrm{PCHIP}(x_b, y_b)(x^*)
$$

where **N_dense** is typically 2000.

---

## 5. Output

The script generates:

* **Plot**

  * Raw points, binned points, and PCHIP-interpolated curve
  * 300 dpi publication-quality PNG (`<outstem>.png`)
* **Optional CSV export** (`<outstem>.csv`)

  * If input X/Y names are distinct, they are preserved.
  * Otherwise, columns default to `X` and `Y`.

---

The default outstem is `<input-stem>-pchip`. An explicit `--outstem result`
writes `result.png` and, with `--save-csv`, `result.csv` without another suffix.
The dense sample count is clamped to at least 10 in `pchip_dense()`.

## 6. Pseudocode

```
Input: csv_path, (xcol|xcoln), (ycol|ycoln), bins, dense

1: Read header, resolve x/y column names
2: Read CSV, coerce to numeric, drop NaN
3: Sort by x ascending
4: Collapse duplicate x by averaging y
5: edges ← unique(quantile(x, linspace(0, 1, max(5, bins)+1)))
   labels ← digitize(x, edges[1:-1], right=True)
6: For each bin b:
       mask ← (labels == b); skip empty bins
       xb[b] ← median(x[mask])
       yb[b] ← mean(y[mask])
7: (xb, yb) ← sort_and_unique(xb, yb)
8: x_dense ← linspace(min(xb), max(xb), max(10, dense))
9: y_dense ← PCHIP(xb, yb)(x_dense)
10: Plot or export (x, y), (xb, yb), (x_dense, y_dense)
```

---

## 7. Parameter Guidance

| Parameter                  | Purpose                             | Typical range            |
| -------------------------- | ----------------------------------- | ------------------------ |
| `--bins`                   | Number of quantile bins             | 150–400                  |
| `--dense`                  | Number of PCHIP evaluation points   | 1000–5000                |
| `--reverse-x`              | Reverse x-axis for cooling datasets | –                        |
| `--no-raw` / `--no-binned` | Hide noisy or intermediate points   | –                        |
| `--pchip-lw`               | PCHIP line width                    | 1.0–2.0                  |
| `--pchip-color`            | Curve color (matplotlib syntax)     | `'k'`, `'#000000'`, etc. |

---

## 8. Notes

* Quantile binning improves robustness when x sampling density varies widely.
* Collapsing duplicate x values ensures numerical stability for PCHIP.
* The interpolation range is limited to the domain of x_b (no extrapolation).
* All results are deterministic given the same input and parameters.

---

## 9. References

* Fritsch, F. N. & Carlson, R. E. (1980). *Monotone Piecewise Cubic Interpolation*.
  *SIAM Journal on Numerical Analysis*, **17**(2), 238–246.
* SciPy Reference Guide: [scipy.interpolate.PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
