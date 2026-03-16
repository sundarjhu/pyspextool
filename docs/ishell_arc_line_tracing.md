# iSHELL Arc-Line Tracing — Developer Notes

## Overview

This document describes the **first-pass 2-D arc-line tracing scaffold** for
iSHELL 2DXD reduction in `pyspextool`.  The tracing scaffold lives in:

    src/pyspextool/instruments/ishell/arc_tracing.py

It is the second quantitative analysis stage of the 2DXD scaffold, following
the flat-field order tracing in `tracing.py` and preceding the (not yet
implemented) coefficient-surface fitting and wavelength calibration stages.

> **Status**: This is a development scaffold, not a finalised wavelength
> calibration.  The outputs are suitable for smoke-testing and data
> exploration, but should **not** be used for science-quality wavelength
> calibration without further validation, ThAr line identification, and
> coefficient-surface fitting.

---

## Purpose and Scope

This module implements **2-D arc-line tracing from ThAr arc frames** —
the stage after flat-order tracing and before wavelength calibration.

What it does:

* loads one or more raw iSHELL ThAr arc FITS files,
* median-combines them to suppress read noise and cosmic rays,
* uses the flat-based traced order geometry (:class:`FlatOrderTrace`) to
  define per-order search strips,
* identifies candidate arc emission lines by finding peaks in a
  row-collapsed spectral profile within each order strip,
* traces each identified line in the row direction across the order band
  by fitting sub-pixel Gaussian centroids, and
* fits a low-order polynomial ``col = f(row)`` to each traced line using
  sigma-clipping.

What it does **not** do (intentionally out of scope for this PR):

* ThAr line identification or atlas matching,
* coefficient-surface fitting across orders,
* rectification-index generation,
* wavelength or spatial calibration.

---

## Data Used

The scaffold was exercised against the real H1-mode calibration dataset at:

    data/testdata/ishell_h1_calibrations/raw/

| File                                          | Type     | Count |
|-----------------------------------------------|----------|-------|
| `icm.2026A060.260214.flat.000{50..54}.a.fits` | QTH flat | 5     |
| `icm.2026A060.260214.arc.00055.a.fits`        | ThAr arc | 1     |
| `icm.2026A060.260214.arc.00056.b.fits`        | ThAr arc | 1     |

These files are stored in Git LFS.  Run `git lfs pull` before attempting
to open them.  The test suite detects LFS pointer files and skips real-data
tests gracefully when the actual pixel data is not present.

---

## What Is Measured Directly from the Arc Frames

For each identified arc emission line within each echelle order, the
following is measured directly from the arc frame pixels:

1. **Seed column position** — the detector column at which the line's
   peak is identified in the row-collapsed spectral profile.

2. **Sub-pixel column centroid at each detector row** — the column
   position of the line (to sub-pixel precision) at each row within the
   order strip, obtained via a moment-based Gaussian centroid estimator.

3. **Per-line polynomial ``col = f(row)``** — a low-order polynomial fit
   (default degree 2) to the column centroids as a function of row,
   using iterative sigma-clipping.

These are purely geometric measurements from the detector frame.  No
wavelength information is assigned to the lines at this stage.

---

## Algorithm

### 1. Frame loading and combination

```python
arc = load_and_combine_arcs(arc_files)
```

All FITS PRIMARY extensions (extension 0) are read as `float32` arrays
and stacked along a new axis.  The pixel-wise median is taken; this
suppresses cosmic rays, hot pixels, and occasional bad reads.

### 2. Order strip definition

For each order in the `FlatOrderTrace`, the algorithm evaluates the
order centre-line polynomial at the seed column to obtain the centre row:

```python
center_row = polyval(seed_col, center_coeffs)
```

The order strip spans rows `[center_row - hw, center_row + hw]`, where
`hw` is the per-order half-width estimate from the flat trace.

### 3. Spectral profile and line-seed identification

Within each order strip, the algorithm takes the pixel-wise **median**
across the row dimension to produce a 1-D spectral profile as a function
of detector column:

```python
spectral_profile = np.median(arc[row_lo:row_hi+1, col_lo:col_hi+1], axis=0)
```

Peaks in this profile are identified using `scipy.signal.find_peaks` with
`prominence` and `distance` criteria.  Each detected peak column is a
candidate arc emission line.

### 4. Row-by-row tracing

For each candidate seed column, the algorithm:

1. Computes an initial sub-pixel centroid at the centre row of the order
   using a moment-based Gaussian estimator over a narrow column window.
2. Walks row-by-row from the centre outward (both upward and downward)
   within the order strip.  At each row, a Gaussian centroid is fitted
   within a ±`gaussian_half_width`-column window centred on the previous
   centroid estimate.
3. Accepts the centroid if it is finite and within `max_line_shift` pixels
   of the previous estimate; otherwise marks that row as missing (NaN).

This design tolerates single-row centroid failures without breaking the
trace at adjacent rows.

### 5. Polynomial fitting with sigma-clipping

For each traced line, the valid (non-NaN) column centroids are fitted with
a degree-`poly_degree` polynomial in row using `numpy.polynomial.polynomial.polyfit`.
Three rounds of σ-clipping (default threshold 3σ) reject outliers caused by
cosmic-ray residuals or detector artefacts.

The fitted polynomial ``col = f(row)`` is stored in the
`TracedArcLine.poly_coeffs` attribute following the
`numpy.polynomial.polynomial` convention: `coeffs[k]` is the coefficient
of `row**k`.

---

## Output Data Structures

### `TracedArcLine`

Represents one traced arc emission line within one echelle order.

| Attribute         | Type                             | Description                                    |
|-------------------|----------------------------------|------------------------------------------------|
| `order_index`     | `int`                            | Zero-based order index                         |
| `seed_col`        | `int`                            | Detector column of the initial seed            |
| `sample_rows`     | `ndarray (n_sample,)`            | Detector rows at which centroids were measured |
| `centroid_cols`   | `ndarray (n_sample,)`            | Sub-pixel column centroids; NaN for failures   |
| `poly_coeffs`     | `ndarray (poly_degree + 1,)`     | `col = f(row)` polynomial coefficients         |
| `fit_rms`         | `float`                          | RMS of polynomial residuals (pixels)           |
| `n_valid`         | `int`                            | Number of valid centroid measurements          |

### `ArcLineTrace`

Collects all traced lines across all orders.

| Attribute         | Type                             | Description                                       |
|-------------------|----------------------------------|---------------------------------------------------|
| `n_orders`        | `int`                            | Number of orders searched                         |
| `n_lines_total`   | `int`                            | Total number of traced lines                      |
| `lines`           | `list[TracedArcLine]`            | Traced lines, order-index ascending               |
| `poly_degree`     | `int`                            | Polynomial degree used                            |
| `seed_col`        | `int`                            | Seed column used                                  |
| `flat_trace`      | `FlatOrderTrace`                 | The flat-field trace used to constrain strips     |

Convenience methods:
- `lines_for_order(order_index)` — filter lines to one order,
- `n_lines_for_order(order_index)` — count lines in one order,
- `valid_lines(min_n_valid)` — filter to lines with sufficient valid points,
- `fit_rms_array()` — per-line RMS values as a NumPy array.

---

## Relationship to the 2DXD Pipeline Stages

The arc-line tracing scaffold sits between flat tracing and the later
coefficient-surface fitting stage in the 2DXD pipeline:

```
Flat tracing  (tracing.py)  ← DONE
    └── centre-line polynomial per order
    └── approximate edge polynomials
            │
            ▼
Arc-line tracing  (this module)  ← THIS PR
    └── spectral-profile peak finding in each order strip
    └── row-by-row Gaussian centroid tracing per line
    └── col = f(row) polynomial per line
            │
            ▼
Coefficient-surface fitting  ← NOT YET IMPLEMENTED
    └── fit 2-D surface  col(order, row)  across all orders and lines
    └── encodes spatial distortion map for rectification
            │
            ▼
Wavelength calibration  ← NOT YET IMPLEMENTED
    └── identify ThAr lines against atlas
    └── fit wavelength polynomial per order
            │
            ▼
Rectification-index generation  ← NOT YET IMPLEMENTED
    └── convert distortion map to resampling indices
            │
            ▼
2DXD spectral rectification  ← NOT YET IMPLEMENTED
    └── resample each order to (wavelength × spatial) grid
```

The per-line polynomials returned by `trace_arc_lines` are the natural
input for the coefficient-surface fitting step, and ultimately for
generating the rectification indices needed for full 2DXD rectification.

---

## Usage

```python
import glob
from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines

flat_files = sorted(glob.glob(
    "data/testdata/ishell_h1_calibrations/raw/*.flat.*.fits"
))
arc_files = sorted(glob.glob(
    "data/testdata/ishell_h1_calibrations/raw/*.arc.*.fits"
))

# Step 1: trace order geometry from flat frames
flat_trace = trace_orders_from_flat(flat_files, col_range=(650, 1550))
print(f"Flat tracing: {flat_trace.n_orders} orders")

# Step 2: trace arc lines constrained by the flat geometry
arc_trace = trace_arc_lines(
    arc_files,
    flat_trace,
    col_range=(650, 1550),
    min_line_prominence=200.0,
    poly_degree=2,
)
print(f"Arc tracing: {arc_trace.n_lines_total} lines across "
      f"{arc_trace.n_orders} orders")

# Inspect lines in the first order
order0_lines = arc_trace.lines_for_order(0)
for line in order0_lines:
    print(f"  seed_col={line.seed_col}, n_valid={line.n_valid}, "
          f"RMS={line.fit_rms:.3f} px")

# Filter to well-traced lines
good_lines = arc_trace.valid_lines(min_n_valid=5)
print(f"Lines with ≥5 valid centroids: {len(good_lines)}")
```

---

## Tests

Tests for the arc-line tracing module are in:

    tests/test_ishell_arc_tracing.py

The test suite contains:

| Class                                    | Coverage                                               |
|------------------------------------------|--------------------------------------------------------|
| `TestTracedArcLineDataclass`             | `TracedArcLine` dataclass construction                 |
| `TestArcLineTraceDataclass`              | `ArcLineTrace` and its helper methods                  |
| `TestLoadAndCombineArcs`                 | FITS loading, median combination, cosmic-ray rejection |
| `TestTraceArcLinesErrors`                | empty input                                            |
| `TestTraceArcLinesSynthetic`             | synthetic arc with known lines, centroid accuracy      |
| `TestModuleImport`                       | importability regression                               |
| `TestTraceArcLinesH1RealData`            | smoke-test on real H1 calibration data (slow)          |

Run all tests with:

```bash
pytest tests/test_ishell_arc_tracing.py -v
```

The `TestTraceArcLinesH1RealData` class is automatically skipped when the
real FITS data is not present (LFS not pulled).

The real-data smoke test uses **intentionally loose acceptance criteria**:
* At least one arc line traced in total (smoke check only).
* All valid lines have finite polynomial coefficients.
* Median polynomial RMS < 5 pixels for lines with ≥ 3 valid centroids.

---

## What Remains Unimplemented

The following steps are explicitly **out of scope** for this PR:

1. **ThAr line identification** — no matching against a ThAr wavelength
   atlas is performed.  The seed column positions are purely geometric
   measurements; no wavelengths are assigned to the traced lines.

2. **Coefficient-surface fitting** — the per-line `col = f(row)` polynomials
   are collected but no 2-D surface `col(order, row)` is fitted across the
   detector.

3. **Rectification-index generation** — the distortion map is not yet
   converted into pixel resampling indices.

4. **Wavelength solution** — no wavelength polynomial is fitted to any order.

5. **Spatial calibration** — no mapping of row offset to arcseconds is
   implemented.

6. **Full 2DXD rectification** — the raw 2-D spectrum is not resampled
   onto a (wavelength × spatial) grid.

7. **Order-resolved edge tracing** — the order edges used to define the
   search strip are the approximate `centre ± half_width` estimates from
   the flat tracing stage, not independently fitted edges.

8. **Tilt variation with column** — the per-order strip is evaluated at
   the seed column only.  A more precise implementation would vary the
   strip boundaries column-by-column following the curved order edges.

---

## References

* `docs/ishell_order_tracing.md` — flat-order tracing design notes.
* `docs/ishell_2dxd_notes.md` — background on iSHELL order geometry and
  the visual diagnostics that preceded the tracing work.
* `docs/ishell_geometry_design_note.md` — design notes on the
  `OrderGeometry` / `OrderGeometrySet` data model.
* `src/pyspextool/instruments/ishell/tracing.py` — flat-field order tracing.
* `src/pyspextool/instruments/ishell/geometry.py` — order geometry classes.
* iSHELL Spextool Manual v10jan2020, Cushing et al.
