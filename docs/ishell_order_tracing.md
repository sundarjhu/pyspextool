# iSHELL Order Tracing — Developer Notes

## Overview

This document describes the order-centre tracing implementation for iSHELL
2DXD reduction in `pyspextool`.  The tracing scaffold lives in:

    src/pyspextool/instruments/ishell/tracing.py

It is the first quantitative analysis stage after the visual diagnostics
in `pyspextool.ishell.diagnostics` and before full 2DXD wavelength
calibration.

---

## Purpose and Scope

This module implements **order-centre tracing from flat-field frames** —
one narrow but essential piece of the 2DXD reduction scaffold.

What it does:

* loads one or more raw iSHELL QTH flat-field FITS files,
* median-combines them to suppress read noise and hot pixels,
* finds order-centre peaks in cross-dispersion profiles at a set of
  evenly-spaced detector columns,
* traces each order centre across all sample columns, and
* fits a low-order polynomial to each traced trajectory.

What it does **not** do (yet):

* full order edge tracing,
* tilt or curvature measurement from arc frames,
* wavelength calibration,
* spectral rectification.

---

## Data Used

The implementation was developed and validated against the real
H1-mode calibration dataset at:

    data/testdata/ishell_h1_calibrations/raw/

| File                                          | Type    | Count |
|-----------------------------------------------|---------|-------|
| `icm.2026A060.260214.flat.000{50..54}.a.fits` | QTH flat | 5    |
| `icm.2026A060.260214.arc.00055.a.fits`        | ThAr arc | 1    |
| `icm.2026A060.260214.arc.00056.b.fits`        | ThAr arc | 1    |

These files are stored in Git LFS.  Run `git lfs pull` before attempting
to open them.  The test suite detects LFS pointer files and skips real-data
tests gracefully when the actual pixel data is not present.

---

## Algorithm

### 1. Frame loading and combination

```python
flat = load_and_combine_flats(flat_files)
```

All FITS PRIMARY extensions (extension 0) are read as `float32` arrays
and stacked along a new axis.  The pixel-wise median is taken; this
suppresses cosmic rays, hot pixels, and occasional bad reads more
robustly than the mean.

### 2. Seed peak detection

At the seed column (default: mid-point of the column range), a
cross-dispersion profile is computed by taking the pixel-wise **median**
over a band of columns `[col − col_half_width, col + col_half_width]`.
A light Gaussian smoothing (σ = 1 pixel) is applied to reduce fringing
features while preserving the broad order peaks.

Order-centre positions are found using `scipy.signal.find_peaks` with:

| Parameter     | Default | Rationale                                               |
|---------------|---------|---------------------------------------------------------|
| `distance`    | 25      | ≈ half the expected inter-order spacing of ≈ 46 px     |
| `prominence`  | 500     | robust against slowly-varying blaze; suppresses fringe  |

### 3. Column-by-column tracking

The algorithm walks from the seed column outward in both directions.  At
each step a **running position estimate** per order is maintained:

```
current_est[i]  ← last matched peak row for order i
```

For each new column:

1. Compute the smoothed cross-dispersion profile.
2. Find peaks with the same `distance` and `prominence` thresholds.
3. For each order, match the nearest detected peak within `max_shift`
   pixels (default 15 px).
4. If a match is found, update `current_est[i]`; otherwise leave it
   unchanged so the next column still has a valid estimate to try.

This design prevents a single missed detection at one column from causing
cascading failures at subsequent columns.

### 4. Polynomial fitting with sigma-clipping

For each order, the valid (non-NaN) centre-row measurements are fitted
with a degree-`poly_degree` polynomial using
`numpy.polynomial.polynomial.polyfit` (default degree 3).

Three rounds of σ-clipping (default threshold 3σ) reject outlier
measurements caused by fringing features that briefly mislead the peak
finder.

### 5. Half-width estimation

The half-maximum half-width of each order peak at the seed column is
estimated by walking outward from the peak until the profile drops below
the half-maximum level.  This is a proxy for the order width and is used
to build approximate edge polynomials.

---

## What Was Confirmed from the H1 Dataset

The implementation was validated against the five H1 flat frames in the
calibration dataset described above.  Key findings:

| Measurement                              | Result                       |
|------------------------------------------|------------------------------|
| Orders detected at col 1100             | 42–43 (45 expected)          |
| Median polynomial fit RMS               | ≈ 3.8 pixels                 |
| Maximum polynomial fit RMS              | < 7 pixels                   |
| Order centre range (rows) at col 1100   | ≈ 4 to 1985                  |
| Order spacing (row)                     | ≈ 46 pixels                  |
| Order half-width (row)                  | ≈ 16 pixels                  |

A notable finding: the traced order centres from the raw 2026 H1 frames
are offset by approximately **−70 rows** relative to the positions
predicted by the packaged `H1_flatinfo.fits` calibration resource.  This
confirms that per-dataset order tracing is necessary; the pre-computed
calibration files cannot be used directly for new observations.

The 3–4 missing orders (out of 45) are a consequence of:
* two orders near the very top of the detector receiving low flat-lamp
  signal, and
* occasional peak-detection ambiguities caused by interference fringing
  at the detector edges.

---

## Usage

```python
from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
import glob

flat_files = sorted(glob.glob(
    "data/testdata/ishell_h1_calibrations/raw/*.flat.*.fits"
))

# Trace with column range matching H1 calibration zone
trace = trace_orders_from_flat(flat_files, col_range=(650, 1550))

print(f"Found {trace.n_orders} orders")
print(f"Median polynomial RMS: {trace.fit_rms.mean():.2f} px")

# Convert to OrderGeometrySet for use in the 2DXD pipeline
from pyspextool.instruments.ishell.geometry import OrderGeometrySet
geom = trace.to_order_geometry_set("H1", col_range=(650, 1550))
print(f"OrderGeometrySet: {geom.n_orders} orders, mode={geom.mode!r}")
```

---

## Output Data Structure

`trace_orders_from_flat` returns a `FlatOrderTrace` dataclass:

| Attribute             | Type                              | Description                                   |
|-----------------------|-----------------------------------|-----------------------------------------------|
| `n_orders`            | `int`                             | Number of traced orders                       |
| `sample_cols`         | `ndarray (n_sample,)`             | Column positions used for profiling           |
| `center_rows`         | `ndarray (n_orders, n_sample)`    | Traced centre rows; NaN for missed detections |
| `center_poly_coeffs`  | `ndarray (n_orders, degree+1)`    | Polynomial coefficients (numpy convention)    |
| `fit_rms`             | `ndarray (n_orders,)`             | Per-order polynomial residual RMS (pixels)    |
| `half_width_rows`     | `ndarray (n_orders,)`             | Order half-width estimate (pixels)            |
| `poly_degree`         | `int`                             | Polynomial degree used                        |
| `seed_col`            | `int`                             | Column used for seed detection                |

The `FlatOrderTrace.to_order_geometry_set(mode)` method converts the
result to an `OrderGeometrySet` using the `OrderGeometry` class defined
in `instruments/ishell/geometry.py`.  The edge polynomials are
approximated by offsetting the centre polynomial by ±`half_width_rows`.

---

## Relationship to the 2DXD Arc-Line Tracing Stage

The tracing scaffold establishes the **order geometry** needed for the
first step of 2DXD rectification.  The pipeline proceeds as follows:

```
Flat tracing (this module)
    └── centre-line polynomial per order
    └── approximate edge polynomials
            │
            ▼
Arc-line tracing (NOT YET IMPLEMENTED)
    └── measure spectral-line tilt at each order
    └── fit tilt polynomial per order
            │
            ▼
Wavelength calibration (NOT YET IMPLEMENTED)
    └── identify arc lines against ThAr atlas
    └── fit wavelength polynomial per order
            │
            ▼
2DXD Rectification (NOT YET IMPLEMENTED)
    └── resample each order to (wavelength × spatial) grid
```

The `OrderGeometry` dataclass already has reserved fields for tilt, curvature,
wavelength, and spatial calibration polynomials (see `geometry.py`); they are
all `None` after the flat-tracing stage.

---

## Tests

Tests for the tracing module are in:

    tests/test_ishell_tracing.py

The test suite contains:

| Class                              | Coverage                                               |
|------------------------------------|--------------------------------------------------------|
| `TestFlatOrderTraceDataclass`      | dataclass construction and `to_order_geometry_set()`   |
| `TestLoadAndCombineFlats`          | FITS loading, median combination, cosmic-ray rejection |
| `TestTraceOrdersFromFlatErrors`    | empty input, blank image                               |
| `TestTraceOrdersFromFlatSynthetic` | known-geometry flat, polynomial accuracy checks        |
| `TestModuleImport`                 | importability regression                               |
| `TestTraceOrdersH1RealData`        | smoke-test on real H1 calibration data (slow)          |

Run all tests with:

```bash
pytest tests/test_ishell_tracing.py -v
```

The `TestTraceOrdersH1RealData` class is automatically skipped when the
real FITS data is not present (LFS not pulled).

---

## What Remains Unimplemented

The following steps are explicitly **out of scope** for this scaffold:

1. **Precise order edge tracing** — currently the edges are estimated as
   `centre ± half_width`, which is an approximation.  A proper edge tracer
   would fit the half-maximum crossing point as a function of column.

2. **Per-order column-range trimming** — the packaged `H1_flatinfo.fits`
   specifies different `[x_start, x_end]` ranges for different orders (e.g.
   order 355 starts at column 960 rather than 650).  The current implementation
   uses a uniform column range for all orders.

3. **Arc-line tilt and curvature measurement** — measuring the tilt of
   emission lines from ThAr arc frames along each order.

4. **Wavelength calibration** — fitting a polynomial wavelength solution
   using identified arc lines.

5. **Spatial calibration** — mapping row offsets to arcseconds on sky.

6. **2DXD rectification** — resampling the raw 2D spectrum onto a
   (wavelength × spatial) grid.

---

## References

* `docs/ishell_2dxd_notes.md` — background notes on iSHELL order geometry
  and the visual diagnostics that preceded this tracing work.
* `docs/ishell_geometry_design_note.md` — design notes on the
  `OrderGeometry` / `OrderGeometrySet` data model.
* `src/pyspextool/instruments/ishell/geometry.py` — order geometry classes.
* `src/pyspextool/instruments/ishell/calibrations.py` — packaged calibration
  resource readers.
* iSHELL Spextool Manual v10jan2020, Cushing et al.
