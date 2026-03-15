# iSHELL 2DXD Tracing Scaffold: Order-Center Tracing from Flat-Field Data

## Purpose

This document describes the first real 2DXD tracing scaffold for iSHELL,
implemented in `src/pyspextool/instruments/ishell/order_trace.py`.  The
scaffold traces echelle order centers as a function of detector column using
the packaged H1 flat-field calibration as real test data.

---

## Background

The IDL Spextool 2DXD wavelength-calibration algorithm traces individual ThAr
arc lines across the detector using a two-step process:

1. **Flat-based order tracing**: Using a normalized QTH flat, locate the
   center of each echelle order at each detector column.  These order-center
   traces define the "zero-tilt" (dispersion-axis) direction.

2. **Arc-line tracing**: For each ThAr line in the arc frame, trace the line
   center across the detector, using the flat-based traces as initial guesses
   and search windows.  Tilt (spectral line curvature relative to the spatial
   axis) is measured at this step.

3. **2DXD polynomial fitting**: Fit a 2D polynomial in `(column, order)` space
   to the arc-line positions, producing a global wavelength solution.

This scaffold implements **step 1 only**.

---

## Data Source: The Packaged H1 Calibration

The packaged file `src/pyspextool/instruments/ishell/data/H1_flatinfo.fits`
(~8.4 MB) contains:

- A `(2048, 2048)` integer array — the **order mask** — in which each pixel
  is set to its echelle order number (0 = inter-order).  This mask was
  derived from real QTH flat frames during the original IDL Spextool
  calibration run.

- FITS header keywords:
  - `ORDERS`: comma-separated list of 45 order numbers (311–355)
  - `OR{n}_XR`: column range `[x_start, x_end]` for each order
  - `OR{n}_B1`…`OR{n}_B5`: 5-coefficient bottom-edge polynomial for each
    order (`numpy.polynomial.polynomial` convention; constant term first)
  - `OR{n}_T1`…`OR{n}_T5`: 5-coefficient top-edge polynomial
  - `EDGEDEG`: polynomial degree (4)
  - `ROTATION`: IDL rotate code (5) — required to bring the image to the
    standard pySpextool orientation

### Coordinate system

The order mask is stored in the **raw IDL orientation**.  After applying
`idl_rotate(image, rotation=5)`, the image is in the standard pySpextool
orientation:

- **Columns (axis 1)** = dispersion axis (left = short wavelength for H band)
- **Rows (axis 0)** = spatial/cross-dispersion axis

All coordinates in `OR{n}_XR`, `OR{n}_B{k}`, and `OR{n}_T{k}` are in this
**rotated** coordinate system.

---

## Algorithm

`trace_order_centers_from_flatinfo(flatinfo_or_mode, *, poly_degree=3, step_size=10)`

1. **Load calibration**: Accept either a `FlatInfo` object or a mode name
   string (e.g. `"H1"`); load via `read_flatinfo` if a string is given.

2. **Rotate image**: Apply `idl_rotate(image, rotation)` to bring the
   order mask to the standard orientation.

3. **Build reference geometry**: Construct an `OrderGeometrySet` from the
   edge polynomial coefficients (`OR{n}_B{k}` / `OR{n}_T{k}`).  This serves
   as the analytical reference against which pixel-level traces are compared.

4. **Column-by-column centroiding**: For each order `n` and each column `c`
   within `[x_start, x_end]` (sampled every `step_size` columns):
   - Find all rows `r` where `image[r, c] == n`.
   - Require at least 2 pixels; otherwise mark the column as invalid.
   - Compute the unweighted centroid: `row_centroid = mean(r)`.

5. **Polynomial fit**: Fit a degree-`poly_degree` polynomial to the valid
   `(column, row_centroid)` pairs using `numpy.polynomial.polynomial.polyfit`.
   Compute the RMS residual.
   - Fallback: if fewer than `poly_degree + 1` valid points are available,
     use the analytical centerline from the edge polynomials and emit a
     `UserWarning`.

6. **Return** an `OrderTraceResult` containing:
   - `trace_columns`, `trace_rows`: raw centroid measurements
   - `center_coeffs`: fitted polynomial coefficients per order
   - `fit_rms`: RMS residual per order (pixels)
   - `n_good`: number of valid centroid measurements per order
   - `geometry`: the reference `OrderGeometrySet`

---

## Confirmed from the H1 Data

Running the tracer on the packaged H1 calibration confirms:

| Property | Value |
|----------|-------|
| Orders successfully traced | 45/45 (order numbers 311–355) |
| Typical RMS residual | ≤ 0.25 pixel |
| Maximum RMS residual | < 0.5 pixel (order 355, near detector edge) |
| Agreement with edge-polynomial centerlines | < 1.5 pixels across all columns |
| Sub-pixel agreement (typical) | < 0.5 pixel |

The excellent agreement between the pixel-level centroid trace and the
analytical edge-polynomial centerlines validates both the order mask and the
stored edge polynomials.

Order centers at `col = 1000` (representative):

| Order | Traced center (row) | Analytical center (row) | Difference |
|-------|---------------------|------------------------|------------|
| 311 | ~86.9 | ~86.9 | ~0.0 |
| 325 | ~766.3 | ~766.4 | ~0.1 |
| 340 | ~1413.7 | ~1413.7 | ~0.0 |
| 355 | ~1987.9 | ~1987.7 | ~0.2 |

Orders are monotonically spaced in the spatial direction (ascending row index
with increasing order number for H1).

---

## Relationship to the 2DXD Arc-Line Tracing Stage

The `OrderTraceResult` produced by this scaffold feeds directly into the 2DXD
arc-line tracing stage (not yet implemented):

1. The `center_coeffs` polynomials provide **initial row positions** for
   searching arc lines at each detector column.

2. The `geometry` (`OrderGeometrySet`) can be augmented with tilt, curvature,
   and wavelength calibration coefficients as the 2DXD pipeline progresses
   (see `geometry.py` and the field documentation in `OrderGeometry`).

3. The `fit_rms` values flag orders where the flat-based trace is noisy;
   these orders may need wider search windows in the arc-line tracing step.

---

## What Remains Unimplemented

The following steps are explicitly **out of scope** for this scaffold:

| Step | Status |
|------|--------|
| Arc-line tracing from ThAr arc frames | Not implemented |
| Flux-weighted centroiding from normalized QTH flat | Not implemented (order mask used instead) |
| Spectral tilt / curvature estimation | Not implemented |
| 2DXD global wavelength polynomial fitting | Not implemented |
| Rectification-index computation | Not implemented |
| Raw flat frame ingestion (QTH frames in `raw/`) | Not implemented |

---

## Test Coverage

Tests are in `tests/test_ishell_order_trace.py` (72 tests).  They use the
packaged `H1_flatinfo.fits` as real test data:

- `TestOrderTraceResultStructure`: result fields, types, mode
- `TestPerOrderArrays`: array shapes, dtypes, detector-range checks
- `TestFitQuality`: polynomial coefficient length, RMS thresholds, internal
  residual consistency
- `TestAgreementWithEdgePolynomials`: traced vs. analytical centerline deviation
- `TestEvalCenter`: `eval_center()` scalar and array inputs
- `TestParameterValidation`: bad `poly_degree`, bad `step_size`
- `TestModeStringInput`: mode-name string accepted (slow marker)
- `TestParameterVariation`: different `step_size` and `poly_degree` values
- `TestAllH1Orders` (slow): all 45 orders valid, monotonic spacing

Run all non-slow tests:

```bash
pytest tests/test_ishell_order_trace.py -m "not slow" -v
```

Run including slow tests:

```bash
pytest tests/test_ishell_order_trace.py -v
```

---

## API Reference

### `trace_order_centers_from_flatinfo(flatinfo_or_mode, *, poly_degree=3, step_size=10)`

Main entry point.  See module docstring for full documentation.

```python
from pyspextool.instruments.ishell.order_trace import (
    trace_order_centers_from_flatinfo,
)

result = trace_order_centers_from_flatinfo("H1")
print(result.n_orders)          # 45
print(result.fit_rms[325])      # ~0.14 pixels
center = result.eval_center(325, 1000.0)   # row position at col 1000
```

### `OrderTraceResult`

Frozen-like result dataclass.  Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `str` | iSHELL mode name |
| `orders` | `list[int]` | Order numbers traced |
| `trace_columns[order]` | `ndarray` | Sampled column positions |
| `trace_rows[order]` | `ndarray` | Measured centroid row at each column |
| `center_coeffs[order]` | `ndarray` | Polynomial coefficients (numpy convention) |
| `fit_rms[order]` | `float` | RMS residual of polynomial fit (pixels) |
| `n_good[order]` | `int` | Number of valid centroid measurements |
| `poly_degree` | `int` | Polynomial degree used |
| `geometry` | `OrderGeometrySet` | Reference geometry from edge polynomials |

Key method: `eval_center(order, cols)` — evaluate fitted centerline at columns.
