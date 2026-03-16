# iSHELL 2-D Arc-Line Tracing — Developer Notes

## Overview

This document describes the **2-D arc-line tracing scaffold** for iSHELL
2DXD reduction in `pyspextool`.  The scaffold lives in:

    src/pyspextool/instruments/ishell/arc_tracing.py

It is the second quantitative analysis stage after flat-field order tracing
(`tracing.py`) and before full 2DXD wavelength calibration.

> **Status**: This is a development scaffold, not a finalised calibration.
> The outputs are suitable for smoke-testing the pipeline and exploring the
> data, but should **not** be used for science-quality wavelength calibration
> without further validation and refinement.

---

## Purpose and Scope

This module implements **2-D arc-line tracing from ThAr arc frames** —
one narrow but essential piece of the 2DXD reduction scaffold.

What it does:

* loads one or more raw iSHELL ThAr arc FITS files and median-combines them,
* uses an `OrderGeometrySet` (from flat-field tracing) to define per-order
  row and column boundaries,
* identifies arc emission-line candidates in each order's collapsed 1-D
  spectrum,
* traces each candidate line row-by-row using flux-weighted centroiding,
* fits a low-order polynomial `col = poly(row)` to each traced line, and
* returns an `ArcLineTraceResult` with all traced lines and their polynomial
  representations.

What it does **not** do (yet):

* wavelength identification (no line matching against a ThAr atlas),
* coefficient-surface fitting (the global `(order, col) → wavelength` surface),
* rectification-index generation,
* full 2DXD wavelength solution.

---

## Data Used

The scaffold is designed to run against the real H1-mode calibration dataset
at:

    data/testdata/ishell_h1_calibrations/raw/

| File                                          | Type     | Count |
|-----------------------------------------------|----------|-------|
| `icm.2026A060.260214.flat.000{50..54}.a.fits` | QTH flat | 5     |
| `icm.2026A060.260214.arc.00055.a.fits`        | ThAr arc | 1     |
| `icm.2026A060.260214.arc.00056.b.fits`        | ThAr arc | 1     |

These files are stored in Git LFS.  Run `git lfs pull` before attempting to
open them.  The test suite detects LFS pointer files and skips real-data tests
gracefully when the actual pixel data is not present.

---

## What Is Measured Directly from the Arc Frames

For each echelle order and each detected arc emission line, the following
are measured directly from the raw arc pixel data:

| Quantity               | Symbol       | Units     | How measured                                                  |
|------------------------|--------------|-----------|---------------------------------------------------------------|
| Seed column            | `seed_col`   | pixels    | Peak of 1-D collapsed spectrum (prominence-based peak finder) |
| Centroid column at row | `trace_cols` | pixels    | Flux-weighted centroid in a ±`col_half_width`-pixel window    |
| Trace rows             | `trace_rows` | pixels    | All rows in order where centroid succeeded                    |
| Polynomial tilt        | `poly_coeffs`| pixels/px | `polyfit(row, centroid_col)` with σ-clipping                  |
| Fit quality            | `fit_rms`    | pixels    | RMS of polynomial residuals                                   |
| Line strength proxy    | `peak_flux`  | counts    | Peak prominence in collapsed spectrum                         |

**What is NOT measured here**:

* The wavelength of each line (that requires ThAr atlas matching).
* The absolute position in a rectified frame (requires a wavelength solution).
* The spatial calibration (arcsec per row).

---

## Algorithm

### 1. Frame loading and combination

```python
arc = load_and_combine_arcs(arc_files)
```

All FITS PRIMARY extensions (extension 0) are read as `float32` arrays and
stacked.  The pixel-wise median is taken; this suppresses cosmic rays, hot
pixels, and residual sky emission more robustly than the mean.

### 2. Per-order strip extraction

For each order `geom` in the `OrderGeometrySet`:

```python
center_col = (geom.x_start + geom.x_end) // 2
row_lo = int(ceil(geom.eval_bottom_edge(center_col))) + 1
row_hi = int(floor(geom.eval_top_edge(center_col))) - 1
strip = arc[row_lo:row_hi+1, x_start:x_end+1]
```

The row bounds are evaluated at the mid-column of the order, which is an
approximation: the true bounds tilt slightly with column due to the order
edge polynomials.  This approximation is adequate for the purposes of finding
line candidates and centroiding within the order strip.

### 3. 1-D spectrum via row-median collapse

```python
spectrum = np.median(strip, axis=0)  # shape (n_cols,)
```

Collapsing along rows (the spatial direction) suppresses the slowly-varying
slit-illumination profile and leaves only the spectrally-resolved emission
features.  The median is preferred over the mean to suppress cosmic rays and
bad pixels that would otherwise produce spurious peaks.

### 4. Line candidate detection

```python
peaks_rel, props = scipy.signal.find_peaks(
    spectrum,
    distance=min_line_distance,   # minimum column separation
    prominence=min_line_prominence,  # minimum peak prominence
)
```

Prominence is robust to a slowly-varying blaze-function background.  The
`distance` parameter prevents splitting of closely-spaced lines.

### 5. Row-by-row centroiding

For each candidate line at absolute column `c_seed`, iterate over all rows
in the order strip:

```
for each row r in [row_lo, row_hi]:
    window = arc[r, c_seed - col_half_width : c_seed + col_half_width + 1]
    bg = min(window)
    net = max(window - bg, 0)
    if sum(net) > 0:
        centroid = sum(cols_in_window * net) / sum(net)
        if |centroid - c_seed| <= max_col_shift:
            accept (r, centroid) as a valid trace point
```

The minimum of the window is used as the local background estimate, which
works well for narrow emission lines on a dark background.  The `max_col_shift`
criterion rejects centroid measurements that have shifted too far from the
seed (e.g. due to bad pixels or blending with an adjacent line).

### 6. Polynomial fitting with sigma-clipping

For each traced line, fit `col = poly(row)` to the valid `(row, centroid_col)`
pairs:

```python
coeffs = np.polynomial.polynomial.polyfit(trace_rows, trace_cols, poly_degree)
```

Three rounds of σ-clipping (default threshold 3σ) reject outliers caused by
bad pixels or cosmic rays that briefly produce erroneous centroids.  The
resulting polynomial captures the spectral-line tilt: how much does the column
position of the line vary across the slit?

### 7. Quality filter

Traces with fewer than `max(poly_degree + 2, int(min_trace_fraction × n_rows))`
valid centroid points are discarded.  This ensures only well-sampled lines
contribute to downstream coefficient-surface fitting.

---

## Output Data Structures

### `TracedArcLine`

One instance per successfully traced arc emission line:

| Attribute       | Type                     | Description                                              |
|-----------------|--------------------------|----------------------------------------------------------|
| `order_index`   | `int`                    | Index into `geometry.geometries` (0-based placeholder)   |
| `seed_col`      | `int`                    | Column of the line seed in the collapsed spectrum        |
| `trace_rows`    | `ndarray (n_valid,)`     | Row positions where centroiding succeeded                |
| `trace_cols`    | `ndarray (n_valid,)`     | Centroid column at each `trace_rows` entry               |
| `poly_coeffs`   | `ndarray (degree+1,)`    | `col = poly(row)` coefficients (numpy convention)        |
| `fit_rms`       | `float`                  | RMS of polynomial residuals (pixels)                     |
| `peak_flux`     | `float`                  | Peak prominence in collapsed spectrum (line strength proxy) |

**Polynomial convention**: `poly_coeffs[k]` is the coefficient of `row**k`.
Evaluate with `np.polynomial.polynomial.polyval(row, poly_coeffs)`.

**Key methods**:

* `eval_col(rows)` — evaluate the polynomial at a set of row coordinates
* `tilt_slope()` — return `poly_coeffs[1]` (the linear tilt slope in px/row)
* `n_trace_points` — number of valid centroid measurements

### `ArcLineTraceResult`

Top-level container for all traced lines:

| Attribute        | Type                  | Description                                       |
|------------------|-----------------------|---------------------------------------------------|
| `mode`           | `str`                 | iSHELL mode (e.g. `"H1"`)                         |
| `arc_files`      | `list[str]`           | Arc file paths used                               |
| `poly_degree`    | `int`                 | Polynomial degree                                 |
| `geometry`       | `OrderGeometrySet`    | Flat-field order geometry used                    |
| `traced_lines`   | `list[TracedArcLine]` | All traced lines, sorted by order then seed_col   |

**Key properties/methods**:

* `n_lines` — total number of traced lines
* `n_orders` — number of orders in the geometry
* `lines_for_order(order_index)` — all lines in a specific order
* `n_lines_per_order()` — array of line counts per order

---

## Relationship to the 2DXD Pipeline

This module fills the **second stage** of the arc-line tracing scaffold:

```
Flat tracing (tracing.py)
    └── centre-line polynomial per order
    └── approximate edge polynomials
            │
            ▼
Arc-line tracing (this module — arc_tracing.py)
    └── 1-D line candidates per order (seed_col)
    └── col(row) polynomial per traced line
    └── ArcLineTraceResult
            │
            ▼
Coefficient-surface fitting (NOT YET IMPLEMENTED)
    └── global surface:  (order_index, seed_col) → wavelength
    └── populate OrderGeometry.wave_coeffs per order
            │
            ▼
Rectification-index generation (NOT YET IMPLEMENTED)
    └── dense grids of src_rows, src_cols per order
    └── RectificationMap per order
            │
            ▼
2DXD Rectification (NOT YET IMPLEMENTED)
    └── resample each order to (wavelength × spatial) grid
```

The `ArcLineTraceResult` is designed to feed directly into the
coefficient-surface fitting step:

* Each `TracedArcLine` contributes one `(order_index, seed_col, poly_coeffs)`
  data point to the surface fit.
* The surface fit will eventually produce `OrderGeometry.wave_coeffs` for each
  order — the wavelength as a function of detector column.
* The `poly_coeffs` from arc tracing may also inform the
  `OrderGeometry.tilt_coeffs` field.

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

# Stage 1: flat-field order tracing → order geometry
flat_trace = trace_orders_from_flat(flat_files, col_range=(650, 1550))
geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))
print(f"Flat tracing: {geom.n_orders} orders")

# Stage 2: arc-line tracing
arc_result = trace_arc_lines(
    arc_files,
    geom,
    poly_degree=2,
    min_line_prominence=200.0,
    col_half_width=5,
)
print(f"Arc tracing: {arc_result.n_lines} lines across {arc_result.n_orders} orders")

# Inspect results
counts = arc_result.n_lines_per_order()
for i, n in enumerate(counts):
    print(f"  Order {i}: {n} lines")

# Example: look at the first line's tilt slope
if arc_result.n_lines > 0:
    line = arc_result.traced_lines[0]
    print(f"First line: order {line.order_index}, col {line.seed_col}, "
          f"tilt {line.tilt_slope():.4f} px/row, RMS {line.fit_rms:.3f} px")
```

---

## Tests

Tests for the arc tracing module are in:

    tests/test_ishell_arc_tracing.py

| Class                              | Coverage                                                    |
|------------------------------------|-------------------------------------------------------------|
| `TestTracedArcLineDataclass`       | dataclass construction, eval_col, tilt_slope                |
| `TestArcLineTraceResult`           | container API, lines_for_order, n_lines_per_order           |
| `TestLoadAndCombineArcs`           | FITS loading, median combination, dtype                     |
| `TestTraceArcLinesErrors`          | empty input, empty geometry, blank image                    |
| `TestTraceArcLinesSynthetic`       | known-geometry arc, line recovery, tilt recovery            |
| `TestModuleImport`                 | importability regression                                    |
| `TestTraceArcLinesH1RealData`      | smoke-test on real H1 calibration data (slow)               |

Run all tests with:

```bash
pytest tests/test_ishell_arc_tracing.py -v
```

The `TestTraceArcLinesH1RealData` class is automatically skipped when the
real FITS data is not present (LFS not pulled).

---

## What Remains Unimplemented

The following steps are explicitly **out of scope** for this scaffold:

1. **Wavelength identification** — matching `seed_col` positions to ThAr
   atlas wavelengths (`data/H1_lines.dat`).  This requires a cross-dispersion
   model to convert column positions to wavelengths, which in turn requires
   a wavelength solution.

2. **Coefficient-surface fitting** — fitting a global 2-D surface
   `(order, col) → wavelength` over all traced lines.  This is the core 2DXD
   step and is not yet implemented.

3. **Rectification-index generation** — computing dense `(src_row, src_col)`
   grids for each output pixel in the rectified frame.

4. **Per-order column-range variation** — the packaged `H1_flatinfo.fits`
   calibration specifies different `[x_start, x_end]` ranges per order (e.g.
   order 355 starts at column 960 rather than 650).  The current scaffold uses
   a uniform column range.

5. **Tilt-corrected centroiding** — the current implementation uses a fixed
   seed column for centroiding at every row.  A production tracer would update
   the expected column using the running polynomial estimate, improving accuracy
   for lines with large tilt.

6. **Full 2DXD wavelength solution and image rectification** — the final
   pipeline step that produces rectified (wavelength × spatial) output.

---

## Known Limitations and Caveats

* **Order geometry is approximate** — the `OrderGeometrySet` produced by flat
  tracing uses `centre ± half_width` as edge estimates rather than
  independently fitted edge polynomials.  This may cause the arc-tracing row
  bounds to include a few pixels of inter-order background.

* **Coordinate offset vs. packaged calibration** — the traced order centres
  from the raw 2026 H1 frames differ by approximately −70 rows from those in
  the packaged `H1_flatinfo.fits`.  This offset is not yet understood (see
  `docs/ishell_order_tracing.md`).  The arc-line positions traced here are
  internally consistent with the flat-tracing geometry, but their relationship
  to the packaged wavelength solution is unknown until this offset is resolved.

* **No bad-pixel handling** — the current implementation relies on the arc
  median combination to suppress cosmic rays and bad pixels.  A production
  pipeline would apply a bad-pixel mask.

---

## References

* `docs/ishell_order_tracing.md` — flat-field order tracing documentation.
* `docs/ishell_2dxd_notes.md` — background notes on iSHELL order geometry.
* `docs/ishell_geometry_design_note.md` — design notes on the
  `OrderGeometry` / `OrderGeometrySet` data model.
* `src/pyspextool/instruments/ishell/arc_tracing.py` — the implementation.
* `src/pyspextool/instruments/ishell/tracing.py` — flat-field order tracing.
* `src/pyspextool/instruments/ishell/geometry.py` — order geometry classes.
* iSHELL Spextool Manual v10jan2020, Cushing et al.
