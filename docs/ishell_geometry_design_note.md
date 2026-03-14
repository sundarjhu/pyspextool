# iSHELL Geometry Data-Model Design Note

**Status:** Minimum geometry support implemented (pre-wavecal)  
**Scope:** NASA IRTF iSHELL spectrograph, J / H / K modes  
**Author:** pySpextool iSHELL integration

---

## 1. Background

This note describes the minimum geometry / data-model changes made in
preparation for full iSHELL wavelength calibration.  It supersedes §5.3 of
`ishell_design_memo.md` on the topic of order-geometry representation.

The problem statement asked for:

1. An inspection of existing wavecal, trace, and rectification code.
2. The minimum internal data structures for iSHELL order geometry.
3. A refactor just large enough to let later iSHELL wavecal plug in cleanly.
4. Tests and this design note.

---

## 2. What the Existing Architecture Gets Right (Reusable)

### 2.1 Generic extraction pipeline

`extract/extraction.py → extract_1dxd()` and
`extract/trace.py → trace_spectrum_1dxd()` operate on three instrument-neutral
arrays:

| Array       | Shape              | Meaning                                 |
|-------------|--------------------|-----------------------------------------|
| `wavecal`   | (nrows, ncols)     | Wavelength in microns at each pixel     |
| `spatcal`   | (nrows, ncols)     | Spatial position in arcsec at each pixel|
| `ordermask` | (nrows, ncols) int | Order number (0 = inter-order region)   |

These arrays abstract away the instrument-specific pixel-to-sky mapping so
the extraction math is reusable unchanged.

### 2.2 Flat-field order tracing

`extract/flat.py → read_flatcal_file()` already reads per-order polynomial
coefficients from the `*_flatinfo.fits` FITS header:

- `OR{n}_B{k}` / `OR{n}_T{k}` – bottom / top edge polynomial coefficients
  for order *n*, term *k* (follows `numpy.polynomial.polynomial` convention
  where `coeffs[k]` multiplies `x**k`).
- `OR{n}_XR` – comma-separated column range `[x_start, x_end]`.

These are returned in `edgecoeffs` (shape `(norders, 2, deg+1)`) and
`xranges` (shape `(norders, 2)`).  The convention is fully generic; it is
already used by all iSHELL `*_flatinfo.fits` files.

### 2.3 iSHELL calibration dataclasses

`instruments/ishell/calibrations.py` already defines typed calibration objects
(`FlatInfo`, `WaveCalInfo`, `LineList`, etc.) keeping iSHELL-specific logic
inside the instrument package and away from generic modules.

---

## 3. What Had to Change for iSHELL Geometry

### 3.1 The fundamental difference

SpeX / uSpeX orders are roughly column-aligned, so the 2-D `wavecal` /
`spatcal` arrays populated from the wavecal FITS file *directly* describe the
geometry needed by `extract_1dxd()`.  No intermediate geometry object is
needed.

iSHELL echelle orders are **tilted** and **curved** with respect to detector
columns.  Spectral lines run at an angle across the order cross-section.
Before `extract_1dxd()` can be applied, the image must be **rectified**:
resampled onto a rectilinear (wavelength × spatial) grid.  Rectification
requires:

1. Per-order tilt polynomial: `tilt_slope = p(column)`.
2. Per-order curvature polynomial (optional second-order term).
3. A wavelength solution along the order centerline: `λ = p(column)`.
4. A spatial calibration: `arcsec = p(row − centerline_row(column))`.

None of these are needed for SpeX/uSpeX and none existed in the codebase.

### 3.2 New module: `instruments/ishell/geometry.py`

Three new dataclasses and one factory function were added:

#### `OrderGeometry`

Describes the geometry of one echelle order.  Required fields are populated
from flat-field data; optional fields are filled during the wavecal step.

| Field                 | Type                    | Availability        |
|-----------------------|-------------------------|---------------------|
| `order`               | `int`                   | After flat tracing  |
| `x_start`, `x_end`   | `int`                   | After flat tracing  |
| `bottom_edge_coeffs`  | `ndarray (n_terms,)`    | After flat tracing  |
| `top_edge_coeffs`     | `ndarray (n_terms,)`    | After flat tracing  |
| `tilt_coeffs`         | `ndarray` or `None`     | After wavecal step  |
| `curvature_coeffs`    | `ndarray` or `None`     | After wavecal step  |
| `wave_coeffs`         | `ndarray` or `None`     | After wavecal step  |
| `spatcal_coeffs`      | `ndarray` or `None`     | After wavecal step  |

The `centerline_coeffs` property is computed as the mean of bottom and top
edge polynomials (term-by-term), exactly as done implicitly in the current
SpeX pipeline.

Evaluation helpers (`eval_bottom_edge`, `eval_top_edge`, `eval_centerline`,
`eval_tilt`) accept array or scalar column arguments and use
`numpy.polynomial.polynomial.polyval`, matching the convention already used in
`read_flatcal_file()`.

#### `RectificationMap`

Holds the *output* of the rectification-coordinate computation: fractional
source detector coordinates `(src_rows, src_cols)` for each output-grid pixel,
plus the output wavelength and spatial axes.  These arrays are the direct
input to `scipy.ndimage.map_coordinates` (or equivalent).

This object intentionally holds only pre-computed index arrays (not
polynomials), making the resampling step trivially vectorisable.

#### `OrderGeometrySet`

A simple collection of `OrderGeometry` objects for all orders in one mode.
Provides `get_order(n)` look-up, aggregate predicates (`has_tilt()`,
`has_wavelength_solution()`), and is the type passed to `_rectify_orders()`.

#### `build_order_geometry_set()`

Factory that converts the `orders`, `xranges`, and `edgecoeffs` arrays
returned by `read_flatcal_file()` into an `OrderGeometrySet`.  This is the
intended bridge between flat tracing and wavecal.

### 3.3 Updated `_rectify_orders()` signature

`ishell.py → _rectify_orders()` previously accepted a plain `dict` (labelled
`wavecalinfo`).  The signature is now:

```python
def _rectify_orders(
    image: npt.ArrayLike,
    geometry: OrderGeometrySet,
) -> npt.ArrayLike:
```

This makes the dependency on the new geometry type explicit and prevents the
anti-pattern of passing a raw dict with undocumented keys.  The function body
still raises `NotImplementedError` (Phase 2 task).

### 3.4 Re-export from `calibrations.py`

`OrderGeometry`, `OrderGeometrySet`, `RectificationMap`, and
`build_order_geometry_set` are re-exported from `calibrations.py` so callers
can import everything from one place:

```python
from pyspextool.instruments.ishell.calibrations import (
    OrderGeometry, OrderGeometrySet, build_order_geometry_set, ...
)
```

---

## 4. What Remains for the Next Wavecal Step

The following numerical work is **explicitly deferred** (Phase 2):

1. **Tilt polynomial fitting** – given identified ThAr arc lines, fit
   `tilt_slope = p(column)` per order by measuring the row offset of each
   line at different spatial positions.

2. **Wavelength polynomial fitting** – apply the 2DXD cross-dispersed
   polynomial model (already sketched in `WaveCalInfo`) to produce
   `wave_coeffs` for each order.

3. **`RectificationMap` construction** – once `tilt_coeffs` and `wave_coeffs`
   are available on every `OrderGeometry`, build `src_rows` / `src_cols`
   coordinate arrays by tracing each output-grid point back through the tilt
   and wavelength polynomials to raw detector coordinates.

4. **`_rectify_orders()` implementation** – call
   `scipy.ndimage.map_coordinates(image, [src_rows, src_cols])` per order
   using the corresponding `RectificationMap`.

5. **Wavecal FITS writer** – write `wavecal` and `spatcal` 2-D arrays in the
   standard pySpextool format so `extract_1dxd()` can proceed unchanged.

---

## 5. Impact on SpeX / uSpeX

None.  All new code lives inside `instruments/ishell/`.  No generic module
in `extract/`, `combine/`, `merge/`, `telluric/`, or `io/` was modified.
The 474+ existing SpeX/uSpeX tests continue to pass.

---

## 6. Polynomial Convention Reference

All polynomial evaluation throughout the iSHELL geometry layer uses
`numpy.polynomial.polynomial.polyval(x, coeffs)` where `coeffs[k]` is the
coefficient of `x**k` (constant term at index 0).  This matches the convention
already used in `read_flatcal_file()` for edge coefficients.

Do **not** use `numpy.polyval(coeffs, x)` (legacy interface, highest degree
first) in iSHELL geometry code.
