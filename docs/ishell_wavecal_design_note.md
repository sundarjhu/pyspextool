# iSHELL Geometry Population and Rectification – Developer Design Note

## Overview

This document describes the iSHELL J/H/K geometry-population and image-
resampling implementation in pySpextool.

> **Important:** This is a structural bootstrap, not a full
> wavecal/rectification pipeline.  The current implementation reads
> pre-packaged calibration files and populates geometry containers with
> structural best-guesses.  The tilt model is a **placeholder zero**,
> so no spectral-line tilt is removed.  The spatial calibration is a
> simple linear plate-scale approximation.  See §5 for a detailed list
> of what remains to be implemented.

This document covers:

1. What the implementation currently does (as opposed to what the full
   legacy IDL Spextool procedure does).
2. How the geometry objects (`OrderGeometry`, `OrderGeometrySet`) are populated.
3. The resampling algorithm (`_rectify_orders`).
4. What remains incomplete relative to legacy IDL Spextool.

**Scope:** J, H, and K bands only.  L/Lp/M modes are out of scope.  This
document is provisional; scientific accuracy requires validation against
real ThAr arc data.

---

## 1. Architecture

### Key modules

| Module | Purpose |
|--------|---------|
| `instruments/ishell/geometry.py` | Data model: `OrderGeometry`, `OrderGeometrySet`, `RectificationMap` |
| `instruments/ishell/calibrations.py` | Typed readers for packaged calibration files |
| `instruments/ishell/wavecal.py` | **NEW** geometry-population and rectification-map builder (structural bootstrap; not a full wavecal pipeline) |
| `instruments/ishell/ishell.py` | **UPDATED** `_rectify_orders()` implementation |

### Data flow

```
read_flatinfo(mode)     → FlatInfo        (edge polynomials, plate scale)
read_wavecalinfo(mode)  → WaveCalInfo     (stored reference arrays, xranges)
          ↓
build_geometry_from_wavecalinfo(wci, fi)
          ↓
OrderGeometrySet  (wave_coeffs, tilt_coeffs, spatcal_coeffs populated)
          ↓
build_rectification_maps(geom_set, plate_scale_arcsec)
          ↓
list[RectificationMap]  (src_rows, src_cols for scipy.ndimage.map_coordinates)
          ↓
_rectify_orders(image, geom_set, plate_scale_arcsec)
          ↓
resampled image (same shape as input; NaN outside all orders; zero-tilt placeholder)
```

---

## 2. Geometry Population (polynomial fit to stored reference arrays)

### What the packaged `*_wavecalinfo.fits` files contain

Every `*_wavecalinfo.fits` file is a data cube of shape
`(n_orders, 4, n_pixels)`.  Based on structural inspection:

| Plane | Status | Observation |
|-------|--------|-------------|
| 0 | **Inferred as wavelengths (µm)** – values match J/H/K band ranges and the FITS header contains 2DXD polynomial coefficients; exact provenance not formally documented | Used by this implementation |
| 1 | **Not confirmed** – non-NaN values of varying sign/scale; possibly a reference arc spectrum but NOT formally documented | Not used |
| 2 | **Not confirmed** – possibly an uncertainty array; NOT formally documented | Not used |
| 3 | **Not confirmed** – possibly quality flags; NOT formally documented | Not used |

> **Important:** Only plane 0 is read by the current implementation.  The
> interpretation of planes 1–3 is entirely speculative and they are not used.

The FITS header also contains:

- `P2W_C00`…`P2W_C11` – 12 polynomial coefficients, consistent with the
  2DXD cross-dispersed polynomial formula described in the iSHELL Spextool
  manual.  These are **not re-evaluated** by the current implementation; they
  are noted here as structural evidence supporting the plane-0 inference.
- `OR{n}_XR` – per-order column ranges, structurally validated against the
  NaN boundary in plane 0.

### What the current implementation does

`build_geometry_from_wavecalinfo()` does **not** implement the 1DXD/2DXD
wavecal procedure from IDL Spextool.  Instead it:

1. Reads the stored values from plane 0 (inferred wavelengths in µm).
2. Maps array indices to detector columns via `OR{n}_XR` ranges.
3. Fits a 1D polynomial (degree = `DISPDEG`) to `(column, plane-0 value)` pairs.
4. Stores the polynomial coefficients in `OrderGeometry.wave_coeffs`.

```python
# Array index j → detector column x_start + j  (structural assumption)
cols = np.arange(n_valid) + x_start
vals = wavecalinfo.data[order_idx, 0, :][~np.isnan(...)]
coeffs = np.polynomial.polynomial.polyfit(cols, vals, degree)
```

This is a compact polynomial representation of the stored reference values,
not a measurement from arc frames.

### Confirmed vs inferred vs not-yet-validated

| Fact | Status |
|------|--------|
| Plane 0 values fall in expected J/H/K band ranges (µm) | **Structurally validated** by range inspection |
| Plane 0 provenance is the 2DXD solution from ThAr arcs | **Inferred** – consistent with header coefficients but not directly confirmed |
| `OR{n}_XR` gives valid pixel range for plane 0 | **Structurally validated** – NaN boundary matches XR range |
| `P2W_C00`…`P2W_C11` are 2DXD polynomial coefficients | **Inferred** from header keyword names and formula structure |
| Planes 1–3 interpretation (arc spectrum / uncertainty / flags) | **Not confirmed** – speculative |
| `LINEDEG = 1` specifies tilt polynomial degree | **Not yet validated** – tilt fitting is not yet implemented |

---

## 3. Order Geometry Population

After calling `build_geometry_from_wavecalinfo()`, each `OrderGeometry`
in the returned `OrderGeometrySet` has:

| Field | Source | Notes |
|-------|--------|-------|
| `order` | `wavecalinfo.orders[i]` | Echelle order number |
| `x_start`, `x_end` | `flatinfo.xranges[i]` | Column range from flat |
| `bottom_edge_coeffs` | `flatinfo.edge_coeffs[i, 0, :]` | From `OR{n}_B1…B5` |
| `top_edge_coeffs` | `flatinfo.edge_coeffs[i, 1, :]` | From `OR{n}_T1…T5` |
| `wave_coeffs` | Polynomial fit to `wavecalinfo.data[i, 0, :]` | Inferred wavelengths; degree = `DISPDEG`; not a live wavecal measurement |
| `tilt_coeffs` | `[0.0]` | **Placeholder** – no tilt measured; identity mapping |
| `spatcal_coeffs` | `[0.0, plate_scale_arcsec]` | **Provisional** linear model; no curvature |
| `curvature_coeffs` | `None` | Not implemented |

### Edge polynomials

The flat FITS header stores per-order edge polynomial coefficients as
`OR{n}_B1`…`OR{n}_B5` (bottom edge) and `OR{n}_T1`…`OR{n}_T5` (top edge).
These are degree-4 polynomials in the detector column.  They follow the
`numpy.polynomial.polynomial` convention: `OR{n}_B1` is the constant term.

The `EDGEDEG` header keyword gives the polynomial degree (4 for all current
iSHELL modes).

---

## 4. Resampling Algorithm (`_rectify_orders`)

> **Note on terminology:** the function is named `_rectify_orders` to match
> legacy IDL Spextool naming, but with the current **placeholder zero-tilt**
> model it does not perform true spectral rectification.  It applies NaN
> masking outside order footprints and bilinearly resamples the input, but
> does not correct spectral-line tilt.

### `build_rectification_maps()`

For each order, `build_rectification_maps()` computes source detector
coordinates `(src_rows, src_cols)` for a regular output grid:

**Output grid:**

- **Spectral axis:** one pixel per detector column from `x_start` to `x_end`
  (integer columns), labelled by the polynomial fitted to plane-0 values.
- **Spatial axis:** uniformly spaced from `−half_slit` to `+half_slit`
  arcseconds (provisional linear model), where `half_slit = 0.5 × slit_height_arcsec`.

**Source coordinate computation** (for output pixel `(spatial_i, spectral_j)`):

```
col_out = x_start + j
row_ctr = eval_centerline(col_out)
row_offset_pix = spatial_arcsec[i] / plate_scale_arcsec
tilt = eval_tilt(col_out)           # = 0.0 – placeholder; no tilt measured
src_col = col_out + tilt * row_offset_pix   # = col_out (identity when tilt=0)
src_row = row_ctr + row_offset_pix
```

With zero tilt, `src_col = col_out` – no horizontal correction is applied.

### `_rectify_orders()`

`_rectify_orders(image, geometry, plate_scale_arcsec)` performs the image
resampling:

1. Calls `build_rectification_maps()` to get `src_rows`, `src_cols` for
   every order.
2. For each order, computes a masked grid covering the order footprint.
3. Applies `scipy.ndimage.map_coordinates` with bilinear interpolation
   (`order=1`) to resample.
4. Pixels outside all order footprints are set to NaN.

**Current behaviour with placeholder zero tilt:**

```
output[r, c] = bilinear_interp(input, r, c)   ∀ (r, c) within an order
output[r, c] = NaN                            ∀ (r, c) outside all orders
```

The output is structurally well-formed (correct NaN masking, edge geometry
applied) but does **not** correct spectral-line tilt.

---

## 5. What Remains Incomplete

### Blocking for science quality

| Gap | Description |
|-----|-------------|
| **Tilt measurement** | Arc-line centroids must be fitted at multiple row positions to measure `d(col)/d(row)` at each column. Until this is done, `_rectify_orders()` does not correct spectral-line tilt. The `LINEDEG` header keyword (= 1) may indicate the expected polynomial degree for the tilt fit in IDL Spextool (not yet validated). |
| **Plane 0 formal validation** | The assumption that plane 0 of `*_wavecalinfo.fits` contains wavelengths in µm is inferred from value ranges, not formally documented. Planes 1–3 are unused and their meaning is unknown. |
| **2DXD re-fitting from arc frames** | Full interactive line identification, robust outlier rejection, and iterative refinement are not implemented; stored reference arrays are used instead. |
| **Slit curvature** | Higher-order spatial distortion (`curvature_coeffs`) is not modelled. |
| **Wavelength uncertainty** | Fit residuals and covariance matrices are not stored in the geometry objects. |

### Non-blocking for structural use

| Gap | Description |
|-----|-------------|
| **Cross-order shift** | `get_spectral_pixelshift()` is available in `extract/wavecal.py` but not wired into the iSHELL-specific path. |
| **Live wavecal path** | The current code always uses the stored reference arrays; re-fitting from new arc frames is not yet exposed. |
| **L/Lp/M modes** | Explicitly out of scope. |

---

## 6. Testing

The test suite in `tests/test_ishell_wavecal.py` covers **structural**
behaviour only:

- `TestBuildGeometryFromWaveCalInfo` – geometry construction from all J/H/K
  modes; structural field types, shapes, and that plane-0 fitted values fall
  in expected band ranges (structural check; semantic not formally confirmed).
- `TestBuildRectificationMaps` – map shapes, that plane-0 values increase
  monotonically with column for stored K1 data, spatial axis symmetry,
  zero-tilt src_col identity.
- `TestRectifyOrders` – identity under placeholder zero-tilt, NaN masking
  outside orders, smoke test with real K1 geometry.
- `TestIntegrationPipeline` – structural pipeline smoke test for all J/H/K modes.
- `TestCalibrationsBackwardCompatibility` – new optional fields do not
  break existing readers or `build_order_geometry_set()`.

Run the test suite with:

```bash
pytest tests/test_ishell_wavecal.py -v
```

---

## 7. References

- iSHELL Spextool Manual v10jan2020, Cushing et al. (IRTF internal).
- IDL Spextool source: `mc_wavecal2dxd.pro`, `mc_findlines1dxd.pro`.
- `docs/ishell_geometry_design_note.md` – geometry data-model design.
- `docs/ishell_design_memo.md` – overall implementation status and blockers.
