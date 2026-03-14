# iSHELL Wavelength Calibration and Rectification – Developer Design Note

## Overview

This document describes the first working iSHELL J/H/K ThAr wavelength-
calibration and rectification implementation in pySpextool.  It covers:

1. The 1DXD/2DXD-style wavelength calibration approach.
2. How the geometry objects (`OrderGeometry`, `OrderGeometrySet`) are populated.
3. The rectification algorithm (`_rectify_orders`).
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
| `instruments/ishell/wavecal.py` | **NEW** ThAr wavecal and rectification-map builder |
| `instruments/ishell/ishell.py` | **UPDATED** `_rectify_orders()` implementation |

### Data flow

```
read_flatinfo(mode)     → FlatInfo        (edge polynomials, plate scale)
read_wavecalinfo(mode)  → WaveCalInfo     (stored wavelength arrays, xranges)
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
rectified image (same shape as input; NaN outside all orders)
```

---

## 2. Wavelength Calibration (1DXD-style from stored solution)

### Stored reference data

Every `*_wavecalinfo.fits` file is a data cube of shape
`(n_orders, 4, n_pixels)` where:

| Plane | Contents |
|-------|---------|
| 0 | Pre-computed reference wavelengths (µm) from the 2DXD solution |
| 1 | Reference ThAr arc spectrum (DN/s) |
| 2 | Uncertainty |
| 3 | Quality flags |

Plane 0 was computed in legacy IDL Spextool by fitting a 2D cross-dispersed
polynomial of the form:

```
λ(col, m) = Σ_i Σ_j  P2W_C[i*(xorder+1)+j]  * col^j  * (m/m0)^i
```

where `m` is the echelle order number, `m0 = HOMEORDR`, `xorder = DISPDEG`,
and the order polynomial degree is `ORDRDEG`.  All 12 polynomial coefficients
are stored in the FITS header as `P2W_C00` … `P2W_C11`.

### Per-order polynomial fit

`build_geometry_from_wavecalinfo()` does **not** re-evaluate the 2D
polynomial.  Instead, it reads the pre-computed wavelength array from plane 0
and fits a 1D polynomial to `(column, wavelength)` for each order:

```python
# Array index j → detector column x_start + j
cols = np.arange(n_valid) + x_start
wavs = wavecalinfo.data[order_idx, 0, :][~np.isnan(...)]
coeffs = np.polynomial.polynomial.polyfit(cols, wavs, degree)
```

The resulting `wave_coeffs` follow the `numpy.polynomial.polynomial`
convention: `coeffs[k]` is the coefficient of `col^k`.  The polynomial is
valid in the range `[x_start, x_end]` for that order.

Column ranges come from the `OR{n}_XR` FITS header keywords, now parsed
into `WaveCalInfo.xranges` and `FlatInfo.xranges`.

### Confirmed vs inferred

| Fact | Status |
|------|--------|
| Plane 0 is wavelength in µm | **Confirmed** – values match expected J/H/K bands |
| `OR{n}_XR` gives valid pixel range | **Confirmed** – matches array valid mask |
| 2DXD polynomial coefficients in header | **Confirmed** – present as `P2W_C00`…`P2W_C11` |
| Plane 1 is the reference arc spectrum | **Confirmed** by sign / scale of values |
| `LINEDEG = 1` is the degree used for line-tilt fitting | **Present** but not yet used |

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
| `wave_coeffs` | Fit to `wavecalinfo.data[i, 0, :]` | 1D poly, degree = `disp_degree` |
| `tilt_coeffs` | `[0.0]` | **Provisional** zero approximation |
| `spatcal_coeffs` | `[0.0, plate_scale_arcsec]` | Linear: arcsec = ps × row_offset |
| `curvature_coeffs` | `None` | Not yet implemented |

### Edge polynomials

The flat FITS header stores per-order edge polynomial coefficients as
`OR{n}_B1`…`OR{n}_B5` (bottom edge) and `OR{n}_T1`…`OR{n}_T5` (top edge).
These are degree-4 polynomials in the detector column.  They follow the
`numpy.polynomial.polynomial` convention: `OR{n}_B1` is the constant term.

The `EDGEDEG` header keyword gives the polynomial degree (4 for all current
iSHELL modes).

---

## 4. Rectification Algorithm

### `build_rectification_maps()`

For each order, `build_rectification_maps()` computes source detector
coordinates `(src_rows, src_cols)` for a regular output grid:

**Output grid:**

- **Spectral axis:** one pixel per detector column from `x_start` to `x_end`
  (integer columns), converted to wavelengths via `wave_coeffs`.
- **Spatial axis:** uniformly spaced from `−half_slit` to `+half_slit`
  arcseconds, where `half_slit = 0.5 × slit_height_arcsec`.

**Source coordinate computation** (for output pixel `(spatial_i, spectral_j)`):

```
col_out = x_start + j
row_ctr = eval_centerline(col_out)
row_offset_pix = spatial_arcsec[i] / plate_scale_arcsec
tilt = eval_tilt(col_out)           # = 0.0 in current implementation
src_col = col_out + tilt * row_offset_pix
src_row = row_ctr + row_offset_pix
```

The tilt correction shifts the source column as a function of spatial offset.
With zero tilt, `src_col = col_out` (no horizontal correction needed).

### `_rectify_orders()`

`_rectify_orders(image, geometry, plate_scale_arcsec)` performs the actual
image resampling:

1. Calls `build_rectification_maps()` to get `src_rows`, `src_cols` for
   every order.
2. For each order, computes a masked grid of `(n_rows, n_cols)` covering the
   order footprint on the detector.
3. Applies `scipy.ndimage.map_coordinates` with bilinear interpolation
   (`order=1`) to look up source pixel values.
4. Writes results into the output image at the same pixel coordinates,
   but using the tilt-corrected source column.
5. Pixels outside all order footprints are set to NaN.

The output image has the same shape `(nrows, ncols)` as the input.

### Behaviour with zero tilt

With the current provisional zero-tilt model (`tilt_coeffs = [0.0]`), the
rectification is equivalent to:

```
output[r, c] = bilinear_interp(input, r, c)   ∀ (r, c) within an order
```

i.e. the identity transformation within each order.  The output differs from
the input only for pixels outside the order footprints (which are set to NaN).

This is correct and expected behaviour: a zero-tilt model introduces no
horizontal shift, so the values are preserved.  Real tilt correction will
become non-trivial once arc-line fitting at multiple row positions is
implemented.

---

## 5. What Remains Incomplete

### Blocking for science use

| Gap | Description |
|-----|-------------|
| **Real tilt measurement** | Arc-line centroids must be fitted at multiple row positions to measure `d(col)/d(row)` at each column. The `LINEDEG` header keyword (= 1) likely gives the polynomial degree expected for the tilt fit in IDL Spextool. |
| **2DXD iteration** | Full interactive line identification, robust outlier rejection, and iterative refinement are not implemented. |
| **Slit curvature** | Higher-order spatial distortion (`curvature_coeffs`) is not modelled. |
| **Wavelength uncertainty** | Fit residuals and covariance matrices are not stored in the geometry objects. |

### Non-blocking for initial use

| Gap | Description |
|-----|-------------|
| **Cross-order shift** | `get_spectral_pixelshift()` is available in `extract/wavecal.py` but not wired into the iSHELL-specific path. |
| **Stored vs. measured** | The current code always uses the stored reference solution; interactive re-fitting from new arc frames is not yet exposed. |
| **L/Lp/M modes** | Explicitly out of scope per `instruments/ishell/modes.yaml` and this PR. |

---

## 6. Testing

The test suite in `tests/test_ishell_wavecal.py` covers:

- `TestBuildGeometryFromWaveCalInfo` – construction from all J/H/K modes,
  field types and values, band-correctness of wavelengths.
- `TestBuildRectificationMaps` – map shapes, wavelength monotonicity,
  spatial symmetry, zero-tilt source column identity.
- `TestRectifyOrders` – identity under zero tilt, NaN outside orders,
  end-to-end smoke test with real K1 geometry.
- `TestIntegrationPipeline` – full pipeline for all J/H/K modes.
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
