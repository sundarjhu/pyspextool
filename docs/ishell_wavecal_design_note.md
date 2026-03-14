# iSHELL Geometry Population and Rectification – Developer Design Note

## Overview

This document describes the iSHELL J/H/K geometry-population and image-
resampling implementation in pySpextool.

The implementation provides two distinct wavelength-calibration paths:

1. **Structural bootstrap** (`build_geometry_from_wavecalinfo`): fits
   polynomials to pre-stored plane-0 reference wavelength arrays.  This is
   a compact polynomial re-encoding of stored values; it is not a
   re-measurement from arc data.

2. **First measured arc-line fitting** (`build_geometry_from_arc_lines`):
   uses the stored ThAr arc-lamp spectra (plane 1 of the wavecalinfo cube)
   together with the packaged line lists to measure arc-line centroids via
   Gaussian profile fitting, then derives a polynomial wavelength solution
   from those measured positions.  This is the **first true measured** arc-
   line fitting stage.

The **tilt model is still a provisional placeholder zero** because tilt
measurement requires 2-D arc images (spatial variation of line positions
across the slit), which are not stored in the packaged calibration files.
See §5 for the complete list of what remains incomplete.

This document covers:

1. What the implementation currently does (and what the two wavecal paths are).
2. The confirmed data-cube plane semantics.
3. How the geometry objects are populated.
4. The resampling algorithm.
5. What remains incomplete relative to legacy IDL Spextool.
6. Testing coverage.

**Scope:** J, H, and K bands only.  L/Lp/M modes are out of scope.

---

## 1. Architecture

### Key modules

| Module | Purpose |
|--------|---------|
| `instruments/ishell/geometry.py` | Data model: `OrderGeometry`, `OrderGeometrySet`, `RectificationMap` |
| `instruments/ishell/calibrations.py` | Typed readers for packaged calibration files |
| `instruments/ishell/wavecal.py` | Geometry population (bootstrap + measured) and rectification-map builder |
| `instruments/ishell/ishell.py` | `_rectify_orders()` implementation |

### Data flow (structural bootstrap path)

```
read_flatinfo(mode)     → FlatInfo        (edge polynomials, plate scale)
read_wavecalinfo(mode)  → WaveCalInfo     (stored reference arrays, xranges)
          ↓
build_geometry_from_wavecalinfo(wci, fi)
          ↓
OrderGeometrySet  (wave_coeffs from plane-0 fit, tilt_coeffs=0 placeholder)
          ↓
build_rectification_maps(geom_set, plate_scale_arcsec)
          ↓
list[RectificationMap]
          ↓
_rectify_orders(image, geom_set, plate_scale_arcsec)
          ↓
resampled image (same shape; NaN outside orders; provisional zero-tilt)
```

### Data flow (first measured arc-line fitting path)

```
read_flatinfo(mode)     → FlatInfo
read_wavecalinfo(mode)  → WaveCalInfo     (plane 0: wavelength grid,
read_line_list(mode)    → LineList         plane 1: arc-lamp spectrum)
          ↓
fit_arc_line_centroids(wci, line_list)
  – for each order: identify expected line positions from plane 0
  – extract window of arc spectrum (plane 1) around each position
  – fit Gaussian profile → measured centroid pixel
  – filter by SNR and reject blended lines
          ↓
build_geometry_from_arc_lines(wci, fi, line_list)
  – fit polynomial to (centroid_pixel, known_wavelength) pairs
  – falls back to plane-0 bootstrap for orders with < 4 accepted lines
          ↓
OrderGeometrySet  (wave_coeffs from measured centroids, tilt_coeffs=0 provisional)
```

---

## 2. Confirmed Data-Cube Plane Semantics

Every `*_wavecalinfo.fits` file is a data cube of shape `(n_orders, 4, n_pixels)`.
Inspection of the FITS header keywords `XUNITS="um"` and `YUNITS="DN / s"`
together with value ranges confirms:

| Plane | Status | Meaning |
|-------|--------|---------|
| 0 | **Confirmed** (XUNITS="um", values in J/H/K bands) | Wavelength grid in µm along the order centerline |
| 1 | **Confirmed** (YUNITS="DN / s", full range of real arc spectrum) | ThAr arc-lamp spectrum in DN/s along the order centerline |
| 2 | **Likely** uncertainty array | Uncertainty (noise) of the arc spectrum |
| 3 | **Confirmed** quality flags (values 0/1/2) | 0=not a line, 1=identified but excluded by IDL pipeline, 2=used in IDL pipeline fit |

The FITS header also contains:

- `P2W_C00`…`P2W_C11` – 12 coefficients of the 2DXD polynomial model from
  the IDL Spextool pipeline.  These are the full pixel-to-wavelength solution
  stored by the IDL pipeline; they are not yet used directly by the Python
  implementation.
- `OR{n}_XR` – per-order column ranges (confirmed to match the valid non-NaN
  region of the data cube).
- `LINEDEG = 1` – degree of the tilt polynomial used by the IDL pipeline.
  Tilt fitting is not yet implemented in Python (requires 2-D arc images).

---

## 3. Geometry Population

### 3a. Structural bootstrap: `build_geometry_from_wavecalinfo()`

Fits a polynomial to the stored plane-0 wavelength values:

```python
cols = np.arange(n_valid) + x_start   # array_index → column mapping
wavs = wavecalinfo.data[order_idx, 0, :][~np.isnan(...)]
coeffs = np.polynomial.polynomial.polyfit(cols, wavs, degree)
```

This is a compact polynomial representation of the stored reference values.

### 3b. First measured arc-line fitting: `build_geometry_from_arc_lines()`

Derives coefficients from measured arc-line centroid positions:

1. `fit_arc_line_centroids(wci, line_list)` identifies each line from the
   line list in the stored arc spectrum (plane 1) using the wavelength grid
   (plane 0) as the reference.  Gaussian profiles are fitted to measure
   precise centroid pixel positions.  Blended lines (two wavelengths at the
   same pixel) are rejected.

2. For each order with ≥ `min_lines_per_order` accepted centroids, a polynomial
   is fitted to `(centroid_pixel, known_wavelength_um)` pairs.

3. Orders with fewer accepted lines fall back to the plane-0 bootstrap and
   emit a `RuntimeWarning`.

### Populated geometry fields

After either path, each `OrderGeometry` in the returned `OrderGeometrySet` has:

| Field | Bootstrap source | Measured source | Notes |
|-------|-----------------|-----------------|-------|
| `order` | `wavecalinfo.orders[i]` | same | Echelle order number |
| `x_start`, `x_end` | `flatinfo.xranges[i]` | same | Column range from flat |
| `bottom_edge_coeffs` | `flatinfo.edge_coeffs[i, 0, :]` | same | From `OR{n}_B1…B5` |
| `top_edge_coeffs` | `flatinfo.edge_coeffs[i, 1, :]` | same | From `OR{n}_T1…T5` |
| `wave_coeffs` | Poly fit to plane-0 values | Poly fit to measured centroids | Different fitting basis |
| `tilt_coeffs` | `[0.0]` **provisional** | `[0.0]` **provisional** | Requires 2-D arc images |
| `spatcal_coeffs` | `[0.0, plate_scale]` provisional | same | Linear model only |
| `curvature_coeffs` | `None` | `None` | Not implemented |

---

## 4. Resampling Algorithm (`_rectify_orders`)

> **Note on tilt:** with the current **provisional zero-tilt** model,
> `_rectify_orders` applies NaN masking outside order footprints and bilinear
> resampling but does **not** correct spectral-line tilt.

### `build_rectification_maps()`

For each order, computes source detector coordinates `(src_rows, src_cols)`:

**Source coordinate computation** (for output pixel `(spatial_i, spectral_j)`):

```
col_out = x_start + j
row_ctr = eval_centerline(col_out)
row_offset_pix = spatial_arcsec[i] / plate_scale_arcsec
tilt = eval_tilt(col_out)           # = 0.0 – provisional; requires 2-D arc images
src_col = col_out + tilt * row_offset_pix   # = col_out when tilt=0
src_row = row_ctr + row_offset_pix
```

### `_rectify_orders()`

1. Calls `build_rectification_maps()` to get `src_rows`, `src_cols`.
2. Applies `scipy.ndimage.map_coordinates` with bilinear interpolation.
3. Pixels outside all order footprints are set to NaN.

---

## 5. What Remains Incomplete

### Blocking for full science quality

| Gap | Description |
|-----|-------------|
| **Tilt measurement** | Requires arc-line centroids at multiple *spatial* row positions (a 2-D arc image). The 1-D centerline arc spectrum stored in plane 1 cannot provide this. `LINEDEG=1` in the FITS header indicates the IDL pipeline used a linear tilt model. |
| **2DXD global model** | Full iterative arc-line identification and outlier rejection against the `P2W_C*` 2DXD polynomial are not implemented. |
| **Slit curvature** | `curvature_coeffs` is not modelled. |
| **Wavelength uncertainty** | Fit residuals and covariance are not stored. |
| **Fallback orders** | Orders with fewer than `min_lines_per_order` accepted centroids fall back to the bootstrap; increasing the line density in the line list files would reduce the fallback count. |

### Non-blocking for structural use

| Gap | Description |
|-----|-------------|
| **L/Lp/M modes** | Explicitly out of scope. |
| **Cross-order shift** | Not wired into the iSHELL-specific path. |

---

## 6. Testing

The test suite in `tests/test_ishell_wavecal.py` covers:

- `TestBuildGeometryFromWaveCalInfo` – bootstrap geometry construction from
  all J/H/K modes; structural field types, shapes, plane-0 band ranges.
- `TestFitArcLineCentroids` – centroid measurement for J0/H1/K1:
  centroids within column range, SNR above threshold, wavelengths matching
  line list, at least one order with accepted measurements, failure cases.
- `TestBuildGeometryFromArcLines` – measured wavelength solution for J0/H1/K1
  and all modes; band-range check, provisional zero-tilt verification,
  fallback behaviour, difference from bootstrap coefficients, synthetic test.
- `TestBuildRectificationMaps` – map shapes, wavelength monotonicity, spatial
  axis symmetry, zero-tilt src_col identity.
- `TestRectifyOrders` – identity under zero-tilt, NaN masking, smoke tests.
- `TestIntegrationPipeline` – structural pipeline smoke test for all J/H/K modes.
- `TestCalibrationsBackwardCompatibility` – new fields do not break existing readers.

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
