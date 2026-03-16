# iSHELL Rectified-Order Image Generation — Developer Notes

## Overview

This document describes the **provisional rectified-order image generation
scaffold** for iSHELL 2DXD reduction in `pyspextool`.  The scaffold lives in:

    src/pyspextool/instruments/ishell/rectified_orders.py

It is the **seventh stage** of the 2DXD calibration scaffold, placed after
the rectification-index generation (`rectification_indices.py`).

> **Status**: This is a development scaffold, not a finalised calibration.
> The outputs are suitable for exploring the data and validating the
> rectification pipeline structure; they should **not** be used for
> science-quality spectral extraction without further validation.

---

## Purpose and Scope

This module generates **provisional rectified order images** by interpolating
a raw detector image at the sub-pixel coordinates defined by the Stage-6
rectification indices.

What it does:

* Accepts a raw detector image (2-D NumPy array) and a
  `RectificationIndexSet` (Stage-6 output).
* For each echelle order, samples the detector image at the `(src_row, src_col)`
  positions given by the rectification indices using **bilinear interpolation**.
* Returns a `RectifiedOrderSet` with one `RectifiedOrder` per echelle order.

What it does **not** do (by design):

* No tilt or curvature correction.
* No physical spatial calibration (spatial axis is fractional slit position).
* No final wavecal/spatcal FITS calibration files.
* No flux-conserving resampling (drizzle-style).
* No optimal extraction.
* No science-quality rectification.

---

## Pipeline Position

```
Flat tracing (tracing.py)
    └── FlatOrderTrace → OrderGeometrySet
            │
            ▼
Arc-line tracing (arc_tracing.py)
    └── ArcLineTraceResult
            │
            ▼
Provisional wavelength matching (wavecal_2d.py)   [Stage 3]
    └── ProvisionalWavelengthMap
            │
            ▼
Global wavelength surface (wavecal_2d_surface.py)   [Stage 4]
    └── GlobalSurfaceFitResult
            │
            ▼
Coefficient-surface refinement (wavecal_2d_refine.py)   [Stage 5]
    └── RefinedCoefficientSurface
            │
            ▼
Rectification indices (rectification_indices.py)   [Stage 6]
    └── RectificationIndexSet
            │
            ▼
Rectified order images (rectified_orders.py)   [Stage 7, THIS MODULE]
    └── RectifiedOrderSet
            │
            ▼
Final wavecal/spatcal products (NOT YET IMPLEMENTED)
```

---

## How Rectified Orders Are Constructed

### 1. Inputs

- **Detector image**: a 2-D NumPy array of shape `(n_rows, n_cols)`.
- **RectificationIndexSet**: from Stage 6, containing per-order index arrays
  `src_rows` (shape `(n_spatial, n_spectral)`) and `src_cols` (shape
  `(n_spectral,)`).

### 2. Interpolator construction

A single `scipy.interpolate.RegularGridInterpolator` is constructed over the
full detector image with:

- `method="linear"` (bilinear interpolation).
- `bounds_error=False`, `fill_value=np.nan` — coordinates outside the
  detector boundary are filled with `NaN`.

### 3. Per-order rectification

For each order *k*:

1. Retrieve `src_rows[j, i]` (detector row for spatial pixel *j*, spectral
   pixel *i*) and `src_cols[i]` (detector column for spectral pixel *i*).

2. Broadcast `src_cols` to shape `(n_spatial, n_spectral)`.

3. Flatten to a `(n_spatial × n_spectral, 2)` query array of `(row, col)`
   points.

4. Evaluate the interpolator at the query points.

5. Reshape the result to `(n_spatial, n_spectral)` to form the rectified
   flux image.

### 4. Output

A `RectifiedOrder` dataclass containing:

- `order` — order number (matches upstream rectification index).
- `order_index` — zero-based position in the parent `RectifiedOrderSet`.
- `wavelength_um` — wavelength axis, shape `(n_spectral,)`, µm.
- `spatial_frac` — spatial axis, shape `(n_spatial,)`, in `[0, 1]`.
- `flux` — rectified flux image, shape `(n_spatial, n_spectral)`.
- `source_image_shape` — `(n_rows, n_cols)` of the input detector image.

---

## Coordinate Conventions

### Detector image

`image[row, col]` — row-major indexing consistent with NumPy convention.
Row 0 is the bottom of the detector; increasing row goes toward the top.
Column 0 is the left edge; increasing column goes toward the right.

### Rectified grid

The rectified output has two axes:

**Axis 0 (rows) — spatial**
Indexed by `spatial_frac ∈ [0.0, 1.0]`:
- `0.0` = bottom order edge (as defined by the flat-tracing edge polynomial).
- `1.0` = top order edge.
- Conversion to physical arcseconds requires a spatial calibration
  (`spatcal_coeffs`) that is not yet available at this scaffold stage.

**Axis 1 (columns) — spectral**
Indexed by `wavelength_um` (uniformly spaced in µm):
- Spans the order's wavelength range as predicted by the refined coefficient
  surface.
- Wavelength increases with increasing column index (except in orders where
  the dispersion direction is reversed; the axis is guaranteed to be monotone
  either way).

### Mapping

`flux[j, i]` is the interpolated detector flux at:
- Detector row: `src_rows[j, i]` (fractional).
- Detector column: `src_cols[i]` (fractional).

---

## Relationship to Rectification Indices

Stage 6 (`rectification_indices.py`) defines the **geometric mapping** from
the rectified grid to the detector frame, without performing any image
interpolation.  Stage 7 (this module) **consumes** those index arrays and
applies them to an actual detector image to produce the rectified flux.

The two stages are intentionally separated so that:

1. The rectification indices can be computed once and reused for multiple
   detector frames (e.g., multiple flat or arc frames).

2. The mapping and the interpolation can be validated independently.

Stage 7 is a thin wrapper around `scipy.interpolate.RegularGridInterpolator`;
the geometric logic lives entirely in Stage 6.

---

## How This Differs from the Final Spextool Rectification Pipeline

The full IDL Spextool 2DXD calibration (Cushing et al. 2004) produces a
complete science-quality rectification that accounts for:

* spectral-line tilt and curvature;
* the spatial calibration in physical arcseconds;
* the wavelength dependence of the spatial sampling;
* flux-conserving resampling.

This provisional scaffold uses only:

* flat-tracing edge polynomials for the spatial mapping;
* the refined coefficient surface for the spectral mapping;
* bilinear interpolation (not flux-conserving).

| Aspect                    | IDL Spextool         | This scaffold          |
|---------------------------|----------------------|------------------------|
| Tilt correction           | Full tilt polynomial | Not applied            |
| Curvature correction      | Included             | Not applied            |
| Spatial calibration       | Physical arcseconds  | Fractional [0, 1]      |
| Resampling method         | Flux-conserving      | Bilinear interpolation |
| Wavelength solution       | Science quality      | Provisional            |
| Science quality           | Yes                  | No (provisional)       |

---

## Public API

### `build_rectified_orders(image, rectification_indices)`

Main entry point.

Parameters:

- `image` — 2-D NumPy array, shape `(n_rows, n_cols)`.  Must be 2-D.
- `rectification_indices` — `RectificationIndexSet` from Stage 6.  Must
  contain at least one order.

Returns a `RectifiedOrderSet`.

Raises `ValueError` if `image` is not 2-D or `rectification_indices` is empty.

### `RectifiedOrder`

Per-order result dataclass.  Key fields:

- `order` — order number.
- `order_index` — zero-based position in the parent `RectifiedOrderSet`.
- `wavelength_um` — shape `(n_spectral,)`, µm, uniformly spaced.
- `spatial_frac` — shape `(n_spatial,)`, values in `[0, 1]`, uniformly spaced.
- `flux` — shape `(n_spatial, n_spectral)`, bilinearly interpolated flux.
  Out-of-bounds pixels are `NaN`.
- `source_image_shape` — `(n_rows, n_cols)` of the source detector image.

Properties:

- `n_spectral`, `n_spatial`, `shape` — convenience accessors.

### `RectifiedOrderSet`

Full result container.  Key interface:

- `mode` — iSHELL mode name.
- `rectified_orders` — list of `RectifiedOrder`, one per matched order.
- `n_orders` — number of orders.
- `orders` — list of order numbers/indices.
- `source_image_shape` — `(n_rows, n_cols)` of the source detector image.
- `get_order(order)` — look up by order number.

---

## What Remains Unimplemented

The following steps are still needed for a full 2DXD solution:

1. **Spectral-line tilt correction** — apply `tilt_coeffs` from arc tracing
   to correctly map detector row onto slit position at each wavelength.

2. **Curvature correction** — apply `curvature_coeffs` for higher-order
   spectral-line shape.

3. **Physical spatial calibration** — convert the fractional slit position
   `[0, 1]` to physical arcseconds using `spatcal_coeffs` from a spatial
   calibration step.

4. **Flux-conserving resampling** — replace bilinear interpolation with a
   drizzle-style or sinc-interpolation approach that conserves flux.

5. **Science-quality wavelength solution** — the refined surface used by the
   rectification indices is provisional; it must be validated against known
   spectral features.

6. **Final wavecal/spatcal FITS products** — the rectified images produced
   here are not yet the calibrated output files expected by later extraction
   stages.

---

## Assumptions and Limitations

* Source coordinates that fall outside the detector boundary are filled with
  `NaN`.  If many output pixels are `NaN`, check that the rectification
  indices are correctly aligned with the detector image dimensions.

* Bilinear interpolation is not flux-conserving.  Integrated flux in the
  rectified frame will differ slightly from the detector frame, especially
  near sharp spectral features.

* The spatial axis is in fractional slit-position units, not physical
  arcseconds.  No conversion is applied at this stage.

* The interpolator is built once over the full detector image and reused for
  all orders.  This is efficient for multi-order instruments like iSHELL.

---

## See Also

- `docs/ishell_rectification_indices.md` — Stage-6 rectification-index scaffold.
- `docs/ishell_wavecal_2d_refine.md` — Stage-5 coefficient-surface refinement.
- `docs/ishell_2dxd_notes.md` — full pipeline overview.
- `docs/ishell_implementation_status.md` — current implementation status.
- `docs/ishell_scaffold_constraints.md` — intentional simplifications.
- Cushing, M. C., Vacca, W. D., & Rayner, J. T. 2004, PASP, 116, 362 —
  IDL Spextool paper describing the 2DXD calibration approach.
