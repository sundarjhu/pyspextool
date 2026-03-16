# iSHELL Rectification-Index Scaffold — Developer Notes

## Overview

This document describes the **provisional rectification-index scaffold** for
iSHELL 2DXD reduction in `pyspextool`.  The scaffold lives in:

    src/pyspextool/instruments/ishell/rectification_indices.py

It is the **sixth stage** of the 2DXD calibration scaffold, placed after the
coefficient-surface refinement (`wavecal_2d_refine.py`).

> **Status**: This is a development scaffold, not a finalised calibration.
> The outputs are suitable for exploring the data and validating the
> geometric mapping structure; they should **not** be used for science-quality
> spectral extraction without further validation.

---

## Purpose and Scope

This module constructs **provisional rectification indices** that describe
a mapping from a rectified (wavelength × spatial) coordinate grid to
approximate detector (row, column) positions for each echelle order.

What it does:

* Accepts an `OrderGeometrySet` (order edge polynomials and column ranges from
  flat tracing) and a `RefinedCoefficientSurface` (Stage-5 output).
* For each echelle order, defines a provisional rectified grid with:
  - a **spectral axis**: uniformly spaced wavelengths (µm) spanning the
    order's wavelength range as predicted by the refined surface; and
  - a **spatial axis**: fractional slit position uniformly spaced in [0, 1],
    where 0 is the bottom order edge and 1 is the top edge.
* For each output grid point, computes approximate detector coordinates:
  - `src_col` via linear interpolation of the surface-sampled wavelength→column curve;
  - `src_row` from the flat-tracing edge polynomials:

        row = bottom_edge(col) + frac × (top_edge(col) − bottom_edge(col))

* Returns a `RectificationIndexSet` collecting `RectificationIndexOrder`
  objects, one per matched order.

What it does **not** do:

* No actual image resampling or interpolation of detector data.
* No final wavecal/spatcal image generation.
* No science-quality rectification.
* No tilt or curvature correction beyond straight edge polynomials.
* No iterative outlier rejection.

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
Rectification indices (rectification_indices.py)   [Stage 6, THIS MODULE]
    └── RectificationIndexSet
            │
            ▼
wavecal/spatcal image generation (NOT YET IMPLEMENTED)
```

---

## What the Rectification Indices Represent

A rectification index answers the question:

> For each pixel in the _rectified_ output array (wavelength pixel `i`,
> spatial pixel `j`), what is the corresponding location in the _detector_
> frame?

The detector location is given as fractional (sub-pixel) coordinates
`(src_row, src_col)`.  These can be passed directly to
`scipy.ndimage.map_coordinates` or equivalent routines for bilinear
interpolation.

### Spectral axis

The output wavelength grid is derived by:

1. Sampling the refined coefficient surface at `n_col_samples` evenly-spaced
   columns across the order's column range.
2. Sorting the resulting (col, wavelength) pairs by wavelength.
3. Defining `n_spectral` uniformly-spaced wavelengths spanning
   `[wavelength_min, wavelength_max]`.
4. Inverting via linear interpolation to get the detector column for each
   output wavelength.

The wavelength axis is uniformly spaced in µm (not in column number).  This
is one of the simplest rectification choices; a more sophisticated scheme
might sample uniformly in `log(λ)` or in velocity space.

### Spatial axis

The output spatial axis is fractional slit position in `[0, 1]`, where
`0` corresponds to the bottom order edge and `1` to the top edge (as defined
by the flat-tracing edge polynomials).

At each output wavelength (i.e., at the corresponding `src_col`), the
detector row is computed by linear interpolation between the bottom and top
edge evaluations:

    src_row = bottom_edge(src_col) + frac × (top_edge(src_col) − bottom_edge(src_col))

No conversion to physical arcseconds is applied at this stage; that requires
a spatial calibration (`spatcal_coeffs`) that is not yet available.

---

## How This Provisional Implementation Differs from the Full IDL 2DXD Method

The full IDL Spextool 2DXD calibration (Cushing et al. 2004) generates a
complete rectification map that accounts for:

* spectral-line tilt (the mapping from detector row to slit position at each
  wavelength);
* spectral-line curvature;
* the spatial calibration in physical arcseconds;
* the wavelength dependence of the spatial sampling.

This provisional scaffold uses only:

* the flat-tracing edge polynomials (no tilt or curvature correction);
* the refined coefficient surface for the spectral mapping;
* linear interpolation for the wavelength→column inversion.

| Aspect | IDL Spextool | This scaffold |
|--------|-------------|---------------|
| Tilt correction | Full tilt polynomial | Not applied |
| Curvature correction | Included | Not applied |
| Spatial calibration | In physical arcseconds | Fractional [0, 1] |
| Wavelength inversion | Exact polynomial root-finding | Linear interpolation |
| Column normalization | Normalized per-order | Raw column units |
| Science quality | Yes | No (provisional) |

---

## Order Matching

Orders are matched between the geometry and the surface by **echelle order
number**.

When the geometry is produced by
`FlatOrderTrace.to_order_geometry_set()`, it uses placeholder 0-indexed
order numbers (0, 1, 2, …) rather than real echelle order numbers.  In
this case, the `ProvisionalWavelengthMap` (`wav_map`) must be passed as an
additional argument to `build_rectification_indices`; its solution list
bridges the placeholder index → real order number correspondence.

When the geometry is constructed with real echelle order numbers (e.g. via
`build_order_geometry_set()` using data from `WaveCalInfo`), no `wav_map`
is needed.

Geometry orders with no corresponding surface fit are silently skipped.
This is expected when some echelle orders have too few arc-line matches to
be fitted by the Stage-5 surface.

---

## Public API

### `build_rectification_indices(geometry, surface, *, wav_map=None, n_spectral=256, n_spatial=64, n_col_samples=1024)`

Main entry point.  Accepts an `OrderGeometrySet` and a
`RefinedCoefficientSurface` (and optionally a `ProvisionalWavelengthMap`
for order-number resolution) and returns a `RectificationIndexSet`.

Parameters:

- `geometry` — `OrderGeometrySet` from flat tracing.
- `surface` — `RefinedCoefficientSurface` from Stage 5.
- `wav_map` — `ProvisionalWavelengthMap` from Stage 3, required when
  geometry uses placeholder order indices.
- `n_spectral` — number of spectral output pixels per order (default 256).
- `n_spatial` — number of spatial output pixels per order (default 64).
- `n_col_samples` — resolution of the wavelength→column inversion
  sampling (default 1024).

### `RectificationIndexOrder`

Per-order result dataclass.  Key fields:

- `order` — order number (or index) from the geometry.
- `order_index` — zero-based position in the parent `RectificationIndexSet`.
- `output_wavelengths_um` — shape `(n_spectral,)`, µm, uniformly spaced.
- `output_spatial_frac` — shape `(n_spatial,)`, values in `[0, 1]`, uniformly
  spaced.
- `src_cols` — shape `(n_spectral,)`, fractional detector columns.
- `src_rows` — shape `(n_spatial, n_spectral)`, fractional detector rows.

Properties:

- `n_spectral`, `n_spatial`, `output_shape` — convenience accessors.

### `RectificationIndexSet`

Full result container.  Key interface:

- `mode` — iSHELL mode name.
- `index_orders` — list of `RectificationIndexOrder`, one per matched order.
- `n_orders` — number of orders.
- `orders` — list of order numbers/indices.
- `get_order(order)` — look up by order number.

---

## What Remains Unimplemented

The following steps are still needed for a full 2DXD solution:

1. **Spectral-line tilt correction** — apply `tilt_coeffs` from arc tracing
   to map detector row accurately onto slit position at each wavelength.

2. **Curvature correction** — apply `curvature_coeffs` for higher-order
   spectral-line shape.

3. **Physical spatial calibration** — convert the fractional slit position
   `[0, 1]` to physical arcseconds using `spatcal_coeffs` from a spatial
   calibration step.

4. **Final wavecal/spatcal image generation** — resample detector frames
   onto the rectified grid using the `src_rows` / `src_cols` arrays with
   `scipy.ndimage.map_coordinates`.

5. **Science-quality wavelength solution** — the refined surface used here
   is still provisional; validated against known line atlases or object spectra.

6. **Column normalization** — the wavelength→column inversion uses raw column
   units; normalizing to `[-1, +1]` within each order would improve numerical
   accuracy for higher-degree fits.

---

## Assumptions and Limitations

* The per-order wavelength function must be approximately monotone in
  detector column.  A `RuntimeWarning` is emitted if the sampled surface
  shows non-monotone behaviour; the resulting `src_cols` may be unreliable
  in that case.

* The spatial mapping uses the flat-tracing edge polynomials.  These are
  approximations: `to_order_geometry_set()` offsets the traced centre-line
  by a constant half-width, not independently-fitted edges.

* The `src_cols` and `src_rows` contain fractional detector coordinates
  intended for interpolation.  No bounds-checking against the detector
  array size is performed.

* The echelle order numbers from the coefficient surface are provisional
  until validated against known spectral features.

---

## See Also

- `docs/ishell_wavecal_2d_refine.md` — Stage-5 coefficient-surface refinement.
- `docs/ishell_2dxd_notes.md` — full pipeline overview.
- `docs/ishell_implementation_status.md` — current implementation status.
- Cushing, M. C., Vacca, W. D., & Rayner, J. T. 2004, PASP, 116, 362 —
  IDL Spextool paper describing the 2DXD calibration approach.
