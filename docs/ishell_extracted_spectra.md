# iSHELL 2DXD Provisional Spectral Extraction (Stage 9)

This document describes the provisional spectral-extraction scaffold
implemented in `src/pyspextool/instruments/ishell/extracted_spectra.py`.

## Overview

Stage 9 takes the rectified order images produced by Stage 7
(`rectified_orders.py`) and collapses each image along the spatial axis
to produce a provisional 1-D spectrum per echelle order.

This is **not** the final science-quality extraction stage.  It is a
scaffold step that makes the rectified images available as 1-D arrays for
downstream development.

## How the provisional extraction works

For each echelle order:

1. The rectified flux image has shape `(n_spatial, n_spectral)`, where
   axis 0 is the spatial direction and axis 1 is the spectral direction.

2. The image is collapsed along axis 0 using either:
   - **sum** (`method="sum"`): `numpy.nansum` along axis 0.
   - **mean** (`method="mean"`): `numpy.nanmean` along axis 0.

3. NaN values (arising from out-of-bounds source coordinates in Stage 7)
   are excluded from the collapse automatically by the `nan*` variants.

4. Spectral columns where *all* spatial pixels are NaN are set to NaN in
   the output.

5. The wavelength axis (`wavelength_um`) is copied unchanged from the
   rectified order.

The result is a 1-D flux array of shape `(n_spectral,)` on the
wavelength axis defined by Stage 7.

## How it differs from final Spextool extraction

The final Spextool extraction (not yet implemented) will differ in the
following ways:

| Feature | Stage 9 scaffold | Final extraction |
|---|---|---|
| Extraction method | Simple sum or mean | Optimal (profile-weighted) |
| Spatial profile | Not modelled | Fitted from data |
| Background | Not subtracted | Sky aperture subtracted |
| Variance / uncertainty | Not propagated (placeholder `None`) | Full Poisson + read-noise |
| Aperture masking | None (all rows included) | Object + sky apertures |
| Telluric correction | Not applied | Integrated |
| Order stitching | Not performed | Merged across orders |

## What remains intentionally unimplemented

The following are **out of scope** for this scaffold stage:

- Optimal extraction (profile-weighted, Horne 1986 style)
- Profile fitting or modelling
- Uncertainty / variance propagation beyond the `None` placeholder
- Background subtraction
- Aperture definition (object / sky)
- Telluric correction
- Stitching or merging across echelle orders
- Science-quality flux calibration

## Assumptions

- The input rectified images have already been flat-field corrected (or
  are flat-field frames used for scaffold development).
- The wavelength axis from Stage 7 is provisional; it has not been
  sigma-clipped or refined to science quality.
- The spatial axis is in fractional slit-position units `[0, 1]`, not
  arcseconds (scaffold simplification documented in
  `docs/ishell_scaffold_constraints.md`).

## Dataclasses

### `ExtractedOrderSpectrum`

Stores the 1-D extracted spectrum for one echelle order.

| Field | Type | Description |
|---|---|---|
| `order` | `int` | Echelle order number |
| `wavelength_um` | `ndarray (n_spectral,)` | Wavelength axis in µm |
| `flux` | `ndarray (n_spectral,)` | Collapsed 1-D flux |
| `method` | `str` | Extraction method (`"sum"` or `"mean"`) |
| `n_spatial_used` | `int` | Number of valid (non-NaN) spatial rows |
| `variance` | `ndarray or None` | Variance placeholder (always `None`) |

### `ExtractedSpectrumSet`

Container for all extracted spectra.

| Field / Property | Description |
|---|---|
| `mode` | iSHELL observing mode (e.g. `"H1"`) |
| `spectra` | List of `ExtractedOrderSpectrum` |
| `orders` | List of order numbers (property) |
| `n_orders` | Number of orders (property) |
| `get_order(n)` | Look up spectrum by order number |

## Constraints

See `docs/ishell_scaffold_constraints.md` for the general scaffold
constraints that apply to all stages.  Stage 9 additionally inherits all
simplifications from Stage 7 (no tilt/curvature correction, fractional
spatial axis).
