# iSHELL Provisional Optimal Extraction

This document describes the provisional optimal-extraction scaffold
implemented in ``optimal_extraction.py`` (Stage 12 of the iSHELL 2DXD
reduction pipeline).

## Status

This is a **scaffold implementation**.  It is intentionally simple and does
**not** represent final science-quality iSHELL or Spextool optimal extraction.
The scaffold exists to establish the pipeline stage, data containers, and
algorithmic skeleton that later stages will build upon.

---

## What is implemented

### Aperture selection

Spatial pixels are selected by fractional slit coordinate:

```
|spatial_frac - center_frac| <= radius_frac
```

This is identical to the aperture selection used in ``aperture_extraction.py``
(Stage 11).

### Background subtraction

If a background annulus is defined (``background_inner``, ``background_outer``)
and ``subtract_background=True``, a per-column background is estimated as the
**median** of pixels in the annulus:

```
background_inner < |dist| <= background_outer
```

and subtracted from every aperture pixel before profile estimation.

This is the same simple median background used in Stage 11.  It is a scaffold
simplification; real Spextool uses more sophisticated sky subtraction.

### Profile estimation

Two profile modes are supported.

#### ``"global_median"`` (default)

The spatial profile is estimated as the **per-row median** of the aperture
sub-image across all spectral columns:

```
profile_1d[i] = nanmedian(flux_ap[i, :])     for i in 0..n_ap-1
```

This produces a single 1-D profile vector that is broadcast across all
spectral columns.  It is robust to spectral emission/absorption features
because those are averaged out over wavelength.

**Limitation:** This profile cannot track spatial-profile variation along
the dispersion axis.

#### ``"columnwise"``

The spatial profile for column `j` is taken directly from the aperture
pixels in that column:

```
profile[:, j] = flux_ap[:, j]
```

This tracks spatial-profile variation with wavelength but is noisier,
especially in columns with low signal.

### Profile normalization

If ``normalize_profile=True`` (default), each column of the profile is
divided by its spatial sum:

```
profile[:, j] /= sum(profile[:, j])
```

Columns where the profile sum is zero or NaN (no valid profile support)
are set to NaN.

Negative profile values are clipped to zero before normalization in both
profile modes.  This prevents negative weights from inverting the sign of
the extracted flux.

### Weighted extraction formula

The 1-D extraction is computed as:

```
flux_1d[j] = sum(P[:, j] * F[:, j]) / sum(P[:, j]^2)
```

where:
- `P` is the spatial-profile array, shape `(n_ap_spatial, n_spectral)`
- `F` is the background-subtracted aperture sub-image, same shape
- summation is over the spatial axis, using NaN-safe reduction

This formula is the **unweighted least-squares estimator** of the flux
scalar `a` for the linear model `F[:, j] = a * P[:, j]`.

Columns where the denominator is zero, NaN, or where all aperture pixels
are NaN return NaN in the output.

---

## How this differs from real Spextool optimal extraction

Real Spextool implements Horne (1986) optimal extraction, which requires:

1. **A noise model** (read noise, gain, Poisson noise per pixel).
2. **Variance-weighted extraction**:
   ```
   flux_1d[j] = sum(M * F / V) / sum(M^2 / V)
   ```
   where `V` is the per-pixel variance.
3. **Iterative cosmic-ray / bad-pixel rejection** using the profile and
   variance model to detect and mask outliers.
4. **A reliable PSF / profile model**, potentially fitted to the data rather
   than taken directly from it.

This scaffold does not implement full Horne (1986) variance-weighted
optimal extraction or iterative outlier rejection.

A first-pass noise model is available via the Stage 14 variance model,
and variance can be propagated through the extraction.  However, the
extraction weights are not derived from an inverse-variance model of
the data, and no iterative refinement or bad-pixel rejection is applied.

---

## What remains intentionally unimplemented

- Horne (1986) variance-weighted optimal extraction
- Iterative outlier rejection
- Sophisticated PSF / profile fitting
- Science-quality uncertainty propagation (current variance support is a
  first-pass approximation)
- Automatic aperture centroiding
- Sky modelling beyond simple median subtraction
- Telluric correction
- Order merging / stitching
- Science-quality validation

---

## First-pass variance propagation

The extraction functions can optionally generate a variance image internally
using the Stage 14 variance model.  Pass either:

* `variance_image` – an explicit per-pixel variance array (takes priority), **or**
* `variance_model` – a `VarianceModelDefinition` that is used to build the
  variance image internally via `build_variance_image()`.

Priority: `variance_image > variance_model > None`.

An optional `variance_image` argument may be passed to `extract_optimal`.
When provided (directly or built from `variance_model`), variance is propagated
through background subtraction and profile-weighted extraction:

```
var_flux[col] = sum(P[:, col]**2 * (var_pixel + var_bg))
```

where:
- `P` is the spatial-profile array (same as used for flux extraction)
- `var_pixel` is the per-pixel variance from the variance image
- `var_bg` is the per-column background variance, approximated as the
  median of the variance image within the background annulus

> **Note:** Background variance is approximated using the median of the
> variance image within the background annulus.  This is a first-order
> approximation and does not account for correlated noise or detailed
> detector characteristics.

When neither `variance_image` nor `variance_model` is supplied (default),
no variance is computed and the `variance` field is `None`.

---

## Data containers

### ``OptimalExtractionDefinition``

Holds the aperture center, radius, optional background annulus, profile
mode, and normalization flag.  Validates all parameters on construction.

### ``OptimalExtractedOrderSpectrum``

Per-order 1-D spectrum.  Fields:

| Field | Description |
|-------|-------------|
| ``order`` | Echelle order number |
| ``wavelength_um`` | Wavelength axis in µm |
| ``flux`` | Weighted 1-D flux |
| ``profile`` | Profile array used as weights, shape `(n_ap, n_spectral)` |
| ``aperture`` | The ``OptimalExtractionDefinition`` used |
| ``method`` | Always ``"optimal_weighted"`` |
| ``n_pixels_used`` | Aperture pixels with at least one finite value |
| ``variance`` | Propagated variance of ``flux``, or ``None`` if no variance source was supplied |

### ``OptimalExtractedSpectrumSet``

Collection of per-order spectra.  Methods: ``get_order(order)``,
``orders`` (list), ``n_orders`` (int).

---

## Scaffold constraints

- Do not retroactively modify earlier pipeline stages.
- All spatial coordinates remain in fractional slit units ``[0, 1]``.
- The wavelength axis is copied from the rectified order (not a reference).
- NaN values from the rectification step are propagated silently.

See ``docs/ishell_scaffold_constraints.md`` for the full list of scaffold
constraints.
