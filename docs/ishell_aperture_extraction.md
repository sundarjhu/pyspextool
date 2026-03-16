# iSHELL Aperture Extraction Scaffold (Stage 11)

This document describes **Stage 11** of the iSHELL 2DXD reduction scaffold:
the aperture-aware spectral extraction implemented in
`src/pyspextool/instruments/ishell/aperture_extraction.py`.

> **Scaffold notice:** This module is intentionally simple.  It is not a
> final science-quality implementation of iSHELL or Spextool extraction.
> See [What remains unimplemented](#what-remains-unimplemented) for a full
> list of limitations.

## Pipeline stage table

| Stage | Module                     | Purpose                              |
|-------|----------------------------|--------------------------------------|
| 1     | `tracing.py`               | Flat-order tracing                   |
| 2     | `arc_tracing.py`           | Arc-line tracing                     |
| 3     | `wavecal_2d.py`            | Per-order provisional wavelength mapping |
| 4     | `wavecal_2d_surface.py`    | Provisional global wavelength surface |
| 5     | `wavecal_2d_refine.py`     | Coefficient-surface refinement       |
| 6     | `rectification_indices.py` | Rectification-index generation       |
| 7     | `rectified_orders.py`      | Rectified order images               |
| 8     | `calibration_products.py`  | Provisional calibration product containers |
| 9     | `calibration_fits.py`      | FITS calibration writer              |
| 10    | `extracted_spectra.py`     | Whole-slit provisional extraction    |
| **11** | **`aperture_extraction.py`** | **Aperture-aware extraction (this module)** |

---

## Overview

Stage 11 adds support for:

* **Object aperture extraction** – only spatial pixels within a user-defined
  aperture contribute to the 1-D spectrum.
* **Optional background subtraction** – a per-column background level is
  estimated from a background annulus and subtracted before collapsing.

It consumes a
[`RectifiedOrderSet`](../src/pyspextool/instruments/ishell/rectified_orders.py)
(Stage 7 output) and produces an `ExtractedApertureSpectrumSet`.

---

## Aperture definition

An aperture is described by an `ApertureDefinition` dataclass:

```python
@dataclass
class ApertureDefinition:
    center_frac: float            # slit center, in [0, 1]
    radius_frac: float            # half-width, > 0
    background_inner: float|None  # background annulus inner edge (optional)
    background_outer: float|None  # background annulus outer edge (optional)
```

All spatial coordinates are **fractional slit coordinates** in `[0, 1]`,
matching the `spatial_frac` axis stored in each `RectifiedOrder`.

* `0.0` = bottom edge of the order on the detector.
* `1.0` = top edge of the order on the detector.

The aperture **half-width** `radius_frac` determines which spatial pixels
are included:

```
|spatial_frac − center_frac| ≤ radius_frac   →   aperture pixel
```

---

## Background region definition

The background annulus is defined by two distances from `center_frac`:

```
background_inner ≤ |spatial_frac − center_frac| ≤ background_outer
```

The background region is **symmetric** about the aperture center: it
includes pixels on both sides of the object (in the slit direction).

### Validation rules

| Condition | Effect |
|-----------|--------|
| `background_inner` and `background_outer` are both `None` | No background subtraction |
| One is `None`, the other is not | `ValueError` |
| `background_inner ≤ radius_frac` | `ValueError` (would overlap aperture) |
| `background_outer ≤ background_inner` | `ValueError` |

---

## Extraction algorithm

For each order in the `RectifiedOrderSet`:

1. **Select aperture pixels.**

   Compute `dist = |spatial_frac − center_frac|` for every spatial pixel.
   Select rows where `dist ≤ radius_frac`.

2. **Estimate background** *(if enabled).*

   Select background rows where
   `background_inner ≤ dist ≤ background_outer`.

   Compute a per-column background estimate as the median of those rows:

   ```python
   background = np.nanmedian(flux_bg, axis=0)   # shape (n_spectral,)
   ```

   Subtract this background from every aperture row:

   ```python
   flux_ap = flux_ap − background[np.newaxis, :]
   ```

3. **Collapse the aperture.**

   * `"sum"` – `np.nansum(flux_ap, axis=0)`; all-NaN columns → NaN.
   * `"mean"` – `np.nanmean(flux_ap, axis=0)`; all-NaN columns → NaN.

4. **Store the result.**

   One `ExtractedApertureSpectrum` per order, containing:

   | Field | Content |
   |-------|---------|
   | `order` | Echelle order number |
   | `wavelength_um` | Wavelength axis in µm (copy) |
   | `flux` | Background-subtracted 1-D flux |
   | `background_flux` | Per-column background estimate, or `None` |
   | `aperture` | The `ApertureDefinition` used |
   | `method` | `"sum"` or `"mean"` |
   | `n_pixels_used` | Number of valid aperture rows used |

---

## NaN handling

`np.nanmedian`, `np.nansum`, and `np.nanmean` are used throughout.
NaN values in the rectified flux image (e.g. out-of-bounds rectification
pixels) are automatically excluded.

Spectral columns where **all** aperture pixels are NaN produce NaN in the
output `flux`.

---

## Differences from real Spextool extraction

The following simplifications are made intentionally in this scaffold:

| Feature | Scaffold behaviour | Real Spextool |
|---------|--------------------|---------------|
| Aperture center | Supplied explicitly by the caller | Located by finding the spectral trace or Gaussian fit |
| Aperture width | Fixed fractional half-width | Profile-adaptive; may vary per column |
| Background estimation | Per-column `np.nanmedian` of symmetric annulus | Polynomial fit to the background region; independent apertures on each side |
| Sky subtraction | Simple subtraction of median background | Separate sky frame or fitted sky model |
| Extraction method | Sum or mean | Optimal (profile-weighted) extraction |
| Uncertainty propagation | Not implemented | Poisson + read-noise error propagation per pixel |
| Spatial units | Fractional slit `[0, 1]` | Physical arcseconds |

---

## What remains unimplemented

The following features are explicitly **deferred** to later stages:

* **Optimal extraction** – profile-weighted extraction (Horne 1986).
* **PSF fitting** – the spatial profile is not modelled.
* **Uncertainty propagation** – no variance arrays are computed.
* **Aperture finding / centroiding** – the aperture center and radius must
  be supplied explicitly.
* **Sky modelling** – only a simple per-column median background is
  subtracted; no polynomial fitting or two-sided background fitting.
* **Telluric correction** – belongs to a later stage.
* **Order merging** – each order is extracted independently.
* **Tilt and curvature correction** – the underlying rectified images
  still ignore arc-line tilt and curvature (see
  `docs/ishell_scaffold_constraints.md`).

---

## Usage example

```python
from pyspextool.instruments.ishell.aperture_extraction import (
    ApertureDefinition,
    extract_with_aperture,
)

# Define the aperture: center at 50% of the slit, half-width 20%,
# background from 30%–45% distance on each side.
aperture = ApertureDefinition(
    center_frac=0.5,
    radius_frac=0.2,
    background_inner=0.3,
    background_outer=0.45,
)

# rectified is a RectifiedOrderSet produced by build_rectified_orders().
result = extract_with_aperture(
    rectified,
    aperture,
    method="sum",
    subtract_background=True,
)

for sp in result.spectra:
    print(f"Order {sp.order}: {sp.n_spectral} spectral pixels, "
          f"aperture pixels used = {sp.n_pixels_used}")
```

---

## Public API summary

| Symbol | Type | Description |
|--------|------|-------------|
| `ApertureDefinition` | dataclass | Aperture + background region definition |
| `ExtractedApertureSpectrum` | dataclass | Per-order 1-D extracted spectrum |
| `ExtractedApertureSpectrumSet` | dataclass | Full set of per-order spectra |
| `extract_with_aperture` | function | Main entry point |
