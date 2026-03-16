# iSHELL Provisional Calibration FITS Writer (Stage 9)

This document describes the **Stage 9** module of the iSHELL 2DXD
reduction scaffold:
`src/pyspextool/instruments/ishell/calibration_fits.py`.

---

## What Stage 9 does

Stage 9 serialises the provisional wavelength-calibration (wavecal) and
spatial-calibration (spatcal) products produced by Stage 8 to structured
FITS calibration files.

The module exposes three public functions:

| Function | Purpose |
|---|---|
| `write_wavecal_fits(cal, path)` | Write wavecal products to a FITS file |
| `write_spatcal_fits(cal, path)` | Write spatcal products to a FITS file |
| `write_calibration_fits(cal, dir)` | Write both files into an output directory |

---

## FITS structure

### Wavecal file (`<mode>_wavecal.fits`)

```
HDU 0   Primary HDU  (no data)
HDU 1   ImageHDU     ORDER_<order_number>  (first order)
HDU 2   ImageHDU     ORDER_<order_number>  (second order)
…
```

**Primary HDU header keywords**

| Keyword   | Value                     | Description                             |
|-----------|---------------------------|-----------------------------------------|
| `PRODTYPE` | `WAVECAL_PROVISIONAL`    | Identifies this as a provisional product |
| `MODE`    | e.g. `H1`                 | iSHELL observing mode                   |
| `NORDERS` | integer                   | Number of echelle orders in the file    |
| `DATE`    | ISO-8601 UTC timestamp    | File creation time                      |

**Per-order extension header keywords**

| Keyword   | Description                                    |
|-----------|------------------------------------------------|
| `ORDER`   | Echelle order number                           |
| `PRODTYPE` | `WAVECAL_PROVISIONAL`                         |
| `NSPECTRA` | Length of the wavelength axis                 |
| `NSPATIAL` | Length of the spatial axis                    |
| `WAVMIN`  | Minimum wavelength in µm                       |
| `WAVMAX`  | Maximum wavelength in µm                       |
| `MODE`    | iSHELL observing mode                          |

**Per-order extension data**

Each extension contains a 2-D `float32` array of shape `(2, n_spectral)`:

```
row 0  →  wavelength axis (µm), length n_spectral
row 1  →  spatial_frac axis, NaN-padded to length n_spectral
           if n_spatial < n_spectral
```

Packing both axes into a single 2-D image keeps the per-order data
self-contained without requiring a binary-table extension.

---

### Spatcal file (`<mode>_spatcal.fits`)

```
HDU 0   Primary HDU  (no data)
HDU 1   ImageHDU     ORDER_<order_number>  (first order)
HDU 2   ImageHDU     ORDER_<order_number>  (second order)
…
```

**Primary HDU header keywords**

| Keyword   | Value                     | Description                             |
|-----------|---------------------------|-----------------------------------------|
| `PRODTYPE` | `SPATCAL_PROVISIONAL`    | Identifies this as a provisional product |
| `MODE`    | e.g. `H1`                 | iSHELL observing mode                   |
| `NORDERS` | integer                   | Number of echelle orders in the file    |
| `DATE`    | ISO-8601 UTC timestamp    | File creation time                      |

**Per-order extension header keywords**

| Keyword   | Description                                        |
|-----------|----------------------------------------------------|
| `ORDER`   | Echelle order number                               |
| `PRODTYPE` | `SPATCAL_PROVISIONAL`                             |
| `NSPATIAL` | Length of the spatial axis                        |
| `SFMIN`   | Minimum spatial fraction (should be 0.0)          |
| `SFMAX`   | Maximum spatial fraction (should be 1.0)          |
| `MODE`    | iSHELL observing mode                              |

**Per-order extension data**

Each extension contains a 2-D `float32` array of shape `(2, n_spatial)`:

```
row 0  →  spatial_frac axis [0, 1], length n_spatial
row 1  →  detector_rows mapping (provisional), length n_spatial
```

---

## Relationship to provisional calibration products

Stage 8 (`calibration_products.py`) extracts coordinate axes from the
rectified order images (Stage 7) and wraps them in typed containers:

```
Stage 7 RectifiedOrder.wavelength_um  →  WaveCalProduct.wavelength_um
Stage 7 RectifiedOrder.spatial_frac   →  WaveCalProduct.spatial_frac
Stage 7 RectifiedOrder.spatial_frac   →  SpatCalProduct.spatial_frac
                       (proxy)         →  SpatCalProduct.detector_rows
```

Stage 9 simply serialises those containers to FITS.  No new information
is introduced; the FITS arrays are identical to the in-memory arrays
held in the `CalibrationProductSet`.

---

## Differences from final Spextool calibration files

The provisional scaffold FITS files differ from the final Spextool
calibration files in several important ways:

| Property | Provisional scaffold | Final Spextool |
|---|---|---|
| PRODTYPE keyword | `WAVECAL_PROVISIONAL` / `SPATCAL_PROVISIONAL` | instrument-specific |
| WCS keywords | Not present | Present |
| Physical spatial axis | Fractional [0,1] only | Arcseconds |
| Wavelength refinement | None (provisional polynomial) | Sigma-clipped, line-list refined |
| Tilt / curvature | Not corrected | Corrected |
| Flux calibration | Not present | Present |
| Format compatibility | Scaffold only | Full Spextool pipeline |

---

## Limitations of the scaffold output

These limitations are intentional scaffold simplifications documented in
`docs/ishell_scaffold_constraints.md`:

1. **Wavelength axis is provisional** – derived from a polynomial
   surface that has not been sigma-clipped or refined against a
   science-quality line list.

2. **Spatial axis is fractional** – physical spatial calibration in
   arcseconds is not yet available; the `spatial_frac` axis is in
   `[0, 1]` only.

3. **Detector-row mapping is a proxy** – `detector_rows` in the spatcal
   product equals `spatial_frac` at this stage; a later stage will
   replace this with physical row coordinates from the rectification
   index arrays.

4. **No WCS keywords** – no FITS WCS keywords are written; downstream
   code must read the axis arrays directly.

5. **No tilt or curvature correction** – the rectification mapping is a
   geometric approximation; tilt and curvature corrections are deferred
   to a later stage.

6. **Not IDL-coefficient compatible** – the polynomial coefficients used
   to derive the wavelength axis are not compatible with the IDL
   Spextool format.

---

## Usage example

```python
from pyspextool.instruments.ishell.calibration_fits import write_calibration_fits

# cal_products is a CalibrationProductSet from Stage 8
wavecal_path, spatcal_path = write_calibration_fits(cal_products, "/output/dir")

print(wavecal_path)   # /output/dir/H1_wavecal.fits
print(spatcal_path)   # /output/dir/H1_spatcal.fits
```

Reading the files back:

```python
import astropy.io.fits as fits
import numpy as np

with fits.open("H1_wavecal.fits") as hdul:
    print(hdul.info())
    for hdu in hdul[1:]:
        order  = hdu.header["ORDER"]
        wav    = hdu.data[0]     # wavelength axis (µm)
        sf_row = hdu.data[1]     # spatial_frac axis (NaN-padded)
        print(f"Order {order}: wav range [{wav.min():.4f}, {wav.max():.4f}] µm")
```
