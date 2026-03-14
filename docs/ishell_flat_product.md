# iSHELL Flat-Field Generation — Developer Notes

## Overview

This document describes the iSHELL flat-generation path added in this PR,
what it reuses from the existing pySpextool infrastructure, and what
remains provisional pending full wavecal/rectification work.

Scope is restricted to **J/H/K modes only** (J0, J1, J2, J3, H1, H2, H3,
K1, K2, K3, Kgas).  L, Lp, and M modes are not supported.

---

## Flat Product Structure

The iSHELL flat field output is a **standard pySpextool flat FITS file**
with five extensions written by `pyspextool.extract.flat.write_flat()`:

| Extension | Contents |
|-----------|----------|
| 0 (PRIMARY) | Header-only HDU; carries all metadata |
| 1 (IMAGE) | Normalised flat image, float32, shape (2048, 2048) |
| 2 (IMAGE) | Variance of the flat, float32, shape (2048, 2048) |
| 3 (IMAGE) | Bitmask (non-linear pixels), int8, shape (2048, 2048) |
| 4 (IMAGE) | Order mask (integer order numbers), int8, shape (2048, 2048) |

**All images are in the post-rotation detector frame** (rotation code stored
in `ROTATION`).  The rotation is applied by `idl_rotate()` during
`read_fits()` and un-applied by `idl_unrotate()` inside `write_flat()` so
that the stored arrays are in the raw detector orientation.

### Primary HDU keywords

Key metadata keywords written by `write_flat()`:

| Keyword | Description |
|---------|-------------|
| `MODE` | iSHELL observing sub-mode (e.g. `K1`) |
| `NORDERS` | Number of echelle orders |
| `ORDERS` | Comma-separated list of echelle order numbers |
| `PLTSCALE` | Plate scale (arcsec/pixel) |
| `SLTH_PIX` | Slit height (pixels) |
| `SLTH_ARC` | Slit height (arcsec) |
| `SLTW_PIX` | Slit width (pixels) |
| `SLTW_ARC` | Slit width (arcsec) |
| `RP` | Nominal spectral resolving power |
| `ROTATION` | IDL-rotate convention code (0–7) |
| `YBUFFER` | Edge buffer (pixels) used during order tracing |
| `OR{NNN}_XR` | Extraction column range for order NNN |
| `OR{NNN}_B{k}` | Bottom edge polynomial coefficient k for order NNN |
| `OR{NNN}_T{k}` | Top edge polynomial coefficient k for order NNN |
| `EDGEDEG` | Polynomial degree used for order edge fitting |

---

## What is Reused from Existing pySpextool Logic

### Generic pipeline (`pyspextool.extract.make_flat`)

The function `pyspextool.extract.make_flat.make_flat()` is the single
entry point for flat-field generation and is **not iSHELL-specific**.  It
discovers and calls the correct instrument module at runtime via
`importlib.import_module('pyspextool.instruments.ishell.ishell')`.

The following pipeline steps are fully generic and unchanged:

* **Image scaling** — `utils.math.scale_data_stack()` scales all frames to
  a common median intensity before combining.
* **Image medianing** — `utils.math.median_data_stack()` combines the scaled
  frames; variance = (1.4826×MAD)² / N.
* **Order location** — `extract.flat.locate_orders()` finds order boundaries
  using Sobel edge detection on the median flat.
* **Order normalisation** — `extract.flat.normalize_flat()` divides each
  order by a smooth 2-D surface fit (fiterpolate).
* **Order masking** — `extract.images.make_ordermask()` creates the integer
  order mask.
* **Flat FITS writing** — `extract.flat.write_flat()` writes the five-
  extension output file.

### iSHELL instrument plugin (`pyspextool.instruments.ishell.ishell`)

`make_flat` calls the iSHELL instrument plugin for:

* **`read_fits()`** — loads each raw iSHELL 3-extension MEF file, flags
  non-linear pixels using pedestal+signal sums (ext 1 + ext 2), divides by
  total exposure time, and applies the requested IDL-rotate.
* **`get_header()`** — maps raw iSHELL FITS keywords to the standard
  pySpextool names (`TCS_AM → AM`, `PASSBAND → MODE`, `SLIT → SLIT`, etc.).

### iSHELL packaged calibration resources

The per-mode order-trace parameters are read from the packaged
`*_flatinfo.fits` files by the **generic** `extract.flat.read_flatcal_file()`
function.  The iSHELL flatinfo files reside in the `data/` subdirectory
of the instrument package (`instruments/ishell/data/<MODE>_flatinfo.fits`).

For each supported mode the flatinfo FITS header encodes:

* Order numbers and per-order column ranges
* Initial guess positions for the order tracer
* Pre-fit polynomial edge coefficients (used as seeds for `locate_orders`)
* Plate scale, slit height, edge-fit degree, step size, and normalisation grid

The `pyspextool.instruments.ishell.calibrations.read_flatinfo()` function
provides a **typed, validated** view of the same data via the `FlatInfo`
dataclass; `make_flat` uses `read_flatcal_file` for compatibility with the
existing pipeline contract.

---

## Changes Made to Generic Code

Two minimal changes were made to `pyspextool.extract.make_flat`:

### 1. Flatinfo path fallback (`data/` subdirectory)

SpeX and uSpeX store per-mode `*_flatinfo.fits` files at the root of their
instrument package directory.  iSHELL stores them in a `data/` subdirectory
(alongside linearity cubes, bad-pixel masks, arc-line lists, etc.).

`make_flat` now falls back to `<instrument_path>/data/<MODE>_flatinfo.fits`
when the file is not found at `<instrument_path>/<MODE>_flatinfo.fits`:

```python
modefile = join(setup.state['instrument_path'], mode + '_flatinfo.fits')
if not os.path.isfile(modefile):
    modefile = join(setup.state['instrument_path'], 'data',
                    mode + '_flatinfo.fits')
```

This is fully backward compatible: SpeX and uSpeX files are found at the
root on the first check; only iSHELL (and any future instrument that uses
the `data/` layout) needs the fallback.

### 2. Robust SLIT keyword parsing

The old code extracted slit width with a fixed 3-character slice:

```python
slitw_arc = float(average_header['SLIT'][0][0:3])  # SpeX: '0.3x15'→0.3
```

This fails for iSHELL slit keywords like `'0.375_K1'` (gives `0.3` instead
of `0.375`) or `'0.75_K1'` (gives `0.7` instead of `0.75`).

The replacement uses a regex to extract the leading numeric field:

```python
m = re.match(r'^([0-9]+(?:\.[0-9]+)?)', str(slit_str))
slitw_arc = float(m.group(1)) if m else float(slit_str[0:3])
```

This is correct for both the SpeX `'0.3x15'` format and the iSHELL
`'0.375_K1'` format.

---

## Provisional / Known Limitations

The following aspects of the iSHELL flat are **provisional** and will require
further work once full wavecal/rectification infrastructure is in place:

### Order rectification (Phase 2)

iSHELL echelle orders are **tilted** with respect to detector columns.
`locate_orders()` and `normalize_flat()` operate on the un-rectified flat
image, so the order edges and the normalised flat are expressed in the
tilted/curved detector frame, not a rectified wavelength-vs-spatial frame.

The `_rectify_orders()` stub in `ishell.py` is a Phase 2 task.  Until
rectification is implemented, the flat can be used for:

* Pixel-level flat-field division to remove detector response variations
* Order masking (the order mask is valid in the un-rectified frame)

It **cannot** yet be used for:

* Wavelength-calibrated extraction that requires a rectified spatial profile

### Linearity correction (Phase 2)

The polynomial H2RG linearity correction (`_correct_ishell_linearity()`) is
not yet applied.  Raw images are flag-only for pixels above `LINCORMAX =
30000 DN` but are not numerically corrected.  The `linearity_correction`
parameter accepted by `read_fits()` and `make_flat()` is a no-op for iSHELL
until Phase 2.

### Reference-pixel bias subtraction (Phase 2)

H2RG reference-pixel bias subtraction (`_subtract_reference_pixels()`) is
not yet applied.  The 4-pixel border reference regions are included in the
raw science image.

---

## Test Coverage

The test suite in `tests/test_ishell_flat.py` covers:

| Test class | What it verifies |
|------------|-----------------|
| `TestReadFlatcalFileIshell` | `read_flatcal_file()` returns a valid dict for all 11 J/H/K modes with all required keys |
| `TestFlatinfoOrderMetadata` | `xranges`, `edgecoeffs`, and `guesspos` shapes agree with the order list; xranges lie within detector bounds |
| `TestSlitParsing` | Regex-based slit extractor gives correct widths for both SpeX and iSHELL formats |
| `TestFlatinfoPathFallback` | `J0_flatinfo.fits` is absent at the instrument root but present in `data/`; `read_flatcal_file` succeeds via the data path |
| `TestMakeFlatIshellEndToEnd` | `make_flat` produces a 5-extension FITS with correct shape, MODE, ORDERS, and SLTW_ARC for representative J, H, K modes using synthetic MEF files |
| `TestMakeFlatIshellFailureModes` | Appropriate errors for missing flatinfo, wrong NINT, missing ITIME, and zero ITIME |

No real iSHELL science data is required by any test.  Synthetic 2048×2048
MEF FITS files are created in temporary directories.  `locate_orders()` is
mocked with the pre-packaged edge coefficients from the relevant flatinfo so
that tests run quickly without needing a realistic illuminated flat.
