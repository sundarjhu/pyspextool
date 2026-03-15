# iSHELL Spectral Extraction Developer Guide

**Scope:** J/H/K modes (J0–J3, H1–H3, K1–K3, Kgas).  L, Lp, and M are out
of scope.

---

## Overview

The iSHELL extraction path connects the preprocessing pipeline output to the
generic pySpextool extraction engine.  Extraction lives in

```
src/pyspextool/instruments/ishell/extract.py
```

and consists of two public functions:

| Function | Purpose |
|---|---|
| `build_extraction_arrays` | Convert `OrderGeometrySet` → `ordermask`, `wavecal`, `spatcal` 2-D arrays |
| `extract_spectra` | Full pipeline: preprocess output → extracted spectra |

---

## How preprocessing output feeds extraction

`preprocess_science_frames` (in `preprocess.py`) returns a dictionary:

```python
{
    'image'           : ndarray (nrows, ncols)   # flat-fielded science, DN/s
    'variance'        : ndarray (nrows, ncols)   # propagated variance
    'bitmask'         : ndarray uint8 (nrows, ncols)
    'hdrinfo'         : dict
    'subtraction_mode': str          # 'A-B' or 'A'
    'flat_applied'    : bool
    'rectified'       : bool
    'tilt_provisional': bool         # True when provisional zero-tilt used
}
```

`extract_spectra` accepts this dictionary as its first argument.  It reads
`image`, `variance`, and `bitmask` and discards all other keys (they are
propagated as-is through `metadata`).

The caller must also supply an `OrderGeometrySet` with
`geometry.has_wavelength_solution() == True`.  This geometry object may be
the same one passed to `preprocess_science_frames` (if rectification was
done) or a newly constructed one.

**Typical sequence:**

```python
from pyspextool.instruments.ishell.calibrations import read_flatinfo, read_wavecalinfo
from pyspextool.instruments.ishell.wavecal import build_geometry_from_wavecalinfo
from pyspextool.instruments.ishell.preprocess import preprocess_science_frames
from pyspextool.instruments.ishell.extract import extract_spectra

fi   = read_flatinfo('K1')
wci  = read_wavecalinfo('K1')
geom = build_geometry_from_wavecalinfo(wci, fi)

pre = preprocess_science_frames(
    files=['/data/ishell_A.fits', '/data/ishell_B.fits'],
    flat_info=fi,
    geometry=geom,
    subtraction_mode='A-B',
)

spectra, meta = extract_spectra(pre, geom)
```

---

## What is generic vs iSHELL-specific

### iSHELL-specific (in `instruments/ishell/extract.py`)

1. **`build_extraction_arrays`** — translates iSHELL `OrderGeometrySet`
   geometry into the three 2-D arrays (`ordermask`, `wavecal`, `spatcal`)
   that the generic extractor expects.

   - `ordermask[row, col]` = echelle order number (0 = inter-order)
   - `wavecal[row, col]` = wavelength in µm from `wave_coeffs` polynomial,
     constant along columns within each order
   - `spatcal[row, col]` = spatial position in arcsec; convention is 0 at
     the bottom edge and `slit_height_arcsec` at the top edge (matching
     `simulate_wavecal_1dxd`)

2. **Bitmask conversion** — the preprocessing `bitmask` (multi-bit uint8) is
   collapsed to a binary `linmax_bitmask` (0 = clean, 1 = any flag set) that
   `extract_1dxd` accepts.

3. **NaN cleanup** — NaN pixels in the preprocessed image and variance are
   replaced with 0 and marked in `linmax_bitmask` so that sum extraction
   does not propagate NaN through `np.sum`.

4. **Trace coefficients** — constant (degree-0) polynomials placing each
   aperture at a fixed arcsec position in the slit.  The default is a single
   centered aperture at `slit_height_arcsec / 2`.

### Generic (in `extract/extraction.py`)

All arithmetic after the setup step is performed by `extract_1dxd`:

- aperture masking (partial-pixel weight maps)
- background fitting and subtraction
- sum extraction and variance propagation
- optimal extraction (when `optimal_info` is passed)
- bad-pixel replacement (when `badpixel_info` is passed)
- output packaging as `(4, nwave)` arrays

`extract_1dxd` is instrument-agnostic; it only needs `ordermask`, `wavecal`,
`spatcal`, and the aperture configuration described above.

---

## Aperture configuration

### Single-aperture (A mode)

```python
spectra, meta = extract_spectra(pre, geom,
                                aperture_radii_arcsec=0.5)
# → 1 spectrum per order
```

The default aperture is centered at the geometric slit midpoint.

### Two-aperture A-B nod pair

For a standard ABBA nod where the A beam falls at `~1/4` and the B beam
at `~3/4` of the slit height:

```python
slit_arcsec = 5.0   # from flat_info.slit_height_arcsec
spectra, meta = extract_spectra(
    pre, geom,
    aperture_positions_arcsec=np.array([slit_arcsec * 0.25,
                                        slit_arcsec * 0.75]),
    aperture_signs=np.array([1, -1]),
    aperture_radii_arcsec=0.5,
)
# → 2 spectra per order (positive A beam + negative B beam)
```

### Aperture radius shapes

`aperture_radii_arcsec` accepts:

| Shape | Meaning |
|---|---|
| scalar | same radius for all orders and apertures |
| 1-D, length `naps` | one radius per aperture, same for all orders |
| 1-D, length `norders` | one radius per order, same for all apertures |
| 2-D `(norders, naps)` | full specification |

---

## Output format

`extract_spectra` returns `(spectra, metadata)`.

### `spectra`

A list of `(4, nwave)` arrays, one per `(order, aperture)` pair, ordered
aperture-minor (i.e. element `i * naps + k` is order index `i`, aperture `k`):

| Row | Content |
|---|---|
| 0 | wavelength (µm) |
| 1 | intensity (DN s⁻¹) |
| 2 | uncertainty (DN s⁻¹) |
| 3 | spectral quality flag (0 = clean; non-zero if any flagged pixel in aperture) |

### `metadata`

```python
{
    'orders'                  : list[int],      # e.g. [254, 255, ..., 270]
    'n_apertures'             : int,
    'aperture_positions_arcsec': ndarray,
    'aperture_radii_arcsec'   : ndarray (norders, naps),
    'aperture_signs'          : ndarray,
    'subtraction_mode'        : str,
    'rectified'               : bool,
    'tilt_provisional'        : bool,
    'plate_scale_arcsec'      : float,
}
```

---

## What remains provisional

### Spectral-line tilt (zero-tilt placeholder)

The current `_rectify_orders` implementation in `ishell.py` uses
`tilt_coeffs = [0.0]` for every order — a placeholder zero-tilt model.
The resampling is structurally valid (NaN masking, edge geometry applied)
but **does not correct spectral-line curvature**.

When preprocessing was performed with `geometry` supplied,
`preprocess_result['tilt_provisional']` will be `True`.  `extract_spectra`
propagates this flag into `meta['tilt_provisional']` and emits a
`RuntimeWarning`.

**Impact on extraction:** Sum extraction integrates along a spatial column at
each wavelength step.  With zero tilt, each column is treated as a single
wavelength element, which is the standard 1DXD assumption.  When the true
tilt is non-zero, lines will be slightly smeared because adjacent rows at the
same column sample slightly different wavelengths.  For iSHELL J/H/K modes
the tilt is small but measurable; science-quality spectra require the true
tilt solution.

### What is needed to remove this limitation

1. Measure tilt slopes from 2-D ThAr arc-lamp frames.
2. Populate `OrderGeometry.tilt_coeffs` with the measured slopes.
3. Re-run `build_rectification_maps` (which already uses `eval_tilt`)
   — no changes to the extraction code are required.

Until then, callers should clearly note `meta['tilt_provisional']` in any
publication or data product header.

---

## Testing

Tests for the extraction module live in `tests/test_ishell_extract.py`.  All
tests use synthetic FITS fixtures and do not require real iSHELL data.

Test categories:

| Class | What is covered |
|---|---|
| `TestModuleAPI` | importability of public functions |
| `TestBuildExtractionArrays` | ordermask/wavecal/spatcal correctness |
| `TestExtractSpectraOutput` | output shape and metadata keys |
| `TestDimensionalConsistency` | J/H/K modes, multiple orders/apertures |
| `TestVariancePropagation` | uncertainty is non-negative; scales with variance |
| `TestBitmaskPropagation` | linearity, flat-zero, and NaN flags reach output |
| `TestApertureParameters` | scalar/array/2-D radius; custom positions; signs |
| `TestFailureCases` | malformed inputs raise the correct exceptions |
| `TestIntegrationWithPreprocess` | full A-mode/A-B-mode/rectified pipelines |

Run the tests with:

```bash
pytest tests/test_ishell_extract.py -v
```

---

## See also

- `docs/ishell_preprocessing_guide.md` — preprocessing pipeline details
- `docs/ishell_wavecal_design_note.md` — wavecal and tilt/curvature design
- `docs/ishell_geometry_design_note.md` — `OrderGeometry` / `OrderGeometrySet` design
- `src/pyspextool/extract/extraction.py` — generic `extract_1dxd` implementation
