# iSHELL Order Merging / Spectrum Assembly — Developer Guide

**Scope**: J/H/K modes only (J0–J3, H1–H3, K1–K3, Kgas).  
L, Lp, and M modes are out of scope for this module.

---

## Overview

This guide describes how the iSHELL extraction output feeds the
order-merging stage, which parts of the merging pipeline are generic
versus iSHELL-specific, and what remains provisional pending a proper
2-D tilt/curvature solution.

---

## Pipeline: Extraction → Merging

```
preprocess_science_frames()
         │
         ▼
  preprocess_result dict
  {'image', 'variance', 'bitmask',
   'subtraction_mode', 'rectified',
   'tilt_provisional', ...}
         │
         ▼
  extract_spectra(preprocess_result, geometry)
         │
         ▼
  spectra  : list of (4, nwave) ndarrays
             len = norders × n_apertures
             ordering: element i*naps+k → order i, aperture k
             rows: 0=wavelength(µm), 1=intensity, 2=uncertainty, 3=flags
  metadata : dict
             {'orders', 'n_apertures', 'tilt_provisional',
              'aperture_positions_arcsec', 'aperture_radii_arcsec',
              'aperture_signs', 'subtraction_mode', 'rectified',
              'plate_scale_arcsec'}
         │
         ▼
  merge_extracted_orders(spectra, metadata)
         │
         ▼
  merged_spectra : list of (4, nwave_merged) ndarrays
                   len = n_apertures_merged
  merge_metadata : dict
                   {'apertures', 'orders', 'n_orders',
                    'n_apertures_merged', 'tilt_provisional', ...}
```

---

## What Is Generic vs iSHELL-Specific

### Generic (reused unchanged)

All spectral merging arithmetic is provided by
`pyspextool.merge.core.merge_spectra`:

- **Overlap detection** — determines whether two spectra overlap, are
  disjoint, or one is wholly inside the other.
- **Weighted averaging** in the overlap region — inverse-variance-weighted
  combination with linear interpolation onto the anchor wavelength grid.
- **Concatenation** in non-overlap regions.
- **Uncertainty propagation** — via inverse-variance weighting.
- **Bitmask combination** — bitwise OR of the input quality flags.

See `src/pyspextool/merge/core.py` and
`tests/test_merge_spectra.py` for details.

### iSHELL-Specific (`instruments/ishell/merge.py`)

The iSHELL-specific work in `merge_extracted_orders` is:

1. **Input translation** — mapping the flat list `spectra[i*naps+k]`
   into per-aperture anchor/add pairs for the generic engine.
2. **Wavelength sorting** — orders are sorted by median wavelength
   (ascending) before merging so that each successive order is added
   to the right of the running merged spectrum.
3. **Tilt-provisional warning** — the `tilt_provisional` flag from the
   extraction metadata is forwarded to callers with a `RuntimeWarning`
   so they can detect upstream data-quality limitations.
4. **Metadata propagation** — optional extraction keys (`subtraction_mode`,
   `rectified`, `plate_scale_arcsec`, aperture geometry) are forwarded
   unchanged into the merge-metadata output.

---

## Provisional Tilt Model

> **Important**: The current iSHELL rectification step uses a
> **placeholder zero-tilt model**.  Spectral-line curvature is **not**
> corrected at this stage.  All merged spectra produced by
> `merge_extracted_orders` are structurally valid (correct shapes,
> finite wavelength grid, propagated uncertainties and flags) and
> suitable for pipeline development and integration testing, but must
> **not** be treated as science-quality until a proper 2-D tilt solution
> is available.

The tilt flag propagates as follows:

1. `preprocess_science_frames` sets `preprocess_result['tilt_provisional'] = True`
   whenever it applies the zero-tilt placeholder.
2. `extract_spectra` copies the flag into `metadata['tilt_provisional']`
   and emits a `RuntimeWarning`.
3. `merge_extracted_orders` re-emits the same `RuntimeWarning` and copies
   the flag into `merge_metadata['tilt_provisional']`.

Callers that wish to suppress the warning during non-science uses (e.g.
pipeline integration tests) can use:

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    merged, merge_meta = merge_extracted_orders(spectra, meta)
```

---

## Usage Example

### Single A-mode observation

```python
from pyspextool.instruments.ishell.preprocess import preprocess_science_frames
from pyspextool.instruments.ishell.extract import extract_spectra
from pyspextool.instruments.ishell.merge import merge_extracted_orders
from pyspextool.instruments.ishell.calibrations import read_flatinfo, read_wavecalinfo
from pyspextool.instruments.ishell.wavecal import build_geometry_from_wavecalinfo

flat_info = read_flatinfo('K1')
wavecal_info = read_wavecalinfo('K1')
geometry = build_geometry_from_wavecalinfo(wavecal_info, flat_info)

pre = preprocess_science_frames(
    ['ishell_0042.fits'],
    flat_info=flat_info,
    geometry=geometry,
    subtraction_mode='A',
)

spectra, meta = extract_spectra(pre, geometry)
# spectra is a list of (4, nwave) arrays, one per order

merged, merge_meta = merge_extracted_orders(spectra, meta)
# merged[0] is the assembled (4, nwave_merged) spectrum

print(merged[0].shape)          # e.g. (4, 8192)
print(merge_meta['n_orders'])   # e.g. 3 for K1 mode
print(merge_meta['tilt_provisional'])  # True until proper tilt solution exists
```

### A-B nodded observation with aperture selection

```python
import numpy as np

pre_ab = preprocess_science_frames(
    ['ishell_A.fits', 'ishell_B.fits'],
    flat_info=flat_info,
    geometry=geometry,
    subtraction_mode='A-B',
)

slit_arcsec = flat_info.slit_height_arcsec
pos = np.array([slit_arcsec * 0.25, slit_arcsec * 0.75])
signs = np.array([1, -1])

spectra, meta = extract_spectra(
    pre_ab, geometry,
    aperture_positions_arcsec=pos,
    aperture_signs=signs,
)

# Merge only the positive-beam aperture
merged, merge_meta = merge_extracted_orders(spectra, meta, apertures=[1])
```

---

## What Remains Provisional

| Feature | Status |
|---------|--------|
| Zero-tilt rectification placeholder | **Provisional** — spectral curvature not corrected |
| 2-D tilt / curvature solution | **Not yet implemented** (Phase 2) |
| H2RG reference-pixel subtraction | **Not yet implemented** (Phase 2) |
| H2RG polynomial linearity correction | **Not yet implemented** (Phase 2) |
| Wavelength, flux, uncertainty, flags in merged output | **Functional** |
| Overlap weighted averaging | **Functional** (generic engine) |
| Bitmask / flag propagation | **Functional** |

When the Phase 2 tilt correction is complete, the
`tilt_provisional` flag will be set to `False` by `preprocess_science_frames`
and no warnings will be emitted.  No API changes to `merge_extracted_orders`
are anticipated.

---

## Related Files

| Path | Purpose |
|------|---------|
| `src/pyspextool/instruments/ishell/merge.py` | iSHELL-specific merge entry point |
| `src/pyspextool/merge/core.py` | Generic `merge_spectra` algorithm |
| `src/pyspextool/instruments/ishell/extract.py` | Extraction stage (upstream) |
| `src/pyspextool/instruments/ishell/preprocess.py` | Preprocessing (upstream) |
| `tests/test_ishell_merge.py` | Merge module tests |
| `tests/test_merge_spectra.py` | Generic merge algorithm tests |
| `docs/ishell_extraction_guide.md` | Extraction stage documentation |
| `docs/ishell_preprocessing_guide.md` | Preprocessing documentation |
