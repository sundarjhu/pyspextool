# iSHELL Science-Frame Preprocessing — Developer Guide

**Scope:** NASA IRTF iSHELL spectrograph, J / H / K modes only  
(L, Lp, and M modes are explicitly out of scope.)

---

## 1. Overview

The iSHELL science-frame preprocessing pipeline is implemented in
`src/pyspextool/instruments/ishell/preprocess.py`.  Its single public entry
point is `preprocess_science_frames()`.

The function accepts raw iSHELL MEF FITS files together with pre-loaded
calibration objects and returns a flat-fielded, (optionally) pair-subtracted,
and (optionally) order-rectified science image ready for the downstream
`extract_1dxd()` extraction step.

### Relationship to the generic `load_image()` pipeline

`pyspextool.extract.load_image.load_image()` provides a stateful,
high-level pipeline for SpeX and uSpeX that dispatches to the instrument
module, loads flat and wavecal calibrations from disk, applies flat-fielding,
and rectifies orders using `extract.images.rectify_order()`.

`preprocess_science_frames()` is the iSHELL-specific counterpart.  It is
designed to be **stateless** (no global `setup.state` mutations) and to
accept fully-constructed calibration objects rather than file paths, so that
each stage is independently testable.

---

## 2. Preprocessing Sequence

```
files (raw MEF FITS)
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Stage 1: Input validation                                        │
 │  - subtraction_mode in {'A-B', 'A'}                             │
 │  - file count matches subtraction_mode                          │
 │  - flat_info.image is non-None and non-zero                     │
 │  - geometry (if supplied) has a wavelength solution             │
 │  - geometry.mode matches flat_info.mode                         │
 └──────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Stage 2: Raw FITS ingestion — load_data() per frame              │
 │  - Read 3-extension MEF (ext 0 = signal diff, 1 = ped sum,      │
 │    2 = sig sum; iSHELL Spextool Manual §2.3)                    │
 │  - Flag non-linear pixels: ped+sig > LINCORMAX (30000 DN)       │
 │  - Normalise to DN s⁻¹ (divide by ITIME × CO_ADDS)             │
 │  - Estimate variance (shot noise + read noise)                  │
 │                                                                  │
 │  NOTE: H2RG polynomial linearity correction and reference-pixel │
 │  bias subtraction are NOT yet applied (Phase 2 tasks).          │
 └──────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Stage 3: Pair subtraction ('A-B') or passthrough ('A')          │
 │  - A-B: img = img_A - img_B; var = var_A + var_B                │
 │         bitmask = bitwise OR of both frames' masks              │
 │  - A:   img = img_A (unchanged); var, bitmask unchanged         │
 └──────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Stage 4: Flat-field division                                     │
 │  - Divide by flat_info.image (pixel-by-pixel)                   │
 │  - Zero or NaN flat pixels → science pixel set to NaN,          │
 │    bitmask bit 4 set                                            │
 │  - var = var / flat²                                            │
 └──────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Stage 5: Order rectification (_rectify_orders)                   │
 │  - Skipped if geometry is None                                  │
 │  - Uses scipy.ndimage.map_coordinates (bilinear) to resample    │
 │    each echelle order                                           │
 │  - Pixels outside all order footprints → NaN                   │
 │  - Variance is rectified through the same mapping               │
 │                                                                  │
 │  ⚠ PROVISIONAL: The tilt model uses zero-tilt placeholder       │
 │  coefficients.  No spectral-line tilt is measured or removed.   │
 └──────────────────────────────────────────────────────────────────┘
        │
        ▼
 dict: image, variance, bitmask, hdrinfo,
       subtraction_mode, flat_applied, rectified, tilt_provisional
```

---

## 3. What Is Scientifically Grounded

### Raw MEF ingestion
The 3-extension iSHELL MEF format is confirmed by the iSHELL Spextool Manual
§2.3:

| Extension | Contents |
|-----------|----------|
| 0 | Signal difference S = Σ pedestal reads − Σ signal reads |
| 1 | Pedestal sum Σ p |
| 2 | Signal sum Σ s |

Non-linearity flagging uses `ped_sum + sig_sum > LINCORMAX = 30000 DN`
(Manual Table 4, LINCRMAX example).

### A-B pair subtraction
Standard sky subtraction for iSHELL J/H/K nodded science observations.
Variance and bitmask propagation follow standard CCD-equivalent rules
for the H2RG readout.

### Flat-field division
Corrects illumination gradients and pixel-response non-uniformity (PRNU).
Pixels where the flat is zero or NaN are flagged in the output bitmask
(bit 4) and set to NaN in the science image so that downstream extraction
can identify them.

### Variance propagation
All stages propagate a variance estimate:

1. **Load**: `var = |S| / (gain × T²) + 2 × (RON/gain)² / T²`
   where `T = ITIME × CO_ADDS`, `gain = 1.8 e⁻/DN`, `RON = 8 e⁻`.
2. **A-B subtraction**: `var_AB = var_A + var_B`.
3. **Flat division**: `var_flat = var / flat²`.

---

## 4. What Remains Provisional

### 4.1 H2RG polynomial linearity correction (Phase 2)

The `_correct_ishell_linearity()` function in `ishell.py` is scaffolded
but raises `NotImplementedError`.  The polynomial correction algorithm for
the H2RG detector differs from the Vacca et al. (2004) `slowcnts` algorithm
used for SpeX.  Until this is implemented, the raw signal difference is
used directly without linearisation.

**Impact:** pixels near the saturation boundary (< 30000 DN) may be slightly
non-linear, introducing a small systematic error in photon counts.  The
non-linearity flag correctly identifies pixels above 30000 DN.

### 4.2 Reference-pixel bias subtraction (Phase 2)

The `_subtract_reference_pixels()` function in `ishell.py` is scaffolded
but raises `NotImplementedError`.  The H2RG detector provides 4-pixel-wide
reference regions on all four edges that track bias and 1/f noise.  Until
this is implemented, no bias correction is applied.

**Impact:** low-frequency bias drifts may remain in the science image.

### 4.3 Spectral-line tilt model (provisional zero)

iSHELL echelle orders are tilted with respect to the detector columns.
The current `_rectify_orders()` implementation uses `tilt_coeffs = [0.0]`
for every order — a **placeholder zero** that does not correct any
spectral-line tilt.

This means that after rectification:
- The output image is **structurally valid**: NaN masking and order edge
  geometry are applied correctly.
- The output image is **not scientifically rectified**: spectral-line tilt
  is not removed.  Features along detector rows will not align with constant
  wavelength after this step.

**Why the placeholder is used:** Measuring tilt requires knowing how a
spectral-line centroid shifts along the spatial (cross-dispersion) direction.
That information requires a 2-D arc-lamp image.  The packaged
`*_wavecalinfo.fits` files contain only the 1-D centerline spectrum; they
do not encode spatial variation.

**What is needed for real tilt correction:**
1. Real 2-D ThAr arc-lamp frames.
2. Gaussian centroid measurement at multiple spatial rows per arc line.
3. A polynomial fit of centroid position vs. spatial row for each arc line.
4. Populating `OrderGeometry.tilt_coeffs` with the measured slope polynomial.

Until this is done, `tilt_provisional = True` in every result dict and a
`RuntimeWarning` is emitted when rectification is applied.

### 4.4 Unsupported subtraction modes

`'A-Sky'` and `'A-Dark'` subtraction modes are listed in
`SUPPORTED_SUBTRACTION_MODES` only as `('A-B', 'A')`.  These require
separate sky or dark FITS files, which are not part of the current iSHELL
data model.

---

## 5. Usage Example

```python
from pyspextool.instruments.ishell.calibrations import (
    read_flatinfo, read_wavecalinfo)
from pyspextool.instruments.ishell.wavecal import (
    build_geometry_from_wavecalinfo)
from pyspextool.instruments.ishell.preprocess import (
    preprocess_science_frames)

# --- Load calibrations ---------------------------------------------------
mode = 'K1'
flat_info = read_flatinfo(mode)
wavecal_info = read_wavecalinfo(mode)
geometry = build_geometry_from_wavecalinfo(wavecal_info, flat_info)
# geometry.has_wavelength_solution() is True at this point.
# geometry tilt_coeffs are zero (provisional placeholder).

# --- Preprocess A-B nod pair ---------------------------------------------
result = preprocess_science_frames(
    files=['/data/ishell_0100.a.fits', '/data/ishell_0101.b.fits'],
    flat_info=flat_info,
    geometry=geometry,
    subtraction_mode='A-B',
    verbose=True,
)

# --- Inspect results -----------------------------------------------------
print(result['image'].shape)          # (2048, 2048)
print(result['rectified'])            # True
print(result['tilt_provisional'])     # True  ← always True until Phase 2
print(result['hdrinfo']['MODE'])      # ['K1', ' Instrument Mode']

# --- Single-frame mode (no sky subtraction) ------------------------------
result_a = preprocess_science_frames(
    files=['/data/ishell_0102.a.fits'],
    flat_info=flat_info,
    geometry=geometry,        # pass None to skip rectification
    subtraction_mode='A',
)
```

---

## 6. Bitmask Flag Register

| Bit | Meaning | Source |
|-----|---------|--------|
| `linearity_info['bit']` (default 0) | Non-linear pixel (ped+sig > 30000 DN) | Stage 2, any input frame |
| 4 | Flat pixel is zero or NaN; unsafe division | Stage 4 |

Bits are combined with bitwise OR across frames in A-B mode.

---

## 7. Testing

Tests are in `tests/test_ishell_preprocess.py` (65 tests as of this writing).
All tests use synthetic MEF FITS files created in `tmp_path` fixtures; no real
iSHELL data is required.

```bash
# Run preprocessing tests only
pytest tests/test_ishell_preprocess.py -v

# Run all iSHELL tests
pytest tests/test_ishell*.py -v
```

Test categories:

| Class | Description |
|-------|-------------|
| `TestModuleAPI` | Module-level imports and constants |
| `TestAMode` | Single A-beam mode: shape, values, flat division |
| `TestABMode` | A-B pair subtraction: values, variance, bitmask union |
| `TestRectification` | Rectification smoke tests (shape, NaN structure, warning) |
| `TestFlatFieldEdgeCases` | Zero flat pixels → NaN, bitmask bit 4 |
| `TestRepresentativeBands` | Parametrised J0/H1/K1 mode checks |
| `TestFailureCases` | Error handling for bad inputs |
| `TestLinearityInfo` | Custom vs. default linearity threshold |

Rectification tests are **smoke tests** — they verify structural properties
(shape, NaN masking, warning emission) rather than exact pixel values,
because the provisional zero-tilt model does not produce scientifically
meaningful rectification.

---

## 8. Module Location and Public API

| Symbol | Location |
|--------|----------|
| `preprocess_science_frames` | `pyspextool.instruments.ishell.preprocess` |
| `SUPPORTED_SUBTRACTION_MODES` | `pyspextool.instruments.ishell.preprocess` |

---

## 9. Phase Roadmap

| Phase | Status | Notes |
|-------|--------|-------|
| MEF ingestion (`read_fits`, `load_data`) | ✅ Complete | Phase 1 |
| A-B pair subtraction | ✅ Complete | Phase 1 |
| Flat-field division | ✅ Complete | Phase 1 |
| Non-linearity flagging | ✅ Complete (flagging only) | Phase 1 |
| Structural order rectification (zero-tilt) | ✅ Complete | Phase 1 / provisional |
| H2RG polynomial linearity correction | ❌ Not yet applied | Phase 2 |
| Reference-pixel bias subtraction | ❌ Not yet applied | Phase 2 |
| 2-D tilt measurement from arc frames | ❌ Not yet implemented | Future PR |
| Full 2DXD global wavelength solution | ❌ Not yet implemented | Future PR |
