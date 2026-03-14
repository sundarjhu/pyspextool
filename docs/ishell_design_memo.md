# iSHELL Backend Integration Design Memo

**Status:** Draft  
**Scope:** NASA IRTF iSHELL spectrograph, J / H / K modes only  
**Restriction:** L, Lp, and M modes are explicitly out of scope.

---

## 1. Architecture Summary

### Existing Instrument Abstraction

pySpextool uses a **duck-typed plugin architecture** for instruments.  There
is no abstract base class; instead, each instrument supplies a Python module
and a configuration file that satisfy an implicit interface:

| Artifact | Role |
|---|---|
| `instruments/<name>/<name>.dat` | Static configuration: detector size, linearity threshold, FITS file suffix, bad-pixel-mask name, extra FITS keywords to retain |
| `instruments/<name>/<name>.py` | Three required callables: `read_fits()`, `get_header()`, `load_data()` |
| `instruments/<name>/<name>_bdpxmk.fits` | 2-D bad-pixel mask (0 = good, 1 = bad) |
| `instruments/<name>/telluric_modeinfo.dat` | Per-mode telluric correction parameters |
| `instruments/<name>/telluric_ewadjustments.dat` | Per-mode / per-order EW scale adjustments |
| `instruments/<name>/telluric_shiftinfo.dat` | Per-mode / per-order wavelength-shift windows |
| `instruments/<name>/IP_coefficients.dat` | Slit-width → instrumental-profile model coefficients |
| `instruments/<name>/<Mode>_flatinfo.fits` | Order-edge map, plate scale, resolving power, slit dimensions per observing mode |
| `instruments/<name>/<Mode>_wavecalinfo.fits` | Reference wavelength-solution grid per mode |
| `instruments/<name>/<Mode>_lines.dat` | Arc-lamp emission-line list per mode |

Instrument dispatch is done at runtime in `extract/load_image.py`:

```python
module = f'pyspextool.instruments.{setup.state["instrument"]}.{setup.state["instrument"]}'
instr  = importlib.import_module(module)
result = instr.read_fits(...)
```

Adding iSHELL requires **no changes to the dispatch mechanism**; only new
files inside a new `instruments/ishell/` package are needed, plus a one-line
addition to `config.state["instruments"]`.

### Key Difference vs. SpeX / uSpeX

SpeX and uSpeX share a 1024 × 1024 ALADDIN InSb array.  Their echelle
orders are roughly aligned with detector columns, so the existing
`extract_1dxd()` / `trace_spectrum_1dxd()` routines can operate directly on
raw (flat-fielded) images.

iSHELL uses a 2048 × 2048 Teledyne H2RG array.  Its cross-dispersed echelle
orders are **tilted** with respect to the detector axes.  Spectra must be
**rectified** (resampled on to a rectilinear wavelength-vs-spatial grid) before
the generic `extract_1dxd()` machinery can be applied.  This rectification step
is the primary new piece of instrument-specific logic required.

---

## 2. SpeX-Specific Assumptions in Current Code

The following locations in the existing code contain SpeX/uSpeX-specific
assumptions that are relevant to iSHELL integration.

### 2.1 `setup_utils.py` — `set_instrument()`

```python
# Line ~411
if instrument_name in ["uspex", "spex"]:
    setup.state["irtf"] = True
```

**Action:** Add `"ishell"` to this list.  iSHELL is also an IRTF instrument.

### 2.2 `setup_utils.py` — bad-pixel-mask filename convention

```python
bad_pixel_mask_file = os.path.join(
    instrument_data_path,
    setup.state["instrument"] + "_bdpxmk.fits"
)
```

This is already generic: it uses the instrument name, so `ishell_bdpxmk.fits`
will be found automatically.

### 2.3 `instruments/spex/spex.py` — `correct_linearity()`

Implements the Vacca et al. (2004) algorithm, which is specific to the SpeX
ALADDIN detector and its `slowcnts` readout parameter.  **This function cannot
be reused** for iSHELL's H2RG detector; a new linearity correction must be
written.

### 2.4 `instruments/uspex/uspex.py` — `correct_uspexbias()`

Corrects 32-amplifier bias drifts specific to the upgraded SpeX InSb
array.  **Not applicable** to iSHELL.

### 2.5 `extract/load_image.py` — `wavecaltype`

Reads `wavecaltype` from `<Mode>_wavecalinfo.fits`.  The pipeline branches on
this value (`'1D'` vs. `'2D'`).  iSHELL will require a `'2D'` calibration type.
Confirm this field is already parsed generically (it is, via the FITS header);
no code change expected here.

### 2.6 `io/read_instrument_file.py`

Parses the `.dat` configuration file.  Parsing is fully generic.  No changes
required.

### 2.7 Telluric module

`telluric/` reads `telluric_modeinfo.dat`, `telluric_ewadjustments.dat`, and
`telluric_shiftinfo.dat` from `setup.state["instrument_path"]`.  Since these
are loaded by path at runtime they are already instrument-neutral.  iSHELL
only needs its own versions of these data files populated with real values.

---

## 3. Proposed Module Tree

```
src/pyspextool/
├── config.py                              MODIFIED  add 'ishell'
├── setup_utils.py                         MODIFIED  add 'ishell' to IRTF list
└── instruments/
    ├── spex/   (unchanged)
    ├── uspex/  (unchanged)
    └── ishell/                            NEW PACKAGE
        ├── __init__.py
        ├── ishell.dat                     detector constants, keywords
        ├── ishell.py                      read_fits / get_header / load_data
        ├── ishell_bdpxmk.fits             2048×2048 bad-pixel mask  (TBD)
        ├── IP_coefficients.dat            slit-width → IP model     (TBD)
        ├── telluric_modeinfo.dat          J / H / K telluric params (TBD)
        ├── telluric_ewadjustments.dat     EW scale corrections      (TBD)
        ├── telluric_shiftinfo.dat         shift windows             (TBD)
        ├── J_lines.dat                    ThAr arc lines, J sub-modes (TBD)
        ├── H_lines.dat                    ThAr arc lines, H sub-modes (TBD)
        └── K_lines.dat                    ThAr arc lines, K sub-modes (TBD)
```

Calibration FITS files (`<Mode>_flatinfo.fits`, `<Mode>_wavecalinfo.fits`)
will be derived from real iSHELL data; they are **not** created in this phase.

---

## 4. Dependency Map

```
pyspextool_setup()
    └─ set_instrument('ishell')
           ├─ loads  ishell.dat              → suffix, nint, lincormax, keywords
           ├─ opens  ishell_bdpxmk.fits      → raw_bad_pixel_mask
           └─ reads  pyspextool_keywords.dat → pyspextool_keywords  [shared]

extract.load_image()
    └─ importlib.import_module('…ishell.ishell')
           └─ ishell.read_fits()
                  ├─ ishell.get_header()     → normalised keyword dict
                  └─ ishell.load_data()
                         ├─ correct_ishell_linearity()   [NEW – ishell.py]
                         └─ rectify_orders()             [NEW – ishell.py]

extract.extract_apertures()
    └─ extract_1dxd()    [REUSED – extract/extraction.py]

combine / telluric / merge
    └─ fully reused, data-file-driven
```

---

## 5. Code Classification

### 5.1 Code reused unchanged

| Module | Reason |
|---|---|
| `extract/extraction.py` — `extract_1dxd()` | Generic optimal-extraction math |
| `extract/trace_spectrum.py` — `trace_spectrum_1dxd()` | Generic polynomial trace |
| `extract/make_flat.py` | Flat-field algorithm is detector-neutral |
| `extract/make_profiles.py` | Spatial-profile fitting is generic |
| `extract/identify_apertures.py` | Aperture identification is generic |
| `extract/define_aperture_parameters.py` | Generic |
| `combine/` (all) | No instrument assumptions |
| `telluric/` (all) | Data-file driven; only `telluric_modeinfo.dat` changes |
| `merge/` (all) | No instrument assumptions |
| `io/` (all) | Generic I/O utilities |
| `fit/`, `plot/`, `utils/` | Fully generic |
| `setup_utils.py` — `pyspextool_setup()` | Only minor addition to IRTF list |

### 5.2 Code requiring instrument-specific overrides (new files only)

| File | What must be provided |
|---|---|
| `instruments/ishell/ishell.dat` | iSHELL-specific constants (gain, readnoise, `LINCORMAX`, keywords) |
| `instruments/ishell/ishell.py` | `read_fits()`, `get_header()`, `load_data()` for H2RG + iSHELL headers |
| `instruments/ishell/ishell_bdpxmk.fits` | H2RG bad-pixel mask (from observatory) |
| `instruments/ishell/telluric_modeinfo.dat` | J/H/K telluric parameters |
| `instruments/ishell/telluric_ewadjustments.dat` | Filled from real data |
| `instruments/ishell/telluric_shiftinfo.dat` | Filled from real data |
| `instruments/ishell/IP_coefficients.dat` | iSHELL slit-width IP parameters |
| `instruments/ishell/{J,H,K}_lines.dat` | ThAr line lists for iSHELL passbands |

### 5.3 Entirely new implementation needed

| Capability | Location | Notes |
|---|---|---|
| H2RG linearity correction | `instruments/ishell/ishell.py` | Polynomial correction; no `slowcnts` equivalent |
| H2RG reference-pixel bias subtraction | `instruments/ishell/ishell.py` | 4-side reference pixel averaging |
| 2D order rectification | `instruments/ishell/ishell.py` | Tilt/curvature correction before extraction; driven by `<Mode>_wavecalinfo.fits` |
| iSHELL FITS keyword normalisation | `instruments/ishell/ishell.py` | Different header keyword names |
| Calibration FITS files | `instruments/ishell/<Mode>_flatinfo.fits` etc. | Must be generated from real data using `make_flat` / `make_wavecal` pipeline runs |

---

## 6. Phased Implementation Plan

### Phase 0 – Scaffolding (this PR)

- [x] Create `instruments/ishell/` package directory
- [x] Stub `ishell.py` with interface-compatible `read_fits()`, `get_header()`,
      and `load_data()` (raise `NotImplementedError` where instrument-specific
      logic is not yet written)
- [x] Create placeholder data files with correct formats but `TBD` / zero data
- [x] Add `'ishell'` to `config.state["instruments"]`
- [x] Add `'ishell'` to the IRTF instrument list in `setup_utils.py`
- [x] Add `tests/test_ishell.py` covering import, config registration, and
      `set_instrument()` call
- [x] Populate confirmed values from iSHELL Spextool Manual (Jan 2020):
      NINT=3, LINCORMAX=30000 DN, pixel scale, arc lamp, telluric method

### Phase 1 – Header and FITS reading

- [ ] Implement `get_header()` with real iSHELL keyword mapping
- [ ] Implement `load_data()` for single-image loading
- [ ] Implement H2RG reference-pixel bias subtraction
- [ ] Implement polynomial linearity correction for H2RG
- [ ] Implement `read_fits()` (sequential and pair-subtract modes)
- [ ] Obtain and commit real `ishell_bdpxmk.fits` and iSHELL linearity file
- [ ] Add pooch registry entry for `ishell_lincorr.fits` (if remote-hosted)

### Phase 2 – Order geometry and rectification

- [ ] Generate `<Mode>_flatinfo.fits` and `<Mode>_wavecalinfo.fits` from
      flat-field and ThAr arc-lamp frames taken at the telescope for each
      J/H/K sub-mode (J0–J3, H1–H3, K1–K3/Kgas)
- [ ] Implement `_rectify_orders()` using the wavecal grid to transform tilted
      orders onto a rectilinear grid
- [ ] Validate that `extract_1dxd()` works correctly on rectified iSHELL data
- [ ] Populate `{J,H,K}_lines.dat` with confirmed ThAr lines in each passband

### Phase 3 – Telluric and calibration files

- [ ] Populate `telluric_modeinfo.dat` with real J/H/K IP parameters
- [ ] Populate `telluric_ewadjustments.dat` from A0 V standard observations
- [ ] Populate `telluric_shiftinfo.dat` from cross-correlation measurements
- [ ] Populate `IP_coefficients.dat` for each iSHELL slit width
      (mandatory: iSHELL does not support the deconvolution method)
- [ ] End-to-end test with real iSHELL science data

### Phase 4 – Validation and release

- [ ] Compare extracted spectra against IDL-based iSHELL pipeline outputs
- [ ] Add integration tests with representative iSHELL test data (Git LFS)
- [ ] Update documentation and notebooks
- [ ] Tag release

---

## 7. Likely Blockers and Unknowns

| # | Item | Impact | Mitigation |
|---|---|---|---|
| B1 | **Exact iSHELL H2RG linearity-correction algorithm** – LINCORMAX is confirmed as 30000 DN (Manual Table 4), but the pixel-by-pixel polynomial coefficient file is not publicly available | High – incorrect photometry / flux calibration | Contact IRTF instrument scientists; request `ishell_lincorr.fits` calibration file |
| B2 | **Order tilt / curvature model format** – the current `wavecalinfo.fits` format may need extension to carry 2D polynomial coefficients describing order tilt | High – rectification impossible without this | Inspect existing iSHELL `.fits` calibration files from observatory; extend format if needed |
| B3 | **Bad-pixel mask** – not yet available for H2RG; stale pixels change over time | Medium – may introduce artefacts | Request current mask from IRTF; add to pooch registry |
| B4 | **ThAr arc-line lists** – confirmed ThAr lamp (Manual Table 1); line positions in J/H/K must be selected from the NIST database | Medium – wavelength calibration fails without real lines | Source Th I, Th II, Ar I, Ar II vacuum wavelengths from NIST Atomic Spectra Database |
| B5 | **Raw FITS keyword names** – the Manual documents *output* Spextool keywords (Table 4) but not the *raw* iSHELL FITS header keyword names | Medium – incorrect flux scaling / header parsing | Confirm from real iSHELL FITS file; output keywords DATE/TIME/NCOADDS/ITIME/RA/DEC/PA/HA/AM are documented |
| B6 | **H2RG pedestal/signal read counts** – must use FITS ext 1 + ext 2 sum to identify non-linear pixels (Manual §2.3), not just ext 0 | Medium – incorrect linearity flagging | Implement `load_data()` to read all three extensions (NINT=3) |
| B7 | **Slit length** – iSHELL J/H/K slit is 5"; L/Lp/M thermal modes use 15" or 25"; spatial-profile assumptions fine for 5" near-IR modes | Low | Confirmed from Manual Table 1 |
| B8 | **Test data availability** – Git LFS test data does not yet exist for iSHELL | Low (for Phase 0–1) | Use synthetic/simulated data for unit tests; add real data in Phase 3 |
| B9 | **Exact raw FITS keyword names** – the table in `get_header()` must be confirmed against real iSHELL FITS files | Medium | Cross-check with IRTF DCS documentation or existing iSHELL reduction scripts |

---

## 8. iSHELL Instrument Reference (J/H/K modes)

The following values are sourced from the iSHELL Spextool Manual (v10jan2020,
Cushing).  Fields marked **TBD** still require confirmation from IRTF staff
or real iSHELL calibration files.

### 8.1 Detector Parameters

| Parameter | Value | Source |
|---|---|---|
| Detector | 2048 × 2048 Teledyne H2RG | Manual §2.2 |
| Pixel scale | **0.125 arcsec/pixel** | Manual Table 4, PLTSCALE example |
| NINT (FITS extensions per file) | **3** | Manual §2.3, Eq. 1 |
| LINCORMAX | **30000 DN** | Manual Table 4, LINCRMAX example |
| Gain | **TBD** (~1.8 e⁻/DN) | Not stated in manual; confirm from IRTF |
| Read noise (Fowler-16) | **TBD** (~5–10 e⁻) | Not stated in manual; confirm from IRTF |
| Reference pixel width | 4 pixels on all 4 edges | H2RG standard |
| Arc lamp (J/H/K/L1) | **ThAr** (thorium-argon) | Manual Table 1 |
| Slit length (J/H/K) | **5 arcseconds** | Manual Table 1 |
| Telluric correction method | **IP only** (deconvolution not available) | Manual §6.2 |
| Resolving power (0.375″ slit) | 75 000 | Manual §2.2 |

### 8.2 Near-IR Mode Table (J / H / K bands only)

All J/H/K modes use a ThAr arc lamp and a 5" slit (Manual Table 1).

| Mode | Orders | Wavelength range (µm) |
|---|---|---|
| J0 | 457–503 | 1.062–1.165 |
| J1 | 432–477 | 1.115–1.228 |
| J2 | 406–442 | 1.197–1.303 |
| J3 | 387–417 | 1.265–1.364 |
| H1 | 311–355 | 1.473–1.683 |
| H2 | 299–388 | 1.544–1.748 |
| H3 | 285–319 | 1.633–1.832 |
| K1 | 233–270 | 1.918–2.228 |
| K2 | 218–248 | 2.084–2.382 |
| Kgas | 211–238 | 2.170–2.468 |
| K3 | 203–229 | 2.253–2.549 |

Available slit widths: 0.375″, 0.75″, 1.5″, 4.0″.
