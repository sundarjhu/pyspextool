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
        ├── J_lines.dat                    Ar + Xe arc lines, J band (TBD)
        ├── H_lines.dat                    Ar + Xe arc lines, H band (TBD)
        └── K_lines.dat                    Ar + Xe arc lines, K band (TBD)
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
| `instruments/ishell/{J,H,K}_lines.dat` | Ar + Xe line lists for iSHELL passbands |

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

### Phase 1 – Header and FITS reading

- [ ] Implement `get_header()` with real iSHELL keyword mapping
- [ ] Implement `load_data()` for single-image loading
- [ ] Implement H2RG reference-pixel bias subtraction
- [ ] Implement polynomial linearity correction for H2RG
- [ ] Implement `read_fits()` (sequential and pair-subtract modes)
- [ ] Obtain and commit real `ishell_bdpxmk.fits` and iSHELL linearity file
- [ ] Add pooch registry entry for `ishell_lincorr.fits` (if remote-hosted)

### Phase 2 – Order geometry and rectification

- [ ] Generate `J_flatinfo.fits`, `H_flatinfo.fits`, `K_flatinfo.fits` from
      flat-field frames taken at the telescope
- [ ] Generate `J_wavecalinfo.fits`, `H_wavecalinfo.fits`, `K_wavecalinfo.fits`
      from arc-lamp frames
- [ ] Implement `rectify_orders()` using the wavecal grid to transform tilted
      orders onto a rectilinear grid
- [ ] Validate that `extract_1dxd()` works correctly on rectified iSHELL data
- [ ] Populate `{J,H,K}_lines.dat` with Ar + Xe lines in each passband

### Phase 3 – Telluric and calibration files

- [ ] Populate `telluric_modeinfo.dat` with real J/H/K parameters
- [ ] Populate `telluric_ewadjustments.dat` from A0 V standard observations
- [ ] Populate `telluric_shiftinfo.dat` from cross-correlation measurements
- [ ] Populate `IP_coefficients.dat` for each iSHELL slit width
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
| B1 | **Exact iSHELL H2RG linearity-correction algorithm** – the IRTF team may use a custom method not publicly documented | High – incorrect photometry / flux calibration | Contact IRTF instrument scientists; inspect existing iSHELL pipeline code |
| B2 | **Order tilt / curvature model format** – the current `wavecalinfo.fits` format may need extension to carry 2D polynomial coefficients describing order tilt | High – rectification impossible without this | Inspect existing iSHELL `.fits` calibration files from observatory; extend format if needed |
| B3 | **Bad-pixel mask** – not yet available for H2RG; stale pixels change over time | Medium – may introduce artefacts | Request current mask from IRTF; add to pooch registry |
| B4 | **Arc-line lists** – Ar + Xe lamp; line positions in J/H/K differ from SpeX Ar-only lamp | Medium – wavelength calibration fails | Source from NIST Atomic Spectra Database filtered to iSHELL wavelength ranges |
| B5 | **`DIVISOR` keyword** – SpeX stores co-adds in `DIVISOR`; iSHELL uses `CO_ADDS` or similar | Medium – incorrect flux scaling | Confirm with real iSHELL FITS header; update `get_header()` accordingly |
| B6 | **Fowler sampling vs. MCDS** – some iSHELL readout modes do not produce a simple DN image; co-add handling may differ | Medium – data loading errors | Inspect real files; add readout-mode-specific handling |
| B7 | **Slit length** – iSHELL slit is 15 arcsec; SpeX is shorter; spatial-profile assumptions may break for faint targets | Low–Medium | Test with actual data; profile fitting should be robust |
| B8 | **Test data availability** – Git LFS test data does not yet exist for iSHELL | Low (for Phase 0–1) | Use synthetic/simulated data for unit tests; add real data in Phase 3 |
| B9 | **Exact FITS keyword names** – the table in `get_header()` must be confirmed against real iSHELL FITS files | Medium | Cross-check with IRTF DCS documentation or existing iSHELL reduction scripts |

---

## 8. iSHELL Instrument Reference (J/H/K modes)

The following values are sourced from publicly available IRTF documentation
and should be **confirmed against real data** before use in production code.
Fields marked **TBD** require confirmation from the IRTF instrument team.

| Parameter | J | H | K | Unit |
|---|---|---|---|---|
| Wavelength range | 1.14 – 1.34 | 1.47 – 1.81 | 1.98 – 2.52 | μm |
| Number of orders | ~8 | ~9 | ~9 | — |
| Detector | 2048 × 2048 H2RG | ← | ← | pixels |
| Pixel scale | ~0.375 | ← | ← | arcsec/pixel |
| Gain | **TBD** (~1.8) | ← | ← | e⁻/DN |
| Read noise (Fowler-16) | **TBD** (~5–10) | ← | ← | e⁻ |
| Linearity threshold | **TBD** (~55 000) | ← | ← | DN |
| Arc lamp | Ar + Xe | ← | ← | — |
| Resolving power (0.375″) | ~75 000 | ~75 000 | ~75 000 | — |
| Resolving power (0.75″) | ~37 500 | ~37 500 | ~37 500 | — |

Slit widths available: 0.375″, 0.75″, 1.5″, 4.0″ (cross × 15″ long).
