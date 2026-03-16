# iSHELL Backend — Implementation Status Report

**Scope:** NASA IRTF iSHELL spectrograph, J / H / K modes only.  
**L, Lp, and M modes are explicitly out of scope.**  
**Date:** 2026-03-14  
**Prepared from repository inspection; reflects Phase 0 scaffolding state.**

---

## 1. Current Repository Structure

### 1.1 Top-level layout

```
pyspextool/
├── .github/workflows/          CI workflows (pytest, codecov)
├── docs/
│   ├── ishell_design_memo.md   Forward-looking design document (already present)
│   ├── ishell_implementation_status.md   THIS FILE
│   └── reference/              API reference docs
├── notebooks/                  Jupyter notebooks (SpeX / uSpeX examples only)
├── src/pyspextool/             Main package (src layout)
├── tests/                      pytest suite
├── pyproject.toml              Build config, dependencies, test config
└── setup.py                    Legacy stub (delegates to pyproject.toml)
```

### 1.2 Package layout under `src/`

```
src/pyspextool/
├── __init__.py
├── config.py                   Global mutable state dict
├── setup_utils.py              pyspextool_setup() / set_instrument()
├── pyspextoolerror.py          Custom exception class
├── batch/                      Batch-reduction helpers (SpeX/uSpeX)
├── combine/                    Spectral combining stage
├── extract/                    Core extraction pipeline
│   ├── background_subtraction.py
│   ├── combine_images.py
│   ├── config.py
│   ├── define_aperture_parameters.py
│   ├── do_all_steps.py
│   ├── extract.py
│   ├── extract_apertures.py
│   ├── extraction.py           ← extract_1dxd() [generic]
│   ├── flat.py
│   ├── identify_apertures.py
│   ├── images.py               ← rectify_order() [generic helper]
│   ├── load_image.py           ← instrument dispatch via importlib
│   ├── make_flat.py
│   ├── make_profiles.py
│   ├── make_wavecal.py
│   ├── override_aperturesigns.py
│   ├── profiles.py
│   ├── select_orders.py
│   ├── trace.py
│   ├── trace_apertures.py
│   └── wavecal.py
├── fit/                        Fitting utilities (all generic)
├── instruments/
│   ├── ishell/                 ← iSHELL package (Phase 0 complete)
│   │   ├── __init__.py
│   │   ├── ishell.dat          Instrument config (NINT=3, SUFFIX=.[ab], …)
│   │   ├── ishell.py           Stub module (NotImplementedError stubs)
│   │   ├── ishell_bdpxmk.fits  Placeholder bad-pixel mask in package root
│   │   ├── modes.yaml          Per-mode registry (J0–J3, H1–H3, K1–K3, Kgas)
│   │   ├── resources.py        importlib.resources accessor / cache layer
│   │   ├── IP_coefficients.dat Placeholder IP coefficients
│   │   ├── J_lines.dat         Placeholder ThAr line list (J-band)
│   │   ├── H_lines.dat         Placeholder ThAr line list (H-band)
│   │   ├── K_lines.dat         Placeholder ThAr line list (K-band)
│   │   ├── telluric_modeinfo.dat  Placeholder (Method=ip confirmed)
│   │   ├── telluric_ewadjustments.dat  Placeholder (empty data rows)
│   │   ├── telluric_shiftinfo.dat      Placeholder (empty data rows)
│   │   └── data/               Per-mode calibration FITS + data files
│   │       ├── ishell.dat       (IDL-era config, superseded by `instruments/ishell/ishell.dat` in the package root)
│   │       ├── ishell_bdpxmk.fits      2048×2048 placeholder bad-pixel mask
│   │       ├── ishell_htpxmk.fits      2048×2048 placeholder hot-pixel mask
│   │       ├── ishell_bias.fits        2048×2048 placeholder bias frame
│   │       ├── ishell_lincorr_CDS.fits Placeholder linearity cube
│   │       ├── IP_coefficients.dat     Mirror of package-root file
│   │       ├── xtellcor_modeinfo.dat   Legacy IDL-era telluric file
│   │       ├── {J0,J1,J2,J3}_flatinfo.fits    Placeholder 2048×2048
│   │       ├── {J0,J1,J2,J3}_wavecalinfo.fits Placeholder shape (47,4,761)
│   │       ├── {J0,…,Kgas}_lines.dat          Placeholder line lists
│   │       └── … (H1–H3, K1–K3, Kgas same pattern)
│   ├── spex/                   SpeX package (complete, production-ready)
│   └── uspex/                  uSpeX package (complete, production-ready)
├── io/                         Generic I/O (FITS, files, SIMBAD, …)
├── merge/                      Order-merging stage
├── plot/                       Plotting helpers
├── telluric/                   Telluric correction stage
└── utils/                      Math, arrays, interpolation, etc.
```

### 1.3 Test layout

```
tests/
├── conftest.py              pytest fixtures (raw_setup, proc_setup, postextraction_setup)
├── test_data/               Git submodule — SpeX / uSpeX raw + processed FITS files
├── test_ishell.py           25 tests: registration, imports, signatures, set_instrument()
├── test_ishell_modes.py     85 tests: modes.yaml, resources.py, FITS loaders
├── test_make_flat.py        Integration tests (SpeX / uSpeX data)
├── test_make_wavecal.py     Integration tests (SpeX / uSpeX data)
├── test_combine.py          Integration tests (SpeX / uSpeX data)
├── test_telluric.py         Integration tests (SpeX / uSpeX data)
├── test_merge_spectra.py    Integration tests (SpeX / uSpeX data)
└── test_*.py                Other unit / utility tests
```

### 1.4 Instrument-specific code / config / data locations

| Instrument | Code | Config | Calibration data |
|---|---|---|---|
| SpeX | `instruments/spex/spex.py` | `instruments/spex/spex.dat` | `instruments/spex/*.fits`, `*.dat` |
| uSpeX | `instruments/uspex/uspex.py` | `instruments/uspex/uspex.dat` | `instruments/uspex/*.fits`, `*.dat` |
| iSHELL | `instruments/ishell/ishell.py` | `instruments/ishell/ishell.dat` | `instruments/ishell/data/*.fits`, `*.dat` |

---

## 2. Existing Instrument Abstraction Points

### 2.1 Duck-typed plugin architecture

There is **no abstract base class**. Each instrument supplies a Python module
that satisfies an implicit three-callable interface. The pipeline discovers it
at runtime in `extract/load_image.py`:

```python
module = ('pyspextool.instruments.' + setup.state['instrument']
          + '.' + setup.state['instrument'])
instr = importlib.import_module(module)
result = instr.read_fits(...)
```

Required callables (signatures must match the interface contract):

| Callable | Signature |
|---|---|
| `read_fits` | `(files, linearity_info, keywords, pair_subtract, rotate, linearity_correction, extra, verbose)` |
| `get_header` | `(header, keywords)` |
| `load_data` | `(file, linearity_info, keywords, coefficients, linearity_correction)` |

### 2.2 `config.state['instruments']`

`src/pyspextool/config.py`:

```python
state = {"instruments": ['uspex', 'spex', 'ishell'], …}
```

**iSHELL is already registered here.**

### 2.3 `setup_utils.py` — `set_instrument()`

- Reads `<name>.dat` config file via `read_instrument_file()` (generic parser).
- Loads bad-pixel mask from `<name>_bdpxmk.fits` (generic filename convention).
- Sets `setup.state['irtf'] = True` for `['uspex', 'spex', 'ishell']` — **iSHELL
  is already in this list** (line 404 of `setup_utils.py`).

### 2.4 `telluric/` stage

Reads `telluric_modeinfo.dat`, `telluric_ewadjustments.dat`, and
`telluric_shiftinfo.dat` from `setup.state['instrument_path']` at runtime.
**Fully instrument-neutral; no code changes needed for iSHELL.**

### 2.5 Still SpeX-specific

| Location | Issue |
|---|---|
| `instruments/spex/spex.py` — `correct_linearity()` | Vacca et al. (2004) algorithm, specific to ALADDIN detector and `slowcnts` readout parameter. Not usable for iSHELL H2RG. |
| `instruments/uspex/uspex.py` — `correct_uspexbias()` | 32-amplifier bias correction specific to upgraded SpeX InSb array. Not applicable to iSHELL. |
| `tests/conftest.py` fixtures | `raw_setup`, `proc_setup`, `postextraction_setup` list only SpeX / uSpeX entries; no iSHELL entries yet. |
| `notebooks/` | All six Jupyter notebooks cover SpeX / uSpeX; none for iSHELL. |
| `batch/batch.py` | Batch-reduction logic appears to reference SpeX-specific file-naming conventions. |

---

## 3. Existing Pipeline Stages

### 3.1 Raw FITS ingestion

| Module | Generic? |
|---|---|
| `extract/load_image.py` | **Generic dispatch** — dispatches to instrument module via `importlib.import_module`. |
| `instruments/spex/spex.py` — `read_fits()` | SpeX-specific (ALADDIN MEF format, 4 extensions, `slowcnts`). |
| `instruments/uspex/uspex.py` — `read_fits()` | uSpeX-specific (H1RG MEF, 5 extensions, 32-amplifier bias). |
| `instruments/ishell/ishell.py` — `read_fits()` | iSHELL-specific stub. Full implementation raises `NotImplementedError`. H2RG MEF, 3 extensions. Pair-subtract logic scaffolded. |

### 3.2 Instrument / mode configuration

| Module | Generic? |
|---|---|
| `config.py` — `state['instruments']` | **Generic** — list-based, iSHELL already present. |
| `setup_utils.py` — `set_instrument()` | **Generic** (file-based dispatch). iSHELL already supported. |
| `io/read_instrument_file.py` | **Generic** — parses `.dat` key-value format. |
| `instruments/ishell/modes.yaml` | iSHELL-specific — 11 modes registered (J0–J3, H1–H3, K1–K3, Kgas). |
| `instruments/ishell/resources.py` | iSHELL-specific — `importlib.resources` accessor. |

### 3.3 Flat-field support

| Module | Generic? |
|---|---|
| `extract/flat.py` — `read_flat_fits()` | **Generic** — reads `<Mode>_flatinfo.fits`. |
| `extract/make_flat.py` — `make_flat()` | **Generic** (detector-neutral algorithm). |
| `instruments/ishell/data/{mode}_flatinfo.fits` | iSHELL-specific — placeholder 2048×2048 FITS files committed for all 11 modes. Must be regenerated from real iSHELL flat-lamp frames. |

### 3.4 Wavecal support

| Module | Generic? |
|---|---|
| `extract/wavecal.py` — `read_wavecal_fits()` | **Generic** — reads `<Mode>_wavecalinfo.fits`. |
| `extract/make_wavecal.py` — `make_wavecal()` | **Generic** (polynomial fit). |
| `instruments/ishell/data/{mode}_wavecalinfo.fits` | iSHELL-specific — placeholder FITS files shape `(47, 4, 761)`. Must be regenerated from real ThAr arc frames. |
| `instruments/ishell/data/{mode}_lines.dat` | iSHELL-specific — placeholder ThAr line lists. No real line positions yet. |
| `instruments/ishell/J_lines.dat`, `H_lines.dat`, `K_lines.dat` | Band-level line list stubs (in package root; note: `data/` has per-mode files). |

### 3.5 Order geometry / rectification

| Module | Generic? |
|---|---|
| `extract/images.py` — `rectify_order()` | **Generic helper** — already present; can be reused. |
| `instruments/ishell/ishell.py` — `_rectify_orders()` | iSHELL-specific — private function stub (raises `NotImplementedError`). iSHELL echelle orders are tilted; rectification is required before extraction. |
| `extract/load_image.py` — `wavecaltype` dispatch | **Generic** — reads `wavecaltype` from FITS header. iSHELL will use `'2D'`. |

### 3.6 Science image preprocessing

| Module | Generic? |
|---|---|
| `extract/combine_images.py` | **Generic** — sigma-clipped mean/median combining. |
| `instruments/ishell/ishell.py` — `_subtract_reference_pixels()` | iSHELL-specific stub — H2RG 4-side reference pixel bias subtraction. Not yet implemented. |
| `instruments/ishell/ishell.py` — `_correct_ishell_linearity()` | iSHELL-specific stub — H2RG polynomial linearity correction. Not yet implemented. |
| `instruments/spex/spex.py` — `correct_linearity()` | SpeX-specific (Vacca 2004 / `slowcnts`). **Not reusable for iSHELL.** |

### 3.7 Extraction

| Module | Generic? |
|---|---|
| `extract/extraction.py` — `extract_1dxd()` | **Generic** — optimal-extraction math. Reusable for iSHELL after rectification. |
| `extract/trace.py` / `trace_apertures.py` | **Generic** — polynomial trace. |
| `extract/profiles.py` / `make_profiles.py` | **Generic** — spatial-profile fitting. |
| `extract/identify_apertures.py` | **Generic** |
| `extract/define_aperture_parameters.py` | **Generic** |
| `extract/background_subtraction.py` | **Generic** |

### 3.8 Spectral combination

| Module | Generic? |
|---|---|
| `combine/` (all modules) | **Generic** — no instrument assumptions. Fully reusable. |

### 3.9 Telluric correction

| Module | Generic? |
|---|---|
| `telluric/` (all modules) | **Generic** — data-file driven. Reads per-instrument `.dat` files. |
| `instruments/ishell/telluric_modeinfo.dat` | iSHELL-specific placeholder — Method=`ip` confirmed; numeric values TBD. |
| `instruments/ishell/telluric_ewadjustments.dat` | iSHELL-specific placeholder — all rows empty (TBD). |
| `instruments/ishell/telluric_shiftinfo.dat` | iSHELL-specific placeholder — all rows empty (TBD). |
| `instruments/ishell/IP_coefficients.dat` | iSHELL-specific placeholder — coefficient values TBD. |

Note: iSHELL supports **only the IP method** for telluric correction
(deconvolution method is explicitly not available per iSHELL Spextool Manual §6.2).

### 3.10 Packaged data / resource loading

| Module | Generic? |
|---|---|
| `setup_utils.py` — `mishu` (pooch registry) | Shared — currently contains SpeX / uSpeX remote files only. |
| `instruments/ishell/resources.py` | iSHELL-specific — uses `importlib.resources.files()` to load from `data/` sub-package. |
| `src/pyspextool/data/` | Shared — Vega model, atmosphere models, HI line list, pyspextool keywords. |

---

## 4. Existing Packaged Resource Model

### 4.1 Storage mechanism

Package data are bundled inside the installed package using setuptools
`include-package-data`. `pyproject.toml` declares:

```toml
[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
"*" = ["*.dat","*.fits","*.csv","*.txt","*.css","*.yaml"]
```

All files matching those extensions under `src/pyspextool/` are bundled.

### 4.2 Access mechanisms

Two patterns are used:

1. **`importlib.resources.files()` (Python ≥ 3.10)** — used in `setup_utils.py`
   and `instruments/ishell/resources.py`:
   ```python
   from importlib.resources import files
   path = files('pyspextool.instruments.ishell') / 'ishell.dat'
   ```

2. **`setup.state['instrument_path']`** — a filesystem path set by
   `set_instrument()` and used throughout the pipeline to locate mode-specific
   files (`<Mode>_flatinfo.fits`, `<Mode>_wavecalinfo.fits`, `<Mode>_lines.dat`,
   `telluric_modeinfo.dat`, etc.).

3. **`pooch` remote registry** (`setup_utils.py` — `mishu`)  — for large files
   not bundled in the package (linearity cubes, Vega model):
   ```python
   mishu = pooch.create(
       base_url="https://pyspextool.s3.us-east-1.amazonaws.com/",
       registry={
           "uspex_lincorr.fits": "9ba8c54…",
           "spex_lincorr.fits":  "47fcbd6…",
           …
       }
   )
   ```

### 4.3 Recommended location for iSHELL resource files

All iSHELL-specific bundled files should live in:

```
src/pyspextool/instruments/ishell/data/
```

This is the sub-package already accessed by `resources.py` as
`DATA_PACKAGE = "pyspextool.instruments.ishell.data"`.

Large files that cannot be bundled (e.g. a full-resolution linearity cube)
should be registered in the `mishu` pooch registry in `setup_utils.py` under
the key `"ishell_lincorr_CDS.fits"` (or similar), pointing to the same S3
bucket.

---

## 5. Existing Tests and Fixtures

### 5.1 How test data are organised

- `tests/test_data/` is a **Git submodule** pointing to an external repository
  containing raw and processed SpeX / uSpeX FITS files.
- No iSHELL test data exists yet.
- `tests/conftest.py` defines `raw_setup`, `proc_setup`, and
  `postextraction_setup` fixtures — all entries are SpeX / uSpeX only.

### 5.2 What the tests assume

| Test file | Assumes |
|---|---|
| `test_ishell.py` (25 tests) | Bundled package data only; no external data. All pass. |
| `test_ishell_modes.py` (85 tests) | Bundled package data only; no external data. All pass. |
| `test_make_flat.py` | Git submodule test data (SpeX / uSpeX). May be skipped if submodule not populated. |
| `test_make_wavecal.py` | Git submodule test data (SpeX / uSpeX). |
| `test_combine.py` | Git submodule test data (SpeX / uSpeX). |
| `test_telluric.py` | Git submodule test data (SpeX / uSpeX). |

### 5.3 Test pattern to follow for iSHELL

The existing `test_ishell.py` and `test_ishell_modes.py` files establish the
correct pattern for Phase 0:

- **Import-only / configuration tests** require no external data.
- **Interface-contract tests** verify callable signatures with `inspect`.
- **`NotImplementedError` tests** confirm Phase 0 stubs behave correctly.
- **Packaged-data tests** use `importlib.resources` paths directly.

For Phase 1 and beyond:
- **Unit tests for `get_header()`** should use synthetic `astropy.io.fits.Header`
  objects; no real FITS file needed.
- **Unit tests for `load_data()`** should create minimal synthetic 2048×2048
  ndarray inputs; avoid Git LFS for unit tests.
- **Integration tests** with real iSHELL FITS files should follow the
  `conftest.py` fixture pattern:
  - Add an `"ishell_j0"` entry to `raw_setup` and `proc_setup`.
  - Store raw iSHELL FITS files in `tests/test_data/raw/ishell-J0/`.
  - Track them via Git LFS (same as SpeX / uSpeX test data).

---

## 6. Recommended Insertion Points for an iSHELL Backend

### 6.1 Proposed new files / modules / directories

**Already in place (Phase 0 complete):**
```
src/pyspextool/instruments/ishell/
├── __init__.py                  ✅
├── ishell.dat                   ✅  (NINT=3, SUFFIX=.[ab], LINCORMAX=30000)
├── ishell.py                    ✅  (stubs; 6 NotImplementedError)
├── ishell_bdpxmk.fits           ✅  (placeholder 2048×2048 in package root)
├── modes.yaml                   ✅  (11 J/H/K modes)
├── resources.py                 ✅  (importlib.resources accessor)
├── IP_coefficients.dat          ✅  (placeholder)
├── J_lines.dat / H_lines.dat / K_lines.dat  ✅  (band-level stubs)
├── telluric_modeinfo.dat        ✅  (Method=ip confirmed; numerics TBD)
├── telluric_ewadjustments.dat   ✅  (placeholder)
├── telluric_shiftinfo.dat       ✅  (placeholder)
└── data/
    ├── ishell_bdpxmk.fits       ✅  (placeholder 2048×2048)
    ├── ishell_htpxmk.fits       ✅  (placeholder 2048×2048)
    ├── ishell_bias.fits         ✅  (placeholder 2048×2048)
    ├── ishell_lincorr_CDS.fits  ✅  (placeholder)
    ├── IP_coefficients.dat      ✅  (mirror)
    ├── xtellcor_modeinfo.dat    ✅  (legacy IDL-era telluric file)
    ├── {J0–J3,H1–H3,K1–K3,Kgas}_flatinfo.fits   ✅  (placeholders)
    ├── {J0–J3,H1–H3,K1–K3,Kgas}_wavecalinfo.fits ✅  (placeholders)
    └── {J0–J3,H1–H3,K1–K3,Kgas}_lines.dat        ✅  (placeholders)
```

**Needed for Phase 1 (header / FITS reading):**
```
src/pyspextool/instruments/ishell/
└── ishell.py   — implement get_header(), load_data(), _subtract_reference_pixels(),
                  _correct_ishell_linearity()
```

No new files or directories are required for Phase 1; only the existing stubs
in `ishell.py` need to be filled in.

**Needed for Phase 2 (real calibration data, from telescope observations):**
```
src/pyspextool/instruments/ishell/data/
├── {J0–J3}_flatinfo.fits       real values from iSHELL flat frames
├── {J0–J3}_wavecalinfo.fits    real wavelength solutions from ThAr frames
├── {J0–J3}_lines.dat           confirmed ThAr line positions
└── … (same for H1–H3, K1–K3, Kgas)
```

### 6.2 Minimal first PR scope (Phase 0 — already merged)

The following items constitute the minimum viable scaffold and are already
present in the repository:

- [x] `'ishell'` in `config.state['instruments']`
- [x] `'ishell'` in the IRTF list in `setup_utils.py`
- [x] `instruments/ishell/` package with all required placeholder files
- [x] `ishell.py` with correct signatures, docstrings, and `NotImplementedError`
      stubs for unimplemented functions
- [x] `modes.yaml` with all 11 J/H/K modes
- [x] `resources.py` for `importlib.resources`-based access
- [x] `tests/test_ishell.py` — 25 tests (all pass)
- [x] `tests/test_ishell_modes.py` — 85 tests (all pass)

### 6.3 What can be added without major refactoring

The following Phase 1–3 items require **no changes to any existing generic
pipeline module**:

1. `get_header()` in `ishell.py` — add keyword mapping table; no other file changes.
2. `load_data()` / `_subtract_reference_pixels()` / `_correct_ishell_linearity()` in `ishell.py`.
3. `read_fits()` body in `ishell.py` (the `read_fits` function wrapper is
   already scaffolded; only `load_data()` needs to be working first).
4. Real calibration FITS files in `instruments/ishell/data/` — drop-in
   replacements for placeholders.
5. Real ThAr line lists in `instruments/ishell/data/{mode}_lines.dat`.
6. Populate `telluric_modeinfo.dat`, `telluric_ewadjustments.dat`,
   `telluric_shiftinfo.dat` numeric values.
7. Add `"ishell"` entries to `tests/conftest.py` fixtures (no code change to
   conftest structure needed, only new dict entries).
8. Register `"ishell_lincorr_CDS.fits"` in the `mishu` pooch registry in
   `setup_utils.py` once the file is hosted.

---

## 7. Risk / Blocker List

### 7.1 Probably reusable (no changes needed)

| Component | Status |
|---|---|
| `extract/extraction.py` — `extract_1dxd()` | Generic optimal-extraction math. Reusable after rectification. |
| `extract/trace*.py`, `profiles.py`, `make_profiles.py` | Generic polynomial trace and profile fitting. |
| `extract/flat.py`, `make_flat.py` | Flat-field algorithm is detector-neutral. |
| `extract/wavecal.py`, `make_wavecal.py` | Wavelength solution fitting is generic. |
| `extract/images.py` — `rectify_order()` | Generic resampling helper; usable in iSHELL rectification. |
| `combine/` (all) | No instrument assumptions. |
| `telluric/` (all) | Data-file driven; only the `.dat` files need iSHELL values. |
| `merge/` (all) | No instrument assumptions. |
| `io/` (all) | Generic I/O utilities. |
| `fit/`, `plot/`, `utils/` | Fully generic. |
| `config.py` | Already includes `'ishell'`. |
| `setup_utils.py` — dispatch machinery | Already works for iSHELL. |

### 7.2 Needs adaptation

| Component | What must change | Risk |
|---|---|---|
| `instruments/ishell/ishell.py` — `get_header()` | Map real iSHELL FITS keyword names to pySpextool standard names. Raw keyword names are **not documented** in the iSHELL Spextool Manual; must be confirmed against real FITS files. (Blocker B9) | **Medium** |
| `instruments/ishell/ishell.py` — `load_data()` | Implement H2RG 3-extension read, reference-pixel bias subtraction, variance estimation. | **Medium** |
| `instruments/ishell/data/{mode}_flatinfo.fits` | Regenerate from real iSHELL flat-lamp frames. Placeholder shape 2048×2048 is correct but contains no real order-edge or plate-scale data. | **High** |
| `instruments/ishell/data/{mode}_wavecalinfo.fits` | Regenerate from real ThAr arc frames. Placeholder shape `(47, 4, 761)` may not match iSHELL order count and dispersion range. | **High** |
| `instruments/ishell/data/{mode}_lines.dat` | Populate with confirmed ThAr vacuum wavelengths from NIST ASD for Th I, Th II, Ar I, Ar II. | **Medium** |
| `telluric_modeinfo.dat` numeric values | Derive from real A0 V standard-star observations. | **Medium** |
| `IP_coefficients.dat` values | Derive from arc-lamp PSF measurements across slit widths. | **Medium** |
| `setup_utils.py` — `mishu` pooch registry | Add `"ishell_lincorr_CDS.fits"` hash once the file is hosted on S3. | **Low** (mechanical change) |
| `tests/conftest.py` fixtures | Add `"ishell_j0"` (and other mode) entries to `raw_setup` / `proc_setup`. | **Low** |

### 7.3 Likely new implementation needed

| Capability | Location | Notes |
|---|---|---|
| **H2RG linearity correction** | `instruments/ishell/ishell.py` — `_correct_ishell_linearity()` | Pixel-by-pixel polynomial correction using `ishell_lincorr_CDS.fits`. The exact algorithm and the coefficient cube file are not publicly available. **(Blocker B1: must request from IRTF instrument scientists.)** |
| **H2RG reference-pixel bias subtraction** | `instruments/ishell/ishell.py` — `_subtract_reference_pixels()` | Average the 4-pixel border on all four edges to subtract 1/f and DC bias. Standard H2RG procedure; no iSHELL-specific documentation needed, but must be implemented. |
| **2D order rectification** | `instruments/ishell/ishell.py` — `_rectify_orders()` | iSHELL orders are tilted with respect to the detector axes. Must transform tilted orders onto a rectilinear wavelength-vs-spatial grid before `extract_1dxd()` can be applied. Driven by `wavecalinfo.fits` 2D polynomial coefficients. **(Blocker B2: wavecalinfo.fits format may need extension to carry tilt/curvature coefficients.)** |
| **iSHELL FITS keyword normalisation** | `instruments/ishell/ishell.py` — `get_header()` | Different header keyword names from SpeX/uSpeX. Raw keyword names unknown (TBD). **(Blocker B9.)** |
| **Real calibration FITS files** | `instruments/ishell/data/` | `{mode}_flatinfo.fits` and `{mode}_wavecalinfo.fits` must be generated from telescope observations using the `make_flat` / `make_wavecal` pipeline. Cannot be synthesised computationally; requires actual iSHELL data. **(Blocker B2, B8.)** |
| **Bad-pixel / hot-pixel masks** | `instruments/ishell/data/ishell_bdpxmk.fits`, `ishell_htpxmk.fits` | Current placeholders are all-zero 2048×2048 images. Real masks must be requested from IRTF. **(Blocker B3.)** |

### 7.4 Summary of blockers by severity

| ID | Blocker | Severity | Mitigation |
|---|---|---|---|
| B1 | H2RG linearity-correction coefficient file (`ishell_lincorr_CDS.fits`) not publicly available | **High** | Contact IRTF instrument scientists; request file |
| B2 | `wavecalinfo.fits` format may need extension for 2D tilt/curvature coefficients | **High** | Inspect existing iSHELL IDL-pipeline calibration files; extend FITS format if needed |
| B3 | Real bad-pixel / hot-pixel masks not yet available | **Medium** | Request current masks from IRTF |
| B4 | ThAr line lists (J/H/K) not yet populated | **Medium** | Source Th I, Th II, Ar I, Ar II vacuum wavelengths from NIST Atomic Spectra Database |
| B5 | Raw iSHELL FITS keyword names not documented in Spextool Manual | **Medium** | Confirm from a real iSHELL raw FITS file |
| B6 | Legacy `data/ishell.dat` (IDL-era) says `NINT=5`; pySpextool `instruments/ishell/ishell.dat` (package root) correctly says `NINT=3` | **Low** | The `data/ishell.dat` file is an IDL-era artefact and is not read by the pySpextool pipeline. Action: add a note or remove `data/ishell.dat` to prevent future confusion. |
| B7 | Test data for iSHELL does not exist | **Low** (Phase 0–1) | Use synthetic data for unit tests; add real data in Phase 3 via Git LFS |
| B8 | Real `wavecalinfo.fits` and `flatinfo.fits` calibration files require telescope time | **High** | No workaround; Phase 2 cannot proceed without real iSHELL observations |
| B9 | Exact raw iSHELL FITS header keyword names unknown | **Medium** | Cross-check with a real iSHELL FITS file or DCS documentation |

---

## 8. Phase Completion Summary

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Scaffolding (package skeleton, stubs, placeholder data, tests) | ✅ **Complete** — 110/110 tests pass |
| **Phase 1** | Header and FITS reading (`get_header`, `load_data`, H2RG corrections) | 🔲 Not started |
| **Phase 2** | Order geometry, rectification, real calibration files | 🔲 Blocked (requires telescope data) |
| **Phase 3** | Telluric and calibration data population | 🔲 Blocked (requires A0V star observations) |
| **Phase 4** | Validation, integration tests, release | 🔲 Not started |

For the detailed phase plan see `docs/ishell_design_memo.md §6`.

---

## 9. 2DXD Calibration Scaffold — Stage Status

The 2DXD wavelength calibration pipeline has a separate scaffold track.

| Stage | Module | Description | Status |
|---|---|---|---|
| **Stage 1** | `tracing.py` | Flat-order tracing | ✅ Complete |
| **Stage 2** | `arc_tracing.py` | Arc-line tracing | ✅ Complete |
| **Stage 3** | `wavecal_2d.py` | Per-order provisional wavelength mapping | ✅ Complete |
| **Stage 4** | `wavecal_2d_surface.py` | Provisional global wavelength surface | ✅ Complete |
| **Stage 5** | `wavecal_2d_refine.py` | Coefficient-surface refinement | ✅ Complete (provisional scaffold) |
| **Stage 6** | *(not yet created)* | 2DXD rectification-index generation | 🔲 Not started |

### Stage 5 details

`wavecal_2d_refine.py` implements the IDL-style two-level fit structure:

1. Per-order polynomial fits: `wavelength_um(col) ≈ Σ_k a_k · col^k`
2. Order-dependence fits: `a_k(order) ≈ Σ_j d_{k,j} · v^j`
   where `v = order_ref / order_number`.

Result is stored in `RefinedCoefficientSurface` with `eval()` and
`eval_array()` helpers.

**What Stage 5 does not do** (still needed for a full 2DXD solution):

- Per-order column normalization (IDL normalizes to `[-1, +1]`).
- Iterative sigma-clipping on per-order or smoothness fits.
- Full IDL coefficient-index (`WDEG`/`ODEG`) compatibility.
- Rectification-index generation.
- Science-quality wavelength solution.

See `docs/ishell_wavecal_2d_refine.md` for full developer notes.
