# iSHELL Resource Layout — Authoritative Inventory

**Status:** Phase 1 (typed calibration readers)
**Scope:** J/H/K modes only (J0–J3, H1–H3, K1–K3, Kgas)  
**Last updated:** 2026-03

This note documents which iSHELL packaged resources are authoritative,
which are placeholder stubs, and which legacy files have been removed.

---

## 1. Package Layout

```
src/pyspextool/instruments/ishell/
├── __init__.py                   # empty (plugin entry point)
├── ishell.py                     # Phase-0 instrument backend (public API)
├── resources.py                  # Python resource-accessor API
├── modes.yaml                    # Mode registry (J/H/K only)
├── ishell.dat                    # pySpextool instrument config (AUTHORITATIVE)
├── ishell_bdpxmk.fits            # Bad-pixel mask (PLACEHOLDER – all zeros)
├── IP_coefficients.dat           # IP model coefficients (PLACEHOLDER – no data rows)
├── telluric_modeinfo.dat         # Telluric mode parameters (PLACEHOLDER – no numeric data)
├── telluric_ewadjustments.dat    # Telluric EW adjustments (PLACEHOLDER – no data rows)
├── telluric_shiftinfo.dat        # Telluric shift windows (PLACEHOLDER – no data rows)
├── J_lines.dat                   # Band-level J ThAr line list (PLACEHOLDER)
├── H_lines.dat                   # Band-level H ThAr line list (PLACEHOLDER)
├── K_lines.dat                   # Band-level K ThAr line list (PLACEHOLDER)
└── data/
    ├── ishell_lincorr_CDS.fits   # Linearity-correction cube (REAL – IDL Dec 2016)
    ├── ishell_bdpxmk.fits        # Bad-pixel mask (PLACEHOLDER – all zeros)
    ├── ishell_htpxmk.fits        # Hot-pixel mask (PLACEHOLDER – all zeros)
    ├── ishell_bias.fits          # Bias frame (CANDIDATE REAL – needs format verification)
    ├── IP_coefficients.dat       # IP coefficients (REAL – Cushing 2014, 3 slit widths)
    ├── J0_lines.dat … Kgas_lines.dat   # Per-mode ThAr line lists (REAL – Cushing 2017)
    ├── J0_flatinfo.fits … Kgas_flatinfo.fits   # Flat calibration (REAL – IDL-era)
    └── J0_wavecalinfo.fits … Kgas_wavecalinfo.fits  # Wavecal metadata (REAL coefficients,
                                                      #  NaN data arrays – needs Phase 1 work)
```

---

## 2. Authoritative Files

| File | Location | Used by | Notes |
|---|---|---|---|
| `ishell.dat` | `ishell/` (top-level) | `set_instrument()` | pySpextool instrument config; NINT=3, LINCORMAX=30000 |
| `ishell_bdpxmk.fits` | `ishell/` (top-level) | `set_instrument()` | Loaded into `state['raw_bad_pixel_mask']` |
| `modes.yaml` | `ishell/` | `resources.py` | Mode registry for J/H/K; all 11 modes present |
| `resources.py` | `ishell/` | Python API callers | Single access point for all packaged resources |
| `data/ishell_lincorr_CDS.fits` | `data/` | `resources.get_linearity_cube()` | Real linearity cube from Dec 2016 IDL pipeline |
| `data/IP_coefficients.dat` | `data/` | telluric module (future) | Real IP coefficients for 3 slit widths (Cushing 2014) |
| `data/<Mode>_lines.dat` | `data/` | wavecal module (future) | Real ThAr line lists (Cushing Oct 2017) |
| `data/<Mode>_flatinfo.fits` | `data/` | flat module (future) | Real IDL-era flat calibrations |
| `data/<Mode>_wavecalinfo.fits` | `data/` | wavecal module (future) | Structure validated; header-keyword coefficients and spectral-array semantics only partially confirmed |

---

## 3. Placeholder Files

These files exist so that the resource registry resolves and tests pass.
They must be replaced with real calibration data before production use.

| File | Location | Status | How to recognize |
|---|---|---|---|
| `ishell_bdpxmk.fits` | `ishell/` | PLACEHOLDER | All-zero int16; FITS COMMENT contains `PLACEHOLDER:` |
| `data/ishell_bdpxmk.fits` | `data/` | PLACEHOLDER | All-zero uint8; FITS COMMENT contains `PLACEHOLDER:` |
| `data/ishell_htpxmk.fits` | `data/` | PLACEHOLDER | All-zero uint8; FITS COMMENT contains `PLACEHOLDER:` |
| `data/ishell_bias.fits` | `data/` | CANDIDATE – format TBC | Real-ish values but epoch 1970; needs verification |
| `IP_coefficients.dat` | `ishell/` | PLACEHOLDER | Comment-only file; no numeric rows |
| `telluric_modeinfo.dat` | `ishell/` | PLACEHOLDER | Mode rows present; all numeric columns empty |
| `telluric_ewadjustments.dat` | `ishell/` | PLACEHOLDER | Header present; no data rows |
| `telluric_shiftinfo.dat` | `ishell/` | PLACEHOLDER | Header present; no data rows |
| `J_lines.dat`, `H_lines.dat`, `K_lines.dat` | `ishell/` | PLACEHOLDER | Band-level lists; comment-only |

`resources.py` exposes `is_placeholder_resource(key)` to programmatically
identify which detector resources are placeholder stubs.

---

## 4. Removed Legacy Files

The following files were present in the original scaffold but have been
**removed** because they are stray copies of IDL-era xSpextool artifacts
that predate pySpextool and use an incompatible format:

| Removed file | Reason |
|---|---|
| `data/ishell.dat` | IDL xSpextool config (NINT=5, XSPEXTOOL_KEYWORD format). Superseded by `ishell/ishell.dat` (NINT=3, pySpextool format). |
| `data/xtellcor_modeinfo.dat` | IDL xtellcor format (numeric method codes). Superseded by `ishell/telluric_modeinfo.dat` (pySpextool format). |

---

## 5. Duplicate / Duality Notes

### `ishell_bdpxmk.fits` appears in two locations

| Location | Used by | Content |
|---|---|---|
| `ishell/ishell_bdpxmk.fits` | `set_instrument()` (via `setup_utils.py`) | Placeholder; all-zero int16; explicit PLACEHOLDER header |
| `data/ishell_bdpxmk.fits` | `resources.get_bad_pixel_mask()` | Placeholder; all-zero uint8; explicit PLACEHOLDER header |

Both are placeholder stubs with the same logical content (all good).
The duality exists because `setup_utils.py` hardcodes the path
`<instrument_dir>/<name>_bdpxmk.fits` for all instruments (consistent with
SpeX and uSpeX), while `resources.py` keeps all detector calibration files
in `data/`.  Both files will be replaced by the same real mask once IRTF
provides calibration data.

### `IP_coefficients.dat` appears in two locations

| Location | Content |
|---|---|
| `ishell/IP_coefficients.dat` | Placeholder (comment-only, no numeric rows) |
| `data/IP_coefficients.dat` | Real coefficients (Cushing 2014, three slit widths) |

The `data/` copy is the authoritative source for the telluric module.
The top-level copy exists for legacy test compatibility and will be
consolidated in Phase 1.

---

## 6. What Still Needs Real Calibration Files

Before Phase 1 reduction logic can be implemented:

1. **`data/ishell_bdpxmk.fits`** – Replace with real bad-pixel map from IRTF.
2. **`data/ishell_htpxmk.fits`** – Replace with real hot-pixel map from IRTF.
3. **`data/ishell_bias.fits`** – Verify format and epoch; replace if necessary.
4. **`data/<Mode>_wavecalinfo.fits`** – Structure is validated (shape, NORDERS,
   NORDERS header keyword).  Header-keyword values that appear to be polynomial
   coefficients are present, but their exact scientific meaning has not been
   confirmed by reverse-engineering the IDL pipeline.  Spectral data arrays are
   all NaN and a valid wavecal solution is needed before reduction logic can use
   this file.
5. **`telluric_modeinfo.dat`** – All numeric columns are empty (TBD).
6. **`telluric_ewadjustments.dat`** – No data rows (TBD).
7. **`telluric_shiftinfo.dat`** – No data rows (TBD).
8. **`J_lines.dat`, `H_lines.dat`, `K_lines.dat`** – Band-level placeholder
   lists; per-mode lists in `data/` are real (Cushing 2017).

---

## 7. Typed Calibration Reader API (`calibrations.py`)

The module `pyspextool.instruments.ishell.calibrations` provides typed
readers that parse each packaged resource into a well-defined Python
dataclass.  It builds on top of `resources.py` (which only returns
`importlib.resources` path objects) and adds:

* File-existence validation,
* Git LFS pointer detection (raises `RuntimeError` with an actionable
  message instead of silently passing a corrupt file to downstream code),
* Structural and dimensional consistency checks,
* Typed Python dataclasses with documented fields.

### Dataclasses

| Class | Returned by | Key fields |
|---|---|---|
| `LineListEntry` | (internal) | `order`, `wavelength_um`, `species`, `fit_type`, `fit_n_terms` |
| `LineList` | `read_line_list(mode)` | `mode`, `entries`, `n_lines`, `orders`, `wavelengths_um` |
| `FlatInfo` | `read_flatinfo(mode)` | `mode`, `orders`, `n_orders`, `rotation`, `plate_scale_arcsec`, `image` (2048×2048) |
| `WaveCalInfo` | `read_wavecalinfo(mode)` | `mode`, `n_orders`, `orders`, `resolving_power`, `data` (NORDERS×4×N), `linelist_name` |
| `LinearityCube` | `read_linearity_cube()` | `data` (9×2048×2048), `dn_lower_limit`, `dn_upper_limit`, `fit_order` |
| `PixelMask` | `read_pixel_mask(type)` | `mask_type`, `data` (2048×2048), `is_placeholder` |
| `BiasFrame` | `read_bias()` | `data` (2048×2048), `divisor` |
| `ModeCalibrations` | `load_mode_calibrations(mode)` | `mode`, `line_list`, `flatinfo`, `wavecalinfo` |

### Runtime-Critical Resources

These resources must be loadable without error before any reduction stage
can run:

| Resource | Reader | Used by |
|---|---|---|
| `data/<Mode>_lines.dat` | `read_line_list(mode)` | wavelength calibration |
| `data/<Mode>_flatinfo.fits` | `read_flatinfo(mode)` | order tracing / flat fielding |
| `data/<Mode>_wavecalinfo.fits` | `read_wavecalinfo(mode)` | wavelength solution |
| `data/ishell_lincorr_CDS.fits` | `read_linearity_cube()` | detector non-linearity correction |
| `data/ishell_bdpxmk.fits` | `read_pixel_mask("bad")` | bad-pixel masking |
| `data/ishell_htpxmk.fits` | `read_pixel_mask("hot")` | hot-pixel masking |
| `data/ishell_bias.fits` | `read_bias()` | bias subtraction |

### Partially-Understood Formats

The following aspects of the packaged files are not yet fully documented
and may require further reverse-engineering of the IDL pipeline:

1. **`*_flatinfo.fits` per-order polynomial coefficients** – Header keywords
   `OR{n}_B1`..`OR{n}_B5` (bottom-edge polynomial) and `OR{n}_T1`..`OR{n}_T5`
   (top-edge polynomial) encode the order-trace geometry, but the polynomial
   variable (pixel column vs. row) is not confirmed.

2. **`*_wavecalinfo.fits` data cube planes** – The cube shape is
   `(NORDERS, 4, N_PIXELS)`.  The four planes are spectral arrays of some
   kind (wavelength, sky, signal, uncertainty?) but the exact labels have
   not yet been recovered from the IDL source.

3. **`ishell_bias.fits`** – The file contains an apparently real iSHELL
   calibration observation from 2016.  The `DIVISOR=10` header suggests
   it is a mean of 10 co-adds.  Whether the epoch is appropriate for
   current observations has not been verified.

### Quick-Start Example

```python
from pyspextool.instruments.ishell.calibrations import load_mode_calibrations, read_linearity_cube

# Load all mode-specific resources at once
cal = load_mode_calibrations("K1")
print(cal.line_list.n_lines)      # e.g. 120
print(cal.flatinfo.n_orders)      # e.g. 38
print(cal.wavecalinfo.data.shape) # e.g. (38, 4, 2028)

# Load detector resources individually
lin = read_linearity_cube()
print(lin.dn_upper_limit)         # 37000.0 DN
```
