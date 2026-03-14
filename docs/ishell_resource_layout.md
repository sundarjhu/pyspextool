# iSHELL Resource Layout — Authoritative Inventory

**Status:** Phase 0 (scaffold)  
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
| `data/<Mode>_wavecalinfo.fits` | `data/` | wavecal module (future) | Real polynomial coefficients; spectral data arrays are NaN |

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
4. **`data/<Mode>_wavecalinfo.fits`** – The polynomial-coefficient headers are
   real, but spectral data arrays are all NaN.  A valid wavecal solution is
   needed before `get_header()` / `load_data()` can be implemented.
5. **`telluric_modeinfo.dat`** – All numeric columns are empty (TBD).
6. **`telluric_ewadjustments.dat`** – No data rows (TBD).
7. **`telluric_shiftinfo.dat`** – No data rows (TBD).
8. **`J_lines.dat`, `H_lines.dat`, `K_lines.dat`** – Band-level placeholder
   lists; per-mode lists in `data/` are real (Cushing 2017).
