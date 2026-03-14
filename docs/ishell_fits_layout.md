# iSHELL Raw FITS Layout

**Status:** Phase 1 (ingestion implemented)  
**Scope:** J/H/K modes only (J0–J3, H1–H3, K1–K3, Kgas)  
**Source:** iSHELL Spextool Manual v10jan2020 (Cushing et al.); iSHELL
xSpextool heritage keyword list.

This document describes the assumed raw FITS file layout for iSHELL science
frames as used by the pySpextool ingestion layer (`load_data()` /
`get_header()`).

---

## 1. File Format

iSHELL writes raw science files as **Multi-Extension FITS (MEF)** with
exactly **3 extensions** (`NINT = 3`):

| Extension | Type | Contents |
|---|---|---|
| 0 (PRIMARY) | Image | Signal difference `S = Σ pedestal_reads − Σ signal_reads` (total DN). All FITS header keywords are stored here. |
| 1 | Image | Pedestal sum `Σ p_{jk,i}` (total DN). |
| 2 | Image | Signal sum `Σ s_{jk,i}` (total DN). |

The signal difference `S` (extension 0) is the science image. Extensions 1
and 2 are used only for non-linearity flagging (see §3).

---

## 2. Primary Header Keywords

The following raw FITS keywords are recognised by `get_header()` and mapped to
standard pySpextool output keys:

| pySpextool key | Raw iSHELL keyword | Notes |
|---|---|---|
| `AM` | `TCS_AM` | Airmass; fallback: nan |
| `HA` | `TCS_HA` | Hour angle string; sign prefix added if absent |
| `PA` | `POSANGLE` | Slit position angle east of north (deg) |
| `RA` | `TCS_RA` | Right ascension string |
| `DEC` | `TCS_DEC` | Declination string; sign prefix added if absent |
| `ITIME` | `ITIME` | Integration time per frame (s) |
| `NCOADDS` | `CO_ADDS` | Number of co-adds |
| `IMGITIME` | (computed) | `ITIME × NCOADDS` (s) |
| `TIME` | `TIME_OBS` | UT observation start time |
| `DATE` | `DATE_OBS` | UT observation start date |
| `MJD` | `MJD_OBS` | Modified Julian date; computed from `DATE_OBS + TIME_OBS` if absent |
| `FILENAME` | `IRAFNAME` | Original FITS filename |
| `MODE` | `PASSBAND` | Observing sub-mode (e.g. `J0`, `K1`) |
| `INSTR` | (hardcoded) | Always `'iSHELL'` |

**Fallback keyword aliases** (tried if the primary keyword is absent):

| Primary | Fallback |
|---|---|
| `TCS_AM` | `AIRMASS` |
| `TCS_RA` | `RA` |
| `TCS_DEC` | `DEC` |
| `IRAFNAME` | `FILENAME` |
| `PASSBAND` | `GRAT` |

Missing keywords produce `nan` (numeric) or `'nan'`/`'unknown'` (string)
rather than raising exceptions.

---

## 3. Non-Linearity Flagging

Following §2.3 of the iSHELL Spextool Manual:

> "We do use the sum of the pedestal and signal reads to identify pixels that
> have counts beyond the linearity curve maximum."

`load_data()` flags pixels where:

```
ext1 + ext2  >  linearity_info['max']
```

The threshold is `LINCORMAX = 30000 DN` (Manual Table 4).  Flagged pixels
have bit `linearity_info['bit']` set in the returned `uint8` bitmask.

---

## 4. Image Normalisation

The science image returned by `load_data()` is in **DN s⁻¹**:

```
img = S / (ITIME × CO_ADDS)
```

where `S` is the signal difference from extension 0 (total DN).

---

## 5. Variance Estimation

A simple Poisson + read-noise variance is estimated (in (DN s⁻¹)²):

```
var = |S| / (gain × T²)  +  2 × (ron / gain)² / T²
```

where `T = ITIME × CO_ADDS`, `gain = GAIN_ELECTRONS_PER_DN = 1.8 e⁻/DN`,
and `ron = READNOISE_ELECTRONS = 8.0 e⁻`.

> **Note:** These detector constants are approximate and should be confirmed
> from IRTF calibration files before production use.

---

## 6. What Is Not Yet Implemented (Phase 2)

| Feature | Status | Function |
|---|---|---|
| H2RG polynomial linearity correction | Not implemented | `_correct_ishell_linearity()` |
| Reference-pixel bias subtraction | Not implemented | `_subtract_reference_pixels()` |
| Order rectification | Not implemented | `_rectify_orders()` |

---

## 7. Synthetic Fixture for Testing

Since real iSHELL raw frames are not bundled in the repository, the test suite
in `tests/test_ishell_ingestion.py` creates **synthetic MEF fixtures** using
`astropy.io.fits`:

```python
ext0 = fits.PrimaryHDU(data=sig_diff_array, header=ishell_header)
ext1 = fits.ImageHDU(data=ped_sum_array)
ext2 = fits.ImageHDU(data=sig_sum_array)
hdul = fits.HDUList([ext0, ext1, ext2])
hdul.writeto(path)
```

The synthetic header includes all keywords listed in §2 above.  Tests
verify normalisation, non-linearity flagging, pair subtraction, rotation,
and error handling.
