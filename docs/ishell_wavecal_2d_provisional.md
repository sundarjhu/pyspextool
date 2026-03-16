# iSHELL Provisional 2-D Wavelength-Mapping Scaffold — Developer Notes

## Overview

This document describes the **provisional wavelength-identification and
wavelength-fit scaffold** for iSHELL 2DXD reduction in `pyspextool`.  The
scaffold lives in:

    src/pyspextool/instruments/ishell/wavecal_2d.py

It is the **third stage** of the 2DXD calibration scaffold, placed between
2-D arc-line tracing (`arc_tracing.py`) and the (not yet implemented) global
coefficient-surface fitting.

> **Status**: This is a development scaffold, not a finalised calibration.
> The outputs are suitable for exploring the data, validating pipeline stages,
> and preparing for the global coefficient-surface fitting step; they should
> **not** be used for science-quality wavelength calibration without further
> validation and refinement.

---

## Purpose and Scope

This module implements the first bridge between **detector-space arc lines**
and **wavelength-space information** for iSHELL H1 (and, in principle, other
J/H/K) modes.

What it does:

* Accepts an `ArcLineTraceResult` produced by 2-D arc-line tracing on real
  iSHELL frames.
* Uses the **plane-0 wavelength array** in the packaged `WaveCalInfo` data cube
  as a *coarse reference grid* to predict, for each traced line, what wavelength
  its detector-column position corresponds to.
* Matches each traced line to the nearest entry in the packaged `LineList`,
  accepting the match only if the residual between the coarse-predicted wavelength
  and the reference wavelength is within a configurable tolerance.
* Fits a per-order polynomial `wavelength_um = poly(centerline_col)` to the
  accepted matches, deriving a provisional dispersion relation from the 2-D
  arc-line positions.
* Returns a `ProvisionalWavelengthMap` recording all matches, fit coefficients,
  and residuals in a form ready for global coefficient-surface fitting.

What it does **not** do:

* Full 2DXD global coefficient-surface fitting.
* Tilt-corrected wavelength assignment (tilt data from traces is available but
  unused at this stage).
* Rectification-index generation.
* Interactive arc-line identification or iterative outlier rejection.
* Science-quality wavelength solutions.

---

## Pipeline Position

```
Flat tracing (tracing.py)
    └── FlatOrderTrace → OrderGeometrySet
            │
            ▼
Arc-line tracing (arc_tracing.py)
    └── ArcLineTraceResult (TracedArcLine objects with col(row) polynomials)
            │
            ▼
Provisional wavelength matching (wavecal_2d.py)          ← THIS DOCUMENT
    └── ProvisionalWavelengthMap (per-order wave_coeffs, match records)
            │
            ▼
Coefficient-surface fitting  (NOT YET IMPLEMENTED)
    └── global (order, col) → wavelength surface
            │
            ▼
2DXD Rectification  (NOT YET IMPLEMENTED)
    └── resample to (wavelength × spatial) grid
```

---

## Data Used

The scaffold is designed to run against the real H1-mode calibration dataset:

    data/testdata/ishell_h1_calibrations/raw/

These files are stored with Git LFS; run `git lfs pull` before opening them.

| Calibration type | Source |
|------------------|--------|
| Flat frames      | 5 QTH flat FITS files |
| Arc frames       | 2 ThAr arc FITS files |
| Line list        | `src/.../ishell/data/H1_lines.dat` |
| WaveCalInfo      | `src/.../ishell/data/H1_wavecalinfo.fits` |

---

## What Is Being Fit

For each echelle order independently, a polynomial

    wavelength_um = poly(centerline_col)

is fitted to a set of `(centroid_column, reference_wavelength)` pairs.
The polynomial follows the `numpy.polynomial.polynomial` convention:
`coeffs[k]` is the coefficient of `col**k`.

**What `centerline_col` is**: For each traced arc line, the representative
column is the **seed column** (`TracedArcLine.seed_col`) — the column at which
the line was detected in the collapsed 1-D median spectrum of the order strip.
This is a single scalar per traced line, not a full 2-D centroid.

**What is fitted**: The standard 1-D dispersion relation:
column position → vacuum wavelength in µm.  A degree-3 polynomial is
used by default; this is consistent with the `disp_degree` stored in the
packaged `WaveCalInfo` for H-band modes.

**What is NOT fitted**: No global 2-D (order × column) surface is fitted at
this stage; the per-order polynomials are fully independent.

---

## What Is Matched Directly to Reference Information

### Coarse reference grid

The column-to-wavelength reference used for matching comes from **plane 0 of
`WaveCalInfo.data`**, which is confirmed (by FITS header `XUNITS="um"` and
value ranges consistent with J/H/K bands) to store wavelengths in µm along
the order centreline.

For each order index `i`:

```
wav_array = wavecalinfo.data[i, 0, :]   # shape (n_pixels,)
x_start   = wavecalinfo.xranges[i, 0]  # start column for this order
cols_ref  = x_start + arange(n_valid)  # column at each valid pixel
predicted_wav = interp(seed_col, cols_ref, wav_array[valid])
```

This is the same coarse reference used by `build_geometry_from_arc_lines` in
`wavecal.py`, but applied to positions of **2-D traced lines** rather than
positions measured from the stored 1-D centreline arc spectrum.

### ThAr line-list matching

For each traced line, the predicted wavelength is compared against all entries
in the `LineList` for the same echelle order number.  The nearest reference
wavelength within `match_tol_um` (default: 0.005 µm = 5 nm) is accepted.

If two traced lines are matched to the same reference wavelength, only the one
with the smaller residual is kept (`_deduplicate_matches`).

### Order-number assignment (heuristic)

The `ArcLineTraceResult` assigns placeholder order numbers `0, 1, 2, …` (not
real echelle order numbers) because flat tracing alone cannot determine the
echelle order number.  The actual echelle order number (e.g. 311, 312, … for
H1 mode) is taken from `wavecalinfo.orders[i]` where `i` is the 0-based order
index.

**This is a heuristic** that relies on:

* the flat-field tracing having found the same orders as are listed in the
  `WaveCalInfo`, in the same bottom-to-top spatial order;
* no orders having been dropped or re-sorted between the two.

When the counts differ (as observed on real H1 data where flat tracing found
42 orders vs. 45 in the `WaveCalInfo`), a `RuntimeWarning` is emitted and the
shorter list is used.  The `ProvisionalWavelengthMap.order_count_mismatch`
flag records this condition.

---

## How This Differs from the Eventual Full 2DXD Coefficient-Surface Fit

| Aspect | This scaffold | Full 2DXD (not yet implemented) |
|--------|--------------|----------------------------------|
| Input arc-line positions | 2-D traced `seed_col` values from `ArcLineTraceResult` | Same (or iteratively refined centroids) |
| Tilt utilised | No — only seed_col is used | Yes — line tilt polynomials from 2-D tracing |
| Scope of fit | Independent per-order polynomial | Global surface over (order, col) |
| Model form | `wavelength = poly(col)` per order | `wavelength = surface(order, col)` |
| Outlier rejection | None (beyond the tolerance filter) | Iterative sigma-clipping against global model |
| Science quality | Provisional / diagnostic | Production-quality (after validation) |
| Order-number assignment | Heuristic (index-based) | Confirmed from global dispersion model |

---

## Key Output: `ProvisionalWavelengthMap`

```python
@dataclass
class ProvisionalWavelengthMap:
    mode: str                              # e.g. "H1"
    order_solutions: list[ProvisionalOrderSolution]
    geometry: OrderGeometrySet             # From arc tracing
    n_total_accepted: int
    match_tol_um: float                    # Tolerance used
    dispersion_degree: int                 # Poly degree used
    order_count_mismatch: bool             # True if n_traced ≠ n_wci_orders
```

### Helper methods

| Method | Purpose |
|--------|---------|
| `solved_orders` | List of orders with a valid polynomial fit |
| `n_solved_orders` | Count of solved orders |
| `collect_for_surface_fit()` | Returns `(order_numbers, cols, wavs)` arrays for global surface fitting |
| `to_geometry_set(fallback=None)` | Returns `OrderGeometrySet` with `wave_coeffs` from provisional fits |

### `collect_for_surface_fit()` usage

The primary output for the next pipeline stage is:

```python
order_numbers, cols, wavs = wav_map.collect_for_surface_fit()
# order_numbers: echelle order numbers for each accepted match
# cols: centroid columns for each accepted match
# wavs: reference wavelengths (µm) for each accepted match
```

These three equal-length arrays provide all the data points for fitting a
global 2-D surface `wavelength = surface(order, col)`.

---

## Performance on Real H1 Data

When run against the real H1 dataset (42 orders traced from flat; 45 orders
in `WaveCalInfo`):

* A `RuntimeWarning` is emitted about the order-count mismatch (42 vs. 45).
* Matching proceeds on the 42 traced orders against `wavecalinfo.orders[0:42]`
  (orders 311–352).
* Multiple orders are successfully solved (≥1 order confirmed in tests).
* Wavelengths of accepted matches fall within the H-band range (~1.49–1.80 µm).

The exact number of solved orders and accepted matches depends on the arc-line
detection parameters used in `trace_arc_lines` (prominence threshold, minimum
distance, etc.) and is not fixed here.

---

## What Remains Intentionally Unimplemented

The following items are explicitly **out of scope** for this PR:

1. **Global 2DXD coefficient-surface fitting** — fitting a surface over all
   orders simultaneously.  `collect_for_surface_fit()` provides the data;
   the fitting itself is the next stage.

2. **Tilt correction during wavelength assignment** — `TracedArcLine.poly_coeffs`
   encodes the col(row) tilt, but is not applied during matching.  All
   wavelength predictions use `seed_col`, which is the centreline column.

3. **Final rectification indices** — the interpolation arrays needed to
   resample raw 2-D orders onto a (wavelength × spatial) grid are not produced.

4. **Final 2-D wavecal/spatcal image construction** — the output is a Python
   data structure, not a FITS calibration file.

5. **Science-quality wavelength solution** — the result is labelled
   "provisional" throughout because:
   - the order-number assignment is a heuristic,
   - the tilt is unused,
   - there is no iterative outlier rejection,
   - and no global dispersion model is imposed.

6. **Confirmed plane-1 semantics of `WaveCalInfo`** — the module docstring in
   `wavecal.py` notes that plane 1 is *interpreted* as an arc spectrum but
   this interpretation has not been verified against the IDL source.  This
   module uses only plane 0 (confirmed wavelengths).

---

## Files Changed

| File | Change |
|------|--------|
| `src/pyspextool/instruments/ishell/wavecal_2d.py` | **New** — provisional wavelength-mapping scaffold |
| `src/pyspextool/instruments/ishell/__init__.py` | Added `wavecal_2d` import |
| `tests/test_ishell_wavecal_2d.py` | **New** — unit tests and H1 smoke test |
| `docs/ishell_wavecal_2d_provisional.md` | **New** — this document |

---

## Testing

```bash
# Run all wavecal_2d tests (synthetic data only, ~3 seconds):
pytest tests/test_ishell_wavecal_2d.py -v

# Run including the real-data smoke test (requires git lfs pull):
pytest tests/test_ishell_wavecal_2d.py -v -m "not skipif"
```

The test suite covers:

* `ProvisionalLineMatch` and `ProvisionalOrderSolution` construction.
* `fit_provisional_wavelength_map` on synthetic data with known answers:
  - correct return type and attributes,
  - known wavelengths recovered within tolerance,
  - fit RMS below the match tolerance,
  - `collect_for_surface_fit()` array shapes and values,
  - `to_geometry_set()` output,
  - order-count mismatch warning,
  - graceful handling of empty line lists and empty traced-order lists.
* `ProvisionalWavelengthMap` helper methods.
* Smoke test on real H1 data (skipped if LFS files absent):
  - function completes without error,
  - at least one order is solved,
  - wavelengths are in the H band,
  - `to_geometry_set()` returns a valid `OrderGeometrySet`.
