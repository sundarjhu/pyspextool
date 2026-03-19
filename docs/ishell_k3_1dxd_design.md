# iSHELL K3 1DXD Wavelength Calibration — Design Document

This document describes the design of the IDL-style K3 1DXD wavelength
calibration implemented in `wavecal_k3_idlstyle.py`.

---

## Overview

The K3 benchmark uses a **global 1DXD polynomial model** as the primary source
of truth for wavelength calibration.  This model is fit to arc-line identifications
obtained from 1D spectra extracted using the traced flat-field order geometry.

The IDL Spextool 2DXD approach fits:

```
λ(col, order) = Σ_{i=0}^{wdeg} Σ_{j=0}^{odeg}  C_{i,j} · col^i · v^j
```

where `v = order_ref / order` is the normalised inverse-order coordinate,
physically motivated by the echelle grating equation `m·λ ≈ const`.

For K3, the IDL defaults are **wdeg = 3**, **odeg = 2**.

---

## Stage 1: 1D Extraction Using Traced Geometry

**Function:** `extract_order_arc_spectra(arc_img, trace, wavecalinfo, aperture_half_width=3)`

### Why not estimate from the arc image?

Estimating the centre row from the arc image (e.g. via cross-dispersion median +
argmax) is unreliable because:

- Arc lines are sparse; many columns have no emission
- The brightest arc line may not be at the order centre
- The estimate changes between arc exposures

### What we do instead

For each echelle order, the **traced centre-line polynomial** from the flat-field
Stage 1 result (`FlatOrderTrace.center_poly_coeffs`) is evaluated at every detector
column.  A symmetric aperture of `±aperture_half_width` pixels is averaged at each
column to produce the 1D arc spectrum.

```
for col in [col_start, …, col_end]:
    row_c = polyval(col, trace.center_poly_coeffs[i])
    flux[col] = mean(arc_img[row_c - hw : row_c + hw + 1, col])
```

### Properties

- **Uses flat-field geometry** — same traced solution used for order extraction
- **Deterministic** — no dependence on arc image statistics
- **Column-aligned** — no resampling; `flux[k]` corresponds to `col_start + k`
- **Robust** — averaging over a small aperture suppresses hot pixels and cosmic rays

### Column range

The valid column range for each order is taken from `wavecalinfo.xranges`, which
stores the column bounds derived from the packaged `*_wavecalinfo.fits`.

---

## Stage 2: Global 1DXD Polynomial Fit

**Function:** `fit_1dxd_wavelength_model(spectra_set, wavecalinfo, line_list, wdeg=3, odeg=2)`

### Step 1: Per-order cross-correlation

For each order, a **synthetic reference comb** is constructed from the expected
column positions of all reference arc lines (predicted by the coarse `WaveCalInfo`
grid).  The comb consists of unit Gaussians (FWHM ≈ 3 px) at each reference-line
column.

The extracted 1D spectrum is **cross-correlated** against this comb using
`scipy.signal.correlate`.  The peak of the cross-correlation within `±50 px`
is located (sub-pixel via parabolic fitting) to obtain the per-order column
shift `xcorr_shift_px`.

This shift is stored in `OrderMatchStats.xcorr_shift_px` and applied to the
coarse reference column grid before peak matching:

```
shifted_coarse_cols = coarse_cols + xcorr_shift
```

A positive shift means the extracted spectrum is shifted right relative to the
reference; subtracting the shift from the predicted columns compensates.

### Step 2: Peak detection

`scipy.signal.find_peaks` detects emission-line peaks in the (shift-aligned)
arc spectrum with:
- `prominence ≥ 50.0` counts (configurable)
- `distance ≥ 5` pixels between peaks

### Step 3: Line matching

For each peak column, a wavelength is predicted by linear interpolation on the
shift-corrected coarse grid.  The nearest reference line within
`match_tol_um = 0.002 µm` is accepted.  Duplicates are resolved by residual.

### Step 4: Global fit

All accepted (col, order, ref_wavelength) triplets are assembled across all orders.
The design matrix has `(wdeg+1) × (odeg+1)` columns, one per basis function
`col^i · v^j`.  A single linear least-squares fit (`np.linalg.lstsq`) solves for
the `(wdeg+1, odeg+1)` coefficient matrix `C`.

### Step 5: Iterative sigma clipping

After the initial fit, residuals are computed for all matched points.  Any point
whose `|residual|` exceeds `sigma_thresh × rms_residual` (default `sigma_thresh=3`)
is rejected.  The fit is rerun on the surviving points.  This is repeated until
convergence or `max_sigma_iter=5` iterations.

The final `IdlStyle1DXDModel` stores:

- `accepted_mask` — boolean array, shape `(n_lines_total,)`
- `n_lines` — accepted count (after clipping)
- `n_lines_total` — all matched points (before clipping)
- `n_lines_rejected` — `n_lines_total - n_lines`
- `fit_rms_um` — RMS on accepted points
- `median_residual_um` — median residual on accepted points
- `per_order_stats` — list of `OrderMatchStats` (one per order)

---

## IdlStyle1DXDModel

The `IdlStyle1DXDModel` dataclass stores:

- `coeffs`: shape `(wdeg+1, odeg+1)` coefficient matrix
- `order_ref`: reference order number (minimum fitted order), used for `v`
- `fitted_order_numbers`: list of echelle orders with accepted arc-line matches
- `fit_rms_um`, `n_lines`, `n_lines_total`, `n_lines_rejected`, `n_orders_fit`
- `accepted_mask`, `median_residual_um`
- `per_order_stats`: list of `OrderMatchStats`

### Evaluation

```python
model.eval(col, order)           # scalar
model.eval_array(cols, orders)   # vectorised

# For rectification:
func = model.as_wavelength_func()  # callable(cols_array, order_number_scalar)
```

---

## Integration with Rectification

`build_rectification_indices` accepts an optional `wavelength_func` callable
plus a `geom_order_map` for direct geometry→echelle order resolution without
the scaffold `prov_map`:

```python
# Build geom_order_map directly from wavecalinfo.orders (no scaffold dependency)
geom_order_map = {i: int(wavecalinfo.orders[i]) for i in range(len(wavecalinfo.orders))}

build_rectification_indices(
    geometry,
    wavelength_func=model.as_wavelength_func(),
    fitted_order_numbers=model.fitted_order_numbers,
    geom_order_map=geom_order_map,   # replaces wav_map for K3 path
    n_spectral=256,
    n_spatial=64,
)
```

When `wavelength_func` is provided:

- The `surface` (scaffold) parameter must be `None`
- Only orders in `fitted_order_numbers` are processed
- Order labels in `RectificationIndexOrder.order` are **real echelle order numbers**
  (not placeholder geometry indices)
- Monotonicity checks are still applied

### Order label consistency

Previously `RectificationIndexOrder.order` contained the **geometry placeholder
index** (`geom.order`).  After this fix it always contains the **real echelle
order number** (`echelle_order`) for both scaffold and 1DXD paths.

---

## K3 Benchmark Pipeline

```
Stage 1  — Flat tracing → FlatOrderTrace (center_poly_coeffs per order)
                │
Stage 2  — Arc tracing → arc_img
                │
Stage 3b — K3 1DXD [PRIMARY]
           1. extract_order_arc_spectra(arc_img, trace, wavecalinfo)
              → OrderArcSpectraSet
           2. fit_1dxd_wavelength_model(spectra_set, wavecalinfo, line_list)
              → xcorr shift per order
              → peak detection with shift-corrected reference
              → global (3,2) polynomial fit
              → iterative sigma clipping
              → IdlStyle1DXDModel
                │
Stage 6  — Rectification (driven by 1DXD model, geom_order_map from wavecalinfo)
           build_rectification_indices(geometry,
               wavelength_func=model.as_wavelength_func(),
               geom_order_map=geom_order_map,  ← no scaffold prov_map needed
               ...)
                │
Stage 7  — Rectified order images
Stage 8  — Calibration FITS (wavecal, spatcal)
```

Stages 3–5 (scaffold) run for reference and diagnostics but do **NOT** drive
K3 results.

---

## Scaffold vs K3 Path

| Stage | Scaffold | K3 primary |
|-------|----------|------------|
| Stage 3 | `fit_provisional_wavelength_map` | (reference only) |
| Stage 4 | `fit_global_wavelength_surface` | (reference only) |
| Stage 5 | `fit_refined_coefficient_surface` | (reference only) |
| Stage 3b | — | `fit_1dxd_wavelength_model` ✓ (xcorr + σ-clip) |
| Stage 6 | `surface` parameter | `wavelength_func` + `geom_order_map` ✓ |

---

## What still prevents full IDL parity

1. **Column normalisation** — IDL normalises columns to `[-1, +1]` within each
   order before fitting.  The current implementation uses raw column values.
   This does not affect the structure of the fit but coefficients will differ
   from IDL values.

2. **Centroid refinement** — IDL refines peak positions by fitting a Gaussian
   or parabola to each peak.  The current code uses the integer `argmax` index
   from `scipy.signal.find_peaks` (no sub-pixel centroiding).

3. **Tilt correction** — IDL applies a spectral-line tilt correction before
   assigning wavelengths to 1D peak positions.  Not implemented.

4. **Per-iteration xcorr update** — IDL may update the cross-correlation shift
   after each sigma-clipping iteration as the model improves.  The current
   implementation computes the xcorr shift once before peak-finding and does
   not update it during sigma-clipping.

5. **Science extraction** — Not implemented (by design).

6. **Telluric correction and order merging** — Not implemented (by design).
