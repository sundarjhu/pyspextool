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

### Line identification

For each order:

1. `scipy.signal.find_peaks` detects emission-line peaks in the 1D arc spectrum
2. Each peak column is matched to the packaged `WaveCalInfo` plane-0 coarse
   wavelength grid (linear interpolation)
3. The predicted wavelength is compared to the reference line list; the nearest
   line within `match_tol_um = 0.002 µm` is accepted
4. Duplicates (two peaks claiming the same reference line) are resolved by keeping
   the match with the smallest residual

### Global fit

All accepted (col, order, ref_wavelength) triplets are assembled across all orders.
The design matrix has `(wdeg+1) × (odeg+1)` columns, one per basis function
`col^i · v^j`.  A single linear least-squares fit (`np.linalg.lstsq`) solves for
the `(wdeg+1, odeg+1)` coefficient matrix `C`.

The fit RMS is reported in nm.

---

## IdlStyle1DXDModel

The `IdlStyle1DXDModel` dataclass stores:

- `coeffs`: shape `(wdeg+1, odeg+1)` coefficient matrix
- `order_ref`: reference order number (minimum fitted order), used for `v`
- `fitted_order_numbers`: list of echelle orders with accepted arc-line matches
- `fit_rms_um`, `n_lines`, `n_orders_fit`: fit statistics

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
(Option B from the design requirements):

```python
build_rectification_indices(
    geometry,
    wavelength_func=model.as_wavelength_func(),
    fitted_order_numbers=model.fitted_order_numbers,
    wav_map=prov_map,          # for placeholder→echelle order mapping
    n_spectral=256,
    n_spatial=64,
)
```

When `wavelength_func` is provided:

- The `surface` (scaffold) parameter must be `None`
- Only orders in `fitted_order_numbers` are processed
- The callable is evaluated as `wavelength_func(col_grid, order_number)` instead of
  `surface.eval_array(order_numbers, col_grid)`
- Monotonicity checks are still applied

---

## K3 Benchmark Pipeline

```
Stage 1  — Flat tracing → FlatOrderTrace (center_poly_coeffs per order)
                │
Stage 2  — Arc tracing → arc_img
                │
Stage 3b — K3 1DXD [PRIMARY]
           extract_order_arc_spectra(arc_img, trace, wavecalinfo)
               → OrderArcSpectraSet
           fit_1dxd_wavelength_model(spectra_set, wavecalinfo, line_list)
               → IdlStyle1DXDModel
                │
Stage 6  — Rectification (driven by 1DXD model)
           build_rectification_indices(geometry,
               wavelength_func=model.as_wavelength_func(), ...)
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
| Stage 3b | — | `fit_1dxd_wavelength_model` ✓ |
| Stage 6 | `surface` parameter | `wavelength_func` parameter ✓ |

---

## Remaining Blockers to IDL K3 Parity

1. **Column normalisation** — IDL normalises columns to `[-1, +1]` within each
   order before fitting.  The current implementation uses raw column values.
   This does not affect the structure of the fit but may require re-scaling
   for exact coefficient reproduction.

2. **Iterative sigma-clipping** — IDL iteratively rejects arc-line matches based
   on residuals from the global fit.  The current implementation is single-pass.

3. **Tilt correction** — IDL applies a spectral-line tilt correction before fitting
   wavelengths.  The current implementation uses the 1D centroid (no tilt).

4. **Science extraction** — Not implemented (by design).

5. **Telluric correction and order merging** — Not implemented (by design).
