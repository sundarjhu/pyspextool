# iSHELL Coefficient-Surface Refinement Scaffold — Developer Notes

## Overview

This document describes the **coefficient-surface refinement scaffold** for
iSHELL 2DXD reduction in `pyspextool`.  The scaffold lives in:

    src/pyspextool/instruments/ishell/wavecal_2d_refine.py

It is the **fifth stage** of the 2DXD calibration scaffold, placed after the
global wavelength surface fit (`wavecal_2d_surface.py`).

> **Status**: This is a development scaffold, not a finalised calibration.
> The outputs are suitable for exploring the data and validating the
> coefficient-surface structure; they should **not** be used for
> science-quality wavelength calibration without further validation and
> refinement.

---

## Purpose and Scope

This module implements the two-level fit structure that mirrors the IDL
Spextool 2DXD calibration approach (Cushing et al. 2004).

What it does:

* Accepts a `ProvisionalWavelengthMap` (Stage-3 output from `wavecal_2d.py`).
* For each echelle order with at least `min_lines_per_order` accepted arc-line
  matches, fits a 1-D polynomial:

      wavelength_um(col) ≈ Σ_k  a_k · col^k

* For each coefficient index *k*, collects the per-order coefficient values
  `a_k(order)` and fits them with a low-degree polynomial in the normalized
  inverse-order coordinate `v = order_ref / order_number`:

      a_k(order) ≈ Σ_j  d_{k,j} · v^j

* Returns a `RefinedCoefficientSurface` dataclass storing the smoothed
  coefficient surface, per-order fits, and residual statistics.
* Provides `eval(order, col)` and `eval_array(orders, cols)` helpers to
  evaluate the smoothed surface at arbitrary detector positions.

What it does **not** do:

* Normalization of per-order column range (the IDL convention normalizes
  columns to `[-1, +1]` within each order; here raw column units are used).
* Iterative sigma-clipping or outlier rejection.
* Full IDL coefficient-index compatibility.
* Rectification-index generation.
* Science-quality wavelength solutions.

---

## Pipeline Position

```
Flat tracing (tracing.py)
    └── FlatOrderTrace → OrderGeometrySet
            │
            ▼
Arc-line tracing (arc_tracing.py)
    └── ArcLineTraceResult
            │
            ▼
Provisional wavelength matching (wavecal_2d.py)
    └── ProvisionalWavelengthMap
            │
            ▼
Global wavelength surface (wavecal_2d_surface.py)   [Stage 4]
    └── GlobalSurfaceFitResult
            │
            ▼
Coefficient-surface refinement (wavecal_2d_refine.py)   [Stage 5, THIS MODULE]
    └── RefinedCoefficientSurface
            │
            ▼
2DXD Rectification (NOT YET IMPLEMENTED)
```

---

## How This Differs from Stage 4 (`wavecal_2d_surface.py`)

### Stage 4 — Direct global surface

Stage 4 fits a single 2-D polynomial directly in `(order_number, col)` space:

    wavelength_um ≈ Σ_{i,j}  c_{i,j} · u(col)^i · v(order)^j

where `u = (col − col_center) / col_half` and `v = order_ref / order_number`.
This is a joint fit over all orders and all matched lines simultaneously.

### Stage 5 — Coefficient surface

Stage 5 instead fits **per-order** 1-D polynomials in `col`:

    wavelength_um(col; order) ≈ Σ_k  a_k(order) · col^k

and then models how each coefficient `a_k` depends on order:

    a_k(order) ≈ Σ_j  d_{k,j} · v^j     (v = order_ref / order_number)

The implicit 2-D surface is:

    wavelength_um(col, order) = Σ_k  [Σ_j d_{k,j} · v^j] · col^k

Both representations can approximate the same function, but the
coefficient-surface form:

* Makes the per-order dispersion structure more transparent.
* Allows inspection of how each coefficient varies across orders.
* More closely mirrors the IDL Spextool 2DXD approach.

---

## How This Approximates the IDL Spextool 2DXD Approach

The IDL 2DXD calibration (Cushing et al. 2004) fits wavelength-dispersion
polynomials per echelle order, then models the *coefficients* as smooth
functions of order to enforce cross-order consistency.  The coefficient
surface is then used to build a rectification map.

This module implements the same two-level structure.

### Differences from the full IDL solution

| Aspect | IDL Spextool | This scaffold |
|--------|-------------|---------------|
| Column normalization | Normalized to `[-1, +1]` within each order | Raw column units |
| Order-dependence basis | Not fully documented publicly | `v = order_ref / order_number` |
| Outlier rejection | Iterative sigma-clipping | None (single-pass OLS) |
| Coefficient indexing | Specific `WDEG`/`ODEG` layout | Arbitrary |
| Rectification map | Generated | Not generated |
| Science quality | Yes | No (provisional scaffold) |

---

## Normalization Basis

The order-dependence fit uses `v = order_ref / order_number` (normalized
inverse order), matching the Stage-4 convention.  This is physically
motivated by the echelle grating equation:

    m · λ ≈ const    (at the blaze angle)

so `λ ∝ 1/m` at fixed column.  A polynomial in `1/m` captures the
leading-order inter-order wavelength variation with a low-degree expansion.

---

## Public API

### `fit_refined_coefficient_surface(wav_map, *, disp_degree=3, order_smooth_degree=2, min_lines_per_order=2)`

Main entry point.  Accepts a `ProvisionalWavelengthMap` and returns a
`RefinedCoefficientSurface`.

Parameters:

- `disp_degree` — polynomial degree for each per-order `wavelength(col)` fit.
  Automatically reduced if an order has too few matched lines.
- `order_smooth_degree` — polynomial degree for the coefficient
  order-dependence fit.  Automatically reduced if too few orders are available.
- `min_lines_per_order` — minimum accepted matches required for an order to
  be included.

### `RefinedCoefficientSurface`

Result dataclass.  Key fields:

- `order_smooth_coeffs` — shape `(disp_degree+1, order_smooth_degree+1)`.
  `order_smooth_coeffs[k, j]` is the *j*-th polynomial coefficient in the
  fit of `a_k` as a function of `v`.
- `per_order_fits` — list of `_PerOrderFit` records, one per fitted order.
- `per_order_rms_um` — RMS residual (µm) for each per-order polynomial fit.
- `smooth_rms_um` — RMS residual (µm) of the order-smoothness fit for each
  coefficient index *k*.
- `n_orders_fit`, `n_points_total`, `n_orders_skipped`.

Helper methods:

- `eval(order_number, col)` — predict wavelength at a single point.
- `eval_array(order_numbers, cols)` — predict wavelengths at arrays of points.

---

## What Remains Unimplemented

The following steps are still needed for a full 2DXD solution:

1. **Column normalization per order** — normalize `col` to `[-1, +1]` within
   each order's column range before fitting the per-order polynomial.  This
   is the IDL convention and improves numerical conditioning.

2. **Iterative sigma-clipping** — re-fit after removing outlier arc lines
   (those with large residuals from the per-order or smoothness fits).

3. **Full IDL coefficient-index compatibility** — adopt the exact `WDEG` and
   `ODEG` parameterization from the IDL code.

4. **Rectification-index generation** — build a look-up table mapping every
   detector pixel to a rectified `(wavelength, slit-position)` coordinate.

5. **Science-quality wavelength solution** — validated against known object
   spectra or laboratory wavelength atlases.

6. **Order-number validation** — the echelle order numbers are still assigned
   heuristically (see `wavecal_2d.py` module docstring).  Until validated,
   the order numbers should be treated as nominal labels.

---

## Assumptions and Limitations

* Per-order polynomials are fitted in **raw detector-column units**.
  This is numerically less stable than normalized columns for high-degree fits.

* The order-dependence fit is a single-pass ordinary least-squares.
  Mis-identified arc lines can noticeably bias the smoothness fit.

* Orders with fewer than `min_lines_per_order` accepted matches are excluded
  from the fit.  If many orders are excluded the smoothness fit may be
  poorly constrained.

* The echelle order numbers are inherited from the `ProvisionalWavelengthMap`
  and should be treated as provisional until validated.

---

## See Also

- `docs/ishell_wavecal_2d_provisional.md` — Stage-3 provisional wavelength mapping.
- `docs/ishell_2dxd_notes.md` — full pipeline overview.
- `docs/ishell_implementation_status.md` — current implementation status.
- Cushing, M. C., Vacca, W. D., & Rayner, J. T. 2004, PASP, 116, 362 — IDL
  Spextool paper describing the 2DXD calibration approach.
