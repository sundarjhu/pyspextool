# iSHELL K3 1DXD Wavelength Calibration — Design Note

This document describes the architectural design of the new K3 benchmark
wavelength-calibration module (`wavecal_k3_idlstyle.py`) and explains why it
was introduced as a replacement for the old per-order scaffold path.

---

## Background

The original Python iSHELL scaffold used a 2D-seed-based, per-order wavelength
fitting approach (Stages 3–8 of the benchmark script).  While this path
produced outputs that could feed into the downstream rectification chain, it
had several structural weaknesses relative to the IDL Spextool approach:

- **Per-order fits** are underconstrained in orders with few arc lines.
- **Per-order degree reduction** produced inconsistent polynomial degrees across
  orders, making the global surface poorly conditioned.
- **Direct 2-D seed matching** is sensitive to the accuracy of the coarse
  reference grid and provides no independent check of the line positions.
- **No cross-correlation step** meant that any bulk offset between the real arc
  image and the stored reference went undetected and uncompensated.
- **No sigma-clipping** in the primary fits meant that a few mis-identified
  lines could significantly bias individual orders.

---

## IDL Spextool sequence (reproduced in Stage 2b)

The IDL Spextool pipeline (Cushing et al. 2004) uses the following sequence
for the K3 wavelength calibration:

1. Extract a 1-D arc spectrum along the order centre-line.
2. Cross-correlate with the stored reference arc spectrum to find the pixel
   offset.
3. Use the shifted expected line positions to identify arc lines in 1-D and
   measure their centroids.
4. Fit a **global 1DXD** wavelength solution across all orders simultaneously:

       lambda = f(column, order)

   with fixed polynomial degrees (`lambda_degree=3`, `order_degree=2`).
5. Apply iterative sigma-clipping to reject outliers.

Stage 2b in the Python benchmark (`run_ishell_k3_example.py`) reproduces this
sequence using the `wavecal_k3_idlstyle.py` module.

---

## Module structure

```
src/pyspextool/instruments/ishell/wavecal_k3_idlstyle.py
```

### Public API (dataclasses)

| Class | Purpose |
|-------|---------|
| `OrderArcSpectrum` | Extracted 1-D arc spectrum for one order |
| `CrossCorrelationResult` | Cross-correlation result per order |
| `LineIdentResult` | 1-D line identifications for one order |
| `GlobalFit1DXD` | Global 1DXD fit result |
| `K3CalibDiagnostics` | Full diagnostics summary |

### Public API (functions)

| Function | Step | Description |
|----------|------|-------------|
| `extract_order_arc_spectra` | 1 | Extract 1-D arc spectra from detector image |
| `cross_correlate_with_reference` | 2 | Estimate per-order pixel shifts |
| `identify_lines_1d` | 3 | Identify and centroid arc lines in 1-D |
| `fit_global_1dxd` | 4+5 | Global 1DXD fit with sigma-clipping |
| `run_k3_1dxd_wavecal` | All | Top-level driver for the full sequence |

### Fixed constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `K3_LAMBDA_DEGREE` | 3 | Polynomial degree in detector column |
| `K3_ORDER_DEGREE` | 2 | Polynomial degree in echelle order number |

These are not selected automatically; they reflect the K3 benchmark
configuration from the IDL Spextool manual.

---

## Coordinate normalization

The global 1DXD fit uses the same normalization as the existing
`wavecal_2d_surface.py` module:

- **Column**: `u = (col - col_center) / col_half` maps observed columns to [-1, +1].
- **Order**: `v = order_ref / order_number` is the normalized inverse order
  (physically motivated by the echelle grating equation `m·λ ≈ const`).

The reference order `order_ref` is the minimum order number in the data, so
`v = 1` for that order and `v < 1` for higher-numbered orders.

---

## 1-D arc spectrum extraction

The 1-D spectrum is extracted as follows:

1. For each order, compute a cross-dispersion median profile of the arc image
   in the order's valid column range.
2. Smooth the profile with a 5-pixel uniform filter to suppress cosmic rays.
3. Identify the peak row (order centre row).
4. Average `n_rows_avg=5` rows centred on the peak row to produce the 1-D
   spectrum.

This is a simple but deterministic and testable method.  The result is a
representative 1-D spectrum that captures the arc-line flux without requiring
a full spatial profile fit.

---

## Cross-correlation

For each order:

1. The extracted spectrum and the reference arc spectrum (plane 1 of
   `K3_wavecalinfo.fits`) are normalised to zero mean and unit variance.
2. A full cross-correlation is computed using `scipy.signal.correlate`.
3. The lag at the peak of the cross-correlation within `[-max_lag, +max_lag]`
   is taken as the pixel shift.

A positive shift means the extracted spectrum is shifted to larger column
numbers relative to the reference.  The shifted expected positions are used
in the line-identification step.

---

## Line identification and centroiding

For each order:

1. Reference line list entries for this order are retrieved from the packaged
   `K3_lines.dat` file.
2. Each reference wavelength is converted to an expected column position by
   interpolation on the reference wavelength grid (plane 0 of
   `K3_wavecalinfo.fits`).
3. The cross-correlation shift is applied to the expected position.
4. A Gaussian is fitted (or flux-weighted centroid used as fallback) to a
   window of `window_half_pixels=8` pixels around the expected position.
5. The centroid is accepted if its peak SNR exceeds `min_snr=3.0` and the
   position is within the window.
6. A minimum column separation filter (`min_col_separation=5.0` pixels)
   removes near-duplicate centroids.

---

## Global 1DXD fit

All accepted centroids from all orders are combined into a single fitting
problem:

```
lambda_um = Σ_{i,j} c_{i,j} · u(col)^i · v(order)^j
```

where `0 ≤ i ≤ 3` and `0 ≤ j ≤ 2`, giving 12 free parameters total.

The fit is performed by ordinary least squares via `numpy.linalg.lstsq`,
with iterative sigma-clipping (3σ threshold, up to 5 iterations).

The fit result is stored in `GlobalFit1DXD`, which exposes:
- `eval(col, order)` — predict wavelength at a single point
- `eval_array(cols, order)` — predict wavelengths along an order
- `per_order_rms_nm()` — per-order RMS in nm

---

## QA outputs

Stage 2b produces a new QA plot `{prefix}_1dxd_qa.png` with four panels:

1. Residuals vs order number (accepted=blue dots, rejected=red ×).
2. Residuals vs column.
3. Histogram of accepted residuals.
4. Per-order accepted-line count (bar chart with RMS labels).

This is analogous in spirit to the IDL manual's 1DXD residual plot (Figure 3).

---

## Diagnostics export

When `--export-diagnostics` is passed, the exported CSV/JSON file includes
additional 1DXD columns:

| Column | Description |
|--------|-------------|
| `xcorr_shift` | Per-order cross-correlation pixel shift |
| `n_1dxd_candidate` | Candidate 1-D line count |
| `n_1dxd_accepted` | Accepted 1-D line count |
| `n_1dxd_rejected` | Rejected 1-D line count |
| `rms_1dxd_nm` | Per-order 1DXD fit RMS in nm |
| `in_1dxd_fit` | Whether this order contributed to the global fit |

---

## What remains different from IDL

| Aspect | IDL Spextool | Python Stage 2b |
|--------|-------------|-----------------|
| Arc image source | Real 2-D ThAr frames | Same (real K3 arc frames) |
| Centre-row geometry | Flat-field traced centre line | Peak of cross-dispersion median |
| Reference spectrum | IDL-stored calibrated spectrum | Plane 1 of `K3_wavecalinfo.fits` (partially confirmed) |
| Sub-pixel cross-correlation | Parabolic peak interpolation | Integer-lag peak (no sub-pixel) |
| Line centroiding | Gaussian + poly background | Gaussian with constant background |
| Global fit normalization | IDL-specific normalization | `u = (col-center)/half`, `v = order_ref/order` |
| Iteration strategy | Full interactive IDL GUI | Automatic σ-clipping |
| Output products | Full IDL wavecal FITS | Python `GlobalFit1DXD` dataclass |

---

## Blockers to matching IDL manual K3 QA

1. **Reference spectrum quality**: Plane 1 of `K3_wavecalinfo.fits` may not
   exactly match the IDL reference arc used for cross-correlation.  If the
   reference arc spectrum has a different scaling or background, the
   cross-correlation shifts may be unreliable.

2. **Sub-pixel cross-correlation**: The current implementation uses the
   integer-lag peak.  A parabolic interpolation around the peak would give
   sub-pixel accuracy.

3. **Line list coverage**: If the K3 line list (`K3_lines.dat`) has fewer
   entries than the IDL line atlas, fewer lines will be identified per order.

4. **Centre-row estimation**: The current method finds the global maximum row
   in the cross-dispersion median, which may not correspond to the order
   centre in all cases.  Using the flat-field traced centre polynomial would
   be more robust.

5. **Rectification path**: The downstream rectification (Stages 6–8) still
   uses the old scaffold wavecal path.  To fully replace the scaffold path,
   the 1DXD global fit result would need to feed into the rectification
   indices.

---

## Files changed by this rewrite

| File | Change |
|------|--------|
| `src/pyspextool/instruments/ishell/wavecal_k3_idlstyle.py` | **New module** — full 1DXD K3 calibration pipeline |
| `scripts/run_ishell_k3_example.py` | Added Stage 2b, `_print_1dxd_summary`, `_plot_1dxd_qa`, `_print_1dxd_diagnostics_table`; updated `_export_diagnostics` |
| `tests/test_ishell_k3_example.py` | Added 8 new test classes for the 1DXD module |
| `docs/ishell_k3_example.md` | Updated stages table, added 1DXD path description, updated mismatches section |
| `docs/ishell_k3_1dxd_design.md` | **This file** — architectural design note |
