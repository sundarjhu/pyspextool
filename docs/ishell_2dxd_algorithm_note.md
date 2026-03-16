# iSHELL 2DXD Wavelength-Mapping Scaffold – Developer Note

**Status:** Provisional scaffold (not a final science-quality calibration)  
**Module:** `src/pyspextool/instruments/ishell/wavecal_2dxd.py`  
**Tests:** `tests/test_ishell_wavecal_2dxd.py`

---

## Overview

This note describes the first bridge between per-order arc-line centroid
measurements and the global 2-D echelle polynomial model ("2DXD") used by
the legacy IDL Spextool pipeline for iSHELL.

The implementation is intentionally limited to a **provisional scaffold**:
it associates traced H1 arc lines with wavelength references, fits a
provisional 2-D polynomial mapping between detector coordinates and
wavelength, and returns a structured result suitable for later
coefficient-surface fitting and full 2DXD calibration.  It does **not**
implement final rectification indices, science-quality wavelength solutions,
or interactive arc-line identification.

---

## What is Being Fit

### The provisional 2DXD polynomial

The module fits a 2-D polynomial of the form

    λ(x, m) = Σ_{i=0}^{Nd} Σ_{j=0}^{No} A_{i,j}
              · ((2·x / (n_px − 1)) − 1)^i
              · (m_home / m)^j

where:

| Symbol   | Meaning |
|----------|---------|
| x        | Detector column position (0-indexed) |
| m        | Echelle order number |
| λ        | Wavelength in µm |
| n_px     | Detector column count = 2048 |
| m_home   | Reference order from the FITS `HOMEORDR` keyword |
| Nd       | Dispersion-axis polynomial degree (`DISPDEG` keyword) |
| No       | Cross-order polynomial degree (`ORDRDEG` keyword) |
| A_{i,j}  | Fitted 2-D polynomial coefficients |

**Normalisation choices:**

- Column: `(2·x/(n_px−1)) − 1` maps column indices to `[−1, 1]`
  (Chebyshev-style normalization, standard for improving polynomial
  conditioning).
- Order: `m_home/m` is the dimensionless inverse order ratio.  This is
  physically motivated by the echelle equation `m·λ ≈ const` (for a given
  grating angle, the product of order and wavelength is approximately
  constant), so `λ ∝ 1/m` at fixed column position.

**Fitting procedure:** Ordinary least squares over all accepted arc-line
centroid measurements across all orders simultaneously.  No iterative
outlier rejection is applied.

---

## What is Matched to Reference Information

### Arc-line centroid input

The provisional fit uses the output of
`wavecal.fit_arc_line_centroids()`, which:

1. For each echelle order and each reference line in the packaged
   `*_lines.dat` file, predicts the expected pixel position from the
   pre-stored plane-0 wavelength grid.
2. Extracts a window of the plane-1 data (interpreted as the ThAr arc-lamp
   spectrum; see caveat below) centred on the predicted position.
3. Fits a Gaussian profile to measure the precise sub-pixel centroid.
4. Accepts only measurements with peak SNR ≥ `snr_min` (default 3.0) and
   centroids within the extraction window.

The result is a flat array of `(x_detector, m, λ_reference)` triples, one
per accepted centroid, spanning all orders that yielded measurements.

### Reference wavelengths

Reference wavelengths come from the packaged `*_lines.dat` files – these
are the ThAr/Ar vacuum wavelengths compiled for each iSHELL mode (e.g.
`H1_lines.dat`).  These files contain real data for H-band modes (as of the
current repository state).  The same reference wavelengths are used in the
legacy IDL pipeline.

### Arc spectrum data

The arc lamp spectrum used for centroid fitting is stored in **plane 1** of
the packaged `*_wavecalinfo.fits` data cube.  The FITS header keyword
`YUNITS = "DN / s"` is consistent with a ThAr arc flux spectrum.  However,
the exact content of this plane has **not** been formally verified against
the IDL source code; it is used on the basis of structural consistency.

---

## How This Differs from the Full 2DXD Coefficient-Surface Fit

| Aspect | This provisional scaffold | Full IDL 2DXD pipeline |
|--------|--------------------------|------------------------|
| Input arc data | 1-D centerline arc spectrum from stored calibration cube | 2-D ThAr arc-lamp frames |
| Line identification | Predicted from stored plane-0 grid; Gaussian centroid fit | Interactive/iterative; full cross-order spatial fitting |
| Outlier rejection | None | Iterative sigma-clipping |
| Tilt measurement | Placeholder zero | Measured from 2-D arc line profiles |
| Slit curvature | Not modelled | Measured from 2-D arc images |
| Coordinate convention | Verified by fitting plane-0 data (RMS < 1 nm) | IDL spextool convention (unverified against this code) |
| Spatial calibration | Linear plate-scale model | Multi-order curved slit model |
| Rectification indices | Not generated | Per-pixel wavelength/spatial index arrays |
| Science quality | Provisional, not validated | Pipeline-quality |

### Stored IDL coefficients

The packaged `*_wavecalinfo.fits` FITS headers contain a set of pre-computed
IDL 2DXD coefficients in keywords `P2W_C00` … `P2W_C{n}`.  These are read
into `TwoDXDCoefficients.coeffs_flat` and stored for reference.

**The exact evaluation convention of these stored coefficients has NOT been
verified against the IDL source code.**  Specifically, the normalization
applied to the column and order coordinates when the IDL pipeline wrote them
is unknown.  The `TwoDXDCoefficients.eval_provisional()` method implements
one plausible form (the same normalization as the fitted polynomial above)
and can be used as a sanity check, but its residuals against the actual
plane-0 data should be inspected before trusting any evaluation.

Numerical experiments show that this specific form reproduces the stored
plane-0 wavelength data with RMS residual ≲ 1 nm *when the coefficients are
re-fitted from scratch*.  Whether the stored P2W_C* values from the IDL
pipeline use the same normalization is an **unresolved question**.

---

## Data Structures Designed for Later Use

The `ProvisionalWaveCal2DXD` result dataclass is structured to be usable as
input to later stages:

- **`fitted_coeffs`** (shape `(Nd+1, No+1)`) – the provisional 2-D
  polynomial coefficient matrix, ready for downstream re-fitting over an
  extended dataset or refinement with new arc frames.
- **`centroid_data`** – the raw `(column, order, λ)` input data, preserved
  for inspection and for use in iterative refinement.
- **`per_order_wave_coeffs`** – per-order 1-D wavelength polynomials in the
  `OrderGeometrySet`-compatible format, for backward compatibility with
  extraction stages that use per-order wavelength solutions.
- **`stored_2dxd`** – the pre-stored IDL 2DXD coefficients, attached for
  comparison and eventual reverse-engineering of the IDL evaluation
  convention.
- **`rms_residual_um`** – fit RMS (in µm) against the centroid input data.
  For H1 with the real packaged calibrations, this is typically < 0.02 nm.

The coefficient matrix `fitted_coeffs[i, j]` multiplies
`x_norm^i · inv_order^j`, where `x_norm = (2x/(n_px−1))−1` and
`inv_order = m_home/m`.  This convention is fully documented in the class
docstring and is designed to match standard practices for 2-D polynomial
fitting of echelle wavelength data.

---

## What Remains Unimplemented in This PR

The following items are **intentionally deferred** to later work:

1. **Tilt measurement.**  The spectral-line tilt (how a monochromatic line's
   centroid shifts along the cross-dispersion direction) requires a 2-D
   ThAr arc-lamp image.  The packaged `*_wavecalinfo.fits` calibration
   contains only a 1-D centerline arc spectrum (plane 1).  Tilt coefficients
   remain the placeholder `[0.0]` in `OrderGeometrySet`.

2. **Slit curvature.**  Higher-order distortion of iSHELL echelle orders is
   not modelled.  `curvature_coeffs` remains `None` in all `OrderGeometry`
   objects.

3. **Final rectification indices.**  The 2-D per-pixel wavelength and
   spatial coordinate arrays needed to remap raw detector frames to
   rectified cubes have not been generated.

4. **Iterative 2DXD fitting.**  The full IDL pipeline uses an interactive
   arc-line identification loop with iterative sigma-clipping and a global
   2DXD model.  This scaffold does one pass only.

5. **IDL polynomial convention.**  The evaluation form of the stored P2W_C*
   coefficients remains unverified.  Reverse-engineering requires access to
   the IDL Spextool source or a detailed comparison of the stored values
   against independently reduced data.

6. **Wavelength uncertainty.**  Fit residuals and covariance matrices are not
   propagated into `OrderGeometrySet.wave_coeffs`.

7. **L/Lp/M-band modes.**  Out of scope per the original problem statement.

---

## File Inventory

| File | Role |
|------|------|
| `src/pyspextool/instruments/ishell/wavecal_2dxd.py` | New: provisional 2DXD scaffold |
| `src/pyspextool/instruments/ishell/calibrations.py` | Modified: `WaveCalInfo.p2w_coeffs` field + `_parse_p2w_coeffs()` |
| `src/pyspextool/instruments/ishell/__init__.py` | Modified: import `wavecal_2dxd` sub-module |
| `tests/test_ishell_wavecal_2dxd.py` | New: 79 tests for the 2DXD scaffold |
| `docs/ishell_2dxd_algorithm_note.md` | New: this document |

---

## Quick-start

```python
import warnings
from pyspextool.instruments.ishell.calibrations import (
    read_flatinfo, read_line_list, read_wavecalinfo
)
from pyspextool.instruments.ishell.wavecal_2dxd import fit_provisional_2dxd

wci = read_wavecalinfo("H1")
fi  = read_flatinfo("H1")
ll  = read_line_list("H1")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # suppress bootstrap fallback warnings
    result = fit_provisional_2dxd(wci, fi, ll)

print(f"Mode:             {result.mode}")
print(f"Centroid points:  {result.centroid_data.n_points}")
print(f"Orders fitted:    {result.n_orders_fitted}/{wci.n_orders}")
print(f"RMS residual:     {result.rms_residual_um*1000:.3f} nm")

# Evaluate provisional wavelength at a detector position
import numpy as np
x = np.array([1000.0])
m = np.array([333.0])    # home order for H1
lam = result.eval_wavelength(x, m)
print(f"λ(x=1000, m=333) ≈ {lam[0]:.4f} µm")
```

Running the tests:

```bash
pytest tests/test_ishell_wavecal_2dxd.py -v
```

---

## References

- `docs/ishell_wavecal_design_note.md` – per-order wavecal paths
  (bootstrap and measured centerline)
- `docs/ishell_geometry_design_note.md` – geometry data model
- `docs/ishell_implementation_status.md` – overall iSHELL implementation
  checklist
- `src/pyspextool/instruments/ishell/wavecal.py` – `fit_arc_line_centroids`
  and `build_geometry_from_arc_lines`
- `src/pyspextool/instruments/ishell/calibrations.py` – `WaveCalInfo`,
  `FlatInfo`, `LineList` dataclasses and readers
