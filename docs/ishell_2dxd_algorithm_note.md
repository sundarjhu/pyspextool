# iSHELL 2DXD Wavelength-Mapping Scaffold – Developer Note

**Status:** Provisional scaffold (not a final science-quality calibration)
**Modules:**
- `src/pyspextool/instruments/ishell/arc_tracing.py` – arc-line tracing scaffold
- `src/pyspextool/instruments/ishell/wavecal_2dxd.py` – provisional 2DXD fit
**Tests:** `tests/test_ishell_wavecal_2dxd.py`

---

## Overview

This note describes the first bridge between per-order arc-line positions
measured in detector space and the global 2-D echelle polynomial model
("2DXD") used by the legacy IDL Spextool pipeline for iSHELL.

The implementation is intentionally limited to a **provisional scaffold**: it
produces traced arc-line centroid positions from available calibration data,
associates those positions with wavelength references from the packaged
line list, fits a provisional 2-D polynomial mapping between detector
coordinates and wavelength, and returns a structured result suitable for
later coefficient-surface fitting and full 2DXD calibration.  It does
**not** implement final rectification indices, science-quality wavelength
solutions, or interactive arc-line identification.

---

## Corrected Data Flow

The corrected path starts from the **arc-tracing scaffold** (`arc_tracing.py`)
rather than directly from the packaged centroid-fitting functions in
`wavecal.py`.

```
packaged H1_wavecalinfo.fits (plane-1 arc spectrum proxy)  [current]
  OR
real 2-D raw arc FITS frame (future; see §4 below)
    ↓
arc_tracing.build_arc_trace_result_from_wavecalinfo()  [proxy]
  or arc_tracing.trace_arc_lines_from_2d_image()       [future stub]
    ↓
ArcTraceResult  (list of TracedArcLine objects)
    ↓  centroid_col is the representative detector coordinate
wavecal_2dxd.collect_centroid_data_from_arc_traces()
    ↓
ArcLineCentroidData  (flat x, m, λ arrays)
    ↓
wavecal_2dxd.fit_provisional_2dxd()
    ↓
ProvisionalWaveCal2DXD  (fitted 2-D polynomial + per-order 1-D coeffs)
```

The previous version used `wavecal.fit_arc_line_centroids()` directly,
bypassing the arc-tracing layer.  The corrected version routes all data
through `arc_tracing.py` so the traced-line data structure is an explicit
intermediate result.

---

## Data Source: What Is Actually Used

**Current proxy path (Phase 0):**

The function `build_arc_trace_result_from_wavecalinfo()` uses **plane 1 of
the packaged `*_wavecalinfo.fits` data cube** as a 1-D centerline arc-lamp
spectrum.  The FITS header keyword `YUNITS = "DN / s"` is consistent with
arc-lamp flux, but the exact content of plane 1 has **not** been formally
verified against the IDL source code.

- This is explicitly a **provisional substitute** for a real 2-D arc image.
- The tracing algorithm (Gaussian centroid fitting) is identical to the
  production path; only the data source differs.
- Each `TracedArcLine` carries a `source = "wavecalinfo_plane1"` label.
- The `ArcLineCentroidData` produced carries
  `source = "arc_tracing:wavecalinfo_plane1"`.

**Intended future path:**

When real 2-D raw arc-lamp FITS frames become available in
`data/testdata/ishell_h1_calibrations/raw/`, the function
`trace_arc_lines_from_2d_image()` should be implemented to:

1. Extract per-order sub-images from the 2-D detector frame.
2. Fit Gaussian profiles at multiple spatial rows.
3. Return `TracedArcLine` objects with both centroid column and tilt info.

Until then, `trace_arc_lines_from_2d_image()` raises `NotImplementedError`.

---

## What Is Matched to Reference Information

For each order:

1. Each entry in the packaged ThAr line list (e.g. `H_lines.dat`) with a
   wavelength in the order's wavelength range is a **candidate**.
2. The candidate's expected column position is predicted from the stored
   plane-0 wavelength grid (linear interpolation).
3. A window of the plane-1 arc spectrum around that predicted position is
   extracted.
4. A Gaussian profile is fit to measure the precise **centroid column**
   (`centroid_col`).
5. Lines with SNR < 3.0 (default), poor centroid fit, or blended centroids
   (within 2 px) are rejected.

The `centroid_col` of each accepted `TracedArcLine` is the **representative
detector coordinate** that is matched to the known vacuum wavelength from
the line list.  This is the quantity used in the 2-D polynomial fit.

---

## What Is Being Fit

### The provisional 2DXD polynomial

```
λ(x, m) = Σ_{i=0}^{Nd} Σ_{j=0}^{No} A_{i,j}
           · ((2·x / (n_px − 1)) − 1)^i
           · (m_home / m)^j
```

| Symbol  | Meaning |
|---------|---------|
| x       | Detector column position (0-indexed); specifically `centroid_col` of each `TracedArcLine` |
| m       | Echelle order number |
| λ       | Wavelength in µm |
| n_px    | Detector column count = 2048 |
| m_home  | Reference order from `HOMEORDR` FITS keyword |
| Nd      | Dispersion-axis polynomial degree (`DISPDEG` keyword) |
| No      | Cross-order polynomial degree (`ORDRDEG` keyword) |
| A_{i,j} | Fitted 2-D polynomial coefficients (ordinary least squares) |

**Normalization:**
- Column: `(2·x / (n_px − 1)) − 1` maps column indices to `[−1, 1]`
- Order: `m_home / m` is dimensionless and motivated by the echelle equation
  `m·λ ≈ const` (so the 0-th order term `A_{0,0}` is approximately the
  constant `m_home · λ_center`)

**Fit quality (H1 proxy path):** OLS fit to 157 centroids across 42 orders
gives RMS residual ≈ 0.018 nm.  This residual reflects how well the
polynomial represents the centroid positions; it does not include systematic
errors from the packaged line list, the plane-1 data interpretation, or the
1-D (single-row) arc extraction.

---

## Stored IDL 2DXD Coefficients (Reference Role)

The packaged `*_wavecalinfo.fits` files carry a pre-computed 2DXD polynomial
in FITS header keywords `P2W_C00` … `P2W_Cnn`.

These are **demoted to a reference / comparison role** in this scaffold.
The provisional fit is computed fresh from traced arc-line positions, not
from the stored coefficients.  The stored coefficients are:

1. Read by `read_stored_2dxd_coeffs()` into a `TwoDXDCoefficients` dataclass.
2. Attached to the `ProvisionalWaveCal2DXD` result in `stored_2dxd`.
3. Accessible via `result.stored_2dxd.eval_provisional(x, m)`.

**Warning:** The exact evaluation convention of the stored coefficients (the
coordinate normalization used when the IDL pipeline wrote them) has **not**
been verified against the IDL source code.  `eval_provisional()` is offered
for comparison only.

---

## How This Differs from the Eventual Full 2DXD Solution

| Feature | This scaffold | Full 2DXD |
|---------|--------------|-----------|
| Arc data source | Plane-1 1-D centerline proxy | Real 2-D arc detector images |
| Line tracing | 1-D Gaussian centroid at centerline | 2-D spatial profile at multiple rows |
| Tilt measurement | Not implemented (placeholder zero) | Measured from spatial variation of line centroids |
| Slit curvature | Not modelled | Measured from arc images |
| Polynomial fit | OLS, no sigma-clipping | Iterative weighted fit |
| Rectification indices | Not generated | Per-pixel λ/spatial coordinate arrays |
| Wavecal accuracy | Engineering check only | Science-quality calibration |

---

## What Remains Intentionally Unimplemented

1. **`trace_arc_lines_from_2d_image()`** – declared as a stub in
   `arc_tracing.py`.  Requires real 2-D arc FITS frames in
   `data/testdata/ishell_h1_calibrations/raw/`.
2. **Tilt measurement** – requires spatial variation of arc-line centroids
   across the slit (available only from a 2-D arc image).
3. **Slit curvature** – not modelled at any stage.
4. **Final rectification indices** – per-pixel wavelength/spatial index
   arrays for `_rectify_orders()`.
5. **Iterative arc-line identification** – no sigma-clipping of outlier
   centroids; no interactive refinement.
6. **IDL `P2W_C*` polynomial convention verification** – the stored
   coefficient evaluation form remains unverified.
7. **Wavelength uncertainty propagation** – fit residuals are not stored in
   `OrderGeometrySet`.

---

## Files Changed vs Previous Version

| File | Change |
|------|--------|
| `src/pyspextool/instruments/ishell/arc_tracing.py` | **New** – arc-line tracing scaffold with `TracedArcLine`, `ArcTraceResult`, `trace_arc_lines_from_1d_spectrum`, `build_arc_trace_result_from_wavecalinfo`, stub `trace_arc_lines_from_2d_image` |
| `src/pyspextool/instruments/ishell/wavecal_2dxd.py` | **Revised** – primary path now uses `ArcTraceResult`; added `collect_centroid_data_from_arc_traces()`; `fit_provisional_2dxd()` now takes `ArcTraceResult`; `collect_centroid_data()` demoted to auxiliary |
| `src/pyspextool/instruments/ishell/__init__.py` | **Revised** – imports `arc_tracing` |
| `src/pyspextool/instruments/ishell/calibrations.py` | **Unchanged** from previous version (still has `WaveCalInfo.p2w_coeffs`) |
| `tests/test_ishell_wavecal_2dxd.py` | **Revised** – tests the corrected `arc_tracing → wavecal_2dxd` path |
| `data/testdata/ishell_h1_calibrations/raw/README.md` | **New** – directory stub with instructions for future raw frames |
| `docs/ishell_2dxd_algorithm_note.md` | **This file** – updated to reflect corrected data flow |

---

## Quick-start

```python
import warnings
from pyspextool.instruments.ishell.arc_tracing import (
    build_arc_trace_result_from_wavecalinfo
)
from pyspextool.instruments.ishell.calibrations import (
    read_flatinfo, read_line_list, read_wavecalinfo
)
from pyspextool.instruments.ishell.wavecal_2dxd import fit_provisional_2dxd

wci = read_wavecalinfo("H1")
fi  = read_flatinfo("H1")
ll  = read_line_list("H1")

# Step 1: build ArcTraceResult (primary arc-tracing path)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    arc_traces = build_arc_trace_result_from_wavecalinfo(wci, ll, fi)

print(f"Traced lines:  {arc_traces.n_lines} across {arc_traces.n_orders_with_lines} orders")
print(f"Source:        {arc_traces.source}")

# Step 2: fit provisional 2DXD polynomial
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = fit_provisional_2dxd(arc_traces, wci, fi, ll)

print(f"Mode:          {result.mode}")
print(f"RMS residual:  {result.rms_residual_um*1000:.3f} nm")
print(f"centroid src:  {result.centroid_data.source}")

# Step 3: evaluate at a detector position
import numpy as np
lam = result.eval_wavelength(np.array([1000.0]), np.array([333.0]))
print(f"λ(x=1000, m=333) ≈ {lam[0]:.4f} µm")
```

Running the tests:

```bash
pytest tests/test_ishell_wavecal_2dxd.py -v
```

---

## References

- `docs/ishell_wavecal_design_note.md` – per-order wavecal paths
- `docs/ishell_geometry_design_note.md` – geometry data model
- `docs/ishell_implementation_status.md` – overall iSHELL status
- `src/pyspextool/instruments/ishell/arc_tracing.py` – arc-line tracing
- `src/pyspextool/instruments/ishell/wavecal_2dxd.py` – provisional 2DXD fit
- `src/pyspextool/instruments/ishell/calibrations.py` – WaveCalInfo, FlatInfo

