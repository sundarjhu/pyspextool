# iSHELL 2DXD Wavelength-Calibration / Line-Tracing / Rectification-Index Algorithm

**Status:** Reference documentation — algorithm analysis only, no Python implementation yet  
**Source:** IDL Spextool routines under `docs/idl_reference/spextool_2dxd/`  
**Scope:** iSHELL J, H, K modes  
**Author:** pySpextool iSHELL integration

---

## 0. Document Conventions

Throughout this note the source of every claim is labelled:

| Label | Meaning |
|-------|---------|
| **[MANUAL]** | Stated in the iSHELL Spextool manual |
| **[IDL]** | Directly confirmed by reading the IDL source in `docs/idl_reference/spextool_2dxd/` |
| **[INFERRED]** | Logically deduced from the IDL code but not a direct quotation |
| **[UNCERTAIN]** | Cannot be confirmed from the available source; judgement required |

---

## 1. Overview

The iSHELL calibration pipeline produces a **wavecal FITS** file that contains
two 2-D calibration images (`wavecal` and `spatcal`) plus per-order
rectification-index arrays.  These products are used downstream to rectify raw
science frames and to map raw-pixel positions to wavelength and spatial angle.

The IDL pipeline follows two sequential phases **[MANUAL, IDL]**:

1. **1DXD wavelength calibration** — a 2-D cross-dispersed polynomial fit of
   arc-line centroids measured along the order centerline.  Produces a
   pixel-to-wavelength mapping for the full focal-plane mosaic of orders.

2. **2DXD distortion mapping** — traces each arc line spatially across the
   full slit height to characterise spectral-line tilt and curvature, fits
   polynomial surfaces to those measurements, then builds rectification-index
   arrays used to resample raw images onto a rectilinear wavelength × arcsecond
   grid.

---

## 2. IDL Routine Inventory

| Routine file | Role |
|---|---|
| `mc_ishellcals2dxd.pro` | Top-level orchestrator for iSHELL calibrations |
| `mc_readwavecalinfo.pro` | Reads `*_wavecalinfo.fits` calibration file |
| `mc_readlinelist.pro` | Reads `|`-delimited line-list text files |
| `mc_wavecal1dxd.pro` | 1DXD wavelength calibration (calls `mc_findlines1dxd`) |
| `mc_findlines1dxd.pro` | Finds arc-line centroids in 1-D extracted spectra |
| `mc_findlines2d.pro` | Traces each arc line spatially in 2-D across the slit |
| `mc_fitlines2dxd.pro` | Fits a polynomial to each traced line's (x, y) path |
| `mc_fitlinecoeffs2d.pro` | Fits polynomial surfaces to the per-line coefficients |
| `mc_mkrectindcs2d.pro` | Generates (ix, iy) rectification-index arrays per order |
| `mc_mkwavecalimgs2d.pro` | Builds 2-D `wavecal` / `spatcal` images from the indices |

Routines **not** present in the provided IDL reference files (therefore
**[UNCERTAIN]** in detail):

| Routine | Inferred role |
|---|---|
| `mc_simwavecal2d` | Creates provisional wavecal/spatcal for 1-D arc extraction |
| `mc_getlinexguess` | Converts line wavelengths to pixel guesses using stored reference |
| `xmc_corspec` | Cross-correlates arc and reference spectra to measure pixel offset |
| `mc_adjustguesspos` | Adjusts order-edge guess positions using data |
| `mc_findorders` | Traces order edges in the flat field |
| `mc_normspecflat` | Normalises the flat field |
| `mc_sumextspec` | Sum-extraction of arc spectra along centerlines |
| `mc_robustpoly2d` | Robust 2-D polynomial fit (Spextool utility) |
| `mc_poly2d` | Evaluates a 2-D polynomial (Spextool utility) |
| `mc_robustpoly1d` | Robust 1-D polynomial fit (Spextool utility) |
| `qtrap` | Numerical integration by trapezoidal rule (Spextool utility) |
| `mc_polyarclenfunc` | Integrand for arc-length along a polynomial curve |

---

## 3. Call Graph

```
mc_ishellcals2dxd          [top-level orchestrator]
│
├─ mc_readwavecalinfo       reads *_wavecalinfo.fits → wavecalinfo struct
│     returns: orders, homeorder, dispdeg, ordrdeg, linedeg,
│              fndystep, fndysum, genystep, c1xdeg, c1ydeg,
│              (c2xdeg, c2ydeg if linedeg=2),
│              xcororder, xcorspec, p2wcoeffs, xranges, wspec
│
├─ mc_simwavecal2d          [UNCERTAIN] provisional wavecal for 1-D extraction
│
├─ mc_sumextspec            sum-extracts arc spectrum along centerlines
│
├─ xmc_corspec              [UNCERTAIN] cross-correlate arc vs. reference → offset
│
├─ mc_readlinelist          reads *.dat line list → lineinfo struct
│
├─ mc_getlinexguess         [UNCERTAIN] predicts pixel positions of lines
│
├─ mc_wavecal1dxd           ─────── 1DXD PHASE ───────
│   │   inputs: arcspec, orders, lineinfo, homeorder, dispdeg, ordrdeg
│   │   outputs: p2wcoeffs (2D polynomial), olineinfo (with xpos added)
│   │
│   └─ mc_findlines1dxd     fits Gaussian/Lorentzian/centroid to 1-D spectra
│         for each line: → xpos, fwhm, inten, fnd_goodbad
│         (uses mpfitpeak for Gaussian/Lorentzian)
│         → mc_robustpoly2d  (2DXD pixel-to-wavelength polynomial fit)
│
├─ mc_findlines2d           ─────── 2DXD PHASE: line tracing ───────
│     inputs: arc (2-D image), edgecoeffs, xranges, orders,
│             lineinfo (with xpos from 1DXD, igoodbad = fit_goodbad)
│             linedeg, fndystep, fndysum
│     outputs: lxy (structure of [x,y] arrays, one per line)
│              ogoodbad (which lines were successfully traced)
│     calls: mpfitpeak (per slit step) or centroid calculation
│            mc_robustpoly1d (running line fit during upward sweep)
│
├─ mc_fitlines2dxd          ─────── 2DXD PHASE: line-polynomial fit ───────
│     inputs: lxy, lorders, edgecoeffs, xranges, orders, linedeg
│     outputs: lstruc [ linecoeffs, xbot, ybot, xmid, ymid,
│                       xtop, ytop, order, deltax, slope, angle ]
│     calls: mc_robustpoly1d (x = p(y) per line)
│
├─ mc_fitlinecoeffs2d       ─────── 2DXD PHASE: coefficient surfaces ───────
│     inputs: lstruc, fitdeg=[c1xdeg,c1ydeg,(c2xdeg,c2ydeg)]
│     outputs: c1coeffs, (c2coeffs if linedeg=2)
│     calls: mc_robustpoly2d  (surface fit of c1/c2 vs detector (x,y))
│
├─ mc_mkrectindcs2d         ─────── 2DXD PHASE: rectification indices ───────
│     inputs: edgecoeffs, xranges, coeffs{c1coeffs,c1xdeg,c1ydeg,...},
│             slith_arc, ds (plate scale), genystep
│     outputs: indices struct (one [nxgrid, nsgrid, 2] array per order)
│              xgrids (one column-grid per order), sgrid
│     calls: mc_poly2d (evaluate c1/c2 at each grid point)
│            mc_polyarclenfunc + qtrap (arc-length integration)
│            interpol (uniform spatial grid)
│
└─ mc_mkwavecalimgs2d       builds 2-D wavecal / spatcal images
      inputs: omask, orders, findices (indices augmented with wgrid/sgrid)
      outputs: wavecal (2-D, wavelength per pixel), spatcal (2-D, arcsec)
      calls: griddata (IDL polynomial regression interpolation)
```

---

## 4. Algorithm Step-by-Step (Plain Language)

### Step 0 — Prerequisites

Before the 2DXD code runs, the flat field has already been processed
**[IDL]**: order edges have been traced (`mc_findorders`), the flat has been
normalised (`mc_normspecflat`), and the output `edgecoeffs` and `xranges` are
available.  A 2-D arc-lamp image is loaded and divided by the flat.

### Step 1 — Load calibration file

`mc_readwavecalinfo` reads `*_wavecalinfo.fits` and returns a structure that
contains **[IDL]**:

- The stored reference wavelength spectrum per order (`wspec`, plane 0 = wavelengths, plane 1 = fluxes).
- Per-order column ranges (`xranges`).
- The 1DXD polynomial degree parameters: `homeorder`, `dispdeg`, `ordrdeg`.
- The 2DXD-specific parameters (only if `wcaltype = '2DXD'`):
  `linedeg`, `fndystep`, `fndysum`, `genystep`, `c1xdeg`, `c1ydeg`
  (and optionally `c2xdeg`, `c2ydeg` when `linedeg = 2`).
- Previously fitted `p2wcoeffs` coefficients (from a prior run or a bootstrap).

### Step 2 — Provisional 1-D extraction of the arc

`mc_simwavecal2d` creates a simplified wavecal/spatcal image pair sufficient
for a 1-D arc extraction **[UNCERTAIN — routine not in reference files]**.
`mc_sumextspec` sums a narrow aperture centred on the midline of each order.

### Step 3 — Cross-correlation offset

`xmc_corspec` cross-correlates the extracted arc spectrum (order `xcororder`)
against a stored reference spectrum (`xcorspec`) to measure the overall
wavelength offset in pixels **[IDL — call site only; routine not in reference
files]**.  This offset is added to every line's pixel-position guess before the
1DXD fit.

### Step 4 — Load line list

`mc_readlinelist` parses a `|`-delimited text file with columns:
order number, wavelength (string), line ID, fit window (in wavelength units),
fit type (`G`/`L`/`C` for Gaussian/Lorentzian/centroid), and number of fit
terms (3–5) **[IDL]**.  The fit window is converted from Å to µm.

Optionally a seventh column (`dmask`) marks whether a line should be used in
the distortion solution **[IDL — `DISTORTION` keyword of `mc_readlinelist`]**.

### Step 5 — 1DXD wavelength calibration (`mc_wavecal1dxd`)

**[IDL — fully confirmed]**

1. `mc_findlines1dxd` iterates over every line in the list.  For each line:
   - A window of the 1-D extracted arc spectrum around `xguess` is extracted.
   - A Gaussian (`G`), Lorentzian (`L`), or centroid (`C`) profile is fitted
     via `mpfitpeak` (or a moment calculation for centroid).
   - The fitted position `xpos`, FWHM, and intensity are stored.
   - A line is marked bad if `|xpos − xguess| > xthresh` or the amplitude is
     non-positive.

2. The scaled wavelength `sclwave = wavelength × order / homeorder` is
   computed for each line.  This folds all orders onto a common dispersion
   using the standard cross-dispersed echelle relation **[IDL, also MANUAL]**.

3. `mc_robustpoly2d(xpos, order, sclwave, dispdeg, ordrdeg)` performs a
   2-D robust polynomial fit:
   `sclwave = P2D(xpos, order)` where `P2D` has degree `dispdeg` in `xpos`
   and degree `ordrdeg` in `order`.

   Output: `p2wcoeffs` (array of `(dispdeg+1) × (ordrdeg+1)` coefficients)
   and `p2wrms` (fit RMS in Å).

### Step 6 — 2-D line tracing (`mc_findlines2d`)

**[IDL — fully confirmed]**

This step uses the **full 2-D arc image** and the `xpos` centroids from Step 5
as starting points.  Only lines that passed the 1DXD fit (`fit_goodbad = 1`)
are processed.

For each surviving line in each order:

1. **Starting position**: the 1DXD centroid `xpos` gives the x-position at
   the slit midpoint.  The y-position is the geometric midline of the order at
   that column.

2. **Upward sweep**: step upward in y by `ystep` pixels at a time.  At each
   step:
   - Collapse `ysum` rows by taking a median in the y direction.
   - Extract a window of width `xwin` pixels around the current x-guess.
   - Fit the same profile type (Gaussian/Lorentzian/centroid) as in Step 5.
   - If the fit amplitude is negative or the centroid deviates by more than
     2 pixels from the guess, discard this step and continue.
   - After ≥ 3 good points, update the x-guess using a running polynomial fit
     of degree `linedeg` to all accepted (y, x) pairs so far.
   - Stop if the position reaches within 1 pixel of the top edge.

3. **Downward sweep**: repeat from the midpoint going downward.  The
   polynomial fit from the upward sweep provides the initial x-guess.

4. If fewer than 10 good (x, y) pairs are accumulated across both sweeps, the
   line is marked bad.

5. Outputs: structure `lxy` (one tag per good line, each holding an `[[x],[y]]`
   array), and `ogoodbad` (integer array flagging which lines were traced).

**Key parameters**: `linedeg` (1 or 2), `fndystep` (typically a few pixels),
`fndysum` (typically 1–3 pixels).

### Step 7 — Per-line polynomial fit (`mc_fitlines2dxd`)

**[IDL — fully confirmed]**

For each successfully traced line:

1. Fit `x = p(y)` of degree `linedeg` (where `x` and `y` are detector column
   and row) using `mc_robustpoly1d`.

2. Find the intercepts of this fitted polynomial with the top and bottom slit
   edges (by root-finding on `x − p(y) = 0`) and with the geometric midline.
   Store the intercept coordinates `(xtop, ytop)`, `(xbot, ybot)`,
   `(xmid, ymid)`.

3. Compute summary statistics: `deltax = xtop − xbot` (column displacement
   top-to-bottom), `slope = deltax / (ytop − ybot)`, `angle` (arctangent of
   slope in degrees).

Output: `lstruc`, an array of records with fields:
`linecoeffs, xbot, ybot, xmid, ymid, xtop, ytop, order, deltax, slope, angle`.

### Step 8 — Coefficient surface fitting (`mc_fitlinecoeffs2d`)

**[IDL — fully confirmed]**

The goal is to describe how line tilt (and optionally curvature) varies as a
smooth function of detector position across all orders.

1. **Re-centre the line polynomial**: the raw polynomial `x = p(y)` from
   Step 7 is converted to one that is centred at the slit midpoint
   `y_mid` by substituting `y' = y − y_mid`.  For a linear fit (`linedeg=1`)
   the centred slope coefficient `c1` is simply `linecoeffs[1]`.  For a
   quadratic fit (`linedeg=2`) the transformation is:
   - `c1_centred = linecoeffs[1] + 2 × linecoeffs[2] × y_mid`
   - `c2_centred = linecoeffs[2]`

2. **Fit c1 surface**: collect the centred c1 value from every good line along
   with its midpoint position `(xmid, ymid)`.  Fit a 2-D robust polynomial
   surface of degree `c1xdeg` in column and `c1ydeg` in row using
   `mc_robustpoly2d` → `c1coeffs`.

3. **Fit c2 surface** (only if `linedeg = 2`): fit the centred c2 values
   similarly, with degrees `c2xdeg`, `c2ydeg` → `c2coeffs`.

The fitted surfaces allow the tilt (and curvature) at any detector position to
be evaluated by `mc_poly2d(col, row, c1xdeg, c1ydeg, c1coeffs)`.

### Step 9 — Rectification-index generation (`mc_mkrectindcs2d`)

**[IDL — fully confirmed]**

This step computes, for every point on the desired output grid
(column × arcsecond), the fractional detector coordinate (column, row) on the
raw detector from which that output pixel should be sampled.

**Output grid definition**:

- Spatial axis: `sgrid = 0, ds, 2*ds, ..., slith_arc` (arcsec),
  giving `nsgrid = round(slith_arc / ds) + 1` points.
- Wavelength/column axis: one grid point per detector column within the
  valid order range `[xranges[0,i], xranges[1,i]]`.

**For each column j and each order i**:

1. Evaluate the tilt at the slit midpoint:
   `c1_at_j = mc_poly2d(xgrid[j], midslit[j], c1xdeg, c1ydeg, c1coeffs)`

2. Re-construct the full local line polynomial **centred at the midpoint**:
   - Linear case (`linedeg=1`): `x = c0 + c1_at_j * y`
     where `c0 = xgrid[j] − c1_at_j * midslit[j]` (pinned at midpoint).
   - Quadratic case (`linedeg=2`): additionally evaluate `c2_at_j`, then
     re-derive `c0, c1, c2` such that the parabola passes through `xgrid[j]`
     at `y = midslit[j]` with the correct slope and curvature.

3. Find the row positions where this polynomial crosses the top and bottom
   slit edges (`xtop, ytop, xbot, ybot`) by interpolation.

4. Integrate the arc length of the polynomial curve from `ybot` to `ytop`
   using `qtrap` (numerical quadrature), obtaining the total slit length in
   pixels.

5. Sample the arc length at intermediate steps of `ystep` rows to get a set
   of (row, arc_length) pairs.  Normalise the arc lengths to the physical
   slit length in arcseconds (`slith_arc`), producing a mapping
   `sline = arc_length / total_arc_length * slith_arc`.

6. Interpolate this mapping to the uniform `sgrid` values:
   `nyline = interpol(yline, sline, sgrid)`.  The output is the row coordinate
   on the raw detector corresponding to each slit-position grid point.

7. The corresponding column coordinate is `ix = poly(nyline, linecoeffs)`.

8. Store `(ix[j, :], iy[j, :])` as the rectification index at column `j`.

After all columns are processed, columns near the order edges where `ix`
strays more than 2 pixels outside the valid range are trimmed.

### Step 10 — Wavelength and spatial calibration images (`mc_mkwavecalimgs2d`)

**[IDL — fully confirmed]**

The rectification indices are augmented with the wavelength grid (`wgrid`)
computed from the 1DXD polynomial `p2wcoeffs`:

```
wgrid = mc_poly2d(xgrids, order/homeorder, dispdeg, ordrdeg, p2wcoeffs) / (order/homeorder)
```

Then `mc_mkwavecalimgs2d` uses IDL `griddata` (degree-3 polynomial regression)
to interpolate from the scattered `(ix, iy, wavelength)` and `(ix, iy, spatcal)`
samples to a uniform per-pixel grid, producing the final 2-D `wavecal` and
`spatcal` images.

### Step 11 — Output FITS file

The output `*_wavecal.fits` file contains **[IDL]**:

- Primary HDU: empty image, header with all calibration parameters.
- Extension 1: `wavecal` image (2-D float, wavelength per pixel in µm).
- Extension 2: `spatcal` image (2-D float, arcsec per pixel).
- Extensions 3…3+norders: per-order rectification-index arrays
  `findices.(i)` of shape `(nxgrid+1, nsgrid+1, 2)`.

Key header keywords written: `EXTTYPE='2D'`, `WCTYPE='2DXD'`, `FLATNAME`,
`ORDERS`, `NORDERS`, `HOMEORDR`, `DISPDEG`, `ORDRDEG`, `LINEDEG`,
`FNDYSTEP`, `FNDYSUM`, `GENYSTEP`, `C1XDEG`, `C1YDEG` (+ `C2XDEG`, `C2YDEG`
when `linedeg=2`), `RMS`, `OR{n}_XR`, `DISPO{n}`, `WAVEFMT`, `SPATFMT`.

---

## 5. Evidence Table — Manual vs. Confirmed vs. Uncertain

| Claim | Source | Confidence |
|---|---|---|
| 1DXD followed by 2DXD | Manual + IDL | **Confirmed** |
| 1DXD fits 2-D polynomial `sclwave = P2D(xpos, order)` | IDL `mc_wavecal1dxd` | **Confirmed** |
| `sclwave = wavelength × order / homeorder` | IDL | **Confirmed** |
| Arc cross-correlation offset before line find | IDL call site | **Confirmed**; internal routine not provided |
| 2-D line tracing sweeps up then down from midpoint | IDL `mc_findlines2d` | **Confirmed** |
| Line traced by iterative Gaussian/Lorentzian/centroid fits to collapsed rows | IDL `mc_findlines2d` | **Confirmed** |
| Running polynomial re-fit of x-guess during sweep | IDL `mc_findlines2d` | **Confirmed** |
| Per-line polynomial `x = p(y)` fit of degree `linedeg` | IDL `mc_fitlines2dxd` | **Confirmed** |
| Slope/angle statistics stored per line | IDL `mc_fitlines2dxd` | **Confirmed** |
| 2-D surface fit of c1 (and c2) vs detector position | IDL `mc_fitlinecoeffs2d` | **Confirmed** |
| Line polynomial re-centred at slit midpoint before surface fit | IDL `mc_fitlinecoeffs2d` | **Confirmed** |
| Arc-length integration for spatial coordinate | IDL `mc_mkrectindcs2d` | **Confirmed** |
| Spatial grid in arcsec, steps = plate scale (`ds`) | IDL `mc_mkrectindcs2d` | **Confirmed** |
| Local line polynomial pinned at midpoint for each column | IDL `mc_mkrectindcs2d` | **Confirmed** |
| Final wavecal/spatcal built by griddata interpolation | IDL `mc_mkwavecalimgs2d` | **Confirmed** |
| `wavecalinfo.wspec` plane 0 = reference wavelengths | IDL `mc_readwavecalinfo` | **Confirmed** (XUNITS="um") |
| `wavecalinfo.wspec` plane 1 = arc spectrum | Header YUNITS="DN/s" | **Likely** — not directly stated in code |
| `mc_simwavecal2d` creates provisional wavecal for 1-D extraction | IDL call site only | **[UNCERTAIN]** |
| `mc_getlinexguess` predicts pixel positions from reference grid | IDL call site only | **[UNCERTAIN]** |
| Default `linedeg=1` for iSHELL | Header value `LINEDEG=1` | **Confirmed** for packaged files |
| `linedeg=2` supported | IDL code branches | **Confirmed** in code; not confirmed for iSHELL modes |

---

## 6. Python Implementation Plan

This section maps each IDL stage onto the existing pyspextool iSHELL
architecture.  All new functions should live in
`src/pyspextool/instruments/ishell/` to avoid touching generic code.

### 6.1 Architecture summary (existing)

The existing Python codebase provides:

- `calibrations.py` — typed readers (`WaveCalInfo`, `FlatInfo`, `LineList`).
- `geometry.py` — `OrderGeometry`, `OrderGeometrySet`, `RectificationMap`,
  `build_order_geometry_set`.
- `wavecal.py` — `fit_arc_line_centroids`, `build_geometry_from_wavecalinfo`,
  `build_geometry_from_arc_lines`, `build_rectification_maps`.
- `ishell.py` — `_rectify_orders` (provisional zero-tilt stub).

The new 2DXD work extends `wavecal.py` and `ishell.py` without touching any
generic module.

### 6.2 Required new functions

> **Important:** The functions below are not yet implemented.  This section
> describes what they should do when implemented.

---

#### Stage A — 2-D line tracing

**Function**: `trace_lines_2d(arc_image, edge_coeffs, x_ranges, orders, lineinfo, line_deg, ystep, ysum) -> tuple[list[np.ndarray], np.ndarray]`

Maps to: `mc_findlines2d`

Inputs:
- `arc_image`: `np.ndarray` shape `(nrows, ncols)`, flat-divided arc lamp.
- `edge_coeffs`: `np.ndarray` shape `(norders, 2, n_terms)`, order-edge polynomials.
- `x_ranges`: `np.ndarray` shape `(norders, 2)`.
- `orders`: `np.ndarray` shape `(norders,)`.
- `lineinfo`: list-of-dicts (or dataclass array) produced by `fit_arc_line_centroids`; each entry carries `order`, `xpos` (1-D centroid), `xwin`, `fittype`, `nterms`, `fit_goodbad`.
- `line_deg`: int, polynomial degree for the running fit (1 or 2).
- `ystep`: int, spatial step size in pixels.
- `ysum`: int, number of rows to collapse at each step.

Outputs:
- `line_xy`: list of `np.ndarray` shape `(n_pts, 2)`, one per good line — column and row pairs.
- `goodbad`: `np.ndarray` shape `(nlines,)`, int, 1 = successfully traced.

Algorithm sketch (Python):
```python
for each order i:
    for each line j where lineinfo[j].fit_goodbad == 1:
        x_guess = lineinfo[j].xpos
        y_guess = midline(x_guess, edge_coeffs[i])
        # sweep upward:
        tx, ty = [], []
        for k in range(1, nrows):
            if y_guess + k*ystep >= top_edge(x_guess):
                break
            col_slice = arc_image[y_guess+k*ystep-ysum : y_guess+k*ystep+ysum+1, :]
            zsub = np.nanmedian(col_slice, axis=0)[window]
            fit x centroid (Gaussian / Lorentzian / centroid)
            if good: tx.append(x_fit); ty.append(y_guess + k*ystep)
            if len(tx) >= 3: update x_guess via poly1d fit
        # sweep downward (mirror logic)
        ...
        if len(all_good_pts) < 10: mark bad; continue
        line_xy[j] = np.column_stack([tx_all, ty_all])
```

Key design notes:
- Use `scipy.optimize.curve_fit` or `scipy.signal.find_peaks` / `astropy.modeling` for profile fitting.
- The running x-guess update should use `np.polynomial.polynomial.polyfit`.
- The `ysum` collapse should be `np.nanmedian` over rows.
- Note that the IDL array layout is `img[col, row]`; Python layout is `img[row, col]` — take care when porting index expressions.

---

#### Stage B — Per-line polynomial fit

**Function**: `fit_line_polynomials(line_xy, line_orders, edge_coeffs, x_ranges, orders, line_deg) -> list[dict]`

Maps to: `mc_fitlines2dxd`

For each traced line:
1. Fit `col = p(row)` of degree `line_deg` to the (col, row) pairs using a
   robust polynomial fitter (e.g., iterative sigma-clipping with
   `np.polynomial.polynomial.polyfit`).
2. Find slit-edge intercepts by root-finding (`scipy.optimize.brentq` or
   linear interpolation on `col − p(row)`).
3. Compute `(xmid, ymid)` from the midline intercept.
4. Compute `deltax`, `slope = deltax / deltay`, `angle`.

Returns a list of dicts (or a dataclass array) with fields:
`line_coeffs, xbot, ybot, xmid, ymid, xtop, ytop, order, deltax, slope, angle`.

---

#### Stage C — Coefficient surface fitting

**Function**: `fit_tilt_surface(line_structs, c1_xdeg, c1_ydeg, c2_xdeg=None, c2_ydeg=None) -> dict`

Maps to: `mc_fitlinecoeffs2d`

1. Extract the centred c1 coefficient from each line record:
   - Linear (`line_deg=1`): `c1_centred = line_coeffs[1]`
   - Quadratic (`line_deg=2`): `c1_centred = line_coeffs[1] + 2*line_coeffs[2]*ymid`
2. Fit `c1_centred = P2D(xmid, ymid)` using a robust 2-D polynomial fitter
   (`np.polynomial.polynomial.polyvander2d` + iterative sigma-clipping).
3. If `line_deg=2`, repeat for `c2_centred = line_coeffs[2]`.

Returns dict with keys: `c1_coeffs`, `c1_xdeg`, `c1_ydeg`, and optionally
`c2_coeffs`, `c2_xdeg`, `c2_ydeg`.

**Polynomial surface convention**: use
`np.polynomial.polynomial.polyval2d(col, row, c_2d)` where `c_2d` has shape
`(c1_xdeg+1, c1_ydeg+1)`.

---

#### Stage D — Rectification-index generation

**Function**: `make_rectification_indices(edge_coeffs, x_ranges, tilt_coeffs, slith_arc, plate_scale, gen_ystep) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]`

Maps to: `mc_mkrectindcs2d`

1. Define `sgrid = np.arange(0, slith_arc + plate_scale, plate_scale)`.
2. For each order i:
   - For each column j in `x_ranges[i, 0] : x_ranges[i, 1]+1`:
     - Evaluate `c1_at_j = polyval2d(xgrid[j], midslit[j], c1_coeffs_2d)`.
     - Reconstruct local line polynomial pinned at `(xgrid[j], midslit[j])`.
     - Find top/bottom slit-edge intercepts by interpolation.
     - Integrate arc length numerically from `ybot` to `ytop`
       (`scipy.integrate.quad` with the polynomial arc-length integrand).
     - Normalise arc length to arcsec.
     - Interpolate to `sgrid` → `nyline`.
     - `iy[j, :] = nyline`, `ix[j, :] = polyval(nyline, line_coeffs)`.
3. Trim edge columns where `ix` strays outside `x_ranges ± 2`.

Returns: `indices` (list of `np.ndarray` shape `(nxgrid, nsgrid, 2)`, one
per order), `xgrids` (list of column arrays), `sgrid`.

**Note on axis order**: IDL stores as `(nxgrid, nsgrid, 2)`.  The Python
convention should be consistent with `scipy.ndimage.map_coordinates` which
expects coordinate arrays over `(row, col)`.

---

#### Stage E — Wavecal / spatcal image construction

**Function**: `make_wavecal_images(order_mask, orders, rect_indices, p2w_coeffs, plate_scale, home_order, disp_deg, ordr_deg) -> tuple[np.ndarray, np.ndarray]`

Maps to: `mc_mkwavecalimgs2d` + wgrid construction in `mc_ishellcals2dxd`

1. For each order, compute `wgrid` from `p2w_coeffs` using the 2DXD polynomial.
2. Augment `rect_indices` with wgrid (as column 0) and sgrid (as row 0).
3. Interpolate from the scattered `(ix, iy, wavelength)` samples to a
   per-pixel grid using `scipy.interpolate.griddata` (method=`'cubic'` or
   `'linear'`) — this replaces IDL `griddata`.
4. Repeat for the spatial coordinate (arcsec).

Returns: `wavecal` shape `(nrows, ncols)`, `spatcal` shape `(nrows, ncols)`.

---

#### Stage F — `_rectify_orders` implementation

**Function**: `_rectify_orders(image, order_geometry_set, plate_scale_arcsec) -> np.ndarray`

Currently a provisional stub in `ishell.py`.  When `tilt_coeffs` are not all
zero, this function should:

1. Call `build_rectification_maps(order_geometry_set, plate_scale_arcsec)`.
2. For each order, call
   `scipy.ndimage.map_coordinates(image, [src_rows, src_cols], order=3)`.
3. Set pixels outside all order footprints to `NaN`.

This function signature already exists; only the body needs to change once
the tilt model is populated.

---

#### Stage G — Updated `build_geometry_from_arc_lines`

After implementing Stages A–D, `build_geometry_from_arc_lines` (in `wavecal.py`)
should be extended to:

1. Accept a `arc_image` parameter (a 2-D arc frame).
2. After the existing 1DXD centroid step, call Stages A–D to populate
   `tilt_coeffs` (and `curvature_coeffs`) on each `OrderGeometry`.

This is the clean integration point with the existing Python architecture.

### 6.3 Integration call sequence (future Python)

```
read_flatinfo(mode)         → FlatInfo
read_wavecalinfo(mode)      → WaveCalInfo
read_line_list(mode)        → LineList
load arc image              → arc_image (np.ndarray)

fit_arc_line_centroids(wci, line_list)   # Stage 5 — 1DXD
    → lineinfo (with xpos per line)

trace_lines_2d(arc_image, ...)           # Stage A — 2-D tracing
    → line_xy, goodbad

fit_line_polynomials(line_xy, ...)       # Stage B — per-line poly
    → line_structs

fit_tilt_surface(line_structs, ...)      # Stage C — coefficient surfaces
    → tilt_coeffs  {c1_coeffs, c1_xdeg, c1_ydeg, ...}

make_rectification_indices(...)          # Stage D — rect indices
    → indices, xgrids, sgrid

make_wavecal_images(...)                 # Stage E — wavecal / spatcal
    → wavecal, spatcal

populate OrderGeometrySet with tilt_coeffs  # updates geometry.py objects

_rectify_orders(science_image,           # Stage F — resampling
               order_geometry_set,
               plate_scale)
    → rectified_image
```

### 6.4 Suggested implementation order

1. Add `trace_lines_2d` (Stage A) — this is the most complex new step.
2. Add `fit_line_polynomials` (Stage B) — straightforward polynomial fitting.
3. Add `fit_tilt_surface` (Stage C) — 2-D polynomial surface fitting.
4. Add `make_rectification_indices` (Stage D) — requires arc-length integration.
5. Add `make_wavecal_images` (Stage E) — griddata interpolation.
6. Update `build_geometry_from_arc_lines` to accept `arc_image` and call A–C.
7. Update `build_rectification_maps` and `_rectify_orders` to use the new tilt model.

---

## 7. Minimum New Data Products Required

The following data and products are **not present** in the existing packaged
calibration files and will be required for a true 2DXD implementation:

| Required input/product | Reason | Notes |
|---|---|---|
| **2-D arc-lamp FITS frame** | `mc_findlines2d` requires spatial variation of line positions across the full slit | Currently only the 1-D centerline spectrum is stored in `*_wavecalinfo.fits` |
| **Traced line (x, y) arrays** | Output of Stage A; needed by Stage B | Not stored anywhere currently |
| **Per-line polynomial coefficients** | Output of Stage B; needed by Stage C | Not stored |
| **Tilt surface coefficient arrays `c1_coeffs` (and `c2_coeffs`)** | Output of Stage C; needed by Stage D | Not stored; currently a placeholder zero in `OrderGeometry.tilt_coeffs` |
| **Rectification index arrays** | Output of Stage D; needed by Stage E and `_rectify_orders` | Not stored in the packaged calibration files |
| **2-D `wavecal` / `spatcal` images** | Final products of Stage E; needed by extraction | Currently populated provisionally from 1-D reference data |

> **Note:** The packaged `*_wavecalinfo.fits` files already store many 1DXD
> parameters (`p2wcoeffs`, `homeorder`, `dispdeg`, `ordrdeg`, `xcorspec`).
> What they lack is the spatial information: 2-D arc images and the products
> derived from them (Stages A–D above).

---

## 8. What the Manual States, What Is Confirmed, and What Remains Uncertain

### Explicitly stated in the iSHELL Spextool manual

- The pipeline performs a 1DXD solution followed by a 2DXD distortion mapping.
- The 2DXD step involves tracing arc lines spatially across the slit.
- A rectification index is derived and used to resample images.

### Confirmed directly from the IDL source

All items in the "Confirmed" rows of §5 above.  In particular:

- The exact algorithm of `mc_findlines2d` (sweep up then down, running polynomial guess, centroid/Gaussian/Lorentzian fit per slit slice, 10-point minimum).
- The re-centring of line polynomials at the midpoint before surface fitting in `mc_fitlinecoeffs2d`.
- The arc-length integration in `mc_mkrectindcs2d` (rather than a simpler linear interpolation).
- The use of `griddata` polynomial regression for the final wavecal/spatcal images.

### Uncertain (not confirmed from available source)

- The precise behaviour of `mc_simwavecal2d`, `mc_getlinexguess`, `xmc_corspec`.
- Whether `linedeg=2` is ever used in practice for iSHELL (only `linedeg=1` appears in the packaged headers).
- The plane-1 semantics of the packaged `*_wavecalinfo.fits` files (likely arc spectrum but not confirmed in code).
- Whether all iSHELL modes use the same `ystep`, `linedeg`, etc., or whether some modes have different settings (only read from the per-mode `*_wavecalinfo.fits` header).

---

## 9. References

- iSHELL Spextool Manual v10jan2020, Cushing et al. (IRTF internal).
- IDL source files: `docs/idl_reference/spextool_2dxd/pro/` and
  `docs/idl_reference/spextool_2dxd/instruments/ishell/pro/`.
- `docs/ishell_wavecal_design_note.md` — current Python wavecal architecture.
- `docs/ishell_geometry_design_note.md` — geometry data-model design.
- `docs/ishell_implementation_status.md` — overall iSHELL implementation status.
