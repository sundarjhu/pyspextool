# iSHELL 2DXD Reduction Notes

## Overview

This document describes how echelle orders appear on the iSHELL H2RG detector,
what the diagnostic tools in `pyspextool.ishell.diagnostics` reveal, and how
this work prepares for implementing automated order tracing.

---

## Test Dataset

The calibration data used for these diagnostics are located at:

    data/testdata/ishell_h1_calibrations/raw/

The directory contains raw iSHELL **H1-mode** (≈1.49–1.80 µm) calibration
frames acquired with the instrument in its standard slit configuration:

| File pattern             | Type          | Description                    |
|--------------------------|---------------|--------------------------------|
| `*.flat.*.fits`          | QTH flat      | Quartz–tungsten–halogen lamp   |
| `*.arc.*.fits`           | ThAr arc      | Thorium–argon hollow-cathode   |

These FITS files are stored with Git LFS.  Run `git lfs pull` before trying
to open them.

---

## iSHELL Detector and Raw Data Format

iSHELL is a cross-dispersed echelle spectrograph at NASA IRTF.  It uses a
**Teledyne H2RG 2048 × 2048 near-IR detector**.

Each raw iSHELL FITS file contains **three extensions**:

| Extension | Name     | Contents                                            |
|-----------|----------|-----------------------------------------------------|
| 0         | PRIMARY  | Signal frame S = Σ pedestal reads − Σ signal reads |
| 1         | SUM_PED  | Sum of all pedestal reads                           |
| 2         | SUM_SAM  | Sum of all signal reads                             |

The PRIMARY extension (extension 0) is sufficient for visual diagnostics
and is the only extension read by the functions in
`pyspextool.ishell.diagnostics`.

---

## How iSHELL Orders Appear on the Detector

In a cross-dispersed echelle spectrograph the grating produces multiple
*diffraction orders* that are separated spatially by a cross-dispersing prism.
On the iSHELL detector this results in:

1. **Multiple horizontal bands** across the image.  Each band is one echelle
   order.  The exact number of orders visible depends on the observing mode;
   use the diagnostic plots to determine this empirically for a given dataset.

2. **Dispersion along columns** (left–right).  Wavelength increases from left
   to right within each order.

3. **Spatial extent along rows** (up–down).  The target spectrum runs along
   the centre of each order band; sky and calibration spectra bracket it above
   and below.

4. **Tilt and curvature**.  iSHELL echelle orders are not perfectly horizontal.
   The spectral trace can be tilted and exhibit a slight curvature along the
   dispersion direction.

5. **Curved order edges**.  The upper and lower boundaries of each order follow
   low-order polynomials in column position, not simple horizontal lines.
   Accurate order tracing requires polynomial fitting.

### QTH Flat-Field Frames

A quartz–tungsten–halogen (QTH) lamp illuminates the full slit width and all
echelle orders simultaneously.  The flat-field frame therefore shows:

* **Bright bands** where echelle orders fall.
* **Dark inter-order gaps** between orders.
* A smooth cross-dispersion envelope following the blaze function of the
  echelle.
* No sharp spectral features — making flats ideal for mapping the order
  geometry.

### ThAr Arc Frames

A thorium–argon (ThAr) hollow-cathode lamp produces narrow emission lines
spread across all echelle orders.  The arc frame reveals:

* The same order pattern as the flat, now as stripes of bright dots.
* **Variable line density per order**: line density depends on wavelength
  coverage and the density of ThAr transitions in that region.
* The spatial extent of each slit position.

---

## Diagnostic Functions

### `plot_flat_orders(flat_file, …)`

**Purpose**: Quick visual assessment of the full flat-field frame.

**What it shows**:

* A 2-D ZScale-scaled grey-scale image of the entire detector.  Echelle
  orders appear as horizontal bright bands.
* Optional **horizontal cut lines** (dashed cyan) for marking positions
  found by eye.
* An optional **median column profile** panel showing the cross-dispersion
  brightness envelope.  Peaks in this profile correspond to order centres;
  troughs correspond to inter-order gaps.

**Diagnostic use**:

1. Confirm that all expected orders are illuminated.
2. Identify row positions of order centres for use as seeds in future
   order-tracing routines.
3. Check for bad columns, vignetting, or detector artefacts.

---

### `plot_detector_cross_section(flat_file, column=1024, width=20, …)`

**Purpose**: Robust quantitative view of order structure along the spatial
(row) axis at a chosen detector location.

**What it shows**:

* **Median signal (DN) versus row number**, computed by averaging over a
  band of columns `column - width` to `column + width`.  Using the median
  over multiple columns suppresses hot pixels and cosmic rays that would
  dominate a single-column slice.
* Optional **candidate order-centre markers** (dashed red lines) derived
  from local maxima in the profile.

**Diagnostic use**:

1. Count the number of illuminated orders at the inspection location.
2. Estimate row positions and widths of each order.
3. Obtain initial seed positions for future peak-finding or order-tracing
   algorithms.

---

### `plot_arc_frame(arc_file, …)`

**Purpose**: Visual inspection of the ThAr arc frame.

**What it shows**:

* A ZScale-scaled grey-scale image of the arc frame, with contrast tuned
  to show both bright emission lines and faint inter-order regions.
* Bright vertical streaks indicate high-intensity ThAr emission lines.
* The pattern of lines across orders reveals tilt and curvature of the
  spectral trace.

**Diagnostic use**:

1. Verify that the arc lamp fired correctly and that lines are sharp.
2. Assess line density per order.
3. Check for uniform slit illumination.
4. Visually confirm the dispersion direction.

---

## How These Diagnostics Prepare for Order Tracing

The information extracted visually from these diagnostics directly informs
future automated order-tracing development:

| Input needed for order tracing         | Diagnostic source                      |
|----------------------------------------|----------------------------------------|
| Number of illuminated orders           | `plot_flat_orders` image               |
| Approximate order-centre rows          | `plot_detector_cross_section` peaks    |
| Order spatial width (rows)             | `plot_detector_cross_section` profile  |
| Order tilt and spectral-trace shape    | `plot_arc_frame` line positions        |
| Inter-order gap width                  | `plot_detector_cross_section` troughs  |

### Recommended Workflow

1. Run `plot_flat_orders` to confirm data quality and get a global view.
2. Run `plot_detector_cross_section` at a few column positions (e.g. 512,
   1024, 1536) to characterise order positions across the detector.
3. Run `plot_arc_frame` to verify arc quality and note any spectral-trace
   tilt.
4. Record the candidate order-centre row positions printed by
   `plot_detector_cross_section` as seeds for future order-tracing work.

---

## Running the Diagnostics Script

A convenience script is provided at `scripts/ishell_diagnostics.py`.  Run
it from the repository root::

    python scripts/ishell_diagnostics.py

This automatically loads the first flat and arc files from
`data/testdata/ishell_h1_calibrations/raw/` and runs all three diagnostics.

To save PNG files instead of displaying interactive windows::

    python scripts/ishell_diagnostics.py --save --output-dir /tmp/diag

The diagnostics have been verified by import checks and interactive plotting
with the H1 calibration dataset described above.

---

## References

* iSHELL Spextool Manual v10jan2020, Cushing et al.
* `docs/ishell_fits_layout.md` — iSHELL raw FITS keyword reference.
* `docs/ishell_geometry_design_note.md` — order geometry design notes.
