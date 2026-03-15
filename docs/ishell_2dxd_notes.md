# iSHELL 2DXD Reduction Notes

## Overview

This document describes how echelle orders appear on the iSHELL H2RG detector,
what the diagnostic tools in `pyspextool.ishell.diagnostics` reveal, and how
this work prepares for implementing automated order tracing.

---

## iSHELL Detector and Raw Data

iSHELL is a cross-dispersed echelle spectrograph at NASA IRTF.  It uses a
**Teledyne H2RG 2048 × 2048 near-IR detector** operating across the
J (1.1–1.4 µm), H (1.5–1.8 µm) and K (2.0–2.4 µm) bands.

### Raw FITS Format

Each raw iSHELL FITS file contains **three extensions**:

| Extension | Name     | Contents                                            |
|-----------|----------|-----------------------------------------------------|
| 0         | PRIMARY  | Signal frame S = Σ pedestal reads − Σ signal reads |
| 1         | SUM_PED  | Sum of all pedestal reads                           |
| 2         | SUM_SAM  | Sum of all signal reads                             |

The PRIMARY extension is sufficient for visual diagnostics and is the only
extension read by the functions in `pyspextool.ishell.diagnostics`.

### Supported H-Band Sub-modes

This test dataset uses H1 mode (1.49–1.80 µm).  The full set of iSHELL
sub-modes supported by pySpextool is:

* **J-band**: J0, J1, J2, J3
* **H-band**: H1, H2, H3
* **K-band**: K1, K2, Kgas, K3

---

## How iSHELL Orders Appear on the Detector

In a cross-dispersed echelle spectrograph the grating produces multiple
*diffraction orders* that are separated spatially by a cross-dispersing prism.
On the iSHELL detector this results in:

1. **Multiple horizontal bands** across the image.  Each band is one echelle
   order.  In H1 mode approximately six to eight orders are visible.

2. **Dispersion along columns** (left–right).  Wavelength increases from left
   to right within each order.

3. **Spatial extent along rows** (up–down).  The target spectrum runs along
   the centre of each order band; sky and calibration spectra bracket it above
   and below.

4. **Tilt and curvature**.  iSHELL echelle orders are not perfectly horizontal.
   The spectral trace is tilted by several degrees and exhibits a slight
   curvature along the dispersion direction.  This is more pronounced in
   shorter-wavelength modes.

5. **Curved order edges**.  The upper and lower boundaries of each order are
   not straight lines; they follow low-order polynomials in column position.
   Accurate order tracing therefore requires polynomial fitting, not simple
   row-centroid tracking.

### QTH Flat-Field Illumination

A quartz–tungsten–halogen (QTH) lamp illuminates the full slit width and all
echelle orders simultaneously.  The flat-field frame therefore shows:

* **Bright bands** where echelle orders fall.
* **Dark inter-order gaps** between orders.
* A smooth cross-dispersion envelope that follows the blaze function of the
  echelle.
* No sharp spectral features — making flats ideal for mapping the order
  geometry.

### ThAr Arc Frame

A thorium–argon (ThAr) hollow-cathode lamp produces thousands of narrow
emission lines.  The arc frame reveals:

* The same order pattern as the flat (now appearing as stripes of bright dots
  rather than continuous bands).
* **Line density varies by order**: shorter-wavelength orders contain more
  ThAr lines per unit wavelength interval than longer-wavelength orders.
* The spatial extent of each slit in both the target and calibration
  positions.

---

## Diagnostic Functions

### `plot_flat_orders(flat_file, …)`

**Purpose**: Quick visual assessment of the flat-field frame.

**What it shows**:

* A 2-D ZScale-scaled grey-scale image of the entire detector.  Echelle
  orders appear as horizontal bright bands.
* Optional **horizontal cut lines** (dashed cyan) for marking order centres.
* An optional **median column profile** panel showing the cross-dispersion
  brightness envelope.  Peaks in this profile correspond to order centres;
  troughs correspond to inter-order gaps.

**Diagnostic use**:

1. Confirm that all expected orders are illuminated.
2. Estimate the row positions of order centres (needed as seeds for order
   tracing).
3. Check for bad columns, vignetting, or other detector artefacts.

---

### `plot_detector_cross_section(flat_file, column=None, …)`

**Purpose**: Quantitative view of order structure at a chosen column.

**What it shows**:

* Intensity (DN) versus row number at a single detector column.
* By default the **central column** (column 1024) is used; pass `column=N`
  to inspect a different location.
* Echelle orders appear as peaks; inter-order gaps are the dips between
  them.

**Diagnostic use**:

1. Count the number of illuminated orders.
2. Measure the approximate row position and FWHM of each order at the
   chosen column.
3. Assess order overlap (if any) at the detector edges.
4. Provide initial seed positions for automated order-finding algorithms
   (e.g. peak-finding on the cross-section profile).

---

### `plot_arc_frame(arc_file, …)`

**Purpose**: Visual inspection of the ThAr arc frame.

**What it shows**:

* A ZScale-scaled grey-scale image of the arc frame, with contrast tuned
  to show both bright lines and faint inter-order regions.
* Bright vertical streaks indicate high-intensity ThAr emission lines.
* The cross-order pattern of lines reveals the tilt and curvature of the
  spectral trace within each order.

**Diagnostic use**:

1. Verify that arc lamps fired correctly and that lines are sharp.
2. Assess line density per order (important for wavelength calibration
   quality).
3. Check that the slit is uniformly illuminated (no vignetting).
4. Visually confirm the dispersion direction (wavelength increasing left
   to right).

---

## How These Diagnostics Prepare for Order Tracing

Automated order tracing requires several inputs that can be estimated
directly from these diagnostic plots:

| Input                    | Source diagnostic                  |
|--------------------------|------------------------------------|
| Number of orders         | `plot_flat_orders` image            |
| Approximate order centres (rows) | `plot_detector_cross_section` peaks |
| Order extent (FWHM in rows)    | `plot_detector_cross_section`  |
| Order tilt and curvature | `plot_arc_frame` line positions    |
| Inter-order gap width    | `plot_detector_cross_section` dips |

Once these parameters have been estimated visually, they can be fed into
the order-tracing algorithms in `pyspextool.instruments.ishell.geometry`
as initial guesses, enabling automated polynomial edge fitting.

### Recommended Workflow

1. Run `plot_flat_orders` to confirm data quality.
2. Run `plot_detector_cross_section` at several columns (e.g. 512, 1024,
   1536) to characterise order positions across the dispersion range.
3. Run `plot_arc_frame` to verify arc quality and inspect tilt.
4. Use the extracted order-centre estimates as seeds for
   `pyspextool.instruments.ishell.geometry.locate_orders`.

---

## Running the Diagnostics Script

A convenience script is provided at `scripts/ishell_diagnostics.py`.  To
run it from the repository root::

    python scripts/ishell_diagnostics.py

This automatically loads the first flat and arc files from::

    data/testdata/ishell_h1_calibrations/raw/

To save PNG files instead of displaying interactive windows::

    python scripts/ishell_diagnostics.py --save --output-dir /tmp/diag

> **Note**: The test FITS files are stored with Git LFS.  Run
> `git lfs pull` to download them before executing the script.

---

## References

* iSHELL Spextool Manual v10jan2020, Cushing et al.
* Vacca et al. (2004), PASP 116, 352 — SpeX spectral reduction procedures.
* `docs/ishell_geometry_design_note.md` — order geometry design for iSHELL.
* `docs/ishell_fits_layout.md` — iSHELL raw FITS keyword reference.
