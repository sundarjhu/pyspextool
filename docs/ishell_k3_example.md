# iSHELL K3 Benchmark Example

This document describes the **K3 benchmark example** for the Python iSHELL
scaffold.  The example exercises the current scaffold code on the canonical K3
dataset from the IDL Spextool manual, and compares the Python reduction flow
to that manual's discussion.

This is **not** a claim of full IDL parity.  The Python scaffold is under
active development and several stages remain unimplemented (see
[What is still missing](#what-is-still-missing)).

---

## Raw data layout

```
data/testdata/ishell_k3_example/
└── raw/
    ├── README.md                                       ← this directory's README
    ├── icm.2017A999.170525.flat.00006.a.fits.gz
    ├── icm.2017A999.170525.flat.00007.a.fits.gz
    ├── icm.2017A999.170525.flat.00008.a.fits.gz
    ├── icm.2017A999.170525.flat.00009.a.fits.gz
    ├── icm.2017A999.170525.flat.00010.a.fits.gz
    ├── icm.2017A999.170525.arc.00011.a.fits.gz
    ├── icm.2017A999.170525.arc.00012.b.fits.gz
    ├── icm.2017A999.170525.dark.00025.a.fits.gz
    │   … (dark.00026 – dark.00029)
    ├── icm.2017A999.170525.spc.00001.a.fits.gz
    │   … (spc.00002 – spc.00005)
    ├── icm.2017A999.170525.spc.00013.a.fits.gz
    │   … (spc.00014 – spc.00017)
    └── …
```

---

## Frame groups from the IDL Spextool manual

| Group            | Frame numbers | IDL output name  |
|------------------|---------------|------------------|
| QTH flat         | 6–10          | `flat6-10`       |
| ThAr arc         | 11–12         | `wavecal11-12`   |
| Dark             | 25–29         | `dark25-29`      |
| Object (spc)     | 1–5           | —                |
| A0 V standard (spc) | 13–17     | —                |

The manual uses **A-Sky/Dark** extraction mode and illustrates the reduction
starting from the flat combination and proceeding through order tracing,
wavelength calibration, and spectral extraction.

---

## Input file format

All K3 files in this repository are stored as **gzip-compressed FITS**
(`.fits.gz`).  The Python benchmark script reads them directly using
`astropy.io.fits` — **no manual decompression is needed**.

The `io_utils` module in the Python scaffold handles both `.fits` and
`.fits.gz` transparently:

```python
from pyspextool.instruments.ishell.io_utils import find_fits_files, is_fits_file

# Both .fits and .fits.gz are returned
files = find_fits_files("data/testdata/ishell_k3_example/raw")
```

Output products written by the benchmark script are always plain `.fits`.

---

## Configuration object

The benchmark driver is controlled by a `K3BenchmarkConfig` dataclass.  All
fields have sensible defaults matching the IDL Spextool manual's canonical K3
example.

```python
from scripts.run_ishell_k3_example import K3BenchmarkConfig, run_k3_example

# Use all defaults
completed = run_k3_example()

# Override specific fields via keyword arguments
completed = run_k3_example(
    raw_dir="/data/K3",
    wavecal_output_name="wavecal11-12",
    no_plots=True,
)

# Pass a fully custom config object
cfg = K3BenchmarkConfig(
    raw_dir="/data/K3",
    output_dir="/tmp/k3_output",
    flat_frames=[6, 7, 8, 9, 10],
    arc_frames=[11, 12],
    wavecal_output_name="wavecal11-12",
    flat_output_name="flat6-10",
    dark_output_name="dark25-29",
    qa_plot_prefix="qa",
    save_plots=True,
    mode_name="K3",
)
completed = run_k3_example(cfg)
```

### K3BenchmarkConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `raw_dir` | str | repo K3 test-data dir | Directory containing raw K3 FITS files |
| `output_dir` | str | `<raw_dir>/../output/` | Output directory for FITS and plots |
| `flat_frames` | list[int] | `[6,7,8,9,10]` | Flat frame sequence numbers |
| `arc_frames` | list[int] | `[11,12]` | Arc frame sequence numbers |
| `dark_frames` | list[int] | `[25,26,27,28,29]` | Dark frame sequence numbers |
| `object_frames` | list[int] | `[1,2,3,4,5]` | Object spc frame numbers |
| `standard_frames` | list[int] | `[13,14,15,16,17]` | A0 V standard spc frame numbers |
| `flat_output_name` | str | `"flat6-10"` | Stem for flat calibration output file |
| `wavecal_output_name` | str | `"wavecal11-12"` | Stem for wavecal output file |
| `dark_output_name` | str | `"dark25-29"` | Stem for dark output file |
| `qa_plot_prefix` | str | `"qa"` | Prefix for all QA plot filenames |
| `save_plots` | bool | `False` | Save plots as PNG instead of displaying |
| `no_plots` | bool | `False` | Skip all QA plotting |
| `mode_name` | str | `"K3"` | iSHELL mode for calibration resource lookup |
| `export_diagnostics` | bool | `False` | Export per-order diagnostics to CSV/JSON |
| `diagnostics_format` | str | `"csv"` | Format for diagnostics export: `"csv"` or `"json"` |

---

## Running the benchmark script

From the top-level repository directory:

```bash
python scripts/run_ishell_k3_example.py
```

### CLI arguments

| Flag | Description |
|------|-------------|
| `--raw-dir PATH` | Override the default raw-data directory |
| `--out-dir PATH` | Override the default output directory |
| `--flat-frames INTS` | Comma-separated flat frame numbers (e.g. `6,7,8,9,10`) |
| `--arc-frames INTS` | Comma-separated arc frame numbers (e.g. `11,12`) |
| `--dark-frames INTS` | Comma-separated dark frame numbers (e.g. `25,26,27,28,29`) |
| `--object-frames INTS` | Comma-separated object spc frame numbers |
| `--standard-frames INTS` | Comma-separated standard spc frame numbers |
| `--flat-output-name NAME` | Stem for flat calibration output (default: `flat6-10`) |
| `--wavecal-output-name NAME` | Stem for wavecal output (default: `wavecal11-12`) |
| `--dark-output-name NAME` | Stem for dark output (default: `dark25-29`) |
| `--qa-plot-prefix PREFIX` | Prefix for QA plot filenames (default: `qa`) |
| `--mode-name MODE` | iSHELL mode (default: `K3`) |
| `--save-plots` | Save QA plots as PNG files instead of displaying them |
| `--no-plots` | Skip all QA plotting |
| `--export-diagnostics` | Export per-order calibration diagnostics to file |
| `--diagnostics-format FMT` | Format for export: `csv` or `json` (default: `csv`) |

### Examples

Save QA plots and specify an alternate output directory:

```bash
python scripts/run_ishell_k3_example.py \
    --save-plots \
    --out-dir /tmp/k3_output
```

Use IDL Spextool manual output-name conventions explicitly:

```bash
python scripts/run_ishell_k3_example.py \
    --wavecal-output-name wavecal11-12 \
    --flat-output-name flat6-10 \
    --no-plots
```

Custom output prefix for QA plots:

```bash
python scripts/run_ishell_k3_example.py \
    --save-plots \
    --qa-plot-prefix bench
```

### Expected output location

By default outputs are written to:

```
data/testdata/ishell_k3_example/output/
```

---

## What the script produces

The script prints a concise stage-by-stage summary and, where implemented,
writes FITS outputs and QA plots.

### FITS outputs

The wavecal output filename is controlled by `--wavecal-output-name` (default
`wavecal11-12`).  A companion spatcal file is written alongside it.

| File (default names) | Description |
|----------------------|-------------|
| `wavecal11-12.fits` | Provisional wavelength calibration product |
| `wavecal11-12_spatcal.fits` | Provisional spatial calibration product |

### QA plots (Python scaffold)

All plots are labelled *"Python scaffold QA"* to distinguish them from the
IDL Spextool manual figures.  The prefix is controlled by `--qa-plot-prefix`
(default `qa`).

| File (default prefix) | Description | Manual analogue |
|-----------------------|-------------|-----------------|
| `qa_flat_orders.png` | Smooth fitted order-centre curves overlaid on the combined flat (27 science orders) | — |
| `qa_arc_lines.png` | Traced arc-line seed positions overlaid on the combined arc | — |
| `qa_wavecal_residuals.png` | Residuals vs order/column (accepted=blue, rejected=red×), histogram, per-order line count with RMS labels | **Figure 3** (1DXD QA plot) |
| `qa_2d_coeff_fit.png` | Arc-line position cloud, tilt-slope map, wavelength solution curves, residual map | **Figures 4–7** (2DCoeffFit.pdf, partial) |
| `qa_rectified_order.png` | First available rectified order flux image | — |

See [K3 order-filtering rule](#k3-order-filtering-rule) for why only 27 orders
appear in the flat-orders plot.

---

## What is currently reproduced in Python

The following stages map to the IDL Spextool manual's K3 walkthrough and
are **implemented and exercised** by the benchmark script:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `tracing.py` | Flat/order-centre tracing from QTH flat frames |
| 2 | `arc_tracing.py` | 2-D arc-line tracing |
| **2b** | **`wavecal_k3_idlstyle.py`** | **IDL-style global 1DXD wavelength calibration (NEW PRIMARY K3 path)** |
| 3 | `wavecal_2d.py` | Provisional per-order wavelength mapping (scaffold, retained for downstream rectification) |
| 4 | `wavecal_2d_surface.py` | Global wavelength surface fit |
| 5 | `wavecal_2d_refine.py` | Coefficient-surface refinement |
| 6 | `rectification_indices.py` | Rectification-index generation |
| 7 | `rectified_orders.py` | Rectified order images |
| 8 | `calibration_fits.py` | Write calibration FITS products |

### New Stage 2b: IDL-style global 1DXD wavelength calibration

Stage 2b is the new **primary K3 wavelength-calibration path** that replaces
the old per-order scaffold path for the K3 benchmark.  It follows the IDL
Spextool sequence more closely:

1. **Extract 1-D arc spectra per order** from the median-combined arc image,
   using the order centre-row geometry.
2. **Cross-correlate with the stored reference spectrum** (plane 1 of
   `K3_wavecalinfo.fits`) to estimate a per-order pixel offset.
3. **Identify and centroid arc lines in 1-D** using shifted expected positions
   from the cross-correlation step.
4. **Fit a global 1DXD wavelength solution** across all accepted K3 line
   centroids together:
   ```
   lambda = f(column, order)
   ```
   with fixed degrees `lambda_degree=3`, `order_degree=2`.
5. **Apply iterative sigma-clipping** (`nsigma=3.0`, up to 5 iterations) in
   the global fit.

The module lives in:

```
src/pyspextool/instruments/ishell/wavecal_k3_idlstyle.py
```

For the architectural design rationale, see
[docs/ishell_k3_1dxd_design.md](ishell_k3_1dxd_design.md).

### First-pass observations on K3 data

- The order-tracing stage detects 29 orders from the combined K3 flat, of
  which 27 are retained as science orders (203–229) after the benchmark
  edge-order filter (see [K3 order-filtering rule](#k3-order-filtering-rule)).
  The 2 dropped orders are partial edge orders clipped at the detector boundary.
- Arc-line tracing recovers ~300–360 lines across the 27 science orders.
- The provisional wavelength mapping produces per-order polynomial fits;
  some orders receive only a low-degree fit due to few matched lines.
  Per-order accepted-line counts and polynomial degrees are printed in the
  benchmark output.
- These are **first-pass scaffold results**, not finalised calibrations.

---

## K3 order-filtering rule

**Benchmark-only rule**: the flat-field tracing algorithm occasionally
detects partially visible orders at the top and bottom edges of the detector.
These manifest as orders with a spatial half-width (`half_width_rows`)
substantially smaller than the typical science-order half-width.

The benchmark applies the following conservative filter (implemented in
`_filter_edge_orders`):

> Exclude any order whose `half_width_rows` is below **30 % of the median
> half-width** across all detected orders.

For K3 data, this criterion removes the 2 partial edge orders (indices 0 and
28 out of 29 detected) whose half-widths are ~2.5 px versus the typical
~16 px for science orders.  The retained 27 orders correspond to the IDL
Spextool manual's K3 science orders 203–229.

This rule is:
- documented and reproducible
- grounded in the actual tracing geometry (not a hard-coded index drop)
- conservative: only truly clipped orders are excluded

---

---

## Calibration diagnostics and failure modes

The K3 benchmark driver produces a **structured per-order diagnostics report**
during each run.  This section explains what the diagnostics capture, how to
interpret them, and what the known failure modes mean.

### Running with diagnostics export

To export the per-order diagnostics as a CSV or JSON file:

```bash
python scripts/run_ishell_k3_example.py \
    --no-plots \
    --export-diagnostics \
    --diagnostics-format csv
```

This writes `<output_dir>/qa_order_diagnostics.csv` (or `.json`).  The default
prefix `qa` is controlled by `--qa-plot-prefix`.

### Diagnostics table fields

The console table printed at the end of every run contains the following
columns:

| Column | Description |
|--------|-------------|
| `Order` | Echelle order number (e.g. 203–229 for K3) |
| `Cand` | Candidate traced arc lines considered for matching |
| `Acc` | Accepted matches used in the polynomial fit |
| `Rej` | Rejected/unmatched lines (`Cand − Acc`) |
| `DegReq` | Polynomial degree that was requested |
| `DegUsd` | Polynomial degree actually used (may be reduced) |
| `RMS(nm)` | Fit RMS in nm (`NaN` when the order was skipped) |
| `Skip` | `YES` when the order had too few lines to fit |
| `NMono` | `YES` when rectification found a non-monotonic wavelength surface |

Orders flagged with `◀` in the table are problematic and listed in the
**Weak-order summary** section below the table.

### Interpreting low line counts

A **low accepted-line count** (`Acc < 3`) means the coarse wavelength
reference from `WaveCalInfo` plane 0 did not predict the correct wavelength
for most traced lines within the matching tolerance (`match_tol_um = 0.005 µm`
by default).

Common causes:
- The coarse reference grid is locally offset from the true dispersion relation
  (especially at the bluemost/redmost orders where the reference is less reliable).
- Arc lines in that order are blended or very faint.
- The traced line positions are shifted (e.g. due to detector flexure).

**Effect**: orders with fewer than `min_lines_per_order = 2` accepted matches
are *skipped* (`wave_coeffs = None`); orders with exactly 2–3 matches receive
a low-degree polynomial fit.

### Interpreting polynomial degree reduction

When fewer than `dispersion_degree + 1` matches are accepted, the polynomial
degree is automatically reduced to `n_accepted − 1`.  For example, with
3 accepted matches and `dispersion_degree = 3`, the fit is reduced to degree 2.

**Effect**: A lower-degree polynomial has fewer degrees of freedom, so it
captures only the gross wavelength trend and not any curvature in the
dispersion relation.  This degrades wavelength accuracy across the order.

### Interpreting non-monotonic wavelength surfaces

A **non-monotonic wavelength surface** is flagged when the global coefficient
surface (computed during Stage 5) predicts wavelengths that are not strictly
increasing (or decreasing) across the detector columns for a given order.

The `rectification_indices` module emits a `RuntimeWarning` for each such
order, and the benchmark driver captures and records these.

Common causes:
- Too few accepted lines in affected orders → the global surface fit
  extrapolates poorly into poorly-constrained orders.
- The per-order polynomial degree was reduced, leaving the global surface
  under-constrained locally.

**Effect**: The column-inversion for rectification (converting from column
space to wavelength space) may be inaccurate for these orders.  Rectified
spectra from non-monotonic orders should be treated as unreliable.

### Interpreting the improved residuals QA plot (`qa_wavecal_residuals.png`)

The residuals plot has been enhanced to show:

- **Blue dots** — accepted matched lines (used in the polynomial fit).
- **Red × markers** at `residual = 0` — rejected or unmatched lines.
- **Bar labels** on the per-order count chart — per-order RMS in nm.

The top two panels (residuals vs order number; residuals vs column) show the
full picture of accepted vs rejected lines.  A large number of red markers
relative to blue markers indicates a poorly-calibrated order.

### Light-touch deduplication of arc lines

The provisional wavelength mapper (`wavecal_2d.py`) now applies a
**minimum column separation filter** (`min_col_separation = 5.0 pixels` by
default) before matching.  When two detected lines in the same order have seed
columns within this threshold, only the one with the higher peak flux is kept.

This removes spurious near-duplicate detections (e.g. from split or blended
emission lines) that would otherwise consume reference wavelength slots and
prevent genuine distinct lines from matching.

---



The following IDL Spextool manual stages are **not yet implemented** in the
Python iSHELL scaffold:

| Stage | Description |
|-------|-------------|
| Dark subtraction | Dark-frame processing and dark subtraction from science frames |
| Science preprocessing | Full preprocessing of K3 object and standard frames beyond the current stub |
| Spectral extraction | Aperture / optimal / profile extraction on K3 object frames |
| Standard extraction | Extraction of the A0 V standard frames (spc 13–17) |
| Telluric correction | Correction of K3 spectra using the A0 V standard |
| Order merging | Stitching per-order 1-D spectra into a single merged spectrum |

These missing stages mean that the Python benchmark cannot yet reproduce the
full K3 reduction flow described in the IDL Spextool manual.

---

## Remaining calibration mismatches vs the IDL manual

The following mismatches are **known and documented** as of this benchmark
pass.  They reflect scaffold-level limitations, not incorrect code:

### New 1DXD path (Stage 2b — primary K3 route)

| Issue | Description |
|-------|-------------|
| Cross-correlation reliability | Per-order pixel shifts from cross-correlation may be inaccurate for orders where the reference arc spectrum (plane 1 of `K3_wavecalinfo.fits`) has low signal or poor overlap with the real arc image.  Robust shift estimation (sub-pixel Gaussian fitting of the XCorr peak) would improve this. |
| Centre-row estimation | The 1-D spectrum extraction estimates the order centre row from a global-maximum profile of the cross-dispersion median.  A flexure-corrected traced centre row would be more accurate. |
| Line centroiding | Gaussian fitting is used with a fixed window; for blended or very faint lines, the centroid may be unreliable. |
| Residual scale (1DXD) | Until real K3 data are exercised, the global fit RMS is not verified against the IDL manual's Figure 3 benchmark (RMS ~0.019 Å = 0.0019 nm). |

### Scaffold path (Stages 3–8 — retained for downstream rectification)

| Issue | Description |
|-------|-------------|
| 2DCoeffFit analogue | The `qa_2d_coeff_fit.png` is a partial analogue only.  No model surface overlay, no curvature coefficient panel, and no per-line rejection indicator are currently available. |
| Non-monotonic wavelength surfaces | Several orders show non-monotonic wavelength surfaces during rectification (logged as `RuntimeWarning`).  These are expected artefacts of the provisional coefficient surface and not a bug in rectification logic. |
| Order polynomial degree reduction | Some orders in the scaffold path use a lower polynomial degree than requested because too few arc lines were accepted.  This degrades the local wavelength solution accuracy. |

---

## Old scaffold path vs new 1DXD path

| Aspect | Old scaffold (Stages 3–8) | New IDL-style path (Stage 2b) |
|--------|--------------------------|-------------------------------|
| Primary calibration model | Per-order polynomial fits | Global 1DXD surface `f(col, order)` |
| Source of λ-IDs | 2-D seed-based direct line matching | 1-D centroiding after cross-correlation shift |
| Weak-order handling | Per-order degree reduction | Global fit regularises via cross-order smoothness |
| Outlier rejection | None | Iterative σ-clipping in global fit |
| Polynomial degrees | Per-order selected automatically | Fixed: lambda=3, order=2 |
| QA plot | `qa_wavecal_residuals.png` | `qa_1dxd_qa.png` (residuals + per-order stats) |

---

## Running the tests

```bash
pytest tests/test_ishell_k3_example.py -v
```

If the K3 raw files are absent, all data-dependent tests skip cleanly:

```
SKIPPED [reason: K3 raw files not present]
```

The non-data tests (file-format recognition, driver import, config defaults,
error-handling) always run regardless of whether the K3 data are present.

---

## Relationship to the IDL Spextool manual

The IDL Spextool manual (see `docs/reference/ishell_atlases/spextoolmanual.pdf`)
provides the authoritative description of the full iSHELL reduction workflow.
This Python benchmark is intended to:

1. Give developers a concrete, runnable exercise of the Python scaffold on
   real K3 data.
2. Make it easy to compare scaffold outputs to the manual's QA figures.
3. Track which stages are implemented and which remain outstanding.

The QA plots produced by this script are labelled *"Python scaffold QA"* and
should not be assumed to match the IDL manual's figures exactly.

