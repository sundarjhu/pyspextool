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

## Running the benchmark script

From the top-level repository directory:

```bash
python scripts/run_ishell_k3_example.py
```

### Optional arguments

| Flag | Description |
|------|-------------|
| `--raw-dir PATH` | Override the default raw-data directory |
| `--out-dir PATH` | Override the default output directory |
| `--save-plots` | Save QA plots as PNG files instead of displaying them |
| `--no-plots` | Skip all QA plotting |

### Example: save QA plots and specify an alternate output directory

```bash
python scripts/run_ishell_k3_example.py \
    --save-plots \
    --out-dir /tmp/k3_output
```

### Expected output location

By default outputs are written to:

```
data/testdata/ishell_k3_example/output/
```

---

## What the script produces

The script prints a concise stage-by-stage summary and, where implemented,
writes FITS outputs and QA plots:

### FITS outputs

| File | Description |
|------|-------------|
| `wavecal11-12.fits` | Provisional wavelength/spatial calibration product |

### QA plots (Python scaffold)

All plots are labelled *"Python scaffold QA"* to distinguish them from the
IDL Spextool manual figures.

| File | Description |
|------|-------------|
| `qa_flat_orders.png` | Traced order centres overlaid on the combined flat |
| `qa_arc_lines.png` | Traced arc-line seed positions overlaid on the combined arc |
| `qa_wavecal_residuals.png` | Wavelength-solution match residuals (histogram + by-order scatter) |
| `qa_rectified_order.png` | First available rectified order image |

---

## What is currently reproduced in Python

The following stages map to the IDL Spextool manual's K3 walkthrough and
are **implemented and exercised** by the benchmark script:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `tracing.py` | Flat/order-centre tracing from QTH flat frames |
| 2 | `arc_tracing.py` | 2-D arc-line tracing |
| 3 | `wavecal_2d.py` | Provisional per-order wavelength mapping |
| 4 | `wavecal_2d_surface.py` | Global wavelength surface fit |
| 5 | `wavecal_2d_refine.py` | Coefficient-surface refinement |
| 6 | `rectification_indices.py` | Rectification-index generation |
| 7 | `rectified_orders.py` | Rectified order images |
| 8 | `calibration_fits.py` | Write calibration FITS products |

### First-pass observations on K3 data

- The order-tracing stage detects approximately 29 orders from the combined
  K3 flat (vs. the ~27 orders expected for K3 mode).  The scaffold traces
  order centres only; bottom/top edges are approximated from half-width
  estimates.
- Arc-line tracing recovers ~360 lines across the traced orders.
- The provisional wavelength mapping produces per-order polynomial fits;
  some orders receive only a low-degree fit due to few matched lines.
- These are **first-pass scaffold results**, not finalised calibrations.

---

## What is still missing

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

## Running the tests

```bash
pytest tests/test_ishell_k3_example.py -v
```

If the K3 raw files are absent, all data-dependent tests skip cleanly:

```
SKIPPED [reason: K3 raw files not present]
```

The non-data tests (file-format recognition, driver import, error-handling)
always run regardless of whether the K3 data are present.

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
