# iSHELL K3 Example Raw Data

This directory holds the raw FITS files for the canonical **K3 benchmark
example** taken from the IDL Spextool manual.

---

## Expected file groups

The manual uses the following frame-number conventions (zero-padded to 5
digits in the actual filenames).

| Group       | Frame numbers | File pattern (example)                                    |
|-------------|--------------|-----------------------------------------------------------|
| Flat        | 6–10         | `icm.*.flat.00006.a.fits.gz` … `icm.*.flat.00010.a.fits.gz` |
| Arc         | 11–12        | `icm.*.arc.00011.a.fits.gz`, `icm.*.arc.00012.b.fits.gz`    |
| Dark        | 25–29        | `icm.*.dark.00025.a.fits.gz` … `icm.*.dark.00029.a.fits.gz` |
| Object (spc)| 1–5          | `icm.*.spc.00001.a.fits.gz` … `icm.*.spc.00005.a.fits.gz`   |
| Standard A0V (spc) | 13–17 | `icm.*.spc.00013.a.fits.gz` … `icm.*.spc.00017.a.fits.gz` |

The nominal IDL Spextool reduction parameters for these frames are:

| Product      | Output name      |
|--------------|------------------|
| Flat         | `flat6-10`       |
| Wavecal      | `wavecal11-12`   |
| Dark         | `dark25-29`      |
| Extraction mode | A-Sky/Dark    |
| Example object image | 1        |

---

## File format

The K3 files in this repository are stored as **gzip-compressed FITS**
(`.fits.gz`).  The Python benchmark script reads them directly via
`astropy.io.fits`; **no manual decompression is required**.

Output products written by the benchmark script are always plain `.fits`.

---

## Where outputs go

The benchmark driver script (`scripts/run_ishell_k3_example.py`) writes
all output FITS files and QA plots to an output directory of your choice.

**Default location** (used when no override is given):

    data/testdata/ishell_k3_example/output/

That directory is created automatically on first run.

**Override the output directory** at the command line:

    python scripts/run_ishell_k3_example.py --out-dir /path/to/my/output

Or via the Python API:

    from scripts.run_ishell_k3_example import K3BenchmarkConfig, run_k3_example

    run_k3_example(output_dir="/path/to/my/output")
    # or
    cfg = K3BenchmarkConfig(output_dir="/path/to/my/output")
    run_k3_example(cfg)

See `docs/ishell_k3_example.md` for a full list of configurable fields and
CLI flags.

---

## Obtaining the data

The `.fits.gz` files in this directory are part of the repository's tracked
assets.  If they are absent (e.g. after a shallow clone), you may need to
run `git lfs pull` or re-download them separately.

The benchmark script (`scripts/run_ishell_k3_example.py`) will exit with a
clear error message if the required input files are not found.
