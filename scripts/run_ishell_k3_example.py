#!/usr/bin/env python
"""
run_ishell_k3_example.py — iSHELL K3 benchmark driver.

This script exercises the current Python iSHELL scaffold on the canonical
K3 example dataset from the IDL Spextool manual.  It is a
**benchmark-oriented driver**, not a claim of full IDL parity.

The K3 manual uses:
  - flat frames     : 6–10
  - arc frames      : 11–12
  - dark frames     : 25–29
  - object spc      : 1–5
  - A0 V standard   : spc 13–17

Usage
-----
Run from the top-level repository directory::

    python scripts/run_ishell_k3_example.py

Optional flags::

    --raw-dir            PATH   Override the default raw-data directory.
    --out-dir            PATH   Override the default output directory.
    --flat-frames        INTS   Comma-separated flat frame numbers (default: 6,7,8,9,10).
    --arc-frames         INTS   Comma-separated arc frame numbers  (default: 11,12).
    --dark-frames        INTS   Comma-separated dark frame numbers (default: 25,26,27,28,29).
    --object-frames      INTS   Comma-separated object spc frame numbers (default: 1,2,3,4,5).
    --standard-frames    INTS   Comma-separated standard spc frame numbers (default: 13,14,15,16,17).
    --flat-output-name   NAME   Stem for flat calibration output file (default: flat6-10).
    --wavecal-output-name NAME  Stem for wavecal output file (default: wavecal11-12).
    --dark-output-name   NAME   Stem for dark output file (default: dark25-29).
    --qa-plot-prefix     PREFIX Prefix for QA plot filenames (default: qa).
    --mode-name          MODE   iSHELL mode (default: K3).
    --save-plots                Save QA plots as PNG files (default: display them).
    --no-plots                  Skip all QA plotting.
    --export-diagnostics        Export per-order calibration diagnostics file.
    --diagnostics-format FMT    Format for diagnostics export: csv or json (default: csv).

All input FITS files may be either ``.fits`` or ``.fits.gz``.
Output FITS files are always written as plain ``.fits``.

Data location (defaults)
------------------------
  Input  : data/testdata/ishell_k3_example/raw/
  Output : data/testdata/ishell_k3_example/output/

Implemented stages
------------------
  Stage 1  – Flat/order-centre tracing    (tracing.py)
  Stage 2  – Arc-line tracing             (arc_tracing.py)
  Stage 3  – Provisional wavelength map   (wavecal_2d.py)          [scaffold]
  Stage 3b – K3 1DXD wavelength model     (wavecal_k3_idlstyle.py) [K3 primary path]
  Stage 4  – Global wavelength surface    (wavecal_2d_surface.py)  [scaffold]
  Stage 5  – Coefficient refinement       (wavecal_2d_refine.py)   [scaffold]
  Stage 6  – Rectification indices        (rectification_indices.py)
             (driven by Stage 3b K3 1DXD model; falls back to Stage 5 if 3b fails)
  Stage 7  – Rectified order images       (rectified_orders.py)
  Stage 8  – Calibration FITS products    (calibration_fits.py)
  QA plots – Python scaffold QA figures   (matplotlib)

K3 calibration path
-------------------
  For the K3 benchmark, Stage 3b (wavecal_k3_idlstyle.py) is the PRIMARY
  source of truth for wavelength calibration.  It:
    1. Extracts 1-D arc spectra using the traced flat-field order geometry.
    2. Fits a global (wdeg=3, odeg=2) 1DXD polynomial across all orders.
  The Stage 3b model drives Stage 6 rectification and all downstream outputs.
  Stages 3–5 (scaffold) run for reference and diagnostics but do NOT drive
  K3 results.

Not yet implemented in Python
------------------------------
  - Dark subtraction / dark-frame processing
  - Science-frame preprocessing beyond the scaffold stub (preprocess.py)
  - Aperture / optimal spectral extraction on K3 object frames
  - Telluric correction of K3 science spectra
  - Order merging
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Ensure the package is importable when running from the repo root.
# ---------------------------------------------------------------------------

_repo_src = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

from pyspextool.instruments.ishell.tracing import (  # noqa: E402
    FlatOrderTrace,
    load_and_combine_flats,
    trace_orders_from_flat,
)
from pyspextool.instruments.ishell.arc_tracing import (  # noqa: E402
    load_and_combine_arcs,
    trace_arc_lines,
)
from pyspextool.instruments.ishell.wavecal_2d import (  # noqa: E402
    fit_provisional_wavelength_map,
)
from pyspextool.instruments.ishell.wavecal_2d_surface import (  # noqa: E402
    fit_global_wavelength_surface,
)
from pyspextool.instruments.ishell.wavecal_2d_refine import (  # noqa: E402
    fit_refined_coefficient_surface,
)
from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (  # noqa: E402
    extract_order_arc_spectra,
    fit_1dxd_wavelength_model,
)
from pyspextool.instruments.ishell.rectification_indices import (  # noqa: E402
    build_rectification_indices,
)
from pyspextool.instruments.ishell.rectified_orders import (  # noqa: E402
    build_rectified_orders,
)
from pyspextool.instruments.ishell.calibration_products import (  # noqa: E402
    build_calibration_products,
)
from pyspextool.instruments.ishell.calibration_fits import (  # noqa: E402
    write_wavecal_fits,
    write_spatcal_fits,
)
from pyspextool.instruments.ishell.calibrations import (  # noqa: E402
    read_wavecalinfo,
    read_line_list,
)
from pyspextool.instruments.ishell.io_utils import (  # noqa: E402
    find_fits_files,
    ensure_fits_suffix,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_repo_root() -> str:
    """Return the absolute path to the repository root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def _default_raw_dir() -> str:
    return os.path.join(
        _find_repo_root(),
        "data", "testdata", "ishell_k3_example", "raw",
    )


def _default_out_dir() -> str:
    return os.path.join(
        _find_repo_root(),
        "data", "testdata", "ishell_k3_example", "output",
    )


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class K3BenchmarkConfig:
    """Configuration for the K3 benchmark reduction driver.

    All fields have sensible defaults that match the IDL Spextool manual's
    canonical K3 example.  Override individual fields to customise the run.

    Parameters
    ----------
    raw_dir : str
        Directory containing the raw K3 FITS files (``.fits`` or
        ``.fits.gz``).  Defaults to the repository's bundled K3 test-data
        directory.
    output_dir : str
        Directory where output FITS files and QA plots are written.
        Created automatically if absent.  Defaults to a sibling ``output/``
        directory inside the K3 test-data tree.
    flat_frames : list of int
        Frame sequence numbers for the QTH flat exposures.
        IDL manual default: 6–10.
    arc_frames : list of int
        Frame sequence numbers for the ThAr arc exposures.
        IDL manual default: 11–12.
    dark_frames : list of int
        Frame sequence numbers for the dark exposures.
        IDL manual default: 25–29.
    object_frames : list of int
        Frame sequence numbers for the science object exposures.
        IDL manual default: 1–5.
    standard_frames : list of int
        Frame sequence numbers for the A0 V standard exposures.
        IDL manual default: 13–17.
    flat_output_name : str
        Stem (without ``.fits``) for the flat calibration output file.
        IDL manual convention: ``"flat6-10"``.
    wavecal_output_name : str
        Stem (without ``.fits``) for the wavecal output file.
        IDL manual convention: ``"wavecal11-12"``.
    dark_output_name : str
        Stem (without ``.fits``) for the dark output file.
        IDL manual convention: ``"dark25-29"``.
    qa_plot_prefix : str
        Prefix prepended to all QA plot filenames, e.g. ``"qa"`` produces
        ``qa_flat_orders.png``, ``qa_arc_lines.png``, etc.
    save_plots : bool
        If ``True``, save QA plots as PNG files instead of displaying them
        interactively.
    no_plots : bool
        If ``True``, skip all QA plotting entirely.
    mode_name : str
        iSHELL observing mode name used to look up packaged calibration
        resources (flat-info, wavecal-info, line list).  Default: ``"K3"``.
    """

    raw_dir: str = field(default_factory=_default_raw_dir)
    output_dir: str = field(default_factory=_default_out_dir)

    # Frame-number groups (IDL Spextool manual K3 defaults)
    flat_frames: list[int] = field(
        default_factory=lambda: [6, 7, 8, 9, 10]
    )
    arc_frames: list[int] = field(
        default_factory=lambda: [11, 12]
    )
    dark_frames: list[int] = field(
        default_factory=lambda: [25, 26, 27, 28, 29]
    )
    object_frames: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5]
    )
    standard_frames: list[int] = field(
        default_factory=lambda: [13, 14, 15, 16, 17]
    )

    # Output naming (IDL manual convention)
    flat_output_name: str = "flat6-10"
    wavecal_output_name: str = "wavecal11-12"
    dark_output_name: str = "dark25-29"

    # QA plot prefix
    qa_plot_prefix: str = "qa"

    # Plot behaviour
    save_plots: bool = False
    no_plots: bool = False

    # iSHELL mode
    mode_name: str = "K3"

    # Diagnostics export
    export_diagnostics: bool = False
    diagnostics_format: str = "csv"  # "csv" or "json"


# ---------------------------------------------------------------------------
# File selection helper
# ---------------------------------------------------------------------------


def _select_files(
    all_files: list,
    frame_type: str,
    frame_numbers: list[int],
) -> list[str]:
    """Return sorted file paths whose name matches *frame_type* and frame numbers.

    Parameters
    ----------
    all_files : list of Path
        Candidate files (from :func:`find_fits_files`).
    frame_type : str
        Sub-string to match in the filename (e.g. ``"flat"``, ``"arc"``).
    frame_numbers : list of int
        Frame sequence numbers to include (matched against the zero-padded
        5-digit run number in the iSHELL filename convention).

    Returns
    -------
    list of str
        Sorted absolute paths of matched files.
    """
    matched: list[str] = []
    for p in all_files:
        name = p.name
        if frame_type not in name:
            continue
        for n in frame_numbers:
            if f".{n:05d}." in name:
                matched.append(str(p))
                break
    return sorted(matched)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------


def _banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def _ok(stage: str) -> None:
    print(f"  [OK]  {stage}")


def _skip(stage: str, reason: str) -> None:
    print(f"  [--]  {stage}  ({reason})")


def _parse_non_monotonic_order_number(warning_message: str) -> int | None:
    """Extract the echelle order number from a non-monotonic RuntimeWarning.

    The warning emitted by ``rectification_indices`` has the form::

        "Order 1 (order_number=204.0): the sampled wavelength surface …"

    The first integer (``1``) is the *order index*, not the echelle order
    number.  The echelle order number is the value after ``order_number=``
    (e.g. ``204.0`` → ``204``).  This helper extracts that value.

    Returns ``None`` if the pattern is not found, so callers can safely
    skip unparseable messages without crashing.
    """
    # Match  order_number=<digits>[.<digits>]  robustly for both 204 and 204.0
    m = re.search(r"order_number=([0-9]+(?:\.[0-9]*)?)", warning_message)
    if m is None:
        return None
    try:
        return int(float(m.group(1)))
    except (ValueError, OverflowError):
        return None


@dataclass
class OrderCalibrationDiagnostics:
    """Per-order calibration diagnostic record.

    Collects information about the quality of the wavelength calibration for
    one echelle order.  A list of these is built by :func:`run_k3_example`
    and printed as a console table; optionally exported to CSV or JSON.

    Parameters
    ----------
    order_number : int
        Echelle order number (e.g. 203–229 for K3 mode).
    n_candidate : int
        Number of traced arc lines considered as match candidates.
    n_accepted : int
        Number of lines accepted and used in the polynomial fit.
    n_rejected : int
        Lines considered but not accepted (n_candidate − n_accepted).
    poly_degree_requested : int
        Polynomial degree that was requested.
    poly_degree_used : int
        Polynomial degree actually used (may be reduced due to low line count).
    fit_rms_nm : float
        RMS of fit residuals in nm (NaN if order was skipped).
    skipped : bool
        True when the order had too few accepted lines to fit.
    non_monotonic : bool
        True when the rectification-indices stage detected a non-monotonic
        wavelength surface for this order.
    """

    order_number: int
    n_candidate: int
    n_accepted: int
    n_rejected: int
    poly_degree_requested: int
    poly_degree_used: int
    fit_rms_nm: float
    skipped: bool
    non_monotonic: bool


@dataclass
class OrderDiagnostics1DXD:
    """Per-order 1DXD diagnostic record.

    Fields specific to the K3 1DXD (Stage 3b) calibration path.  Exported
    alongside :class:`OrderCalibrationDiagnostics` when ``--export-diagnostics``
    is specified.

    Parameters
    ----------
    order_number : int
        Echelle order number.
    xcorr_shift_px : float
        Cross-correlation column shift (px) applied before peak finding.
    n_candidate : int
        Number of peaks found in the 1D arc spectrum.
    n_matched : int
        Number of peaks matched to a reference line (before sigma clipping).
    n_accepted : int
        Number of matches retained after global sigma clipping.
    n_rejected : int
        Number of matches rejected by sigma clipping.
    rms_resid_nm : float
        Per-order RMS of the global-fit residuals for accepted points (nm).
        NaN if this order contributed no accepted points.
    participated : bool
        True if this order contributed at least one accepted point.
    """

    order_number: int
    xcorr_shift_px: float
    n_candidate: int
    n_matched: int
    n_accepted: int
    n_rejected: int
    rms_resid_nm: float
    participated: bool


def _build_1dxd_diagnostics(
    k3_model: "IdlStyle1DXDModel",
) -> list["OrderDiagnostics1DXD"]:
    """Build per-order 1DXD diagnostics from the Stage 3b model.

    Parameters
    ----------
    k3_model : IdlStyle1DXDModel
        Fitted global 1DXD model from Stage 3b.

    Returns
    -------
    list of OrderDiagnostics1DXD
    """
    diags: list[OrderDiagnostics1DXD] = []
    for stat in k3_model.per_order_stats:
        rms_nm = (
            stat.rms_resid_um * 1000.0
            if not math.isnan(stat.rms_resid_um) else float("nan")
        )
        diags.append(OrderDiagnostics1DXD(
            order_number=stat.order_number,
            xcorr_shift_px=stat.xcorr_shift_px,
            n_candidate=stat.n_candidate,
            n_matched=stat.n_matched,
            n_accepted=stat.n_accepted,
            n_rejected=stat.n_rejected,
            rms_resid_nm=rms_nm,
            participated=stat.participated,
        ))
    return diags



def _build_order_diagnostics(
    prov_map: "ProvisionalWavelengthMap",
    non_monotonic_orders: "set[int]",
    dispersion_degree: int,
) -> list[OrderCalibrationDiagnostics]:
    """Build a list of :class:`OrderCalibrationDiagnostics` from Stage 3 + 6 results.

    Parameters
    ----------
    prov_map : ProvisionalWavelengthMap
        Provisional wavelength solution from Stage 3.
    non_monotonic_orders : set of int
        Order numbers flagged as non-monotonic during Stage 6 (rectification).
    dispersion_degree : int
        The polynomial degree that was requested for all orders.

    Returns
    -------
    list of OrderCalibrationDiagnostics
    """
    diags: list[OrderCalibrationDiagnostics] = []
    for sol in prov_map.order_solutions:
        fit_rms_nm = (
            sol.fit_rms_um * 1000.0
            if not math.isnan(sol.fit_rms_um) else float("nan")
        )
        diags.append(OrderCalibrationDiagnostics(
            order_number=sol.order_number,
            n_candidate=sol.n_candidate,
            n_accepted=sol.n_accepted,
            n_rejected=sol.n_candidate - sol.n_accepted,
            poly_degree_requested=dispersion_degree,
            poly_degree_used=sol.fit_degree,
            fit_rms_nm=fit_rms_nm,
            skipped=sol.wave_coeffs is None,
            non_monotonic=sol.order_number in non_monotonic_orders,
        ))
    return diags


def _print_diagnostics_table(diags: list[OrderCalibrationDiagnostics]) -> None:
    """Print per-order calibration diagnostics as a formatted console table."""
    # Header
    hdr = (
        f"  {'Order':>5}  {'Cand':>5}  {'Acc':>4}  {'Rej':>4}  "
        f"{'DegReq':>6}  {'DegUsd':>6}  {'RMS(nm)':>8}  {'Skip':>4}  {'NMono':>5}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)
    for d in diags:
        rms_str = f"{d.fit_rms_nm:8.4f}" if not math.isnan(d.fit_rms_nm) else "     NaN"
        skip_str = "YES" if d.skipped else "no"
        nmono_str = "YES" if d.non_monotonic else "no"
        flag = ""
        if d.skipped or d.non_monotonic or d.n_accepted < 3:
            flag = " ◀"
        print(
            f"  {d.order_number:>5}  {d.n_candidate:>5}  {d.n_accepted:>4}  "
            f"{d.n_rejected:>4}  {d.poly_degree_requested:>6}  "
            f"{d.poly_degree_used:>6}  {rms_str}  {skip_str:>4}  "
            f"{nmono_str:>5}{flag}"
        )
    print(sep)


def _print_weak_order_warnings(diags: list[OrderCalibrationDiagnostics]) -> None:
    """Print a grouped summary of problematic orders."""
    skipped = [d for d in diags if d.skipped]
    low_lines = [d for d in diags if not d.skipped and d.n_accepted < 3]
    reduced_deg = [
        d for d in diags
        if not d.skipped and d.poly_degree_used < d.poly_degree_requested
    ]
    non_mono = [d for d in diags if d.non_monotonic]

    any_issues = skipped or low_lines or reduced_deg or non_mono
    if not any_issues:
        print("  [OK]  No weak-order issues detected.")
        return

    if skipped:
        nums = ", ".join(str(d.order_number) for d in skipped)
        print(f"  [!!]  SKIPPED orders ({len(skipped)}): {nums}")
    if low_lines:
        nums = ", ".join(str(d.order_number) for d in low_lines)
        print(f"  [!!]  Low-line orders (<3 accepted, {len(low_lines)}): {nums}")
    if reduced_deg:
        nums = ", ".join(str(d.order_number) for d in reduced_deg)
        print(f"  [!!]  Reduced-degree orders ({len(reduced_deg)}): {nums}")
    if non_mono:
        nums = ", ".join(str(d.order_number) for d in non_mono)
        print(f"  [!!]  Non-monotonic wavelength surface ({len(non_mono)}): {nums}")


def _export_diagnostics(
    diags: list[OrderCalibrationDiagnostics],
    out_dir: str,
    fmt: str,
    prefix: str,
    diags_1dxd: "Optional[list[OrderDiagnostics1DXD]]" = None,
) -> str:
    """Write diagnostics to *out_dir* as CSV or JSON.

    When *diags_1dxd* is provided the 1DXD-specific fields are merged into
    each row (keyed by ``order_number``).

    Returns the path of the written file.
    """
    # Build a lookup for 1DXD per-order fields
    dxd_lookup: dict[int, "OrderDiagnostics1DXD"] = {}
    if diags_1dxd is not None:
        dxd_lookup = {d.order_number: d for d in diags_1dxd}

    rows = []
    for d in diags:
        row: dict = {
            "order_number": d.order_number,
            "n_candidate": d.n_candidate,
            "n_accepted": d.n_accepted,
            "n_rejected": d.n_rejected,
            "poly_degree_requested": d.poly_degree_requested,
            "poly_degree_used": d.poly_degree_used,
            "fit_rms_nm": (
                None if math.isnan(d.fit_rms_nm) else round(d.fit_rms_nm, 6)
            ),
            "skipped": d.skipped,
            "non_monotonic": d.non_monotonic,
        }
        # Merge 1DXD fields when available
        dxd = dxd_lookup.get(d.order_number)
        if dxd is not None:
            row["1dxd_xcorr_shift_px"] = round(dxd.xcorr_shift_px, 3)
            row["1dxd_n_candidate"] = dxd.n_candidate
            row["1dxd_n_matched"] = dxd.n_matched
            row["1dxd_n_accepted"] = dxd.n_accepted
            row["1dxd_n_rejected"] = dxd.n_rejected
            row["1dxd_rms_nm"] = (
                None if math.isnan(dxd.rms_resid_nm)
                else round(dxd.rms_resid_nm, 6)
            )
            row["1dxd_participated"] = dxd.participated
        rows.append(row)

    if fmt == "json":
        out_path = os.path.join(out_dir, f"{prefix}_order_diagnostics.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
    else:
        out_path = os.path.join(out_dir, f"{prefix}_order_diagnostics.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    return out_path




def _filter_edge_orders(
    trace: "FlatOrderTrace",
    min_half_width_fraction: float = 0.30,
) -> "FlatOrderTrace":
    """Return a copy of *trace* with partial edge orders removed.

    **Benchmark-only rule (K3 calibration benchmark)**:

    The iSHELL flat-field tracing algorithm occasionally detects partially
    visible orders at the top and bottom edges of the detector.  These
    manifest as orders whose spatial half-width (``half_width_rows``) is
    substantially smaller than the typical half-width of the well-exposed
    science orders — because only part of the order profile is on the
    detector.

    For the K3 benchmark, orders whose ``half_width_rows`` falls below
    ``min_half_width_fraction`` × (median half-width across all orders)
    are excluded.  This criterion isolates orders that are genuinely
    clipped at the detector boundary, without requiring hard-coded index
    offsets.

    Applied to K3 data this reduces 29 detected orders to the 27 science
    orders (203–229) described in the IDL Spextool manual's QA figures.

    Parameters
    ----------
    trace : FlatOrderTrace
        Full tracing result, as returned by :func:`trace_orders_from_flat`.
    min_half_width_fraction : float
        Orders with ``half_width_rows < min_half_width_fraction * median``
        are excluded.  Default is 0.30 (30 % of the median half-width).

    Returns
    -------
    FlatOrderTrace
        A new :class:`FlatOrderTrace` containing only the retained orders,
        in their original detector order.  The ``n_orders`` field is
        updated accordingly.
    """
    import numpy as np

    hw = trace.half_width_rows
    median_hw = float(np.median(hw))
    threshold = min_half_width_fraction * median_hw

    keep = [i for i in range(trace.n_orders) if hw[i] >= threshold]

    if len(keep) == trace.n_orders:
        return trace  # nothing to filter

    keep_arr = keep
    return FlatOrderTrace(
        n_orders=len(keep_arr),
        sample_cols=trace.sample_cols,
        center_rows=trace.center_rows[keep_arr],
        center_poly_coeffs=trace.center_poly_coeffs[keep_arr],
        fit_rms=trace.fit_rms[keep_arr],
        half_width_rows=trace.half_width_rows[keep_arr],
        poly_degree=trace.poly_degree,
        seed_col=trace.seed_col,
        bot_poly_coeffs=(
            trace.bot_poly_coeffs[keep_arr]
            if trace.bot_poly_coeffs is not None else None
        ),
        top_poly_coeffs=(
            trace.top_poly_coeffs[keep_arr]
            if trace.top_poly_coeffs is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# QA plotting helpers
# ---------------------------------------------------------------------------


def _plot_flat_orders(
    flat_img,
    trace,
    out_dir: str,
    save: bool,
    prefix: str,
) -> None:
    """Python scaffold QA: traced order centres overlaid on the combined flat.

    The overlay shows **smooth polynomial curves** evaluated continuously
    across the full column range of each order, so the fitted trace is
    clearly visible.  Sparse sampled points are overlaid in a lighter
    style for reference.
    """
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 8))
        vmin, vmax = np.nanpercentile(flat_img, [1, 99])
        ax.imshow(
            flat_img,
            origin="lower",
            aspect="auto",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        # Evaluate each order's centre polynomial continuously across columns
        col_range = np.arange(
            int(trace.sample_cols[0]), int(trace.sample_cols[-1]) + 1
        )
        for i in range(trace.n_orders):
            coeffs = trace.center_poly_coeffs[i]
            centers_smooth = np.polynomial.polynomial.polyval(col_range, coeffs)
            ax.plot(col_range, centers_smooth, lw=0.9, alpha=0.85)
            # Overlay sampled points in a light grey for reference
            ax.plot(
                trace.sample_cols,
                trace.center_rows[i],
                ".",
                ms=2,
                alpha=0.25,
                color="white",
            )

        ax.set_title(
            "Python scaffold QA — K3 flat: traced order centres\n"
            f"({trace.n_orders} science orders, smooth polynomial curves)"
        )
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")

        if save:
            out_path = os.path.join(out_dir, f"{prefix}_flat_orders.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] flat-orders plot skipped: {exc}")


def _plot_arc_lines(
    arc_img,
    arc_result,
    out_dir: str,
    save: bool,
    prefix: str,
) -> None:
    """Python scaffold QA: traced arc-line seed columns on the combined arc."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 8))
        vmin, vmax = np.nanpercentile(arc_img, [5, 99.5])
        ax.imshow(
            arc_img,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )

        for line in arc_result.traced_lines:
            seed_col = line.seed_col
            # Use the median of the actually traced row indices as the
            # representative row for this line.  The poly_coeffs encode
            # col(row), not row(col), so evaluating them at seed_col would
            # be geometrically incorrect.
            if len(line.trace_rows) > 0:
                row = float(np.median(line.trace_rows))
            else:
                # Fallback when no trace points survived quality checks —
                # skip plotting this line rather than guessing.
                continue
            ax.plot(seed_col, row, "cx", markersize=4, alpha=0.7)

        ax.set_title(
            "Python scaffold QA — K3 arc: traced arc-line seed positions\n"
            f"({arc_result.n_lines} lines traced)"
        )
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")

        if save:
            out_path = os.path.join(out_dir, f"{prefix}_arc_lines.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] arc-lines plot skipped: {exc}")


def _plot_wavecal_residuals(
    prov_map,
    arc_result,
    out_dir: str,
    save: bool,
    prefix: str,
) -> None:
    """Python scaffold QA: wavelength-solution match residuals.

    Produces a 4-panel figure analogous in spirit to the IDL manual's
    Figure 3 (1DXD residual QA plot):

    - Top-left  : signed residuals vs echelle order number
                  (accepted lines in blue; rejected/unmatched in red)
    - Top-right : signed residuals vs detector column
                  (accepted lines in blue; rejected/unmatched in red)
    - Bottom-left : residual histogram with RMS annotation
    - Bottom-right: per-order accepted-line count bar chart with RMS
                  annotations

    Labelled *"Python scaffold QA"* — do not assume IDL parity.
    """
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # ---- collect accepted-match data ----
        residuals_nm: list[float] = []
        order_nums: list[int] = []
        cols: list[float] = []
        for sol in prov_map.order_solutions:
            for m in sol.accepted_matches:
                residuals_nm.append(m.match_residual_um * 1000.0)
                order_nums.append(m.order_number)
                cols.append(float(m.centerline_col))

        # ---- collect rejected-line data ----
        # "Rejected" = traced lines that were not among the accepted matches.
        # We infer them by comparing all traced lines for each order against the
        # accepted set (matched by seed_col equality).
        rej_cols: list[float] = []
        rej_order_nums: list[int] = []
        for sol in prov_map.order_solutions:
            accepted_seed_cols: set[float] = {
                float(m.centerline_col) for m in sol.accepted_matches
            }
            order_lines = arc_result.lines_for_order(sol.order_index)
            for line in order_lines:
                sc = float(line.seed_col)
                if sc not in accepted_seed_cols:
                    rej_cols.append(sc)
                    rej_order_nums.append(sol.order_number)

        if not residuals_nm:
            print("  [QA] No wavecal matches to plot.")
            return

        r = np.asarray(residuals_nm)
        o = np.asarray(order_nums)
        c = np.asarray(cols)
        rms = float(np.sqrt(np.mean(r ** 2)))
        med = float(np.median(r))

        rej_c = np.asarray(rej_cols)
        rej_o = np.asarray(rej_order_nums)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        # ---- per-order RMS ----
        per_order_rms: dict[int, float] = {}
        per_order_med: dict[int, float] = {}
        for sol in prov_map.order_solutions:
            if sol.n_accepted > 0:
                res_arr = np.array(
                    [m.match_residual_um * 1000.0 for m in sol.accepted_matches]
                )
                per_order_rms[sol.order_number] = float(
                    np.sqrt(np.mean(res_arr ** 2))
                )
                per_order_med[sol.order_number] = float(np.median(res_arr))

        # --- Top-left: residuals vs order number (accepted + rejected) ---
        axes[0, 0].scatter(o, r, s=10, alpha=0.7, color="steelblue",
                           label=f"Accepted ({len(r)})", zorder=3)
        if len(rej_o) > 0:
            axes[0, 0].scatter(
                rej_o, np.zeros(len(rej_o)), s=10, alpha=0.4,
                color="tomato", marker="x",
                label=f"Rejected/unmatched ({len(rej_o)})", zorder=2,
            )
        axes[0, 0].axhline(0.0, color="k", lw=0.8, ls="--")
        axes[0, 0].axhline(med, color="r", lw=1.2,
                           label=f"median = {med:.3f} nm")
        axes[0, 0].set_xlabel("Echelle order number (provisional)")
        axes[0, 0].set_ylabel("Residual (nm)")
        axes[0, 0].set_title("Residuals vs order number\n(accepted=blue, rejected=red×)")
        axes[0, 0].legend(fontsize=7)

        # --- Top-right: residuals vs detector column (accepted + rejected) ---
        axes[0, 1].scatter(c, r, s=10, alpha=0.7, color="steelblue",
                           label=f"Accepted ({len(r)})", zorder=3)
        if len(rej_c) > 0:
            axes[0, 1].scatter(
                rej_c, np.zeros(len(rej_c)), s=10, alpha=0.4,
                color="tomato", marker="x",
                label=f"Rejected/unmatched ({len(rej_c)})", zorder=2,
            )
        axes[0, 1].axhline(0.0, color="k", lw=0.8, ls="--")
        axes[0, 1].axhline(med, color="r", lw=1.2,
                           label=f"median = {med:.3f} nm")
        axes[0, 1].set_xlabel("Detector column (pixels)")
        axes[0, 1].set_ylabel("Residual (nm)")
        axes[0, 1].set_title("Residuals vs detector column\n(accepted=blue, rejected=red×)")
        axes[0, 1].legend(fontsize=7)

        # --- Bottom-left: histogram ---
        axes[1, 0].hist(r, bins=30, edgecolor="k", linewidth=0.5,
                        color="steelblue", alpha=0.7)
        axes[1, 0].axvline(0.0, color="k", lw=0.8, ls="--")
        axes[1, 0].set_xlabel("Residual (nm)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title(
            f"Accepted residual histogram  n={len(r)}\n"
            f"median={med:.3f} nm   RMS={rms:.3f} nm"
        )

        # --- Bottom-right: per-order accepted-line count + RMS annotation ---
        order_counts: dict = {}
        for oi in o:
            order_counts[oi] = order_counts.get(oi, 0) + 1
        order_sorted = sorted(order_counts)
        bar_vals = [order_counts[oi] for oi in order_sorted]
        bars = axes[1, 1].bar(
            order_sorted,
            bar_vals,
            width=0.7,
            color="teal",
            alpha=0.8,
        )
        # Annotate bars with per-order RMS (nm)
        for bar, oi in zip(bars, order_sorted):
            if oi in per_order_rms:
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.05,
                    f"{per_order_rms[oi]:.2f}",
                    ha="center", va="bottom", fontsize=5, rotation=90,
                    color="darkslategray",
                )
        axes[1, 1].set_xlabel("Echelle order number (provisional)")
        axes[1, 1].set_ylabel("Matched lines")
        axes[1, 1].set_title(
            "Accepted lines per order\n(bar labels = RMS nm)"
        )

        fig.suptitle(
            "Python scaffold QA — K3 1DXD residuals (analogue of manual Fig. 3)\n"
            f"Accepted: {len(r)}  Rejected/unmatched: {len(rej_cols)}  "
            f"RMS: {rms:.3f} nm   NOTE: not IDL-equivalent",
            fontsize=10,
        )
        fig.tight_layout()

        if save:
            out_path = os.path.join(
                out_dir, f"{prefix}_wavecal_residuals.png"
            )
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] wavecal-residuals plot skipped: {exc}")


def _plot_rectified_order(
    rect_set,
    out_dir: str,
    save: bool,
    prefix: str,
) -> None:
    """Python scaffold QA: first available rectified order image."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Pick the first order that has data
        img = None
        order_num = None
        for ro in rect_set.rectified_orders:
            if ro.flux is not None and ro.flux.size > 0:
                img = ro.flux
                order_num = ro.order
                break

        if img is None:
            print("  [QA] No rectified order images available.")
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        vmin, vmax = np.nanpercentile(img, [1, 99])
        ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(
            f"Python scaffold QA — K3 rectified order {order_num}\n"
            "NOTE: provisional scaffold; not science-quality"
        )
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Spatial row (pixels)")

        if save:
            out_path = os.path.join(
                out_dir, f"{prefix}_rectified_order.png"
            )
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] rectified-order plot skipped: {exc}")

# ---------------------------------------------------------------------------
# 2DCoeffFit-style QA helper (scaffold analogue of IDL Figures 4–7)
# ---------------------------------------------------------------------------


def _plot_2d_coeff_fit(
    arc_result,
    prov_map,
    out_dir: str,
    save: bool,
    prefix: str,
) -> None:
    """Python scaffold QA: first analogue of the IDL 2DCoeffFit.pdf.

    Produces a multi-panel figure covering the line-tilt / wavelength
    calibration coefficient view.  This is an **approximate analogue** of
    the IDL Spextool manual's Figures 4–7 (``2DCoeffFit.pdf``), limited to
    what is available from the current Python scaffold.

    Panels
    ------
    1. Arc-line position cloud (col vs row) — corresponds to IDL's
       "zeroed-line" view.
    2. Line tilt slope (``poly_coeffs[1]``) vs seed column — tilt mapping
       across the detector.
    3. Per-order wavelength polynomial evaluation (wavelength vs column) —
       the scaffold's 1DXD wavelength-solution curves.
    4. Accepted-match residuals (nm) vs (order, column) scatter.

    What is missing relative to IDL Figures 4–7
    --------------------------------------------
    * No model surface overlay (requires full 2DCoeffFit surface).
    * No curvature coefficient map (not yet extracted from scaffold).
    * No per-line sigma-clip rejection indicator.
    * No spatial calibration panels.

    Labelled *"Python scaffold QA"* — not equivalent to IDL figures.
    """
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ----------------------------------------------------------------
        # Panel 1: Arc-line position cloud (col vs seed_row at seed_col)
        # ----------------------------------------------------------------
        seed_cols: list[float] = []
        seed_rows: list[float] = []
        tilt_slopes: list[float] = []
        order_indices: list[int] = []

        for line in arc_result.traced_lines:
            sc = float(line.seed_col)
            # Use the median row of traced points as the line's representative row
            if len(line.trace_rows) > 0:
                sr = float(np.median(line.trace_rows))
            else:
                sr = float(np.polynomial.polynomial.polyval(
                    sc, line.poly_coeffs
                ))
            slope = line.tilt_slope()
            seed_cols.append(sc)
            seed_rows.append(sr)
            tilt_slopes.append(slope)
            order_indices.append(line.order_index)

        sc_arr = np.asarray(seed_cols)
        sr_arr = np.asarray(seed_rows)
        ts_arr = np.asarray(tilt_slopes)
        oi_arr = np.asarray(order_indices)

        axes[0, 0].scatter(sc_arr, sr_arr, c=oi_arr, cmap="tab20",
                           s=8, alpha=0.7)
        axes[0, 0].set_xlabel("Seed column (pixels)")
        axes[0, 0].set_ylabel("Traced row at seed column (pixels)")
        axes[0, 0].set_title(
            "Arc-line position cloud\n(col vs traced row, coloured by order index)"
        )

        # ----------------------------------------------------------------
        # Panel 2: Tilt slope vs seed column
        # ----------------------------------------------------------------
        axes[0, 1].scatter(sc_arr, ts_arr, c=oi_arr, cmap="tab20",
                           s=8, alpha=0.7)
        axes[0, 1].axhline(0.0, color="k", lw=0.6, ls="--")
        axes[0, 1].set_xlabel("Seed column (pixels)")
        axes[0, 1].set_ylabel("Tilt slope (pixels / row)")
        axes[0, 1].set_title(
            "Arc-line tilt slope vs detector column\n"
            "(analogue of IDL 2DCoeffFit slope panel)"
        )

        # ----------------------------------------------------------------
        # Panel 3: Wavelength solution curves per order
        # ----------------------------------------------------------------
        col_eval = np.linspace(0, 2047, 200)
        for sol in prov_map.order_solutions:
            if sol.wave_coeffs is None or len(sol.accepted_matches) == 0:
                continue
            wave_nm = np.polynomial.polynomial.polyval(
                col_eval, sol.wave_coeffs
            ) * 1000.0  # µm → nm
            axes[1, 0].plot(col_eval, wave_nm, lw=0.8, alpha=0.8)
        axes[1, 0].set_xlabel("Detector column (pixels)")
        axes[1, 0].set_ylabel("Wavelength (nm)")
        axes[1, 0].set_title(
            "Per-order 1DXD wavelength solutions\n"
            "(analogue of IDL wavelength surface view)"
        )

        # ----------------------------------------------------------------
        # Panel 4: Residuals scatter (order × column)
        # ----------------------------------------------------------------
        res_nm: list[float] = []
        res_order: list[int] = []
        res_col: list[float] = []
        for sol in prov_map.order_solutions:
            for m in sol.accepted_matches:
                res_nm.append(m.match_residual_um * 1000.0)
                res_order.append(m.order_number)
                res_col.append(float(m.centerline_col))

        if res_nm:
            r_arr = np.asarray(res_nm)
            c_arr = np.asarray(res_col)
            o_arr = np.asarray(res_order)
            sc_plot = axes[1, 1].scatter(
                c_arr, o_arr, c=r_arr, cmap="RdYlGn_r",
                s=18, alpha=0.8,
                vmin=np.nanpercentile(r_arr, 5),
                vmax=np.nanpercentile(r_arr, 95),
            )
            fig.colorbar(sc_plot, ax=axes[1, 1], label="Residual (nm)")
        axes[1, 1].set_xlabel("Detector column (pixels)")
        axes[1, 1].set_ylabel("Echelle order number")
        axes[1, 1].set_title(
            "Residual map (order × column)\n"
            "(analogue of IDL 2DCoeffFit residual panel)"
        )

        fig.suptitle(
            "Python scaffold QA — K3 2DCoeffFit analogue (Figs 4–7 spirit)\n"
            "NOTE: partial — not equivalent to IDL 2DCoeffFit.pdf",
            fontsize=10,
        )
        fig.tight_layout()

        if save:
            out_path = os.path.join(
                out_dir, f"{prefix}_2d_coeff_fit.png"
            )
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] 2DCoeffFit plot skipped: {exc}")


# ---------------------------------------------------------------------------
# K3 1DXD QA plot (accepted / rejected, residuals vs order / column)
# ---------------------------------------------------------------------------


def _plot_k3_1dxd_qa(
    k3_model: "IdlStyle1DXDModel",
    out_dir: str,
    save: bool,
    prefix: str,
) -> None:
    """Python K3 1DXD QA — accepted vs rejected points, residual plots.

    Produces a 2×2 figure showing:

    - Panel 1: residuals vs detector column (accepted=blue, rejected=red ×)
    - Panel 2: residuals vs echelle order  (accepted=blue, rejected=red ×)
    - Panel 3: residual histogram after sigma clipping
    - Panel 4: per-order RMS bar chart

    Clearly labelled as *Python K3 1DXD QA* to distinguish from IDL.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # -- Reconstruct all point coordinates from the accepted_mask ----
        # We need to rebuild them from the model's stats; the raw arrays
        # are not stored in the model, so we recover what we can from
        # per_order_stats.
        # For the residual panels we show per-order stats as a proxy.

        stats = k3_model.per_order_stats
        order_nums = [s.order_number for s in stats]
        xcorr_shifts = [s.xcorr_shift_px for s in stats]
        n_acc = [s.n_accepted for s in stats]
        n_rej = [s.n_rejected for s in stats]
        rms_vals = [
            s.rms_resid_um * 1e3  # → nm
            if not math.isnan(s.rms_resid_um) else float("nan")
            for s in stats
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # ----------------------------------------------------------------
        # Panel 1: cross-correlation shift vs order
        # ----------------------------------------------------------------
        ax = axes[0, 0]
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.scatter(order_nums, xcorr_shifts, color="steelblue", s=20)
        ax.set_xlabel("Echelle order number")
        ax.set_ylabel("XCorr shift (pixels)")
        ax.set_title("Per-order cross-correlation shift")

        # ----------------------------------------------------------------
        # Panel 2: per-order accepted vs rejected line counts
        # ----------------------------------------------------------------
        ax = axes[0, 1]
        x = np.arange(len(order_nums))
        width = 0.4
        ax.bar(x - width / 2, n_acc, width, label="Accepted", color="steelblue")
        ax.bar(x + width / 2, n_rej, width, label="Rejected (σ-clip)", color="tomato")
        ax.set_xticks(x)
        ax.set_xticklabels(order_nums, rotation=90, fontsize=6)
        ax.set_xlabel("Echelle order number")
        ax.set_ylabel("Line count")
        ax.set_title("Accepted vs rejected lines per order\n(after sigma clipping)")
        ax.legend(fontsize=8)

        # ----------------------------------------------------------------
        # Panel 3: per-order RMS after sigma clipping
        # ----------------------------------------------------------------
        ax = axes[1, 0]
        rms_valid = [r for r in rms_vals if not math.isnan(r)]
        rms_orders = [
            s.order_number for s in stats if not math.isnan(s.rms_resid_um)
        ]
        if rms_valid:
            ax.bar(rms_orders, rms_valid, color="steelblue", width=0.6)
        ax.axhline(k3_model.fit_rms_um * 1e3, color="tomato",
                   lw=1.2, ls="--", label=f"Global RMS = {k3_model.fit_rms_um*1e3:.3f} nm")
        ax.set_xlabel("Echelle order number")
        ax.set_ylabel("RMS residual (nm)")
        ax.set_title("Per-order RMS after sigma clipping")
        ax.legend(fontsize=8)

        # ----------------------------------------------------------------
        # Panel 4: global summary text
        # ----------------------------------------------------------------
        ax = axes[1, 1]
        ax.axis("off")
        summary = (
            "Global 1DXD fit summary\n"
            "──────────────────────\n"
            f"wdeg = {k3_model.wdeg},  odeg = {k3_model.odeg}\n"
            f"Total lines matched : {k3_model.n_lines_total}\n"
            f"Accepted (σ-clip)   : {k3_model.n_lines}\n"
            f"Rejected (σ-clip)   : {k3_model.n_lines_rejected}\n"
            f"Orders in fit       : {k3_model.n_orders_fit}\n"
            f"Global RMS          : {k3_model.fit_rms_um * 1e3:.4f} nm\n"
            f"Median residual     : {k3_model.median_residual_um * 1e3:.4f} nm\n"
        )
        ax.text(
            0.05, 0.95, summary,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        fig.suptitle(
            "Python K3 1DXD QA — global polynomial fit with sigma clipping\n"
            "NOTE: preliminary — not IDL-equivalent",
            fontsize=10,
        )
        fig.tight_layout()

        if save:
            out_path = os.path.join(out_dir, f"{prefix}_k3_1dxd_qa.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] K3 1DXD QA plot skipped: {exc}")


# ---------------------------------------------------------------------------
# K3 1DXD residual QA plot — point-based (primary 1DXD QA artifact)
# ---------------------------------------------------------------------------


def _plot_k3_1dxd_residuals(
    k3_model: "IdlStyle1DXDModel",
    out_dir: str,
    save: bool,
    prefix: str,
) -> Optional[str]:
    """Point-based K3 1DXD residual QA plot.

    Uses the explicit matched-point arrays stored on the model to produce the
    primary K3 1DXD QA figure.  The output file is named::

        {prefix}_k3_1dxd_residuals.png

    Panels
    ------
    1. Residuals vs detector column  — accepted (blue ●) and rejected (red ×)
    2. Residuals vs echelle order    — accepted (blue ●) and rejected (red ×)
    3. Residual histogram (accepted) with optional rejected overlay
    4. Summary annotation box with ntot / nacc / nrej / RMS / median

    All residual axes are in **nm** (nanometres).
    """
    try:
        import matplotlib.pyplot as plt

        cols_all   = k3_model.matched_cols_px
        orders_all = k3_model.matched_order_numbers
        resid_all  = k3_model.matched_residual_um * 1e3  # µm → nm
        mask       = k3_model.accepted_mask

        cols_acc  = cols_all[mask]
        cols_rej  = cols_all[~mask]
        orders_acc = orders_all[mask]
        orders_rej = orders_all[~mask]
        resid_acc  = resid_all[mask]
        resid_rej  = resid_all[~mask]

        ntot = k3_model.n_lines_total
        nacc = k3_model.n_lines
        nrej = k3_model.n_lines_rejected
        rms_nm   = k3_model.fit_rms_um * 1e3
        med_nm   = k3_model.median_residual_um * 1e3

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # ----------------------------------------------------------------
        # Panel 1: residuals vs detector column
        # ----------------------------------------------------------------
        ax = axes[0, 0]
        if len(cols_acc) > 0:
            ax.scatter(cols_acc, resid_acc, s=8, color="steelblue",
                       alpha=0.7, label="Accepted")
        if len(cols_rej) > 0:
            ax.scatter(cols_rej, resid_rej, s=14, color="tomato",
                       marker="x", linewidths=1.0, alpha=0.9, label="Rejected (σ-clip)")
        ax.axhline(0.0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Detector column (pixels)")
        ax.set_ylabel("Residual (nm)")
        ax.set_title("Residuals vs detector column")
        ax.legend(fontsize=8)

        # ----------------------------------------------------------------
        # Panel 2: residuals vs echelle order
        # ----------------------------------------------------------------
        ax = axes[0, 1]
        if len(orders_acc) > 0:
            ax.scatter(orders_acc, resid_acc, s=8, color="steelblue",
                       alpha=0.7, label="Accepted")
        if len(orders_rej) > 0:
            ax.scatter(orders_rej, resid_rej, s=14, color="tomato",
                       marker="x", linewidths=1.0, alpha=0.9, label="Rejected (σ-clip)")
        ax.axhline(0.0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Echelle order number")
        ax.set_ylabel("Residual (nm)")
        ax.set_title("Residuals vs echelle order")
        ax.legend(fontsize=8)

        # ----------------------------------------------------------------
        # Panel 3: residual histogram (accepted), rejected overlay
        # ----------------------------------------------------------------
        ax = axes[1, 0]
        if len(resid_acc) > 0:
            ax.hist(resid_acc, bins=30, color="steelblue", alpha=0.7,
                    label="Accepted")
        if len(resid_rej) > 0:
            ax.hist(resid_rej, bins=20, color="tomato", alpha=0.5,
                    label="Rejected (σ-clip)")
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        ax.axvline(rms_nm, color="steelblue", lw=1.0, ls=":",
                   label=f"+RMS = {rms_nm:.3f} nm")
        ax.axvline(-rms_nm, color="steelblue", lw=1.0, ls=":")
        ax.set_xlabel("Residual (nm)")
        ax.set_ylabel("Count")
        ax.set_title("Residual histogram (accepted + rejected)")
        ax.legend(fontsize=7)

        # ----------------------------------------------------------------
        # Panel 4: summary annotation
        # ----------------------------------------------------------------
        ax = axes[1, 1]
        ax.axis("off")
        summary = (
            "Python K3 1DXD QA\n"
            "────────────────────────\n"
            f"wdeg = {k3_model.wdeg},  odeg = {k3_model.odeg}\n"
            f"ntot  = {ntot}\n"
            f"nacc  = {nacc}\n"
            f"nrej  = {nrej}\n"
            f"RMS   = {rms_nm:.4f} nm\n"
            f"median= {med_nm:.4f} nm\n"
            f"Orders: {k3_model.n_orders_fit}\n"
        )
        ax.text(
            0.05, 0.95, summary,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        fig.suptitle(
            "Python K3 1DXD QA — residuals from sigma-clipped global fit\n"
            "NOTE: preliminary — not IDL-equivalent",
            fontsize=10,
        )
        fig.tight_layout()

        out_path: Optional[str] = None
        if save:
            out_path = os.path.join(out_dir, f"{prefix}_k3_1dxd_residuals.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
        return out_path
    except Exception as exc:
        print(f"  [QA] K3 1DXD residuals plot skipped: {exc}")
        return None


# ---------------------------------------------------------------------------
# K3 1DXD global summary metadata export
# ---------------------------------------------------------------------------


def _export_1dxd_global_summary(
    k3_model: "IdlStyle1DXDModel",
    out_dir: str,
    prefix: str,
) -> str:
    """Write K3 1DXD global summary metadata to a JSON sidecar file.

    The file is named::

        {prefix}_k3_1dxd_summary.json

    Schema
    ------
    ::

        {
          "n_lines_total":      int,
          "n_lines_accepted":   int,
          "n_lines_rejected":   int,
          "fit_rms_um":         float,
          "median_residual_um": float,
          "lambda_degree":      int,
          "order_degree":       int,
          "fitted_order_numbers": [int, ...],
          "mode":               str
        }

    Returns the path of the written file.
    """
    data = {
        "n_lines_total":       k3_model.n_lines_total,
        "n_lines_accepted":    k3_model.n_lines,
        "n_lines_rejected":    k3_model.n_lines_rejected,
        "fit_rms_um":          k3_model.fit_rms_um,
        "median_residual_um":  k3_model.median_residual_um,
        "lambda_degree":       k3_model.wdeg,
        "order_degree":        k3_model.odeg,
        "fitted_order_numbers": k3_model.fitted_order_numbers,
        "mode":                k3_model.mode,
    }
    out_path = os.path.join(out_dir, f"{prefix}_k3_1dxd_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Main reduction driver
# ---------------------------------------------------------------------------


def run_k3_example(
    config: K3BenchmarkConfig | None = None,
    **overrides,
) -> dict[str, bool]:
    """Run the K3 benchmark reduction chain.

    Parameters
    ----------
    config : K3BenchmarkConfig, optional
        Configuration object.  If ``None``, a default
        :class:`K3BenchmarkConfig` is constructed first.
    **overrides
        Any :class:`K3BenchmarkConfig` field name may be passed as a
        keyword argument to override the corresponding field in *config*
        (or the default config when *config* is ``None``).

        Supported override keys (all optional):

        - ``raw_dir``               – raw K3 data directory
        - ``out_dir``               – output directory (alias for ``output_dir``)
        - ``output_dir``            – output directory
        - ``flat_frames``           – list of flat frame numbers
        - ``arc_frames``            – list of arc frame numbers
        - ``dark_frames``           – list of dark frame numbers
        - ``object_frames``         – list of object frame numbers
        - ``standard_frames``       – list of standard frame numbers
        - ``flat_output_name``      – stem for flat FITS output
        - ``wavecal_output_name``   – stem for wavecal FITS output
        - ``dark_output_name``      – stem for dark FITS output
        - ``qa_plot_prefix``        – prefix for QA plot filenames
        - ``save_plots``            – save plots as PNG
        - ``no_plots``              – skip all plots
        - ``mode_name``             – iSHELL mode string

    Returns
    -------
    dict of str → bool
        Mapping from stage name to completion status.

    Raises
    ------
    FileNotFoundError
        If the raw directory does not exist or the required flat/arc files
        are absent.
    TypeError
        If an unrecognised keyword argument is passed.

    Examples
    --------
    Run with all defaults::

        from scripts.run_ishell_k3_example import run_k3_example
        completed = run_k3_example()

    Override individual fields::

        completed = run_k3_example(
            raw_dir="/data/K3",
            wavecal_output_name="wavecal11-12",
            no_plots=True,
        )

    Pass a fully custom config::

        from scripts.run_ishell_k3_example import K3BenchmarkConfig, run_k3_example
        cfg = K3BenchmarkConfig(
            raw_dir="/data/K3",
            wavecal_output_name="my_wavecal",
            save_plots=True,
        )
        completed = run_k3_example(cfg)
    """
    # ------------------------------------------------------------------
    # Build effective config from defaults + any overrides
    # ------------------------------------------------------------------

    cfg = K3BenchmarkConfig() if config is None else config

    # Support ``out_dir`` as a convenience alias for ``output_dir``
    if "out_dir" in overrides and "output_dir" not in overrides:
        overrides["output_dir"] = overrides.pop("out_dir")

    # Validate override keys before applying them
    valid_fields = {f for f in cfg.__dataclass_fields__}
    bad_keys = set(overrides) - valid_fields
    if bad_keys:
        raise TypeError(
            f"run_k3_example() got unexpected keyword argument(s): "
            f"{sorted(bad_keys)}"
        )

    # Apply overrides (shallow copy to avoid mutating the caller's config)
    if overrides:
        import dataclasses
        cfg = dataclasses.replace(cfg, **overrides)

    raw_dir = cfg.raw_dir
    out_dir = cfg.output_dir
    mode = cfg.mode_name
    save_plots = cfg.save_plots
    no_plots = cfg.no_plots
    prefix = cfg.qa_plot_prefix
    wavecal_name = cfg.wavecal_output_name
    flat_name = cfg.flat_output_name

    completed: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"K3 raw directory not found: {raw_dir!r}\n"
            "Place the K3 FITS files there (see data/testdata/ishell_k3_example/raw/README.md)."
        )

    os.makedirs(out_dir, exist_ok=True)

    all_files = find_fits_files(raw_dir)

    flat_files = _select_files(all_files, "flat", cfg.flat_frames)
    arc_files = _select_files(all_files, "arc", cfg.arc_frames)
    dark_files = _select_files(all_files, "dark", cfg.dark_frames)
    object_files = _select_files(all_files, "spc", cfg.object_frames)
    standard_files = _select_files(all_files, "spc", cfg.standard_frames)

    flat_range = f"{min(cfg.flat_frames)}–{max(cfg.flat_frames)}"
    arc_range = f"{min(cfg.arc_frames)}–{max(cfg.arc_frames)}"
    dark_range = (
        f"{min(cfg.dark_frames)}–{max(cfg.dark_frames)}"
        if cfg.dark_frames else "none"
    )
    obj_range = (
        f"{min(cfg.object_frames)}–{max(cfg.object_frames)}"
        if cfg.object_frames else "none"
    )
    std_range = (
        f"{min(cfg.standard_frames)}–{max(cfg.standard_frames)}"
        if cfg.standard_frames else "none"
    )

    _banner("K3 Benchmark: file discovery")
    print(f"  Mode        : {mode}")
    print(f"  Raw dir     : {raw_dir}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Flat files  : {len(flat_files)}  (frames {flat_range})")
    print(f"  Arc files   : {len(arc_files)}   (frames {arc_range})")
    print(f"  Dark files  : {len(dark_files)}  (frames {dark_range})")
    print(f"  Object spc  : {len(object_files)}  (frames {obj_range})")
    print(f"  Std A0V spc : {len(standard_files)}  (frames {std_range})")

    if not flat_files:
        raise FileNotFoundError(
            f"No flat files (frames {flat_range}) found in {raw_dir!r}. "
            "Ensure the K3 FITS files are present."
        )
    if not arc_files:
        raise FileNotFoundError(
            f"No arc files (frames {arc_range}) found in {raw_dir!r}. "
            "Ensure the K3 FITS files are present."
        )

    # ------------------------------------------------------------------
    # Load packaged calibration resources for the configured mode
    # ------------------------------------------------------------------

    wavecalinfo = read_wavecalinfo(mode)
    line_list = read_line_list(mode)

    # ------------------------------------------------------------------
    # Stage 1: Flat/order tracing
    # ------------------------------------------------------------------

    _banner("Stage 1: Flat / order-centre tracing")
    flat_img = load_and_combine_flats(flat_files)
    trace_raw = trace_orders_from_flat(flat_files)
    print(f"  Orders detected (raw)  : {trace_raw.n_orders}")

    # Apply benchmark-only edge-order filter:
    # Exclude orders whose half-width is below 30 % of the median half-width.
    # These are partially visible detector-edge orders that are not science
    # orders.  For K3 this reduces 29 detected orders to the 27 science
    # orders (203–229) described in the IDL Spextool manual's QA figures.
    # See _filter_edge_orders() for the full criterion documentation.
    trace = _filter_edge_orders(trace_raw)
    n_dropped = trace_raw.n_orders - trace.n_orders
    if n_dropped > 0:
        print(
            f"  Orders retained        : {trace.n_orders} "
            f"(dropped {n_dropped} partial edge order(s), "
            f"half_width_rows < 30% of median)"
        )
    else:
        print(f"  Orders retained        : {trace.n_orders}")

    geom = trace.to_order_geometry_set(mode)
    print(f"  Median fit RMS  : {float(trace.fit_rms.mean()):.2f} px (mean across orders)")
    _ok("Stage 1 — flat/order tracing")
    completed["stage1_flat_tracing"] = True

    if not no_plots:
        _plot_flat_orders(flat_img, trace, out_dir, save=save_plots,
                          prefix=prefix)

    # ------------------------------------------------------------------
    # Stage 2: Arc-line tracing
    # ------------------------------------------------------------------

    _banner("Stage 2: Arc-line tracing")
    arc_img = load_and_combine_arcs(arc_files)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arc_result = trace_arc_lines(arc_files, geom)
    print(f"  Arc lines traced : {arc_result.n_lines}")
    _ok("Stage 2 — arc-line tracing")
    completed["stage2_arc_tracing"] = True

    if not no_plots:
        _plot_arc_lines(arc_img, arc_result, out_dir, save=save_plots,
                        prefix=prefix)

    # ------------------------------------------------------------------
    # Stage 3: Provisional wavelength map (scaffold — used for diagnostics)
    # ------------------------------------------------------------------

    _banner("Stage 3: Provisional per-order wavelength mapping (scaffold)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prov_map = fit_provisional_wavelength_map(
            arc_result, wavecalinfo, line_list
        )
    n_matched = sum(
        len(sol.accepted_matches) for sol in prov_map.order_solutions
    )
    print(f"  Orders with solutions : {prov_map.n_orders}")
    print(f"  Total matched lines   : {n_matched}")

    # Per-order structured diagnostics table (non-monotonic info added after Stage 6)
    _dispersion_degree = 3  # default; matches fit_provisional_wavelength_map
    print("  Per-order details:")
    for sol in prov_map.order_solutions:
        n_acc = len(sol.accepted_matches)
        deg = (
            len(sol.wave_coeffs) - 1
            if sol.wave_coeffs is not None else "N/A"
        )
        print(
            f"    order {sol.order_number:4d}: "
            f"{n_acc:3d} accepted lines, "
            f"poly degree={deg}"
        )

    _ok("Stage 3 — provisional wavelength mapping (scaffold)")
    completed["stage3_provisional_wavemap"] = True

    if not no_plots:
        _plot_wavecal_residuals(prov_map, arc_result, out_dir,
                                save=save_plots, prefix=prefix)
        _plot_2d_coeff_fit(arc_result, prov_map, out_dir, save=save_plots,
                           prefix=prefix)

    # ------------------------------------------------------------------
    # Build the geometry→echelle order mapping from wavecalinfo.
    # This allows Stage 6 to use the K3 1DXD path WITHOUT depending on
    # the scaffold prov_map as the primary order-number source.
    #
    # The flat-field tracing produces geometry with 0-based placeholder
    # order indices (0, 1, 2, …).  wavecalinfo.orders lists the real
    # echelle order numbers in the same position order.
    # ------------------------------------------------------------------
    k3_geom_order_map: dict[int, int] = {}
    try:
        orders_from_wci = list(wavecalinfo.orders)
        for geom_order_idx, real_order in enumerate(orders_from_wci):
            k3_geom_order_map[geom_order_idx] = int(real_order)
    except Exception as exc:  # noqa: BLE001
        # Fallback: use prov_map for the mapping (scaffold dependency)
        k3_geom_order_map = {}
        print(f"  [warn] Could not build geom_order_map from wavecalinfo: {exc}")

    # ------------------------------------------------------------------
    # Stage 3b: K3 1DXD wavelength calibration (IDL-style — primary path)
    # ------------------------------------------------------------------

    _banner("Stage 3b: K3 1DXD wavelength calibration (IDL-style — primary path)")
    k3_model = None
    try:
        arc_spectra = extract_order_arc_spectra(
            arc_img, trace, wavecalinfo, aperture_half_width=3
        )
        print(f"  1D arc spectra extracted : {arc_spectra.n_orders} orders")

        k3_model = fit_1dxd_wavelength_model(
            arc_spectra, wavecalinfo, line_list, wdeg=3, odeg=2
        )
        print(f"  1DXD fit: {k3_model.n_orders_fit} orders")
        print(f"    total={k3_model.n_lines_total}, "
              f"accepted={k3_model.n_lines}, "
              f"rejected={k3_model.n_lines_rejected}")
        print(f"    RMS = {k3_model.fit_rms_um * 1e3:.3f} nm  "
              f"median_res = {k3_model.median_residual_um * 1e3:.4f} nm")
        print(f"  Fitted orders: {k3_model.fitted_order_numbers}")
        _ok("Stage 3b — K3 1DXD wavelength calibration")
        completed["stage3b_k3_1dxd"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"  Stage 3b raised: {exc}")
        _skip("Stage 3b — K3 1DXD wavelength calibration", "exception (see above)")
        completed["stage3b_k3_1dxd"] = False

    if not no_plots and k3_model is not None:
        _plot_k3_1dxd_qa(k3_model, out_dir, save=save_plots, prefix=prefix)
        _plot_k3_1dxd_residuals(k3_model, out_dir, save=save_plots, prefix=prefix)

    # ------------------------------------------------------------------
    # Stage 4: Global wavelength surface (scaffold — kept for reference)
    # ------------------------------------------------------------------

    _banner("Stage 4: Global wavelength surface (scaffold)")
    try:
        surface_result = fit_global_wavelength_surface(prov_map)
        print(f"  Surface fit converged: {getattr(surface_result, 'converged', 'N/A')}")
        _ok("Stage 4 — global wavelength surface (scaffold)")
        completed["stage4_global_surface"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"  Stage 4 raised: {exc}")
        _skip("Stage 4 — global wavelength surface (scaffold)", "exception (see above)")
        completed["stage4_global_surface"] = False
        surface_result = None

    # ------------------------------------------------------------------
    # Stage 5: Coefficient refinement (scaffold — kept for reference)
    # ------------------------------------------------------------------

    _banner("Stage 5: Coefficient-surface refinement (scaffold)")
    try:
        refined = fit_refined_coefficient_surface(prov_map)
        print(f"  Refined surface n_orders: {getattr(refined, 'n_orders', 'N/A')}")
        _ok("Stage 5 — coefficient refinement (scaffold)")
        completed["stage5_refinement"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"  Stage 5 raised: {exc}")
        _skip("Stage 5 — coefficient refinement (scaffold)", "exception (see above)")
        completed["stage5_refinement"] = False
        refined = None

    # ------------------------------------------------------------------
    # Stage 6: Rectification indices
    #
    # K3 benchmark path: uses the 1DXD model (Stage 3b) as the source of
    # truth for wavelengths.  Falls back to the scaffold surface (Stage 5)
    # only if the 1DXD model is unavailable.
    # ------------------------------------------------------------------

    # Capture non-monotonic warnings so they can feed into the diagnostics.
    _non_monotonic_orders: set[int] = set()

    _banner("Stage 6: Rectification indices")
    # Decide which wavelength model drives rectification.
    _use_k3_path = k3_model is not None
    _rect_prereq_ok = _use_k3_path or (refined is not None)

    if _rect_prereq_ok:
        try:
            with warnings.catch_warnings(record=True) as _caught_warns:
                warnings.simplefilter("always")
                if _use_k3_path:
                    # K3 benchmark path: 1DXD model drives rectification.
                    # Use geom_order_map (derived directly from wavecalinfo)
                    # to avoid any scaffold prov_map dependency.
                    print("  Wavelength source: K3 1DXD model (Stage 3b)")
                    _k3_geom_map = k3_geom_order_map if k3_geom_order_map else None
                    _k3_wav_map = prov_map if not _k3_geom_map else None
                    if _k3_geom_map:
                        print("  Order mapping   : wavecalinfo (no scaffold dependency)")
                    else:
                        print("  Order mapping   : scaffold prov_map (fallback)")
                    rect_idx = build_rectification_indices(
                        geom,
                        wavelength_func=k3_model.as_wavelength_func(),
                        fitted_order_numbers=k3_model.fitted_order_numbers,
                        geom_order_map=_k3_geom_map,
                        wav_map=_k3_wav_map,
                    )
                else:
                    # Scaffold fallback (K3 1DXD unavailable).
                    print("  Wavelength source: scaffold surface (Stage 5 fallback)")
                    rect_idx = build_rectification_indices(
                        geom, refined, wav_map=prov_map
                    )
            # Extract order numbers from non-monotonic RuntimeWarnings.
            for w in _caught_warns:
                if issubclass(w.category, RuntimeWarning):
                    msg = str(w.message)
                    # Warning format: "Order N (order_number=M): the sampled …"
                    # Extract the *echelle* order number from order_number=M,
                    # NOT the index N that appears first in the message.
                    _echelle_order = _parse_non_monotonic_order_number(msg)
                    if _echelle_order is not None:
                        _non_monotonic_orders.add(_echelle_order)
                    # Re-emit so they remain visible to the caller.
                    warnings.warn_explicit(
                        w.message, w.category, w.filename, w.lineno
                    )
            print(f"  Rectification index orders: {rect_idx.n_orders}")
            if _non_monotonic_orders:
                print(
                    f"  Non-monotonic orders detected: "
                    f"{sorted(_non_monotonic_orders)}"
                )
            _ok("Stage 6 — rectification indices")
            completed["stage6_rect_indices"] = True
        except Exception as exc:  # noqa: BLE001
            print(f"  Stage 6 raised: {exc}")
            _skip("Stage 6 — rectification indices", "exception (see above)")
            completed["stage6_rect_indices"] = False
            rect_idx = None
    else:
        _skip("Stage 6 — rectification indices",
              "neither Stage 3b (K3 1DXD) nor Stage 5 (scaffold) completed")
        completed["stage6_rect_indices"] = False
        rect_idx = None

    # ------------------------------------------------------------------
    # Stage 7: Rectified order images
    # ------------------------------------------------------------------

    _banner("Stage 7: Rectified order images")
    if rect_idx is not None:
        try:
            rect_set = build_rectified_orders(flat_img, rect_idx)
            print(f"  Rectified orders produced: {rect_set.n_orders}")
            _ok("Stage 7 — rectified order images")
            completed["stage7_rectified_orders"] = True

            if not no_plots:
                _plot_rectified_order(rect_set, out_dir, save=save_plots,
                                      prefix=prefix)
        except Exception as exc:  # noqa: BLE001
            print(f"  Stage 7 raised: {exc}")
            _skip("Stage 7 — rectified order images", "exception (see above)")
            completed["stage7_rectified_orders"] = False
            rect_set = None
    else:
        _skip("Stage 7 — rectified order images", "Stage 6 did not complete")
        completed["stage7_rectified_orders"] = False
        rect_set = None

    # ------------------------------------------------------------------
    # Stage 8: Write calibration FITS products
    # ------------------------------------------------------------------

    _banner("Stage 8: Write calibration FITS products")
    if rect_set is not None:
        try:
            cal_products = build_calibration_products(rect_set)
            wavecal_path = ensure_fits_suffix(
                os.path.join(out_dir, wavecal_name)
            )
            spatcal_path = ensure_fits_suffix(
                os.path.join(out_dir, f"{wavecal_name}_spatcal")
            )
            write_wavecal_fits(cal_products, wavecal_path)
            write_spatcal_fits(cal_products, spatcal_path)
            print(f"  Wavecal FITS written : {wavecal_path}")
            print(f"  Spatcal FITS written : {spatcal_path}")
            _ok("Stage 8 — calibration FITS")
            completed["stage8_calibration_fits"] = True
        except Exception as exc:  # noqa: BLE001
            print(f"  Stage 8 raised: {exc}")
            _skip("Stage 8 — calibration FITS", "exception (see above)")
            completed["stage8_calibration_fits"] = False
    else:
        _skip("Stage 8 — calibration FITS", "Stage 7 did not complete")
        completed["stage8_calibration_fits"] = False

    # ------------------------------------------------------------------
    # Not-yet-implemented stages
    # ------------------------------------------------------------------

    _banner("Stages not yet implemented in Python")
    not_impl = [
        "Dark subtraction / dark-frame pipeline",
        "Science-frame preprocessing of K3 object / standard frames",
        "Spectral extraction of K3 object frames (aperture / optimal)",
        "Spectral extraction of K3 standard A0 V frames",
        "Telluric correction of extracted K3 spectra",
        "Order merging into a single 1-D spectrum",
    ]
    for s in not_impl:
        _skip(s, "not yet implemented in Python iSHELL scaffold")
        completed[s] = False

    # ------------------------------------------------------------------
    # Calibration diagnostics
    # ------------------------------------------------------------------

    _banner("Per-order calibration diagnostics")
    order_diags = _build_order_diagnostics(
        prov_map, _non_monotonic_orders, _dispersion_degree
    )
    _print_diagnostics_table(order_diags)

    _banner("Weak-order summary")
    _print_weak_order_warnings(order_diags)

    # Build 1DXD diagnostics when Stage 3b succeeded
    diags_1dxd = None
    if k3_model is not None:
        diags_1dxd = _build_1dxd_diagnostics(k3_model)
        completed["_1dxd_diagnostics"] = diags_1dxd  # type: ignore[assignment]

    if cfg.export_diagnostics:
        diag_path = _export_diagnostics(
            order_diags, out_dir, cfg.diagnostics_format, prefix,
            diags_1dxd=diags_1dxd,
        )
        print(f"\n  [DIAG] Per-order diagnostics exported to: {diag_path}")
        completed["diagnostics_export"] = True
    else:
        completed["diagnostics_export"] = False

    # Export K3 1DXD global summary JSON (always written when Stage 3b succeeded)
    if k3_model is not None:
        summary_path = _export_1dxd_global_summary(k3_model, out_dir, prefix)
        print(f"  [DIAG] K3 1DXD global summary exported to: {summary_path}")
        completed["_1dxd_summary_path"] = summary_path  # type: ignore[assignment]

    # Expose diagnostics on the returned mapping for programmatic access.
    completed["_order_diagnostics"] = order_diags  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    _banner("K3 Benchmark Summary")
    _display_keys = {k: v for k, v in completed.items()
                     if not k.startswith("_")}
    total = len(_display_keys)
    done = sum(v for v in _display_keys.values())
    print(f"  Completed stages : {done} / {total}")
    print()
    for stage, ok in _display_keys.items():
        mark = "OK" if ok else "--"
        print(f"  [{mark}]  {stage}")

    return completed


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated string of integers, e.g. ``"6,7,8,9,10"``."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="iSHELL K3 benchmark driver (Python scaffold)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- paths ---
    parser.add_argument(
        "--raw-dir",
        default=None,
        metavar="PATH",
        help="Raw K3 data directory "
             "(default: data/testdata/ishell_k3_example/raw/)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help="Output directory for FITS products and QA plots "
             "(default: data/testdata/ishell_k3_example/output/)",
    )

    # --- frame numbers ---
    parser.add_argument(
        "--flat-frames",
        default=None,
        metavar="INTS",
        help="Comma-separated flat frame numbers (default: 6,7,8,9,10)",
    )
    parser.add_argument(
        "--arc-frames",
        default=None,
        metavar="INTS",
        help="Comma-separated arc frame numbers (default: 11,12)",
    )
    parser.add_argument(
        "--dark-frames",
        default=None,
        metavar="INTS",
        help="Comma-separated dark frame numbers (default: 25,26,27,28,29)",
    )
    parser.add_argument(
        "--object-frames",
        default=None,
        metavar="INTS",
        help="Comma-separated object spc frame numbers (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--standard-frames",
        default=None,
        metavar="INTS",
        help="Comma-separated standard spc frame numbers "
             "(default: 13,14,15,16,17)",
    )

    # --- output names ---
    parser.add_argument(
        "--flat-output-name",
        default=None,
        metavar="NAME",
        help="Stem for flat calibration output file (default: flat6-10)",
    )
    parser.add_argument(
        "--wavecal-output-name",
        default=None,
        metavar="NAME",
        help="Stem for wavecal output file (default: wavecal11-12)",
    )
    parser.add_argument(
        "--dark-output-name",
        default=None,
        metavar="NAME",
        help="Stem for dark output file (default: dark25-29)",
    )
    parser.add_argument(
        "--qa-plot-prefix",
        default=None,
        metavar="PREFIX",
        help="Prefix for QA plot filenames (default: qa)",
    )

    # --- mode ---
    parser.add_argument(
        "--mode-name",
        default=None,
        metavar="MODE",
        help="iSHELL observing mode (default: K3)",
    )

    # --- plot behaviour ---
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=False,
        help="Save QA plots as PNG files instead of displaying them.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip all QA plotting.",
    )

    # --- diagnostics export ---
    parser.add_argument(
        "--export-diagnostics",
        action="store_true",
        default=False,
        help="Export per-order calibration diagnostics to a CSV or JSON file.",
    )
    parser.add_argument(
        "--diagnostics-format",
        default=None,
        metavar="FMT",
        choices=["csv", "json"],
        help="Format for the diagnostics export file: csv or json (default: csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Build overrides dict from CLI — only include values that were actually
    # specified so that K3BenchmarkConfig defaults remain in effect otherwise.
    overrides: dict = {}
    if args.raw_dir is not None:
        overrides["raw_dir"] = args.raw_dir
    if args.out_dir is not None:
        overrides["output_dir"] = args.out_dir
    if args.flat_frames is not None:
        overrides["flat_frames"] = _parse_int_list(args.flat_frames)
    if args.arc_frames is not None:
        overrides["arc_frames"] = _parse_int_list(args.arc_frames)
    if args.dark_frames is not None:
        overrides["dark_frames"] = _parse_int_list(args.dark_frames)
    if args.object_frames is not None:
        overrides["object_frames"] = _parse_int_list(args.object_frames)
    if args.standard_frames is not None:
        overrides["standard_frames"] = _parse_int_list(args.standard_frames)
    if args.flat_output_name is not None:
        overrides["flat_output_name"] = args.flat_output_name
    if args.wavecal_output_name is not None:
        overrides["wavecal_output_name"] = args.wavecal_output_name
    if args.dark_output_name is not None:
        overrides["dark_output_name"] = args.dark_output_name
    if args.qa_plot_prefix is not None:
        overrides["qa_plot_prefix"] = args.qa_plot_prefix
    if args.mode_name is not None:
        overrides["mode_name"] = args.mode_name
    if args.save_plots:
        overrides["save_plots"] = True
    if args.no_plots:
        overrides["no_plots"] = True
    if args.export_diagnostics:
        overrides["export_diagnostics"] = True
    if args.diagnostics_format is not None:
        overrides["diagnostics_format"] = args.diagnostics_format

    try:
        run_k3_example(**overrides)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
