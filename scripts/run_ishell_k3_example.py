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

    --raw-dir   PATH   Override the default raw-data directory.
    --out-dir   PATH   Override the default output directory.
    --save-plots       Save QA plots as PNG files (default: display them).
    --no-plots         Skip all QA plotting.

All input FITS files may be either ``.fits`` or ``.fits.gz``.
Output FITS files are always written as plain ``.fits``.

Data location (defaults)
------------------------
  Input  : data/testdata/ishell_k3_example/raw/
  Output : data/testdata/ishell_k3_example/output/

Implemented stages
------------------
  Stage 1  – Flat/order-centre tracing  (tracing.py)
  Stage 2  – Arc-line tracing           (arc_tracing.py)
  Stage 3  – Provisional wavelength map (wavecal_2d.py)
  Stage 4  – Global wavelength surface  (wavecal_2d_surface.py)
  Stage 5  – Coefficient refinement     (wavecal_2d_refine.py)
  Stage 6  – Rectification indices      (rectification_indices.py)
  Stage 7  – Rectified order images     (rectified_orders.py)
  Stage 8  – Calibration FITS products  (calibration_fits.py)
  QA plots – Python scaffold QA figures (matplotlib)

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
import logging
import os
import sys
import warnings

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
    write_calibration_fits,
)
from pyspextool.instruments.ishell.calibrations import (  # noqa: E402
    read_wavecalinfo,
    read_line_list,
)
from pyspextool.instruments.ishell.io_utils import (  # noqa: E402
    find_fits_files,
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
# K3 mode name
# ---------------------------------------------------------------------------

_MODE = "K3"

# ---------------------------------------------------------------------------
# Frame-number groups from the IDL Spextool manual
# ---------------------------------------------------------------------------

_FLAT_FRAME_NUMBERS = list(range(6, 11))    # frames 6–10
_ARC_FRAME_NUMBERS = list(range(11, 13))    # frames 11–12
_DARK_FRAME_NUMBERS = list(range(25, 30))   # frames 25–29
_OBJECT_FRAME_NUMBERS = list(range(1, 6))   # spc frames 1–5
_STANDARD_FRAME_NUMBERS = list(range(13, 18))  # spc frames 13–17


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


def _banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def _ok(stage: str) -> None:
    print(f"  [OK]  {stage}")


def _skip(stage: str, reason: str) -> None:
    print(f"  [--]  {stage}  ({reason})")


# ---------------------------------------------------------------------------
# QA plotting helpers
# ---------------------------------------------------------------------------


def _plot_flat_orders(flat_img, trace, out_dir: str, save: bool) -> None:
    """Python scaffold QA: traced order centres overlaid on the combined flat."""
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
        cols = trace.sample_cols
        for i in range(trace.n_orders):
            centers = trace.center_rows[i]
            ax.plot(cols, centers, lw=0.6, alpha=0.8)

        ax.set_title(
            "Python scaffold QA — K3 flat: traced order centres\n"
            f"({trace.n_orders} orders detected)"
        )
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")

        if save:
            out_path = os.path.join(out_dir, "qa_flat_orders.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] flat-orders plot skipped: {exc}")


def _plot_arc_lines(arc_img, arc_result, out_dir: str, save: bool) -> None:
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

        for line in arc_result.lines:
            seed_col = line.seed_col
            # Evaluate the traced row at the seed column using the polynomial
            row = float(
                sum(
                    c * (seed_col ** k)
                    for k, c in enumerate(line.poly_coeffs)
                )
            )
            ax.plot(seed_col, row, "cx", markersize=4, alpha=0.7)

        ax.set_title(
            "Python scaffold QA — K3 arc: traced arc-line seed positions\n"
            f"({arc_result.n_lines} lines traced)"
        )
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")

        if save:
            out_path = os.path.join(out_dir, "qa_arc_lines.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] arc-lines plot skipped: {exc}")


def _plot_wavecal_residuals(prov_map, out_dir: str, save: bool) -> None:
    """Python scaffold QA: wavelength-solution match residuals."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        residuals_nm: list[float] = []
        order_indices: list[int] = []
        for sol in prov_map.order_solutions:
            for m in sol.accepted_matches:
                residuals_nm.append(m.match_residual_um * 1000.0)
                order_indices.append(sol.order_number)

        if not residuals_nm:
            print("  [QA] No wavecal matches to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(order_indices, residuals_nm, s=8, alpha=0.6)
        axes[0].axhline(np.median(residuals_nm), color="r", lw=1.5,
                        label=f"median = {np.median(residuals_nm):.2f} nm")
        axes[0].set_xlabel("Echelle order number (provisional)")
        axes[0].set_ylabel("|residual| (nm)")
        axes[0].set_title("Residuals by order")
        axes[0].legend(fontsize=8)

        axes[1].hist(residuals_nm, bins=30, edgecolor="k", linewidth=0.5)
        axes[1].set_xlabel("|residual| (nm)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(
            f"Residual histogram  n={len(residuals_nm)}\n"
            f"median={np.median(residuals_nm):.2f} nm  "
            f"RMS={np.std(residuals_nm):.2f} nm"
        )

        fig.suptitle(
            "Python scaffold QA — K3 provisional wavelength-solution residuals",
            y=1.01,
        )
        fig.tight_layout()

        if save:
            out_path = os.path.join(out_dir, "qa_wavecal_residuals.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] wavecal-residuals plot skipped: {exc}")


def _plot_rectified_order(rect_set, out_dir: str, save: bool) -> None:
    """Python scaffold QA: first available rectified order image."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Pick the first order that has data
        img = None
        order_idx = None
        for i, order in enumerate(rect_set.orders):
            if order.image is not None and order.image.size > 0:
                img = order.image
                order_idx = i
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
            f"Python scaffold QA — K3 rectified order (index {order_idx})\n"
            "NOTE: provisional scaffold; not science-quality"
        )
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Spatial row (pixels)")

        if save:
            out_path = os.path.join(out_dir, "qa_rectified_order.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  [QA] Saved: {out_path}")
        else:
            plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"  [QA] rectified-order plot skipped: {exc}")


# ---------------------------------------------------------------------------
# Main reduction driver
# ---------------------------------------------------------------------------


def run_k3_example(
    raw_dir: str,
    out_dir: str,
    *,
    save_plots: bool = False,
    no_plots: bool = False,
) -> dict[str, bool]:
    """Run the K3 benchmark reduction chain.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw K3 FITS files (``.fits`` or ``.fits.gz``).
    out_dir : str
        Directory where output FITS files and QA plots are written.
    save_plots : bool
        If ``True``, save QA plots as PNG files instead of displaying them.
    no_plots : bool
        If ``True``, skip all QA plotting.

    Returns
    -------
    dict of str → bool
        Mapping from stage name to completion status.

    Raises
    ------
    FileNotFoundError
        If *raw_dir* does not exist or the required K3 flat/arc files are
        absent.
    """
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

    flat_files = _select_files(all_files, "flat", _FLAT_FRAME_NUMBERS)
    arc_files = _select_files(all_files, "arc", _ARC_FRAME_NUMBERS)
    dark_files = _select_files(all_files, "dark", _DARK_FRAME_NUMBERS)
    object_files = _select_files(all_files, "spc", _OBJECT_FRAME_NUMBERS)
    standard_files = _select_files(all_files, "spc", _STANDARD_FRAME_NUMBERS)

    _banner("K3 Benchmark: file discovery")
    print(f"  Raw dir     : {raw_dir}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Flat files  : {len(flat_files)}  (frames 6–10)")
    print(f"  Arc files   : {len(arc_files)}   (frames 11–12)")
    print(f"  Dark files  : {len(dark_files)}  (frames 25–29)")
    print(f"  Object spc  : {len(object_files)}  (frames 1–5)")
    print(f"  Std A0V spc : {len(standard_files)}  (frames 13–17)")

    if not flat_files:
        raise FileNotFoundError(
            f"No flat files (frames 6–10) found in {raw_dir!r}. "
            "Ensure the K3 FITS files are present."
        )
    if not arc_files:
        raise FileNotFoundError(
            f"No arc files (frames 11–12) found in {raw_dir!r}. "
            "Ensure the K3 FITS files are present."
        )

    # ------------------------------------------------------------------
    # Load packaged K3 calibration resources
    # ------------------------------------------------------------------

    wavecalinfo = read_wavecalinfo(_MODE)
    line_list = read_line_list(_MODE)

    # ------------------------------------------------------------------
    # Stage 1: Flat/order tracing
    # ------------------------------------------------------------------

    _banner("Stage 1: Flat / order-centre tracing")
    flat_img = load_and_combine_flats(flat_files)
    trace = trace_orders_from_flat(flat_files)
    geom = trace.to_order_geometry_set(_MODE)
    print(f"  Orders detected : {trace.n_orders}")
    print(f"  Median fit RMS  : {float(trace.fit_rms.mean()):.2f} px (mean across orders)")
    _ok("Stage 1 — flat/order tracing")
    completed["stage1_flat_tracing"] = True

    if not no_plots:
        _plot_flat_orders(flat_img, trace, out_dir, save=save_plots)

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
        _plot_arc_lines(arc_img, arc_result, out_dir, save=save_plots)

    # ------------------------------------------------------------------
    # Stage 3: Provisional wavelength map
    # ------------------------------------------------------------------

    _banner("Stage 3: Provisional per-order wavelength mapping")
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
    _ok("Stage 3 — provisional wavelength mapping")
    completed["stage3_provisional_wavemap"] = True

    if not no_plots:
        _plot_wavecal_residuals(prov_map, out_dir, save=save_plots)

    # ------------------------------------------------------------------
    # Stage 4: Global wavelength surface
    # ------------------------------------------------------------------

    _banner("Stage 4: Global wavelength surface")
    try:
        surface_result = fit_global_wavelength_surface(prov_map)
        print(f"  Surface fit converged: {getattr(surface_result, 'converged', 'N/A')}")
        _ok("Stage 4 — global wavelength surface")
        completed["stage4_global_surface"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"  Stage 4 raised: {exc}")
        _skip("Stage 4 — global wavelength surface", "exception (see above)")
        completed["stage4_global_surface"] = False
        surface_result = None

    # ------------------------------------------------------------------
    # Stage 5: Coefficient refinement
    # ------------------------------------------------------------------

    _banner("Stage 5: Coefficient-surface refinement")
    try:
        refined = fit_refined_coefficient_surface(prov_map)
        print(f"  Refined surface n_orders: {getattr(refined, 'n_orders', 'N/A')}")
        _ok("Stage 5 — coefficient refinement")
        completed["stage5_refinement"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"  Stage 5 raised: {exc}")
        _skip("Stage 5 — coefficient refinement", "exception (see above)")
        completed["stage5_refinement"] = False
        refined = None

    # ------------------------------------------------------------------
    # Stage 6: Rectification indices
    # ------------------------------------------------------------------

    _banner("Stage 6: Rectification indices")
    if refined is not None:
        try:
            rect_idx = build_rectification_indices(
                geom, refined, wav_map=prov_map
            )
            print(f"  Rectification index orders: {rect_idx.n_orders}")
            _ok("Stage 6 — rectification indices")
            completed["stage6_rect_indices"] = True
        except Exception as exc:  # noqa: BLE001
            print(f"  Stage 6 raised: {exc}")
            _skip("Stage 6 — rectification indices", "exception (see above)")
            completed["stage6_rect_indices"] = False
            rect_idx = None
    else:
        _skip("Stage 6 — rectification indices", "Stage 5 did not complete")
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
                _plot_rectified_order(rect_set, out_dir, save=save_plots)
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
            paths = write_calibration_fits(cal_products, out_dir)
            print(f"  Wavecal FITS written : {paths[0]}")
            print(f"  Spatcal FITS written : {paths[1]}")
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
    # Summary
    # ------------------------------------------------------------------

    _banner("K3 Benchmark Summary")
    total = len(completed)
    done = sum(v for v in completed.values())
    print(f"  Completed stages : {done} / {total}")
    print()
    for stage, ok in completed.items():
        mark = "OK" if ok else "--"
        print(f"  [{mark}]  {stage}")

    return completed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="iSHELL K3 benchmark driver (Python scaffold)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        metavar="PATH",
        help="Raw K3 data directory (default: data/testdata/ishell_k3_example/raw/)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help="Output directory for FITS products and QA plots "
             "(default: data/testdata/ishell_k3_example/output/)",
    )
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    raw_dir = args.raw_dir or _default_raw_dir()
    out_dir = args.out_dir or _default_out_dir()

    try:
        run_k3_example(
            raw_dir=raw_dir,
            out_dir=out_dir,
            save_plots=args.save_plots,
            no_plots=args.no_plots,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
