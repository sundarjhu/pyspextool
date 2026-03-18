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
    out_dir: str,
    save: bool,
    prefix: str,
) -> None:
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
    trace = trace_orders_from_flat(flat_files)
    geom = trace.to_order_geometry_set(mode)
    print(f"  Orders detected : {trace.n_orders}")
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
        _plot_wavecal_residuals(prov_map, out_dir, save=save_plots,
                                prefix=prefix)

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

    try:
        run_k3_example(**overrides)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
