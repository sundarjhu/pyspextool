"""
Stage 3b pixel-error diagnostics for K3 weak orders.

This module measures the pixel distance (dx) between predicted arc-line
column positions (from the current global 1DXD model) and the nearest
actual peak detected in the extracted 1-D arc spectrum.  It is a
**pure diagnostic** — it does NOT modify any pipeline state.

Public API
----------
- :func:`run_k3_arc_dx_diagnostics` – run the full diagnostic and save plots.
- :class:`OrderDxStats` – per-order summary statistics.

Typical usage
-------------
After :func:`~pyspextool.instruments.ishell.wavecal_k3_idlstyle.fit_1dxd_wavelength_model`
has returned a fitted :class:`~pyspextool.instruments.ishell.wavecal_k3_idlstyle.IdlStyle1DXDModel`::

    from pyspextool.instruments.ishell.k3_arc_dx_diagnostics import (
        run_k3_arc_dx_diagnostics,
    )

    stats = run_k3_arc_dx_diagnostics(
        model=k3_model,
        arc_spectra=arc_spectra,
        line_list=line_list,
        out_dir="qa_dx",
    )
    for s in stats:
        print(s)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .calibrations import LineList
    from .wavecal_k3_idlstyle import IdlStyle1DXDModel, OrderArcSpectraSet

__all__ = [
    "OrderDxStats",
    "run_k3_arc_dx_diagnostics",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded weak orders (K3 Stage 3b confirmed failures)
# ---------------------------------------------------------------------------

WEAK_ORDERS: list[int] = [204, 205, 212, 213, 214, 217, 221, 228]

# Number of column samples used for model inversion (wavelength → column)
_N_COL_SAMPLES = 4096


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OrderDxStats:
    """Per-order pixel-error statistics between predicted positions and peaks.

    Parameters
    ----------
    order_number : int
        Echelle order number.
    n_predicted : int
        Number of reference wavelengths whose predicted column fell inside
        the valid column range of the extracted spectrum.
    n_peaks : int
        Number of peaks detected in the smoothed spectrum above threshold.
    dx_values : ndarray, shape (n_predicted,)
        Distance (pixels) from each predicted column to the nearest detected
        peak.  Values are stored even when the nearest peak is far away.
    median_dx : float
        Median of *dx_values* (pixels).  ``NaN`` if *n_predicted* == 0.
    min_dx : float
        Minimum of *dx_values* (pixels).  ``NaN`` if *n_predicted* == 0.
    max_dx : float
        Maximum of *dx_values* (pixels).  ``NaN`` if *n_predicted* == 0.
    """

    order_number: int
    n_predicted: int
    n_peaks: int
    dx_values: npt.NDArray = field(default_factory=lambda: np.empty(0, dtype=float))
    median_dx: float = float("nan")
    min_dx: float = float("nan")
    max_dx: float = float("nan")

    def __str__(self) -> str:
        return (
            f"Order {self.order_number:3d}: "
            f"n_pred={self.n_predicted:3d}  n_peaks={self.n_peaks:3d}  "
            f"median_dx={self.median_dx:6.2f}  "
            f"min_dx={self.min_dx:6.2f}  "
            f"max_dx={self.max_dx:6.2f}  px"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _smooth_spectrum(flux: npt.NDArray, sigma: float = 1.5) -> npt.NDArray:
    """Return a Gaussian-smoothed copy of *flux* (NaN-safe)."""
    from scipy.ndimage import gaussian_filter1d

    # Replace NaN with 0 for smoothing, then restore NaN mask
    nan_mask = np.isnan(flux)
    f = flux.copy()
    f[nan_mask] = 0.0
    smoothed = gaussian_filter1d(f.astype(float), sigma=sigma)
    smoothed[nan_mask] = np.nan
    return smoothed


def _detect_peaks(
    flux: npt.NDArray,
    columns: npt.NDArray,
) -> npt.NDArray:
    """Detect local maxima above ``median + 3 * MAD`` threshold.

    Parameters
    ----------
    flux : ndarray
        1-D spectrum values (may contain NaN).
    columns : ndarray
        Corresponding detector column indices.

    Returns
    -------
    ndarray
        Detector column indices of detected peaks.
    """
    from scipy.signal import find_peaks as _find_peaks

    valid = ~np.isnan(flux)
    if not np.any(valid):
        return np.empty(0, dtype=float)

    f_valid = flux[valid]
    median_f = float(np.median(f_valid))
    mad_f = float(np.median(np.abs(f_valid - median_f)))
    threshold = median_f + 3.0 * mad_f

    # find_peaks operates on indices; we need full-array indices
    peak_indices, _ = _find_peaks(flux, height=threshold)
    if len(peak_indices) == 0:
        return np.empty(0, dtype=float)
    return columns[peak_indices].astype(float)


def _invert_model_for_order(
    model: "IdlStyle1DXDModel",
    order_number: int,
    col_start: int,
    col_end: int,
    ref_wavelengths_um: npt.NDArray,
) -> npt.NDArray:
    """Compute predicted column positions for *ref_wavelengths_um*.

    Inverts the 1DXD polynomial by evaluating it on a dense column grid
    and using linear interpolation to map wavelength → column.

    Parameters
    ----------
    model : IdlStyle1DXDModel
        Fitted 1DXD model.
    order_number : int
        Echelle order number.
    col_start, col_end : int
        Valid column range of the extracted spectrum.
    ref_wavelengths_um : ndarray
        Reference wavelengths (µm) to predict column positions for.

    Returns
    -------
    ndarray, shape (len(ref_wavelengths_um),)
        Predicted detector columns.  Values outside ``[col_start, col_end]``
        are kept so the caller can filter them.
    """
    # Dense column grid over the valid range
    cols_dense = np.linspace(col_start, col_end, _N_COL_SAMPLES)
    wavs_dense = model.eval_array(
        cols_dense,
        np.full_like(cols_dense, float(order_number)),
    )

    # Sort by wavelength (monotonicity required for np.interp)
    sort_idx = np.argsort(wavs_dense)
    wavs_sorted = wavs_dense[sort_idx]
    cols_sorted = cols_dense[sort_idx]

    # Invert: wavelength → column via linear interpolation
    x_pred = np.interp(ref_wavelengths_um, wavs_sorted, cols_sorted)
    return x_pred


def _compute_dx(
    x_pred: npt.NDArray,
    peak_cols: npt.NDArray,
) -> npt.NDArray:
    """Return nearest-peak distance for each predicted position.

    Parameters
    ----------
    x_pred : ndarray, shape (n,)
        Predicted detector columns.
    peak_cols : ndarray, shape (m,)
        Detected peak column positions.

    Returns
    -------
    ndarray, shape (n,)
        ``dx[i] = min_j |peak_cols[j] - x_pred[i]|``.
        If *peak_cols* is empty, ``dx`` is filled with ``np.inf``.
    """
    if len(peak_cols) == 0:
        return np.full(len(x_pred), np.inf)
    dx = np.array(
        [float(np.min(np.abs(peak_cols - xp))) for xp in x_pred],
        dtype=float,
    )
    return dx


# ---------------------------------------------------------------------------
# Per-order diagnostic
# ---------------------------------------------------------------------------


def _diagnose_order(
    order_number: int,
    model: "IdlStyle1DXDModel",
    arc_spectra: "OrderArcSpectraSet",
    line_list: "LineList",
    out_dir: Optional[str],
    save_plots: bool,
    sigma: float = 1.5,
) -> Optional[OrderDxStats]:
    """Run the diagnostic for a single order and return :class:`OrderDxStats`.

    Returns ``None`` if the order is not present in *arc_spectra*.
    """
    # ------------------------------------------------------------------
    # Step 1 – Load extracted 1-D arc spectrum
    # ------------------------------------------------------------------
    try:
        spectrum = arc_spectra.get_spectrum(order_number)
    except KeyError:
        logger.debug("_diagnose_order: order %d not in arc_spectra — skipping", order_number)
        return None

    columns = spectrum.columns.astype(float)   # pixel x-axis
    flux = spectrum.flux.astype(float)          # raw spectrum

    # ------------------------------------------------------------------
    # Step 2 – Smooth spectrum
    # ------------------------------------------------------------------
    flux_smooth = _smooth_spectrum(flux, sigma=sigma)

    # ------------------------------------------------------------------
    # Step 3 – Detect peaks
    # ------------------------------------------------------------------
    peak_cols = _detect_peaks(flux_smooth, columns)

    # ------------------------------------------------------------------
    # Step 4 – Get reference wavelengths for this order (predicted positions)
    # ------------------------------------------------------------------
    ref_entries = [
        e.wavelength_um
        for e in line_list.entries
        if e.order == order_number
    ]
    if len(ref_entries) == 0:
        logger.debug(
            "_diagnose_order: order %d has no reference lines — skipping", order_number
        )
        return None

    ref_wavs = np.array(ref_entries, dtype=float)

    # ------------------------------------------------------------------
    # Step 5 – Invert 1DXD model: wavelength → column
    # ------------------------------------------------------------------
    col_start = int(spectrum.col_start)
    col_end = int(spectrum.col_end)

    x_pred_all = _invert_model_for_order(
        model, order_number, col_start, col_end, ref_wavs
    )

    # Keep only predictions that fall within the valid column range
    in_range = (x_pred_all >= col_start) & (x_pred_all <= col_end)
    x_pred = x_pred_all[in_range]

    # ------------------------------------------------------------------
    # Step 6 – Measure nearest-peak distance
    # ------------------------------------------------------------------
    dx = _compute_dx(x_pred, peak_cols)

    # ------------------------------------------------------------------
    # Step 7 – Compute stats
    # ------------------------------------------------------------------
    n_predicted = len(x_pred)
    n_peaks = len(peak_cols)

    if n_predicted > 0 and not np.all(np.isinf(dx)):
        finite_dx = dx[np.isfinite(dx)]
        median_dx = float(np.median(finite_dx)) if len(finite_dx) > 0 else float("nan")
        min_dx = float(np.min(finite_dx)) if len(finite_dx) > 0 else float("nan")
        max_dx = float(np.max(finite_dx)) if len(finite_dx) > 0 else float("nan")
    else:
        median_dx = min_dx = max_dx = float("nan")

    stats = OrderDxStats(
        order_number=order_number,
        n_predicted=n_predicted,
        n_peaks=n_peaks,
        dx_values=dx,
        median_dx=median_dx,
        min_dx=min_dx,
        max_dx=max_dx,
    )

    # ------------------------------------------------------------------
    # Step 8 – Plot
    # ------------------------------------------------------------------
    if out_dir is not None:
        _plot_order(
            order_number=order_number,
            columns=columns,
            flux_smooth=flux_smooth,
            peak_cols=peak_cols,
            x_pred=x_pred,
            stats=stats,
            out_dir=out_dir,
            save=save_plots,
        )

    return stats


def _plot_order(
    order_number: int,
    columns: npt.NDArray,
    flux_smooth: npt.NDArray,
    peak_cols: npt.NDArray,
    x_pred: npt.NDArray,
    stats: OrderDxStats,
    out_dir: str,
    save: bool = True,
) -> None:
    """Create and optionally save the per-order diagnostic plot."""
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))

    # Spectrum (gray)
    ax.plot(columns, flux_smooth, color="gray", linewidth=0.8, label="spectrum (smoothed)")

    # Detected peaks (blue dots)
    if len(peak_cols) > 0:
        peak_flux = np.interp(peak_cols, columns, flux_smooth)
        ax.plot(
            peak_cols, peak_flux,
            "o", color="blue", markersize=5, label=f"peaks (n={stats.n_peaks})",
        )

    # Predicted positions (red vertical lines)
    y_min, y_max = ax.get_ylim()
    for xp in x_pred:
        ax.axvline(xp, color="red", linewidth=0.7, alpha=0.7)

    # Dummy handle for predicted lines in legend
    from matplotlib.lines import Line2D
    pred_handle = Line2D([], [], color="red", linewidth=0.7, label=f"predicted (n={stats.n_predicted})")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(pred_handle)

    ax.legend(handles=handles, fontsize=8, loc="upper right")

    title = (
        f"Order {order_number}   "
        f"median_dx={stats.median_dx:.2f} px   "
        f"min_dx={stats.min_dx:.2f} px   "
        f"max_dx={stats.max_dx:.2f} px"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Detector column (pixels)")
    ax.set_ylabel("Flux (smoothed)")

    fig.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"order_{order_number}.png")
        fig.savefig(out_path, dpi=100)
        logger.info("Saved diagnostic plot: %s", out_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_k3_arc_dx_diagnostics(
    model: "IdlStyle1DXDModel",
    arc_spectra: "OrderArcSpectraSet",
    line_list: "LineList",
    *,
    out_dir: Optional[str] = "qa_dx",
    save_plots: bool = True,
    weak_orders: Optional[list[int]] = None,
    sigma: float = 1.5,
) -> list[OrderDxStats]:
    """Run the Stage 3b arc-line pixel-error diagnostic for K3 weak orders.

    For each weak order this function:

    1. Loads the extracted 1-D arc spectrum from *arc_spectra*.
    2. Smooths it with a Gaussian (``sigma`` pixels).
    3. Detects peaks above ``median + 3 × MAD``.
    4. Inverts the 1DXD *model* to predict detector column positions for
       every reference wavelength in *line_list*.
    5. Measures the nearest-peak distance for each prediction.
    6. Computes per-order statistics and saves a diagnostic plot to *out_dir*.

    This function does **not** modify any pipeline state.

    Parameters
    ----------
    model : IdlStyle1DXDModel
        Fitted global 1DXD wavelength model from Stage 3b.
    arc_spectra : OrderArcSpectraSet
        Extracted 1-D arc spectra (one per order) from Stage 3b.
    line_list : LineList
        Packaged K3 ThAr arc-line reference list.
    out_dir : str or None, optional
        Directory to write per-order PNG plots.  Created if it does not
        exist.  Pass ``None`` to skip saving plots.  Default: ``"qa_dx"``.
    save_plots : bool, optional
        If *True* (default), save plots as PNG files.  If *False*, display
        them interactively (useful in notebooks).  Ignored if *out_dir* is
        ``None``.
    weak_orders : list of int or None, optional
        Override the hardcoded list of weak orders.  If *None* (default),
        uses :data:`WEAK_ORDERS`.
    sigma : float, optional
        Gaussian smoothing width in pixels.  Default: ``1.5``.

    Returns
    -------
    list of OrderDxStats
        Per-order statistics, one entry per order that was found in
        *arc_spectra* and had at least one reference wavelength in
        *line_list*.  Orders that are missing from *arc_spectra* are
        silently skipped.
    """
    orders_to_check = weak_orders if weak_orders is not None else WEAK_ORDERS

    results: list[OrderDxStats] = []
    for order_number in orders_to_check:
        stats = _diagnose_order(
            order_number=order_number,
            model=model,
            arc_spectra=arc_spectra,
            line_list=line_list,
            out_dir=out_dir,
            save_plots=save_plots,
            sigma=sigma,
        )
        if stats is not None:
            results.append(stats)
            logger.info(str(stats))

    return results
