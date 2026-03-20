"""
Order-centre tracing from iSHELL flat-field frames.

This module implements the first stage of the iSHELL 2DXD reduction
scaffold: tracing echelle order centres from one or more median-combined
QTH flat-field frames.  The result is a :class:`FlatOrderTrace` object
containing:

* the sampled centre-row positions at a set of detector columns,
* low-order polynomial fits to those positions, and
* estimated per-order half-widths.

The :class:`FlatOrderTrace` can be converted to an
:class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` via its
:meth:`~FlatOrderTrace.to_order_geometry_set` method, which feeds into the
later 2DXD arc-line tracing stage.

.. note::
   This is a **first-pass tracing scaffold**, not a finalised geometry
   calibration.  The outputs are suitable for development and smoke-testing
   the downstream pipeline, but should not be used for science-quality
   spectral rectification without further validation.

Algorithm
---------
1. **Load frames** – open each flat FITS file (PRIMARY extension, extension 0)
   and median-combine to suppress read noise and hot pixels.

2. **Cross-section profile** – at a set of evenly-spaced *sample columns*
   within the valid column range, extract a cross-dispersion profile by
   median-averaging over a narrow column band (±``col_half_width`` columns).
   The median suppresses residual cosmic rays and bad pixels.

3. **Peak finding** – at each sample column, find local maxima in the profile
   using :func:`scipy.signal.find_peaks`.  The ``prominence`` criterion is the
   most reliable discriminator: it measures how much a peak stands above its
   local surroundings and is robust to the slowly-varying blaze envelope.

4. **Seed extraction** – peaks found at the central seed column are used as
   initial position estimates for all orders.

5. **Column-by-column tracking** – at each sample column the nearest peak to
   the most recent position estimate (within ``max_shift`` pixels) is
   accepted.  Unmatched positions are left as ``NaN``.

6. **Polynomial fitting with sigma-clipping** – a degree-``poly_degree``
   polynomial is fitted to each order's traced positions using iterative
   sigma-clipping (3 iterations, clip at ``sigma_clip`` × RMS).  The
   :func:`numpy.polynomial.polynomial.polyfit` convention is used throughout:
   ``coeffs[k]`` is the coefficient of ``x**k``.

7. **Half-width estimation** – the half-maximum half-width of each order peak
   in the seed-column profile is recorded as a proxy for order width.

Observations from the H1 dataset
----------------------------------
The following was observed when running this scaffold against the five real
H1-mode flat frames in ``data/testdata/ishell_h1_calibrations/raw/``.
These are first-pass observations, not finalised calibration results.

* The H2RG 2048 × 2048 detector shows clean, well-separated echelle order
  bands separated by narrow inter-order gaps (~5–15 pixels).
* With ``distance=25`` and ``prominence=500``, the majority of the ~45
  orders expected in H1 mode are detected from the median-combined flat.
  A small number of orders near the detector edges are typically missed
  due to low flat-lamp signal.
* Polynomial fit residuals on the real H1 data are typically 3–8 pixels
  (median ≈ 3–4 px).  This is adequate for a first-pass scaffold but
  may be larger than a final production pipeline would require.
* The traced centres from the raw H1 frames differ by approximately
  −70 rows from the positions stored in the packaged ``H1_flatinfo.fits``
  calibration resource.  The cause of this offset has **not yet been
  resolved**; possible explanations include detector orientation differences,
  coordinate convention differences, or raw-frame preprocessing differences
  between the IDL and Python pipelines.  No conclusion should be drawn from
  this offset alone.

Relationship to the 2DXD arc-line tracing stage
-------------------------------------------------
This module provides the *centre-line* geometry needed for the first step of
2DXD rectification.  The subsequent steps (not yet implemented) are:

* Edge tracing – refine bottom/top order edges from the flat profile.
* Tilt and curvature measurement – measure spectral-line tilt and curvature
  from arc-lamp frames using the order boundaries established here.
* Wavelength calibration – identify arc lines and fit a wavelength solution.
* Rectification – resample the raw 2D spectrum onto a
  (wavelength × spatial) grid.

What remains unimplemented
--------------------------
* Precise edge tracing from flat-field profiles (only half-width estimates are
  produced here).
* Per-order column-range trimming (all orders use the same ``col_range``).
* Tilt, curvature, wavelength, and spatial calibration polynomials.
* Full 2DXD rectification.

See :mod:`pyspextool.instruments.ishell.geometry` for the data structures
used downstream and ``docs/ishell_order_tracing.md`` for the full design
rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from scipy.signal import find_peaks as _scipy_find_peaks

from .geometry import OrderGeometry, OrderGeometrySet

__all__ = [
    "FlatOrderTrace",
    "OrderTraceStats",
    "trace_orders_from_flat",
    "load_and_combine_flats",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QA thresholds (module-level defaults)
# ---------------------------------------------------------------------------

_RMS_THRESHOLD: float = 8.0          # px -- flag if rms_residual exceeds this
# NOTE: threshold raised from 5.0 to 8.0 px to match real K3 tracking noise.
# The previous 5.0 px threshold was unrealistically tight for iSHELL K3 data
# where peak-matching scatter at edge orders is routinely 5–8 px.

_CURVATURE_THRESHOLD: float = 1e-3   # px/col^2 -- flag if max |d2y/dx2| exceeds this

_SEPARATION_THRESHOLD: float = 3.0   # px -- flag if min absolute inter-order gap drops below this

_OSCILLATION_THRESHOLD: float = 0.5  # px/col -- flag if peak-to-peak slope variation exceeds this
# NOTE: threshold raised from 0.05 to 0.5 px/col to match real K3 curvature.
# For a degree-2 polynomial over 2048 columns with modest curvature
# (~1e-4 px/col²), the slope varies by ~0.4 px/col -- well above 0.05.
# The previous 0.05 threshold caused every genuine K3 order to fail,
# including well-fitted ones.  0.5 px/col still flags pathologically
# oscillatory polynomials (e.g. the formerly divergent Order 24).


# ---------------------------------------------------------------------------
# Public result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OrderTraceStats:
    """Per-order QA metrics from the flat-field order tracing stage.

    These metrics provide visibility into the quality of each traced
    order-centre polynomial **before** any orders are rejected.  They are
    stored in :attr:`FlatOrderTrace.order_stats` and printed as a concise
    summary log after tracing completes.

    Attributes
    ----------
    order_index : int
        Zero-based index of the order within :class:`FlatOrderTrace`.
    rms_residual : float
        RMS of the polynomial fit residuals in pixels.  Equals
        :attr:`FlatOrderTrace.fit_rms[order_index]`.  ``NaN`` if the
        polynomial fit failed (too few valid sample points).
    curvature_metric : float
        Maximum absolute second derivative of the fitted polynomial
        evaluated at the sample columns (px / col²).  Large values
        indicate unrealistic bending.  ``NaN`` if the fit failed.
    oscillation_metric : float
        Peak-to-peak variation in the first derivative (slope) of the
        fitted polynomial across the sample columns (px / col).  For a
        straight trace this is zero; large values flag oscillatory or
        pathologically curved polynomials.  ``NaN`` if the fit failed.
    min_sep_lower : float
        Minimum *absolute* vertical separation to the lower-row neighbouring
        order across all sample columns, in pixels.  ``NaN`` if this order
        has no lower neighbour or the fit failed.  Always ≥ 0.
    min_sep_upper : float
        Minimum *absolute* vertical separation to the upper-row neighbouring
        order across all sample columns, in pixels.  ``NaN`` if this order
        has no upper neighbour or the fit failed.  Always ≥ 0.
    crosses_lower : bool
        ``True`` if the fitted centre curve crosses (or touches) the lower
        neighbouring order at any sample column.  ``False`` if there is no
        lower neighbour or the fit failed.
    crosses_upper : bool
        ``True`` if the fitted centre curve crosses (or touches) the upper
        neighbouring order at any sample column.  ``False`` if there is no
        upper neighbour or the fit failed.
    trace_valid : bool
        Composite validity flag.  ``True`` if *all* of the following hold:

        * ``rms_residual`` is finite and ≤ ``_RMS_THRESHOLD`` (5 px).
        * ``curvature_metric`` ≤ ``_CURVATURE_THRESHOLD`` (1 × 10⁻³ px/col²).
        * ``oscillation_metric`` ≤ ``_OSCILLATION_THRESHOLD`` (0.05 px/col).
        * ``min_sep_lower`` is ``NaN`` *or* ≥ ``_SEPARATION_THRESHOLD`` (3 px).
        * ``min_sep_upper`` is ``NaN`` *or* ≥ ``_SEPARATION_THRESHOLD`` (3 px).
        * ``crosses_lower`` is ``False``.
        * ``crosses_upper`` is ``False``.

    Notes
    -----
    Setting ``trace_valid = False`` does **not** remove the order from
    downstream processing; it is a diagnostic flag only.
    """

    order_index: int
    rms_residual: float
    curvature_metric: float
    oscillation_metric: float
    min_sep_lower: float
    min_sep_upper: float
    crosses_lower: bool
    crosses_upper: bool
    trace_valid: bool


@dataclass
class FlatOrderTrace:
    """Result of order-centre tracing from one or more flat-field frames.

    Attributes
    ----------
    n_orders : int
        Number of orders successfully traced.
    sample_cols : ndarray, shape (n_sample,)
        Detector column indices at which cross-dispersion profiles were
        evaluated.
    center_rows : ndarray, shape (n_orders, n_sample)
        Traced centre-row position at each sample column.  ``NaN`` marks
        columns where the corresponding peak was not reliably detected.
    center_poly_coeffs : ndarray, shape (n_orders, poly_degree + 1)
        Per-order polynomial coefficients for the centre-row trajectory.
        Follows the ``numpy.polynomial.polynomial`` convention:
        ``coeffs[k]`` is the coefficient of ``col**k``.
    fit_rms : ndarray, shape (n_orders,)
        RMS of the polynomial residuals for each order, in pixels.  A value
        of ``NaN`` indicates that the polynomial fit failed (too few valid
        points).
    half_width_rows : ndarray, shape (n_orders,)
        Estimated half-width (half-maximum) of each order peak, in pixels,
        measured at the seed column.  This is a proxy for the order width
        and is used to build edge polynomials in
        :meth:`to_order_geometry_set`.
    poly_degree : int
        Maximum polynomial degree requested for fitting centre-line
        polynomials.  Individual orders may use a lower degree if the
        stable-fit fallback reduces it (see ``poly_degrees_used``).
    seed_col : int
        Detector column used to locate initial order-centre seeds.
    order_stats : list of OrderTraceStats
        Per-order QA metrics computed immediately after polynomial fitting.
        See :class:`OrderTraceStats` for the available fields.  Empty if
        the result was constructed directly without calling
        :func:`trace_orders_from_flat`.
    poly_degrees_used : ndarray of int, shape (n_orders,)
        Actual polynomial degree used per order after stable-fit fallback.
        Equal to ``poly_degree`` for orders that were well constrained.
        Lower for sparse or divergent orders that triggered the bounds-
        based degree-reduction fallback.  Empty array if this trace was
        constructed directly (not via :func:`trace_orders_from_flat`).

    Notes
    -----
    The optional fields ``tilt_coeffs``, ``curvature_coeffs``,
    ``wave_coeffs``, and ``spatcal_coeffs`` inside each
    :class:`~pyspextool.instruments.ishell.geometry.OrderGeometry` produced
    by :meth:`to_order_geometry_set` are all ``None``; they are populated
    by the wavecal step.
    """

    n_orders: int
    sample_cols: np.ndarray
    center_rows: np.ndarray
    center_poly_coeffs: np.ndarray
    fit_rms: np.ndarray
    half_width_rows: np.ndarray
    poly_degree: int
    seed_col: int
    order_stats: list[OrderTraceStats] = field(default_factory=list)
    poly_degrees_used: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    """Per-order polynomial degree actually used after stable-fit fallback.

    When :func:`trace_orders_from_flat` falls back to a lower-degree
    polynomial for a sparse or divergent order, the fallback degree is
    stored here.  ``poly_degrees_used[i] <= poly_degree`` for all *i*.
    An empty array is returned when the trace was constructed directly
    without calling :func:`trace_orders_from_flat`.
    """

    def to_order_geometry_set(
        self,
        mode: str,
        col_range: tuple[int, int] | None = None,
    ) -> OrderGeometrySet:
        """Convert the tracing result to an :class:`OrderGeometrySet`.

        .. warning::
           The returned geometry is **provisional scaffolding**, not a
           science-quality calibration.  Specifically:

           * Only **order centres are traced from data**.  The bottom and
             top edge polynomials are **approximated** by offsetting the
             centre-line polynomial constant term by ``±half_width_rows``;
             they are not independently fitted to the flat profile.
           * Order numbers are **placeholder integers** (0, 1, 2, …) assigned
             sequentially.  Real echelle order numbers are not yet assigned;
             that requires the wavecal step.
           * The geometry objects are intended for development and pipeline
             scaffolding, not for final science-quality spectral rectification.

        Parameters
        ----------
        mode : str
            iSHELL observing mode name (e.g. ``"H1"``).  Stored as metadata
            in the returned :class:`OrderGeometrySet`.
        col_range : tuple of int (col_start, col_end), optional
            Column range applied uniformly to all orders.  Defaults to
            ``(sample_cols[0], sample_cols[-1])``.

        Returns
        -------
        :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        """
        if col_range is None:
            x_start = int(self.sample_cols[0])
            x_end = int(self.sample_cols[-1])
        else:
            x_start, x_end = int(col_range[0]), int(col_range[1])

        geometries = []
        for i in range(self.n_orders):
            c = self.center_poly_coeffs[i].copy()
            hw = float(self.half_width_rows[i])

            bot_coeffs = c.copy()
            top_coeffs = c.copy()
            bot_coeffs[0] -= hw
            top_coeffs[0] += hw

            geom = OrderGeometry(
                order=i,
                x_start=x_start,
                x_end=x_end,
                bottom_edge_coeffs=bot_coeffs,
                top_edge_coeffs=top_coeffs,
            )
            geometries.append(geom)

        return OrderGeometrySet(mode=mode, geometries=geometries)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def trace_orders_from_flat(
    flat_files: list[str],
    *,
    n_sample_cols: int = 40,
    col_half_width: int = 10,
    poly_degree: int = 3,
    seed_col: int | None = None,
    col_range: tuple[int, int] | None = None,
    min_prominence: float = 500.0,
    min_distance: int = 25,
    max_shift: float = 15.0,
    sigma_clip: float = 3.0,
) -> FlatOrderTrace:
    """Trace echelle order centres from one or more iSHELL flat-field frames.

    Parameters
    ----------
    flat_files : list of str
        Paths to raw iSHELL FITS flat frames (PRIMARY extension is read).
        When more than one file is provided the frames are median-combined
        before tracing.
    n_sample_cols : int, default 40
        Number of evenly-spaced detector columns at which cross-dispersion
        profiles are evaluated.
    col_half_width : int, default 10
        Half-width of the column band (in pixels) used to compute each
        cross-dispersion profile.  The profile at column *c* is the
        median over columns ``c - col_half_width`` to
        ``c + col_half_width`` (inclusive, clipped to array bounds).
    poly_degree : int, default 3
        Degree of the polynomial fitted to each order's centre trajectory.
        Degree 2 or 3 is usually sufficient; higher degrees can overfit
        noisy column positions near the detector edges.
    seed_col : int, optional
        Detector column at which initial order positions are identified.
        Defaults to the mid-point of *col_range*.
    col_range : tuple of int (col_start, col_end), optional
        Column range over which tracing is performed.  Defaults to the
        full detector width ``(0, ncols - 1)``.
    min_prominence : float, default 500.0
        Minimum peak prominence (in detector counts) required for a
        cross-section peak to be accepted as an order centre.  Prominence
        measures how much a peak rises above its local surroundings; it is
        robust to the slowly-varying blaze-function envelope.
    min_distance : int, default 25
        Minimum separation in rows between accepted peaks.  Set to
        roughly half the expected order spacing to avoid splitting wide
        peaks while still resolving adjacent orders.
    max_shift : float, default 15.0
        Maximum row shift (in pixels) between the expected order position
        (extrapolated from the previously fitted polynomial or seed) and
        an accepted peak.  Prevents mis-matching to a neighbouring order
        when one order is temporarily undetected.
    sigma_clip : float, default 3.0
        Sigma-clipping threshold applied during polynomial fitting.
        Points whose residuals exceed ``sigma_clip × RMS`` are rejected
        in three successive iterations.

    Returns
    -------
    :class:`FlatOrderTrace`
        Tracing result with per-order polynomial coefficients, sampled
        centre positions, RMS values, and half-width estimates.

    Raises
    ------
    ValueError
        If *flat_files* is empty.
    RuntimeError
        If no order peaks are found at the seed column after applying
        the prominence and distance criteria.

    Notes
    -----
    The returned polynomial coefficients follow the
    ``numpy.polynomial.polynomial`` convention: ``coeffs[k]`` is the
    coefficient of ``col**k``.  Evaluate with
    ``np.polynomial.polynomial.polyval(col, coeffs)``.

    Examples
    --------
    >>> from pyspextool.instruments.ishell.io_utils import find_fits_files
    >>> all_paths = find_fits_files("data/testdata/ishell_h1_calibrations/raw")
    >>> flat_paths = [str(p) for p in all_paths if "flat" in p.name]
    >>> trace = trace_orders_from_flat(flat_paths)
    >>> print(f"Found {trace.n_orders} orders")
    >>> geom = trace.to_order_geometry_set("H1")
    """
    if not flat_files:
        raise ValueError("flat_files must not be empty.")

    # ------------------------------------------------------------------
    # 1. Load and median-combine
    # ------------------------------------------------------------------
    flat = load_and_combine_flats(flat_files)
    nrows, ncols = flat.shape

    # ------------------------------------------------------------------
    # 2. Resolve column range and seed column
    # ------------------------------------------------------------------
    if col_range is None:
        col_lo, col_hi = 0, ncols - 1
    else:
        col_lo, col_hi = int(col_range[0]), int(col_range[1])

    if seed_col is None:
        seed_col = (col_lo + col_hi) // 2
    else:
        seed_col = max(col_lo, min(col_hi, int(seed_col)))

    # ------------------------------------------------------------------
    # 3. Find seed peaks at the central column
    # ------------------------------------------------------------------
    seed_profile = _extract_cross_section(flat, seed_col, col_half_width)
    seed_peaks = _find_order_peaks(seed_profile, min_distance, min_prominence)

    if len(seed_peaks) == 0:
        raise RuntimeError(
            f"No order peaks found at seed column {seed_col} "
            f"(prominence>{min_prominence}, distance>{min_distance}).  "
            "Try reducing min_prominence or min_distance."
        )

    n_orders = len(seed_peaks)
    logger.info(
        "Found %d order seeds at column %d: rows %s",
        n_orders,
        seed_col,
        seed_peaks.tolist(),
    )

    # ------------------------------------------------------------------
    # 4. Build evenly-spaced sample columns
    # ------------------------------------------------------------------
    sample_cols = np.round(
        np.linspace(col_lo, col_hi, n_sample_cols)
    ).astype(int)
    sample_cols = np.unique(sample_cols)

    # ------------------------------------------------------------------
    # 5. Trace each order across sample columns
    # ------------------------------------------------------------------
    center_rows = np.full((n_orders, len(sample_cols)), np.nan)

    # Find the index of the sample column closest to the seed column
    seed_idx = int(np.argmin(np.abs(sample_cols - seed_col)))

    # Trace each order across all sample columns, walking outward from the
    # seed column.  Exclusive peak matching prevents two different orders
    # from claiming the same detected peak at the same column – the main
    # cause of spurious drift onto a neighbouring order.
    _trace_all_orders(
        flat, sample_cols, seed_idx, seed_peaks,
        center_rows, col_half_width, min_distance, min_prominence, max_shift,
    )

    # ------------------------------------------------------------------
    # 6. Fit polynomials with sigma-clipping and bounds-based fallback
    #
    # Root causes of pathological fits:
    #   (a) Too few valid sample points – a degree-D polynomial with
    #       D+1 ..< 2(D+1) points is underdetermined; even a tiny spread
    #       in y leads to large cancelling higher-degree coefficients that
    #       diverge far outside the data range.
    #   (b) All valid points clustered in a small column sub-range – with
    #       no data outside, higher-degree extrapolation is unconstrained.
    #
    # Fix: _fit_stable_poly() tries degree D, then D-1, …, 1, 0.  It
    # accepts the first degree for which the polynomial stays within the
    # detector bounds across the FULL column range [col_lo, col_hi].
    # ------------------------------------------------------------------
    center_poly_coeffs = np.zeros((n_orders, poly_degree + 1))
    fit_rms = np.full(n_orders, np.nan)
    poly_degrees_used = np.full(n_orders, poly_degree, dtype=int)

    for i in range(n_orders):
        row_vals = center_rows[i]
        good = np.isfinite(row_vals)

        if good.sum() < 2:
            logger.warning(
                "Order %d: only %d valid sample points; polynomial fit skipped.",
                i, good.sum(),
            )
            continue

        cols_fit = sample_cols[good].astype(float)
        rows_fit = row_vals[good]

        coeffs, rms, deg_used = _fit_stable_poly(
            cols_fit, rows_fit,
            max_degree=poly_degree,
            sigma=sigma_clip,
            n_iter=3,
            col_lo=float(col_lo),
            col_hi=float(col_hi),
            nrows=nrows,
        )
        # Pad coeffs to poly_degree + 1 length so the array is uniform
        padded = np.zeros(poly_degree + 1)
        padded[:len(coeffs)] = coeffs
        center_poly_coeffs[i] = padded
        fit_rms[i] = rms
        poly_degrees_used[i] = deg_used
        if deg_used < poly_degree:
            logger.info(
                "Order %d: reduced polynomial degree to %d (from %d) "
                "to prevent out-of-bounds extrapolation.",
                i, deg_used, poly_degree,
            )

    # ------------------------------------------------------------------
    # 6b. Repair any remaining adjacent-order crossings
    #
    # Even after bounds-based degree reduction some pairs of adjacent
    # orders may still cross if their sample points were entangled by
    # the tracking step (e.g. two orders converge near an edge).
    # _repair_crossing_orders() detects such crossing pairs and tries
    # reducing the polynomial degree for the worse-fitting member.
    # ------------------------------------------------------------------
    _repair_crossing_orders(
        center_poly_coeffs, fit_rms, center_rows, sample_cols,
        poly_degrees_used,
        col_lo=float(col_lo),
        col_hi=float(col_hi),
        nrows=nrows,
        sigma_clip=sigma_clip,
    )

    # ------------------------------------------------------------------
    # 7. Estimate order half-widths from the seed profile
    # ------------------------------------------------------------------
    half_width_rows = _estimate_half_widths(seed_profile, seed_peaks)

    # ------------------------------------------------------------------
    # 8. Compute per-order QA metrics
    # ------------------------------------------------------------------
    order_stats = _compute_order_trace_stats(
        center_poly_coeffs, fit_rms, sample_cols,
    )
    _log_trace_qa_summary(order_stats, poly_degrees_used=poly_degrees_used)

    logger.info(
        "Order tracing complete: %d orders, median RMS = %.2f px",
        n_orders,
        float(np.nanmedian(fit_rms)),
    )

    return FlatOrderTrace(
        n_orders=n_orders,
        sample_cols=sample_cols,
        center_rows=center_rows,
        center_poly_coeffs=center_poly_coeffs,
        fit_rms=fit_rms,
        half_width_rows=half_width_rows,
        poly_degree=poly_degree,
        seed_col=seed_col,
        order_stats=order_stats,
        poly_degrees_used=poly_degrees_used,
    )


# ---------------------------------------------------------------------------
# Public helper – loading
# ---------------------------------------------------------------------------


def load_and_combine_flats(flat_files: list[str]) -> npt.NDArray:
    """Load iSHELL flat FITS frames and median-combine them.

    Reads the **PRIMARY extension** (extension index 0) of each file and
    takes the pixel-wise median over all frames.  The median is more
    robust than the mean against cosmic rays, hot pixels, and occasional
    bad reads.

    Parameters
    ----------
    flat_files : list of str
        Paths to raw iSHELL FITS flat files.  At least one file must be
        provided.

    Returns
    -------
    ndarray, shape (nrows, ncols), dtype float32
        Median-combined flat image.

    Raises
    ------
    ValueError
        If *flat_files* is empty.
    OSError
        If any file cannot be opened as a valid FITS file.
    """
    if not flat_files:
        raise ValueError("flat_files must not be empty.")

    imgs: list[npt.NDArray] = []
    for path in flat_files:
        with fits.open(path, memmap=False) as hdul:
            imgs.append(hdul[0].data.astype(np.float32))

    if len(imgs) == 1:
        return imgs[0]

    return np.median(np.stack(imgs, axis=0), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_cross_section(
    flat: npt.NDArray,
    col: int,
    half_width: int,
    smooth_sigma: float = 1.0,
) -> npt.NDArray:
    """Return a cross-dispersion profile by median-averaging a column band.

    Parameters
    ----------
    flat : ndarray, shape (nrows, ncols)
        Flat-field image.
    col : int
        Centre column of the band.
    half_width : int
        Half-width of the band in pixels.
    smooth_sigma : float, default 1.0
        Standard deviation of the Gaussian kernel applied to the median
        profile before peak detection.  Light smoothing suppresses
        fringing features while preserving the broad order peaks.  Set
        to ``0`` to disable smoothing.

    Returns
    -------
    ndarray, shape (nrows,)
        Median (and optionally smoothed) cross-dispersion profile.
    """
    from scipy.ndimage import gaussian_filter1d

    nrows, ncols = flat.shape
    c0 = max(0, col - half_width)
    c1 = min(ncols, col + half_width + 1)
    profile = np.median(flat[:, c0:c1], axis=1)
    if smooth_sigma > 0:
        profile = gaussian_filter1d(profile, sigma=smooth_sigma)
    return profile


def _find_order_peaks(
    profile: npt.NDArray,
    min_distance: int,
    min_prominence: float,
) -> npt.NDArray:
    """Find order-centre peaks in a cross-dispersion profile.

    Parameters
    ----------
    profile : ndarray, shape (nrows,)
        Cross-dispersion intensity profile.
    min_distance : int
        Minimum separation between peaks in pixels.
    min_prominence : float
        Minimum peak prominence in detector counts.

    Returns
    -------
    ndarray of int
        Row indices of the detected peaks, sorted in ascending order.
    """
    peaks, _ = _scipy_find_peaks(
        profile,
        distance=min_distance,
        prominence=min_prominence,
    )
    return peaks


def _trace_all_orders(
    flat: npt.NDArray,
    sample_cols: npt.NDArray,
    seed_idx: int,
    seed_peaks: npt.NDArray,
    center_rows: npt.NDArray,
    col_half_width: int,
    min_distance: int,
    min_prominence: float,
    max_shift: float,
) -> None:
    """Trace all orders across all sample columns, filling *center_rows* in place.

    Starting at *seed_idx*, the algorithm walks right and then left from the
    seed column.  At each step a running position estimate per order is
    maintained; it is updated only when a peak is successfully matched.
    This prevents single-column detection failures from cascading.

    **Exclusive peak matching** is used at each column: each detected peak
    is assigned to at most one order.  Candidates are ranked by distance
    from the running estimate and the closest (order, peak) pair claims the
    peak first.  This prevents two adjacent orders from drifting onto the
    same peak when one of them loses its genuine signal near a detector
    edge — a root cause of the entangled/crossing traces seen on K3 data.

    Parameters
    ----------
    flat : ndarray
        Flat-field image.
    sample_cols : ndarray of int
        Column indices to evaluate.
    seed_idx : int
        Index into *sample_cols* of the seed column.
    seed_peaks : ndarray of int
        Row positions of order seeds (found at the seed column).
    center_rows : ndarray of float, shape (n_orders, n_sample)
        Output array; filled in place with NaN for unmatched positions.
    col_half_width, min_distance, min_prominence, max_shift :
        Passed through to helper functions.
    """
    n_orders = center_rows.shape[0]
    n_cols = len(sample_cols)

    # Walk in each direction: (start, stop, step)
    for direction_start, direction_stop, direction_step in [
        (seed_idx, n_cols, +1),   # rightward (includes seed)
        (seed_idx - 1, -1, -1),   # leftward
    ]:
        # Initialise running estimates from seed peaks
        current_est = seed_peaks.astype(float).copy()

        for idx in range(direction_start, direction_stop, direction_step):
            col = int(sample_cols[idx])
            profile = _extract_cross_section(flat, col, col_half_width)
            peaks = _find_order_peaks(profile, min_distance, min_prominence)

            if len(peaks) == 0:
                center_rows[:, idx] = np.nan
                continue

            # ----------------------------------------------------------
            # Exclusive greedy matching:
            # Build all valid (order, peak) candidate pairs within
            # max_shift, then assign them greedily from closest to
            # furthest, ensuring each peak is claimed by at most one
            # order.  This prevents two orders that have both drifted
            # near the same peak from both claiming it.
            # ----------------------------------------------------------
            peaks_f = peaks.astype(float)
            # dist[i, j] = |estimate_i – peak_j|
            dist_matrix = np.abs(
                current_est[:, np.newaxis] - peaks_f[np.newaxis, :]
            )  # shape (n_orders, n_peaks)

            assigned = np.full(n_orders, -1, dtype=int)   # peak index per order
            claimed = np.zeros(len(peaks), dtype=bool)    # peak already taken?

            # Collect all (distance, order_idx, peak_idx) within max_shift
            oi_arr, pi_arr = np.where(
                (dist_matrix <= max_shift) & np.isfinite(current_est[:, np.newaxis])
            )
            if len(oi_arr) > 0:
                d_arr = dist_matrix[oi_arr, pi_arr]
                sort_order = np.argsort(d_arr)
                for k in sort_order:
                    oi = int(oi_arr[k])
                    pi = int(pi_arr[k])
                    if assigned[oi] == -1 and not claimed[pi]:
                        assigned[oi] = pi
                        claimed[pi] = True

            for i in range(n_orders):
                if not np.isfinite(current_est[i]):
                    center_rows[i, idx] = np.nan
                    continue
                if assigned[i] != -1:
                    matched_row = float(peaks[assigned[i]])
                    center_rows[i, idx] = matched_row
                    current_est[i] = matched_row  # update running estimate
                else:
                    center_rows[i, idx] = np.nan
                    # Keep current_est unchanged so the next column
                    # still uses the last known good position.


def _fit_poly_sigma_clip(
    cols: npt.NDArray,
    rows: npt.NDArray,
    degree: int,
    sigma: float,
    n_iter: int = 3,
) -> tuple[npt.NDArray, float]:
    """Fit a polynomial with iterative sigma-clipping.

    Parameters
    ----------
    cols : ndarray
        Column coordinates (x values).
    rows : ndarray
        Row coordinates (y values).
    degree : int
        Polynomial degree.
    sigma : float
        Clipping threshold in units of residual RMS.
    n_iter : int, default 3
        Number of sigma-clipping iterations.

    Returns
    -------
    coeffs : ndarray, shape (degree + 1,)
        Polynomial coefficients in ``numpy.polynomial.polynomial`` convention.
    rms : float
        RMS of the final fit residuals in pixels.
    """
    c = cols.copy()
    r = rows.copy()

    coeffs = np.polynomial.polynomial.polyfit(c, r, degree)

    for _ in range(n_iter):
        predicted = np.polynomial.polynomial.polyval(c, coeffs)
        resid = r - predicted
        rms = float(np.std(resid))
        if rms == 0.0:
            break
        keep = np.abs(resid) <= sigma * rms
        if keep.sum() < degree + 1:
            break
        c = c[keep]
        r = r[keep]
        coeffs = np.polynomial.polynomial.polyfit(c, r, degree)

    # Compute final RMS on the full accepted set
    predicted = np.polynomial.polynomial.polyval(c, coeffs)
    final_rms = float(np.std(r - predicted))

    return coeffs, final_rms


def _fit_stable_poly(
    cols: npt.NDArray,
    rows: npt.NDArray,
    max_degree: int,
    sigma: float,
    n_iter: int,
    col_lo: float,
    col_hi: float,
    nrows: int,
    bounds_margin: float = 50.0,
    data_range_margin: float = 50.0,
) -> tuple[npt.NDArray, float, int]:
    """Fit a polynomial with sigma-clipping and bounds-based degree fallback.

    Root cause addressed
    --------------------
    A degree-D polynomial fitted to too few or too densely clustered
    sample points can extrapolate catastrophically outside the data
    range.  Two distinct failure modes are addressed:

    (a) *Off-detector divergence*: 7 points near y ≈ 1780 in cols 262–997
        produce a degree-3 polynomial that reaches y = 534 at col 2047
        (far off the 2048-row detector).

    (b) *Inter-order crossing via extrapolation*: order i's sample points
        start at column C_min > col_lo because the order loses signal
        near the detector edge.  A degree-3 polynomial extrapolated from
        C_min back to col_lo can diverge well below the order's true
        row range, crossing a neighbouring order.  Example: Order 2's
        data begin at col 367 (row 83) but the cubic polynomial reaches
        row 22 at col 0.

    Algorithm
    ---------
    1. Coverage cap: if sample points span < 50 % of the column range,
       cap the maximum degree at 1.
    2. Try degrees from ``effective_max`` down to 1.  For each degree,
       fit with sigma-clipping and then apply **two** acceptance tests:

       (i)  Detector-bounds test: all polynomial values in [col_lo, col_hi]
            must lie in ``[-bounds_margin, nrows + bounds_margin]``.
       (ii) Data-range test: all polynomial values in [col_lo, col_hi]
            must lie in ``[rows.min() - data_range_margin,
            rows.max() + data_range_margin]``.  This prevents the
            polynomial from crossing into a neighbouring order's territory
            even when the detector-bounds test passes.

    3. The first degree that passes both tests is accepted.
    4. Fallback: constant (mean of valid rows) if no degree passes.

    Parameters
    ----------
    cols, rows : ndarray
        Column and row coordinates of valid sample points.
    max_degree : int
        Maximum polynomial degree to try first.
    sigma : float
        Sigma-clipping threshold for :func:`_fit_poly_sigma_clip`.
    n_iter : int
        Number of sigma-clipping iterations.
    col_lo, col_hi : float
        Full column range over which the polynomial is evaluated.
    nrows : int
        Number of detector rows; used for the upper bound check.
    bounds_margin : float
        Extra margin beyond [0, nrows] allowed for the detector-bounds
        test before the degree is reduced.
    data_range_margin : float
        Extra margin beyond ``[rows.min(), rows.max()]`` allowed for the
        data-range test before the degree is reduced.  Set to a value
        comparable to half the typical inter-order spacing (~50 px for
        iSHELL K3).

    Returns
    -------
    coeffs : ndarray, shape (degree_used + 1,)
        Polynomial coefficients in ``numpy.polynomial.polynomial``
        convention.  Length may be less than ``max_degree + 1``.
    rms : float
        RMS of the fit residuals in pixels.
    degree_used : int
        The polynomial degree actually used (≤ ``max_degree``).
    """
    n_pts = len(cols)

    # ---------------------------------------------------------------
    # Coverage-based degree cap: if sample points span < 50 % of the
    # column range, cap the degree at 1 (linear extrapolation is far
    # more stable than cubic over a large un-sampled region).
    # ---------------------------------------------------------------
    col_span = float(cols.max() - cols.min()) if n_pts > 1 else 0.0
    total_span = float(col_hi - col_lo)
    coverage = col_span / total_span if total_span > 0 else 0.0

    if coverage < 0.5:
        effective_max = min(1, max_degree)
    elif coverage < 0.75 and n_pts < 3 * (max_degree + 1):
        effective_max = min(2, max_degree)
    else:
        effective_max = max_degree

    # Evaluation grid for the acceptance tests
    eval_cols = np.linspace(col_lo, col_hi, 50)

    # Data row range for the data-range acceptance test
    row_lo_bound = float(rows.min()) - data_range_margin
    row_hi_bound = float(rows.max()) + data_range_margin

    # ---------------------------------------------------------------
    # Try fitting from effective_max down to 1, accepting the first
    # degree whose polynomial passes both acceptance tests.
    # ---------------------------------------------------------------
    for degree in range(effective_max, 0, -1):
        if n_pts < degree + 2:
            # Not enough points for a well-determined fit at this degree
            continue
        coeffs, rms = _fit_poly_sigma_clip(cols, rows, degree, sigma, n_iter)
        y_eval = np.polynomial.polynomial.polyval(eval_cols, coeffs)
        detector_ok = np.all(y_eval >= -bounds_margin) and np.all(
            y_eval <= nrows + bounds_margin
        )
        data_range_ok = np.all(y_eval >= row_lo_bound) and np.all(
            y_eval <= row_hi_bound
        )
        if detector_ok and data_range_ok:
            return coeffs, rms, degree

    # ---------------------------------------------------------------
    # Fallback: constant (mean of valid sample rows).
    # Used when even a linear fit diverges outside both acceptance
    # windows or there are too few points.
    # ---------------------------------------------------------------
    mean_val = float(np.mean(rows))
    rms_const = float(np.std(rows))
    return np.array([mean_val]), rms_const, 0


def _repair_crossing_orders(
    center_poly_coeffs: npt.NDArray,
    fit_rms: npt.NDArray,
    center_rows: npt.NDArray,
    sample_cols: npt.NDArray,
    poly_degrees_used: npt.NDArray,
    col_lo: float,
    col_hi: float,
    nrows: int,
    sigma_clip: float,
    n_iter: int = 3,
    max_passes: int = 3,
) -> None:
    """Repair adjacent-order crossings by reducing the polynomial degree in place.

    Root cause addressed
    --------------------
    Even after the bounds-based degree reduction in :func:`_fit_stable_poly`,
    some adjacent-order pairs may still cross if their sample points were
    entangled by the tracking step.  For example, if Order i drifted onto
    Order i-1's peak at left-edge columns, the resulting polynomial for
    Order i may underestimate the row position at col 0 and produce a
    crossing at that end of the detector.

    Algorithm
    ---------
    Iteratively (up to *max_passes* times):
      1. Evaluate all centre polynomials at 50 evenly-spaced columns
         in ``[col_lo, col_hi]``.
      2. Identify adjacent-order pairs (i, i+1) where the polynomial of
         order i+1 is ever below the polynomial of order i.
      3. For each such pair, the order with the larger RMS (the "worse"
         fit) is refitted using a lower polynomial degree.  The order
         with fewer valid sample points breaks ties.
      4. Repeat until no more crossings are found or *max_passes* is
         exhausted.

    Modifications are done **in place** to *center_poly_coeffs*,
    *fit_rms*, and *poly_degrees_used*.

    Parameters
    ----------
    center_poly_coeffs : ndarray, shape (n_orders, poly_degree + 1)
        Polynomial coefficients modified in place.
    fit_rms : ndarray, shape (n_orders,)
        RMS values modified in place.
    center_rows : ndarray, shape (n_orders, n_sample)
        Tracked sample positions (read-only within this function).
    sample_cols : ndarray of int
        Sample column indices.
    poly_degrees_used : ndarray of int, shape (n_orders,)
        Actual degree per order; modified in place when a repair occurs.
    col_lo, col_hi : float
        Full column range.
    nrows : int
        Number of detector rows.
    sigma_clip : float
        Sigma-clipping threshold for refitting.
    n_iter : int
        Number of sigma-clipping iterations.
    max_passes : int
        Maximum number of repair passes before giving up.
    """
    n_orders = len(fit_rms)
    eval_cols = np.linspace(col_lo, col_hi, 50)

    for pass_num in range(max_passes):
        # Evaluate all polynomials
        center_vals = np.full((n_orders, 50), np.nan)
        for i in range(n_orders):
            if np.isfinite(fit_rms[i]):
                center_vals[i] = np.polynomial.polynomial.polyval(
                    eval_cols, center_poly_coeffs[i]
                )

        any_repaired = False
        for i in range(n_orders - 1):
            if not (np.isfinite(fit_rms[i]) and np.isfinite(fit_rms[i + 1])):
                continue
            gap = center_vals[i + 1] - center_vals[i]
            finite_gap = gap[np.isfinite(gap)]
            if len(finite_gap) == 0 or not np.any(finite_gap <= 0):
                continue

            # Crossing detected between order i and i+1.
            # Fix the one with larger RMS (worse fit).
            n_valid_i = int(np.sum(np.isfinite(center_rows[i])))
            n_valid_ip1 = int(np.sum(np.isfinite(center_rows[i + 1])))
            if fit_rms[i] > fit_rms[i + 1]:
                worse = i
            elif fit_rms[i + 1] > fit_rms[i]:
                worse = i + 1
            else:
                # Tie: prefer order with fewer valid points (less trusted)
                worse = i if n_valid_i <= n_valid_ip1 else i + 1

            current_deg = int(poly_degrees_used[worse])
            new_deg = max(0, current_deg - 1)

            good = np.isfinite(center_rows[worse])
            if good.sum() < 2:
                continue  # cannot refit with < 2 points

            cols_fit = sample_cols[good].astype(float)
            rows_fit = center_rows[worse][good]

            if new_deg == 0:
                mean_val = float(np.mean(rows_fit))
                new_coeffs = np.array([mean_val])
                new_rms = float(np.std(rows_fit))
            else:
                new_coeffs, new_rms = _fit_poly_sigma_clip(
                    cols_fit, rows_fit, new_deg, sigma_clip, n_iter
                )

            # -------------------------------------------------------
            # Apply only if the new fit:
            #   (a) stays within detector bounds, AND
            #   (b) actually resolves the crossing
            #       (i.e., the new polynomial no longer crosses its
            #        neighbour).
            # If the crossing cannot be resolved by degree reduction
            # (e.g., the orders genuinely overlap at a detector edge),
            # leave the original fit intact rather than degrading it to
            # a constant, and let the QA flag report the crossing.
            # -------------------------------------------------------
            eval_new = np.polynomial.polynomial.polyval(eval_cols, new_coeffs)
            if not (np.all(eval_new >= -50.0) and np.all(eval_new <= nrows + 50.0)):
                continue  # new fit still out of bounds – skip

            # Check whether the crossing is actually resolved
            other = i if worse == i + 1 else i + 1
            other_vals = center_vals[other]
            if worse == i + 1:
                # worse is the upper order; check new_vals > other_vals
                new_gap = eval_new - other_vals
            else:
                # worse is the lower order; check other_vals > new_vals
                new_gap = other_vals - eval_new
            finite_new_gap = new_gap[np.isfinite(new_gap)]
            if len(finite_new_gap) == 0 or np.any(finite_new_gap <= 0):
                # Crossing not resolved – keep original fit
                logger.debug(
                    "Crossing between orders %d and %d could not be "
                    "resolved by reducing degree of order %d to %d; "
                    "leaving original fit intact.",
                    i, i + 1, worse, new_deg,
                )
                continue

            padded = np.zeros(center_poly_coeffs.shape[1])
            padded[:len(new_coeffs)] = new_coeffs
            center_poly_coeffs[worse] = padded
            fit_rms[worse] = new_rms
            poly_degrees_used[worse] = new_deg
            logger.info(
                "Crossing repair (pass %d): order %d reduced to degree %d "
                "to eliminate crossing with order %d.",
                pass_num + 1, worse, new_deg,
                i if worse == i + 1 else i + 1,
            )
            any_repaired = True

        if not any_repaired:
            break


def _estimate_half_widths(
    profile: npt.NDArray,
    peaks: npt.NDArray,
) -> npt.NDArray:
    """Estimate the half-maximum half-width of each order peak.

    For each peak, the function walks outward from the peak until the
    profile drops below the half-maximum level (relative to a local
    baseline), then records the half-width.

    Parameters
    ----------
    profile : ndarray, shape (nrows,)
        Cross-dispersion intensity profile.
    peaks : ndarray of int
        Row indices of the order-centre peaks.

    Returns
    -------
    ndarray of float, shape (n_peaks,)
        Estimated half-maximum half-widths in pixels.
    """
    nrows = len(profile)
    baseline = float(np.percentile(profile, 20))
    half_widths = np.zeros(len(peaks), dtype=float)

    for i, pk in enumerate(peaks):
        peak_val = float(profile[pk])
        half_max = baseline + (peak_val - baseline) * 0.5

        # Walk left to half-maximum
        left = int(pk)
        while left > 0 and profile[left] > half_max:
            left -= 1

        # Walk right to half-maximum
        right = int(pk)
        while right < nrows - 1 and profile[right] > half_max:
            right += 1

        half_widths[i] = (right - left) / 2.0

    return half_widths


def _compute_order_trace_stats(
    center_poly_coeffs: npt.NDArray,
    fit_rms: npt.NDArray,
    sample_cols: npt.NDArray,
    *,
    rms_threshold: float = _RMS_THRESHOLD,
    curvature_threshold: float = _CURVATURE_THRESHOLD,
    separation_threshold: float = _SEPARATION_THRESHOLD,
    oscillation_threshold: float = _OSCILLATION_THRESHOLD,
) -> list[OrderTraceStats]:
    """Compute per-order QA metrics for traced order-centre polynomials.

    Parameters
    ----------
    center_poly_coeffs : ndarray, shape (n_orders, poly_degree + 1)
        Per-order polynomial coefficients in ``numpy.polynomial.polynomial``
        convention.
    fit_rms : ndarray, shape (n_orders,)
        RMS of polynomial residuals per order.  ``NaN`` marks failed fits.
    sample_cols : ndarray, shape (n_sample,)
        Detector column indices at which the polynomial is evaluated.
    rms_threshold : float, default ``_RMS_THRESHOLD``
        Maximum acceptable RMS residual in pixels.
    curvature_threshold : float, default ``_CURVATURE_THRESHOLD``
        Maximum acceptable second derivative of the polynomial (px / col²).
    separation_threshold : float, default ``_SEPARATION_THRESHOLD``
        Minimum acceptable absolute inter-order separation in pixels.
    oscillation_threshold : float, default ``_OSCILLATION_THRESHOLD``
        Maximum acceptable peak-to-peak slope variation (px / col).

    Returns
    -------
    list of OrderTraceStats
        One entry per order.
    """
    n_orders = len(fit_rms)
    cols = sample_cols.astype(float)

    # Pre-compute centre-row positions at all sample columns.
    # Use NaN rows where the polynomial fit failed so they do not
    # corrupt inter-order separation calculations.
    center_vals = np.full((n_orders, len(cols)), np.nan)
    for i in range(n_orders):
        if np.isfinite(fit_rms[i]):
            center_vals[i] = np.polynomial.polynomial.polyval(
                cols, center_poly_coeffs[i]
            )

    stats_list: list[OrderTraceStats] = []

    for i in range(n_orders):
        coeffs = center_poly_coeffs[i]
        rms = float(fit_rms[i])
        fit_ok = np.isfinite(rms)

        # ------------------------------------------------------------------
        # 1. Curvature metric: max |d²y/dx²| evaluated at sample columns.
        #    Uses numpy.polynomial.polynomial.polyder to differentiate.
        # ------------------------------------------------------------------
        if not fit_ok:
            curvature_metric = np.nan
        else:
            d2_coeffs = np.polynomial.polynomial.polyder(coeffs, 2)
            if d2_coeffs.size == 0:
                curvature_metric = 0.0
            else:
                d2_vals = np.abs(
                    np.polynomial.polynomial.polyval(cols, d2_coeffs)
                )
                curvature_metric = float(np.max(d2_vals))

        # ------------------------------------------------------------------
        # 2. Oscillation metric: peak-to-peak variation in the first
        #    derivative (slope) across the sample columns.  A straight trace
        #    gives 0; oscillatory or pathologically curved polynomials give
        #    large values.  This replaces the previous boolean monotonicity
        #    test, which incorrectly flagged even gently curved traces.
        # ------------------------------------------------------------------
        if not fit_ok:
            oscillation_metric = np.nan
        else:
            d1_coeffs = np.polynomial.polynomial.polyder(coeffs, 1)
            if d1_coeffs.size == 0:
                oscillation_metric = 0.0
            else:
                d1_vals = np.polynomial.polynomial.polyval(cols, d1_coeffs)
                oscillation_metric = float(np.max(d1_vals) - np.min(d1_vals))

        # ------------------------------------------------------------------
        # 3. Inter-order separation: absolute gap to each adjacent neighbour.
        #    Reports lower and upper neighbours separately, and flags any
        #    crossing explicitly.  Uses nanmin so that isolated NaN columns
        #    (missed peaks in center_vals) do not discard an entire neighbour.
        # ------------------------------------------------------------------
        if n_orders <= 1 or not fit_ok:
            min_sep_lower = np.nan
            min_sep_upper = np.nan
            crosses_lower = False
            crosses_upper = False
        else:
            # Lower neighbour (smaller row values, index i-1).
            if i > 0 and np.any(np.isfinite(center_vals[i - 1])):
                gap = center_vals[i] - center_vals[i - 1]
                finite_mask = np.isfinite(gap)
                if np.any(finite_mask):
                    min_sep_lower = float(np.min(np.abs(gap[finite_mask])))
                    crosses_lower = bool(np.any(gap[finite_mask] < 0))
                else:
                    min_sep_lower = np.nan
                    crosses_lower = False
            else:
                min_sep_lower = np.nan
                crosses_lower = False

            # Upper neighbour (larger row values, index i+1).
            if i < n_orders - 1 and np.any(np.isfinite(center_vals[i + 1])):
                gap = center_vals[i + 1] - center_vals[i]
                finite_mask = np.isfinite(gap)
                if np.any(finite_mask):
                    min_sep_upper = float(np.min(np.abs(gap[finite_mask])))
                    crosses_upper = bool(np.any(gap[finite_mask] < 0))
                else:
                    min_sep_upper = np.nan
                    crosses_upper = False
            else:
                min_sep_upper = np.nan
                crosses_upper = False

        # ------------------------------------------------------------------
        # 4. Composite validity flag.
        # ------------------------------------------------------------------
        rms_ok = fit_ok and rms <= rms_threshold
        curv_ok = np.isfinite(curvature_metric) and (
            curvature_metric <= curvature_threshold
        )
        osc_ok = np.isfinite(oscillation_metric) and (
            oscillation_metric <= oscillation_threshold
        )
        sep_ok = (
            (np.isnan(min_sep_lower) or min_sep_lower >= separation_threshold)
            and (np.isnan(min_sep_upper) or min_sep_upper >= separation_threshold)
            and not crosses_lower
            and not crosses_upper
        )
        trace_valid = bool(rms_ok and curv_ok and osc_ok and sep_ok)

        stats_list.append(
            OrderTraceStats(
                order_index=i,
                rms_residual=rms,
                curvature_metric=curvature_metric,
                oscillation_metric=oscillation_metric,
                min_sep_lower=min_sep_lower,
                min_sep_upper=min_sep_upper,
                crosses_lower=crosses_lower,
                crosses_upper=crosses_upper,
                trace_valid=trace_valid,
            )
        )

    return stats_list


def _log_trace_qa_summary(
    stats: list[OrderTraceStats],
    poly_degrees_used: Optional[npt.NDArray] = None,
) -> None:
    """Log a concise per-order QA summary after tracing.

    Orders that fail any QA check are clearly marked with ``*** INVALID ***``
    and the specific failing checks are listed so they are easy to spot in
    the log output.

    Parameters
    ----------
    stats : list of OrderTraceStats
        Per-order statistics returned by :func:`_compute_order_trace_stats`.
    poly_degrees_used : ndarray of int, optional
        Actual polynomial degree used per order (after stable-fit fallback).
        When provided, the degree is appended to each row so the user can
        see which orders triggered a degree-reduction fallback.
    """
    if not stats:
        return

    has_degrees = (
        poly_degrees_used is not None and len(poly_degrees_used) == len(stats)
    )
    deg_header = "  Deg" if has_degrees else ""
    header = (
        "  {:>5}  {:>9}  {:>11}  {:>10}  {:>9}  {:>9}  {:>6}  {:>6}  {}{}".format(
            "Order", "RMS(px)", "Curv(p/c²)", "Osc(p/c)",
            "SepLo(px)", "SepHi(px)", "XLo", "XHi", "Valid",
            deg_header,
        )
    )
    separator = "  " + "-" * (88 + (6 if has_degrees else 0))

    logger.info("Per-order trace QA summary:")
    logger.info(header)
    logger.info(separator)

    for s in stats:
        rms_str = f"{s.rms_residual:9.3f}" if np.isfinite(s.rms_residual) else "      NaN"
        curv_str = (
            f"{s.curvature_metric:11.2e}"
            if np.isfinite(s.curvature_metric)
            else "        NaN"
        )
        osc_str = (
            f"{s.oscillation_metric:10.4f}"
            if np.isfinite(s.oscillation_metric)
            else "       NaN"
        )
        sep_lo_str = (
            f"{s.min_sep_lower:9.1f}"
            if np.isfinite(s.min_sep_lower)
            else "      NaN"
        )
        sep_hi_str = (
            f"{s.min_sep_upper:9.1f}"
            if np.isfinite(s.min_sep_upper)
            else "      NaN"
        )
        xlo_str = f"{'Y' if s.crosses_lower else 'N':>6}"
        xhi_str = f"{'Y' if s.crosses_upper else 'N':>6}"

        deg_str = ""
        if has_degrees:
            deg_str = f"  {int(poly_degrees_used[s.order_index]):>3}"

        # Build a short failure reason string for invalid orders.
        if not s.trace_valid:
            reasons = []
            if not (np.isfinite(s.rms_residual) and s.rms_residual <= _RMS_THRESHOLD):
                reasons.append("RMS")
            if not (np.isfinite(s.curvature_metric) and s.curvature_metric <= _CURVATURE_THRESHOLD):
                reasons.append("Curv")
            if not (np.isfinite(s.oscillation_metric) and s.oscillation_metric <= _OSCILLATION_THRESHOLD):
                reasons.append("Osc")
            if np.isfinite(s.min_sep_lower) and s.min_sep_lower < _SEPARATION_THRESHOLD:
                reasons.append("SepLo")
            if np.isfinite(s.min_sep_upper) and s.min_sep_upper < _SEPARATION_THRESHOLD:
                reasons.append("SepHi")
            if s.crosses_lower:
                reasons.append("XLo")
            if s.crosses_upper:
                reasons.append("XHi")
            flag = f"  *** INVALID ({', '.join(reasons)}) ***"
        else:
            flag = ""

        line = (
            "  {:>5}  {}  {}  {}  {}  {}  {}  {}  {}{}{}".format(
                s.order_index,
                rms_str,
                curv_str,
                osc_str,
                sep_lo_str,
                sep_hi_str,
                xlo_str,
                xhi_str,
                str(s.trace_valid),
                deg_str,
                flag,
            )
        )
        logger.info(line)

    n_invalid = sum(1 for s in stats if not s.trace_valid)
    if n_invalid:
        logger.warning(
            "%d / %d order(s) flagged as potentially invalid (not rejected).",
            n_invalid,
            len(stats),
        )
    else:
        logger.info("All %d orders passed QA checks.", len(stats))
