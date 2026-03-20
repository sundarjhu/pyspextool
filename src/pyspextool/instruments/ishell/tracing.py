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

_RMS_THRESHOLD: float = 5.0          # px -- flag if rms_residual exceeds this
_CURVATURE_THRESHOLD: float = 1e-3   # px/col^2 -- flag if max |d2y/dx2| exceeds this
_SEPARATION_THRESHOLD: float = 3.0   # px -- flag if min absolute inter-order gap drops below this
_OSCILLATION_THRESHOLD: float = 0.05 # px/col -- flag if peak-to-peak slope variation exceeds this


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
        Degree of the fitted centre-line polynomials.
    seed_col : int
        Detector column used to locate initial order-centre seeds.
    order_stats : list of OrderTraceStats
        Per-order QA metrics computed immediately after polynomial fitting.
        See :class:`OrderTraceStats` for the available fields.  Empty if
        the result was constructed directly without calling
        :func:`trace_orders_from_flat`.

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

    # Trace each order independently across all sample columns,
    # walking outward from the seed column in both directions so that
    # the running position estimate is always based on the most recent
    # good match.
    _trace_all_orders(
        flat, sample_cols, seed_idx, seed_peaks,
        center_rows, col_half_width, min_distance, min_prominence, max_shift,
    )

    # ------------------------------------------------------------------
    # 6. Fit polynomials with sigma-clipping
    # ------------------------------------------------------------------
    center_poly_coeffs = np.zeros((n_orders, poly_degree + 1))
    fit_rms = np.full(n_orders, np.nan)

    for i in range(n_orders):
        row_vals = center_rows[i]
        good = np.isfinite(row_vals)

        if good.sum() < poly_degree + 1:
            logger.warning(
                "Order %d: only %d valid sample points; polynomial fit skipped.",
                i, good.sum(),
            )
            continue

        cols_fit = sample_cols[good].astype(float)
        rows_fit = row_vals[good]

        coeffs, rms = _fit_poly_sigma_clip(
            cols_fit, rows_fit, poly_degree, sigma_clip, n_iter=3,
        )
        center_poly_coeffs[i] = coeffs
        fit_rms[i] = rms

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
    _log_trace_qa_summary(order_stats)

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

            for i in range(n_orders):
                est = current_est[i]
                if not np.isfinite(est):
                    center_rows[i, idx] = np.nan
                    continue

                if len(peaks) == 0:
                    center_rows[i, idx] = np.nan
                    continue

                dists = np.abs(peaks.astype(float) - est)
                j = int(np.argmin(dists))
                if dists[j] <= max_shift:
                    matched_row = float(peaks[j])
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


def _log_trace_qa_summary(stats: list[OrderTraceStats]) -> None:
    """Log a concise per-order QA summary after tracing.

    Orders that fail any QA check are clearly marked with ``*** INVALID ***``
    and the specific failing checks are listed so they are easy to spot in
    the log output.

    Parameters
    ----------
    stats : list of OrderTraceStats
        Per-order statistics returned by :func:`_compute_order_trace_stats`.
    """
    if not stats:
        return

    header = (
        "  {:>5}  {:>9}  {:>11}  {:>10}  {:>9}  {:>9}  {:>6}  {:>6}  {}"
    ).format(
        "Order", "RMS(px)", "Curv(p/c²)", "Osc(p/c)",
        "SepLo(px)", "SepHi(px)", "XLo", "XHi", "Valid"
    )
    separator = "  " + "-" * 88

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
            "  {:>5}  {}  {}  {}  {}  {}  {}  {}  {}{}".format(
                s.order_index,
                rms_str,
                curv_str,
                osc_str,
                sep_lo_str,
                sep_hi_str,
                xlo_str,
                xhi_str,
                str(s.trace_valid),
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
