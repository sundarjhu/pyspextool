"""
2-D arc-line tracing from iSHELL ThAr arc frames.

This module implements the second stage of the iSHELL 2DXD reduction
scaffold: tracing ThAr arc emission lines in two dimensions from one or
more median-combined arc frames, constrained by the order geometry
established by the flat-field tracing stage.

The result is an :class:`ArcLineTrace` object containing:

* the column and row position of each identified arc line at each
  sampled detector row within each order,
* low-order polynomial fits  ``col(row)``  for each traced line, and
* the order index and approximate detector-column seed used to find
  each line.

These per-line polynomials are the natural input for the later
*coefficient-surface fitting* step that builds a smooth
``col(order, row)`` surface across the entire detector.

.. note::
   This is a **first-pass arc-line tracing scaffold**, not a finalised
   wavelength calibration.  Specifically:

   * No ThAr wavelength identification is performed here — that
     requires matching the measured line positions against a ThAr atlas,
     which is a later step.
   * No wavelength or spatial rectification index is computed.
   * No 2DXD coefficient surface is fitted.

   The outputs are suitable for smoke-testing and data exploration but
   should not be used for science-quality wavelength calibration without
   further validation.

Algorithm
---------
1. **Load arc frames** – open each arc FITS file (PRIMARY extension,
   extension 0) and median-combine to suppress read noise and hot pixels.

2. **Define order regions** – use the :class:`FlatOrderTrace` (from the
   flat-tracing stage) to determine, for each order, the centre-row
   trajectory and the half-width of the order band.  The search strip for
   each order spans ``centre ± strip_half_width`` rows at each column.

3. **Identify line seeds** – at the seed column, extract a spectral
   profile (the row-collapsed sum within the order strip) and find peaks
   with ``scipy.signal.find_peaks``.  These are candidate arc emission
   lines.

4. **Row-by-row tracing** – for each identified seed line (at seed column
   ``col_seed``), walk along the detector-row axis within the order strip.
   At each new row position, extract a narrow column profile centred on
   the current line estimate and fit a Gaussian to locate the line centroid
   more precisely.  Walk in both directions from the seed row.

5. **Polynomial fitting** – fit a degree-``poly_degree`` polynomial
   ``col = f(row)``  to each traced line using iterative sigma-clipping
   (3 iterations, threshold ``sigma_clip``).

6. **Collect results** – store the traced positions, polynomial
   coefficients, and RMS values in an :class:`ArcLineTrace` object.

Relationship to later pipeline stages
--------------------------------------
The polynomials returned here represent the detector-column position of
each arc line as a function of detector row within one order.  They
constitute the raw material for:

* **Tilt measurement** – the slope ``d(col)/d(row)`` of each line
  polynomial encodes the local spectral-line tilt at that column.
* **Coefficient-surface fitting** – fitting a 2-D surface
  ``col(order_index, row)`` jointly across all orders and all identified
  lines gives the 2DXD "spatial distortion map".
* **Rectification-index generation** – converting the distortion map
  into resampling indices to produce a spatially rectified 2-D spectrum
  (NOT implemented in this module).
* **Wavelength calibration** – once lines are matched to the ThAr atlas
  (NOT implemented in this module), the col→wavelength mapping can be
  established.

What remains intentionally unimplemented in this PR
----------------------------------------------------
* ThAr line identification and atlas matching.
* Coefficient-surface fitting across orders.
* Rectification-index generation.
* Spatial or wavelength calibration.
* Full 2DXD pipeline integration.

See ``docs/ishell_arc_line_tracing.md`` for a full design rationale.
See ``docs/ishell_order_tracing.md`` for the preceding flat-tracing stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from scipy.signal import find_peaks as _scipy_find_peaks

from .tracing import FlatOrderTrace

__all__ = [
    "ArcLineTrace",
    "TracedArcLine",
    "trace_arc_lines",
    "load_and_combine_arcs",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TracedArcLine:
    """A single traced arc emission line within one echelle order.

    Attributes
    ----------
    order_index : int
        Zero-based index of the echelle order (index into the
        :class:`FlatOrderTrace` order list; not an echelle order number).
    seed_col : int
        Detector column at which this line was first identified.
    sample_rows : ndarray, shape (n_sample,)
        Detector row positions at which the line centroid was measured.
        Rows are within the order strip defined by the flat-field trace.
    centroid_cols : ndarray, shape (n_sample,)
        Sub-pixel detector column centroid of the arc line at each sample
        row.  ``NaN`` marks rows where centroid measurement failed.
    poly_coeffs : ndarray, shape (poly_degree + 1,)
        Polynomial coefficients for ``col = f(row)`` in the
        ``numpy.polynomial.polynomial`` convention:
        ``coeffs[k]`` is the coefficient of ``row**k``.
        Evaluate with ``np.polynomial.polynomial.polyval(row, coeffs)``.
    fit_rms : float
        RMS of the polynomial residuals, in pixels.  ``NaN`` if the fit
        failed (too few valid centroids).
    n_valid : int
        Number of valid (non-NaN) centroid measurements used in the fit.

    Notes
    -----
    The polynomial ``col = f(row)`` represents the horizontal (column)
    position of the line as it traverses the order band from bottom edge
    to top edge.  For perfectly vertical lines the polynomial reduces to
    a constant; real iSHELL arc lines exhibit a tilt (non-zero linear
    term) and a small curvature (non-zero quadratic term).
    """

    order_index: int
    seed_col: int
    sample_rows: np.ndarray
    centroid_cols: np.ndarray
    poly_coeffs: np.ndarray
    fit_rms: float
    n_valid: int


@dataclass
class ArcLineTrace:
    """Result of 2-D arc-line tracing from one or more iSHELL arc frames.

    This is the primary output of :func:`trace_arc_lines`.  It collects
    all traced arc lines across all echelle orders together with metadata
    about the tracing run.

    Attributes
    ----------
    n_orders : int
        Number of echelle orders that were searched for arc lines.
    n_lines_total : int
        Total number of arc lines traced across all orders.
    lines : list of :class:`TracedArcLine`
        Traced arc lines, in the order they were found (order index
        ascending, seed column ascending within each order).
    poly_degree : int
        Degree of the ``col = f(row)`` polynomials fitted to each line.
    seed_col : int
        Detector column used to identify the initial line seeds.
    flat_trace : :class:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace`
        The flat-field order geometry used to constrain the search strips.

    Notes
    -----
    The :attr:`lines` list can be iterated to access per-line polynomials.
    To retrieve only the lines in a single order, filter by
    ``line.order_index``.

    Examples
    --------
    >>> arc_trace = trace_arc_lines(arc_files, flat_trace)
    >>> print(f"Total lines traced: {arc_trace.n_lines_total}")
    >>> for line in arc_trace.lines:
    ...     if line.order_index == 0:
    ...         print(f"  seed col={line.seed_col}, RMS={line.fit_rms:.2f} px")
    """

    n_orders: int
    n_lines_total: int
    lines: list[TracedArcLine] = field(default_factory=list)
    poly_degree: int = 2
    seed_col: int = 0
    flat_trace: Optional[FlatOrderTrace] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def lines_for_order(self, order_index: int) -> list[TracedArcLine]:
        """Return the traced lines for a given order index.

        Parameters
        ----------
        order_index : int
            Zero-based order index.

        Returns
        -------
        list of :class:`TracedArcLine`
        """
        return [ln for ln in self.lines if ln.order_index == order_index]

    def n_lines_for_order(self, order_index: int) -> int:
        """Return the number of traced lines for a given order index.

        Parameters
        ----------
        order_index : int
            Zero-based order index.

        Returns
        -------
        int
        """
        return sum(1 for ln in self.lines if ln.order_index == order_index)

    def valid_lines(self, min_n_valid: int = 3) -> list[TracedArcLine]:
        """Return lines with at least *min_n_valid* valid centroid measurements.

        Parameters
        ----------
        min_n_valid : int, default 3
            Minimum number of valid centroids required.

        Returns
        -------
        list of :class:`TracedArcLine`
        """
        return [ln for ln in self.lines if ln.n_valid >= min_n_valid]

    def fit_rms_array(self) -> npt.NDArray:
        """Return per-line polynomial RMS values as an ndarray.

        Returns
        -------
        ndarray, shape (n_lines_total,)
            ``NaN`` for lines whose polynomial fit failed.
        """
        return np.array([ln.fit_rms for ln in self.lines])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def trace_arc_lines(
    arc_files: list[str],
    flat_trace: FlatOrderTrace,
    *,
    seed_col: Optional[int] = None,
    strip_half_width: Optional[int] = None,
    row_step: int = 1,
    poly_degree: int = 2,
    min_line_prominence: float = 100.0,
    min_line_distance: int = 5,
    max_line_shift: float = 3.0,
    gaussian_half_width: int = 4,
    sigma_clip: float = 3.0,
    col_range: Optional[tuple[int, int]] = None,
) -> ArcLineTrace:
    """Trace ThAr arc emission lines in 2-D from iSHELL arc frames.

    Uses the flat-field order geometry from *flat_trace* to define the
    search strip for each order, identifies candidate arc lines at a seed
    column, and traces each line in the row direction within the strip by
    fitting sub-pixel Gaussian centroids.

    Parameters
    ----------
    arc_files : list of str
        Paths to raw iSHELL FITS arc frames (PRIMARY extension is read).
        Multiple files are median-combined before tracing.
    flat_trace : :class:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace`
        Flat-field order geometry result.  Provides the centre-row
        polynomial and half-width for each order.
    seed_col : int, optional
        Detector column at which to identify initial line seeds.
        Defaults to the middle of the column range.
    strip_half_width : int, optional
        Half-width (in rows) of the order strip searched for arc lines.
        Defaults to ``flat_trace.half_width_rows`` rounded to the nearest
        integer for each order.  Passing an explicit value overrides this
        and applies the same strip width to all orders.
    row_step : int, default 1
        Step size (in rows) for the row-by-row tracing.  A value of 1
        traces every row; larger values speed up the tracing at the cost
        of sampling density.
    poly_degree : int, default 2
        Degree of the ``col = f(row)`` polynomial fitted to each line.
    min_line_prominence : float, default 100.0
        Minimum peak prominence (in detector counts) for a spectral peak
        at the seed column to be accepted as a candidate arc line.
    min_line_distance : int, default 5
        Minimum separation (in columns) between candidate arc lines at
        the seed column.
    max_line_shift : float, default 3.0
        Maximum column shift (in pixels) between the predicted line
        position and the measured Gaussian centroid at each row step.
        Prevents mis-matching to a neighbouring line.
    gaussian_half_width : int, default 4
        Half-width (in columns) of the narrow profile extracted around
        the current line position estimate for Gaussian centroid fitting.
    sigma_clip : float, default 3.0
        Sigma-clipping threshold for polynomial fitting.
    col_range : tuple of int (col_start, col_end), optional
        Column range within which seeds are sought and to which the seed
        column is clipped.  Defaults to the full detector width.

    Returns
    -------
    :class:`ArcLineTrace`
        Contains all traced lines for all orders with per-line polynomial
        coefficients and statistics.

    Raises
    ------
    ValueError
        If *arc_files* is empty.
    RuntimeError
        If the arc image shape is inconsistent with the flat-trace result.

    Notes
    -----
    The returned polynomial coefficients follow the
    ``numpy.polynomial.polynomial`` convention: ``coeffs[k]`` is the
    coefficient of ``row**k``.

    Examples
    --------
    >>> import glob
    >>> from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
    >>> from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
    >>>
    >>> flat_files = sorted(glob.glob(
    ...     "data/testdata/ishell_h1_calibrations/raw/*.flat.*.fits"))
    >>> arc_files = sorted(glob.glob(
    ...     "data/testdata/ishell_h1_calibrations/raw/*.arc.*.fits"))
    >>> flat_trace = trace_orders_from_flat(flat_files, col_range=(650, 1550))
    >>> arc_trace = trace_arc_lines(arc_files, flat_trace)
    >>> print(f"Traced {arc_trace.n_lines_total} lines across "
    ...       f"{arc_trace.n_orders} orders")
    """
    if not arc_files:
        raise ValueError("arc_files must not be empty.")

    # ------------------------------------------------------------------
    # 1. Load and median-combine arc frames
    # ------------------------------------------------------------------
    arc = load_and_combine_arcs(arc_files)
    nrows, ncols = arc.shape

    # ------------------------------------------------------------------
    # 2. Resolve column range and seed column
    # ------------------------------------------------------------------
    if col_range is None:
        col_lo, col_hi = 0, ncols - 1
    else:
        col_lo, col_hi = int(col_range[0]), int(col_range[1])

    if seed_col is None:
        _seed_col = (col_lo + col_hi) // 2
    else:
        _seed_col = max(col_lo, min(col_hi, int(seed_col)))

    logger.info(
        "Arc loaded: shape %s, seed col %d, col range [%d, %d]",
        arc.shape,
        _seed_col,
        col_lo,
        col_hi,
    )

    # ------------------------------------------------------------------
    # 3. Trace lines in each order
    # ------------------------------------------------------------------
    all_lines: list[TracedArcLine] = []

    for order_idx in range(flat_trace.n_orders):
        center_coeffs = flat_trace.center_poly_coeffs[order_idx]

        if strip_half_width is not None:
            hw = int(strip_half_width)
        else:
            hw = max(3, int(round(float(flat_trace.half_width_rows[order_idx]))))

        # Centre row at the seed column
        center_at_seed = float(
            np.polynomial.polynomial.polyval(float(_seed_col), center_coeffs)
        )

        # Define row limits for this order at seed column
        row_lo = max(0, int(round(center_at_seed - hw)))
        row_hi = min(nrows - 1, int(round(center_at_seed + hw)))

        if row_hi <= row_lo:
            logger.debug(
                "Order %d: degenerate row strip [%d, %d], skipping.",
                order_idx,
                row_lo,
                row_hi,
            )
            continue

        # ------------------------------------------------------------------
        # 3a. Find line seeds: spectral (column-axis) profile within the
        #     order strip, collapsed over rows.
        # ------------------------------------------------------------------
        # Average across rows within the order strip to get a 1-D spectral
        # profile as a function of detector column.  Peaks in this profile
        # correspond to arc emission lines.
        spectral_profile = np.median(
            arc[row_lo : row_hi + 1, col_lo : col_hi + 1].astype(float),
            axis=0,
        )

        seed_peaks_local = _find_line_peaks(
            spectral_profile, min_line_distance, min_line_prominence
        )

        if len(seed_peaks_local) == 0:
            logger.debug(
                "Order %d: no arc lines found in spectral profile "
                "(col range [%d, %d]).",
                order_idx, col_lo, col_hi,
            )
            continue

        # Convert local column indices (relative to col_lo) to detector columns
        seed_col_positions = seed_peaks_local + col_lo  # absolute column positions

        logger.debug(
            "Order %d: found %d line seeds at cols %s",
            order_idx,
            len(seed_col_positions),
            seed_col_positions.tolist(),
        )

        # The seed row for tracing is the order centre at the seed column
        seed_row = int(round(center_at_seed))

        # ------------------------------------------------------------------
        # 3b. For each seed column, trace the line in the row direction
        #     within the order strip
        # ------------------------------------------------------------------
        for seed_col_pos in seed_col_positions:
            # Initial column centroid at the seed row, centred on this line
            col_seed_centroid = _gaussian_centroid(
                arc,
                row=int(seed_row),
                col_est=float(seed_col_pos),
                half_width=gaussian_half_width,
                nrows=nrows,
                ncols=ncols,
            )
            if not np.isfinite(col_seed_centroid):
                continue

            # Define the row range for tracing this line within the order strip
            # Recompute the order strip boundaries at each row using the
            # centre-polynomial evaluated at _seed_col (conservative: we use the
            # column strip width as defined by the flat trace half-width).
            trace_row_lo = row_lo
            trace_row_hi = row_hi

            sample_rows = np.arange(
                trace_row_lo, trace_row_hi + 1, row_step, dtype=int
            )
            n_rows = len(sample_rows)
            centroid_cols = np.full(n_rows, np.nan)

            # Find seed index in sample_rows
            seed_in_samples = np.argmin(np.abs(sample_rows - seed_row))

            # Walk rightward (increasing row index) from seed
            current_est = col_seed_centroid
            for k in range(seed_in_samples, n_rows):
                r = int(sample_rows[k])
                c = _gaussian_centroid(
                    arc,
                    row=r,
                    col_est=current_est,
                    half_width=gaussian_half_width,
                    nrows=nrows,
                    ncols=ncols,
                )
                if np.isfinite(c) and abs(c - current_est) <= max_line_shift:
                    centroid_cols[k] = c
                    current_est = c
                # else: leave as NaN; keep current_est for next step

            # Walk leftward (decreasing row index) from seed-1
            current_est = col_seed_centroid
            for k in range(seed_in_samples - 1, -1, -1):
                r = int(sample_rows[k])
                c = _gaussian_centroid(
                    arc,
                    row=r,
                    col_est=current_est,
                    half_width=gaussian_half_width,
                    nrows=nrows,
                    ncols=ncols,
                )
                if np.isfinite(c) and abs(c - current_est) <= max_line_shift:
                    centroid_cols[k] = c
                    current_est = c

            # ------------------------------------------------------------------
            # 3c. Polynomial fit: col = f(row)
            # ------------------------------------------------------------------
            good = np.isfinite(centroid_cols)
            n_valid = int(good.sum())

            if n_valid < poly_degree + 1:
                poly_coeffs = np.full(poly_degree + 1, np.nan)
                fit_rms = np.nan
            else:
                rows_fit = sample_rows[good].astype(float)
                cols_fit = centroid_cols[good]
                poly_coeffs, fit_rms = _fit_poly_sigma_clip(
                    rows_fit, cols_fit, poly_degree, sigma_clip, n_iter=3
                )

            all_lines.append(
                TracedArcLine(
                    order_index=order_idx,
                    seed_col=int(round(col_seed_centroid)),
                    sample_rows=sample_rows,
                    centroid_cols=centroid_cols,
                    poly_coeffs=poly_coeffs,
                    fit_rms=float(fit_rms) if np.isfinite(fit_rms) else np.nan,
                    n_valid=n_valid,
                )
            )

    n_lines = len(all_lines)
    logger.info(
        "Arc-line tracing complete: %d orders searched, %d lines traced.",
        flat_trace.n_orders,
        n_lines,
    )
    valid_rms = [ln.fit_rms for ln in all_lines if np.isfinite(ln.fit_rms)]
    if valid_rms:
        logger.info("Median line polynomial RMS: %.3f px", float(np.median(valid_rms)))

    return ArcLineTrace(
        n_orders=flat_trace.n_orders,
        n_lines_total=n_lines,
        lines=all_lines,
        poly_degree=poly_degree,
        seed_col=_seed_col,
        flat_trace=flat_trace,
    )


# ---------------------------------------------------------------------------
# Public helper – loading
# ---------------------------------------------------------------------------


def load_and_combine_arcs(arc_files: list[str]) -> npt.NDArray:
    """Load iSHELL arc FITS frames and median-combine them.

    Reads the **PRIMARY extension** (extension index 0) of each file and
    takes the pixel-wise median over all frames.  The median is more
    robust than the mean against cosmic rays, hot pixels, and occasional
    bad reads.

    Parameters
    ----------
    arc_files : list of str
        Paths to raw iSHELL FITS arc files.  At least one file must be
        provided.

    Returns
    -------
    ndarray, shape (nrows, ncols), dtype float32
        Median-combined arc image.

    Raises
    ------
    ValueError
        If *arc_files* is empty.
    OSError
        If any file cannot be opened as a valid FITS file.
    """
    if not arc_files:
        raise ValueError("arc_files must not be empty.")

    imgs: list[npt.NDArray] = []
    for path in arc_files:
        with fits.open(path, memmap=False) as hdul:
            imgs.append(hdul[0].data.astype(np.float32))

    if len(imgs) == 1:
        return imgs[0]

    return np.median(np.stack(imgs, axis=0), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_line_peaks(
    profile: npt.NDArray,
    min_distance: int,
    min_prominence: float,
) -> npt.NDArray:
    """Find emission-line peaks in a 1-D spectral or spatial profile.

    Parameters
    ----------
    profile : ndarray, shape (n,)
        1-D profile to search for peaks.
    min_distance : int
        Minimum separation between peaks in pixels.
    min_prominence : float
        Minimum peak prominence in detector counts.

    Returns
    -------
    ndarray of int
        Indices of detected peaks, sorted in ascending order.
    """
    peaks, _ = _scipy_find_peaks(
        profile,
        distance=min_distance,
        prominence=min_prominence,
    )
    return peaks


def _gaussian_centroid(
    image: npt.NDArray,
    *,
    row: int,
    col_est: float,
    half_width: int,
    nrows: int,
    ncols: int,
) -> float:
    """Fit a Gaussian to a narrow profile and return the sub-pixel centroid.

    Extracts a 1-D profile from *image* along the column axis at *row*,
    centred on *col_est* with a half-width of *half_width* pixels.  A
    Gaussian is fitted to the profile using the method of moments to
    obtain a sub-pixel centroid.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Arc image.
    row : int
        Detector row at which the profile is extracted.
    col_est : float
        Current estimate of the arc-line column position.
    half_width : int
        Half-width (in columns) of the extraction window.
    nrows, ncols : int
        Image dimensions, used for boundary checking.

    Returns
    -------
    float
        Sub-pixel column centroid of the Gaussian fit.  ``NaN`` if the
        fit failed (e.g. profile is too noisy or the peak is too faint).
    """
    col_lo = max(0, int(round(col_est)) - half_width)
    col_hi = min(ncols, int(round(col_est)) + half_width + 1)

    if col_hi <= col_lo + 2:
        return np.nan
    if not (0 <= row < nrows):
        return np.nan

    profile = image[row, col_lo:col_hi].astype(float)

    # Subtract local minimum to remove any background pedestal
    bg = np.min(profile)
    profile -= bg

    total = np.sum(profile)
    if total <= 0.0:
        return np.nan

    cols = np.arange(col_lo, col_hi, dtype=float)
    centroid = float(np.sum(cols * profile) / total)

    # Sanity check: centroid must be within the extraction window
    if not (col_lo <= centroid <= col_hi - 1):
        return np.nan

    return centroid


def _fit_poly_sigma_clip(
    x: npt.NDArray,
    y: npt.NDArray,
    degree: int,
    sigma: float,
    n_iter: int = 3,
) -> tuple[npt.NDArray, float]:
    """Fit a polynomial with iterative sigma-clipping.

    Parameters
    ----------
    x : ndarray
        Independent variable values.
    y : ndarray
        Dependent variable values.
    degree : int
        Polynomial degree.
    sigma : float
        Clipping threshold in units of residual RMS.
    n_iter : int, default 3
        Number of sigma-clipping iterations.

    Returns
    -------
    coeffs : ndarray, shape (degree + 1,)
        Polynomial coefficients in ``numpy.polynomial.polynomial``
        convention.
    rms : float
        RMS of the final fit residuals in pixels.
    """
    xc = x.copy()
    yc = y.copy()

    coeffs = np.polynomial.polynomial.polyfit(xc, yc, degree)

    for _ in range(n_iter):
        predicted = np.polynomial.polynomial.polyval(xc, coeffs)
        resid = yc - predicted
        rms = float(np.std(resid))
        if rms == 0.0:
            break
        keep = np.abs(resid) <= sigma * rms
        if keep.sum() < degree + 1:
            break
        xc = xc[keep]
        yc = yc[keep]
        coeffs = np.polynomial.polynomial.polyfit(xc, yc, degree)

    predicted = np.polynomial.polynomial.polyval(xc, coeffs)
    final_rms = float(np.std(yc - predicted))

    return coeffs, final_rms
