"""
2-D arc-line tracing from iSHELL ThAr arc frames.

This module implements the second stage of the iSHELL 2DXD reduction
scaffold: using flat-field order geometry (from
:mod:`~pyspextool.instruments.ishell.tracing`) to identify and trace ThAr
arc emission lines in 2-D across the iSHELL H2RG detector.

What this module does
---------------------
* Loads one or more raw ThAr arc FITS frames and median-combines them.
* Uses an :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
  (produced by flat-field tracing) to define per-order row and column
  boundaries.
* For each order, collapses the order strip along the spatial (row) direction
  to produce a 1-D emission-line spectrum, then detects line candidates using
  prominence-based peak finding.
* For each candidate line, traces its column position row-by-row within the
  order using flux-weighted centroiding, yielding a set of
  ``(row, col)`` measurements that characterise the line's 2-D position on
  the detector.
* Fits a low-order polynomial ``col = poly(row)`` to each traced line using
  iterative sigma-clipping, recording the coefficients and residual RMS.
* Returns an :class:`ArcLineTraceResult` containing all traced lines,
  suitable for downstream coefficient-surface fitting.

What this module does NOT do (by design)
-----------------------------------------
* **No wavelength identification** — arc lines are found and traced
  purely from the pixel data; no matching against a ThAr line atlas is
  performed at this stage.
* **No coefficient-surface fitting** — fitting a global 2-D surface over
  (order, column) → wavelength is a subsequent step, not implemented here.
* **No rectification-index generation** — the interpolation arrays needed
  to resample the raw 2-D orders onto a (wavelength × spatial) grid are
  not produced here.
* **No full 2DXD wavelength solution** — the full echelle dispersion
  relation is not fitted.

Algorithm
---------
1. **Load frames** – open each arc FITS file (PRIMARY extension, index 0)
   and median-combine to suppress read noise and residual sky emission.

2. **Order strip extraction** – for each order in the
   :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`,
   compute approximate row bounds from the bottom and top edge polynomials
   evaluated at the order mid-column.  Extract the sub-image
   ``arc[row_lo:row_hi+1, x_start:x_end+1]``.

3. **1-D spectrum** – collapse the strip by taking the pixel-wise *median*
   along the row axis.  The median suppresses cosmic rays, bad pixels, and
   the slowly-varying spatial profile, leaving only the spectrally-resolved
   emission features.

4. **Line candidate detection** – find local maxima in the 1-D spectrum
   using ``scipy.signal.find_peaks`` with a prominence criterion.  The
   prominence is robust to the varying blaze-function background.

5. **Row-by-row centroiding** – for each candidate line at seed column
   ``c_seed``, iterate over detector rows within the order.  At each row
   ``r``, extract a narrow window of width ``2 × col_half_width + 1``
   pixels centred on ``c_seed`` and compute the flux-weighted centroid
   (centre of mass after local-minimum background subtraction).  Centroids
   that shift more than ``max_col_shift`` pixels from ``c_seed`` are
   discarded.

6. **Polynomial fitting with sigma-clipping** – fit
   ``col = poly(row)`` to the valid ``(row, centroid_col)`` pairs using
   a degree-``poly_degree`` polynomial.  Three rounds of σ-clipping
   (threshold ``sigma_clip × RMS``) reject outliers caused by bad pixels
   or cosmic rays.

7. **Quality filter** – discard traces that have fewer than
   ``min_trace_fraction × n_order_rows`` valid centroid measurements (after
   sigma-clipping), ensuring only well-sampled lines contribute to the
   result.

Relationship to the 2DXD pipeline
-----------------------------------
This module fills in the second stage of the arc-line tracing scaffold::

    Flat tracing (tracing.py)
        └── centre-line polynomial per order
        └── approximate edge polynomials
                │
                ▼
    Arc-line tracing (this module)
        └── 1-D line candidates per order
        └── col(row) polynomial per traced line
        └── ArcLineTraceResult
                │
                ▼
    Coefficient-surface fitting (NOT YET IMPLEMENTED)
        └── global (order × col) → wavelength surface
                │
                ▼
    2DXD Rectification (NOT YET IMPLEMENTED)
        └── resample to (wavelength × spatial) grid

The :class:`ArcLineTraceResult` stores the per-line polynomial
coefficients and is designed to feed directly into the coefficient-surface
fitting step: each traced line contributes one ``(order, col, row, poly)``
data point to the surface fit.

What remains unimplemented
--------------------------
* Arc-line identification (matching detector positions to ThAr wavelengths).
* Tilt-corrected centroiding (the current implementation uses a fixed seed
  column; a production tracer would update the seed using the running
  polynomial estimate from previous rows).
* Per-order column-range variation (all orders use the same ``x_start``
  and ``x_end`` from the geometry; the packaged calibration uses per-order
  ranges that differ by up to 400 columns).
* Coefficient-surface fitting over all orders.
* Full 2DXD wavelength solution and rectification.

See :mod:`pyspextool.instruments.ishell.geometry` for the geometry data
structures and ``docs/ishell_arc_line_tracing.md`` for the full design
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

from .geometry import OrderGeometrySet

__all__ = [
    "TracedArcLine",
    "ArcLineTraceResult",
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

    Each instance records the detector-frame measurements of one ThAr
    emission line traced row-by-row through the order.  The polynomial
    ``col = poly(row)`` captures the spectral-line tilt: how much the
    column position of the line changes along the spatial (slit) direction.

    Parameters
    ----------
    order_index : int
        Index of the parent order in
        :attr:`ArcLineTraceResult.geometry` ``.geometries`` (0-based
        placeholder).  This is **not** an echelle order number; order
        numbers are assigned during the wavelength-calibration step.
    seed_col : int
        Detector column at which the line was identified in the
        collapsed 1-D spectrum (the initial centroiding seed).
    trace_rows : ndarray, shape (n_valid,)
        Detector row indices at which a valid centroid column was
        measured.  Rows where the centroid failed quality checks are
        absent from this array.
    trace_cols : ndarray, shape (n_valid,)
        Flux-weighted centroid column positions at each row in
        *trace_rows*, with sub-pixel precision.
    poly_coeffs : ndarray, shape (poly_degree + 1,)
        Polynomial coefficients for ``col = poly(row)``, following the
        ``numpy.polynomial.polynomial`` convention: ``coeffs[k]`` is the
        coefficient of ``row**k``.  Evaluate with
        ``np.polynomial.polynomial.polyval(row, poly_coeffs)``.
    fit_rms : float
        RMS of the polynomial residuals in pixels.  ``NaN`` if the
        polynomial fit failed (too few valid points after sigma-clipping).
    peak_flux : float
        Prominence of the peak in the collapsed 1-D spectrum (a proxy for
        line strength).  Defined as the prominence value returned by
        ``scipy.signal.find_peaks``.

    Notes
    -----
    The ``poly_coeffs`` represent ``col(row)`` — the column position of
    the arc line as a function of detector row.  This convention is chosen
    because:

    * The spatial (row) direction is the independent variable across the
      slit.
    * The column shift captures the spectral-line tilt, which is needed
      for 2DXD rectification.
    * The polynomial is evaluated at a dense grid of rows during
      rectification-index generation (not yet implemented).

    The constant term ``poly_coeffs[0]`` is approximately the column
    position of the line at row 0, and the linear term
    ``poly_coeffs[1]`` approximates the tilt slope in pixels per row.
    """

    order_index: int
    seed_col: int
    trace_rows: np.ndarray
    trace_cols: np.ndarray
    poly_coeffs: np.ndarray
    fit_rms: float
    peak_flux: float

    @property
    def n_trace_points(self) -> int:
        """Number of valid (row, col) trace points."""
        return len(self.trace_rows)

    def eval_col(self, rows: npt.ArrayLike) -> np.ndarray:
        """Evaluate the col(row) polynomial at *rows*.

        Parameters
        ----------
        rows : array_like
            Detector row coordinates.

        Returns
        -------
        ndarray
            Predicted column positions at each row.
        """
        return np.polynomial.polynomial.polyval(
            np.asarray(rows, dtype=float), self.poly_coeffs
        )

    def tilt_slope(self) -> float:
        """Linear tilt slope in pixels per row (first-order term).

        Returns
        -------
        float
            ``poly_coeffs[1]`` if available, else 0.0 for a
            degree-0 polynomial.
        """
        if len(self.poly_coeffs) >= 2:
            return float(self.poly_coeffs[1])
        return 0.0


@dataclass
class ArcLineTraceResult:
    """Result of 2-D arc-line tracing for one iSHELL observing mode.

    This is the primary output of :func:`trace_arc_lines`.  It collects
    all traced arc lines from all echelle orders and stores the flat-field
    geometry used to define the order boundaries.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"H1"``).
    arc_files : list of str
        Paths to the arc FITS files that were loaded and combined.
    poly_degree : int
        Degree of the ``col(row)`` polynomial fitted to each traced line.
    geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        The flat-field order geometry used to define order regions.
    traced_lines : list of :class:`TracedArcLine`
        All successfully traced arc lines, ordered by order index and
        then by seed column within each order.

    Notes
    -----
    This object is designed to feed directly into the (not yet implemented)
    coefficient-surface fitting step, which will:

    * collect ``(order_index, seed_col, poly_coeffs)`` from all traced
      lines,
    * fit a global 2-D surface mapping ``(order, col) → wavelength``, and
    * populate :attr:`~pyspextool.instruments.ishell.geometry.OrderGeometry.wave_coeffs`
      on each order in the geometry set.
    """

    mode: str
    arc_files: list[str]
    poly_degree: int
    geometry: OrderGeometrySet
    traced_lines: list[TracedArcLine] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_lines(self) -> int:
        """Total number of successfully traced arc lines."""
        return len(self.traced_lines)

    @property
    def n_orders(self) -> int:
        """Number of orders in the geometry set."""
        return self.geometry.n_orders

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def lines_for_order(self, order_index: int) -> list[TracedArcLine]:
        """Return all traced lines for the given order index.

        Parameters
        ----------
        order_index : int
            Index into :attr:`geometry` ``.geometries`` (0-based
            placeholder integer).

        Returns
        -------
        list of :class:`TracedArcLine`
            Traced lines belonging to that order, in seed-column order.
        """
        return [ln for ln in self.traced_lines if ln.order_index == order_index]

    def n_lines_per_order(self) -> npt.NDArray:
        """Number of traced lines per order.

        Returns
        -------
        ndarray of int, shape (n_orders,)
            ``result[i]`` is the number of lines traced in order *i*.
        """
        counts = np.zeros(self.n_orders, dtype=int)
        for ln in self.traced_lines:
            if 0 <= ln.order_index < self.n_orders:
                counts[ln.order_index] += 1
        return counts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def trace_arc_lines(
    arc_files: list[str],
    order_geometry: OrderGeometrySet,
    *,
    poly_degree: int = 2,
    min_line_prominence: float = 200.0,
    min_line_distance: int = 5,
    col_half_width: int = 5,
    min_trace_fraction: float = 0.3,
    max_col_shift: float = 5.0,
    sigma_clip: float = 3.0,
) -> ArcLineTraceResult:
    """Trace ThAr arc emission lines in 2-D using flat-field order geometry.

    For each order in *order_geometry*, identifies arc emission-line
    candidates from a median-collapsed 1-D spectrum, then traces each
    line row-by-row using flux-weighted centroiding.  A low-order
    polynomial ``col = poly(row)`` is fitted to each traced line to
    capture the spectral-line tilt across the slit.

    Parameters
    ----------
    arc_files : list of str
        Paths to raw iSHELL ThAr arc FITS files (PRIMARY extension is
        read).  When more than one file is provided the frames are
        median-combined before tracing.
    order_geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        Order geometry obtained from flat-field tracing.  Provides the
        per-order row and column boundaries used to define the strip for
        arc-line detection and tracing.
    poly_degree : int, default 2
        Degree of the polynomial fitted to each traced line's
        ``col(row)`` trajectory.  Degree 1 (linear) captures a constant
        tilt; degree 2 adds a curvature term.  Higher degrees are
        generally not recommended without a significantly wider order
        (more rows).
    min_line_prominence : float, default 200.0
        Minimum peak prominence (in detector counts) required for a
        feature in the collapsed 1-D spectrum to be accepted as an arc
        line candidate.  Prominence is robust to a slowly-varying blaze
        background.  Decrease this value if lines are being missed;
        increase it to suppress noise peaks.
    min_line_distance : int, default 5
        Minimum separation in columns between accepted arc line
        candidates.  Should be set to roughly the expected minimum line
        spacing in pixels.
    col_half_width : int, default 5
        Half-width (in columns) of the window used for flux-weighted
        centroiding at each row.  The centroiding window spans
        ``[seed_col - col_half_width, seed_col + col_half_width]``.
        Should be wide enough to include the full line profile but
        narrow enough to avoid contamination from adjacent lines.
    min_trace_fraction : float, default 0.3
        Minimum fraction of order rows that must yield valid centroid
        measurements for a traced line to be retained.  Traces with fewer
        valid points than ``max(poly_degree + 2,
        int(min_trace_fraction × n_rows))`` are discarded.
    max_col_shift : float, default 5.0
        Maximum allowed displacement (in columns) of the measured centroid
        from the seed column.  Centroids that shift by more than this
        value are rejected as likely bad-pixel or blending artefacts.
    sigma_clip : float, default 3.0
        Sigma-clipping threshold applied during polynomial fitting.
        Centroid measurements whose residuals exceed
        ``sigma_clip × RMS`` are rejected in three successive iterations.

    Returns
    -------
    :class:`ArcLineTraceResult`
        Tracing result with per-line polynomial coefficients, trace
        points, and residual RMS values.

    Raises
    ------
    ValueError
        If *arc_files* is empty or *order_geometry* contains no orders.

    Notes
    -----
    The returned polynomial coefficients follow the
    ``numpy.polynomial.polynomial`` convention: ``coeffs[k]`` is the
    coefficient of ``row**k``.  Evaluate with
    ``np.polynomial.polynomial.polyval(row, coeffs)``.

    The arc-line positions returned here are in raw detector (pixel)
    coordinates.  Wavelength identification — matching these positions
    to ThAr atlas wavelengths — is a subsequent step not implemented in
    this module.

    Examples
    --------
    Typical usage after obtaining an :class:`OrderGeometrySet` from flat
    tracing:

    >>> import glob
    >>> from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
    >>> from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
    >>>
    >>> flat_files = sorted(glob.glob(
    ...     "data/testdata/ishell_h1_calibrations/raw/*.flat.*.fits"
    ... ))
    >>> arc_files = sorted(glob.glob(
    ...     "data/testdata/ishell_h1_calibrations/raw/*.arc.*.fits"
    ... ))
    >>>
    >>> # Stage 1: flat-field order tracing
    >>> trace = trace_orders_from_flat(flat_files, col_range=(650, 1550))
    >>> geom = trace.to_order_geometry_set("H1", col_range=(650, 1550))
    >>>
    >>> # Stage 2: arc-line tracing
    >>> arc_result = trace_arc_lines(arc_files, geom)
    >>> print(f"Traced {arc_result.n_lines} lines across {arc_result.n_orders} orders")
    """
    if not arc_files:
        raise ValueError("arc_files must not be empty.")
    if order_geometry.n_orders == 0:
        raise ValueError("order_geometry must contain at least one order.")

    # ------------------------------------------------------------------
    # 1. Load and median-combine arc frames
    # ------------------------------------------------------------------
    arc = load_and_combine_arcs(arc_files)
    nrows, ncols = arc.shape

    all_lines: list[TracedArcLine] = []

    # ------------------------------------------------------------------
    # 2. Process each order
    # ------------------------------------------------------------------
    for idx, geom in enumerate(order_geometry.geometries):

        # Determine the valid column range for this order, clipped to
        # the detector boundary.
        col_lo = max(0, geom.x_start)
        col_hi = min(ncols - 1, geom.x_end)

        if col_hi - col_lo < 5:
            logger.warning(
                "Order %d: column range too small (%d–%d); skipping.",
                idx, col_lo, col_hi,
            )
            continue

        # Evaluate order row bounds at the mid-column.
        center_col = (col_lo + col_hi) // 2
        row_bot_f = geom.eval_bottom_edge(float(center_col))
        row_top_f = geom.eval_top_edge(float(center_col))

        row_lo = max(0, int(np.ceil(float(row_bot_f))) + 1)
        row_hi = min(nrows - 1, int(np.floor(float(row_top_f))) - 1)

        if row_hi - row_lo < poly_degree + 2:
            logger.warning(
                "Order %d: row range too narrow (%d–%d); skipping.",
                idx, row_lo, row_hi,
            )
            continue

        # ------------------------------------------------------------------
        # 3. Collapse the order strip to a 1-D spectrum
        # ------------------------------------------------------------------
        strip = arc[row_lo : row_hi + 1, col_lo : col_hi + 1].astype(float)
        spectrum = np.median(strip, axis=0)  # shape (col_hi - col_lo + 1,)

        # ------------------------------------------------------------------
        # 4. Detect arc line candidates
        # ------------------------------------------------------------------
        peaks_rel, props = _scipy_find_peaks(
            spectrum,
            distance=min_line_distance,
            prominence=min_line_prominence,
        )

        if len(peaks_rel) == 0:
            logger.debug("Order %d: no arc line candidates found.", idx)
            continue

        # Convert relative column indices to absolute detector columns
        peaks_abs = peaks_rel + col_lo
        prominences = props["prominences"]

        logger.debug(
            "Order %d: found %d arc line candidates at columns %s.",
            idx, len(peaks_abs), peaks_abs.tolist(),
        )

        # ------------------------------------------------------------------
        # 5. Trace each candidate line row-by-row
        # ------------------------------------------------------------------
        rows_in_order = np.arange(row_lo, row_hi + 1, dtype=int)
        min_valid = max(
            poly_degree + 2,
            int(np.round(min_trace_fraction * len(rows_in_order))),
        )

        for k, c_seed in enumerate(peaks_abs):
            trace_rows, trace_cols = _trace_single_line(
                arc, int(c_seed), rows_in_order,
                col_half_width, max_col_shift, nrows, ncols,
            )

            if len(trace_rows) < min_valid:
                logger.debug(
                    "Order %d, line at col %d: only %d valid trace points "
                    "(minimum %d); skipping.",
                    idx, int(c_seed), len(trace_rows), min_valid,
                )
                continue

            # ------------------------------------------------------------------
            # 6. Fit col = poly(row) with sigma-clipping
            # ------------------------------------------------------------------
            coeffs, rms = _fit_poly_sigma_clip(
                trace_rows.astype(float),
                trace_cols,
                poly_degree,
                sigma_clip,
                n_iter=3,
            )

            all_lines.append(
                TracedArcLine(
                    order_index=idx,
                    seed_col=int(c_seed),
                    trace_rows=trace_rows,
                    trace_cols=trace_cols,
                    poly_coeffs=coeffs,
                    fit_rms=rms,
                    peak_flux=float(prominences[k]),
                )
            )

    logger.info(
        "Arc-line tracing complete: %d lines traced across %d orders.",
        len(all_lines), order_geometry.n_orders,
    )

    return ArcLineTraceResult(
        mode=order_geometry.mode,
        arc_files=list(arc_files),
        poly_degree=poly_degree,
        geometry=order_geometry,
        traced_lines=all_lines,
    )


# ---------------------------------------------------------------------------
# Public helper — loading
# ---------------------------------------------------------------------------


def load_and_combine_arcs(arc_files: list[str]) -> npt.NDArray:
    """Load iSHELL ThAr arc FITS frames and median-combine them.

    Reads the **PRIMARY extension** (extension index 0) of each file and
    takes the pixel-wise median over all frames.  Median combination
    suppresses cosmic rays, hot pixels, and occasional bad reads more
    robustly than the mean.

    Parameters
    ----------
    arc_files : list of str
        Paths to raw iSHELL ThAr arc FITS files.  At least one file must
        be provided.

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


def _trace_single_line(
    arc: npt.NDArray,
    seed_col: int,
    rows: npt.NDArray,
    col_half_width: int,
    max_col_shift: float,
    nrows: int,
    ncols: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Trace one arc line across a set of rows via flux-weighted centroiding.

    At each row in *rows*, the function extracts a narrow column window
    centred on *seed_col* and computes the flux-weighted centroid.
    Measurements that shift too far from the seed or that have insufficient
    flux are discarded.

    Parameters
    ----------
    arc : ndarray, shape (nrows, ncols)
        Median-combined arc image.
    seed_col : int
        Initial column estimate (from the collapsed 1-D spectrum peak).
    rows : ndarray of int
        Detector rows to trace over (typically all rows in the order strip).
    col_half_width : int
        Half-width of the centroiding window in columns.
    max_col_shift : float
        Maximum allowed displacement of the centroid from *seed_col*.
    nrows, ncols : int
        Detector dimensions (for bounds checking).

    Returns
    -------
    trace_rows : ndarray of int
        Rows at which a valid centroid was obtained.
    trace_cols : ndarray of float
        Centroid column positions (sub-pixel) at each row in *trace_rows*.
    """
    valid_rows: list[int] = []
    valid_cols: list[float] = []

    for r in rows:
        if r < 0 or r >= nrows:
            continue

        c0 = max(0, seed_col - col_half_width)
        c1 = min(ncols - 1, seed_col + col_half_width)

        if c1 <= c0:
            continue

        centroid = _flux_centroid(arc[r, c0 : c1 + 1], c0)

        if np.isnan(centroid):
            continue

        if abs(centroid - seed_col) > max_col_shift:
            continue

        valid_rows.append(int(r))
        valid_cols.append(float(centroid))

    if not valid_rows:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    return np.array(valid_rows, dtype=int), np.array(valid_cols, dtype=float)


def _flux_centroid(window: npt.NDArray, col_offset: int) -> float:
    """Compute the flux-weighted centroid column of a 1-D pixel window.

    The local background is estimated as the minimum of the window.  Net
    flux is clipped to zero before computing the centroid to avoid
    negative-flux artefacts.

    Parameters
    ----------
    window : ndarray, shape (n,)
        Pixel values in the centroiding window.
    col_offset : int
        Absolute detector column index of the first element of *window*.

    Returns
    -------
    float
        Flux-weighted centroid in absolute column coordinates.
        Returns ``NaN`` if the net flux is non-positive (line not
        detected in this row).
    """
    bg = float(np.min(window))
    net = np.asarray(window, dtype=float) - bg
    net = np.maximum(net, 0.0)
    total = float(net.sum())

    if total <= 0.0:
        return np.nan

    cols = np.arange(col_offset, col_offset + len(window), dtype=float)
    return float((net * cols).sum() / total)


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
        Independent variable (e.g. row).
    y : ndarray
        Dependent variable (e.g. centroid column).
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
        RMS of the final fit residuals in pixels.  Returns ``NaN`` if
        the fit failed (too few points remaining after clipping).
    """
    if len(x) < degree + 1:
        return np.zeros(degree + 1), np.nan

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
