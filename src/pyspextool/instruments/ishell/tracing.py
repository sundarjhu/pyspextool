"""
Order-edge and order-centre tracing from iSHELL flat-field frames.

This module implements the first stage of the iSHELL 2DXD reduction
scaffold: tracing echelle order bottom/top edges from one or more
median-combined QTH flat-field frames.  The result is a
:class:`FlatOrderTrace` object containing:

* the sampled bottom-edge and top-edge row positions at a set of detector
  columns (the **primary traced quantities**, matching IDL),
* robust polynomial fits to those edge positions (``bot_poly_coeffs``,
  ``top_poly_coeffs``),
* centre-line polynomial coefficients derived from the edge fits,
* estimated per-order half-widths derived from edge separation, and
* per-order valid column ranges (``order_xranges``), matching the IDL
  ``xranges`` output of ``mc_adjustguesspos``.

The :class:`FlatOrderTrace` can be converted to an
:class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` via its
:meth:`~FlatOrderTrace.to_order_geometry_set` method, which feeds into the
later 2DXD arc-line tracing and wavecal stages.

IDL Caller Chain (iSHELL flat-field geometry)
----------------------------------------------
The IDL procedures relevant to iSHELL flat/order geometry, in call order:

1. ``mc_ishellcals2dxd.pro`` (``ishellcals2dxd_mkflat`` sub-procedure) –
   top-level iSHELL calibration driver.  Calls the following in sequence:

2. ``mc_readflatinfo(flatinfofile)`` – reads the reference
   ``K3_flatinfo.fits`` (or other mode's flatinfo) file.  Returns a
   structure containing stored reference ``edgecoeffs`` (shape
   ``[degree+1, 2, norders]``), ``xranges``, ``omask``, ``ycororder``,
   ``ybuffer``, tracing parameters ``step``, ``flatfrac``, ``comwin``,
   ``slith_range``, ``edgedeg``.

3. ``mc_adjustguesspos(edgecoeffs, xranges, flat, omask, orders,
   ycororder, ybuffer)`` – adjusts the reference guess positions by
   performing a vertical cross-correlation between the stored order mask
   and the new flat to measure any spatial shift.  Returns adjusted
   ``guesspos`` and ``xranges``.

4. ``mc_findorders(flat, adj.guesspos, adj.xranges, step, slith_range,
   edgedeg, ybuffer, flatfrac, comwin, edgecoeffs, xranges)`` – the
   core tracing routine.  For each order it sweeps the per-order sample
   columns, tracks bottom and top edges via Sobel-image COM, fits robust
   polynomials to each edge separately, and returns ``edgecoeffs``
   (shape ``[degree+1, 2, norders]``) and ``xranges``.

The **output geometry product** from IDL is ``edgecoeffs`` —
polynomials for the bottom edge (index 0) and top edge (index 1) of
each order.  Centres are derived from the edges; they are not the
primary fit target.

Algorithm (Python port)
-----------------------
1. **Load frames** – median-combine flat FITS files.

2. **Order initialization**:

   - *With flatinfo (full path)*: call :func:`_adjust_guess_positions` to
     perform the IDL ``mc_adjustguesspos`` vertical cross-correlation
     between the stored order mask and the new flat.  Requires
     ``flatinfo.omask`` and ``flatinfo.ycororder`` to be populated.
   - *With flatinfo (DEFAULT path)*: when ``omask`` or ``ycororder`` is
     absent, use :func:`_compute_guess_positions_from_flatinfo` (IDL
     ``mc_adjustguesspos /DEFAULT`` equivalent).
   - *Without flatinfo (explicit path)*: caller must supply ``guess_rows``
     — a list of one centre-row estimate per order at the seed column.
     IDL ``mc_findorders`` always requires explicit ``guesspos`` from outside;
     there is no auto-detection mode in the IDL algorithm.

3. **Per-order sample columns** – in flatinfo mode, each order has its
   own valid column range from the adjusted ``xranges``; the sweep is
   restricted to that range.  In fallback mode all orders share a single
   common column range.

4. **Sobel enhancement** – compute gradient magnitude (IDL ``sobel()``).

5. **Edge tracking** – for each order, sweep left then right from the
   guess column, tracking bottom and top edges via flux threshold +
   Sobel COM (IDL ``mc_findorders`` inner loop).  Only columns within
   the order's valid xrange are traced.

6. **Polynomial fitting** – fit a robust polynomial to the **bottom
   edge** array and a separate one to the **top edge** array (IDL
   ``mc_robustpoly1d``).  Centres are derived from the fitted edges.

IDL-to-Python fidelity status
------------------------------
**Matches IDL:**

- Bottom and top edges are the primary tracked/fitted quantities
- Sobel-image COM edge refinement
- ``edges[k,j]`` stored and fitted (not centres)
- Robust polynomial fitting for both edges
- ``goto cont1`` / boundary-flag stop logic
- Initialization window around guess column index
- ``mc_adjustguesspos`` cross-correlation: vertical sweep over
  ``nshifts = slith_pix * 1.8 + 1`` integer offsets; offset is subtracted
  from all guess y-positions; per-order xranges recomputed with shifted edges
- Per-order valid column ranges (``order_xranges``)
- Per-order step-based sample column sets
- Post-fit xrange restriction: after edge polynomial fitting, ``order_xranges``
  is narrowed to only those columns where BOTH fitted edges evaluate to valid
  pixel positions (``0 < edge < nrows-1``), matching IDL's end-of-loop
  ``where(top gt 0 and top lt nrows-1 and bot gt 0 and bot lt nrows-1)`` block
- Per-order polynomial domain enforcement: QA statistics (curvature,
  oscillation, inter-order separation) are computed only within each order's
  valid ``[x_start, x_end]`` range; extrapolated tails return ``NaN`` via
  :func:`_polyval_with_xrange`

**Does NOT fully match IDL:**

- Step-based column sampling is used when ``flatinfo.step > 0``; otherwise
  ``n_sample_cols`` evenly-spaced columns are used (no IDL equivalent for
  that fallback).
- When neither ``flatinfo`` nor explicit ``guess_rows`` are provided, a
  ``ValueError`` is raised; IDL always requires external ``guesspos``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from scipy.ndimage import sobel as _scipy_sobel

from pyspextool.fit.polyfit import polyfit_1d as _polyfit_1d

from .geometry import OrderGeometry, OrderGeometrySet

__all__ = [
    "FlatOrderTrace",
    "OrderTraceSamples",
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
class OrderTraceSamples:
    """Per-order traced sample data — the authoritative internal representation.

    This is the primary output of the per-order tracing loop in
    :func:`trace_orders_from_flat`.  Each order has its own 1-D arrays of
    column indices and the corresponding traced edge/centre rows, which
    may span a different column range from other orders (matching the IDL
    ``sranges[*,i]`` design).

    Attributes
    ----------
    order_index : int
        Zero-based index of this order within the :class:`FlatOrderTrace`.
    sample_cols : ndarray, shape (n_samp,)
        Column indices at which this order was traced.  These are the
        original sampling grid built from the initial flatinfo/input xranges.
        **After post-fit xrange restriction (step 6a in
        :func:`trace_orders_from_flat`), some entries may fall outside
        ``[x_start, x_end]``.  The authoritative final valid domain is
        ``[x_start, x_end]``, not the span of ``sample_cols``.**
    center_rows : ndarray, shape (n_samp,)
        Derived centre-row at each sampled column (mean of bot/top edges).
        NaN where edges were not reliably detected.
    bot_rows : ndarray, shape (n_samp,)
        Bottom-edge row at each sampled column.  NaN where not detected.
    top_rows : ndarray, shape (n_samp,)
        Top-edge row at each sampled column.  NaN where not detected.
    x_start : int
        First column of the order's **final post-fit restricted valid range**
        (IDL ``xranges[0, i]``).  Set during step 6a to ensure the fitted
        edge polynomials stay within the detector.  May be larger than
        ``sample_cols[0]`` when the polynomial overshoots at the low end.
    x_end : int
        Last column of the order's **final post-fit restricted valid range**
        (IDL ``xranges[1, i]``).  May be smaller than ``sample_cols[-1]``
        when the polynomial overshoots at the high end.
    """

    order_index: int
    sample_cols: np.ndarray
    center_rows: np.ndarray
    bot_rows: np.ndarray
    top_rows: np.ndarray
    x_start: int
    x_end: int


@dataclass
class FlatOrderTrace:
    """Result of order-centre tracing from one or more flat-field frames.

    Attributes
    ----------
    n_orders : int
        Number of orders successfully traced.
    order_samples : list of OrderTraceSamples
        **Authoritative per-order traced samples.**  Each element holds the
        traced bottom-edge, top-edge, and centre arrays for one order, on
        that order's own column grid (IDL ``sranges[*,i]`` domain).  This
        is the native internal representation produced by the tracing loop.
        Length equals ``n_orders`` when populated by :func:`trace_orders_from_flat`;
        empty list when the object is constructed directly (backward-compatibility).
    sample_cols : ndarray, shape (n_sample,)
        **Compatibility view — MUST NOT be used for internal computation.**
        Union of all per-order sample column sets, populated from
        ``order_samples`` after tracing.  Exists only to satisfy callers that
        expect a shared column grid.  All polynomial fitting, edge detection,
        and geometry construction MUST use ``order_samples[i].sample_cols``
        instead.
    center_rows : ndarray, shape (n_orders, n_sample)
        **Compatibility view — MUST NOT be used for internal computation.**
        Centre-row positions on the shared ``sample_cols`` grid; NaN for
        columns outside each order's post-fit restricted xrange
        ``[x_start, x_end]``.  Columns that appear in an order's
        ``sample_cols`` but lie outside its ``[x_start, x_end]`` are also
        NaN here.  Derived from ``order_samples`` after tracing.  All
        polynomial fitting, edge detection, and geometry construction MUST
        use ``order_samples[i].center_rows`` instead.
    center_poly_coeffs : ndarray, shape (n_orders, poly_degree + 1)
        Per-order polynomial coefficients for the centre-row trajectory,
        derived as the arithmetic mean of ``bot_poly_coeffs`` and
        ``top_poly_coeffs``.  Follows the ``numpy.polynomial.polynomial``
        convention: ``coeffs[k]`` is the coefficient of ``col**k``.
    fit_rms : ndarray, shape (n_orders,)
        RMS of the centre polynomial residuals for each order, in pixels.
    half_width_rows : ndarray, shape (n_orders,)
        Estimated half-width of each order in pixels, derived from the mean
        bottom-to-top edge separation at the guess/seed column.
    poly_degree : int
        Degree of the fitted polynomials.
    seed_col : int
        Detector column used for initial seed detection (auto-detect mode)
        or the midpoint of the guess column range (flatinfo mode).
    bot_poly_coeffs : ndarray, shape (n_orders, poly_degree + 1) or None
        Per-order polynomial coefficients for the **bottom edge** of each
        order.  This is the primary IDL output quantity (IDL
        ``edgecoeffs[*,0,i]``).  ``None`` when the object was constructed
        without edge tracking (e.g. directly in tests).
    top_poly_coeffs : ndarray, shape (n_orders, poly_degree + 1) or None
        Per-order polynomial coefficients for the **top edge** of each
        order (IDL ``edgecoeffs[*,1,i]``).  ``None`` as above.
    order_xranges : ndarray, shape (n_orders, 2) or None
        Per-order valid column ranges ``[x_start, x_end]``, matching the
        IDL ``xranges[*,i]`` output of ``mc_adjustguesspos``.  Derived from
        ``order_samples`` when populated.
    order_stats : list of OrderTraceStats
        Per-order QA metrics.  Empty if constructed directly.
    """

    n_orders: int
    sample_cols: np.ndarray
    center_rows: np.ndarray
    center_poly_coeffs: np.ndarray
    fit_rms: np.ndarray
    half_width_rows: np.ndarray
    poly_degree: int
    seed_col: int
    bot_poly_coeffs: Optional[np.ndarray] = None
    top_poly_coeffs: Optional[np.ndarray] = None
    order_xranges: Optional[np.ndarray] = None
    order_samples: list[OrderTraceSamples] = field(default_factory=list)
    order_stats: list[OrderTraceStats] = field(default_factory=list)

    def to_order_geometry_set(
        self,
        mode: str,
        col_range: tuple[int, int] | None = None,
    ) -> OrderGeometrySet:
        """Convert the tracing result to an :class:`OrderGeometrySet`.

        When ``bot_poly_coeffs`` and ``top_poly_coeffs`` are available on
        this object (produced by :func:`trace_orders_from_flat` with edge
        tracking), the bottom and top edge polynomials are taken directly
        from the fitted edge polynomials.  This matches the IDL path, where
        ``edgecoeffs`` is the primary output of ``mc_findorders``.

        When ``bot_poly_coeffs`` / ``top_poly_coeffs`` are ``None`` (the
        object was constructed directly without running the full tracing),
        the edges are approximated as ``centre ± half_width_rows`` —
        a fallback for backward-compatibility with code that builds
        :class:`FlatOrderTrace` objects directly (e.g. in tests).

        When ``order_xranges`` is available, each order's ``x_start`` /
        ``x_end`` is taken from the per-order range (matching IDL
        ``xranges``).  Otherwise, *col_range* is applied to all orders.

        Parameters
        ----------
        mode : str
            iSHELL observing mode name (e.g. ``"H1"``).
        col_range : tuple of int (col_start, col_end), optional
            Column range applied uniformly to all orders when
            ``order_xranges`` is ``None``.  Defaults to
            ``(sample_cols[0], sample_cols[-1])``.

        Returns
        -------
        :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        """
        if col_range is None:
            if self.order_samples:
                # Use first/last order x_start/x_end as representative range.
                default_x_start = min(os_i.x_start for os_i in self.order_samples)
                default_x_end = max(os_i.x_end for os_i in self.order_samples)
            else:
                # Backward-compat: direct-construction path (no order_samples).
                default_x_start = int(self.sample_cols[0])
                default_x_end = int(self.sample_cols[-1])
        else:
            default_x_start, default_x_end = int(col_range[0]), int(col_range[1])

        geometries = []
        for i in range(self.n_orders):
            # Per-order x-range: use order_xranges when available (IDL path).
            if self.order_xranges is not None:
                x_start = int(self.order_xranges[i, 0])
                x_end = int(self.order_xranges[i, 1])
            else:
                x_start = default_x_start
                x_end = default_x_end

            if self.bot_poly_coeffs is not None and self.top_poly_coeffs is not None:
                # Use the directly-traced edge polynomials (IDL output path).
                bot_coeffs = self.bot_poly_coeffs[i].copy()
                top_coeffs = self.top_poly_coeffs[i].copy()
            else:
                # Fallback: approximate edges from centre ± half_width.
                # Used when bot/top polynomials are not available (e.g. when
                # this object was constructed directly in tests).
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
    flatinfo=None,
    n_sample_cols: int = 40,
    poly_degree: int = 3,
    seed_col: int | None = None,
    col_range: tuple[int, int] | None = None,
    guess_rows: list[float] | None = None,
    max_shift: float = 15.0,
    sigma_clip: float = 3.0,
    intensity_fraction: float = 0.85,
    com_half_width: int = 2,
    slit_height_range: tuple[float, float] | None = None,
    ybuffer: int = 1,
) -> FlatOrderTrace:
    """Trace echelle order edges from one or more iSHELL flat-field frames.

    This function implements a direct port of IDL ``mc_findorders``.  It
    requires explicit initial guess positions for each order — either via
    *flatinfo* (the normal production path) or via *guess_rows* (a list of
    one approximate centre-row value per order at the seed column, used for
    development or testing when flatinfo is not available).

    IDL ``mc_findorders`` always receives ``guesspos`` from outside (typically
    from ``mc_adjustguesspos``); there is no auto-detection mode.  Accordingly,
    this function raises ``ValueError`` if neither *flatinfo* nor *guess_rows*
    is provided.

    When *flatinfo* is supplied, order initialization follows the IDL
    ``mc_readflatinfo`` / ``mc_adjustguesspos`` path: guess positions are
    derived from the stored reference edge polynomials in the flatinfo
    FITS file, and tracing parameters (``intensity_fraction``,
    ``com_half_width``, ``slit_height_range``, ``ybuffer``, ``poly_degree``)
    are taken from the flatinfo fields, overriding any explicit keyword
    arguments.

    In both cases the tracking algorithm follows IDL ``mc_findorders``:
    for each order it sweeps left and right, tracking the **bottom and top
    edges** via flux-threshold + Sobel-image COM, then fits a robust
    polynomial to each edge separately.  The centre polynomial is derived
    from the edge fits.

    Parameters
    ----------
    flat_files : list of str
        Paths to raw iSHELL FITS flat frames (PRIMARY extension is read).
        When more than one file is provided the frames are median-combined.
    flatinfo : :class:`~pyspextool.instruments.ishell.calibrations.FlatInfo`, optional
        Mode calibration metadata loaded from the flatinfo FITS file (e.g.
        via :func:`~pyspextool.instruments.ishell.calibrations.read_flatinfo`).
        When supplied:

        - Order guess positions are computed from ``flatinfo.edge_coeffs``
          and ``flatinfo.xranges`` (IDL ``mc_readflatinfo`` output).
        - ``intensity_fraction``, ``com_half_width``, ``slit_height_range``,
          ``ybuffer``, and ``poly_degree`` are overridden from the
          corresponding ``flatinfo`` fields.
        - Step-based column sampling uses ``flatinfo.step``.

        When ``None`` (default), *guess_rows* must be provided.
    n_sample_cols : int, default 40
        Number of evenly-spaced sample columns (used only when ``flatinfo``
        is ``None`` or when ``flatinfo.step`` is 0).
    poly_degree : int, default 3
        Degree of the polynomial fitted to the edges (overridden by
        ``flatinfo.edge_degree`` when *flatinfo* is supplied).
    seed_col : int, optional
        Seed column for explicit-guess mode.  Defaults to mid-point of
        *col_range*.  Ignored when *flatinfo* is supplied.
    col_range : tuple of int (col_start, col_end), optional
        Column range over which tracing is performed.  Defaults to the
        full detector width ``(0, ncols - 1)``.
    guess_rows : list of float, optional
        One approximate centre-row estimate per order, measured at *seed_col*
        (or at the midpoint of *col_range* when *seed_col* is ``None``).
        Used when *flatinfo* is ``None``.  This is the Python equivalent of
        IDL's ``guesspos[1, i]`` — the y-coordinates that ``mc_findorders``
        receives from ``mc_adjustguesspos``.  Required when *flatinfo* is
        not provided.
    max_shift : float, default 15.0
        Retained for API compatibility; not used in the IDL tracking logic.
    sigma_clip : float, default 3.0
        Robust-fit rejection threshold (IDL ``mc_robustpoly1d`` ``thresh``
        argument).
    intensity_fraction : float, default 0.85
        Fraction of the centre flux used to locate the order edge (IDL
        ``frac`` / ``FLATFRAC``).  Overridden by ``flatinfo.flat_fraction``
        when *flatinfo* is supplied.
    com_half_width : int, default 2
        Half-width of the Sobel-image COM window in pixels (IDL
        ``halfwin = round(comwidth/2)``).  Overridden by
        ``round(flatinfo.comm_window / 2)`` when *flatinfo* is supplied.
    slit_height_range : tuple of float (min, max), optional
        Allowed slit-height range in pixels (IDL ``slith_pix``).
        Overridden by ``flatinfo.slit_range_pixels`` when *flatinfo* is
        supplied.  ``None`` disables the check.
    ybuffer : int, default 1
        Detector-edge buffer in rows (IDL ``bufpix``).  Overridden by
        ``flatinfo.ybuffer`` when *flatinfo* is supplied.

    Returns
    -------
    :class:`FlatOrderTrace`
        Tracing result with per-order bottom/top edge polynomial
        coefficients, derived centre coefficients, sampled positions,
        RMS values, and half-width estimates.

    Raises
    ------
    ValueError
        If *flat_files* is empty, or if neither *flatinfo* nor *guess_rows*
        is provided.
    """
    if not flat_files:
        raise ValueError("flat_files must not be empty.")

    # ------------------------------------------------------------------
    # 1. Load and median-combine
    # ------------------------------------------------------------------
    flat = load_and_combine_flats(flat_files)
    nrows, ncols = flat.shape

    # ------------------------------------------------------------------
    # 2. Override tracing parameters from flatinfo (IDL mc_readflatinfo path)
    # ------------------------------------------------------------------
    if flatinfo is not None:
        # Use flatinfo fields where they override keyword arguments.
        if flatinfo.edge_degree is not None:
            poly_degree = int(flatinfo.edge_degree)
        intensity_fraction = float(flatinfo.flat_fraction)
        com_half_width = int(round(flatinfo.comm_window / 2))
        slit_height_range = (
            float(flatinfo.slit_range_pixels[0]),
            float(flatinfo.slit_range_pixels[1]),
        )
        # Use ybuffer from flatinfo if present (FlatInfo may not have this field
        # yet; fall back to 1 to match IDL's typical value).
        ybuffer = max(1, int(getattr(flatinfo, "ybuffer", 1)))

        logger.info(
            "Using flatinfo parameters: poly_degree=%d, intensity_fraction=%.3f, "
            "com_half_width=%d, slit_height_range=%s, ybuffer=%d",
            poly_degree, intensity_fraction, com_half_width, slit_height_range, ybuffer,
        )

    # ------------------------------------------------------------------
    # 3. Resolve column range
    # ------------------------------------------------------------------
    if col_range is None:
        col_lo, col_hi = 0, ncols - 1
    else:
        col_lo, col_hi = int(col_range[0]), int(col_range[1])

    # ------------------------------------------------------------------
    # 4. Compute guess positions and build per-order sample column arrays
    # ------------------------------------------------------------------
    if flatinfo is not None and flatinfo.edge_coeffs is not None and flatinfo.xranges is not None:
        # IDL path: adjust guess positions via cross-correlation (mc_adjustguesspos),
        # then build per-order sample columns respecting the adjusted xranges.
        can_adjust = (
            flatinfo.omask is not None
            and flatinfo.ycororder is not None
            and flatinfo.ycororder in flatinfo.orders
        )
        if can_adjust:
            guess_positions, guess_xranges = _adjust_guess_positions(
                flatinfo.edge_coeffs,
                flatinfo.xranges,
                flat,
                flatinfo.omask,
                flatinfo.orders,
                flatinfo.ycororder,
                ybuffer,
            )
            logger.info(
                "Flatinfo mode: %d orders, guess positions adjusted via "
                "cross-correlation (ycororder=%d)",
                len(guess_positions), flatinfo.ycororder,
            )
        else:
            # DEFAULT branch: no omask or ycororder available; use reference
            # positions unchanged (IDL mc_adjustguesspos /DEFAULT equivalent).
            guess_positions, guess_xranges = _compute_guess_positions_from_flatinfo(
                flatinfo.edge_coeffs, flatinfo.xranges
            )
            logger.info(
                "Flatinfo mode: %d orders, guess positions from stored "
                "edgecoeffs (no cross-correlation: omask or ycororder absent)",
                len(guess_positions),
            )

        n_orders = len(guess_positions)

        # seed_col: representative column used for logging
        seed_col = int(np.round(np.mean(guess_xranges[:, 0] + guess_xranges[:, 1]) / 2.0))
        seed_peaks = np.array([gp[1] for gp in guess_positions])

        # Build per-order sample column arrays (IDL sranges[*,i]).
        step = int(getattr(flatinfo, "step", 0))
        order_sample_cols_list: list[npt.NDArray] = []
        order_seed_indices: list[int] = []
        for i in range(n_orders):
            x0_i = max(col_lo, int(guess_xranges[i, 0]))
            x1_i = min(col_hi, int(guess_xranges[i, 1]))
            if step > 0:
                starts_i = x0_i + step - 1
                stops_i = x1_i - step + 1
                s_cols_i = np.arange(starts_i, stops_i + 1, step)
            else:
                s_cols_i = np.round(np.linspace(x0_i, x1_i, n_sample_cols)).astype(int)
            s_cols_i = np.unique(np.clip(s_cols_i, x0_i, x1_i))
            order_sample_cols_list.append(s_cols_i)
            seed_idx_i = int(np.argmin(np.abs(s_cols_i - guess_positions[i][0])))
            order_seed_indices.append(seed_idx_i)

        # Store the adjusted per-order xranges for downstream output
        traced_xranges = guess_xranges.copy()

    else:
        # Explicit-guess path (no flatinfo).
        # IDL mc_findorders always requires guesspos from mc_adjustguesspos;
        # there is no auto-detection mode in the IDL algorithm.
        if guess_rows is None:
            raise ValueError(
                "Either 'flatinfo' or 'guess_rows' must be provided.  "
                "IDL mc_findorders always requires explicit guess positions "
                "(guesspos[1,i] = y-centre of each order); provide them via "
                "guess_rows=[row0, row1, ...] or supply a flatinfo object."
            )

        seed_peaks = np.asarray(guess_rows, dtype=float)
        if len(seed_peaks) == 0:
            raise ValueError("guess_rows must not be empty.")

        n_orders = len(seed_peaks)
        guess_positions = None
        traced_xranges = None

        # Resolve seed column (IDL: tabinv(scols, guesspos[0,i], idx)).
        if seed_col is None:
            seed_col = (col_lo + col_hi) // 2
        else:
            seed_col = max(col_lo, min(col_hi, int(seed_col)))

        # All orders share the same column grid (no per-order xranges available).
        # IDL: scols = findgen(fix((stops-starts)/step)+1)*step + starts
        shared_cols = np.round(np.linspace(col_lo, col_hi, n_sample_cols)).astype(int)
        shared_cols = np.unique(shared_cols)
        seed_idx = int(np.argmin(np.abs(shared_cols - seed_col)))
        order_sample_cols_list = [shared_cols.copy() for _ in range(n_orders)]
        order_seed_indices = [seed_idx] * n_orders

        logger.info(
            "Explicit-guess mode: %d order seeds at column %d: rows %s",
            n_orders, seed_col, seed_peaks.tolist(),
        )

    # ------------------------------------------------------------------
    # 5. Trace each order on its own per-order sample column array.
    #    Build OrderTraceSamples — the authoritative per-order representation.
    #    (IDL mc_findorders inner loop with per-order sranges[*,i].)
    # ------------------------------------------------------------------
    sflat = _compute_sobel_magnitude(flat)

    order_samples: list[OrderTraceSamples] = []
    for i in range(n_orders):
        s_cols_i = order_sample_cols_list[i]
        seed_idx_i = order_seed_indices[i]
        y_seed_i = float(seed_peaks[i])

        cen_i, bot_i, top_i = _trace_single_order(
            flat, sflat, s_cols_i, seed_idx_i, y_seed_i, poly_degree,
            ybuffer=ybuffer,
            intensity_fraction=intensity_fraction,
            com_half_width=com_half_width,
            slit_height_range=slit_height_range,
        )

        if traced_xranges is not None:
            x_start_i = int(traced_xranges[i, 0])
            x_end_i = int(traced_xranges[i, 1])
        else:
            x_start_i = int(s_cols_i[0])
            x_end_i = int(s_cols_i[-1])

        order_samples.append(OrderTraceSamples(
            order_index=i,
            sample_cols=s_cols_i,
            center_rows=cen_i,
            bot_rows=bot_i,
            top_rows=top_i,
            x_start=x_start_i,
            x_end=x_end_i,
        ))

    # ------------------------------------------------------------------
    # 5a. Validate order_samples — assert internal consistency invariants
    #     before any downstream computation uses them.
    #
    #     NOTE: the xrange check below (sample_cols within [x_start, x_end])
    #     is a PRE-FIT invariant only.  At construction time (step 5),
    #     sample_cols is built from the initial input xranges so it lies
    #     within [x_start, x_end] by design.  After step 6a (post-fit
    #     xrange restriction), x_start/x_end may be narrowed inward, making
    #     some sample_cols entries legitimately outside [x_start, x_end].
    #     Do NOT re-run this check after step 6a.
    # ------------------------------------------------------------------
    if len(order_samples) != n_orders:
        raise RuntimeError(
            f"Internal consistency error after per-order tracing: "
            f"order_samples length {len(order_samples)} != n_orders {n_orders}"
        )
    for os_i in order_samples:
        n = len(os_i.sample_cols)
        if n == 0:
            raise RuntimeError(f"Order {os_i.order_index}: empty sample_cols array")
        if os_i.center_rows.shape != (n,) or os_i.bot_rows.shape != (n,) or os_i.top_rows.shape != (n,):
            raise RuntimeError(
                f"Order {os_i.order_index}: traced array shapes inconsistent with "
                f"sample_cols length {n}"
            )
        if n > 1 and not np.all(np.diff(os_i.sample_cols) > 0):
            raise RuntimeError(
                f"Order {os_i.order_index}: sample_cols is not strictly increasing"
            )
        if os_i.sample_cols[0] < os_i.x_start or os_i.sample_cols[-1] > os_i.x_end:
            raise RuntimeError(
                f"Order {os_i.order_index}: sample_cols [{os_i.sample_cols[0]}, "
                f"{os_i.sample_cols[-1]}] outside xrange [{os_i.x_start}, {os_i.x_end}]"
            )

    # ------------------------------------------------------------------
    # 6. Fit robust polynomials to bottom and top edges per order,
    #    using each order's own sample column array.
    #    (IDL: mc_robustpoly1d on edges[*,0] and edges[*,1])
    # ------------------------------------------------------------------
    bot_poly_coeffs = np.zeros((n_orders, poly_degree + 1))
    top_poly_coeffs = np.zeros((n_orders, poly_degree + 1))
    center_poly_coeffs = np.zeros((n_orders, poly_degree + 1))
    fit_rms = np.full(n_orders, np.nan)

    for i, os_i in enumerate(order_samples):
        good_bot = np.isfinite(os_i.bot_rows)
        good_top = np.isfinite(os_i.top_rows)
        good_cen = np.isfinite(os_i.center_rows)
        n_min = poly_degree + 1

        if good_bot.sum() >= n_min:
            b_coeffs, _ = _fit_poly_robust(
                os_i.sample_cols[good_bot].astype(float),
                os_i.bot_rows[good_bot],
                poly_degree, sigma_clip,
            )
            bot_poly_coeffs[i] = b_coeffs
        else:
            logger.warning(
                "Order %d: only %d valid bottom-edge points; "
                "polynomial fit skipped.", i, good_bot.sum(),
            )
            bot_poly_coeffs[i][:] = np.nan

        if good_top.sum() >= n_min:
            t_coeffs, _ = _fit_poly_robust(
                os_i.sample_cols[good_top].astype(float),
                os_i.top_rows[good_top],
                poly_degree, sigma_clip,
            )
            top_poly_coeffs[i] = t_coeffs
        else:
            logger.warning(
                "Order %d: only %d valid top-edge points; "
                "polynomial fit skipped.", i, good_top.sum(),
            )
            top_poly_coeffs[i][:] = np.nan

        center_poly_coeffs[i] = (bot_poly_coeffs[i] + top_poly_coeffs[i]) / 2.0

        if good_cen.sum() >= n_min:
            pred = np.polynomial.polynomial.polyval(
                os_i.sample_cols[good_cen].astype(float), center_poly_coeffs[i]
            )
            fit_rms[i] = float(np.std(os_i.center_rows[good_cen] - pred))

    # ------------------------------------------------------------------
    # 6a. Update per-order xranges from fitted edge polynomials.
    #     IDL mc_findorders (verbatim port of the end-of-order-loop block):
    #
    #       x = findgen(stop-start+1)+start   ; all integer cols in srange
    #       bot = poly(x, edgecoeffs[*,0,i])
    #       top = poly(x, edgecoeffs[*,1,i])
    #       z = where(top gt 0.0 and top lt nrows-1
    #                 and bot gt 0 and bot lt nrows-1)
    #       xranges[*,i] = [min(x[z],MAX=max), max]
    #
    #     i.e. restrict xrange to columns where BOTH fitted edge polynomials
    #     evaluate to strictly valid pixel positions (inside [0, nrows-1]).
    #     This prevents a pathological polynomial (one that shoots outside
    #     the detector at one end of the input xrange) from being applied
    #     over that extrapolated, unsupported region.
    #
    #     Python mismatch that this corrects: before this step the Python
    #     code carried the input sranges straight through to order_xranges
    #     without ever evaluating whether the fitted polynomial remained
    #     within the detector.  IDL always performs this restriction.
    # ------------------------------------------------------------------
    for i, os_i in enumerate(order_samples):
        b_coeffs = bot_poly_coeffs[i]
        t_coeffs = top_poly_coeffs[i]
        if np.any(np.isnan(b_coeffs)) or np.any(np.isnan(t_coeffs)):
            # Fit failed — keep original xrange (degenerate fallback,
            # matching IDL's implicit cancel-on-failure behaviour).
            logger.warning(
                "Order %d: edge fit has NaN coefficients; xrange not updated.", i,
            )
            continue
        # Evaluate on every integer column in the order's current xrange
        # (IDL: x = findgen(stop-start+1)+start).
        x_dense = np.arange(os_i.x_start, os_i.x_end + 1, dtype=float)
        bot_vals = np.polynomial.polynomial.polyval(x_dense, b_coeffs)
        top_vals = np.polynomial.polynomial.polyval(x_dense, t_coeffs)
        # IDL condition: top gt 0.0 and top lt nrows-1 and bot gt 0 and bot lt nrows-1
        valid_mask = (
            (bot_vals > 0.0) & (bot_vals < float(nrows - 1)) &
            (top_vals > 0.0) & (top_vals < float(nrows - 1))
        )
        valid_x = x_dense[valid_mask].astype(int)
        if len(valid_x) > 0:
            os_i.x_start = int(valid_x[0])
            os_i.x_end = int(valid_x[-1])
        else:
            # No valid columns — set degenerate range matching IDL's
            # initialisation (xranges[0,i] = sranges[0,i], xranges[1,i] = sranges[0,i]).
            logger.warning(
                "Order %d: fitted edge polynomials evaluate outside detector in "
                "all columns [%d, %d]; setting degenerate xrange.",
                i, os_i.x_start, os_i.x_end,
            )
            os_i.x_end = os_i.x_start  # degenerate: start == end

    # ------------------------------------------------------------------
    # 7. Estimate order half-widths from traced edge separation.
    #    IDL approach: use mean (top − bottom) separation per order.
    #    This is correct for all paths (flatinfo and explicit-guess).
    # ------------------------------------------------------------------
    half_width_rows = np.zeros(n_orders, dtype=float)
    for i, os_i in enumerate(order_samples):
        sep = os_i.top_rows - os_i.bot_rows
        valid = np.isfinite(sep) & (sep > 0)
        if valid.sum() > 0:
            half_width_rows[i] = float(np.mean(sep[valid]) / 2.0)
        else:
            half_width_rows[i] = 0.0

    # ------------------------------------------------------------------
    # 8. Build legacy compatibility arrays: union sample_cols + 2D center_rows
    #    These are derived from order_samples for callers that still use the
    #    shared-grid API.  They are NOT used internally for polynomial fitting,
    #    edge logic, or geometry construction — only for backward compatibility
    #    and QA polynomial evaluation (see step 9).
    # ------------------------------------------------------------------
    union_cols = np.unique(np.concatenate([os_i.sample_cols for os_i in order_samples]))
    compat_center_rows = np.full((n_orders, len(union_cols)), np.nan)
    for i, os_i in enumerate(order_samples):
        for j, col in enumerate(os_i.sample_cols):
            # Only copy values within the UPDATED (post-fit restricted) xrange.
            # Columns outside [x_start, x_end] correspond to extrapolated
            # polynomial territory and must remain NaN in the compat arrays.
            if not (os_i.x_start <= col <= os_i.x_end):
                continue
            k = int(np.searchsorted(union_cols, col))
            if k < len(union_cols) and union_cols[k] == col:
                compat_center_rows[i, k] = os_i.center_rows[j]

    # ------------------------------------------------------------------
    # 8a. Validate compatibility arrays — internal consistency invariants.
    #     These checks guard against bugs in the compat-array construction
    #     above; they do NOT validate user inputs.
    # ------------------------------------------------------------------
    if compat_center_rows.shape != (n_orders, len(union_cols)):
        raise RuntimeError(
            f"compat_center_rows shape {compat_center_rows.shape} does not match "
            f"(n_orders={n_orders}, len(union_cols)={len(union_cols)})"
        )
    if len(union_cols) > 1 and not np.all(np.diff(union_cols) > 0):
        bad_indices = np.where(np.diff(union_cols) <= 0)[0]
        raise RuntimeError(
            "union_cols is not strictly increasing after np.unique — "
            "this should never happen and indicates a construction bug. "
            f"First violation at indices {bad_indices.tolist()}: "
            f"values {union_cols[bad_indices].tolist()} -> {union_cols[bad_indices + 1].tolist()}"
        )
    # Every finite entry in compat_center_rows must correspond to a column
    # that actually exists in that order's own order_samples[i].sample_cols
    # AND lies within the post-fit restricted [x_start, x_end].
    for i, os_i in enumerate(order_samples):
        order_col_set = set(os_i.sample_cols.tolist())
        for k, col in enumerate(union_cols):
            val = compat_center_rows[i, k]
            if np.isfinite(val) and col not in order_col_set:
                raise RuntimeError(
                    f"compat_center_rows[{i}, {k}] is finite but column {col} "
                    f"is not in order_samples[{i}].sample_cols — construction bug."
                )
            if np.isfinite(val) and (col < os_i.x_start or col > os_i.x_end):
                raise RuntimeError(
                    f"compat_center_rows[{i}, {k}] is finite but column {col} "
                    f"is outside post-fit xrange [{os_i.x_start}, {os_i.x_end}] "
                    f"for order {i} — construction bug."
                )

    # ------------------------------------------------------------------
    # 9. Compute per-order QA metrics
    #    Build traced_xranges_out first so we can pass it to
    #    _compute_order_trace_stats, which will restrict polynomial
    #    evaluation to each order's valid column range.
    # ------------------------------------------------------------------
    traced_xranges_out = np.array(
        [[os_i.x_start, os_i.x_end] for os_i in order_samples], dtype=int
    )
    order_stats = _compute_order_trace_stats(
        center_poly_coeffs, fit_rms, union_cols,
        order_xranges=traced_xranges_out,
    )
    _log_trace_qa_summary(order_stats)

    logger.info(
        "Order tracing complete: %d orders, median RMS = %.2f px",
        n_orders,
        float(np.nanmedian(fit_rms)),
    )

    return FlatOrderTrace(
        n_orders=n_orders,
        order_samples=order_samples,
        sample_cols=union_cols,
        center_rows=compat_center_rows,
        center_poly_coeffs=center_poly_coeffs,
        fit_rms=fit_rms,
        half_width_rows=half_width_rows,
        poly_degree=poly_degree,
        seed_col=seed_col,
        bot_poly_coeffs=bot_poly_coeffs,
        top_poly_coeffs=top_poly_coeffs,
        order_xranges=traced_xranges_out,
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


def _adjust_guess_positions(
    edge_coeffs: npt.NDArray,
    xranges: npt.NDArray,
    flat: npt.NDArray,
    omask: npt.NDArray,
    orders: list[int],
    ycororder: int,
    ybuffer: int,
) -> tuple[list[tuple[float, float]], npt.NDArray]:
    """Adjust stored guess positions via cross-correlation.

    This is a Python port of the IDL ``mc_adjustguesspos`` procedure from
    Spextool.  It computes the vertical shift between the reference order
    mask stored in the flatinfo FITS file and the raw flat being calibrated,
    then subtracts that shift from the reference guess positions and
    recomputes the per-order valid column ranges.

    **Algorithm (IDL ``mc_adjustguesspos`` transliteration):**

    1. Compute initial guess positions from the stored ``edge_coeffs`` and
       ``xranges`` (midpoint x of each order; midpoint y between bot/top
       edges at that x).  [IDL lines: ``guesspos[*,i]``]

    2. Isolate ``ycororder`` in the mask (set all other order pixels to 0).

    3. Clip a sub-image of ``flat`` and the binary mask for ``ycororder``
       spanning ``[min(botedge) − slith_pix, max(topedge) + slith_pix]``
       rows.

    4. Sweep ``nshifts = int(slith_pix * 1.8) + 1`` integer vertical shifts
       centred on zero.  For each shift *s*, compute the overlap integral
       ``sum(flat_sub * roll(mask_sub, s))``.  The shift that maximises this
       sum is the vertical offset of the raw flat relative to the reference
       mask.

    5. Subtract the detected shift from all guess y-positions and recompute
       per-order valid column ranges (columns where both edges, shifted by
       the offset, remain within ``[ybuffer, nrows − ybuffer − 1]``).

    Parameters
    ----------
    edge_coeffs : ndarray, shape (n_orders, 2, n_terms)
        Reference edge polynomial coefficients (``FlatInfo.edge_coeffs``).
    xranges : ndarray, shape (n_orders, 2)
        Reference per-order column ranges (``FlatInfo.xranges``).
    flat : ndarray, shape (ncols, nrows) in IDL convention, (nrows, ncols) here
        Raw flat-field image (not yet Sobel-enhanced).  The array is in
        Python (row-major, rows=0 axis) convention.
    omask : ndarray, shape (nrows, ncols)
        Reference order mask (integer dtype; zero = inter-order).
    orders : list of int
        Echelle order numbers, in the same order as ``edge_coeffs``.
    ycororder : int
        Echelle order number to use for the cross-correlation.
    ybuffer : int
        Detector-edge buffer in rows.

    Returns
    -------
    guess_positions : list of (x_guess, y_guess) tuples
        Adjusted guess positions, one per order.
    new_xranges : ndarray, shape (n_orders, 2)
        Adjusted per-order valid column ranges.

    Notes
    -----
    IDL uses column-major arrays (IDL ``image[col, row]``), so ``xranges``
    refers to column indices and ``omask[col,row]``.  The Python arrays are
    row-major (``flat[row, col]``), which is handled transparently because
    the cross-correlation is purely row-based and the poly evaluations use
    column as the independent variable.

    The IDL ``nshifts = slith_pix * 1.8 + 1`` expression (IDL 2019-04-14
    change to avoid picking up adjacent orders).
    """
    nrows, ncols = flat.shape
    n_orders = len(orders)
    polyval = np.polynomial.polynomial.polyval

    # ------------------------------------------------------------------ #
    # 1. Compute base guess positions from stored edge_coeffs + xranges    #
    #    IDL: for i = 0,norders-1 do guesspos[*,i] = [x, (bot+top)/2]     #
    # ------------------------------------------------------------------ #
    guesspos_y = np.zeros(n_orders, dtype=float)
    guesspos_x = np.zeros(n_orders, dtype=float)
    for i in range(n_orders):
        x0, x1 = float(xranges[i, 0]), float(xranges[i, 1])
        x_guess = (x0 + x1) / 2.0
        bot_y = float(polyval(x_guess, edge_coeffs[i, 0, :]))
        top_y = float(polyval(x_guess, edge_coeffs[i, 1, :]))
        guesspos_x[i] = x_guess
        guesspos_y[i] = (bot_y + top_y) / 2.0

    # ------------------------------------------------------------------ #
    # 2. Isolate ycororder in the mask                                      #
    #    IDL: z = where(omask ne ycororder); omask[z]=0; omask[good]=1      #
    # ------------------------------------------------------------------ #
    cor_idx = orders.index(ycororder)
    binary_mask = (omask == ycororder).astype(np.float32)  # 0/1

    # ------------------------------------------------------------------ #
    # 3. Clip sub-image for ycororder                                       #
    #    IDL: find max slit height for ycororder; clip flat/mask            #
    # ------------------------------------------------------------------ #
    x0_c, x1_c = int(xranges[cor_idx, 0]), int(xranges[cor_idx, 1])
    cols_c = np.arange(x0_c, x1_c + 1)
    bot_edge_c = polyval(cols_c.astype(float), edge_coeffs[cor_idx, 0, :])
    top_edge_c = polyval(cols_c.astype(float), edge_coeffs[cor_idx, 1, :])
    slith_pix = int(np.ceil(np.max(top_edge_c - bot_edge_c)))

    botidx = max(0, int(np.round(np.min(bot_edge_c))) - slith_pix)
    topidx = min(nrows - 1, int(np.round(np.max(top_edge_c))) + slith_pix)

    subflat = flat[botidx: topidx + 1, :].astype(np.float32)    # (sub_rows, ncols)
    submask = binary_mask[botidx: topidx + 1, :]                  # (sub_rows, ncols)
    sub_nrows = subflat.shape[0]

    # ------------------------------------------------------------------ #
    # 4. Cross-correlation sweep                                            #
    #    IDL: nshifts = slith_pix*1.8+1; shifts = indgen(nshifts)-nshifts/2#
    # ------------------------------------------------------------------ #
    nshifts = int(slith_pix * 1.8) + 1
    half = nshifts // 2
    shifts = np.arange(nshifts) - half   # centred integer shifts

    overlap = np.zeros(nshifts, dtype=float)
    for k, s in enumerate(shifts):
        # IDL shift convention:
        #   hbot = -s > 0  (max(-s, 0))
        #   htop = (sub_nrows-1-s) < (sub_nrows-1)
        #   mbot = s > 0
        #   mtop = (sub_nrows-1+s) < (sub_nrows-1)
        hbot = max(-s, 0)
        htop = min(sub_nrows - 1 - s, sub_nrows - 1)
        mbot = max(s, 0)
        mtop = min(sub_nrows - 1 + s, sub_nrows - 1)
        if htop < hbot or mtop < mbot:
            continue
        overlap[k] = float(np.sum(subflat[hbot: htop + 1, :] * submask[mbot: mtop + 1, :]))

    offset = int(shifts[int(np.argmax(overlap))])
    logger.debug(
        "_adjust_guess_positions: ycororder=%d, slith_pix=%d, "
        "nshifts=%d, detected offset=%d rows",
        ycororder, slith_pix, nshifts, offset,
    )

    # ------------------------------------------------------------------ #
    # 5. Subtract offset from guess y-positions                            #
    #    IDL: guesspos[1,*] = guesspos[1,*] - offset                       #
    # ------------------------------------------------------------------ #
    guesspos_y -= offset

    # ------------------------------------------------------------------ #
    # 6. Recompute per-order xranges with shifted edges                    #
    #    IDL: for each order: botedge-offset, topedge-offset;              #
    #         keep cols where both within [ybuffer, nrows-ybuffer-1]       #
    # ------------------------------------------------------------------ #
    new_xranges = np.zeros_like(xranges)
    for i in range(n_orders):
        x0, x1 = int(xranges[i, 0]), int(xranges[i, 1])
        cols_i = np.arange(x0, x1 + 1)
        bot_i = polyval(cols_i.astype(float), edge_coeffs[i, 0, :]) - offset
        top_i = polyval(cols_i.astype(float), edge_coeffs[i, 1, :]) - offset

        valid = np.where(
            (bot_i > ybuffer - 1) & (top_i < nrows - ybuffer - 1)
        )[0]

        if len(valid) == 0:
            # All columns fall outside detector; keep original range.
            logger.warning(
                "_adjust_guess_positions: order index %d: no valid columns "
                "after offset=%d; keeping original xrange.", i, offset,
            )
            new_xranges[i] = xranges[i]
        else:
            new_xranges[i, 0] = cols_i[valid[0]]
            new_xranges[i, 1] = cols_i[valid[-1]]

    guess_positions = [(float(guesspos_x[i]), float(guesspos_y[i])) for i in range(n_orders)]
    return guess_positions, new_xranges


def _compute_guess_positions_from_flatinfo(
    edge_coeffs: npt.NDArray,
    xranges: npt.NDArray,
) -> tuple[list[tuple[float, float]], npt.NDArray]:
    """Compute IDL-style guess positions from stored edge polynomials.

    This is the ``DEFAULT=True`` branch of IDL ``mc_adjustguesspos``:
    compute initial guess positions from stored reference coefficients
    without any cross-correlation shift adjustment.

    Parameters
    ----------
    edge_coeffs : ndarray, shape (n_orders, 2, n_terms)
        Edge polynomial coefficients from ``FlatInfo.edge_coeffs``.
        ``edge_coeffs[i, 0, :]`` = bottom edge, ``[i, 1, :]`` = top edge.
    xranges : ndarray, shape (n_orders, 2)
        Per-order column ranges ``[x_start, x_end]``.

    Returns
    -------
    guess_positions : list of (x_guess, y_guess) tuples
        One entry per order.
    xranges : ndarray
        The input xranges array (returned unchanged).
    """
    guess_positions = []
    for i in range(len(edge_coeffs)):
        x0, x1 = float(xranges[i, 0]), float(xranges[i, 1])
        x_guess = (x0 + x1) / 2.0
        bot_y = float(np.polynomial.polynomial.polyval(x_guess, edge_coeffs[i, 0, :]))
        top_y = float(np.polynomial.polynomial.polyval(x_guess, edge_coeffs[i, 1, :]))
        y_guess = (bot_y + top_y) / 2.0
        guess_positions.append((x_guess, y_guess))
    return guess_positions, xranges


def _trace_single_order(
    flat: npt.NDArray,
    sflat: npt.NDArray,
    s_cols: npt.NDArray,
    seed_idx: int,
    y_seed: float,
    poly_degree: int,
    *,
    ybuffer: int = 1,
    intensity_fraction: float = 0.85,
    com_half_width: int = 2,
    slit_height_range: tuple[float, float] | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Trace one order on its own per-order sample column array.

    This is the IDL ``mc_findorders`` inner loop for a single order, operating
    on the per-order column range (``sranges[*,i]`` in IDL), not a shared
    union of all-order columns.

    Parameters
    ----------
    flat : ndarray, shape (nrows, ncols)
        Flat-field image.
    sflat : ndarray, shape (nrows, ncols)
        Sobel gradient magnitude image (pre-computed).
    s_cols : ndarray of int, shape (n_samp,)
        Per-order sample column indices (IDL ``sranges`` for this order).
    seed_idx : int
        Index into *s_cols* of the seed/guess column.
    y_seed : float
        Seed row (order centre estimate at the seed column).
    poly_degree : int
        Polynomial degree for centre-extrapolation fits.
    ybuffer, intensity_fraction, com_half_width, slit_height_range :
        Passed through to edge-finding helpers.

    Returns
    -------
    cen : ndarray of float, shape (n_samp,)
    bot : ndarray of float, shape (n_samp,)
    top : ndarray of float, shape (n_samp,)
        Traced centre, bottom-edge, and top-edge positions at each column in
        *s_cols*.  NaN for columns where tracing was not reliable.
    """
    n_samp = len(s_cols)
    nrows = flat.shape[0]
    row = np.arange(nrows, dtype=float)
    pred_deg = max(1, poly_degree - 2)

    cen = np.full(n_samp, np.nan)
    bot = np.full(n_samp, np.nan)
    top = np.full(n_samp, np.nan)

    # Initialise centre window around the seed index.
    lo = max(0, seed_idx - poly_degree)
    hi = min(n_samp - 1, seed_idx + poly_degree)
    cen[lo: hi + 1] = y_seed

    # Left sweep (seed_idx → 0, inclusive)
    do_top = True
    do_bot = True
    for k in range(seed_idx, -1, -1):
        fcol = flat[:, s_cols[k]]
        rcol = sflat[:, s_cols[k]]
        y_guess_f, com_top, com_bot = _predict_and_find_edges(
            cen, s_cols, k, pred_deg, fcol, rcol, row,
            nrows, ybuffer, intensity_fraction, com_half_width,
            do_top, do_bot,
        )
        cont = _update_centre_and_flags(
            cen, bot, top, k, y_guess_f, com_top, com_bot,
            nrows, ybuffer, slit_height_range,
            do_top, do_bot,
        )
        if cont == "break":
            break
        if cont == "continue":
            continue
        do_top, do_bot = cont

    # Right sweep (seed_idx+1 → n_samp-1)
    do_top = True
    do_bot = True
    for k in range(seed_idx + 1, n_samp):
        fcol = flat[:, s_cols[k]]
        rcol = sflat[:, s_cols[k]]
        y_guess_f, com_top, com_bot = _predict_and_find_edges(
            cen, s_cols, k, pred_deg, fcol, rcol, row,
            nrows, ybuffer, intensity_fraction, com_half_width,
            do_top, do_bot,
        )
        cont = _update_centre_and_flags(
            cen, bot, top, k, y_guess_f, com_top, com_bot,
            nrows, ybuffer, slit_height_range,
            do_top, do_bot,
        )
        if cont == "break":
            break
        if cont == "continue":
            continue
        do_top, do_bot = cont

    return cen, bot, top


def _trace_all_orders(
    flat: npt.NDArray,
    sample_cols: npt.NDArray,
    order_seed_indices: list[int],
    seed_peaks: npt.NDArray,
    center_rows: npt.NDArray,
    bot_rows: npt.NDArray,
    top_rows: npt.NDArray,
    poly_degree: int,
    *,
    ybuffer: int = 1,
    intensity_fraction: float = 0.85,
    com_half_width: int = 2,
    slit_height_range: tuple[float, float] | None = None,
    order_sample_cols_list: list[npt.NDArray] | None = None,
) -> None:
    """Trace all orders using the IDL ``mc_findorders`` algorithm.

    When *order_sample_cols_list* is provided, each order is traced on its own
    per-order sample column array (the IDL ``sranges`` approach).  Results are
    mapped back into the shared *center_rows* / *bot_rows* / *top_rows* output
    arrays (indexed against *sample_cols*; NaN for columns not in an order's
    range).

    When *order_sample_cols_list* is ``None``, all orders share *sample_cols*
    directly (fallback / auto-detect mode).

    Parameters
    ----------
    flat : ndarray, shape (nrows, ncols)
        Flat-field image.
    sample_cols : ndarray of int
        Shared column index array for the output arrays.
    order_seed_indices : list of int
        Per-order index into *sample_cols* of the seed/guess column.
    seed_peaks : ndarray of float, shape (n_orders,)
        Initial order-centre row estimates at the seed column.
    center_rows : ndarray of float, shape (n_orders, n_sample)
        Centre array; filled in place.
    bot_rows : ndarray of float, shape (n_orders, n_sample)
        Bottom-edge array; filled in place.
    top_rows : ndarray of float, shape (n_orders, n_sample)
        Top-edge array; filled in place.
    poly_degree : int
        Polynomial degree for the centre-extrapolation fits.
    ybuffer, intensity_fraction, com_half_width, slit_height_range :
        Passed through to the single-order tracing helper.
    order_sample_cols_list : list of int ndarrays, optional
        Per-order sample column arrays (IDL ``sranges``).  When provided,
        each order is traced on its own column set; results are mapped back
        to the shared *sample_cols* index space.  ``None`` → all orders use
        *sample_cols* directly (auto-detect fallback).
    """
    n_orders = center_rows.shape[0]
    nrows = flat.shape[0]

    # Compute the Sobel gradient magnitude once for all orders.
    sflat = _compute_sobel_magnitude(flat)

    for order_i in range(n_orders):
        # Per-order column array: IDL sranges[*,i] approach
        if order_sample_cols_list is not None:
            s_cols_i = order_sample_cols_list[order_i]
            # Seed index within this order's own column array
            guess_col = sample_cols[order_seed_indices[order_i]]
            seed_idx_i = int(np.argmin(np.abs(s_cols_i - guess_col)))
        else:
            s_cols_i = sample_cols
            seed_idx_i = order_seed_indices[order_i]

        cen_i, bot_i, top_i = _trace_single_order(
            flat, sflat, s_cols_i, seed_idx_i, float(seed_peaks[order_i]),
            poly_degree,
            ybuffer=ybuffer,
            intensity_fraction=intensity_fraction,
            com_half_width=com_half_width,
            slit_height_range=slit_height_range,
        )

        if order_sample_cols_list is not None:
            # Map per-order results back to the shared sample_cols index space.
            # For each column in s_cols_i, find its index in sample_cols and store.
            for j, col in enumerate(s_cols_i):
                k = int(np.searchsorted(sample_cols, col))
                if k < len(sample_cols) and sample_cols[k] == col:
                    center_rows[order_i, k] = cen_i[j]
                    bot_rows[order_i, k] = bot_i[j]
                    top_rows[order_i, k] = top_i[j]
        else:
            center_rows[order_i] = cen_i
            bot_rows[order_i] = bot_i
            top_rows[order_i] = top_i


def _compute_sobel_magnitude(flat: npt.NDArray) -> npt.NDArray:
    """Return the Sobel gradient magnitude of *flat*.

    Equivalent to IDL's ``sobel()`` built-in, which the IDL ``mc_findorders``
    routine uses to enhance order edges before the centre-of-mass step.
    The image is normalised to ``[0, 1]`` before differentiation so that
    the magnitude is scale-independent (matching IDL: ``sobel(image*1000/max)``).
    """
    scl = float(np.max(flat))
    if scl == 0.0:
        return np.zeros_like(flat, dtype=float)
    img_norm = flat.astype(float) / scl
    s0 = _scipy_sobel(img_norm, axis=0)
    s1 = _scipy_sobel(img_norm, axis=1)
    return np.sqrt(s0 ** 2 + s1 ** 2)


def _predict_and_find_edges(
    cen: npt.NDArray,
    sample_cols: npt.NDArray,
    k: int,
    pred_deg: int,
    fcol: npt.NDArray,
    rcol: npt.NDArray,
    row: npt.NDArray,
    nrows: int,
    ybuffer: int,
    intensity_fraction: float,
    com_half_width: int,
    do_top: bool,
    do_bot: bool,
) -> tuple[float, float, float]:
    """Predict the centre and find top/bottom edges at column index *k*.

    Returns ``(y_guess_f, com_top, com_bot)`` where edge values are
    ``NaN`` when the corresponding edge could not be located.
    """
    # Fit centre polynomial to predict y_guess.
    valid = np.isfinite(cen)
    if valid.sum() >= pred_deg + 1:
        r = _polyfit_1d(
            sample_cols[valid].astype(float),
            cen[valid],
            pred_deg,
            justfit=True,
            silent=True,
        )
        y_guess_f = float(
            np.polynomial.polynomial.polyval(float(sample_cols[k]), r["coeffs"])
        )
    else:
        # Not enough points yet – use the nearest finite value.
        finite_vals = cen[valid]
        if len(finite_vals) > 0:
            y_guess_f = float(finite_vals[0])
        elif np.isfinite(cen[k]):
            y_guess_f = float(cen[k])
        else:
            y_guess_f = 0.0

    y_guess_f = float(np.clip(y_guess_f, ybuffer, nrows - ybuffer - 1))
    y_guess = int(round(y_guess_f))
    z_guess = float(fcol[y_guess])
    threshold = intensity_fraction * z_guess

    # Find top edge (IDL: ztop = where(fcol lt frac*z_guess and row gt y_guess)).
    if do_top:
        ztop_idx = np.where((fcol < threshold) & (row > y_guess_f))[0]
        if len(ztop_idx) > 0:
            guessyt = int(ztop_idx[0])
            bidx = max(0, guessyt - com_half_width)
            tidx = min(nrows - 1, guessyt + com_half_width)
            y_sub = row[bidx: tidx + 1]
            z_sub = rcol[bidx: tidx + 1]
            total_z = float(np.sum(z_sub))
            com_top = float(np.sum(y_sub * z_sub) / total_z) if total_z != 0.0 else np.nan
        else:
            com_top = np.nan
    else:
        com_top = np.nan

    # Find bot edge (IDL: zbot = where(fcol lt frac*z_guess and row lt y_guess)).
    if do_bot:
        zbot_idx = np.where((fcol < threshold) & (row < y_guess_f))[0]
        if len(zbot_idx) > 0:
            guessyb = int(zbot_idx[-1])
            bidx = max(0, guessyb - com_half_width)
            tidx = min(nrows - 1, guessyb + com_half_width)
            y_sub = row[bidx: tidx + 1]
            z_sub = rcol[bidx: tidx + 1]
            total_z = float(np.sum(z_sub))
            com_bot = float(np.sum(y_sub * z_sub) / total_z) if total_z != 0.0 else np.nan
        else:
            com_bot = np.nan
    else:
        com_bot = np.nan

    return y_guess_f, com_top, com_bot


def _update_centre_and_flags(
    cen: npt.NDArray,
    bot: npt.NDArray,
    top: npt.NDArray,
    k: int,
    y_guess_f: float,
    com_top: float,
    com_bot: float,
    nrows: int,
    ybuffer: int,
    slit_height_range: tuple[float, float] | None,
    do_top: bool,
    do_bot: bool,
) -> tuple[bool, bool] | str:
    """Update ``cen[k]``, ``bot[k]``, ``top[k]`` and return updated flags.

    Returns the string ``"break"`` when the sweep should stop, or
    ``"continue"`` when the column should be skipped (IDL ``goto cont1``).

    Implements the exact IDL ``mc_findorders`` update logic:

    * If both edge COMs are finite and the slit height is within range:
      store edges, update ``cen[k]``, and check boundary flags.
    * If the slit height is *outside* range: skip (``goto cont1``);
      edges remain NaN for this column.
    * If either edge is NaN: store NaN edges and fall back to ``y_guess_f``
      for the centre.
    """
    if do_top and do_bot and np.isfinite(com_bot) and np.isfinite(com_top):
        slit_h = abs(com_bot - com_top)
        height_ok = (
            slit_height_range is None
            or (slit_height_range[0] <= slit_h <= slit_height_range[1])
        )
        if height_ok:
            # IDL: edges[k,*] = [com_bot, com_top]; cen[k] = (com_bot+com_top)/2
            bot[k] = com_bot
            top[k] = com_top
            cen[k] = (com_bot + com_top) / 2.0
            # Boundary checks – stop tracking an edge that reaches the border.
            # IDL source:
            #   if com_top le bufpix or com_top ge (nrows-1-bufpix) then dotop=0
            #   if com_bot le bufpix or com_bot gt (nrows-1-bufpix) then dobot=0
            # Note the intentional asymmetry: top uses `ge` (>=) while bot uses `gt` (>).
            if com_top <= ybuffer or com_top >= nrows - 1 - ybuffer:
                do_top = False
            if com_bot <= ybuffer or com_bot > nrows - 1 - ybuffer:
                do_bot = False
            if not do_top and not do_bot:
                return "break"
            return (do_top, do_bot)
        else:
            # IDL: goto cont1 – skip column; edges remain NaN.
            return "continue"
    else:
        # Edge detection failed; IDL: edges[k,*] = [nan, nan]; cen[k] = y_guess
        bot[k] = np.nan
        top[k] = np.nan
        cen[k] = y_guess_f
        # Boundary checks on NaN edges are no-ops (np.isfinite guards).
        # Same IDL asymmetry as above: top uses `ge` (>=), bot uses `gt` (>).
        if np.isfinite(com_top) and (com_top <= ybuffer or com_top >= nrows - 1 - ybuffer):
            do_top = False
        if np.isfinite(com_bot) and (com_bot <= ybuffer or com_bot > nrows - 1 - ybuffer):
            do_bot = False
        if not do_top and not do_bot:
            return "break"
        return (do_top, do_bot)


def _fit_poly_robust(
    cols: npt.NDArray,
    rows: npt.NDArray,
    degree: int,
    thresh: float = 3.0,
) -> tuple[npt.NDArray, float]:
    """Fit a polynomial using robust least-squares (IDL ``mc_robustpoly1d``).

    Parameters
    ----------
    cols : ndarray
        Column coordinates (x values).
    rows : ndarray
        Row coordinates (y values).
    degree : int
        Polynomial degree.
    thresh : float, default 3.0
        Rejection threshold in units of residual RMS.  Corresponds to
        IDL ``mc_robustpoly1d(..., 3, 0.01, ...)``.

    Returns
    -------
    coeffs : ndarray, shape (degree + 1,)
        Polynomial coefficients in ``numpy.polynomial.polynomial`` convention.
    rms : float
        RMS of the fit residuals on the accepted (non-rejected) points.
    """
    r = _polyfit_1d(
        cols,
        rows,
        degree,
        robust={"thresh": thresh, "eps": 0.01},
        justfit=True,
        silent=True,
    )
    coeffs = r["coeffs"]
    good = r["goodbad"] == 1
    if good.sum() > 0:
        predicted = np.polynomial.polynomial.polyval(cols[good], coeffs)
        rms = float(np.std(rows[good] - predicted))
    else:
        rms = np.nan
    return coeffs, rms


def _estimate_half_widths_from_edges(
    bot_rows: npt.NDArray,
    top_rows: npt.NDArray,
    seed_idx: int,
) -> npt.NDArray:
    """Estimate per-order half-widths from the traced edge separation.

    For each order, returns half the mean (top − bottom) edge separation
    across the available sample columns near the seed index.

    Parameters
    ----------
    bot_rows : ndarray, shape (n_orders, n_sample)
        Bottom-edge row positions.
    top_rows : ndarray, shape (n_orders, n_sample)
        Top-edge row positions.
    seed_idx : int
        Index of the seed/guess sample column (used as a preference centre
        for the local estimation window).

    Returns
    -------
    ndarray of float, shape (n_orders,)
        Estimated half-widths in pixels.
    """
    n_orders = bot_rows.shape[0]
    half_widths = np.zeros(n_orders, dtype=float)

    for i in range(n_orders):
        sep = top_rows[i] - bot_rows[i]
        valid = np.isfinite(sep) & (sep > 0)
        if valid.sum() > 0:
            half_widths[i] = float(np.mean(sep[valid]) / 2.0)
        else:
            half_widths[i] = 0.0

    return half_widths



def _polyval_with_xrange(
    coeffs: npt.NDArray,
    x: npt.NDArray,
    x_start: int,
    x_end: int,
) -> npt.NDArray:
    """Evaluate a polynomial, returning NaN outside ``[x_start, x_end]``.

    Polynomials are only physically meaningful within the detector region
    where edge samples were actually collected.  This helper enforces that
    constraint so that extrapolated tails never contaminate QA stats or
    compatibility arrays.

    Parameters
    ----------
    coeffs : ndarray, shape (degree + 1,)
        Polynomial coefficients in ``numpy.polynomial.polynomial`` convention.
    x : ndarray, shape (n,)
        Column values at which to evaluate.
    x_start : int
        First valid column (inclusive).
    x_end : int
        Last valid column (inclusive).

    Returns
    -------
    ndarray, shape (n,)
        Polynomial values where ``x_start <= x <= x_end``; ``NaN`` elsewhere.
    """
    result = np.full(len(x), np.nan)
    valid = (x >= x_start) & (x <= x_end)
    if np.any(valid):
        result[valid] = np.polynomial.polynomial.polyval(x[valid], coeffs)
    return result


def _compute_order_trace_stats(
    center_poly_coeffs: npt.NDArray,
    fit_rms: npt.NDArray,
    sample_cols: npt.NDArray,
    *,
    order_xranges: npt.NDArray | None = None,
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
    order_xranges : ndarray, shape (n_orders, 2), optional
        Per-order valid column ranges ``[x_start, x_end]``.  When supplied,
        polynomial evaluation is restricted to columns inside each order's
        valid range via :func:`_polyval_with_xrange` — extrapolated tails
        outside that range contribute ``NaN`` and do not corrupt metrics.
        When ``None``, the full ``sample_cols`` grid is used for every order
        (backward-compatible behaviour).
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

    # Defensive: assert that we only perform polynomial evaluation below —
    # never indexing into raw traced-sample arrays (center_rows, bot_rows,
    # top_rows).  This guard documents the contract and will fire if the
    # function body is ever accidentally extended to use raw data.
    assert isinstance(cols, np.ndarray) and cols.ndim == 1, (
        "_compute_order_trace_stats: sample_cols must be a 1-D array used "
        "for polynomial evaluation only — do not pass raw traced-sample arrays."
    )

    # Pre-compute centre-row positions at all sample columns.
    # When order_xranges is supplied, evaluate only within [x_start, x_end]
    # for each order — extrapolated tails return NaN and do not corrupt
    # inter-order separation calculations.
    # When order_xranges is None (backward compat), use the full column grid.
    # NOTE: the only operation on `cols` throughout this function is
    # ``polyval(cols, coeffs)`` — no indexing into raw traced arrays occurs.
    center_vals = np.full((n_orders, len(cols)), np.nan)
    for i in range(n_orders):
        if np.isfinite(fit_rms[i]):
            if order_xranges is not None:
                center_vals[i] = _polyval_with_xrange(
                    center_poly_coeffs[i], cols,
                    int(order_xranges[i, 0]), int(order_xranges[i, 1]),
                )
            else:
                center_vals[i] = np.polynomial.polynomial.polyval(
                    cols, center_poly_coeffs[i]
                )

    stats_list: list[OrderTraceStats] = []

    for i in range(n_orders):
        coeffs = center_poly_coeffs[i]
        rms = float(fit_rms[i])
        fit_ok = np.isfinite(rms)

        # Per-order valid column sub-array: restrict metrics to the valid range.
        if order_xranges is not None:
            x_start_i = int(order_xranges[i, 0])
            x_end_i = int(order_xranges[i, 1])
            valid_col_mask = (cols >= x_start_i) & (cols <= x_end_i)
            cols_i = cols[valid_col_mask]
        else:
            cols_i = cols

        # ------------------------------------------------------------------
        # 1. Curvature metric: max |d²y/dx²| evaluated at valid sample columns.
        #    Uses numpy.polynomial.polynomial.polyder to differentiate.
        # ------------------------------------------------------------------
        if not fit_ok:
            curvature_metric = np.nan
        else:
            d2_coeffs = np.polynomial.polynomial.polyder(coeffs, 2)
            if d2_coeffs.size == 0 or len(cols_i) == 0:
                curvature_metric = 0.0
            else:
                d2_vals = np.abs(
                    np.polynomial.polynomial.polyval(cols_i, d2_coeffs)
                )
                curvature_metric = float(np.max(d2_vals))

        # ------------------------------------------------------------------
        # 2. Oscillation metric: peak-to-peak variation in the first
        #    derivative (slope) across the valid sample columns.  A straight
        #    trace gives 0; oscillatory or pathologically curved polynomials
        #    give large values.  This replaces the previous boolean monotonicity
        #    test, which incorrectly flagged even gently curved traces.
        # ------------------------------------------------------------------
        if not fit_ok:
            oscillation_metric = np.nan
        else:
            d1_coeffs = np.polynomial.polynomial.polyder(coeffs, 1)
            if d1_coeffs.size == 0 or len(cols_i) == 0:
                oscillation_metric = 0.0
            else:
                d1_vals = np.polynomial.polynomial.polyval(cols_i, d1_coeffs)
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


# ---------------------------------------------------------------------------
# IDL mc_findorders building-block helpers (blocks A–F)
# ---------------------------------------------------------------------------
# These functions are named, minimal Python ports of the corresponding IDL
# code blocks inside mc_findorders.pro.  They are called by _trace_single_order
# (and may be called directly by tests).
# ---------------------------------------------------------------------------


def _compute_sobel_image(image: npt.NDArray) -> npt.NDArray:
    """Return the Sobel-filtered image with IDL-matching normalisation.

    IDL block (mc_findorders.pro, line 158):
        rimage = sobel(image*1000./max(image))

    The factor of 1000 and the division by max(image) are kept explicit to
    match the IDL semantics.  The actual COM step uses relative weights so the
    absolute scale is cosmetic, but correctness requires the normalisation.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Raw flat-field image in row-major order (Python convention).

    Returns
    -------
    ndarray, shape (nrows, ncols), dtype float
        Sobel gradient magnitude of the normalised image.
    """
    max_val = float(np.max(image))
    if max_val == 0.0:
        return np.zeros_like(image, dtype=float)
    # IDL: image*1000./max(image)
    scaled = image.astype(float) * (1000.0 / max_val)
    # IDL sobel() = sqrt(Gx^2 + Gy^2) applied along both axes.
    s0 = _scipy_sobel(scaled, axis=0)
    s1 = _scipy_sobel(scaled, axis=1)
    return np.sqrt(s0 ** 2 + s1 ** 2)


def _build_sample_cols(x_lo: int, x_hi: int, step: int) -> npt.NDArray:
    """Construct the sample-column array for one order.

    IDL block (mc_findorders.pro, lines 169–171):
        starts = start + step - 1
        stops  = stop  - step + 1
        scols  = findgen(fix((stops-starts)/step)+1)*step + starts

    Parameters
    ----------
    x_lo : int
        First column of the order range (IDL ``start`` / ``sranges[0,i]``).
    x_hi : int
        Last column of the order range (IDL ``stop``  / ``sranges[1,i]``).
    step : int
        Step size in columns (IDL ``step``).

    Returns
    -------
    ndarray of int
        Sample column indices.
    """
    starts = x_lo + step - 1
    stops = x_hi - step + 1
    # IDL: findgen(fix((stops-starts)/step)+1)*step + starts
    n = int((stops - starts) / step) + 1
    return (np.arange(n, dtype=float) * step + starts).astype(int)


def _initialize_order_trace_arrays(
    sample_cols: npt.NDArray,
    guess_col: float,
    guess_row: float,
    poly_degree: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, int]:
    """Create and seed the per-order trace arrays.

    IDL block (mc_findorders.pro, lines 173–195):
        edges = replicate(!values.f_nan, nscols, 2)
        cen   = fltarr(nscols) + !values.f_nan
        tabinv, scols, guesspos[0,i], idx
        gidx  = round(idx)
        cen[(gidx-degree):(gidx+degree)] = guesspos[1,i]

    Parameters
    ----------
    sample_cols : ndarray of int, shape (n_samp,)
        Per-order sample column indices (``scols`` in IDL).
    guess_col : float
        X-coordinate of the guess position (``guesspos[0,i]``).
    guess_row : float
        Y-coordinate of the guess position (``guesspos[1,i]``).
    poly_degree : int
        Polynomial degree (IDL ``degree``).  Seeds the range
        ``[gidx-degree, gidx+degree]`` (inclusive).

    Returns
    -------
    edges : ndarray of float, shape (n_samp, 2)
        NaN-filled array for bottom (col 0) and top (col 1) edge positions.
    cen : ndarray of float, shape (n_samp,)
        NaN-filled centre array seeded with *guess_row* near *gidx*.
    bot : ndarray of float, shape (n_samp,)
        NaN-filled bottom-edge array (alias for edges[:, 0]).
    gidx : int
        Index in *sample_cols* closest to *guess_col* (IDL ``gidx``).
    """
    n_samp = len(sample_cols)
    edges = np.full((n_samp, 2), np.nan)
    cen = np.full(n_samp, np.nan)

    # IDL: tabinv(scols, guesspos[0,i], idx); gidx = round(idx)
    # tabinv finds the floating-point index by linear interpolation.
    # Equivalent: find the position in the sorted array.
    idx_f = float(np.interp(guess_col, sample_cols.astype(float),
                             np.arange(n_samp, dtype=float)))
    gidx = max(0, min(n_samp - 1, int(round(idx_f))))

    # IDL: cen[(gidx-degree):(gidx+degree)] = guesspos[1,i]
    lo = max(0, gidx - poly_degree)
    hi = min(n_samp - 1, gidx + poly_degree)
    cen[lo: hi + 1] = guess_row

    return edges, cen, edges[:, 0], gidx


def _predict_center_from_known_samples(
    sample_cols: npt.NDArray,
    center_samples: npt.NDArray,
    predict_col: float,
    poly_degree: int,
    nrows: int,
    bufpix: int,
) -> float:
    """Predict the order centre at *predict_col* from previously-traced samples.

    IDL block (mc_findorders.pro, lines 208–210):
        coeff   = mc_polyfit1d(scols, cen, 1 > (degree-2), /SILENT, /JUSTFIT)
        y_guess = bufpix > poly(scols[k], coeff) < (nrows-bufpix-1)

    Only finite entries in *center_samples* are used for the fit
    (IDL's ``mc_polyfit1d`` with ``/JUSTFIT`` ignores NaN points by design).

    Parameters
    ----------
    sample_cols : ndarray of float-or-int, shape (n_samp,)
        All sample column positions for this order.
    center_samples : ndarray of float, shape (n_samp,)
        Current centre estimates; NaN where not yet determined.
    predict_col : float
        Column at which to predict the centre (``scols[k]`` in IDL).
    poly_degree : int
        Full polynomial degree (IDL ``degree``).  The temporary fit degree is
        ``max(1, poly_degree - 2)`` (IDL: ``1 > (degree-2)``).
    nrows : int
        Number of detector rows.
    bufpix : int
        Detector-edge buffer in rows (IDL ``bufpix``).

    Returns
    -------
    float
        Predicted centre row, clamped to ``[bufpix, nrows - bufpix - 1]``.
    """
    # IDL: 1 > (degree-2)  →  max(1, degree-2)
    fit_deg = max(1, poly_degree - 2)

    valid_mask = np.isfinite(center_samples)
    n_valid = int(valid_mask.sum())

    if n_valid >= fit_deg + 1:
        r = _polyfit_1d(
            sample_cols[valid_mask].astype(float),
            center_samples[valid_mask],
            fit_deg,
            justfit=True,
            silent=True,
        )
        y_pred = float(np.polynomial.polynomial.polyval(float(predict_col), r["coeffs"]))
    elif n_valid > 0:
        # Not enough points for the requested degree; fall back to nearest finite.
        y_pred = float(center_samples[valid_mask][0])
    else:
        y_pred = 0.0

    # IDL: bufpix > y_guess < (nrows-bufpix-1)  →  clamp
    return float(np.clip(y_pred, bufpix, nrows - bufpix - 1))


def _extract_column_profiles(
    image: npt.NDArray,
    sobel_image: npt.NDArray,
    col: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Extract the flux and Sobel column profiles at *col*.

    IDL block (mc_findorders.pro, lines 205–206):
        fcol = reform( image[scols[k],*] )
        rcol = reform( rimage[scols[k],*] )

    In IDL, ``image`` is column-major, so ``image[col, *]`` yields a 1-D
    array of all rows at that column.  Python uses row-major storage, so the
    equivalent is ``image[:, col]``.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Flat-field image.
    sobel_image : ndarray, shape (nrows, ncols)
        Sobel-filtered image (from :func:`_compute_sobel_image`).
    col : int
        Column index to extract.

    Returns
    -------
    fcol : ndarray of float, shape (nrows,)
        Flux column profile (IDL ``fcol``).
    rcol : ndarray of float, shape (nrows,)
        Sobel column profile (IDL ``rcol``).
    """
    return image[:, col].astype(float), sobel_image[:, col].astype(float)


def _local_flux_at_guess(flux_col: npt.NDArray, y_guess: float) -> float:
    """Return the flux at the predicted order-centre row.

    IDL block (mc_findorders.pro, line 210):
        z_guess = fcol[y_guess]

    In IDL ``y_guess`` is already a rounded integer (from the ``bufpix >
    poly(...) < (nrows-bufpix-1)`` expression).  This function rounds to the
    nearest integer explicitly to match that behaviour.

    Parameters
    ----------
    flux_col : ndarray of float, shape (nrows,)
        Flux column profile.
    y_guess : float
        Predicted order-centre row (may be fractional).

    Returns
    -------
    float
        Flux value at the integer row nearest to *y_guess*.
    """
    return float(flux_col[int(round(y_guess))])


def _find_threshold_edge_guesses(
    flux_col: npt.NDArray,
    y_guess: float,
    frac: float,
) -> tuple[int | None, int | None]:
    """Find the first-crossing row indices above and below the centre guess.

    IDL block (mc_findorders.pro, lines 220–248):
        ; Top:
        ztop = where(fcol lt frac*z_guess and row gt y_guess, cnt)
        if cnt ne 0 then guessyt = ztop[0]
        ; Bottom:
        zbot = where(fcol lt frac*z_guess and row lt y_guess, cnt)
        if cnt ne 0 then guessyb = zbot[cnt-1]

    These are *threshold guess* rows used to anchor the Sobel COM window —
    not final edge positions.

    Parameters
    ----------
    flux_col : ndarray of float, shape (nrows,)
        Flux column profile (IDL ``fcol``).
    y_guess : float
        Predicted order-centre row (IDL ``y_guess``).
    frac : float
        Flux fraction threshold (IDL ``frac``).

    Returns
    -------
    guessyt : int or None
        Row index of the first pixel *above* ``y_guess`` where
        ``flux < frac * z_guess`` (IDL: ``ztop[0]``).
        ``None`` if no such row exists (IDL: ``cnt eq 0``).
    guessyb : int or None
        Row index of the last pixel *below* ``y_guess`` where
        ``flux < frac * z_guess`` (IDL: ``zbot[cnt-1]``).
        ``None`` if no such row exists (IDL: ``cnt eq 0``).
    """
    z_guess = _local_flux_at_guess(flux_col, y_guess)
    threshold = frac * z_guess
    row = np.arange(len(flux_col), dtype=float)

    # IDL: ztop = where(fcol lt frac*z_guess and row gt y_guess, cnt)
    ztop_idx = np.where((flux_col < threshold) & (row > y_guess))[0]
    guessyt: int | None = int(ztop_idx[0]) if len(ztop_idx) > 0 else None

    # IDL: zbot = where(fcol lt frac*z_guess and row lt y_guess, cnt)
    zbot_idx = np.where((flux_col < threshold) & (row < y_guess))[0]
    guessyb: int | None = int(zbot_idx[-1]) if len(zbot_idx) > 0 else None

    return guessyt, guessyb


# ---------------------------------------------------------------------------
# IDL mc_findorders building-block helpers (blocks G–I)
# ---------------------------------------------------------------------------


def _sobel_centroid(
    sobel_col: npt.NDArray,
    guess_row: int,
    halfwin: int,
    nrows: int,
) -> float:
    """Compute the Sobel-weighted centroid (COM) around a threshold-guess row.

    IDL block (mc_findorders.pro, lines 224–229 for top, 240–246 for bottom):
        bidx    = 0 > (guessyt - halfwin)
        tidx    = (guessyt + halfwin) < (nrows-1)
        y       = row[bidx:tidx]
        z       = rcol[bidx:tidx]
        COM_top = total(y*z) / total(z)

    In IDL the subscript ``bidx:tidx`` is *inclusive* on both ends; the
    Python equivalent is ``bidx : tidx+1``.

    Parameters
    ----------
    sobel_col : ndarray of float, shape (nrows,)
        Sobel column profile (IDL ``rcol``).
    guess_row : int
        Threshold-guess row index (IDL ``guessyt`` / ``guessyb``).
    halfwin : int
        Half-width of the COM window in rows (IDL ``halfwin``).
    nrows : int
        Number of detector rows (used to clip ``tidx``).

    Returns
    -------
    float
        Sobel-weighted centroid row position, or ``NaN`` when the local
        Sobel sum is zero (undefined COM).
    """
    # IDL: bidx = 0 > (guess_row - halfwin)   → max(0, guess_row - halfwin)
    bidx = max(0, guess_row - halfwin)
    # IDL: tidx = (guess_row + halfwin) < (nrows-1)  → min(nrows-1, guess_row + halfwin)
    tidx = min(nrows - 1, guess_row + halfwin)

    row = np.arange(nrows, dtype=float)
    # IDL subscript bidx:tidx is inclusive; Python slice is bidx:tidx+1.
    y = row[bidx: tidx + 1]
    z = sobel_col[bidx: tidx + 1]

    # IDL: COM = total(y*z) / total(z)
    total_z = float(np.sum(z))
    if total_z == 0.0:
        return np.nan
    return float(np.sum(y * z) / total_z)


def _accept_edge_pair(
    bottom_edge: float,
    top_edge: float,
    slit_height_min: float,
    slit_height_max: float,
) -> bool:
    """Return True when the edge pair passes the IDL slit-height sanity check.

    IDL block (mc_findorders.pro, lines 277–285, repeated at lines 367–375):
        if dotop and dobot and finite(COM_bot) and finite(COM_top) then begin
            if abs(com_bot-com_top) gt slith_pix[0] and
               abs(com_bot-com_top) lt slith_pix[1] then begin
                ; accept
            endif else goto, cont1   ; reject
        endif else begin
            ; NaN case → store NaN, fall back to y_guess
        endelse

    Parameters
    ----------
    bottom_edge : float
        COM position of the bottom edge (IDL ``com_bot``).
    top_edge : float
        COM position of the top edge (IDL ``com_top``).
    slit_height_min : float
        Minimum acceptable slit height in pixels (IDL ``slith_pix[0]``).
    slit_height_max : float
        Maximum acceptable slit height in pixels (IDL ``slith_pix[1]``).

    Returns
    -------
    bool
        ``True`` when both edges are finite and the slit height satisfies
        ``slit_height_min < |bottom - top| < slit_height_max``.
        ``False`` otherwise (IDL ``goto cont1/cont2`` path).
    """
    if not (np.isfinite(bottom_edge) and np.isfinite(top_edge)):
        return False
    slit_h = abs(bottom_edge - top_edge)
    # IDL: abs(com_bot-com_top) gt slith_pix[0] and abs(...) lt slith_pix[1]
    return bool(slit_h > slit_height_min and slit_h < slit_height_max)


def _update_edge_activity_flags(
    top_edge: float,
    bottom_edge: float,
    nrows: int,
    bufpix: int,
    trace_top_active: bool,
    trace_bottom_active: bool,
) -> tuple[bool, bool]:
    """Disable a tracing side when its edge COM reaches the detector guard band.

    IDL block (mc_findorders.pro, lines 302–303, repeated at lines 393–394):
        if com_top le bufpix or com_top ge (nrows-1-bufpix) then dotop = 0
        if com_bot le bufpix or com_bot gt (nrows-1-bufpix) then dobot = 0

    Note the intentional asymmetry between the top and bottom tests:
    * Top uses ``ge`` (>=) for the upper boundary.
    * Bottom uses ``gt`` (>)  for the upper boundary.

    NaN edge values leave the flag unchanged (NaN comparisons are always
    False in Python, matching IDL's ``finite()`` guard).

    Parameters
    ----------
    top_edge : float
        COM position of the top edge (IDL ``com_top``).
    bottom_edge : float
        COM position of the bottom edge (IDL ``com_bot``).
    nrows : int
        Number of detector rows.
    bufpix : int
        Guard-band width in rows (IDL ``bufpix``).
    trace_top_active : bool
        Current state of the top-edge tracing flag (IDL ``dotop``).
    trace_bottom_active : bool
        Current state of the bottom-edge tracing flag (IDL ``dobot``).

    Returns
    -------
    (trace_top_active, trace_bottom_active) : (bool, bool)
        Updated flags.
    """
    upper = nrows - 1 - bufpix

    # IDL: if com_top le bufpix or com_top ge (nrows-1-bufpix) then dotop = 0
    if trace_top_active and (top_edge <= bufpix or top_edge >= upper):
        trace_top_active = False

    # IDL: if com_bot le bufpix or com_bot gt (nrows-1-bufpix) then dobot = 0
    # Note: 'gt' (strictly greater than) for bottom, 'ge' (>=) for top.
    if trace_bottom_active and (bottom_edge <= bufpix or bottom_edge > upper):
        trace_bottom_active = False

    return trace_top_active, trace_bottom_active


# ---------------------------------------------------------------------------
# IDL mc_findorders building-block helpers (blocks J–K)
# ---------------------------------------------------------------------------


def _trace_order_left(
    image: npt.NDArray,
    sobel_image: npt.NDArray,
    sample_cols: npt.NDArray,
    center_samples: npt.NDArray,
    bottom_edge_samples: npt.NDArray,
    top_edge_samples: npt.NDArray,
    gidx: int,
    nrows: int,
    bufpix: int,
    poly_degree: int,
    frac: float,
    com_half_width: int,
    slit_height_min: float,
    slit_height_max: float,
) -> None:
    """Trace left from the seeded column index, updating edge and centre arrays.

    IDL left-sweep (mc_findorders.pro, lines 197–309):
        dotop = 1 & dobot = 1
        for j = 0, gidx do begin
            k = gidx - j
            fcol = reform(image[scols[k], *])
            rcol = reform(rimage[scols[k], *])
            coeff   = mc_polyfit1d(scols, cen, 1>(degree-2), /SILENT, /JUSTFIT)
            y_guess = bufpix > poly(scols[k], coeff) < (nrows-bufpix-1)
            ...COM for top / bottom...
            ...slit-height check...
            ...boundary flags...
            if not dotop and not dobot then goto, moveon1
        endfor
        moveon1:

    Arrays *center_samples*, *bottom_edge_samples*, and *top_edge_samples* are
    mutated in-place, exactly as in IDL's shared ``cen`` / ``edges`` arrays.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
    sobel_image : ndarray, shape (nrows, ncols)
    sample_cols : ndarray of int, shape (n_samp,)
    center_samples : ndarray of float, shape (n_samp,)
        Pre-seeded at the seed window; updated in-place.
    bottom_edge_samples : ndarray of float, shape (n_samp,)
        Updated in-place.
    top_edge_samples : ndarray of float, shape (n_samp,)
        Updated in-place.
    gidx : int
        Index of the seed column in *sample_cols* (IDL ``gidx``).
    nrows : int
    bufpix : int
    poly_degree : int
        Full polynomial degree (IDL ``degree``).
    frac : float
        Intensity threshold fraction (IDL ``frac``).
    com_half_width : int
        Half-width of the Sobel COM window (IDL ``halfwin``).
    slit_height_min : float
        IDL ``slith_pix[0]``.
    slit_height_max : float
        IDL ``slith_pix[1]``.
    """
    # IDL: dotop = 1 & dobot = 1
    do_top = True
    do_bot = True

    # IDL: for j = 0, gidx do begin  …  k = gidx-j  …  endfor
    for j in range(gidx + 1):
        k = gidx - j

        # IDL: fcol = reform(image[scols[k],*]) ; rcol = reform(rimage[scols[k],*])
        fcol, rcol = _extract_column_profiles(image, sobel_image, int(sample_cols[k]))

        # IDL: coeff = mc_polyfit1d(scols,cen,1>(degree-2),/SILENT,/JUSTFIT)
        #      y_guess = bufpix > poly(scols[k],coeff) < (nrows-bufpix-1)
        y_guess_f = _predict_center_from_known_samples(
            sample_cols, center_samples, float(sample_cols[k]),
            poly_degree, nrows, bufpix,
        )
        # Clamp to valid row indices before indexing fcol.
        y_guess = int(np.clip(round(y_guess_f), 0, nrows - 1))

        # IDL: z_guess = fcol[y_guess]
        z_guess = float(fcol[y_guess])
        threshold = frac * z_guess

        # IDL: if dotop then begin
        #        ztop = where(fcol lt frac*z_guess and row gt y_guess, cnt)
        #        if cnt ne 0 then … COM_top … else COM_top = !values.f_nan
        #      endif else COM_top = !values.f_nan
        if do_top:
            guessyt, _ = _find_threshold_edge_guesses(fcol, y_guess_f, frac)
            if guessyt is not None:
                com_top = _sobel_centroid(rcol, guessyt, com_half_width, nrows)
            else:
                com_top = np.nan
        else:
            com_top = np.nan

        # IDL: if dobot then begin
        #        zbot = where(fcol lt frac*z_guess and row lt y_guess, cnt)
        #        if cnt ne 0 then … COM_bot … else COM_bot = !values.f_nan
        #      endif else COM_bot = !values.f_nan
        if do_bot:
            _, guessyb = _find_threshold_edge_guesses(fcol, y_guess_f, frac)
            if guessyb is not None:
                com_bot = _sobel_centroid(rcol, guessyb, com_half_width, nrows)
            else:
                com_bot = np.nan
        else:
            com_bot = np.nan

        # IDL: if dotop and dobot and finite(COM_bot) and finite(COM_top) then begin
        #        if abs(com_bot-com_top) gt slith_pix[0] and … lt slith_pix[1] then begin
        #          edges[k,*] = [com_bot,com_top]; cen[k] = (com_bot+com_top)/2.
        #        endif else goto, cont1
        #      endif else begin
        #        com_bot = !values.f_nan; com_top = !values.f_nan
        #        edges[k,*] = [com_bot,com_top]; cen[k] = y_guess
        #      endelse
        if do_top and do_bot and _accept_edge_pair(com_bot, com_top, slit_height_min, slit_height_max):
            bottom_edge_samples[k] = com_bot
            top_edge_samples[k]    = com_top
            center_samples[k]      = (com_bot + com_top) / 2.0
        else:
            if do_top and do_bot:
                # Both active but edge pair rejected (IDL: goto cont1 path)
                # Edges stay NaN; center falls back to y_guess.
                pass
            # NaN case (IDL: else-branch after the outer if)
            bottom_edge_samples[k] = np.nan
            top_edge_samples[k]    = np.nan
            center_samples[k]      = y_guess_f
            # IDL: goto cont1 — activity flags still checked via fall-through
            # IDL lines 302-303 run for the NaN branch as well; use NaN values
            # so the float comparisons in _update_edge_activity_flags are safe.
            do_top, do_bot = _update_edge_activity_flags(
                com_top, com_bot, nrows, bufpix, do_top, do_bot,
            )
            if not do_top and not do_bot:
                break
            continue

        # IDL: if com_top le bufpix or com_top ge (nrows-1-bufpix) then dotop=0
        # IDL: if com_bot le bufpix or com_bot gt  (nrows-1-bufpix) then dobot=0
        do_top, do_bot = _update_edge_activity_flags(
            com_top, com_bot, nrows, bufpix, do_top, do_bot,
        )

        # IDL: if not dotop and not dobot then goto, moveon1
        if not do_top and not do_bot:
            break


def _trace_order_right(
    image: npt.NDArray,
    sobel_image: npt.NDArray,
    sample_cols: npt.NDArray,
    center_samples: npt.NDArray,
    bottom_edge_samples: npt.NDArray,
    top_edge_samples: npt.NDArray,
    gidx: int,
    nrows: int,
    bufpix: int,
    poly_degree: int,
    frac: float,
    com_half_width: int,
    slit_height_min: float,
    slit_height_max: float,
) -> None:
    """Trace right from the seeded column index, updating edge and centre arrays.

    IDL right-sweep (mc_findorders.pro, lines 312–401):
        dotop = 1 & dobot = 1
        for j = gidx+1, nscols-1 do begin
            k = j
            fcol = reform(image[scols[k],*])
            rcol = reform(rimage[scols[k],*])
            coeff   = mc_polyfit1d(scols,cen,1>(degree-2),/SILENT,/JUSTFIT)
            y_guess = bufpix > poly(scols[k],coeff) < (nrows-bufpix-1)
            ...COM for top / bottom...
            ...slit-height check...
            ...boundary flags...
            if not dotop and not dobot then goto, moveon2
        endfor
        moveon2:

    Parameters are identical to :func:`_trace_order_left` (arrays mutated
    in-place for the right half of *sample_cols*).
    """
    # IDL: dotop = 1 & dobot = 1
    do_top = True
    do_bot = True

    # IDL: for j = gidx+1, nscols-1 do begin  …  k = j  …  endfor
    for k in range(gidx + 1, len(sample_cols)):

        # IDL: fcol = reform(image[scols[k],*]) ; rcol = reform(rimage[scols[k],*])
        fcol, rcol = _extract_column_profiles(image, sobel_image, int(sample_cols[k]))

        # IDL: coeff = mc_polyfit1d(scols,cen,1>(degree-2),/SILENT,/JUSTFIT)
        #      y_guess = bufpix > poly(scols[k],coeff) < (nrows-bufpix-1)
        y_guess_f = _predict_center_from_known_samples(
            sample_cols, center_samples, float(sample_cols[k]),
            poly_degree, nrows, bufpix,
        )
        # Clamp to valid row indices before indexing fcol.
        y_guess = int(np.clip(round(y_guess_f), 0, nrows - 1))

        # IDL: z_guess = fcol[y_guess]
        z_guess = float(fcol[y_guess])
        threshold = frac * z_guess

        # IDL: if dotop then begin
        #        z = where(fcol lt frac*z_guess and row gt y_guess, cnt)
        #        if cnt ne 0 then … COM_top … else com_top = !values.f_nan
        #      endif else com_top = !values.f_nan
        if do_top:
            guessyt, _ = _find_threshold_edge_guesses(fcol, y_guess_f, frac)
            if guessyt is not None:
                com_top = _sobel_centroid(rcol, guessyt, com_half_width, nrows)
            else:
                com_top = np.nan
        else:
            com_top = np.nan

        # IDL: if dobot then begin
        #        z = where(fcol lt frac*z_guess and row lt y_guess, cnt)
        #        if cnt ne 0 then … COM_bot … else com_bot = !values.f_nan
        #      endif else com_bot = !values.f_nan
        if do_bot:
            _, guessyb = _find_threshold_edge_guesses(fcol, y_guess_f, frac)
            if guessyb is not None:
                com_bot = _sobel_centroid(rcol, guessyb, com_half_width, nrows)
            else:
                com_bot = np.nan
        else:
            com_bot = np.nan

        # IDL: if dotop and dobot and finite(com_top) and finite(com_bot) then begin
        #        if abs(com_bot-com_top) gt slith_pix[0] and … lt slith_pix[1] then begin
        #          edges[k,*] = [com_bot,com_top]; cen[k] = (com_bot+com_top)/2.
        #        endif else goto, cont2
        #      endif else begin
        #        com_bot = !values.f_nan; com_top = !values.f_nan
        #        edges[k,*] = [com_bot,com_top]; cen[k] = y_guess
        #      endelse
        if do_top and do_bot and _accept_edge_pair(com_bot, com_top, slit_height_min, slit_height_max):
            bottom_edge_samples[k] = com_bot
            top_edge_samples[k]    = com_top
            center_samples[k]      = (com_bot + com_top) / 2.0
        else:
            # NaN case or rejected pair (IDL else-branch / cont2 path)
            bottom_edge_samples[k] = np.nan
            top_edge_samples[k]    = np.nan
            center_samples[k]      = y_guess_f
            do_top, do_bot = _update_edge_activity_flags(
                com_top, com_bot, nrows, bufpix, do_top, do_bot,
            )
            if not do_top and not do_bot:
                break
            continue

        # IDL: if com_top le bufpix or com_top ge (nrows-1-bufpix) then dotop=0
        # IDL: if com_bot le bufpix or com_bot gt  (nrows-1-bufpix) then dobot=0
        do_top, do_bot = _update_edge_activity_flags(
            com_top, com_bot, nrows, bufpix, do_top, do_bot,
        )

        # IDL: if not dotop and not dobot then goto, moveon2
        if not do_top and not do_bot:
            break


# ---------------------------------------------------------------------------
# IDL mc_findorders building-block helpers (blocks L–N)
# ---------------------------------------------------------------------------


def _fit_order_edge_polynomials(
    sample_cols: npt.NDArray,
    bottom_edge_samples: npt.NDArray,
    top_edge_samples: npt.NDArray,
    poly_degree: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Fit final robust polynomials to the traced bottom and top edges.

    IDL block (mc_findorders.pro, lines 405–421):
        for j = 0,1 do begin
            coeff = mc_robustpoly1d(scols, edges[*,j], degree, 3, 0.01,
                                    OGOODBAD=goodbad, /SILENT, /GAUSSJ,
                                    CANCEL=cancel)
            if cancel then return
            edgecoeffs[*,j,i] = coeff
        endfor

    IDL's ``mc_robustpoly1d`` automatically skips NaN samples via its
    internal goodbad mask.  Here we filter finite samples explicitly before
    passing to :func:`_fit_poly_robust`, which wraps ``_polyfit_1d`` with
    equivalent robust-rejection settings (threshold=3, eps=0.01).

    .. note::
        Temporary drift: the Python :func:`_fit_poly_robust` uses iterative
        sigma-clipping via ``_polyfit_1d`` rather than IDL's Gauss-Jordan
        implementation (``/GAUSSJ``).  The robust interface is otherwise
        parameter-compatible (thresh=3, eps=0.01).

    Parameters
    ----------
    sample_cols : ndarray of float, shape (n_samp,)
        Column positions of all samples (IDL ``scols``).
    bottom_edge_samples : ndarray of float, shape (n_samp,)
        Traced bottom-edge row values; NaN where not traced (IDL ``edges[*,0]``).
    top_edge_samples : ndarray of float, shape (n_samp,)
        Traced top-edge row values; NaN where not traced (IDL ``edges[*,1]``).
    poly_degree : int
        Polynomial degree (IDL ``degree``).

    Returns
    -------
    bottom_coeffs : ndarray, shape (poly_degree + 1,)
        Coefficients for the bottom edge (IDL ``edgecoeffs[*,0,i]``), in
        ``numpy.polynomial.polynomial`` convention.
    top_coeffs : ndarray, shape (poly_degree + 1,)
        Coefficients for the top edge (IDL ``edgecoeffs[*,1,i]``).
    """
    cols = np.asarray(sample_cols, dtype=float)

    # IDL j=0: bottom edge
    bot = np.asarray(bottom_edge_samples, dtype=float)
    bot_ok = np.isfinite(bot)
    bottom_coeffs, _ = _fit_poly_robust(cols[bot_ok], bot[bot_ok], poly_degree)

    # IDL j=1: top edge
    top = np.asarray(top_edge_samples, dtype=float)
    top_ok = np.isfinite(top)
    top_coeffs, _ = _fit_poly_robust(cols[top_ok], top[top_ok], poly_degree)

    return bottom_coeffs, top_coeffs


def _derive_order_xrange_from_fitted_edges(
    x_lo: int,
    x_hi: int,
    bottom_edge_coeffs: npt.NDArray,
    top_edge_coeffs: npt.NDArray,
    nrows: int,
) -> tuple[int, int]:
    """Derive the valid column range where both fitted edges remain on detector.

    IDL block (mc_findorders.pro, lines 403+426–430):
        x   = findgen(stop-start+1)+start
        bot = poly(x, edgecoeffs[*,0,i])
        top = poly(x, edgecoeffs[*,1,i])
        z   = where(top gt 0.0 and top lt nrows-1 and
                    bot gt 0  and bot lt nrows-1)
        xranges[*,i] = [min(x[z],MAX=max), max]

    Both fitted-edge polynomials are evaluated on the dense integer grid
    ``x = [x_lo, x_lo+1, …, x_hi]``.  The valid sub-range is determined
    purely from where both evaluated curves lie strictly inside the detector
    row extent, i.e.:
        0 < bot < nrows-1  AND  0 < top < nrows-1

    No sample-coverage heuristics and no center-based logic are used.

    Parameters
    ----------
    x_lo : int
        First column of the nominal order range (IDL ``start = sranges[0,i]``).
    x_hi : int
        Last column of the nominal order range (IDL ``stop = sranges[1,i]``).
    bottom_edge_coeffs : ndarray, shape (degree + 1,)
        ``numpy.polynomial.polynomial`` coefficients for the bottom edge.
    top_edge_coeffs : ndarray, shape (degree + 1,)
        ``numpy.polynomial.polynomial`` coefficients for the top edge.
    nrows : int
        Number of detector rows.

    Returns
    -------
    xrange_lo : int
        Minimum column where both edges are on detector.
    xrange_hi : int
        Maximum column where both edges are on detector.
        If no valid column exists, returns ``(x_lo, x_lo)`` (IDL would
        error on an empty ``where``; we return a degenerate range instead).
    """
    # IDL: x = findgen(stop-start+1)+start
    x = np.arange(x_lo, x_hi + 1, dtype=float)

    # IDL: bot = poly(x, edgecoeffs[*,0,i])
    #      top = poly(x, edgecoeffs[*,1,i])
    bot = np.polynomial.polynomial.polyval(x, bottom_edge_coeffs)
    top = np.polynomial.polynomial.polyval(x, top_edge_coeffs)

    # IDL: z = where(top gt 0.0 and top lt nrows-1 and
    #                bot gt 0  and bot lt nrows-1)
    valid = (top > 0.0) & (top < nrows - 1) & (bot > 0) & (bot < nrows - 1)

    # IDL: xranges[*,i] = [min(x[z],MAX=max), max]
    if not np.any(valid):
        # Degenerate: no column has both edges on detector.
        return int(x_lo), int(x_lo)

    valid_x = x[valid]
    return int(np.min(valid_x)), int(np.max(valid_x))


def _center_coeffs_from_edge_coeffs(
    bottom_edge_coeffs: npt.NDArray,
    top_edge_coeffs: npt.NDArray,
) -> npt.NDArray:
    """Derive centre polynomial coefficients from the fitted edge coefficients.

    IDL semantic (mc_findorders.pro, line 283 / line 373):
        cen[k] = (com_bot + com_top) / 2.

    At the polynomial level this means:
        center(x) = (bot(x) + top(x)) / 2

    For polynomials with the same degree this is equivalent to averaging the
    corresponding coefficients element-wise:
        center_coeffs = (bottom_coeffs + top_coeffs) / 2

    The centre is therefore derived from the edge polynomials, never fitted
    independently.

    Parameters
    ----------
    bottom_edge_coeffs : ndarray, shape (degree + 1,)
        ``numpy.polynomial.polynomial`` coefficients for the bottom edge.
    top_edge_coeffs : ndarray, shape (degree + 1,)
        ``numpy.polynomial.polynomial`` coefficients for the top edge.

    Returns
    -------
    ndarray, shape (degree + 1,)
        Center polynomial coefficients.

    Raises
    ------
    ValueError
        If the two coefficient arrays have different lengths.
    """
    bot = np.asarray(bottom_edge_coeffs, dtype=float)
    top = np.asarray(top_edge_coeffs, dtype=float)
    if bot.shape != top.shape:
        raise ValueError(
            f"bottom_edge_coeffs and top_edge_coeffs must have the same shape; "
            f"got {bot.shape} vs {top.shape}"
        )
    return (bot + top) / 2.0
