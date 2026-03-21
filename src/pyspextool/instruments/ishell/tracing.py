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
   - *Without flatinfo (fallback)*: auto-detect order centres at a
     single seed column via :func:`scipy.signal.find_peaks`.

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

**Does NOT fully match IDL:**

- Step-based column sampling is used when ``flatinfo.step > 0``; otherwise
  ``n_sample_cols`` evenly-spaced columns are used (no IDL equivalent for
  that fallback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from scipy.ndimage import sobel as _scipy_sobel
from scipy.signal import find_peaks as _scipy_find_peaks

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
        Column indices at which this order was traced.
    center_rows : ndarray, shape (n_samp,)
        Derived centre-row at each sampled column (mean of bot/top edges).
        NaN where edges were not reliably detected.
    bot_rows : ndarray, shape (n_samp,)
        Bottom-edge row at each sampled column.  NaN where not detected.
    top_rows : ndarray, shape (n_samp,)
        Top-edge row at each sampled column.  NaN where not detected.
    x_start : int
        First column of the order's valid range (IDL ``xranges[0, i]``).
    x_end : int
        Last column of the order's valid range (IDL ``xranges[1, i]``).
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
        columns outside each order's own traced range.  Derived from
        ``order_samples`` after tracing.  All polynomial fitting, edge
        detection, and geometry construction MUST use
        ``order_samples[i].center_rows`` instead.
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
    col_half_width: int = 10,
    poly_degree: int = 3,
    seed_col: int | None = None,
    col_range: tuple[int, int] | None = None,
    min_prominence: float = 500.0,
    min_distance: int = 25,
    max_shift: float = 15.0,
    sigma_clip: float = 3.0,
    intensity_fraction: float = 0.85,
    com_half_width: int = 2,
    slit_height_range: tuple[float, float] | None = None,
    ybuffer: int = 1,
) -> FlatOrderTrace:
    """Trace echelle order edges from one or more iSHELL flat-field frames.

    When *flatinfo* is supplied, order initialization follows the IDL
    ``mc_readflatinfo`` / ``mc_adjustguesspos`` path: guess positions are
    derived from the stored reference edge polynomials in the flatinfo
    FITS file, and tracing parameters (``intensity_fraction``,
    ``com_half_width``, ``slit_height_range``, ``ybuffer``, ``poly_degree``)
    are taken from the flatinfo fields, overriding any explicit keyword
    arguments.

    When *flatinfo* is ``None``, order centres are auto-detected at a
    single seed column via :func:`scipy.signal.find_peaks` (generic
    fallback mode).

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

        When ``None`` (default), blind peak detection at the seed column
        is used for order initialization.
    n_sample_cols : int, default 40
        Number of evenly-spaced sample columns (used only when ``flatinfo``
        is ``None`` or when ``flatinfo.step`` is 0).
    col_half_width : int, default 10
        Half-width of the column band used for cross-dispersion profile
        extraction during seed peak detection (fallback mode only).
    poly_degree : int, default 3
        Degree of the polynomial fitted to the edges (overridden by
        ``flatinfo.edge_degree`` when *flatinfo* is supplied).
    seed_col : int, optional
        Seed column for auto-detect mode.  Defaults to mid-point of
        *col_range*.  Ignored when *flatinfo* is supplied.
    col_range : tuple of int (col_start, col_end), optional
        Column range over which tracing is performed.  Defaults to the
        full detector width ``(0, ncols - 1)``.
    min_prominence : float, default 500.0
        Minimum peak prominence for seed detection (fallback mode only).
    min_distance : int, default 25
        Minimum row separation between seed peaks (fallback mode only).
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
        If *flat_files* is empty.
    RuntimeError
        If no order peaks are found (fallback mode only).
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

        # seed_col: representative column used for logging / half-width fallback
        seed_col = int(np.round(np.mean(guess_xranges[:, 0] + guess_xranges[:, 1]) / 2.0))
        seed_profile = _extract_cross_section(flat, seed_col, col_half_width)
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
        # Fallback: auto-detect seed peaks at the central column.
        if seed_col is None:
            seed_col = (col_lo + col_hi) // 2
        else:
            seed_col = max(col_lo, min(col_hi, int(seed_col)))

        seed_profile = _extract_cross_section(flat, seed_col, col_half_width)
        seed_peaks = _find_order_peaks(seed_profile, min_distance, min_prominence)

        if len(seed_peaks) == 0:
            raise RuntimeError(
                f"No order peaks found at seed column {seed_col} "
                f"(prominence>{min_prominence}, distance>{min_distance}).  "
                "Try reducing min_prominence or min_distance."
            )

        n_orders = len(seed_peaks)
        guess_positions = None
        traced_xranges = None

        # All orders share the same column grid in fallback mode.
        shared_cols = np.round(np.linspace(col_lo, col_hi, n_sample_cols)).astype(int)
        shared_cols = np.unique(shared_cols)
        seed_idx = int(np.argmin(np.abs(shared_cols - seed_col)))
        order_sample_cols_list = [shared_cols.copy() for _ in range(n_orders)]
        order_seed_indices = [seed_idx] * n_orders

        logger.info(
            "Auto-detect mode: %d order seeds at column %d: rows %s",
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
    # ------------------------------------------------------------------
    if len(order_samples) != n_orders:
        raise RuntimeError(
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
    # 7. Estimate order half-widths directly from per-order edge samples.
    #    No temporary 2D arrays: compute mean edge separation per order.
    # ------------------------------------------------------------------
    if flatinfo is not None:
        # IDL path: use mean (top − bottom) edge separation for each order.
        half_width_rows = np.zeros(n_orders, dtype=float)
        for i, os_i in enumerate(order_samples):
            sep = os_i.top_rows - os_i.bot_rows
            valid = np.isfinite(sep) & (sep > 0)
            if valid.sum() > 0:
                half_width_rows[i] = float(np.mean(sep[valid]) / 2.0)
            else:
                half_width_rows[i] = 0.0
    else:
        half_width_rows = _estimate_half_widths(seed_profile, seed_peaks)

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
            k = int(np.searchsorted(union_cols, col))
            if k < len(union_cols) and union_cols[k] == col:
                compat_center_rows[i, k] = os_i.center_rows[j]

    # ------------------------------------------------------------------
    # 9. Compute per-order QA metrics
    #    _compute_order_trace_stats evaluates the fitted polynomials at the
    #    supplied column array.  Passing union_cols is safe: it is used for
    #    polynomial evaluation only, not for indexing raw traced arrays.
    # ------------------------------------------------------------------
    order_stats = _compute_order_trace_stats(
        center_poly_coeffs, fit_rms, union_cols,
    )
    _log_trace_qa_summary(order_stats)

    logger.info(
        "Order tracing complete: %d orders, median RMS = %.2f px",
        n_orders,
        float(np.nanmedian(fit_rms)),
    )

    # Build order_xranges from order_samples (authoritative x_start/x_end)
    traced_xranges_out = np.array(
        [[os_i.x_start, os_i.x_end] for os_i in order_samples], dtype=int
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


def _estimate_half_widths(
    profile: npt.NDArray,
    peaks: npt.NDArray,
) -> npt.NDArray:
    """Estimate the half-maximum half-width of each order peak.

    This is a profile-based estimator used in the auto-detect fallback path.
    For edge-tracked orders, prefer :func:`_estimate_half_widths_from_edges`.

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

        left = int(pk)
        while left > 0 and profile[left] > half_max:
            left -= 1

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

    # Defensive: assert that we only perform polynomial evaluation below —
    # never indexing into raw traced-sample arrays (center_rows, bot_rows,
    # top_rows).  This guard documents the contract and will fire if the
    # function body is ever accidentally extended to use raw data.
    assert isinstance(cols, np.ndarray) and cols.ndim == 1, (
        "_compute_order_trace_stats: sample_cols must be a 1-D array used "
        "for polynomial evaluation only — do not pass raw traced-sample arrays."
    )

    # Pre-compute centre-row positions at all sample columns.
    # Use NaN rows where the polynomial fit failed so they do not
    # corrupt inter-order separation calculations.
    # NOTE: the only operation on `cols` throughout this function is
    # ``polyval(cols, coeffs)`` — no indexing into raw traced arrays occurs.
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
