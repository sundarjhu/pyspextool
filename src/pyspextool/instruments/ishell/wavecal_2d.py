"""
Provisional wavelength-identification scaffold for iSHELL 2DXD reduction.

This module implements the third stage of the iSHELL 2DXD reduction scaffold:
connecting traced 2-D arc lines (from
:mod:`~pyspextool.instruments.ishell.arc_tracing`) to wavelength-space
information and fitting a first provisional per-order wavelength solution.

What this module does
---------------------
* Accepts an :class:`~pyspextool.instruments.ishell.arc_tracing.ArcLineTraceResult`
  produced by 2-D arc-line tracing on real iSHELL frames, together with the
  packaged :class:`~pyspextool.instruments.ishell.calibrations.WaveCalInfo` and
  :class:`~pyspextool.instruments.ishell.calibrations.LineList` for the same
  mode.
* For each echelle order, uses the **plane-0 wavelength array** stored in the
  ``WaveCalInfo`` data cube as a *coarse reference grid* to predict, for each
  traced line, what wavelength its detector-column position corresponds to.
* Matches each traced line to the nearest entry in the ``LineList``, accepting
  the match only if the residual between the coarse-predicted wavelength and the
  reference wavelength is within a configurable tolerance.
* Fits a per-order polynomial ``wavelength_um = poly(centerline_col)`` to the
  accepted match set, measuring the provisional dispersion relation directly
  from the 2-D arc-line positions.
* Returns a :class:`ProvisionalWavelengthMap` that records all matches, fit
  coefficients, and residuals, in a form designed for later global
  coefficient-surface fitting.

What this module does NOT do (by design)
-----------------------------------------
* **No full 2DXD coefficient-surface fitting** – fitting a global surface
  over ``(order, column) → wavelength`` is a subsequent step, not implemented
  here.  :meth:`ProvisionalWavelengthMap.collect_for_surface_fit` returns the
  per-order match arrays ready for that step.
* **No tilt correction** – the current implementation uses the seed column of
  each traced line (the column position in the collapsed 1-D median spectrum)
  as the representative column for wavelength assignment.  The 2-D tilt
  polynomial is available from the traced lines but is not used during
  wavelength matching.
* **No rectification-index generation** – interpolation arrays for 2-D
  resampling are not produced here.
* **No interactive arc-line identification** – the matching is entirely
  automatic and based on proximity to the coarse reference grid; iterative
  outlier rejection against a 2-D global model is not implemented.
* **No science-quality wavelength solution** – the result is a development
  scaffold, suitable for exploring the data and validating the pipeline stages,
  not for final scientific reduction.

Relationship to other modules
------------------------------
This module occupies the third stage of the arc-calibration scaffold::

    Flat tracing (tracing.py)
        └── FlatOrderTrace → OrderGeometrySet
                │
                ▼
    Arc-line tracing (arc_tracing.py)
        └── ArcLineTraceResult
                │
                ▼
    Provisional wavelength matching (this module)
        └── ProvisionalWavelengthMap
                │
                ▼
    Coefficient-surface fitting (NOT YET IMPLEMENTED)
        └── global (order × col) → wavelength surface
                │
                ▼
    2DXD Rectification (NOT YET IMPLEMENTED)

Order-number assignment (provisional heuristic)
------------------------------------------------
The ``OrderGeometrySet`` produced by flat-field tracing assigns placeholder
order numbers ``0, 1, 2, …`` (see
:meth:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace.to_order_geometry_set`).
These are *not* the echelle order numbers (e.g. 311–355 for H1 mode).

This module resolves the mapping by assuming that the *i*-th traced order
(index ``i`` in the ``ArcLineTraceResult``) corresponds to the *i*-th entry
in ``wavecalinfo.orders``.  This is a heuristic that relies on:

* flat-field tracing and the packaged ``WaveCalInfo`` covering the same set of
  echelle orders, in the same bottom-to-top (or top-to-bottom) spatial order;
* no orders having been dropped or re-sorted between the two.

If the number of traced orders differs from the number of orders in the
``WaveCalInfo``, a ``RuntimeWarning`` is issued and the shorter list is used.
This limitation is noted clearly in the returned
:attr:`ProvisionalWavelengthMap.order_count_mismatch` flag.

Coarse wavelength reference (plane 0 of WaveCalInfo)
-----------------------------------------------------
The column-to-wavelength reference used for matching is derived from
**plane 0** of ``WaveCalInfo.data``, which is confirmed (by FITS header
``XUNITS="um"`` and value ranges consistent with J/H/K bands) to store
wavelengths in µm along the order centreline.

For each order index ``i``:

* ``wav_array = wavecalinfo.data[i, 0, :]``
* Valid pixels are those where ``wav_array`` is not NaN.
* The column mapping is ``col = wavecalinfo.xranges[i, 0] + array_index``.
* The predicted wavelength for any column is obtained by linear interpolation
  on this grid.

This is the same coarse reference used by
:func:`~pyspextool.instruments.ishell.wavecal.build_geometry_from_arc_lines`
in ``wavecal.py``, but here it is applied to detector positions of **2-D
traced lines** rather than positions measured from the stored 1-D spectrum.

Public API
----------
- :func:`fit_provisional_wavelength_map` – main entry point: accepts an
  ``ArcLineTraceResult``, ``WaveCalInfo``, and ``LineList``; returns a
  :class:`ProvisionalWavelengthMap`.
- :class:`ProvisionalLineMatch` – one traced line matched to one reference
  wavelength.
- :class:`ProvisionalOrderSolution` – per-order polynomial fit result.
- :class:`ProvisionalWavelengthMap` – full result container with helpers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .geometry import OrderGeometry, OrderGeometrySet

if TYPE_CHECKING:
    from .arc_tracing import ArcLineTraceResult, TracedArcLine
    from .calibrations import LineList, WaveCalInfo

__all__ = [
    "ProvisionalLineMatch",
    "ProvisionalOrderSolution",
    "ProvisionalWavelengthMap",
    "fit_provisional_wavelength_map",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProvisionalLineMatch:
    """A single traced arc line provisionally matched to a reference wavelength.

    This records the outcome of matching one
    :class:`~pyspextool.instruments.ishell.arc_tracing.TracedArcLine` to one
    entry in the :class:`~pyspextool.instruments.ishell.calibrations.LineList`
    using a coarse ``col → wavelength`` reference grid from the stored
    ``WaveCalInfo``.

    Parameters
    ----------
    order_index : int
        0-based index of this order in the :class:`ArcLineTraceResult`
        geometry (placeholder; not the echelle order number).
    order_number : int
        Echelle order number derived from ``wavecalinfo.orders[order_index]``.
        This is a provisional assignment; see the module docstring for caveats.
    traced_line : :class:`~pyspextool.instruments.ishell.arc_tracing.TracedArcLine`
        The 2-D traced arc line.
    centerline_col : float
        Detector column used as the representative position for this line.
        Currently set to ``traced_line.seed_col`` (the column where the line
        was detected in the collapsed 1-D median spectrum).
    predicted_wavelength_um : float
        Wavelength predicted for ``centerline_col`` by linear interpolation
        on the coarse ``WaveCalInfo`` plane-0 grid (µm).
    reference_wavelength_um : float
        Vacuum wavelength of the nearest ``LineList`` entry for this order
        within the matching tolerance (µm).
    match_residual_um : float
        Absolute difference between ``reference_wavelength_um`` and
        ``predicted_wavelength_um`` (µm).  This is the quantity used to
        accept or reject the match.
    reference_species : str
        Atomic/ionic species label from the ``LineList`` entry (e.g.
        ``"Ar I"``).

    Notes
    -----
    Only the ``centerline_col`` (``seed_col``) is used during matching.  The
    2-D tilt polynomial stored in ``traced_line.poly_coeffs`` is available for
    later use (e.g. tilt-corrected wavelength assignment) but is not applied
    at this stage.
    """

    order_index: int
    order_number: int
    traced_line: "TracedArcLine"
    centerline_col: float
    predicted_wavelength_um: float
    reference_wavelength_um: float
    match_residual_um: float
    reference_species: str


@dataclass
class ProvisionalOrderSolution:
    """Provisional per-order polynomial wavelength solution.

    Contains the polynomial fit to accepted :class:`ProvisionalLineMatch`
    objects for one echelle order, plus the raw matched arrays needed by a
    downstream coefficient-surface fitter.

    Parameters
    ----------
    order_index : int
        0-based index into the :class:`ArcLineTraceResult` geometry.
    order_number : int
        Echelle order number (from ``wavecalinfo.orders``).
    n_candidate : int
        Number of traced lines in this order that were considered for matching.
    n_matched : int
        Number of traced lines that had a candidate reference line within
        ``match_tol_um`` of the coarse-predicted wavelength.
    n_accepted : int
        Number of accepted matches used in the polynomial fit.  May be less
        than ``n_matched`` if the polynomial degree requires a minimum number
        of points.
    wave_coeffs : ndarray, shape (fit_degree + 1,)
        Polynomial coefficients: ``wavelength_um = polyval(centerline_col, wave_coeffs)``.
        Following the ``numpy.polynomial.polynomial`` convention (``coeffs[k]``
        is the coefficient of ``col**k``).  ``None`` if fewer than two
        matches were accepted.
    fit_degree : int
        Actual polynomial degree used for this order (may be reduced below the
        requested degree if fewer match points are available).
    fit_rms_um : float
        RMS of the residuals ``reference_wavelength_um − polyval(col, wave_coeffs)``
        over the accepted match set (µm).  ``NaN`` if no fit was possible.
    accepted_matches : list of :class:`ProvisionalLineMatch`
        All matches that were accepted and used in the polynomial fit.
    centerline_cols : ndarray, shape (n_accepted,)
        Centroid columns of accepted matched lines; convenience copy for
        downstream surface fitting.
    reference_wavs : ndarray, shape (n_accepted,)
        Reference wavelengths of accepted matched lines; convenience copy.
    """

    order_index: int
    order_number: int
    n_candidate: int
    n_matched: int
    n_accepted: int
    wave_coeffs: npt.NDArray | None
    fit_degree: int
    fit_rms_um: float
    accepted_matches: list[ProvisionalLineMatch] = field(default_factory=list)
    centerline_cols: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    reference_wavs: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )


@dataclass
class ProvisionalWavelengthMap:
    """Provisional per-order wavelength solutions from 2-D traced arc lines.

    This is the primary output of :func:`fit_provisional_wavelength_map`.
    It records per-order polynomial wavelength solutions derived from matching
    2-D traced arc lines (from
    :mod:`~pyspextool.instruments.ishell.arc_tracing`) to reference
    wavelengths from the packaged line list.

    The design is intentionally provisional:

    * The per-order polynomials are independent fits; no global
      coefficient-surface model is imposed.
    * Order-number assignments are a heuristic (see module docstring).
    * Tilt information from the 2-D traces is available but not used in
      matching.
    * The result is suitable for diagnostics and as an input to a later
      global coefficient-surface fitting step.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    order_solutions : list of :class:`ProvisionalOrderSolution`
        One entry per order that was attempted, in the same order as the
        :class:`~pyspextool.instruments.ishell.arc_tracing.ArcLineTraceResult`.
    geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        The flat-field order geometry used during arc-line tracing.
    n_total_accepted : int
        Total number of accepted line matches across all orders.
    match_tol_um : float
        Matching tolerance in µm that was used.
    dispersion_degree : int
        Polynomial degree used for the per-order fits.
    order_count_mismatch : bool
        ``True`` if the number of traced orders did not match the number of
        orders in the ``WaveCalInfo``.  See the module docstring for the
        implications.
    """

    mode: str
    order_solutions: list[ProvisionalOrderSolution]
    geometry: OrderGeometrySet
    n_total_accepted: int
    match_tol_um: float
    dispersion_degree: int
    order_count_mismatch: bool = False

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_orders(self) -> int:
        """Number of orders for which a solution was attempted."""
        return len(self.order_solutions)

    @property
    def solved_orders(self) -> list["ProvisionalOrderSolution"]:
        """Orders that have at least one accepted match (``wave_coeffs`` is not None)."""
        return [s for s in self.order_solutions if s.wave_coeffs is not None]

    @property
    def n_solved_orders(self) -> int:
        """Number of orders with a valid polynomial fit."""
        return len(self.solved_orders)

    # ------------------------------------------------------------------
    # Data extraction for downstream coefficient-surface fitting
    # ------------------------------------------------------------------

    def collect_for_surface_fit(
        self,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Collect all accepted matches as arrays for global surface fitting.

        Returns one row per accepted ``(order_number, centerline_col,
        reference_wavelength_um)`` triple, ready to be passed to a global
        2-D surface fitter ``wavelength = surface(order, col)``.

        Returns
        -------
        order_numbers : ndarray, shape (n,)
            Echelle order number for each accepted match.
        centerline_cols : ndarray, shape (n,)
            Detector column for each accepted match.
        reference_wavs : ndarray, shape (n,)
            Reference wavelength (µm) for each accepted match.

        Notes
        -----
        This method returns **all accepted matches across all orders**,
        including orders where the per-order polynomial fit was skipped
        (because fewer than two matches were accepted for that order).
        Individual per-order data are available via
        :attr:`ProvisionalOrderSolution.centerline_cols` and
        :attr:`ProvisionalOrderSolution.reference_wavs`.
        """
        order_nums: list[float] = []
        cols: list[float] = []
        wavs: list[float] = []
        for sol in self.order_solutions:
            for m in sol.accepted_matches:
                order_nums.append(float(m.order_number))
                cols.append(m.centerline_col)
                wavs.append(m.reference_wavelength_um)
        return (
            np.array(order_nums, dtype=float),
            np.array(cols, dtype=float),
            np.array(wavs, dtype=float),
        )

    def to_geometry_set(
        self,
        fallback_geometry: OrderGeometrySet | None = None,
    ) -> OrderGeometrySet:
        """Convert to an :class:`~.geometry.OrderGeometrySet` with ``wave_coeffs`` populated.

        For orders with a valid polynomial fit, replaces ``wave_coeffs`` with
        the provisional fit coefficients.  For orders without a fit (fewer than
        two accepted matches), copies the existing ``wave_coeffs`` from
        ``fallback_geometry`` if provided, or leaves the field as ``None``.

        Parameters
        ----------
        fallback_geometry : :class:`~.geometry.OrderGeometrySet` or None, optional
            A geometry set (e.g. from
            :func:`~pyspextool.instruments.ishell.wavecal.build_geometry_from_wavecalinfo`)
            whose ``wave_coeffs`` are used as fallback for orders without a
            provisional fit.

        Returns
        -------
        :class:`~.geometry.OrderGeometrySet`
            New geometry set with ``wave_coeffs`` updated from the provisional
            fits where available.
        """
        geoms_out = []
        n_sol = len(self.order_solutions)

        for i, geom_in in enumerate(self.geometry.geometries):
            sol = self.order_solutions[i] if i < n_sol else None

            if sol is not None and sol.wave_coeffs is not None:
                wave_coeffs_out = sol.wave_coeffs.copy()
            elif fallback_geometry is not None and i < fallback_geometry.n_orders:
                fb_geom = fallback_geometry.geometries[i]
                wave_coeffs_out = (
                    fb_geom.wave_coeffs.copy()
                    if fb_geom.wave_coeffs is not None
                    else None
                )
            else:
                wave_coeffs_out = geom_in.wave_coeffs

            geoms_out.append(
                OrderGeometry(
                    order=geom_in.order,
                    x_start=geom_in.x_start,
                    x_end=geom_in.x_end,
                    bottom_edge_coeffs=geom_in.bottom_edge_coeffs.copy(),
                    top_edge_coeffs=geom_in.top_edge_coeffs.copy(),
                    wave_coeffs=wave_coeffs_out,
                    tilt_coeffs=geom_in.tilt_coeffs,
                    curvature_coeffs=geom_in.curvature_coeffs,
                    spatcal_coeffs=geom_in.spatcal_coeffs,
                )
            )

        return OrderGeometrySet(mode=self.geometry.mode, geometries=geoms_out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_provisional_wavelength_map(
    trace_result: "ArcLineTraceResult",
    wavecalinfo: "WaveCalInfo",
    line_list: "LineList",
    *,
    dispersion_degree: int = 3,
    match_tol_um: float = 0.005,
    min_lines_per_order: int = 2,
) -> ProvisionalWavelengthMap:
    """Match traced 2-D arc lines to reference wavelengths and fit per-order solutions.

    This is the main entry point for the provisional wavelength-mapping
    scaffold.  It connects the detector-space arc-line positions produced by
    :func:`~pyspextool.instruments.ishell.arc_tracing.trace_arc_lines` to the
    reference wavelengths in a packaged ThAr line list.

    For each echelle order:

    1. Builds a coarse ``col → wavelength`` lookup by linear interpolation on
       the plane-0 wavelength array stored in ``wavecalinfo.data``.
    2. For each traced line in that order, predicts the wavelength at its
       ``seed_col`` using the coarse lookup.
    3. Finds the nearest reference line in ``line_list`` (for the corresponding
       echelle order number) and accepts the match if the residual is within
       ``match_tol_um``.
    4. Fits a degree-``dispersion_degree`` polynomial
       ``wavelength_um = poly(col)`` to the accepted matches.
    5. Records all results in a :class:`ProvisionalWavelengthMap`.

    Parameters
    ----------
    trace_result : :class:`~pyspextool.instruments.ishell.arc_tracing.ArcLineTraceResult`
        2-D arc-line tracing result from
        :func:`~pyspextool.instruments.ishell.arc_tracing.trace_arc_lines`.
        Provides per-order lists of :class:`~pyspextool.instruments.ishell.arc_tracing.TracedArcLine`
        objects with their detector-column positions and tilt polynomials.
    wavecalinfo : :class:`~pyspextool.instruments.ishell.calibrations.WaveCalInfo`
        Packaged wavelength calibration for the same mode.  Must have
        ``xranges`` populated (from ``OR{n}_XR`` FITS headers) and ``data``
        with valid plane 0 (wavelengths in µm).
    line_list : :class:`~pyspextool.instruments.ishell.calibrations.LineList`
        ThAr arc-line reference list for the same mode.
    dispersion_degree : int, default 3
        Polynomial degree for per-order ``wavelength_um = poly(col)`` fits.
        Automatically reduced if fewer than ``dispersion_degree + 1`` matches
        are accepted for an order.
    match_tol_um : float, default 0.005
        Maximum allowed residual between the coarse-predicted wavelength and
        the nearest reference wavelength (µm).  Increasing this value accepts
        more matches but risks misidentifications; decreasing it may reject
        good matches when the coarse calibration is slightly offset.  The
        default of 0.005 µm (5 nm) is conservative for H-band ThAr lines
        with a coarse calibration.
    min_lines_per_order : int, default 2
        Minimum number of accepted matches required to attempt a polynomial
        fit.  Orders with fewer matches receive ``wave_coeffs = None`` and a
        ``RuntimeWarning``.

    Returns
    -------
    :class:`ProvisionalWavelengthMap`
        Provisional per-order wavelength solutions.  Call
        :meth:`ProvisionalWavelengthMap.collect_for_surface_fit` to obtain
        arrays suitable for global coefficient-surface fitting.

    Raises
    ------
    ValueError
        If ``wavecalinfo.xranges`` is ``None``, or if ``trace_result``
        contains no orders.

    Warnings
    --------
    RuntimeWarning
        Emitted for each order where the number of accepted matches falls
        below ``min_lines_per_order``.
    RuntimeWarning
        Emitted if the number of traced orders does not match the number of
        orders in ``wavecalinfo``.

    Notes
    -----
    **Order-number assignment** — The i-th traced order is assumed to
    correspond to the i-th entry in ``wavecalinfo.orders``.  This is a
    heuristic; see the module docstring for the full discussion.

    **Tilt** — The 2-D tilt polynomials in the traced lines are available
    from the returned :class:`ProvisionalLineMatch` objects but are not used
    during matching.  The seed column (``traced_line.seed_col``) serves as the
    representative column position.

    **Provisional status** — This function intentionally produces a
    per-order, non-global wavelength solution.  The result should not be used
    for science-quality spectral extraction without further validation and the
    full 2DXD coefficient-surface fitting step.

    Examples
    --------
    Typical usage after obtaining flat-order geometry and arc-line traces:

    >>> import glob
    >>> from pyspextool.instruments.ishell.calibrations import (
    ...     read_line_list, read_wavecalinfo)
    >>> from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
    >>> from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
    >>> from pyspextool.instruments.ishell.wavecal_2d import (
    ...     fit_provisional_wavelength_map)
    >>>
    >>> flat_files = sorted(glob.glob(
    ...     "data/testdata/ishell_h1_calibrations/raw/*.flat.*.fits"))
    >>> arc_files = sorted(glob.glob(
    ...     "data/testdata/ishell_h1_calibrations/raw/*.arc.*.fits"))
    >>>
    >>> # Stage 1: flat-field order tracing
    >>> flat_trace = trace_orders_from_flat(flat_files, col_range=(650, 1550))
    >>> geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))
    >>>
    >>> # Stage 2: arc-line tracing
    >>> arc_result = trace_arc_lines(arc_files, geom)
    >>>
    >>> # Stage 3: provisional wavelength matching
    >>> wci = read_wavecalinfo("H1")
    >>> ll = read_line_list("H1")
    >>> wav_map = fit_provisional_wavelength_map(arc_result, wci, ll)
    >>>
    >>> print(f"Solved {wav_map.n_solved_orders}/{wav_map.n_orders} orders")
    >>> order_nums, cols, wavs = wav_map.collect_for_surface_fit()
    >>> print(f"Total accepted matches: {len(cols)}")
    """
    if wavecalinfo.xranges is None:
        raise ValueError(
            "wavecalinfo.xranges is None; cannot build a column-to-wavelength "
            "lookup.  Re-read with a version that parses OR{n}_XR headers."
        )
    if trace_result.n_orders == 0:
        raise ValueError("trace_result contains no orders.")

    n_traced = trace_result.n_orders
    n_wci = len(wavecalinfo.orders)
    order_count_mismatch = n_traced != n_wci

    if order_count_mismatch:
        warnings.warn(
            f"trace_result has {n_traced} orders but wavecalinfo has {n_wci} "
            "orders.  Matching by index up to the shorter list.  "
            "See the wavecal_2d module docstring for the order-assignment "
            "heuristic and its limitations.",
            RuntimeWarning,
            stacklevel=2,
        )

    n_orders_to_process = min(n_traced, n_wci)

    # Build a reference dict: order_number → list of (wavelength_um, species)
    ref_by_order: dict[int, list[tuple[float, str]]] = {}
    for entry in line_list.entries:
        ref_by_order.setdefault(entry.order, []).append(
            (entry.wavelength_um, entry.species)
        )

    order_solutions: list[ProvisionalOrderSolution] = []

    for i in range(n_orders_to_process):
        order_num = int(wavecalinfo.orders[i])
        traced_lines = trace_result.lines_for_order(i)

        # Build coarse col → wavelength lookup for this order
        coarse_cols, coarse_wavs = _build_coarse_lookup(wavecalinfo, i)

        # Get reference lines for this echelle order
        ref_entries = ref_by_order.get(order_num, [])

        # Match each traced line to a reference wavelength
        accepted_matches, n_candidate, n_matched = _match_lines_for_order(
            order_index=i,
            order_number=order_num,
            traced_lines=traced_lines,
            coarse_cols=coarse_cols,
            coarse_wavs=coarse_wavs,
            ref_entries=ref_entries,
            match_tol_um=match_tol_um,
        )

        # Fit polynomial to accepted matches
        sol = _fit_order_solution(
            order_index=i,
            order_number=order_num,
            n_candidate=n_candidate,
            n_matched=n_matched,
            accepted_matches=accepted_matches,
            dispersion_degree=dispersion_degree,
            min_lines_per_order=min_lines_per_order,
        )
        order_solutions.append(sol)

    n_total_accepted = sum(s.n_accepted for s in order_solutions)

    return ProvisionalWavelengthMap(
        mode=trace_result.mode,
        order_solutions=order_solutions,
        geometry=trace_result.geometry,
        n_total_accepted=n_total_accepted,
        match_tol_um=match_tol_um,
        dispersion_degree=dispersion_degree,
        order_count_mismatch=order_count_mismatch,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_coarse_lookup(
    wavecalinfo: "WaveCalInfo",
    order_idx: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Return (cols, wavs) arrays for the coarse col→wavelength lookup.

    Plane 0 of the WaveCalInfo data cube contains wavelengths in µm.
    NaN values are excluded.  The column mapping is:
    ``col = xranges[order_idx, 0] + array_index``.

    Returns
    -------
    cols : ndarray
        Detector column positions (1-D, monotone increasing).
    wavs : ndarray
        Corresponding wavelengths in µm.

    Returns ``(empty, empty)`` if no valid data exist for this order.
    """
    wav_array = wavecalinfo.data[order_idx, 0, :]
    valid = ~np.isnan(wav_array)
    if not valid.any():
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    x_start = int(wavecalinfo.xranges[order_idx, 0])
    n_valid = int(valid.sum())
    cols = np.arange(n_valid, dtype=float) + x_start
    wavs = wav_array[valid]
    return cols, wavs


def _match_lines_for_order(
    order_index: int,
    order_number: int,
    traced_lines: list["TracedArcLine"],
    coarse_cols: npt.NDArray,
    coarse_wavs: npt.NDArray,
    ref_entries: list[tuple[float, str]],
    match_tol_um: float,
) -> tuple[list[ProvisionalLineMatch], int, int]:
    """Match traced lines to reference wavelengths for one order.

    Returns
    -------
    accepted_matches : list of ProvisionalLineMatch
    n_candidate : int
        Number of traced lines considered.
    n_matched : int
        Number of traced lines where a candidate match was found (within
        ``match_tol_um``).
    """
    accepted: list[ProvisionalLineMatch] = []
    n_candidate = len(traced_lines)
    n_matched = 0

    if not traced_lines or len(coarse_cols) == 0 or not ref_entries:
        return accepted, n_candidate, n_matched

    ref_wavs = np.array([e[0] for e in ref_entries], dtype=float)
    ref_species = [e[1] for e in ref_entries]

    col_min = float(coarse_cols[0])
    col_max = float(coarse_cols[-1])

    for line in traced_lines:
        col = float(line.seed_col)

        # Skip if seed column is outside the coarse reference range
        if col < col_min or col > col_max:
            continue

        # Coarse wavelength prediction by linear interpolation
        pred_wav = float(np.interp(col, coarse_cols, coarse_wavs))

        # Nearest reference line
        diffs = np.abs(ref_wavs - pred_wav)
        best_idx = int(np.argmin(diffs))
        residual = float(diffs[best_idx])

        if residual > match_tol_um:
            continue

        n_matched += 1
        accepted.append(
            ProvisionalLineMatch(
                order_index=order_index,
                order_number=order_number,
                traced_line=line,
                centerline_col=col,
                predicted_wavelength_um=pred_wav,
                reference_wavelength_um=float(ref_wavs[best_idx]),
                match_residual_um=residual,
                reference_species=ref_species[best_idx],
            )
        )

    # Remove duplicate reference-wavelength assignments: if two traced lines
    # match the same reference line, keep the one with the smaller residual.
    accepted = _deduplicate_matches(accepted)

    return accepted, n_candidate, n_matched


def _deduplicate_matches(
    matches: list[ProvisionalLineMatch],
) -> list[ProvisionalLineMatch]:
    """Resolve duplicate reference-wavelength assignments.

    If two or more traced lines are matched to the same reference wavelength,
    keep only the match with the smallest ``match_residual_um``.  This avoids
    degenerate polynomial fits from two centroid measurements claiming the same
    reference wavelength.
    """
    if len(matches) <= 1:
        return list(matches)

    # Group by reference wavelength (use exact float equality; reference
    # wavelengths come from a finite set, so collisions are exact duplicates)
    from collections import defaultdict
    by_ref: dict[float, list[ProvisionalLineMatch]] = defaultdict(list)
    for m in matches:
        by_ref[m.reference_wavelength_um].append(m)

    result: list[ProvisionalLineMatch] = []
    for ref_wav, group in by_ref.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Keep the match with the smallest residual
            best = min(group, key=lambda m: m.match_residual_um)
            result.append(best)

    # Sort by centroid column for deterministic output
    result.sort(key=lambda m: m.centerline_col)
    return result


def _fit_order_solution(
    order_index: int,
    order_number: int,
    n_candidate: int,
    n_matched: int,
    accepted_matches: list[ProvisionalLineMatch],
    dispersion_degree: int,
    min_lines_per_order: int,
) -> ProvisionalOrderSolution:
    """Fit a polynomial wavelength solution to accepted matches for one order.

    If fewer than ``min_lines_per_order`` matches were accepted, emits a
    RuntimeWarning and returns a solution with ``wave_coeffs = None`` and
    ``fit_rms_um = NaN``.
    """
    n_accepted = len(accepted_matches)

    if n_accepted < min_lines_per_order:
        warnings.warn(
            f"Order {order_number} (index {order_index}): only {n_accepted} "
            f"accepted match(es) (minimum {min_lines_per_order}); "
            "skipping polynomial fit for this order.",
            RuntimeWarning,
            stacklevel=4,
        )
        cols_arr = np.array(
            [m.centerline_col for m in accepted_matches], dtype=float
        )
        wavs_arr = np.array(
            [m.reference_wavelength_um for m in accepted_matches], dtype=float
        )
        return ProvisionalOrderSolution(
            order_index=order_index,
            order_number=order_number,
            n_candidate=n_candidate,
            n_matched=n_matched,
            n_accepted=n_accepted,
            wave_coeffs=None,
            fit_degree=dispersion_degree,
            fit_rms_um=float("nan"),
            accepted_matches=accepted_matches,
            centerline_cols=cols_arr,
            reference_wavs=wavs_arr,
        )

    cols_arr = np.array([m.centerline_col for m in accepted_matches], dtype=float)
    wavs_arr = np.array(
        [m.reference_wavelength_um for m in accepted_matches], dtype=float
    )

    fit_degree = min(dispersion_degree, n_accepted - 1)
    if fit_degree < dispersion_degree:
        warnings.warn(
            f"Order {order_number} (index {order_index}): requested degree "
            f"{dispersion_degree} but only {n_accepted} accepted matches; "
            f"reducing polynomial degree to {fit_degree}.",
            RuntimeWarning,
            stacklevel=4,
        )

    wave_coeffs = np.polynomial.polynomial.polyfit(cols_arr, wavs_arr, fit_degree)

    # Compute fit RMS
    wav_pred = np.polynomial.polynomial.polyval(cols_arr, wave_coeffs)
    fit_rms = float(np.sqrt(np.mean((wavs_arr - wav_pred) ** 2)))

    return ProvisionalOrderSolution(
        order_index=order_index,
        order_number=order_number,
        n_candidate=n_candidate,
        n_matched=n_matched,
        n_accepted=n_accepted,
        wave_coeffs=wave_coeffs,
        fit_degree=fit_degree,
        fit_rms_um=fit_rms,
        accepted_matches=accepted_matches,
        centerline_cols=cols_arr,
        reference_wavs=wavs_arr,
    )
