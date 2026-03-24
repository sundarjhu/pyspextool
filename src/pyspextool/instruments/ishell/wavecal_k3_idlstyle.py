"""
IDL-style K3 1DXD wavelength calibration for iSHELL.

This module implements the **K3 benchmark 1DXD** wavelength-calibration path:
a clean Python approximation of the IDL Spextool 2DXD approach that operates
from traced order geometry.

Overview
--------
The IDL 1DXD pipeline operates in two steps:

1. **1D extraction** — for each echelle order, extract a 1-D arc spectrum
   along the order by averaging a small aperture around the traced order
   centre.  This relies on the flat-field traced geometry (Stage 1), *not* a
   crude image median.

2. **Global 1DXD fit** — collect arc-line positions from all orders, match
   them to a reference line list, and fit a single global 2-D polynomial:

   .. math::

       \\lambda(\\text{col}, \\text{order}) =
           \\sum_{i=0}^{w} \\sum_{j=0}^{p}
           C_{i,j}\\, \\text{col}^{i}\\, v^{j}

   where ``v = order_ref / order`` is the normalised inverse-order coordinate
   (physically motivated by the echelle grating equation ``m·λ ≈ const``),
   ``w`` is the dispersion degree, and ``p`` is the order degree.

   For K3 the IDL defaults are ``wdeg=3``, ``odeg=2``.

This model then drives rectification and FITS output — *not* the scaffold
per-order polynomial fits.

Public API
----------
- :func:`extract_order_arc_spectra` – extract 1-D arc spectra using traced geometry.
- :func:`fit_1dxd_wavelength_model` – fit the global 1DXD model.
- :class:`OrderArcSpectrum` – 1-D arc spectrum for one order.
- :class:`OrderArcSpectraSet` – collection of per-order spectra.
- :class:`IdlStyle1DXDModel` – fitted 1DXD model with evaluation helpers.

Constraints
-----------
* Does NOT implement science extraction.
* Does NOT implement telluric correction.
* Does NOT implement order merging.
* Uses only :class:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace`
  from Stage 1 for geometry — never estimates centre rows from the arc image.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate as _scipy_correlate
from scipy.signal import find_peaks as _scipy_find_peaks

if TYPE_CHECKING:
    from .calibrations import LineList, WaveCalInfo
    from .tracing import FlatOrderTrace

__all__ = [
    "OrderArcSpectrum",
    "OrderArcSpectraSet",
    "OrderMatchStats",
    "IdlStyle1DXDModel",
    "extract_order_arc_spectra",
    "fit_1dxd_wavelength_model",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OrderArcSpectrum:
    """1-D arc spectrum extracted for one echelle order.

    Parameters
    ----------
    order_index : int
        Zero-based index of this order in the parent
        :class:`OrderArcSpectraSet`.
    order_number : int
        Echelle order number assigned from the packaged
        :class:`~pyspextool.instruments.ishell.calibrations.WaveCalInfo`.
    col_start : int
        First detector column (inclusive) over which the spectrum was
        extracted.
    col_end : int
        Last detector column (inclusive).
    flux : ndarray, shape (n_cols,)
        Averaged flux at each column.  Columns outside the valid range are
        ``NaN``.

    Notes
    -----
    ``flux[i]`` corresponds to detector column ``col_start + i``.  The
    extraction uses the traced centre-line polynomial from the flat-field
    :class:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace`, *not*
    the arc image itself, to locate the aperture.
    """

    order_index: int
    order_number: int
    col_start: int
    col_end: int
    flux: npt.NDArray  # shape (n_cols,)

    @property
    def n_cols(self) -> int:
        """Number of detector columns in this spectrum."""
        return len(self.flux)

    @property
    def columns(self) -> npt.NDArray:
        """Integer column indices for each flux element."""
        return np.arange(self.col_start, self.col_start + self.n_cols)


@dataclass
class OrderArcSpectraSet:
    """Collection of per-order 1-D arc spectra.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"K3"``).
    spectra : list of :class:`OrderArcSpectrum`
        One entry per echelle order.
    aperture_half_width : int
        Half-width of the extraction aperture (in pixels) used when
        building each spectrum.
    """

    mode: str
    spectra: list[OrderArcSpectrum] = field(default_factory=list)
    aperture_half_width: int = 3

    @property
    def n_orders(self) -> int:
        """Number of orders."""
        return len(self.spectra)

    @property
    def order_numbers(self) -> list[int]:
        """List of echelle order numbers in storage order."""
        return [s.order_number for s in self.spectra]

    def get_spectrum(self, order_number: int) -> OrderArcSpectrum:
        """Return the spectrum for *order_number*.

        Raises
        ------
        KeyError
            If *order_number* is not present.
        """
        for s in self.spectra:
            if s.order_number == order_number:
                return s
        raise KeyError(
            f"Order {order_number} not found in OrderArcSpectraSet "
            f"(mode={self.mode!r}).  Available: {self.order_numbers}"
        )


@dataclass
class OrderMatchStats:
    """Per-order match statistics from the K3 1DXD fitting pipeline.

    Collected during :func:`fit_1dxd_wavelength_model` and stored inside
    :attr:`IdlStyle1DXDModel.per_order_stats`.

    Parameters
    ----------
    order_number : int
        Echelle order number.
    xcorr_shift_px : float
        Cross-correlation shift (in pixels) applied before peak centroiding
        to align the extracted spectrum with the reference comb.  ``0.0``
        if cross-correlation could not be computed for this order.
    n_candidate : int
        Number of reference-line windows that yielded a convincing local peak
        during the local expected-line search.  This is the count of
        successfully found local candidates, one per reference line whose
        predicted column fell within the spectrum and whose local window
        contained a peak above the prominence threshold.
    n_matched : int
        Number of peaks that matched a reference line within the tolerance
        window and passed monotonicity filtering (before global sigma
        clipping).
    n_ambiguous_removed : int
        Number of reference-line matches rejected because multiple peaks
        claimed the same reference line (ambiguity filtering).
    n_monotonic_removed : int
        Number of matches removed by monotonicity enforcement (wavelength
        must increase monotonically with detector column).
    n_accepted : int
        Number of matches retained after global iterative sigma clipping.
    n_rejected : int
        Number of matches rejected by sigma clipping.
    rms_resid_um : float
        Per-order RMS of the global-fit residuals for the *accepted* points
        (µm).  ``NaN`` if no accepted points remain for this order.
    participated : bool
        ``True`` if this order contributed at least one accepted point to
        the global fit.
    xcorr_shift_clipped : bool
        ``True`` if the cross-correlation peak landed at the boundary of the
        allowed search window, indicating the true shift may lie outside it.
    skipped_insufficient_matches : bool
        ``True`` if the order was excluded from the global fit because it
        had fewer than the required minimum number of matched lines after
        filtering.
    min_lines_required : int
        The per-order minimum-match threshold used when deciding whether
        this order should participate in the global fit.
    affine_a : float
        Slope of the per-order affine column correction fitted from the
        first-pass local search results.  ``1.0`` when the identity
        transform was used (fewer than 2 first-pass matches, or the fitted
        slope was outside the allowed range ``[0.95, 1.05]``).
    affine_b : float
        Intercept (pixels) of the per-order affine column correction.
        ``0.0`` when the identity transform was used.
    affine_applied : bool
        ``True`` if a valid non-identity affine correction was fitted and
        applied before the second-pass local search.  ``False`` when the
        identity transform was used.
    affine_n_points : int
        Number of first-pass candidate matches used to fit the affine
        correction.  ``0`` when no first-pass candidates were available.
    used_fallback : bool
        ``True`` if the fallback global peak search was triggered for this
        order (because the local search produced fewer than
        ``min_lines_per_order`` matches) and the fallback produced enough
        matches to rescue the order.  ``False`` in the normal case.
    fallback_n_matches : int
        Number of matches returned by the fallback global peak search.
        ``0`` when fallback was not triggered or produced no matches.
    """

    order_number: int
    xcorr_shift_px: float
    n_candidate: int
    n_matched: int
    n_ambiguous_removed: int
    n_monotonic_removed: int
    n_accepted: int
    n_rejected: int
    rms_resid_um: float
    participated: bool
    xcorr_shift_clipped: bool
    skipped_insufficient_matches: bool
    min_lines_required: int
    affine_a: float = 1.0
    affine_b: float = 0.0
    affine_applied: bool = False
    affine_n_points: int = 0
    used_fallback: bool = False
    fallback_n_matches: int = 0


@dataclass
class IdlStyle1DXDModel:
    """Global IDL-style 1DXD wavelength model.

    Stores the 2-D polynomial fit

    .. math::

        \\lambda(\\text{col}, \\text{order}) =
            \\sum_{i=0}^{\\text{wdeg}} \\sum_{j=0}^{\\text{odeg}}
            C_{i,j}\\, \\text{col}^{i}\\, v^{j}

    where ``v = order_ref / order``.

    Parameters
    ----------
    mode : str
        iSHELL observing mode.
    wdeg : int
        Polynomial degree in detector column.
    odeg : int
        Polynomial degree in ``v = order_ref / order``.
    order_ref : float
        Reference order number (minimum fitted order).  Used to compute ``v``.
    coeffs : ndarray, shape (wdeg+1, odeg+1)
        Coefficient matrix.  ``coeffs[i, j]`` is the coefficient of
        ``col**i * v**j``.
    fitted_order_numbers : list of int
        Echelle order numbers that contributed at least one accepted point.
    fit_rms_um : float
        RMS residual of the global fit after sigma clipping (µm).
    n_lines : int
        Number of arc-line matches *accepted* in the final fit (after sigma
        clipping).
    n_lines_total : int
        Total arc-line matches before sigma clipping.
    n_lines_rejected : int
        Number of arc-line matches rejected by sigma clipping
        (``n_lines_total - n_lines``).
    accepted_mask : ndarray of bool, shape (n_lines_total,)
        Boolean mask: ``True`` for accepted points, ``False`` for rejected.
    median_residual_um : float
        Median residual of the accepted points (µm).  Zero for a perfectly
        symmetric residual distribution.
    n_orders_fit : int
        Number of orders that contributed at least one accepted match.
    per_order_stats : list of :class:`OrderMatchStats`
        Per-order match statistics (one entry per order in *spectra_set*).
    matched_cols_px : ndarray, shape (n_lines_total,)
        Detector column of each matched arc-line point (before or after the
        xcorr shift correction; the value stored is the detected peak column).
    matched_order_numbers : ndarray, shape (n_lines_total,)
        Echelle order number for each matched point.
    matched_ref_wavelength_um : ndarray, shape (n_lines_total,)
        Reference catalogue wavelength (µm) for each matched point.
    matched_fit_wavelength_um : ndarray, shape (n_lines_total,)
        Wavelength predicted by the **final** (sigma-clipped) fit for each
        matched point (µm).
    matched_residual_um : ndarray, shape (n_lines_total,)
        Residual ``ref - fit`` (µm) for each matched point from the final fit.
        Accepted points satisfy ``|residual| ≤ sigma_thresh × rms``.

    Notes
    -----
    Use :meth:`eval` (scalar) or :meth:`eval_array` (vectorised) to predict
    wavelength at arbitrary ``(col, order)`` combinations.

    The five ``matched_*`` arrays all have the same length ``n_lines_total``
    and are aligned element-wise with ``accepted_mask``.  To recover accepted
    vs rejected subsets::

        cols_acc  = model.matched_cols_px[model.accepted_mask]
        cols_rej  = model.matched_cols_px[~model.accepted_mask]
        resid_acc = model.matched_residual_um[model.accepted_mask]
    """

    mode: str
    wdeg: int
    odeg: int
    order_ref: float
    coeffs: npt.NDArray  # shape (wdeg+1, odeg+1)
    fitted_order_numbers: list[int]
    fit_rms_um: float
    n_lines: int
    n_lines_total: int
    n_lines_rejected: int
    accepted_mask: npt.NDArray  # bool, shape (n_lines_total,)
    median_residual_um: float
    n_orders_fit: int
    per_order_stats: list[OrderMatchStats] = field(default_factory=list)
    # Per-point arrays aligned with accepted_mask (length = n_lines_total)
    matched_cols_px: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    matched_order_numbers: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    matched_ref_wavelength_um: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    matched_fit_wavelength_um: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    matched_residual_um: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )

    # ------------------------------------------------------------------
    # Coordinate helper
    # ------------------------------------------------------------------

    def _v(self, order: float | npt.NDArray) -> float | npt.NDArray:
        """Normalised inverse-order coordinate ``v = order_ref / order``."""
        return self.order_ref / np.asarray(order, dtype=float)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def eval(self, col: float, order: float) -> float:
        """Evaluate the model at a single ``(col, order)`` point.

        Parameters
        ----------
        col : float
            Detector column.
        order : float
            Echelle order number.

        Returns
        -------
        float
            Predicted wavelength in µm.
        """
        v = float(self._v(float(order)))
        c = float(col)
        result = 0.0
        for i in range(self.wdeg + 1):
            for j in range(self.odeg + 1):
                result += self.coeffs[i, j] * (c ** i) * (v ** j)
        return result

    def eval_array(
        self,
        cols: npt.ArrayLike,
        orders: npt.ArrayLike,
    ) -> npt.NDArray:
        """Evaluate the model at arrays of ``(col, order)`` points.

        Parameters
        ----------
        cols : array_like, shape (n,)
            Detector columns.
        orders : array_like, shape (n,)
            Echelle order numbers.

        Returns
        -------
        ndarray, shape (n,)
            Predicted wavelengths in µm.
        """
        cs = np.asarray(cols, dtype=float)
        vs = self._v(np.asarray(orders, dtype=float))
        result = np.zeros_like(cs)
        for i in range(self.wdeg + 1):
            for j in range(self.odeg + 1):
                result += self.coeffs[i, j] * (cs ** i) * (vs ** j)
        return result

    def as_wavelength_func(self):
        """Return a callable ``wavelength_func(cols, order_number)``.

        The returned callable is compatible with the ``wavelength_func``
        parameter of
        :func:`~pyspextool.instruments.ishell.rectification_indices.build_rectification_indices`.

        Returns
        -------
        callable
            ``f(cols_array, order_number_scalar) -> wavelengths_array``
        """
        def _func(cols: npt.ArrayLike, order_number: float) -> npt.NDArray:
            cs = np.asarray(cols, dtype=float)
            return self.eval_array(cs, np.full_like(cs, float(order_number)))

        return _func


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def extract_order_arc_spectra(
    arc_img: npt.NDArray,
    trace: "FlatOrderTrace",
    wavecalinfo: "WaveCalInfo",
    *,
    aperture_half_width: int = 3,
) -> OrderArcSpectraSet:
    """Extract 1-D arc spectra using traced flat-field order geometry.

    For each echelle order, the traced centre-line polynomial (row as a
    function of column) is evaluated at every detector column.  A symmetric
    aperture of ``±aperture_half_width`` rows is averaged at each column
    to produce the 1-D arc spectrum.

    This function uses *only* the flat-field trace geometry from Stage 1.
    It does NOT estimate a centre row from the arc image itself.

    Parameters
    ----------
    arc_img : ndarray, shape (nrows, ncols)
        Combined arc-lamp image (output of
        :func:`~pyspextool.instruments.ishell.arc_tracing.load_and_combine_arcs`
        or equivalent).
    trace : :class:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace`
        Flat-field order-centre tracing result from Stage 1.  The
        ``center_poly_coeffs`` array (shape ``(n_orders, poly_degree+1)``)
        gives the column-to-row polynomial for each order.
    wavecalinfo : :class:`~pyspextool.instruments.ishell.calibrations.WaveCalInfo`
        Packaged calibration metadata for the mode.  Used to assign echelle
        order numbers and valid column ranges (``xranges``) to each traced
        order.  The *i*-th traced order is assumed to correspond to the
        *i*-th entry in ``wavecalinfo.orders`` (same ordering convention as
        the rest of the scaffold).
    aperture_half_width : int, default 3
        Half-width of the extraction aperture in pixels.  Pixels at rows
        ``[row_c - aperture_half_width, row_c + aperture_half_width]``
        (inclusive, clipped to detector bounds) are averaged for each
        column.

    Returns
    -------
    :class:`OrderArcSpectraSet`
        One :class:`OrderArcSpectrum` per order, in the same order as
        *trace*.

    Raises
    ------
    ValueError
        If *arc_img* is not 2-D, or if *aperture_half_width* < 1.

    Notes
    -----
    If the number of traced orders differs from the number of orders in
    ``wavecalinfo``, a :exc:`RuntimeWarning` is emitted and the shorter
    list is used.

    The extraction preserves column alignment: ``flux[k]`` for
    :class:`OrderArcSpectrum` corresponds to detector column
    ``col_start + k``.  No resampling is applied.
    """
    if arc_img.ndim != 2:
        raise ValueError(
            f"arc_img must be a 2-D array; got shape {arc_img.shape}"
        )
    if aperture_half_width < 1:
        raise ValueError(
            f"aperture_half_width must be >= 1; got {aperture_half_width}"
        )

    nrows, ncols = arc_img.shape
    arc = arc_img.astype(float)

    n_trace = trace.n_orders
    n_wci = wavecalinfo.n_orders
    n_orders = min(n_trace, n_wci)

    if n_trace != n_wci:
        warnings.warn(
            f"extract_order_arc_spectra: number of traced orders ({n_trace}) "
            f"differs from wavecalinfo orders ({n_wci}).  "
            f"Using the first {n_orders} orders.",
            RuntimeWarning,
            stacklevel=2,
        )

    mode = wavecalinfo.mode
    spectra: list[OrderArcSpectrum] = []

    for i in range(n_orders):
        order_num = int(wavecalinfo.orders[i])
        coeffs = trace.center_poly_coeffs[i]  # shape (poly_degree+1,)

        # Determine valid column range from wavecalinfo.xranges if available,
        # otherwise use the full detector width.
        if wavecalinfo.xranges is not None:
            col_start = int(wavecalinfo.xranges[i, 0])
            col_end = int(wavecalinfo.xranges[i, 1])
        else:
            col_start = 0
            col_end = ncols - 1

        col_start = max(0, col_start)
        col_end = min(ncols - 1, col_end)

        cols_int = np.arange(col_start, col_end + 1)
        n_cols = len(cols_int)

        # Evaluate the traced centre row at each column.
        center_rows = np.polynomial.polynomial.polyval(
            cols_int.astype(float), coeffs
        )  # shape (n_cols,)

        flux = np.empty(n_cols, dtype=float)

        for k, (col, row_c) in enumerate(zip(cols_int, center_rows)):
            row_lo = max(0, int(round(row_c)) - aperture_half_width)
            row_hi = min(nrows - 1, int(round(row_c)) + aperture_half_width)
            if row_lo > row_hi:
                flux[k] = np.nan
            else:
                flux[k] = float(np.mean(arc[row_lo: row_hi + 1, col]))

        spectra.append(
            OrderArcSpectrum(
                order_index=i,
                order_number=order_num,
                col_start=col_start,
                col_end=col_end,
                flux=flux,
            )
        )

    logger.info(
        "extract_order_arc_spectra: extracted %d spectra (aperture ±%d px)",
        len(spectra),
        aperture_half_width,
    )

    return OrderArcSpectraSet(
        mode=mode,
        spectra=spectra,
        aperture_half_width=aperture_half_width,
    )


def fit_1dxd_wavelength_model(
    spectra_set: OrderArcSpectraSet,
    wavecalinfo: "WaveCalInfo",
    line_list: "LineList",
    *,
    wdeg: int = 2,
    odeg: int = 1,
    min_prominence: float = 50.0,
    min_distance: int = 5,
    match_tol_um: float = 0.002,
    max_col_residual_px: float = 5.0,
    min_lines_per_order: int = 4,
    min_lines_total: int = 10,
    sigma_thresh: float = 3.0,
    max_sigma_iter: int = 5,
    xcorr_max_shift_px: int = 50,
    local_search_window_px: int = 20,
) -> IdlStyle1DXDModel:
    """Fit a global IDL-style 1DXD wavelength model across all echelle orders.

    For each order in *spectra_set*:

    1. **Cross-correlate** the extracted 1-D arc spectrum against a synthetic
       reference comb (Gaussians placed at each reference-line column position
       predicted by the coarse wavelength grid) to determine a per-order
       column shift.
    2. **Two-pass local expected-line search** — for each reference line
       expected in this order:

       * **Pass 1**: search within ``±adaptive_window`` pixels of the
         xcorr-shifted predicted column.  Collect provisional matches.
       * **Affine correction**: fit ``detected_col ≈ a × predicted_col + b``
         from the provisional matches (falls back to identity when fewer
         than 2 matches or if the slope is outside ``[0.95, 1.05]``).
       * **Pass 2**: re-run the same local search using the affine-corrected
         column predictions.  This second pass is the final match set used
         for the global fit.

       This two-pass approach corrects both the per-order translation (xcorr
       shift) and a small residual slope/stretch, improving match yield for
       orders where the coarse wavelength grid has a slight dispersion error.

    Then fit a single global 2-D polynomial across all matched points:

    .. math::

        \\lambda(\\text{col}, \\text{order}) =
            \\sum_{i=0}^{w} \\sum_{j=0}^{p}
            C_{i,j}\\, \\text{col}^{i}\\, v^{j}

    where ``v = order_ref / order``, ``order_ref`` is the minimum echelle
    order number, ``w = wdeg``, and ``p = odeg``.

    After the initial fit, **iterative sigma clipping** rejects points whose
    residual exceeds ``sigma_thresh × rms_residual``.  The fit is repeated
    on the surviving points until convergence or *max_sigma_iter* iterations.

    Parameters
    ----------
    spectra_set : :class:`OrderArcSpectraSet`
        1-D arc spectra from :func:`extract_order_arc_spectra`.
    wavecalinfo : :class:`~pyspextool.instruments.ishell.calibrations.WaveCalInfo`
        Packaged calibration metadata for coarse wavelength prediction.
    line_list : :class:`~pyspextool.instruments.ishell.calibrations.LineList`
        Reference arc-line list.
    wdeg : int, default 2
        Polynomial degree in detector column.
    odeg : int, default 1
        Polynomial degree in ``v = order_ref / order``.
    min_prominence : float, default 50.0
        Minimum local prominence (peak value minus window floor, in detector
        counts) required for a local peak to be accepted as an arc-line
        candidate.
    min_distance : int, default 5
        Kept for API compatibility; not used by the local search strategy.
    match_tol_um : float, default 0.002
        Kept for API compatibility; not used by the local search strategy.
    max_col_residual_px : float, default 5.0
        Kept for API compatibility; not used by the local search strategy.
    min_lines_per_order : int, default 4
        Minimum number of matched (post-monotonicity-filter) arc lines
        required for an order to contribute to the global fit.  Orders
        with fewer accepted matches are recorded in ``per_order_stats``
        with ``participated=False`` and are excluded from the global
        polynomial to avoid destabilising the fit with poorly constrained
        orders.
    min_lines_total : int, default 10
        Minimum total number of matched arc lines required across all orders.
        Raises :exc:`ValueError` if this threshold is not met.
    sigma_thresh : float, default 3.0
        Sigma-clipping threshold.  Points whose residual exceeds
        ``sigma_thresh × rms`` are rejected from the global fit.
    max_sigma_iter : int, default 5
        Maximum number of sigma-clipping iterations.  Iteration stops
        earlier if no new points are rejected.
    xcorr_max_shift_px : int, default 50
        Maximum absolute column shift (pixels) allowed by the
        cross-correlation.  Shifts larger than this are clipped to ±50.
    local_search_window_px : int, default 20
        Base half-width of the local search window in pixels.  The effective
        window used per order is:
        ``max(local_search_window_px, int(abs(xcorr_shift)) + 5)``,
        so that the window expands automatically when the cross-correlation
        shift is large (indicating a coarser-than-expected wavelength grid).

    Returns
    -------
    :class:`IdlStyle1DXDModel`
        Fitted global model with sigma-clipping statistics and per-order
        match metadata.

    Raises
    ------
    ValueError
        If *wdeg* or *odeg* is negative, or if fewer than *min_lines_total*
        arc lines are matched (before sigma clipping).

    Warns
    -----
    RuntimeWarning
        If an order has no valid coarse reference grid and is skipped.
    """
    if wdeg < 0:
        raise ValueError(f"wdeg must be >= 0; got {wdeg}")
    if odeg < 0:
        raise ValueError(f"odeg must be >= 0; got {odeg}")

    # ------------------------------------------------------------------
    # Collect (col, order_num, ref_wavelength) triplets across all orders
    # ------------------------------------------------------------------
    all_cols: list[float] = []
    all_orders: list[float] = []
    all_wavs: list[float] = []
    orders_with_matches: set[int] = set()
    per_order_stats: list[OrderMatchStats] = []

    for spec in spectra_set.spectra:
        order_num = spec.order_number
        order_idx = spec.order_index

        # Build coarse col→wavelength reference for this order
        coarse_cols, coarse_wavs = _build_coarse_lookup_1d(
            wavecalinfo, order_idx
        )
        if len(coarse_cols) == 0:
            warnings.warn(
                f"Order {order_num}: no valid coarse reference grid; "
                "skipping line matching for this order.",
                RuntimeWarning,
                stacklevel=2,
            )
            per_order_stats.append(OrderMatchStats(
                order_number=order_num, xcorr_shift_px=0.0,
                n_candidate=0, n_matched=0, n_ambiguous_removed=0,
                n_monotonic_removed=0,
                n_accepted=0, n_rejected=0,
                rms_resid_um=float("nan"), participated=False,
                xcorr_shift_clipped=False,
                skipped_insufficient_matches=False,
                min_lines_required=min_lines_per_order,
            ))
            continue

        # Reference line wavelengths for this order
        ref_entries = _get_ref_entries(line_list, order_num)

        # Find peaks in the 1D arc spectrum
        flux = spec.flux
        valid_mask = np.isfinite(flux)
        if not valid_mask.any():
            per_order_stats.append(OrderMatchStats(
                order_number=order_num, xcorr_shift_px=0.0,
                n_candidate=0, n_matched=0, n_ambiguous_removed=0,
                n_monotonic_removed=0,
                n_accepted=0, n_rejected=0,
                rms_resid_um=float("nan"), participated=False,
                xcorr_shift_clipped=False,
                skipped_insufficient_matches=False,
                min_lines_required=min_lines_per_order,
            ))
            continue

        # ------------------------------------------------------------------
        # Step 1: Cross-correlate extracted spectrum against reference comb
        # to find the per-order column shift.
        # ------------------------------------------------------------------
        xcorr_shift, xcorr_shift_clipped = _xcorr_order_shift(
            flux, coarse_cols, coarse_wavs, ref_entries,
            spec.col_start, max_shift_px=xcorr_max_shift_px,
        )
        logger.debug(
            "Order %d: xcorr shift = %.2f px", order_num, xcorr_shift
        )

        # Shift the coarse column grid to align with the extracted spectrum.
        # A positive shift means the extracted spectrum is shifted right
        # relative to the reference; we subtract the shift from predicted
        # columns to compensate.
        shifted_coarse_cols = coarse_cols + xcorr_shift

        # Expand the search window when xcorr reports a large shift — this
        # compensates for coarse-grid prediction errors that exceed the base
        # window size.
        adaptive_window = max(local_search_window_px, int(abs(xcorr_shift)) + 5)

        # ------------------------------------------------------------------
        # Step 2a: Pass 1 — local expected-line search with xcorr-shifted grid.
        # For each reference line, search within ±adaptive_window pixels of
        # its predicted column.  Results feed the per-order affine correction.
        # ------------------------------------------------------------------
        pass1_candidates, _ = _find_local_line_peaks(
            flux,
            spec.col_start,
            shifted_coarse_cols,
            coarse_wavs,
            ref_entries,
            local_window_px=adaptive_window,
            min_prominence=min_prominence,
        )

        # ------------------------------------------------------------------
        # Step 2b: Fit a per-order affine column correction from pass-1 results.
        # col_corrected = affine_a * shifted_coarse_col + affine_b
        # Falls back to identity (a=1, b=0) when < 2 provisional matches or
        # when the fitted slope is outside [0.95, 1.05].
        # ------------------------------------------------------------------
        affine_a, affine_b, affine_applied, affine_n_pts = _fit_affine_col_correction(
            pass1_candidates, shifted_coarse_cols, coarse_wavs,
        )
        affine_coarse_cols = affine_a * shifted_coarse_cols + affine_b
        logger.debug(
            "Order %d: affine correction a=%.5f b=%.2f applied=%s (n_pts=%d)",
            order_num, affine_a, affine_b, affine_applied, affine_n_pts,
        )

        # ------------------------------------------------------------------
        # Step 2c: Pass 2 — local expected-line search with affine-corrected grid.
        # This is the final match set used for the global fit.
        # ------------------------------------------------------------------
        candidates, local_diag = _find_local_line_peaks(
            flux,
            spec.col_start,
            affine_coarse_cols,
            coarse_wavs,
            ref_entries,
            local_window_px=adaptive_window,
            min_prominence=min_prominence,
        )
        n_candidate = local_diag["n_windows_with_peak"]
        if n_candidate == 0:
            logger.debug("Order %d: no local peaks found near reference lines", order_num)
            per_order_stats.append(OrderMatchStats(
                order_number=order_num, xcorr_shift_px=xcorr_shift,
                n_candidate=0, n_matched=0, n_ambiguous_removed=0,
                n_monotonic_removed=0,
                n_accepted=0, n_rejected=0,
                rms_resid_um=float("nan"), participated=False,
                xcorr_shift_clipped=xcorr_shift_clipped,
                skipped_insufficient_matches=False,
                min_lines_required=min_lines_per_order,
                affine_a=affine_a, affine_b=affine_b,
                affine_applied=affine_applied, affine_n_points=affine_n_pts,
            ))
            continue

        # The local search already produces matched (col, ref_wav) pairs — one
        # per reference line — so n_ambiguous_removed is 0 by construction.
        n_ambiguous_removed = 0
        matches = candidates

        # Enforce monotonicity: wavelength must increase with column
        matches, n_monotonic_removed = _enforce_monotonic_matches(matches)

        n_matched = len(matches)
        if not matches:
            logger.debug(
                "Order %d: candidate=%d matched=%d used=0 rejected=0 "
                "(ambig_removed=%d, mono_removed=%d)",
                order_num, n_candidate, n_matched,
                n_ambiguous_removed, n_monotonic_removed,
            )
            per_order_stats.append(OrderMatchStats(
                order_number=order_num, xcorr_shift_px=xcorr_shift,
                n_candidate=n_candidate, n_matched=0,
                n_ambiguous_removed=n_ambiguous_removed,
                n_monotonic_removed=n_monotonic_removed,
                n_accepted=0, n_rejected=0,
                rms_resid_um=float("nan"), participated=False,
                xcorr_shift_clipped=xcorr_shift_clipped,
                skipped_insufficient_matches=False,
                min_lines_required=min_lines_per_order,
                affine_a=affine_a, affine_b=affine_b,
                affine_applied=affine_applied, affine_n_points=affine_n_pts,
            ))
            continue

        # ------------------------------------------------------------------
        # Per-order minimum-match threshold: exclude weakly constrained
        # orders from the global fit to avoid destabilising the polynomial.
        # If local search produces too few matches, try the fallback global
        # peak search before giving up.
        # ------------------------------------------------------------------
        used_fallback = False
        fallback_n_matches = 0
        if n_matched < min_lines_per_order:
            logger.debug(
                "Order %d: local matches insufficient (%d < %d); "
                "trying fallback global peak search",
                order_num, n_matched, min_lines_per_order,
            )
            fallback_matches = _fallback_global_line_match(
                flux,
                spec.col_start,
                affine_coarse_cols,
                coarse_wavs,
                ref_entries,
                min_prominence=min_prominence,
            )
            fallback_n_matches = len(fallback_matches)
            logger.debug(
                "Order %d: fallback used: %s, fallback matches = %d",
                order_num, fallback_n_matches >= min_lines_per_order,
                fallback_n_matches,
            )
            if fallback_n_matches >= min_lines_per_order:
                used_fallback = True
                matches = fallback_matches
                n_matched = fallback_n_matches
            else:
                logger.debug(
                    "Order %d skipped: fallback insufficient (%d < %d)",
                    order_num, fallback_n_matches, min_lines_per_order,
                )
                per_order_stats.append(OrderMatchStats(
                    order_number=order_num, xcorr_shift_px=xcorr_shift,
                    n_candidate=n_candidate, n_matched=n_matched,
                    n_ambiguous_removed=n_ambiguous_removed,
                    n_monotonic_removed=n_monotonic_removed,
                    n_accepted=0, n_rejected=0,
                    rms_resid_um=float("nan"), participated=False,
                    xcorr_shift_clipped=xcorr_shift_clipped,
                    skipped_insufficient_matches=True,
                    min_lines_required=min_lines_per_order,
                    affine_a=affine_a, affine_b=affine_b,
                    affine_applied=affine_applied, affine_n_points=affine_n_pts,
                    used_fallback=False, fallback_n_matches=fallback_n_matches,
                ))
                continue

        for col_m, wav_m in matches:
            all_cols.append(col_m)
            all_orders.append(float(order_num))
            all_wavs.append(wav_m)

        orders_with_matches.add(order_num)

        # Store preliminary stats (n_accepted / n_rejected updated after sigma clip)
        per_order_stats.append(OrderMatchStats(
            order_number=order_num, xcorr_shift_px=xcorr_shift,
            n_candidate=n_candidate, n_matched=n_matched,
            n_ambiguous_removed=n_ambiguous_removed,
            n_monotonic_removed=n_monotonic_removed,
            n_accepted=n_matched, n_rejected=0,
            rms_resid_um=float("nan"), participated=True,
            xcorr_shift_clipped=xcorr_shift_clipped,
            affine_a=affine_a, affine_b=affine_b,
            affine_applied=affine_applied, affine_n_points=affine_n_pts,
            skipped_insufficient_matches=False,
            min_lines_required=min_lines_per_order,
            used_fallback=used_fallback, fallback_n_matches=fallback_n_matches,
        ))

        logger.debug(
            "Order %d: xcorr=%.1fpx, candidate=%d ambig_removed=%d "
            "mono_removed=%d matched=%d",
            order_num, xcorr_shift, n_candidate,
            n_ambiguous_removed, n_monotonic_removed, n_matched,
        )

    n_lines_total = len(all_cols)
    n_orders_fit = len(orders_with_matches)

    logger.info(
        "fit_1dxd_wavelength_model: %d lines from %d orders (before sigma clip)",
        n_lines_total,
        n_orders_fit,
    )

    if n_lines_total < min_lines_total:
        raise ValueError(
            f"Only {n_lines_total} arc lines matched (minimum is {min_lines_total}).  "
            "Try reducing match_tol_um or min_prominence."
        )

    cols_arr = np.array(all_cols, dtype=float)
    orders_arr = np.array(all_orders, dtype=float)
    wavs_arr = np.array(all_wavs, dtype=float)

    # ------------------------------------------------------------------
    # Build design matrix
    # ------------------------------------------------------------------
    order_ref = float(np.min(orders_arr))
    vs = order_ref / orders_arr  # normalised inverse-order coordinate

    n_terms = (wdeg + 1) * (odeg + 1)

    def _build_design(c_arr, v_arr):
        X = np.empty((len(c_arr), n_terms), dtype=float)
        idx = 0
        for ii in range(wdeg + 1):
            for jj in range(odeg + 1):
                X[:, idx] = (c_arr ** ii) * (v_arr ** jj)
                idx += 1
        return X

    # ------------------------------------------------------------------
    # Iterative sigma clipping
    # ------------------------------------------------------------------
    accepted_mask = np.ones(n_lines_total, dtype=bool)

    for _iter in range(max_sigma_iter):
        X_sub = _build_design(cols_arr[accepted_mask], vs[accepted_mask])
        c_flat, _, _, _ = np.linalg.lstsq(X_sub, wavs_arr[accepted_mask], rcond=None)
        residuals_all = wavs_arr - (_build_design(cols_arr, vs) @ c_flat)
        rms_acc = float(np.sqrt(np.mean(residuals_all[accepted_mask] ** 2)))
        if rms_acc == 0.0:
            break
        new_mask = accepted_mask & (np.abs(residuals_all) <= sigma_thresh * rms_acc)
        n_newly_rejected = int(np.sum(accepted_mask) - np.sum(new_mask))
        accepted_mask = new_mask
        logger.debug(
            "Sigma clip iter %d: rms=%.4f nm, rejected %d new points (%d total)",
            _iter + 1, rms_acc * 1e3, n_newly_rejected,
            int(np.sum(~accepted_mask)),
        )
        if n_newly_rejected == 0:
            break

    # Final fit on accepted points
    X_final = _build_design(cols_arr[accepted_mask], vs[accepted_mask])
    c_flat_final, _, _, _ = np.linalg.lstsq(
        X_final, wavs_arr[accepted_mask], rcond=None
    )
    coeffs = c_flat_final.reshape(wdeg + 1, odeg + 1)

    # Final residuals (for accepted points only)
    residuals_final = wavs_arr[accepted_mask] - (X_final @ c_flat_final)
    fit_rms_um = float(np.sqrt(np.mean(residuals_final ** 2)))
    median_residual_um = float(np.median(residuals_final))

    n_lines_accepted = int(np.sum(accepted_mask))
    n_lines_rejected = n_lines_total - n_lines_accepted

    # ------------------------------------------------------------------
    # Update per-order stats with final residuals and sigma-clip counts
    # ------------------------------------------------------------------
    orders_arr_all = orders_arr  # shape (n_lines_total,)
    residuals_all_final = wavs_arr - (_build_design(cols_arr, vs) @ c_flat_final)

    for stat in per_order_stats:
        if not stat.participated:
            continue
        order_mask = orders_arr_all == float(stat.order_number)
        accepted_order = accepted_mask & order_mask
        rejected_order = (~accepted_mask) & order_mask
        n_acc = int(np.sum(accepted_order))
        n_rej = int(np.sum(rejected_order))
        rms_order = (
            float(np.sqrt(np.mean(residuals_all_final[accepted_order] ** 2)))
            if n_acc > 0 else float("nan")
        )
        # Mutate (OrderMatchStats is a dataclass, not frozen)
        stat.n_accepted = n_acc
        stat.n_rejected = n_rej
        stat.rms_resid_um = rms_order
        stat.participated = n_acc > 0
        logger.debug(
            "Order %d: candidate=%d matched=%d used=%d rejected=%d",
            stat.order_number, stat.n_candidate, stat.n_matched,
            stat.n_accepted, stat.n_rejected,
        )

    # Rebuild orders_with_matches from accepted points only
    fitted_orders_final = sorted(
        int(o) for o in set(orders_arr[accepted_mask].tolist())
    )
    n_orders_fit_final = len(fitted_orders_final)

    # ------------------------------------------------------------------
    # Build per-point arrays for reproducible QA
    # ------------------------------------------------------------------
    # matched_fit_wavelength_um: predicted wavelength from the final fit
    matched_fit_wavs = _build_design(cols_arr, vs) @ c_flat_final

    logger.info(
        "fit_1dxd_wavelength_model: wdeg=%d, odeg=%d, "
        "total=%d acc=%d rej=%d rms=%.4f nm",
        wdeg, odeg, n_lines_total, n_lines_accepted,
        n_lines_rejected, fit_rms_um * 1e3,
    )

    return IdlStyle1DXDModel(
        mode=spectra_set.mode,
        wdeg=wdeg,
        odeg=odeg,
        order_ref=order_ref,
        coeffs=coeffs,
        fitted_order_numbers=fitted_orders_final,
        fit_rms_um=fit_rms_um,
        n_lines=n_lines_accepted,
        n_lines_total=n_lines_total,
        n_lines_rejected=n_lines_rejected,
        accepted_mask=accepted_mask,
        median_residual_um=median_residual_um,
        n_orders_fit=n_orders_fit_final,
        per_order_stats=per_order_stats,
        matched_cols_px=cols_arr,
        matched_order_numbers=orders_arr,
        matched_ref_wavelength_um=wavs_arr,
        matched_fit_wavelength_um=matched_fit_wavs,
        matched_residual_um=residuals_all_final,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _xcorr_order_shift(
    flux: npt.NDArray,
    coarse_cols: npt.NDArray,
    coarse_wavs: npt.NDArray,
    ref_entries: list[tuple[float, str]],
    col_start: int,
    *,
    max_shift_px: int = 50,
    fwhm_px: float = 3.0,
) -> tuple[float, bool]:
    """Compute the per-order column shift via cross-correlation.

    Builds a synthetic reference comb from the expected arc-line column
    positions (Gaussians of width *fwhm_px* at each reference-line column
    predicted by the coarse wavelength grid) and cross-correlates it
    against the extracted 1-D arc spectrum.

    The shift is found from the peak of the cross-correlation within
    ``±max_shift_px`` pixels.  Sub-pixel accuracy is obtained by
    fitting a parabola through the three points around the peak.

    The returned shift is always within ``[-max_shift_px, max_shift_px]``.
    The boolean flag indicates whether the correlation peak landed at the
    boundary of the search window — which suggests the true peak is outside
    the allowed shift range and the result should be treated with caution.

    Parameters
    ----------
    flux : ndarray, shape (n_cols,)
        Extracted 1-D arc spectrum for this order.
    coarse_cols, coarse_wavs : ndarray
        Coarse column→wavelength lookup from the packaged ``WaveCalInfo``.
    ref_entries : list of (float, str)
        Reference line ``(wavelength_um, species)`` pairs for this order.
    col_start : int
        Detector column corresponding to ``flux[0]``.
    max_shift_px : int, default 50
        Maximum absolute shift in pixels to search.
    fwhm_px : float, default 3.0
        FWHM of the Gaussian used for each reference line in the comb.

    Returns
    -------
    shift : float
        Cross-correlation shift in pixels, clipped to
        ``[-max_shift_px, max_shift_px]``.  A positive value means the
        extracted spectrum is shifted *right* relative to the reference.
        Returns ``0.0`` if no reference lines are available, the coarse
        grid is empty, or correlation fails.
    was_clipped : bool
        ``True`` if the correlation peak was found at the boundary of the
        search window (index 0 or ``len(window)-1``), which indicates the
        true peak is likely outside the allowed shift range and the returned
        shift has been constrained.  ``False`` in the normal case, including
        early-return paths that yield ``0.0``.
    """
    if len(ref_entries) == 0 or len(coarse_cols) == 0:
        return 0.0, False

    n = len(flux)
    if n == 0:
        return 0.0, False

    # Replace NaNs with zero for correlation
    flux_clean = np.where(np.isfinite(flux), flux, 0.0)
    # Remove DC offset
    flux_clean = flux_clean - float(np.mean(flux_clean))

    # Build synthetic reference comb: Gaussian at each expected line column
    ref_wavs = np.array([e[0] for e in ref_entries], dtype=float)
    sigma = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    col_min = float(coarse_cols[0])
    col_max = float(coarse_cols[-1])

    cols_abs = col_start + np.arange(n, dtype=float)  # absolute detector columns
    comb = np.zeros(n, dtype=float)
    n_lines_in_comb = 0
    for wav in ref_wavs:
        # Predicted column for this reference line
        if wav < float(coarse_wavs[0]) or wav > float(coarse_wavs[-1]):
            continue
        pred_col = float(np.interp(wav, coarse_wavs, coarse_cols))
        if pred_col < col_min or pred_col > col_max:
            continue
        # Add Gaussian at that column (in local index space)
        pred_idx = pred_col - col_start
        comb += np.exp(-0.5 * ((cols_abs - col_start - pred_idx) / sigma) ** 2)
        n_lines_in_comb += 1

    if n_lines_in_comb == 0:
        return 0.0, False

    # Full cross-correlation (mode="full") gives a (2n-1,)-length result.
    # The zero-lag is at index n-1.
    xcorr = _scipy_correlate(flux_clean, comb, mode="full")
    zero_lag = n - 1
    lo = max(0, zero_lag - max_shift_px)
    hi = min(len(xcorr) - 1, zero_lag + max_shift_px)
    xcorr_window = xcorr[lo: hi + 1]

    peak_idx_local = int(np.argmax(xcorr_window))
    peak_idx_global = lo + peak_idx_local
    # True when the correlation argmax is at the boundary of the search window,
    # indicating the true shift is likely outside the allowed range.
    was_clipped = (peak_idx_local == 0 or peak_idx_local == len(xcorr_window) - 1)

    # Sub-pixel refinement via parabolic fit through three points
    if 0 < peak_idx_global < len(xcorr) - 1:
        y0 = xcorr[peak_idx_global - 1]
        y1 = xcorr[peak_idx_global]
        y2 = xcorr[peak_idx_global + 1]
        denom = 2.0 * y1 - y0 - y2
        if denom != 0.0:
            sub_shift = 0.5 * (y2 - y0) / denom
        else:
            sub_shift = 0.0
    else:
        sub_shift = 0.0

    # Raw subpixel shift relative to zero lag
    raw_shift = float(peak_idx_global - zero_lag) + sub_shift
    # Clip to allowed window (raw_shift could slightly exceed ±max_shift_px
    # due to the sub-pixel parabolic correction at the boundary)
    clipped_shift = float(np.clip(raw_shift, -max_shift_px, max_shift_px))
    return clipped_shift, was_clipped


def _build_coarse_lookup_1d(
    wavecalinfo: "WaveCalInfo",
    order_idx: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Return ``(cols, wavs)`` for the coarse column→wavelength lookup.

    Reads plane 0 of the ``WaveCalInfo`` data cube (confirmed to store
    wavelengths in µm).  NaN values are excluded.  The column mapping is::

        col = xranges[order_idx, 0] + array_index

    Returns empty arrays if no valid data exist.
    """
    wav_array = wavecalinfo.data[order_idx, 0, :]
    valid = np.isfinite(wav_array)
    if not valid.any():
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    x_start = int(wavecalinfo.xranges[order_idx, 0])
    wavs = wav_array[valid]
    idxs = np.where(valid)[0].astype(float)
    cols = idxs + x_start
    return cols, wavs


def _get_ref_entries(
    line_list: "LineList",
    order_number: int,
) -> list[tuple[float, str]]:
    """Return ``(wavelength_um, species)`` pairs from *line_list* for *order_number*.

    An empty list is returned if no entries are present for this order.
    """
    entries = [
        (float(e.wavelength_um), str(e.species))
        for e in line_list.entries
        if e.order == order_number
    ]
    return entries


def _match_1d_peaks(
    peak_cols: npt.NDArray,
    coarse_cols: npt.NDArray,
    coarse_wavs: npt.NDArray,
    ref_entries: list[tuple[float, str]],
    match_tol_um: float,
    max_col_residual_px: float = 5.0,
) -> tuple[list[tuple[float, float]], int]:
    """Match detected 1-D peak positions to reference wavelengths.

    For each peak column, predicts a wavelength by linear interpolation on
    the coarse grid, then finds the nearest reference line.  A match is
    accepted only when **both** criteria are satisfied:

    1. ``|predicted_wavelength - reference_wavelength| < match_tol_um``
    2. ``|peak_col - predicted_col_for_ref_line| < max_col_residual_px``

    The second criterion uses the coarse grid in reverse
    (wavelength → predicted column) to ensure the peak column is
    consistent with where the reference line is expected on the detector.
    This prevents incorrect associations when the coarse grid is slightly
    offset or peaks are dense.

    Deduplicates: if two peaks match the same reference line, only the
    one with the smallest wavelength residual is retained.

    Ambiguity rejection: if a reference line has two detected peaks within
    both tolerances, the match is rejected entirely (both candidates are
    dropped) rather than keeping the closer one.  This prevents a single
    bright reference line from pulling in wrong peak columns.

    Parameters
    ----------
    peak_cols : ndarray
        Detector columns of detected peaks.
    coarse_cols, coarse_wavs : ndarray
        Coarse reference grid from ``WaveCalInfo``.
    ref_entries : list of (float, str)
        Reference line ``(wavelength_um, species)`` pairs for this order.
    match_tol_um : float
        Maximum allowed wavelength residual (µm) for a match to be accepted.
    max_col_residual_px : float, default 5.0
        Maximum allowed column residual (pixels) between the detected peak
        column and the predicted detector column for the reference line.

    Returns
    -------
    matches : list of (col, ref_wavelength_um)
        Accepted matches after deduplication and ambiguity rejection.
    n_ambiguous_removed : int
        Number of reference lines that were flagged as ambiguous (i.e. two
        or more detected peaks fell within both tolerances of the same
        reference line).  Each such reference line contributes zero accepted
        matches regardless of how many peaks claimed it.
    """
    if len(ref_entries) == 0 or len(coarse_cols) == 0:
        return [], 0

    ref_wavs = np.array([e[0] for e in ref_entries], dtype=float)

    col_min = float(coarse_cols[0])
    col_max = float(coarse_cols[-1])

    # best: ref_idx → (col, ref_wav, wav_residual)
    best: dict[int, tuple[float, float, float]] = {}
    # ambiguous: ref indices that had more than one candidate within tolerance
    ambiguous: set[int] = set()

    for col in peak_cols:
        col = float(col)
        if col < col_min or col > col_max:
            continue

        pred_wav = float(np.interp(col, coarse_cols, coarse_wavs))
        diffs = np.abs(ref_wavs - pred_wav)
        best_ref_idx = int(np.argmin(diffs))
        wav_residual = float(diffs[best_ref_idx])

        if wav_residual > match_tol_um:
            continue

        # Column consistency check: predicted column for the reference line
        # must be within max_col_residual_px of the detected peak column.
        ref_wav = float(ref_wavs[best_ref_idx])
        pred_col_for_ref = float(np.interp(ref_wav, coarse_wavs, coarse_cols))
        col_residual = abs(col - pred_col_for_ref)
        if col_residual >= max_col_residual_px:
            continue

        if best_ref_idx in ambiguous:
            # Already flagged ambiguous; ignore any further candidates
            continue

        if best_ref_idx in best:
            # Second candidate for this reference line — flag as ambiguous
            ambiguous.add(best_ref_idx)
            del best[best_ref_idx]
        else:
            best[best_ref_idx] = (col, ref_wav, wav_residual)

    n_ambiguous_removed = len(ambiguous)
    matches = [(col, wav) for col, wav, _ in best.values()]
    return matches, n_ambiguous_removed


def _enforce_monotonic_matches(
    matches: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], int]:
    """Remove non-monotonic matches so wavelength increases with column.

    Sorts the match list by detector column and then applies a greedy
    longest-increasing-subsequence filter on the assigned reference
    wavelengths.  Any match whose wavelength is not strictly greater than
    the previous accepted wavelength is removed.

    Parameters
    ----------
    matches : list of (col, ref_wavelength_um)
        Candidate matches, in any order.

    Returns
    -------
    filtered : list of (col, ref_wavelength_um)
        Matches with non-monotonic entries removed, sorted by column.
    n_removed : int
        Number of matches removed.
    """
    if len(matches) <= 1:
        return list(matches), 0

    # Sort by column
    sorted_matches = sorted(matches, key=lambda m: m[0])

    # Greedy forward pass: keep a match only if its wavelength is strictly
    # greater than the last accepted wavelength.
    filtered: list[tuple[float, float]] = []
    last_wav = -float("inf")
    for col, wav in sorted_matches:
        if wav > last_wav:
            filtered.append((col, wav))
            last_wav = wav

    n_removed = len(sorted_matches) - len(filtered)
    return filtered, n_removed


def _fit_affine_col_correction(
    candidates: list[tuple[float, float]],
    coarse_cols: npt.NDArray,
    coarse_wavs: npt.NDArray,
    *,
    a_lo: float = 0.95,
    a_hi: float = 1.05,
) -> tuple[float, float, bool, int]:
    """Fit an affine column correction from provisional first-pass candidates.

    Given provisional ``(peak_col, ref_wav)`` pairs from the first-pass local
    expected-line search, fits the affine mapping::

        detected_col ≈ a * predicted_col + b

    where ``predicted_col = interp(ref_wav, coarse_wavs, coarse_cols)``.

    Parameters
    ----------
    candidates : list of (peak_col, ref_wav)
        Provisional matches from the first-pass local search.
    coarse_cols, coarse_wavs : ndarray
        Coarse column→wavelength lookup (already xcorr-shifted).
    a_lo, a_hi : float, default (0.95, 1.05)
        Allowed range for the fitted slope.  If the slope falls outside this
        range (pathological fit), the identity transform is returned.

    Returns
    -------
    a : float
        Fitted slope (``1.0`` for the identity transform).
    b : float
        Fitted intercept in pixels (``0.0`` for the identity transform).
    applied : bool
        ``True`` if a valid non-identity affine correction was fitted.
    n_points : int
        Number of provisional candidate matches used in the fit.
    """
    n_pts = len(candidates)
    if n_pts < 2:
        return 1.0, 0.0, False, n_pts

    pred_cols = np.array(
        [float(np.interp(ref_wav, coarse_wavs, coarse_cols)) for _, ref_wav in candidates],
        dtype=float,
    )
    detected_cols = np.array([float(peak_col) for peak_col, _ in candidates], dtype=float)

    # Fit detected_col = a * pred_col + b via least squares
    X = np.column_stack([pred_cols, np.ones(n_pts)])
    try:
        coeffs_fit, _residuals, _rank, _singular = np.linalg.lstsq(X, detected_cols, rcond=None)
        a, b = float(coeffs_fit[0]), float(coeffs_fit[1])
    except (np.linalg.LinAlgError, ValueError):
        return 1.0, 0.0, False, n_pts

    if not (a_lo <= a <= a_hi):
        return 1.0, 0.0, False, n_pts

    return a, b, True, n_pts


def _find_local_line_peaks(
    flux: npt.NDArray,
    col_start: int,
    shifted_coarse_cols: npt.NDArray,
    coarse_wavs: npt.NDArray,
    ref_entries: list[tuple[float, str]],
    *,
    local_window_px: int = 20,
    min_prominence: float = 50.0,
) -> tuple[list[tuple[float, float]], dict]:
    """Search locally near each expected reference-line column for a peak.

    For each reference line in *ref_entries*, predicts its detector column by
    interpolating on the (shifted) coarse wavelength grid, then searches
    within ``±local_window_px`` pixels of that predicted column for the
    strongest local peak.  This is the IDL-style approach: instead of
    detecting all peaks globally and then matching against a reference list,
    we search *only* near where each line is expected, avoiding the ambiguity
    that arises when hundreds of global peaks are matched against a sparse
    reference list.

    Peak selection is a two-step process:

    1. **Smoothing** — a 3-pixel box-car is applied to suppress single-pixel
       noise spikes before candidate identification.
    2. **Distance+height scoring** — the top-3 candidates by smoothed
       amplitude are scored by ``raw_height - 2 × distance_from_prediction``.
       The highest-scoring candidate is selected.  This prefers peaks that are
       both strong *and* close to the predicted position, so a slightly taller
       spurious peak that is well off-centre will not mask the genuine line.

    After the best candidate is selected, a **local-maximum condition** is
    applied: the candidate pixel must be ≥ both immediate neighbours in the
    raw flux, ensuring we are on a genuine peak crest rather than a plateau
    shoulder.

    Prominence is evaluated on the **raw** (unsmoothed) window flux so that
    the threshold retains its physical meaning in detector counts.

    Parameters
    ----------
    flux : ndarray, shape (n_cols,)
        Extracted 1-D arc spectrum for this order.
    col_start : int
        Detector column corresponding to ``flux[0]``.
    shifted_coarse_cols : ndarray
        Coarse column lookup already shifted by the xcorr-derived offset.
    coarse_wavs : ndarray
        Coarse wavelength values (µm) corresponding to *shifted_coarse_cols*.
        Must be monotonically ordered for :func:`numpy.interp` to work
        correctly.
    ref_entries : list of (float, str)
        Reference line ``(wavelength_um, species)`` pairs for this order.
    local_window_px : int, default 20
        Half-width of the local search window in pixels.  For each reference
        line the window spans
        ``[predicted_col - local_window_px, predicted_col + local_window_px]``.
    min_prominence : float, default 50.0
        Minimum local prominence (peak value minus window floor, in detector
        counts) required for a peak to be accepted as a candidate.
        Prominence is measured on the raw (unsmoothed) window flux of the
        selected candidate.

    Returns
    -------
    candidates : list of (col, ref_wavelength_um)
        One entry per reference line for which a convincing local peak was
        found.  ``col`` is the detector column of the peak (float).
    diagnostics : dict
        Contains three integer counts:

        ``n_reference_lines_considered``
            Number of reference lines whose predicted column fell within the
            valid flux array range.
        ``n_windows_with_peak``
            Number of those windows that yielded a convincing local peak
            (i.e., local prominence ≥ *min_prominence*).
        ``n_windows_empty``
            Number of windows searched but yielding no convincing peak
            (``n_reference_lines_considered - n_windows_with_peak``).
    """
    candidates: list[tuple[float, float]] = []
    n_considered = 0
    n_with_peak = 0
    n_flux = len(flux)

    if n_flux == 0 or len(ref_entries) == 0 or len(shifted_coarse_cols) == 0:
        return [], {
            "n_reference_lines_considered": 0,
            "n_windows_with_peak": 0,
            "n_windows_empty": 0,
        }

    wav_lo = float(np.min(coarse_wavs))
    wav_hi = float(np.max(coarse_wavs))

    for ref_wav, _species in ref_entries:
        ref_wav = float(ref_wav)
        # Skip reference lines outside the coarse wavelength range
        if ref_wav < wav_lo or ref_wav > wav_hi:
            continue

        # Predict detector column for this reference line via interpolation
        pred_col = float(np.interp(ref_wav, coarse_wavs, shifted_coarse_cols))

        # Convert predicted column to a flux-array index and build the window
        pred_idx = pred_col - col_start
        pred_idx_int = int(round(pred_idx))
        lo_idx = max(0, pred_idx_int - local_window_px)
        hi_idx = min(n_flux - 1, pred_idx_int + local_window_px)
        if lo_idx > hi_idx:
            continue

        n_considered += 1
        window_flux = flux[lo_idx: hi_idx + 1]
        finite_mask = np.isfinite(window_flux)
        if not finite_mask.any():
            continue

        # Smooth the window with a 3-pixel box-car to suppress single-pixel
        # noise spikes.  Smoothed signal is used only for candidate ranking;
        # raw flux is used for the distance+height score and prominence.
        raw_flux = np.where(finite_mask, window_flux, 0.0)
        if len(raw_flux) >= 3:
            kernel = np.ones(3) / 3.0
            smoothed = np.convolve(raw_flux, kernel, mode="same")
            # Edges of the convolution are less reliable; reset to raw there
            smoothed[0] = raw_flux[0]
            smoothed[-1] = raw_flux[-1]
        else:
            smoothed = raw_flux

        # Step 1: pick the top-3 candidates by smoothed amplitude.
        smooth_masked = np.where(finite_mask, smoothed, -np.inf)
        n_top = min(3, len(raw_flux))
        top_indices = np.argsort(smooth_masked)[-n_top:]

        # Step 2: score each candidate by raw height minus a 2-per-pixel distance
        # penalty from the predicted column.  This prefers peaks that are
        # both strong and close to the expected line position.
        pred_idx_float = pred_col - col_start  # fractional flux-array index
        candidate_scores: list[tuple[float, int]] = []
        for i in top_indices:
            distance = abs((lo_idx + i) - pred_idx_float)
            height = float(raw_flux[i])
            candidate_scores.append((height - 2.0 * distance, int(i)))

        # Step 3: select the highest-scoring candidate.
        peak_local_idx = max(candidate_scores)[1]

        # Step 4: enforce local-maximum condition — the selected pixel must be
        # ≥ both immediate neighbours so we land on a genuine peak crest.
        if 0 < peak_local_idx < len(raw_flux) - 1:
            if not (raw_flux[peak_local_idx] >= raw_flux[peak_local_idx - 1]
                    and raw_flux[peak_local_idx] >= raw_flux[peak_local_idx + 1]):
                continue

        # Step 5: measure prominence on the raw window flux.
        peak_val = float(window_flux[peak_local_idx])
        window_floor = float(np.nanmin(window_flux))
        local_prominence = peak_val - window_floor

        if local_prominence < min_prominence:
            # No convincing peak in this window; skip this reference line
            continue

        n_with_peak += 1
        peak_col = float(col_start + lo_idx + peak_local_idx)
        candidates.append((peak_col, ref_wav))

    n_empty = n_considered - n_with_peak
    return candidates, {
        "n_reference_lines_considered": n_considered,
        "n_windows_with_peak": n_with_peak,
        "n_windows_empty": n_empty,
    }


def _fallback_global_line_match(
    flux: npt.NDArray,
    col_start: int,
    affine_coarse_cols: npt.NDArray,
    coarse_wavs: npt.NDArray,
    ref_entries: list[tuple[float, str]],
    *,
    min_prominence: float = 50.0,
    min_distance: int = 5,
    match_tol_um: float = 0.0005,
) -> list[tuple[float, float]]:
    """Fallback global peak search for orders where local search fails.

    Detects peaks across the full 1-D arc spectrum and attempts to match
    them to reference lines in wavelength space.  This is called ONLY when
    the two-pass local expected-line search produces fewer than
    ``min_lines_per_order`` matches.  It must NOT replace the IDL-style
    local search for well-behaved orders.

    Parameters
    ----------
    flux : ndarray, shape (n_cols,)
        Extracted 1-D arc spectrum for this order.
    col_start : int
        Detector column corresponding to ``flux[0]``.
    affine_coarse_cols : ndarray
        Coarse column lookup already shifted by xcorr and affine correction.
    coarse_wavs : ndarray
        Coarse wavelength values (µm) corresponding to *affine_coarse_cols*.
        Must be monotonically ordered for :func:`numpy.interp`.
    ref_entries : list of (float, str)
        Reference line ``(wavelength_um, species)`` pairs for this order.
    min_prominence : float, default 50.0
        Minimum prominence (in detector counts) for a global peak to be
        considered as an arc-line candidate.
    min_distance : int, default 5
        Minimum separation (in pixels) between detected peaks.
    match_tol_um : float, default 0.0005
        Maximum allowed wavelength residual (µm) between a detected peak's
        predicted wavelength and the nearest reference line.  The default
        0.0005 µm equals 0.5 nm (1 µm = 1000 nm).

    Returns
    -------
    matches : list of (col, ref_wavelength_um)
        Matched and monotonicity-filtered (col, reference-wavelength) pairs.
        Returns an empty list when fewer matches are found or when the
        fallback cannot improve on the local result.

    Notes
    -----
    Monotonicity (wavelength must increase with column) is enforced via
    :func:`_enforce_monotonic_matches`.  Duplicate matches (two detected
    peaks claiming the same reference line) are resolved by keeping the
    one with the smallest wavelength residual.
    """
    if len(ref_entries) == 0 or len(affine_coarse_cols) == 0:
        return []

    n_flux = len(flux)
    if n_flux == 0:
        return []

    # Replace NaNs with zero for peak finding
    flux_clean = np.where(np.isfinite(flux), flux, 0.0)

    # Step 1: Detect peaks across the full 1-D spectrum
    peak_indices, _ = _scipy_find_peaks(
        flux_clean,
        prominence=min_prominence,
        distance=min_distance,
    )
    if len(peak_indices) == 0:
        return []

    # Convert peak local indices to absolute detector columns
    peak_cols = (peak_indices + col_start).astype(float)

    # Step 2: Convert peak columns to wavelengths using the current model
    col_min = float(affine_coarse_cols[0])
    col_max = float(affine_coarse_cols[-1])
    ref_wavs = np.array([e[0] for e in ref_entries], dtype=float)

    # Step 3 & 4: Match each detected peak to nearest reference line within
    # tolerance, keeping only the best match per reference line.
    # best: ref_idx → (col, ref_wav, wav_residual)
    best: dict[int, tuple[float, float, float]] = {}

    for peak_col in peak_cols:
        if peak_col < col_min or peak_col > col_max:
            continue

        pred_wav = float(np.interp(peak_col, affine_coarse_cols, coarse_wavs))
        diffs = np.abs(ref_wavs - pred_wav)
        best_ref_idx = int(np.argmin(diffs))
        wav_residual = float(diffs[best_ref_idx])

        if wav_residual > match_tol_um:
            continue

        ref_wav = float(ref_wavs[best_ref_idx])

        # Keep only the closest peak for each reference line
        if best_ref_idx not in best or wav_residual < best[best_ref_idx][2]:
            best[best_ref_idx] = (peak_col, ref_wav, wav_residual)

    raw_matches = [(col, wav) for col, wav, _ in best.values()]

    # Step 5: Enforce monotonicity (wavelength must increase with column)
    filtered, _ = _enforce_monotonic_matches(raw_matches)
    return filtered
