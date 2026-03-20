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
        Number of arc-line peaks detected in the 1-D spectrum before
        matching.
    n_matched : int
        Number of peaks that matched a reference line within the tolerance
        window (before global sigma clipping).
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
    """

    order_number: int
    xcorr_shift_px: float
    n_candidate: int
    n_matched: int
    n_accepted: int
    n_rejected: int
    rms_resid_um: float
    participated: bool


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
    wdeg: int = 3,
    odeg: int = 2,
    min_prominence: float = 50.0,
    min_distance: int = 5,
    match_tol_um: float = 0.002,
    min_lines_total: int = 10,
    sigma_thresh: float = 3.0,
    max_sigma_iter: int = 5,
    xcorr_max_shift_px: int = 50,
) -> IdlStyle1DXDModel:
    """Fit a global IDL-style 1DXD wavelength model across all echelle orders.

    For each order in *spectra_set*:

    1. **Cross-correlate** the extracted 1-D arc spectrum against a synthetic
       reference comb (Gaussians placed at each reference-line column position
       predicted by the coarse wavelength grid) to determine a per-order
       column shift.
    2. **Detect peaks** in the shift-corrected arc spectrum using
       :func:`scipy.signal.find_peaks`.
    3. **Match peaks** to the nearest reference line within *match_tol_um*,
       using the shifted coarse grid for wavelength prediction.

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
    wdeg : int, default 3
        Polynomial degree in detector column.
    odeg : int, default 2
        Polynomial degree in ``v = order_ref / order``.
    min_prominence : float, default 50.0
        Minimum peak prominence (detector counts) for a peak to be
        considered as an arc line.
    min_distance : int, default 5
        Minimum peak separation in pixels.
    match_tol_um : float, default 0.002
        Maximum allowed residual (µm) between predicted and reference
        wavelength for a match to be accepted.
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
                n_candidate=0, n_matched=0, n_accepted=0, n_rejected=0,
                rms_resid_um=float("nan"), participated=False,
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
                n_candidate=0, n_matched=0, n_accepted=0, n_rejected=0,
                rms_resid_um=float("nan"), participated=False,
            ))
            continue

        # ------------------------------------------------------------------
        # Step 1: Cross-correlate extracted spectrum against reference comb
        # to find the per-order column shift.
        # ------------------------------------------------------------------
        xcorr_shift = _xcorr_order_shift(
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

        # Replace NaNs with zero for peak finding
        flux_for_peaks = np.where(valid_mask, flux, 0.0)

        peak_idxs, _ = _scipy_find_peaks(
            flux_for_peaks,
            prominence=min_prominence,
            distance=min_distance,
        )

        n_candidate = len(peak_idxs)
        if n_candidate == 0:
            logger.debug("Order %d: no peaks found in 1D arc spectrum", order_num)
            per_order_stats.append(OrderMatchStats(
                order_number=order_num, xcorr_shift_px=xcorr_shift,
                n_candidate=0, n_matched=0, n_accepted=0, n_rejected=0,
                rms_resid_um=float("nan"), participated=False,
            ))
            continue

        # Convert peak indices to detector columns
        peak_cols = spec.col_start + peak_idxs.astype(float)

        # Match peaks to reference lines using the shift-corrected coarse grid
        matches = _match_1d_peaks(
            peak_cols,
            shifted_coarse_cols,
            coarse_wavs,
            ref_entries,
            match_tol_um,
        )

        n_matched = len(matches)
        if not matches:
            logger.debug("Order %d: no arc lines matched", order_num)
            per_order_stats.append(OrderMatchStats(
                order_number=order_num, xcorr_shift_px=xcorr_shift,
                n_candidate=n_candidate, n_matched=0, n_accepted=0, n_rejected=0,
                rms_resid_um=float("nan"), participated=False,
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
            n_accepted=n_matched, n_rejected=0,
            rms_resid_um=float("nan"), participated=True,
        ))

        logger.debug(
            "Order %d: xcorr=%.1fpx, %d peaks, %d matched",
            order_num, xcorr_shift, n_candidate, n_matched,
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
) -> float:
    """Compute the per-order column shift via cross-correlation.

    Builds a synthetic reference comb from the expected arc-line column
    positions (Gaussians of width *fwhm_px* at each reference-line column
    predicted by the coarse wavelength grid) and cross-correlates it
    against the extracted 1-D arc spectrum.

    The shift is found from the peak of the cross-correlation within
    ``±max_shift_px`` pixels.  Sub-pixel accuracy is obtained by
    fitting a parabola through the three points around the peak.

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
    float
        Cross-correlation shift in pixels.  A positive value means the
        extracted spectrum is shifted *right* relative to the reference.
        Returns ``0.0`` if no reference lines are available, the coarse
        grid is empty, or correlation fails.
    """
    if len(ref_entries) == 0 or len(coarse_cols) == 0:
        return 0.0

    n = len(flux)
    if n == 0:
        return 0.0

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
        return 0.0

    # Full cross-correlation (mode="full") gives a (2n-1,)-length result.
    # The zero-lag is at index n-1.
    xcorr = _scipy_correlate(flux_clean, comb, mode="full")
    zero_lag = n - 1
    lo = max(0, zero_lag - max_shift_px)
    hi = min(len(xcorr) - 1, zero_lag + max_shift_px)
    xcorr_window = xcorr[lo: hi + 1]

    peak_idx_local = int(np.argmax(xcorr_window))
    peak_idx_global = lo + peak_idx_local

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

    # Shift relative to zero lag
    shift = float(peak_idx_global - zero_lag) + sub_shift
    # Clip to max_shift
    shift = float(np.clip(shift, -max_shift_px, max_shift_px))
    return shift


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
) -> list[tuple[float, float]]:
    """Match detected 1-D peak positions to reference wavelengths.

    For each peak column, predicts a wavelength by linear interpolation on
    the coarse grid, then finds the nearest reference line.  Accepts the
    match only if the residual is within *match_tol_um*.

    Deduplicates: if two peaks match the same reference line, only the
    one with the smallest residual is retained.

    Parameters
    ----------
    peak_cols : ndarray
        Detector columns of detected peaks.
    coarse_cols, coarse_wavs : ndarray
        Coarse reference grid from ``WaveCalInfo``.
    ref_entries : list of (float, str)
        Reference line ``(wavelength_um, species)`` pairs for this order.
    match_tol_um : float
        Acceptance tolerance.

    Returns
    -------
    list of (col, ref_wavelength_um)
        Accepted matches.
    """
    if len(ref_entries) == 0 or len(coarse_cols) == 0:
        return []

    ref_wavs = np.array([e[0] for e in ref_entries], dtype=float)

    col_min = float(coarse_cols[0])
    col_max = float(coarse_cols[-1])

    best: dict[int, tuple[float, float, float]] = {}  # ref_idx → (col, ref_wav, residual)

    for col in peak_cols:
        col = float(col)
        if col < col_min or col > col_max:
            continue

        pred_wav = float(np.interp(col, coarse_cols, coarse_wavs))
        diffs = np.abs(ref_wavs - pred_wav)
        best_ref_idx = int(np.argmin(diffs))
        residual = float(diffs[best_ref_idx])

        if residual > match_tol_um:
            continue

        # Deduplication: keep best residual per reference line
        if best_ref_idx not in best or residual < best[best_ref_idx][2]:
            best[best_ref_idx] = (col, float(ref_wavs[best_ref_idx]), residual)

    return [(col, wav) for col, wav, _ in best.values()]
