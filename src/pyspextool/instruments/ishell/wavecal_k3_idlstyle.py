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
from scipy.signal import find_peaks as _scipy_find_peaks

if TYPE_CHECKING:
    from .calibrations import LineList, WaveCalInfo
    from .tracing import FlatOrderTrace

__all__ = [
    "OrderArcSpectrum",
    "OrderArcSpectraSet",
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
        Echelle order numbers included in the fit.
    fit_rms_um : float
        RMS residual of the global fit (µm).
    n_lines : int
        Number of arc-line matches used in the fit.
    n_orders_fit : int
        Number of orders that contributed at least one match.

    Notes
    -----
    Use :meth:`eval` (scalar) or :meth:`eval_array` (vectorised) to predict
    wavelength at arbitrary ``(col, order)`` combinations.
    """

    mode: str
    wdeg: int
    odeg: int
    order_ref: float
    coeffs: npt.NDArray  # shape (wdeg+1, odeg+1)
    fitted_order_numbers: list[int]
    fit_rms_um: float
    n_lines: int
    n_orders_fit: int

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
) -> IdlStyle1DXDModel:
    """Fit a global IDL-style 1DXD wavelength model across all echelle orders.

    For each order in *spectra_set*:

    1. Detect arc-line peaks in the 1-D arc spectrum using
       :func:`scipy.signal.find_peaks`.
    2. For each peak, predict a wavelength from the packaged
       ``WaveCalInfo`` plane-0 coarse grid.
    3. Match the predicted wavelength to the nearest entry in *line_list*,
       accepting only matches within *match_tol_um*.

    Then fit a single global 2-D polynomial across all accepted matches:

    .. math::

        \\lambda(\\text{col}, \\text{order}) =
            \\sum_{i=0}^{w} \\sum_{j=0}^{p}
            C_{i,j}\\, \\text{col}^{i}\\, v^{j}

    where ``v = order_ref / order``, ``order_ref`` is the minimum echelle
    order number, ``w = wdeg``, and ``p = odeg``.

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

    Returns
    -------
    :class:`IdlStyle1DXDModel`
        Fitted global model.

    Raises
    ------
    ValueError
        If *wdeg* or *odeg* is negative, or if fewer than *min_lines_total*
        arc lines are matched.

    Warns
    -----
    RuntimeWarning
        If an order has no valid coarse reference grid and is skipped.
    RuntimeWarning
        If the total number of matched lines is below *min_lines_total*
        but the fit is still attempted with however many points are
        available.
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
            continue

        # Reference line wavelengths for this order
        ref_entries = _get_ref_entries(line_list, order_num)

        # Find peaks in the 1D arc spectrum
        flux = spec.flux
        valid_mask = np.isfinite(flux)
        if not valid_mask.any():
            continue

        # Replace NaNs with zero for peak finding (NaN positions won't be
        # selected because prominence requires finite neighbours)
        flux_for_peaks = np.where(valid_mask, flux, 0.0)

        peak_idxs, _ = _scipy_find_peaks(
            flux_for_peaks,
            prominence=min_prominence,
            distance=min_distance,
        )

        if len(peak_idxs) == 0:
            logger.debug("Order %d: no peaks found in 1D arc spectrum", order_num)
            continue

        # Convert peak indices to detector columns
        peak_cols = spec.col_start + peak_idxs.astype(float)

        # Match peaks to reference lines
        matches = _match_1d_peaks(
            peak_cols,
            coarse_cols,
            coarse_wavs,
            ref_entries,
            match_tol_um,
        )

        if not matches:
            logger.debug("Order %d: no arc lines matched", order_num)
            continue

        for col_m, wav_m in matches:
            all_cols.append(col_m)
            all_orders.append(float(order_num))
            all_wavs.append(wav_m)

        orders_with_matches.add(order_num)
        logger.debug(
            "Order %d: %d peaks detected, %d lines matched",
            order_num,
            len(peak_idxs),
            len(matches),
        )

    n_lines = len(all_cols)
    n_orders_fit = len(orders_with_matches)

    logger.info(
        "fit_1dxd_wavelength_model: %d lines from %d orders",
        n_lines,
        n_orders_fit,
    )

    if n_lines < min_lines_total:
        raise ValueError(
            f"Only {n_lines} arc lines matched (minimum is {min_lines_total}).  "
            "Try reducing match_tol_um or min_prominence."
        )

    cols_arr = np.array(all_cols, dtype=float)
    orders_arr = np.array(all_orders, dtype=float)
    wavs_arr = np.array(all_wavs, dtype=float)

    # ------------------------------------------------------------------
    # Build design matrix and fit
    # ------------------------------------------------------------------
    order_ref = float(np.min(orders_arr))
    vs = order_ref / orders_arr  # normalised inverse-order coordinate

    # Design matrix: columns are col^i * v^j for i in 0..wdeg, j in 0..odeg
    n_terms = (wdeg + 1) * (odeg + 1)
    X = np.empty((n_lines, n_terms), dtype=float)
    term_idx = 0
    for i in range(wdeg + 1):
        for j in range(odeg + 1):
            X[:, term_idx] = (cols_arr ** i) * (vs ** j)
            term_idx += 1

    # Least-squares fit
    c_flat, _, _, _ = np.linalg.lstsq(X, wavs_arr, rcond=None)
    coeffs = c_flat.reshape(wdeg + 1, odeg + 1)

    # Compute fit RMS
    wavs_pred = X @ c_flat
    residuals = wavs_arr - wavs_pred
    fit_rms_um = float(np.sqrt(np.mean(residuals ** 2)))

    logger.info(
        "fit_1dxd_wavelength_model: wdeg=%d, odeg=%d, "
        "n_lines=%d, fit_rms=%.4f nm",
        wdeg,
        odeg,
        n_lines,
        fit_rms_um * 1e3,
    )

    return IdlStyle1DXDModel(
        mode=spectra_set.mode,
        wdeg=wdeg,
        odeg=odeg,
        order_ref=order_ref,
        coeffs=coeffs,
        fitted_order_numbers=sorted(orders_with_matches),
        fit_rms_um=fit_rms_um,
        n_lines=n_lines,
        n_orders_fit=n_orders_fit,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
