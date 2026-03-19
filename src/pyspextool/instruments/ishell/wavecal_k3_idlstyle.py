"""
IDL-style global 1DXD wavelength calibration for iSHELL K3 mode.

This module implements the K3 benchmark wavelength-calibration workflow that
mirrors the IDL Spextool reduction sequence more closely than the earlier
2D-seed-based scaffold path.

IDL Spextool sequence (reproduced here)
----------------------------------------
1. Extract a 1-D arc spectrum per order from the median-combined arc image,
   using the order centre-line geometry from flat-field tracing.
2. Cross-correlate the extracted 1-D spectrum with the stored reference
   arc spectrum (plane 1 of the packaged ``K3_wavecalinfo.fits``) to
   estimate a per-order pixel offset.
3. Use the shifted expected line positions to identify and centroid arc lines
   in 1-D for each order.
4. Fit a single **global 1DXD** wavelength solution

       lambda = f(column, order)

   across all accepted K3 line centroids using fixed polynomial degrees::

       lambda_degree = 3   (column)
       order_degree  = 2   (echelle order number)

5. Apply iterative sigma-clipping outlier rejection in the global fit.
6. Report QA statistics analogous to the IDL 1DXD residual plot.

What this module does NOT do (by design)
-----------------------------------------
* Science-side spectral extraction.
* Telluric correction.
* Order merging.
* Full 2-D tilt / curvature measurement.
* Claim parity with IDL unless explicitly demonstrated.

The 2-D traced arc-line results from :mod:`arc_tracing` are still used for
downstream rectification work but are **not** the primary source of
wavelength identifications for the K3 benchmark in this path.

Relationship to the old scaffold path
---------------------------------------
The old scaffold path (wavecal_2d.py → wavecal_2d_surface.py →
wavecal_2d_refine.py) used 2-D seed-based direct line matching as the
primary calibration model, with per-order polynomial fitting as the
central fit and per-order degree reduction for weak orders.

This module replaces that approach for the K3 benchmark with:
- 1-D arc extraction per order (simpler, deterministic, testable),
- cross-correlation-based offset estimation (no hard-coded pixel shifts),
- 1-D line identification and centroiding (primary source of λ-IDs),
- global 1DXD fit across all orders (no per-order degree selection),
- sigma-clipping in the global fit (not in per-order fits).

Public API
----------
- :class:`OrderArcSpectrum` – extracted 1-D arc spectrum for one order.
- :class:`CrossCorrelationResult` – cross-correlation result per order.
- :class:`LineIdentResult` – 1-D line identifications for one order.
- :class:`GlobalFit1DXD` – global 1DXD fit result.
- :class:`K3CalibDiagnostics` – full diagnostics summary.
- :func:`extract_order_arc_spectra` – Step 1: extract 1-D spectra.
- :func:`cross_correlate_with_reference` – Step 2: per-order offset.
- :func:`identify_lines_1d` – Step 3: 1-D identification and centroiding.
- :func:`fit_global_1dxd` – Step 4+5: global fit with sigma-clipping.
- :func:`run_k3_1dxd_wavecal` – top-level driver for the full sequence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate, find_peaks
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as nppoly

if TYPE_CHECKING:
    from .calibrations import FlatInfo, LineList, WaveCalInfo

__all__ = [
    "OrderArcSpectrum",
    "CrossCorrelationResult",
    "LineIdentResult",
    "GlobalFit1DXD",
    "K3CalibDiagnostics",
    "extract_order_arc_spectra",
    "cross_correlate_with_reference",
    "identify_lines_1d",
    "fit_global_1dxd",
    "run_k3_1dxd_wavecal",
    # Fixed benchmark degrees
    "K3_LAMBDA_DEGREE",
    "K3_ORDER_DEGREE",
]

# ---------------------------------------------------------------------------
# Fixed global polynomial degrees for the K3 benchmark (IDL manual §3.2.3)
# ---------------------------------------------------------------------------

K3_LAMBDA_DEGREE: int = 3  #: Polynomial degree in detector column
K3_ORDER_DEGREE: int = 2   #: Polynomial degree in echelle order number


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OrderArcSpectrum:
    """Extracted 1-D arc spectrum for one echelle order.

    Parameters
    ----------
    order_index : int
        0-based index of this order in the :class:`~.calibrations.WaveCalInfo`
        order list.
    order_number : int
        Echelle order number (e.g. 203–229 for K3).
    col_start : int
        First detector column of the valid column range for this order.
    col_end : int
        Last detector column of the valid column range for this order.
    spectrum : ndarray, shape (n_cols,)
        Extracted 1-D flux values along the order centre-line.  Length
        equals ``col_end - col_start + 1``.
    reference_spectrum : ndarray, shape (n_cols,)
        Packaged reference arc spectrum (plane 1 of WaveCalInfo.data) for
        this order, resampled to the same column range.
    reference_wavelength_um : ndarray, shape (n_cols,)
        Packaged reference wavelength grid (plane 0 of WaveCalInfo.data) for
        this order, in microns.
    """

    order_index: int
    order_number: int
    col_start: int
    col_end: int
    spectrum: npt.NDArray
    reference_spectrum: npt.NDArray
    reference_wavelength_um: npt.NDArray

    @property
    def n_cols(self) -> int:
        """Number of columns in this order's spectrum."""
        return len(self.spectrum)

    @property
    def columns(self) -> npt.NDArray:
        """Detector column indices corresponding to ``spectrum``."""
        return np.arange(self.col_start, self.col_end + 1)


@dataclass
class CrossCorrelationResult:
    """Cross-correlation result between extracted and reference arc spectra.

    Parameters
    ----------
    order_index : int
        0-based order index.
    order_number : int
        Echelle order number.
    pixel_shift : float
        Estimated pixel offset of the extracted spectrum relative to the
        reference spectrum.  A positive value means the extracted spectrum
        is shifted to larger column numbers relative to the reference.
    xcorr : ndarray
        Full cross-correlation function array (for diagnostics).
    xcorr_lag_range : tuple[int, int]
        ``(lag_min, lag_max)`` range over which ``xcorr`` was computed.
    """

    order_index: int
    order_number: int
    pixel_shift: float
    xcorr: npt.NDArray
    xcorr_lag_range: tuple[int, int]


@dataclass
class LineIdentResult:
    """1-D arc-line identifications and centroids for one echelle order.

    Parameters
    ----------
    order_index : int
        0-based order index.
    order_number : int
        Echelle order number.
    n_candidate : int
        Number of reference lines tried.
    centroids_col : ndarray, shape (n_accepted,)
        Detector column centroids of accepted lines.
    wavelengths_um : ndarray, shape (n_accepted,)
        Reference vacuum wavelengths (µm) of accepted lines.
    centroid_snr : ndarray, shape (n_accepted,)
        Peak signal-to-noise of each accepted centroid window.
    rejected_wavelengths_um : ndarray, shape (n_rejected,)
        Reference wavelengths of lines that were tried but not accepted.
    """

    order_index: int
    order_number: int
    n_candidate: int
    centroids_col: npt.NDArray
    wavelengths_um: npt.NDArray
    centroid_snr: npt.NDArray
    rejected_wavelengths_um: npt.NDArray

    @property
    def n_accepted(self) -> int:
        """Number of accepted line identifications."""
        return len(self.centroids_col)

    @property
    def n_rejected(self) -> int:
        """Number of rejected candidate lines."""
        return len(self.rejected_wavelengths_um)


@dataclass
class GlobalFit1DXD:
    """Result of the global 1DXD wavelength fit across all K3 orders.

    The fit models::

        lambda = f(column, order)

    as a 2-D polynomial with fixed degrees::

        lambda_degree = 3   (column)
        order_degree  = 2   (echelle order number)

    using normalized coordinates:

    * ``u = (col - col_center) / col_half``  maps observed columns to [-1, +1].
    * ``v = order_ref / order_number``        normalized inverse order.

    Parameters
    ----------
    lambda_degree : int
        Polynomial degree in the column direction (always
        :data:`K3_LAMBDA_DEGREE` = 3).
    order_degree : int
        Polynomial degree in the order direction (always
        :data:`K3_ORDER_DEGREE` = 2).
    coeffs : ndarray, shape ((lambda_degree+1) * (order_degree+1),)
        Flattened 2-D polynomial coefficients.  Reshape to
        ``(lambda_degree+1, order_degree+1)`` to index as ``[i, j]``.
    col_center : float
        Centre used for column normalization.
    col_half : float
        Half-range used for column normalization (always > 0).
    order_ref : float
        Reference order number for the inverse-order normalization
        (equal to the minimum order number in the data).
    order_numbers_all : ndarray, shape (n_total,)
        Echelle order number for every input point (before rejection).
    cols_all : ndarray, shape (n_total,)
        Detector column for every input point.
    wavs_ref_all : ndarray, shape (n_total,)
        Reference wavelength (µm) for every input point.
    accepted_mask : ndarray of bool, shape (n_total,)
        ``True`` for points used in the final fit; ``False`` for rejected.
    residuals_um : ndarray, shape (n_total,)
        Fit residuals in µm for all input points (``NaN`` for rejected
        points).
    rms_um : float
        RMS of fit residuals (µm) over accepted points.
    median_residual_um : float
        Median absolute residual (µm) over accepted points.
    n_total : int
        Total number of input points.
    n_accepted : int
        Number of points used in the final fit.
    n_rejected : int
        Number of points rejected by sigma-clipping.
    sigma_clip_nsigma : float
        Sigma-clipping threshold used.
    sigma_clip_niter : int
        Number of sigma-clipping iterations performed.
    """

    lambda_degree: int
    order_degree: int
    coeffs: npt.NDArray
    col_center: float
    col_half: float
    order_ref: float
    order_numbers_all: npt.NDArray
    cols_all: npt.NDArray
    wavs_ref_all: npt.NDArray
    accepted_mask: npt.NDArray
    residuals_um: npt.NDArray
    rms_um: float
    median_residual_um: float
    n_total: int
    n_accepted: int
    n_rejected: int
    sigma_clip_nsigma: float
    sigma_clip_niter: int

    @property
    def coeffs_2d(self) -> npt.NDArray:
        """Coefficients reshaped to (lambda_degree+1, order_degree+1)."""
        return self.coeffs.reshape(self.lambda_degree + 1, self.order_degree + 1)

    def eval(self, col: float, order_number: float) -> float:
        """Evaluate the fitted wavelength at a single (col, order) point."""
        u = (col - self.col_center) / self.col_half
        v = self.order_ref / order_number
        result = 0.0
        c2d = self.coeffs_2d
        for i in range(self.lambda_degree + 1):
            for j in range(self.order_degree + 1):
                result += c2d[i, j] * (u ** i) * (v ** j)
        return result

    def eval_array(
        self,
        cols: npt.ArrayLike,
        order_number: float,
    ) -> npt.NDArray:
        """Evaluate the fitted wavelength at an array of columns for one order."""
        cols_arr = np.asarray(cols, dtype=float)
        u = (cols_arr - self.col_center) / self.col_half
        v = self.order_ref / order_number
        c2d = self.coeffs_2d
        result = np.zeros_like(cols_arr)
        for i in range(self.lambda_degree + 1):
            for j in range(self.order_degree + 1):
                result += c2d[i, j] * (u ** i) * (v ** j)
        return result

    def per_order_rms_nm(self) -> dict[int, float]:
        """Return per-order RMS in nm using accepted points only."""
        orders = np.unique(self.order_numbers_all[self.accepted_mask])
        result: dict[int, float] = {}
        for ord_n in orders:
            mask_ord = self.accepted_mask & (self.order_numbers_all == ord_n)
            resid = self.residuals_um[mask_ord]
            if len(resid) > 0:
                result[int(ord_n)] = float(np.sqrt(np.mean(resid ** 2)) * 1e3)
            else:
                result[int(ord_n)] = float("nan")
        return result


@dataclass
class K3CalibDiagnostics:
    """Full diagnostics summary for the K3 1DXD calibration run.

    Parameters
    ----------
    order_spectra : list of :class:`OrderArcSpectrum`
        Per-order extracted 1-D arc spectra.
    xcorr_results : list of :class:`CrossCorrelationResult`
        Per-order cross-correlation results.
    line_idents : list of :class:`LineIdentResult`
        Per-order 1-D line identifications.
    global_fit : :class:`GlobalFit1DXD` or None
        Global 1DXD fit result, or ``None`` if the fit could not be
        performed (e.g. too few identified lines).
    pixel_shifts : dict[int, float]
        Per-order pixel shift (order_number → shift).
    per_order_n_candidate : dict[int, int]
        Per-order candidate line count.
    per_order_n_accepted : dict[int, int]
        Per-order accepted 1-D line count.
    per_order_n_rejected : dict[int, int]
        Per-order rejected 1-D line count.
    per_order_rms_nm : dict[int, float]
        Per-order global-fit RMS in nm (NaN if not in fit).
    orders_in_fit : list[int]
        Order numbers that contributed to the global fit.
    """

    order_spectra: list[OrderArcSpectrum]
    xcorr_results: list[CrossCorrelationResult]
    line_idents: list[LineIdentResult]
    global_fit: "GlobalFit1DXD | None"
    pixel_shifts: dict[int, float]
    per_order_n_candidate: dict[int, int]
    per_order_n_accepted: dict[int, int]
    per_order_n_rejected: dict[int, int]
    per_order_rms_nm: dict[int, float]
    orders_in_fit: list[int]

    @property
    def total_accepted(self) -> int:
        """Total accepted 1-D line count across all orders."""
        return sum(self.per_order_n_accepted.values())

    @property
    def total_rejected(self) -> int:
        """Total rejected 1-D line count across all orders."""
        return sum(self.per_order_n_rejected.values())

    @property
    def global_rms_nm(self) -> float:
        """Global fit RMS in nm, or NaN if fit unavailable."""
        if self.global_fit is None:
            return float("nan")
        return self.global_fit.rms_um * 1e3

    @property
    def global_median_residual_nm(self) -> float:
        """Global fit median residual in nm, or NaN if fit unavailable."""
        if self.global_fit is None:
            return float("nan")
        return self.global_fit.median_residual_um * 1e3


# ---------------------------------------------------------------------------
# Step 1: extract 1-D arc spectra per order
# ---------------------------------------------------------------------------


def extract_order_arc_spectra(
    arc_image: npt.NDArray,
    wavecalinfo: "WaveCalInfo",
    *,
    n_rows_avg: int = 5,
) -> list[OrderArcSpectrum]:
    """Extract a representative 1-D arc spectrum for each K3 order.

    Uses the order centre-row geometry stored in plane 0 of the packaged
    ``WaveCalInfo`` together with the actual detector arc image to produce a
    1-D flux spectrum per order.

    For each order the extraction is:

    1. Identify the centre row from the flat-field order geometry.
       Since the packaged ``WaveCalInfo.xranges`` provides the column range
       but not explicit row positions, the centre row is estimated from a
       cross-dispersion median profile of the arc image in the valid column
       range.
    2. Average ``n_rows_avg`` rows centred on the estimated centre row to
       produce the 1-D spectrum, providing modest SNR averaging while
       remaining close to the centre-line.

    Parameters
    ----------
    arc_image : ndarray, shape (n_rows, n_cols)
        Median-combined arc image, as returned by
        :func:`~.arc_tracing.load_and_combine_arcs`.
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Packaged wavelength-calibration metadata for K3 mode.
    n_rows_avg : int, default 5
        Number of rows to average around the estimated centre row.  Must be
        odd and ≥ 1.

    Returns
    -------
    list of :class:`OrderArcSpectrum`
        One entry per order in ``wavecalinfo.orders``, in the same order.
    """
    arc_image = np.asarray(arc_image, dtype=float)
    nrows, ncols = arc_image.shape
    n_half = max(0, (n_rows_avg - 1) // 2)

    result: list[OrderArcSpectrum] = []
    for idx, order_number in enumerate(wavecalinfo.orders):
        x_start = int(wavecalinfo.xranges[idx, 0])
        x_end = int(wavecalinfo.xranges[idx, 1])
        x_end = min(x_end, ncols - 1)
        x_start = max(0, x_start)
        n_pix = x_end - x_start + 1

        # Reference wavelength and spectrum arrays from the packaged data
        ref_wav_full = wavecalinfo.data[idx, 0, :]   # plane 0: wavelengths (µm)
        ref_spec_full = wavecalinfo.data[idx, 1, :]  # plane 1: reference arc spec

        # Slice to the valid column range
        ref_wav = ref_wav_full[x_start: x_end + 1].copy()
        ref_spec = ref_spec_full[x_start: x_end + 1].copy()

        # Estimate the order centre row from a cross-dispersion collapse of
        # the arc image in the valid column range.
        strip = arc_image[:, x_start: x_end + 1]
        col_median = np.nanmedian(strip, axis=1)  # shape (nrows,)

        # Find the peak row (brightest centre-line region)
        # Use a robust argmax of a smoothed profile to avoid cosmic rays.
        from scipy.ndimage import uniform_filter1d as _smooth
        smoothed = _smooth(col_median, size=5)
        centre_row = int(np.argmax(smoothed))

        # Clamp rows to valid range
        row_lo = max(0, centre_row - n_half)
        row_hi = min(nrows - 1, centre_row + n_half)

        # Extract and average the spectrum strip
        spec_strip = arc_image[row_lo: row_hi + 1, x_start: x_end + 1]
        spectrum = np.nanmean(spec_strip, axis=0)

        result.append(
            OrderArcSpectrum(
                order_index=idx,
                order_number=int(order_number),
                col_start=x_start,
                col_end=x_end,
                spectrum=spectrum.astype(float),
                reference_spectrum=ref_spec.astype(float),
                reference_wavelength_um=ref_wav.astype(float),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Step 2: cross-correlate with stored reference spectrum
# ---------------------------------------------------------------------------


def cross_correlate_with_reference(
    order_spectra: list[OrderArcSpectrum],
    *,
    max_lag: int = 100,
) -> list[CrossCorrelationResult]:
    """Estimate per-order pixel shifts by cross-correlating with the reference.

    For each order, cross-correlates the extracted 1-D arc spectrum with the
    packaged reference arc spectrum (plane 1 of WaveCalInfo) to find the
    shift that best aligns them.

    Both spectra are first normalised to zero mean and unit variance (within
    the region where the reference is non-NaN and positive) before computing
    the cross-correlation, so that the result is not dominated by the
    bright-line envelope.

    Parameters
    ----------
    order_spectra : list of :class:`OrderArcSpectrum`
        Per-order spectra, as returned by
        :func:`extract_order_arc_spectra`.
    max_lag : int, default 100
        Maximum pixel lag to consider.  Lags outside ``[-max_lag, +max_lag]``
        are ignored to avoid false-peak aliasing.

    Returns
    -------
    list of :class:`CrossCorrelationResult`
        One entry per input order spectrum, in the same order.

    Notes
    -----
    A positive ``pixel_shift`` means the extracted spectrum appears at larger
    column numbers than the reference, i.e. the reference must be shifted
    *right* (to larger columns) to match the extracted spectrum.  Equivalently,
    the expected arc-line positions should be offset by ``+pixel_shift`` before
    centroiding.
    """
    results: list[CrossCorrelationResult] = []
    for os in order_spectra:
        spec = os.spectrum.copy()
        ref = os.reference_spectrum.copy()

        # Valid mask: both must be finite and reference must be non-negative
        valid = np.isfinite(spec) & np.isfinite(ref) & (ref >= 0)
        if valid.sum() < 10:
            # Not enough data: fall back to zero shift
            results.append(
                CrossCorrelationResult(
                    order_index=os.order_index,
                    order_number=os.order_number,
                    pixel_shift=0.0,
                    xcorr=np.array([1.0]),
                    xcorr_lag_range=(0, 0),
                )
            )
            continue

        # Zero-mean, unit-variance normalisation on the valid region
        spec_v = spec.copy()
        ref_v = ref.copy()
        spec_v[~valid] = 0.0
        ref_v[~valid] = 0.0

        sp_mean = np.mean(spec_v[valid])
        sp_std = np.std(spec_v[valid])
        sp_std = sp_std if sp_std > 0 else 1.0
        spec_v = (spec_v - sp_mean) / sp_std

        rf_mean = np.mean(ref_v[valid])
        rf_std = np.std(ref_v[valid])
        rf_std = rf_std if rf_std > 0 else 1.0
        ref_v = (ref_v - rf_mean) / rf_std

        # Full cross-correlation
        xcorr_full = correlate(spec_v, ref_v, mode="full")
        n = len(spec_v)
        lags = np.arange(-(n - 1), n)

        # Restrict to [-max_lag, +max_lag]
        keep = (lags >= -max_lag) & (lags <= max_lag)
        xcorr = xcorr_full[keep]
        lags_kept = lags[keep]

        peak_idx = int(np.argmax(xcorr))
        pixel_shift = float(lags_kept[peak_idx])

        results.append(
            CrossCorrelationResult(
                order_index=os.order_index,
                order_number=os.order_number,
                pixel_shift=pixel_shift,
                xcorr=xcorr,
                xcorr_lag_range=(int(lags_kept[0]), int(lags_kept[-1])),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Step 3: 1-D line identification and centroiding
# ---------------------------------------------------------------------------


_MAX_GAUSSIAN_FIT_ITERATIONS: int = 800  #: Max function evaluations for Gaussian fits


def _gaussian(x: npt.NDArray, amp: float, mu: float, sigma: float,
               background: float) -> npt.NDArray:
    """Simple Gaussian + constant background."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + background


def _centroid_gaussian(
    spec: npt.NDArray,
    col_start: int,
    centre_col: float,
    half_window: int,
) -> tuple[float, float]:
    """Fit a Gaussian to a window of spec and return (centroid_col, peak_snr).

    Returns ``(nan, nan)`` if the fit fails or the centroid is outside the
    window.
    """
    n = len(spec)
    i_lo = max(0, int(round(centre_col - col_start)) - half_window)
    i_hi = min(n - 1, int(round(centre_col - col_start)) + half_window)
    if i_hi - i_lo < 3:
        return float("nan"), float("nan")

    y = spec[i_lo: i_hi + 1].copy()
    x = np.arange(i_lo, i_hi + 1, dtype=float) + col_start

    if not np.all(np.isfinite(y)):
        return float("nan"), float("nan")

    bg_est = np.min(y)
    amp_est = np.max(y) - bg_est
    if amp_est <= 0:
        return float("nan"), float("nan")

    try:
        p0 = [amp_est, centre_col, 1.5, bg_est]
        bounds = (
            [0, x[0], 0.3, -np.inf],
            [np.inf, x[-1], 10.0, np.inf],
        )
        popt, _ = curve_fit(_gaussian, x, y, p0=p0, bounds=bounds,
                            maxfev=_MAX_GAUSSIAN_FIT_ITERATIONS)
        fitted_col = popt[1]
        fitted_amp = popt[0]
        fitted_bg = popt[3]
        fitted_sigma = abs(popt[2])
        # Check centroid is within window
        if fitted_col < x[0] or fitted_col > x[-1]:
            return float("nan"), float("nan")
        noise = max(np.std(y - _gaussian(x, *popt)), 1e-10)
        snr = fitted_amp / noise
        return fitted_col, snr
    except (RuntimeError, ValueError):
        pass

    # Fall back to flux-weighted centroid
    y_sub = y - np.min(y)
    total = np.sum(y_sub)
    if total <= 0:
        return float("nan"), float("nan")
    centroid = float(np.sum(x * y_sub) / total)
    if centroid < x[0] or centroid > x[-1]:
        return float("nan"), float("nan")
    peak_val = np.max(y_sub)
    noise = max(np.std(y - y_sub - np.min(y)), 1e-10)
    snr = peak_val / noise
    return centroid, snr


def identify_lines_1d(
    order_spectra: list[OrderArcSpectrum],
    xcorr_results: list[CrossCorrelationResult],
    linelist: "LineList",
    *,
    window_half_pixels: int = 8,
    min_snr: float = 3.0,
    min_col_separation: float = 5.0,
) -> list[LineIdentResult]:
    """Identify and centroid arc lines in 1-D for each K3 order.

    For each order:

    1. Retrieve the reference line list entries for this order.
    2. Convert reference wavelengths to expected pixel positions using the
       packaged reference wavelength grid (plane 0 of WaveCalInfo), applying
       the cross-correlation pixel shift from Step 2.
    3. Fit a Gaussian (or fall back to flux-weighted centroid) to a window
       centred on each expected position.
    4. Accept the centroid if its peak SNR exceeds ``min_snr`` and the
       column is within the fitting window.
    5. Apply a minimum column separation filter to remove near-duplicate
       centroids.

    Parameters
    ----------
    order_spectra : list of :class:`OrderArcSpectrum`
        Per-order 1-D arc spectra.
    xcorr_results : list of :class:`CrossCorrelationResult`
        Per-order cross-correlation results (same ordering as
        ``order_spectra``).
    linelist : :class:`~.calibrations.LineList`
        Packaged ThAr arc-line list for K3 mode.
    window_half_pixels : int, default 8
        Half-width (in pixels) of the centroiding window around the
        expected line position.
    min_snr : float, default 3.0
        Minimum peak-SNR for an accepted centroid.
    min_col_separation : float, default 5.0
        Minimum column separation (pixels) between accepted centroids.
        When two accepted centroids are closer than this, only the one
        with higher SNR is kept.

    Returns
    -------
    list of :class:`LineIdentResult`
        One entry per order, in the same order as ``order_spectra``.
    """
    # Index linelist by order number
    from collections import defaultdict
    ll_by_order: dict[int, list] = defaultdict(list)
    for entry in linelist.entries:
        ll_by_order[entry.order].append(entry)

    results: list[LineIdentResult] = []
    for os, xcr in zip(order_spectra, xcorr_results):
        ord_n = os.order_number
        entries = ll_by_order.get(ord_n, [])
        shift = xcr.pixel_shift

        spec = os.spectrum
        ref_wav = os.reference_wavelength_um
        cols = os.columns

        # Build a wavelength→column interpolator from the reference grid
        valid_ref = np.isfinite(ref_wav) & (ref_wav > 0)
        if valid_ref.sum() < 2 or len(entries) == 0:
            results.append(
                LineIdentResult(
                    order_index=os.order_index,
                    order_number=ord_n,
                    n_candidate=len(entries),
                    centroids_col=np.array([]),
                    wavelengths_um=np.array([]),
                    centroid_snr=np.array([]),
                    rejected_wavelengths_um=np.array(
                        [e.wavelength_um for e in entries]
                    ),
                )
            )
            continue

        # Interpolate reference wavelengths to column positions
        # ref_wav is indexed from col_start; cols = np.arange(col_start, col_end+1)
        ref_wav_valid = ref_wav[valid_ref]
        cols_valid = cols[valid_ref]

        accepted_cols: list[float] = []
        accepted_wavs: list[float] = []
        accepted_snrs: list[float] = []
        rejected_wavs: list[float] = []

        for entry in entries:
            # Predict the column position of this line in the reference
            if (entry.wavelength_um < ref_wav_valid[0]
                    or entry.wavelength_um > ref_wav_valid[-1]):
                rejected_wavs.append(entry.wavelength_um)
                continue
            predicted_col = float(
                np.interp(entry.wavelength_um, ref_wav_valid, cols_valid.astype(float))
            )
            # Apply the cross-correlation shift
            shifted_col = predicted_col + shift

            # Clamp to valid column range
            col_lo = float(cols[0])
            col_hi = float(cols[-1])
            if shifted_col < col_lo + window_half_pixels or shifted_col > col_hi - window_half_pixels:
                rejected_wavs.append(entry.wavelength_um)
                continue

            centroid, snr = _centroid_gaussian(
                spec, os.col_start, shifted_col, window_half_pixels
            )
            if not np.isfinite(centroid) or not np.isfinite(snr) or snr < min_snr:
                rejected_wavs.append(entry.wavelength_um)
                continue

            accepted_cols.append(centroid)
            accepted_wavs.append(entry.wavelength_um)
            accepted_snrs.append(snr)

        # Apply minimum column separation filter
        if len(accepted_cols) > 1:
            order_idx = np.argsort(accepted_cols)
            acc_cols_s = np.array(accepted_cols)[order_idx]
            acc_wavs_s = np.array(accepted_wavs)[order_idx]
            acc_snrs_s = np.array(accepted_snrs)[order_idx]
            keep = np.ones(len(acc_cols_s), dtype=bool)
            for i in range(1, len(acc_cols_s)):
                if not keep[i - 1]:
                    continue
                if (acc_cols_s[i] - acc_cols_s[i - 1]) < min_col_separation:
                    # Keep the one with higher SNR
                    if acc_snrs_s[i] > acc_snrs_s[i - 1]:
                        keep[i - 1] = False
                    else:
                        keep[i] = False
            accepted_cols = acc_cols_s[keep].tolist()
            accepted_wavs = acc_wavs_s[keep].tolist()
            accepted_snrs = acc_snrs_s[keep].tolist()

        results.append(
            LineIdentResult(
                order_index=os.order_index,
                order_number=ord_n,
                n_candidate=len(entries),
                centroids_col=np.array(accepted_cols),
                wavelengths_um=np.array(accepted_wavs),
                centroid_snr=np.array(accepted_snrs),
                rejected_wavelengths_um=np.array(rejected_wavs),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Step 4+5: Global 1DXD fit with sigma-clipping
# ---------------------------------------------------------------------------


def fit_global_1dxd(
    line_idents: list[LineIdentResult],
    *,
    lambda_degree: int = K3_LAMBDA_DEGREE,
    order_degree: int = K3_ORDER_DEGREE,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_niter: int = 5,
    min_lines_total: int = 6,
) -> "GlobalFit1DXD | None":
    """Fit a global 1DXD wavelength solution across all K3 orders.

    Fits a single 2-D polynomial::

        lambda = f(column, order)

    using fixed polynomial degrees (``lambda_degree=3``, ``order_degree=2``)
    and iterative sigma-clipping outlier rejection.

    The coordinate normalization is:

    * ``u = (col - col_center) / col_half`` → maps observed columns to [-1, +1]
    * ``v = order_ref / order_number``       → normalized inverse order

    Parameters
    ----------
    line_idents : list of :class:`LineIdentResult`
        Per-order 1-D line identifications from :func:`identify_lines_1d`.
    lambda_degree : int, default :data:`K3_LAMBDA_DEGREE` (3)
        Polynomial degree in the column direction.  Fixed for the K3 benchmark.
    order_degree : int, default :data:`K3_ORDER_DEGREE` (2)
        Polynomial degree in the order direction.  Fixed for the K3 benchmark.
    sigma_clip_nsigma : float, default 3.0
        Sigma-clipping threshold in units of the current fit RMS.
    sigma_clip_niter : int, default 5
        Maximum number of sigma-clipping iterations.
    min_lines_total : int, default 6
        Minimum total number of accepted 1-D line identifications required
        to attempt the global fit.  Returns ``None`` if below this threshold.

    Returns
    -------
    :class:`GlobalFit1DXD` or None
        Fit result, or ``None`` if there are not enough points to fit.

    Notes
    -----
    The fixed degrees (3, 2) reflect the K3 benchmark configuration from
    the IDL Spextool manual.  They are **not** selected automatically.
    """
    # Collect all accepted points across orders
    all_orders: list[float] = []
    all_cols: list[float] = []
    all_wavs: list[float] = []

    for lid in line_idents:
        if lid.n_accepted == 0:
            continue
        for col, wav in zip(lid.centroids_col, lid.wavelengths_um):
            all_orders.append(float(lid.order_number))
            all_cols.append(float(col))
            all_wavs.append(float(wav))

    n_total = len(all_wavs)
    if n_total < min_lines_total:
        return None

    orders_arr = np.array(all_orders)
    cols_arr = np.array(all_cols)
    wavs_arr = np.array(all_wavs)

    # Normalization
    col_center = (cols_arr.max() + cols_arr.min()) / 2.0
    col_half = (cols_arr.max() - cols_arr.min()) / 2.0
    col_half = max(col_half, 1.0)
    order_ref = float(orders_arr.min())

    u_arr = (cols_arr - col_center) / col_half
    v_arr = order_ref / orders_arr

    # Number of free parameters
    n_params = (lambda_degree + 1) * (order_degree + 1)

    # Iterative sigma-clipping fit
    accepted = np.ones(n_total, dtype=bool)
    residuals = np.full(n_total, np.nan)
    coeffs = None
    iter_used = 0

    for iteration in range(sigma_clip_niter + 1):
        n_acc = accepted.sum()
        if n_acc < n_params:
            warnings.warn(
                f"fit_global_1dxd: too few accepted points ({n_acc}) for "
                f"{n_params} parameters on iteration {iteration}.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # Build design matrix
        u_acc = u_arr[accepted]
        v_acc = v_arr[accepted]
        wav_acc = wavs_arr[accepted]

        A = _build_design_matrix(u_acc, v_acc, lambda_degree, order_degree)
        coeffs_flat, _, _, _ = np.linalg.lstsq(A, wav_acc, rcond=None)

        # Residuals for all points (accepted + previously rejected)
        A_all = _build_design_matrix(u_arr, v_arr, lambda_degree, order_degree)
        wavs_pred_all = A_all @ coeffs_flat
        residuals_all = wavs_arr - wavs_pred_all

        # Sigma clip using accepted-point RMS
        rms = float(np.sqrt(np.mean(residuals_all[accepted] ** 2)))
        if rms == 0.0:
            break

        new_accepted = accepted & (np.abs(residuals_all) <= sigma_clip_nsigma * rms)
        iter_used = iteration

        if np.array_equal(new_accepted, accepted):
            # Converged
            break
        accepted = new_accepted

    if coeffs_flat is None:
        # Fallback: single-pass fit without any rejection
        A = _build_design_matrix(u_arr, v_arr, lambda_degree, order_degree)
        coeffs_flat, _, _, _ = np.linalg.lstsq(A, wavs_arr, rcond=None)
        A_all = A
        wavs_pred_all = A_all @ coeffs_flat
        residuals_all = wavs_arr - wavs_pred_all
        accepted = np.ones(n_total, dtype=bool)
        rms = float(np.sqrt(np.mean(residuals_all ** 2)))

    # Final residuals (NaN for rejected points)
    residuals_final = np.full(n_total, np.nan)
    residuals_final[accepted] = residuals_all[accepted]

    rms_final = float(np.sqrt(np.mean(residuals_all[accepted] ** 2)))
    median_resid = float(np.median(np.abs(residuals_all[accepted])))

    return GlobalFit1DXD(
        lambda_degree=lambda_degree,
        order_degree=order_degree,
        coeffs=coeffs_flat,
        col_center=col_center,
        col_half=col_half,
        order_ref=order_ref,
        order_numbers_all=orders_arr,
        cols_all=cols_arr,
        wavs_ref_all=wavs_arr,
        accepted_mask=accepted,
        residuals_um=residuals_final,
        rms_um=rms_final,
        median_residual_um=median_resid,
        n_total=n_total,
        n_accepted=int(accepted.sum()),
        n_rejected=int((~accepted).sum()),
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_niter=iter_used,
    )


def _build_design_matrix(
    u: npt.NDArray,
    v: npt.NDArray,
    lambda_degree: int,
    order_degree: int,
) -> npt.NDArray:
    """Build the 2-D polynomial design matrix for the global 1DXD fit.

    Returns an array of shape ``(n_points, (lambda_degree+1)*(order_degree+1))``
    where column ``i*(order_degree+1) + j`` contains ``u**i * v**j``.
    """
    n = len(u)
    n_params = (lambda_degree + 1) * (order_degree + 1)
    A = np.zeros((n, n_params), dtype=float)
    for i in range(lambda_degree + 1):
        for j in range(order_degree + 1):
            col_idx = i * (order_degree + 1) + j
            A[:, col_idx] = (u ** i) * (v ** j)
    return A


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_k3_1dxd_wavecal(
    arc_image: npt.NDArray,
    wavecalinfo: "WaveCalInfo",
    linelist: "LineList",
    *,
    n_rows_avg: int = 5,
    xcorr_max_lag: int = 100,
    window_half_pixels: int = 8,
    min_snr: float = 3.0,
    min_col_separation: float = 5.0,
    lambda_degree: int = K3_LAMBDA_DEGREE,
    order_degree: int = K3_ORDER_DEGREE,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_niter: int = 5,
    min_lines_total: int = 6,
) -> K3CalibDiagnostics:
    """Run the full IDL-style K3 1DXD wavelength-calibration sequence.

    Implements the following IDL Spextool-like steps:

    1. Extract a 1-D arc spectrum per order from the arc image.
    2. Cross-correlate each extracted spectrum with the stored reference.
    3. Identify and centroid arc lines in 1-D using shifted expected positions.
    4. Fit a global 1DXD wavelength solution (lambda_degree=3, order_degree=2).
    5. Apply iterative sigma-clipping in the global fit.
    6. Collect diagnostics.

    Parameters
    ----------
    arc_image : ndarray, shape (n_rows, n_cols)
        Median-combined ThAr arc image.
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Packaged calibration metadata for K3 mode.
    linelist : :class:`~.calibrations.LineList`
        Packaged ThAr line list for K3 mode.
    n_rows_avg : int, default 5
        Number of rows to average for the 1-D spectrum extraction.
    xcorr_max_lag : int, default 100
        Maximum pixel lag for the cross-correlation.
    window_half_pixels : int, default 8
        Half-width of the centroiding window in pixels.
    min_snr : float, default 3.0
        Minimum SNR for a line centroid to be accepted.
    min_col_separation : float, default 5.0
        Minimum column separation between accepted centroids (pixels).
    lambda_degree : int, default 3
        Polynomial degree in column direction for the global 1DXD fit.
    order_degree : int, default 2
        Polynomial degree in order direction for the global 1DXD fit.
    sigma_clip_nsigma : float, default 3.0
        Sigma-clipping threshold.
    sigma_clip_niter : int, default 5
        Maximum number of sigma-clipping iterations.
    min_lines_total : int, default 6
        Minimum accepted 1-D line count to attempt the global fit.

    Returns
    -------
    :class:`K3CalibDiagnostics`
        Full diagnostics object containing per-order and global results.
    """
    # Step 1: extract 1-D arc spectra
    order_spectra = extract_order_arc_spectra(
        arc_image, wavecalinfo, n_rows_avg=n_rows_avg
    )

    # Step 2: cross-correlation
    xcorr_results = cross_correlate_with_reference(
        order_spectra, max_lag=xcorr_max_lag
    )

    # Step 3: 1-D line identification
    line_idents = identify_lines_1d(
        order_spectra,
        xcorr_results,
        linelist,
        window_half_pixels=window_half_pixels,
        min_snr=min_snr,
        min_col_separation=min_col_separation,
    )

    # Step 4+5: global 1DXD fit with sigma-clipping
    global_fit = fit_global_1dxd(
        line_idents,
        lambda_degree=lambda_degree,
        order_degree=order_degree,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_niter=sigma_clip_niter,
        min_lines_total=min_lines_total,
    )

    # Build diagnostics
    pixel_shifts: dict[int, float] = {
        xcr.order_number: xcr.pixel_shift for xcr in xcorr_results
    }
    per_order_n_candidate: dict[int, int] = {
        lid.order_number: lid.n_candidate for lid in line_idents
    }
    per_order_n_accepted: dict[int, int] = {
        lid.order_number: lid.n_accepted for lid in line_idents
    }
    per_order_n_rejected: dict[int, int] = {
        lid.order_number: lid.n_rejected for lid in line_idents
    }

    # Per-order RMS from global fit
    if global_fit is not None:
        per_order_rms_nm = global_fit.per_order_rms_nm()
        orders_in_fit = [
            int(o) for o in np.unique(
                global_fit.order_numbers_all[global_fit.accepted_mask]
            )
        ]
    else:
        per_order_rms_nm = {
            lid.order_number: float("nan") for lid in line_idents
        }
        orders_in_fit = []

    return K3CalibDiagnostics(
        order_spectra=order_spectra,
        xcorr_results=xcorr_results,
        line_idents=line_idents,
        global_fit=global_fit,
        pixel_shifts=pixel_shifts,
        per_order_n_candidate=per_order_n_candidate,
        per_order_n_accepted=per_order_n_accepted,
        per_order_n_rejected=per_order_n_rejected,
        per_order_rms_nm=per_order_rms_nm,
        orders_in_fit=orders_in_fit,
    )
