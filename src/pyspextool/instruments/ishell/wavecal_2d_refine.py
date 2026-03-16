"""
Coefficient-surface refinement scaffold for iSHELL 2DXD reduction.

This module implements the **fifth stage** of the iSHELL 2DXD reduction
scaffold: fitting per-order wavelength polynomials whose coefficients
themselves vary smoothly with echelle order number.

What this module does
---------------------
* Accepts a
  :class:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalWavelengthMap`
  (the Stage-3 output).

* For each echelle order that has enough accepted arc-line matches, fits an
  independent 1-D polynomial::

      wavelength_um(col) ≈ Σ_k  a_k · col^k

  using the matched arc-line data already stored in the map.

* For each coefficient index *k*, collects the per-order coefficient values
  ``a_k(order)`` and fits them with a low-degree polynomial in echelle order
  number::

      a_k(order) ≈ Σ_j  d_{k,j} · v(order)^j

  where ``v = order_ref / order_number`` is the normalized inverse-order
  coordinate (the same encoding used by
  :mod:`~pyspextool.instruments.ishell.wavecal_2d_surface`, physically
  motivated by the echelle grating equation ``m · λ ≈ const``).

* Stores the result in a :class:`RefinedCoefficientSurface` dataclass that
  exposes evaluation helpers.

What this module does NOT do (by design)
-----------------------------------------
* **No rectification-index generation** – interpolation arrays for 2-D
  detector resampling are not produced.

* **No iterative sigma-clipping** – the fits are single-pass ordinary
  least-squares with no outlier rejection.

* **No full IDL coefficient-indexing compatibility** – the IDL ``2DXD``
  model uses a specific coefficient ordering and normalization that is not
  reproduced here.  This scaffold captures the *structure* (per-order
  polynomials whose coefficients vary with order) but not the exact
  parameterisation.

* **No science-quality wavelength solution** – the result is a provisional
  scaffold for development and pipeline validation.

How this differs from Stage 4 (``wavecal_2d_surface.py``)
----------------------------------------------------------
Stage 4 fits a single 2-D polynomial directly in ``(order_number, col)``
space::

    wavelength_um ≈ Σ_{i,j}  c_{i,j} · u(col)^i · v(order)^j

Stage 5 instead fits **per-order** 1-D polynomials in ``col``, and then
models how each polynomial coefficient depends on order number.  The result
is a *coefficient surface*: an implicit 2-D surface expressed through
smoothly varying per-order polynomial coefficients.

Formally the two representations can approximate the same function, but the
coefficient-surface form is closer to the IDL Spextool 2DXD approach and
makes the per-order dispersion structure more transparent.

How this approximates the IDL Spextool 2DXD approach
------------------------------------------------------
The IDL 2DXD calibration (Cushing et al. 2004) fits wavelength-dispersion
polynomials per echelle order, then models the *coefficients* as smooth
functions of order to enforce cross-order consistency and enable
interpolation to orders with few or no arc lines.

This module implements the same two-level structure:

1. Per-order polynomial fit in detector column.
2. Smooth polynomial fit of each coefficient as a function of order number.

Differences from the full IDL solution:

* The per-order column polynomials here are fitted in **raw column units**
  (not normalized to the order-specific column range).  The IDL code
  normalizes columns to ``[-1, +1]`` within each order.  Raw columns are
  used here for simplicity; this does not affect correctness for the
  scaffold stage.

* The order-dependence fit uses ``v = order_ref / order_number``
  (normalized inverse order) rather than a plain integer offset, to match
  the Stage-4 convention.

* No iterative outlier rejection is performed.

* The coefficient indexing does not match the IDL ``WDEG``/``ODEG``
  parameter layout exactly.

What remains unimplemented
--------------------------
* Normalization of per-order column range (IDL convention).
* Iterative sigma-clipping on the coefficient-smoothness fits.
* Full IDL coefficient-index compatibility.
* Rectification-index generation.
* Science-quality wavelength solution.

Public API
----------
- :func:`fit_refined_coefficient_surface` – main entry point.
- :class:`RefinedCoefficientSurface` – result container with evaluation
  helpers and residual statistics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .wavecal_2d import ProvisionalWavelengthMap

__all__ = [
    "RefinedCoefficientSurface",
    "fit_refined_coefficient_surface",
]


# ---------------------------------------------------------------------------
# Per-order fit record (internal helper, not part of the public API)
# ---------------------------------------------------------------------------


@dataclass
class _PerOrderFit:
    """Internal record of one per-order polynomial fit.

    Parameters
    ----------
    order_number : float
        Echelle order number.
    n_points : int
        Number of accepted arc-line matches used in this fit.
    col_coeffs : ndarray, shape (disp_degree+1,)
        Polynomial coefficients from ``polyfit`` in *increasing degree*
        order (i.e. ``col_coeffs[0]`` is the constant term).
    fit_rms_um : float
        RMS residual (µm) of this per-order fit.
    cols : ndarray, shape (n_points,)
        Detector columns used in the fit.
    wavs_true : ndarray, shape (n_points,)
        Reference wavelengths (µm) used in the fit.
    wavs_pred : ndarray, shape (n_points,)
        Wavelengths predicted by this per-order polynomial.
    """

    order_number: float
    n_points: int
    col_coeffs: npt.NDArray
    fit_rms_um: float
    cols: npt.NDArray
    wavs_true: npt.NDArray
    wavs_pred: npt.NDArray


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RefinedCoefficientSurface:
    """Result of the coefficient-surface refinement fit.

    Stores the two-level fit result:

    1. **Per-order polynomials** – for each solved echelle order, a 1-D
       polynomial ``wavelength_um(col) ≈ Σ_k a_k · col^k`` fitted to the
       accepted arc-line matches.

    2. **Order-dependence polynomials** – for each coefficient index *k*,
       a 1-D polynomial in the normalized inverse-order coordinate
       ``v = order_ref / order_number`` that smooths the coefficient
       values across orders::

           a_k(order) ≈ Σ_j  d_{k,j} · v^j

    Evaluation
    ----------
    Use :meth:`eval` (scalar) or :meth:`eval_array` (array) to predict
    wavelength at arbitrary ``(order_number, col)`` combinations.  The
    prediction uses the **smoothed** (order-polynomial) coefficients, not
    the raw per-order fitted coefficients.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    disp_degree : int
        Polynomial degree used for each per-order ``wavelength(col)`` fit.
    order_smooth_degree : int
        Polynomial degree used to fit each coefficient as a function of
        normalized inverse order.
    order_ref : float
        Reference order number (minimum order in the fit data).  Used to
        compute the normalized inverse-order coordinate
        ``v = order_ref / order_number``.
    order_smooth_coeffs : ndarray, shape (disp_degree+1, order_smooth_degree+1)
        Smoothed coefficient array.
        ``order_smooth_coeffs[k, j]`` is the *j*-th polynomial coefficient
        in the fit of ``a_k`` as a function of ``v``.  That is::

            a_k(v) ≈ Σ_j  order_smooth_coeffs[k, j] · v^j

    per_order_fits : list of :class:`_PerOrderFit`
        One entry per echelle order that had enough data to fit.
    per_order_rms_um : ndarray, shape (n_orders_fit,)
        RMS residual (µm) for each per-order polynomial fit.
    per_order_order_numbers : ndarray, shape (n_orders_fit,)
        Echelle order numbers for the orders that were fitted.
    smooth_rms_um : ndarray, shape (disp_degree+1,)
        RMS residual (µm) of the order-smoothness fit for each coefficient
        index *k*.
    n_orders_fit : int
        Number of echelle orders for which a per-order polynomial was
        fitted.
    n_points_total : int
        Total number of accepted arc-line matches used across all per-order
        fits.
    n_orders_skipped : int
        Number of orders that were skipped because they had fewer than
        ``min_lines_per_order`` accepted matches.

    Notes
    -----
    All fits are single-pass ordinary least-squares; no iterative outlier
    rejection is performed.

    The per-order polynomial coefficients are in **raw column units** (not
    normalized within each order).  The evaluation helpers use the smoothed
    order-dependence model, not the raw per-order fit coefficients directly.

    This is a provisional scaffold.  See the module docstring for the full
    list of limitations.
    """

    mode: str
    disp_degree: int
    order_smooth_degree: int
    order_ref: float
    order_smooth_coeffs: npt.NDArray  # shape (disp_degree+1, order_smooth_degree+1)
    per_order_fits: list[_PerOrderFit]
    per_order_rms_um: npt.NDArray  # shape (n_orders_fit,)
    per_order_order_numbers: npt.NDArray  # shape (n_orders_fit,)
    smooth_rms_um: npt.NDArray  # shape (disp_degree+1,)
    n_orders_fit: int
    n_points_total: int
    n_orders_skipped: int

    # ------------------------------------------------------------------
    # Internal normalization helper
    # ------------------------------------------------------------------

    def _v(self, order_number: float | npt.NDArray) -> float | npt.NDArray:
        """Compute normalized inverse-order coordinate.

        Returns ``v = order_ref / order_number``, motivated by the echelle
        grating equation ``m · λ ≈ const``: wavelength scales as ``1/m`` at
        fixed column, so a polynomial in ``1/m`` captures the leading
        inter-order variation with a low-degree expansion.
        """
        return self.order_ref / np.asarray(order_number, dtype=float)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def eval(self, order_number: float, col: float) -> float:
        """Evaluate the coefficient surface at a single (order_number, col).

        Uses the smoothed order-dependence model to predict each polynomial
        coefficient ``a_k`` at the given order number, then evaluates the
        resulting 1-D column polynomial at ``col``.

        Parameters
        ----------
        order_number : float
            Echelle order number.
        col : float
            Detector column.

        Returns
        -------
        float
            Predicted wavelength in µm.
        """
        v = float(self._v(float(order_number)))
        # Compute smoothed per-order coefficients at this order
        # order_smooth_coeffs[k, :] are the poly coefficients in v for a_k
        # np.polynomial.polynomial.polyval(v, c) evaluates Σ c[j] * v^j
        col_coeffs = np.array(
            [
                float(
                    np.polynomial.polynomial.polyval(
                        v, self.order_smooth_coeffs[k]
                    )
                )
                for k in range(self.disp_degree + 1)
            ]
        )
        # Evaluate Σ col_coeffs[k] * col^k
        return float(np.polynomial.polynomial.polyval(float(col), col_coeffs))

    def eval_array(
        self,
        order_numbers: npt.NDArray,
        cols: npt.NDArray,
    ) -> npt.NDArray:
        """Evaluate the coefficient surface at arrays of (order_number, col).

        Parameters
        ----------
        order_numbers : array_like, shape (n,)
            Echelle order numbers.
        cols : array_like, shape (n,)
            Detector columns.

        Returns
        -------
        ndarray, shape (n,)
            Predicted wavelengths in µm.
        """
        ords = np.asarray(order_numbers, dtype=float)
        cs = np.asarray(cols, dtype=float)
        v = self._v(ords)  # shape (n,)

        # Build smoothed per-point column-polynomial coefficients
        # col_coeffs_mat[k, i] = a_k evaluated at order i
        col_coeffs_mat = np.stack(
            [
                np.polynomial.polynomial.polyval(v, self.order_smooth_coeffs[k])
                for k in range(self.disp_degree + 1)
            ],
            axis=0,
        )  # shape (disp_degree+1, n)

        # Evaluate Σ_k col_coeffs_mat[k, i] * col[i]^k for each i
        # Build powers of col: shape (disp_degree+1, n)
        col_powers = np.stack(
            [cs**k for k in range(self.disp_degree + 1)], axis=0
        )
        return np.sum(col_coeffs_mat * col_powers, axis=0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_refined_coefficient_surface(
    wav_map: "ProvisionalWavelengthMap",
    *,
    disp_degree: int = 3,
    order_smooth_degree: int = 2,
    min_lines_per_order: int = 2,
) -> RefinedCoefficientSurface:
    """Fit the IDL-style coefficient-surface refinement on per-order data.

    Implements the two-level fit structure used by the IDL Spextool 2DXD
    calibration:

    1. For each echelle order with at least ``min_lines_per_order`` accepted
       arc-line matches, fit a 1-D polynomial::

           wavelength_um(col) ≈ Σ_k  a_k · col^k

    2. For each coefficient index *k*, fit a 1-D polynomial in the
       normalized inverse-order coordinate ``v = order_ref / order_number``
       to smooth the coefficient values across orders::

           a_k(order) ≈ Σ_j  d_{k,j} · v^j

    The result is a :class:`RefinedCoefficientSurface` that can evaluate
    the smoothed surface at arbitrary ``(order_number, col)`` combinations.

    Parameters
    ----------
    wav_map : :class:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalWavelengthMap`
        Per-order provisional wavelength solutions from
        :func:`~pyspextool.instruments.ishell.wavecal_2d.fit_provisional_wavelength_map`.
        The accepted arc-line matches from each order are used for fitting.
    disp_degree : int, default 3
        Polynomial degree for the per-order ``wavelength(col)`` fits.
        If an individual order has fewer than ``disp_degree + 1`` accepted
        matches, the degree is automatically reduced for that order.
    order_smooth_degree : int, default 2
        Polynomial degree for the order-dependence fit of each coefficient
        ``a_k(order)``.  Must be strictly less than the number of fitted
        orders; if not, it is automatically reduced.
    min_lines_per_order : int, default 2
        Minimum number of accepted arc-line matches required for an order
        to be included in the per-order fit.  Orders with fewer accepted
        matches are skipped and a :exc:`RuntimeWarning` is issued.

    Returns
    -------
    :class:`RefinedCoefficientSurface`
        Result container with smoothed coefficient surface, per-order fits,
        and residual statistics.

    Raises
    ------
    ValueError
        If ``disp_degree < 0`` or ``order_smooth_degree < 0``.
    ValueError
        If ``min_lines_per_order < 1``.
    ValueError
        If ``wav_map`` contains no accepted matches (nothing to fit).
    ValueError
        If fewer than 2 orders have enough data to fit (the coefficient
        smoothness step requires at least 2 data points).

    Warnings
    --------
    RuntimeWarning
        Emitted for each order that is skipped due to insufficient matches.
    RuntimeWarning
        Emitted if ``order_smooth_degree`` must be reduced because there are
        too few fitted orders.

    Notes
    -----
    **Column units:** Per-order polynomials are fitted in raw detector-column
    units, not normalized within each order.  This differs from the IDL
    convention (which normalizes columns to ``[-1, +1]`` within each order).
    Evaluation via :meth:`~RefinedCoefficientSurface.eval` uses the same raw
    column units.

    **Normalization basis:** The order-dependence fit uses
    ``v = order_ref / order_number`` (normalized inverse order), matching the
    convention in :mod:`~pyspextool.instruments.ishell.wavecal_2d_surface`.

    **Provisional status:** This function produces a development scaffold, not
    a science-quality wavelength solution.  See the module docstring for the
    complete list of intentional limitations.

    Examples
    --------
    Typical usage following the per-order provisional fit:

    >>> from pyspextool.instruments.ishell.wavecal_2d import (
    ...     fit_provisional_wavelength_map)
    >>> from pyspextool.instruments.ishell.wavecal_2d_refine import (
    ...     fit_refined_coefficient_surface)
    >>>
    >>> # … obtain arc_result, wci, ll as in wavecal_2d examples …
    >>> wav_map = fit_provisional_wavelength_map(arc_result, wci, ll)
    >>> refined = fit_refined_coefficient_surface(wav_map, disp_degree=3)
    >>>
    >>> print(f"Orders fitted: {refined.n_orders_fit}")
    >>> print(f"Smooth-coeff RMS (k=0): {refined.smooth_rms_um[0]*1e3:.3f} nm")
    >>> # Evaluate at a specific (order, column):
    >>> wav = refined.eval(order_number=330, col=1024)
    >>> print(f"Predicted wavelength at order 330, col 1024: {wav:.6f} µm")
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if disp_degree < 0:
        raise ValueError(f"disp_degree must be >= 0; got {disp_degree}.")
    if order_smooth_degree < 0:
        raise ValueError(
            f"order_smooth_degree must be >= 0; got {order_smooth_degree}."
        )
    if min_lines_per_order < 1:
        raise ValueError(
            f"min_lines_per_order must be >= 1; got {min_lines_per_order}."
        )

    # Check that there is at least some data
    order_numbers_all, cols_all, wavs_all = wav_map.collect_for_surface_fit()
    if len(order_numbers_all) == 0:
        raise ValueError(
            "wav_map contains no accepted arc-line matches. "
            "Cannot fit a refined coefficient surface."
        )

    # ------------------------------------------------------------------
    # Stage 1: Per-order polynomial fits
    # ------------------------------------------------------------------
    per_order_fits: list[_PerOrderFit] = []
    n_orders_skipped = 0

    for sol in wav_map.order_solutions:
        n_matches = sol.n_accepted
        if n_matches < min_lines_per_order:
            warnings.warn(
                f"Order {sol.order_number} (index {sol.order_index}): "
                f"only {n_matches} accepted match(es) "
                f"(minimum {min_lines_per_order}); skipping.",
                RuntimeWarning,
                stacklevel=2,
            )
            n_orders_skipped += 1
            continue

        cols_ord = sol.centerline_cols
        wavs_ord = sol.reference_wavs

        # Reduce degree if not enough points for the requested degree.
        # A warning is emitted so the caller knows which orders were affected.
        effective_degree = min(disp_degree, n_matches - 1)
        if effective_degree < disp_degree:
            warnings.warn(
                f"Order {sol.order_number} (index {sol.order_index}): "
                f"requested disp_degree={disp_degree} but only {n_matches} "
                f"accepted match(es); reducing polynomial degree to "
                f"{effective_degree}.",
                RuntimeWarning,
                stacklevel=2,
            )

        # np.polynomial.polynomial.polyfit returns coefficients in
        # increasing degree: coeffs[0] is constant, coeffs[k] is col^k
        col_coeffs = np.polynomial.polynomial.polyfit(
            cols_ord, wavs_ord, effective_degree
        )

        wavs_pred_ord = np.polynomial.polynomial.polyval(cols_ord, col_coeffs)
        residuals_ord = wavs_ord - wavs_pred_ord
        fit_rms = float(np.sqrt(np.mean(residuals_ord**2)))

        per_order_fits.append(
            _PerOrderFit(
                order_number=float(sol.order_number),
                n_points=n_matches,
                col_coeffs=col_coeffs,
                fit_rms_um=fit_rms,
                cols=cols_ord.copy(),
                wavs_true=wavs_ord.copy(),
                wavs_pred=wavs_pred_ord,
            )
        )

    if len(per_order_fits) < 2:
        raise ValueError(
            f"Only {len(per_order_fits)} order(s) had enough data to fit "
            f"(minimum 2 required for the order-smoothness step). "
            "Reduce min_lines_per_order or collect more arc-line matches."
        )

    # ------------------------------------------------------------------
    # Stage 2: Fit order-dependence of each polynomial coefficient
    # ------------------------------------------------------------------
    # Collect per-order values
    fitted_order_numbers = np.array(
        [f.order_number for f in per_order_fits], dtype=float
    )
    order_ref = float(fitted_order_numbers.min())
    v = order_ref / fitted_order_numbers  # normalized inverse order

    # Determine number of available (filled) coefficients.
    # Each per-order fit may have fewer than disp_degree+1 coefficients if
    # the degree was reduced.  Pad shorter arrays with zeros so we always
    # have a (disp_degree+1)-element array per order.
    n_coeff = disp_degree + 1
    coeff_matrix = np.zeros((len(per_order_fits), n_coeff), dtype=float)
    for row, pof in enumerate(per_order_fits):
        nc = len(pof.col_coeffs)
        coeff_matrix[row, :nc] = pof.col_coeffs

    # Reduce order_smooth_degree if there are too few orders
    effective_smooth_degree = order_smooth_degree
    if effective_smooth_degree >= len(per_order_fits):
        effective_smooth_degree = len(per_order_fits) - 1
        warnings.warn(
            f"order_smooth_degree reduced from {order_smooth_degree} to "
            f"{effective_smooth_degree} because only {len(per_order_fits)} "
            "order(s) were fitted.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Fit each coefficient's order-dependence
    smooth_coeffs_list = []
    smooth_rms_list = []

    for k in range(n_coeff):
        a_k = coeff_matrix[:, k]  # shape (n_orders_fit,)
        d_k = np.polynomial.polynomial.polyfit(
            v, a_k, effective_smooth_degree
        )
        a_k_pred = np.polynomial.polynomial.polyval(v, d_k)
        rms_k = float(np.sqrt(np.mean((a_k - a_k_pred) ** 2)))
        smooth_coeffs_list.append(d_k)
        smooth_rms_list.append(rms_k)

    # Shape: (disp_degree+1, effective_smooth_degree+1)
    order_smooth_coeffs = np.array(smooth_coeffs_list, dtype=float)
    smooth_rms_um = np.array(smooth_rms_list, dtype=float)

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    per_order_rms_um = np.array([f.fit_rms_um for f in per_order_fits])
    per_order_order_numbers = fitted_order_numbers
    n_points_total = sum(f.n_points for f in per_order_fits)

    return RefinedCoefficientSurface(
        mode=wav_map.mode,
        disp_degree=disp_degree,
        order_smooth_degree=effective_smooth_degree,
        order_ref=order_ref,
        order_smooth_coeffs=order_smooth_coeffs,
        per_order_fits=per_order_fits,
        per_order_rms_um=per_order_rms_um,
        per_order_order_numbers=per_order_order_numbers,
        smooth_rms_um=smooth_rms_um,
        n_orders_fit=len(per_order_fits),
        n_points_total=n_points_total,
        n_orders_skipped=n_orders_skipped,
    )
