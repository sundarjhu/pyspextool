"""
Provisional global wavelength-surface scaffold for iSHELL 2DXD reduction.

This module implements the **fourth stage** of the iSHELL 2DXD reduction
scaffold: fitting a first provisional global surface

    wavelength_um = surface(order_number, col)

from the per-order provisional solutions already produced by
:mod:`~pyspextool.instruments.ishell.wavecal_2d`.

What this module does
---------------------
* Accepts a
  :class:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalWavelengthMap`
  (or the ``(order_numbers, cols, wavs)`` arrays returned by
  :meth:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalWavelengthMap.collect_for_surface_fit`).

* Normalizes the inputs:

  - **Columns** are mapped to ``u ∈ [−1, +1]`` by centering and scaling to
    the observed column range::

        u = (col − col_center) / col_half

    where ``col_center = (col_max + col_min) / 2`` and
    ``col_half = (col_max − col_min) / 2``.

  - **Order numbers** are mapped to ``v = order_ref / order_number``
    (normalized inverse order).  This encoding is physically motivated by
    the echelle grating equation: ``m · λ ≈ const``, so at fixed column the
    wavelength scales as ``1/m``.  The reference order ``order_ref`` is the
    **minimum** order number in the data, giving ``v = 1`` for that order
    and ``v < 1`` for higher-numbered (shorter-wavelength) orders.

* Fits a 2-D polynomial in ``(u, v)`` to the collected match data by
  ordinary least squares::

      wavelength_um ≈ Σ_{i,j} c_{i,j} · u^i · v^j

  where ``0 ≤ i ≤ col_degree`` and ``0 ≤ j ≤ order_degree``.  The full
  tensor-product basis is used (all cross terms included), giving
  ``(col_degree + 1) × (order_degree + 1)`` free parameters.  The design
  matrix is built with :func:`numpy.polynomial.polynomial.polyvander2d`.

* Returns a :class:`GlobalSurfaceFitResult` that records the coefficients,
  normalization parameters, point-wise residuals, and basic quality metrics.

What this module does NOT do (by design)
-----------------------------------------
* **No coefficient-surface refinement** – the IDL-style 2DXD reduction
  iteratively refines a *coefficient surface* by fitting polynomial
  descriptions of per-order polynomial coefficients as a function of order
  number.  That multi-level fit is **not** implemented here.

* **No sigma-clipping or iterative outlier rejection** – the current fit is
  a single ordinary-least-squares pass with no outlier masking.  Residuals
  are reported but the fit is not re-run after removing outliers.

* **No rectification-index generation** – interpolation arrays for 2-D
  detector resampling are not produced.

* **No science-quality wavelength solution** – the result is a provisional
  scaffold intended for exploratory development and pipeline validation, not
  for final scientific spectral extraction.

How this differs from the per-order provisional fits
------------------------------------------------------
The per-order fits produced by
:func:`~pyspextool.instruments.ishell.wavecal_2d.fit_provisional_wavelength_map`
are **independent 1-D polynomials**, one per echelle order::

    wavelength_um(col) ≈ Σ_i  a_i · col^i    [per-order]

Each per-order polynomial is fitted only to the arc lines matched within
that order.

The global surface fit produced by this module combines **all orders
simultaneously** into a single 2-D model::

    wavelength_um(order, col) ≈ Σ_{i,j}  c_{i,j} · u(col)^i · v(order)^j

This single fit enforces cross-order smoothness: information from well-
constrained orders regularizes the wavelength assignments for orders that
have fewer matched lines.  It is therefore more robust than per-order fits
when individual orders have sparse arc lines.

How this differs from the eventual full IDL-style 2DXD coefficient-surface fit
--------------------------------------------------------------------------------
In the full Cushing et al. (2004) 2DXD approach, the per-order polynomial
coefficients are themselves modelled as smooth functions of echelle order
number, producing a *coefficient surface*.  The surface is then used to
generate a *rectification map*: a look-up table that, for every detector
pixel, gives the rectified (wavelength, slit-position) coordinate.

This module is a simplified precursor:

* It fits **wavelength directly** (not the polynomial coefficients) as a
  surface over ``(order_number, col)`` space.
* It uses a single least-squares pass with no iterative refinement.
* It produces no rectification map.

The intention is to validate that the per-order matched-line data are
internally consistent before the more elaborate coefficient-surface machinery
is built.

Order-number assignment caveats
---------------------------------
The order numbers used here are inherited from
:attr:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalLineMatch.order_number`
in the input ``ProvisionalWavelengthMap``.  Those numbers are themselves
assigned heuristically (the *i*-th traced order → *i*-th entry in
``wavecalinfo.orders``).  Until the order-number assignment is validated
against an independent cross-check (e.g. comparison with a laboratory
wavelength atlas or a known object spectrum), the echelle order numbers
should be treated as *nominal* labels, not confirmed identifications.

Normalization basis choice
----------------------------
The choice of ``v = order_ref / order_number`` (normalized inverse order) is
motivated by echelle grating physics.  For a perfect echelle:

    m · λ = const   (at the blaze angle)

so ``λ ∝ 1/m`` at fixed column.  A polynomial in ``1/m`` (or equivalently
in ``v``) therefore captures the leading-order inter-order wavelength
variation with a low-degree expansion.

An alternative – fitting a plain polynomial in normalized order number,
``v′ = (order_number − order_mid) / order_half`` – is mathematically
equivalent if enough terms are included, but requires higher degree to
approximate the ``1/m`` dependence well.  The inverse-order basis is
therefore preferred here.

Both the ``col_center``, ``col_half``, and ``order_ref`` normalization
parameters are stored in :class:`GlobalSurfaceFitResult` so that the
fitted surface can be evaluated at any detector column and order number.

Public API
----------
- :func:`fit_global_wavelength_surface` – main entry point.
- :class:`GlobalSurfaceFitResult` – result container with evaluation
  helpers and quality metrics.
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
    "GlobalSurfaceFitResult",
    "fit_global_wavelength_surface",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GlobalSurfaceFitResult:
    """Result of provisional global wavelength-surface fitting.

    Contains the 2-D polynomial coefficients, normalization parameters,
    point-wise residuals, and basic quality metrics from a fit of the form::

        wavelength_um ≈ Σ_{i,j}  c_{i,j} · u(col)^i · v(order_number)^j

    where ``u = (col − col_center) / col_half`` and
    ``v = order_ref / order_number``.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    n_points : int
        Number of arc-line matches used in the fit.
    col_degree : int
        Maximum power of the normalized-column basis variable ``u``.
    order_degree : int
        Maximum power of the normalized inverse-order basis variable ``v``.
    coeffs : ndarray, shape ((col_degree+1) * (order_degree+1),)
        Flattened 2-D polynomial coefficients in row-major (C) order.
        ``coeffs[i*(order_degree+1) + j]`` is the coefficient of
        ``u^i · v^j``.  Reshape to ``(col_degree+1, order_degree+1)`` to
        index as ``coeffs_2d[i, j]``.
    col_center : float
        Column used as the centre of the normalized column coordinate:
        ``u = (col − col_center) / col_half``.  Equal to
        ``(col_max + col_min) / 2`` of the input columns.
    col_half : float
        Column half-range used in the normalization:
        ``u = (col − col_center) / col_half``.  Equal to
        ``(col_max − col_min) / 2`` of the input columns, or ``1.0`` if all
        input columns are identical (degenerate case).  Always positive.
    order_ref : float
        Reference order number for the inverse-order normalization:
        ``v = order_ref / order_number``.  Set to the minimum order number
        in the input data (so ``v = 1`` for that order).
    rms_um : float
        Root-mean-square fit residual (µm) over all input points.
    max_abs_residual_um : float
        Maximum absolute residual (µm) over all input points.
    order_numbers : ndarray, shape (n_points,)
        Echelle order number for each input point.
    cols : ndarray, shape (n_points,)
        Detector column for each input point.
    wavs_true : ndarray, shape (n_points,)
        Reference wavelength (µm) for each input point.
    wavs_pred : ndarray, shape (n_points,)
        Wavelength predicted by the fitted surface for each input point.
    residuals_um : ndarray, shape (n_points,)
        Signed residuals ``wavs_true − wavs_pred`` (µm) for each input point.
    n_orders_used : int
        Number of distinct echelle order numbers present in the input.
    rank : int
        Numerical rank of the design matrix as returned by
        :func:`numpy.linalg.lstsq`.  A rank smaller than the number of
        fit parameters indicates a degenerate or nearly-degenerate fit.
    n_params : int
        Total number of free parameters: ``(col_degree+1)*(order_degree+1)``.

    Notes
    -----
    All residuals and quality metrics refer to the single-pass ordinary
    least-squares fit.  No iterative outlier rejection is performed.

    The fit is **provisional**: it should not be used for science-quality
    spectral extraction.  See the module docstring for the full list of
    limitations.
    """

    mode: str
    n_points: int
    col_degree: int
    order_degree: int
    coeffs: npt.NDArray
    col_center: float
    col_half: float
    order_ref: float
    rms_um: float
    max_abs_residual_um: float
    order_numbers: npt.NDArray
    cols: npt.NDArray
    wavs_true: npt.NDArray
    wavs_pred: npt.NDArray
    residuals_um: npt.NDArray
    n_orders_used: int
    rank: int
    n_params: int

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def coeffs_2d(self) -> npt.NDArray:
        """2-D view of the coefficient array.

        Returns
        -------
        ndarray, shape (col_degree+1, order_degree+1)
            ``coeffs_2d[i, j]`` is the coefficient of ``u^i · v^j``.
        """
        return self.coeffs.reshape(self.col_degree + 1, self.order_degree + 1)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _normalize(
        self,
        order_numbers: npt.NDArray,
        cols: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Return (u, v) normalized coordinates for the given inputs."""
        u = (cols - self.col_center) / self.col_half
        v = self.order_ref / order_numbers
        return u, v

    def eval(self, order_number: float, col: float) -> float:
        """Evaluate the fitted surface at a single (order_number, col) point.

        Parameters
        ----------
        order_number : float
            Echelle order number (use the same nominal values as in the fit).
        col : float
            Detector column.

        Returns
        -------
        float
            Predicted wavelength in µm.
        """
        u, v = self._normalize(
            np.array([float(order_number)]),
            np.array([float(col)]),
        )
        A = np.polynomial.polynomial.polyvander2d(
            u, v, [self.col_degree, self.order_degree]
        )
        return float(A @ self.coeffs)

    def eval_array(
        self,
        order_numbers: npt.NDArray,
        cols: npt.NDArray,
    ) -> npt.NDArray:
        """Evaluate the fitted surface at arrays of (order_number, col) points.

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
        u, v = self._normalize(ords, cs)
        A = np.polynomial.polynomial.polyvander2d(
            u, v, [self.col_degree, self.order_degree]
        )
        return A @ self.coeffs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_global_wavelength_surface(
    wav_map: "ProvisionalWavelengthMap",
    *,
    col_degree: int = 4,
    order_degree: int = 2,
) -> GlobalSurfaceFitResult:
    """Fit a provisional global wavelength surface over (order_number, col).

    Takes the accepted arc-line matches from all orders in ``wav_map`` and
    fits a single 2-D polynomial::

        wavelength_um ≈ Σ_{i,j}  c_{i,j} · u(col)^i · v(order)^j

    where:

    * ``u = (col − col_center) / col_half`` maps the observed column range
      to ``[−1, +1]``,
    * ``v = order_ref / order_number`` is the normalized inverse order
      (``order_ref`` is the minimum order number in the data, so
      ``v = 1`` for that order and ``v < 1`` for higher-numbered orders).

    The full tensor-product basis is used (``(col_degree+1)*(order_degree+1)``
    free parameters) and the coefficients are estimated by ordinary
    least squares via :func:`numpy.linalg.lstsq`.

    Parameters
    ----------
    wav_map : :class:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalWavelengthMap`
        Provisional per-order wavelength solutions from
        :func:`~pyspextool.instruments.ishell.wavecal_2d.fit_provisional_wavelength_map`.
        The accepted-match arrays are extracted via
        :meth:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalWavelengthMap.collect_for_surface_fit`.
    col_degree : int, default 4
        Maximum polynomial degree in the normalized column coordinate ``u``.
        A higher degree captures more complex dispersion curvature but
        increases the risk of over-fitting when the number of matched lines
        per order is small.
    order_degree : int, default 2
        Maximum polynomial degree in the normalized inverse-order coordinate
        ``v``.  For an ideal echelle, the leading variation with order number
        is captured by degree 1; degree 2 allows for a small second-order
        correction.

    Returns
    -------
    :class:`GlobalSurfaceFitResult`
        Result container with fitted coefficients, normalization parameters,
        and quality metrics.

    Raises
    ------
    ValueError
        If ``wav_map`` contains no accepted matches (nothing to fit).
    ValueError
        If ``col_degree < 0`` or ``order_degree < 0``.
    ValueError
        If the number of matched points is less than the number of free
        parameters ``(col_degree+1)*(order_degree+1)``.

    Warnings
    --------
    RuntimeWarning
        Emitted if the design matrix is numerically rank-deficient.

    Notes
    -----
    **Heuristic assumptions:**

    * The order-number → echelle-order mapping in ``wav_map`` is assumed to
      be correct.  See the module docstring and
      :mod:`~pyspextool.instruments.ishell.wavecal_2d` for the heuristic
      and its limitations.

    * The single-pass least-squares fit contains no outlier rejection.  A
      small number of mis-identified arc lines can noticeably bias the
      surface.

    * The normalization choice (``v = order_ref / order_number``) is
      physically motivated but is not the only valid choice.  See the module
      docstring for a discussion.

    **Provisional status:** This function produces a development scaffold, not
    a science-quality wavelength solution.  See the module docstring for the
    complete list of intentional limitations.

    Examples
    --------
    Typical usage following the per-order provisional fit:

    >>> from pyspextool.instruments.ishell.wavecal_2d import (
    ...     fit_provisional_wavelength_map)
    >>> from pyspextool.instruments.ishell.wavecal_2d_surface import (
    ...     fit_global_wavelength_surface)
    >>>
    >>> # … obtain arc_result, wci, ll as in wavecal_2d examples …
    >>> wav_map = fit_provisional_wavelength_map(arc_result, wci, ll)
    >>> surface = fit_global_wavelength_surface(wav_map)
    >>>
    >>> print(f"Global surface RMS: {surface.rms_um*1e3:.2f} nm")
    >>> print(f"Coefficients shape: {surface.coeffs_2d.shape}")
    >>> # Evaluate at a specific (order, column):
    >>> wav = surface.eval(order_number=330, col=1024)
    >>> print(f"Predicted wavelength at order 330, col 1024: {wav:.6f} µm")
    """
    if col_degree < 0:
        raise ValueError(f"col_degree must be >= 0; got {col_degree}.")
    if order_degree < 0:
        raise ValueError(f"order_degree must be >= 0; got {order_degree}.")

    order_numbers, cols, wavs = wav_map.collect_for_surface_fit()
    n_points = len(order_numbers)

    if n_points == 0:
        raise ValueError(
            "wav_map contains no accepted arc-line matches. "
            "Cannot fit a global wavelength surface."
        )

    n_params = (col_degree + 1) * (order_degree + 1)
    if n_points < n_params:
        raise ValueError(
            f"Not enough matched arc lines ({n_points}) to fit a surface with "
            f"{n_params} free parameters (col_degree={col_degree}, "
            f"order_degree={order_degree}).  Reduce the polynomial degrees or "
            "accept more arc-line matches."
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    col_min = float(cols.min())
    col_max = float(cols.max())

    if col_max == col_min:
        # Degenerate: all points at the same column.  Use half-range = 1
        # to avoid division by zero.
        col_center = col_min
        col_half = 1.0
    else:
        col_center = 0.5 * (col_min + col_max)
        col_half = 0.5 * (col_max - col_min)

    u = (cols - col_center) / col_half

    order_ref = float(order_numbers.min())
    v = order_ref / order_numbers  # shape (n_points,)

    # ------------------------------------------------------------------
    # Build 2-D Vandermonde design matrix and solve by least squares
    # ------------------------------------------------------------------
    # polyvander2d returns shape (n_points, (col_degree+1)*(order_degree+1))
    # Column ordering: v (order axis) varies fastest within each u-block.
    # i.e. column index k = i*(order_degree+1) + j  <->  u^i * v^j
    A = np.polynomial.polynomial.polyvander2d(u, v, [col_degree, order_degree])

    coeffs, _residuals, rank, _sv = np.linalg.lstsq(A, wavs, rcond=None)

    if rank < n_params:
        warnings.warn(
            f"Global wavelength-surface design matrix is rank-deficient: "
            f"rank={rank} < n_params={n_params}.  The fit may be unreliable. "
            "Consider reducing col_degree or order_degree, or collecting more "
            "arc-line matches.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Evaluate residuals and quality metrics
    # ------------------------------------------------------------------
    wavs_pred = A @ coeffs
    residuals_um = wavs - wavs_pred
    rms_um = float(np.sqrt(np.mean(residuals_um**2)))
    max_abs_residual_um = float(np.max(np.abs(residuals_um)))
    n_orders_used = int(len(np.unique(order_numbers)))

    return GlobalSurfaceFitResult(
        mode=wav_map.mode,
        n_points=n_points,
        col_degree=col_degree,
        order_degree=order_degree,
        coeffs=coeffs,
        col_center=col_center,
        col_half=col_half,
        order_ref=order_ref,
        rms_um=rms_um,
        max_abs_residual_um=max_abs_residual_um,
        order_numbers=order_numbers,
        cols=cols,
        wavs_true=wavs,
        wavs_pred=wavs_pred,
        residuals_um=residuals_um,
        n_orders_used=n_orders_used,
        rank=rank,
        n_params=n_params,
    )
