"""
Proto-Horne variance-weighted optimal-extraction scaffold for iSHELL 2DXD reduction.

This module implements **Stages 17, 18, 19, and 20** of the iSHELL 2DXD
reduction scaffold.  Stage 17 introduced proto-Horne inverse-variance weighted
extraction using the existing profile estimate and a per-pixel variance image.
Stage 18 adds an optional iterative sigma-clipping outlier-rejection loop.
Stage 19 adds an optional profile re-estimation step inside the rejection loop.
Stage 20 adds support for smoothed-empirical and externally supplied spatial
profiles as alternative profile sources for the weighted extraction.

.. note::
    This is a **scaffold implementation**.  It is intentionally simple and
    does **not** represent final science-quality iSHELL or Spextool optimal
    extraction.  See ``docs/ishell_weighted_optimal_extraction.md`` for a
    full description of the algorithm, its relationship to Horne (1986), and
    a list of what remains unimplemented.

Pipeline stage summary
----------------------
.. list-table::
   :header-rows: 1

   * - Stage
     - Module
     - Purpose
   * - 1
     - ``tracing.py``
     - Flat-order tracing
   * - 2
     - ``arc_tracing.py``
     - Arc-line tracing
   * - 3
     - ``wavecal_2d.py``
     - Per-order provisional wavelength mapping
   * - 4
     - ``wavecal_2d_surface.py``
     - Provisional global wavelength surface
   * - 5
     - ``wavecal_2d_refine.py``
     - Coefficient-surface refinement
   * - 6
     - ``rectification_indices.py``
     - Rectification-index generation
   * - 7
     - ``rectified_orders.py``
     - Rectified order images
   * - 8
     - ``calibration_products.py``
     - Provisional calibration product containers
   * - 9
     - ``calibration_fits.py``
     - FITS calibration writer
   * - 10
     - ``extracted_spectra.py``
     - Whole-slit provisional extraction
   * - 11
     - ``aperture_extraction.py``
     - Aperture-aware extraction
   * - 12
     - ``optimal_extraction.py``
     - Profile-weighted extraction (unweighted scaffold)
   * - 14
     - ``variance_model.py``
     - Provisional variance-image generation
   * - **17**
     - **``weighted_optimal_extraction.py``**
     - **Proto-Horne variance-weighted extraction (this module)**
   * - **18**
     - **``weighted_optimal_extraction.py``**
     - **Iterative sigma-clipping outlier rejection (this module)**
   * - **19**
     - **``weighted_optimal_extraction.py``**
     - **Profile re-estimation after iterative rejection (this module)**
   * - **20**
     - **``weighted_optimal_extraction.py``**
     - **Smoothed / external profile sources (this module)**

What this module does
---------------------
* Accepts a
  :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
  (the Stage-7 output) and a :class:`WeightedExtractionDefinition`.

* For each order:

  1. Selects aperture pixels defined by *center_frac* ± *radius_frac* in
     fractional slit coordinates ``[0, 1]``.

  2. Optionally estimates and subtracts a per-column background as the
     median of pixels in a background annulus.

  3. Resolves a variance image (see **Variance source priority** below).

  4. Estimates a provisional spatial profile using one of three **profile
     source** modes (controlled by ``profile_source`` in
     :class:`WeightedExtractionDefinition`):

     - ``"empirical"`` (default): estimate the profile from the aperture
       sub-image using *profile_mode* (``"global_median"`` or
       ``"columnwise"``).

     - ``"smoothed_empirical"``: estimate the empirical profile then apply
       a Gaussian smoothing kernel along the **spatial axis** with standard
       deviation *profile_smooth_sigma* pixels.  Smoothing reduces
       sensitivity to single-pixel outliers in the profile.

     - ``"external"``: use the caller-supplied *external_profile* array
       instead of estimating from the data.  The profile is normalized
       (when *normalize_profile* is ``True``) but never re-estimated during
       iterative rejection.

  5. Normalizes the profile so its spatial sum equals 1.0 per column
     (when *normalize_profile* is ``True``).

  6. Applies proto-Horne inverse-variance weighting::

         weights[i, col] = P[i, col]^2 / V[i, col]

         flux_1d[col] = sum_i( P[i,col] * F[i,col] / V[i,col] )
                      / sum_i( P[i,col]^2 / V[i,col] )

         var_1d[col] = 1 / sum_i( P[i,col]^2 / V[i,col] )

     where ``P`` is the normalized profile, ``F`` is the
     background-subtracted aperture flux, and ``V`` is the per-pixel
     variance.

     Pixels with non-positive variance, non-finite data, or flagged by the
     user mask are excluded from both the numerator and the denominator.

     Columns with no valid support return NaN flux and NaN variance.

* Returns a :class:`WeightedExtractedSpectrumSet` collecting one
  :class:`WeightedExtractedOrderSpectrum` per echelle order.

Weighting formula
-----------------
The extraction formula is::

    flux_1d[col] = sum( P * F / V, axis=0 ) / sum( P^2 / V, axis=0 )
    var_1d[col]  = 1 / sum( P^2 / V, axis=0 )

This is the standard Horne (1986) optimal-extraction estimator applied to
the provisional empirical profile.  It is closer to Horne-style extraction
than Stage 12 (which uses ``sum(P*F)/sum(P^2)`` with no variance weighting)
because:

- the per-pixel variance ``V`` is explicitly used in the weights,
- the extracted variance ``var_1d = 1/sum(P^2/V)`` is the Cramér-Rao lower
  bound for unbiased linear estimation with known profile and variance.

When ``V`` is uniform (e.g. unit variance), the formula reduces to the
Stage-12 unweighted estimator for the flux, but the variance formula differs:
``var_1d = 1/sum(P^2)`` instead of ``sum(P^2 * V)``.

What this module does NOT do (by design)
-----------------------------------------
* **No full Horne iterative convergence** – Stage 19 adds profile
  re-estimation after each rejection step (when *reestimate_profile* is
  ``True``), but does not iterate to full convergence of the joint
  (profile, rejection mask) pair as in Horne (1986).
* **No PSF fitting beyond the empirical/smoothed profile estimate** –
  ``"empirical"`` and ``"smoothed_empirical"`` profiles are still estimated
  directly from the data being extracted.  ``"external"`` is the first
  scaffold step toward a truly independent profile source.
* **No automatic aperture centroiding** – center and radius are explicit.
* **No sophisticated sky modelling** – simple per-column median subtraction.
* **No correlated noise propagation** – only diagonal (per-pixel) variance.
* **No telluric correction** – that belongs to a later stage.
* **No order merging** – each order is extracted independently.

Variance source priority
------------------------
``explicit variance_image > variance_model (function arg) > definition's
variance_model (when use_variance_model=True) > unit variance (fallback)``

If no variance source is available, **unit variance (V=1 everywhere) is used
automatically**.  This is documented behaviour, not an error.  The extracted
flux is then identical to the Stage-12 estimator, and the variance is
``1/sum(P^2)``.  Pass an explicit *variance_image* or *variance_model* for
meaningful uncertainty estimates.

NaN handling
------------
:func:`numpy.sum` is applied after replacing invalid pixels with zero via
``numpy.where``.  Pixels excluded by the combined mask (non-finite data,
non-finite or non-positive variance, user mask) contribute zero to both the
numerator and denominator, ensuring they do not bias the result.  Spectral
columns with no valid aperture support return NaN flux and NaN variance.

Public API
----------
- :func:`extract_weighted_optimal` – main entry point.
- :class:`WeightedExtractionDefinition` – extraction aperture, profile-mode,
  and optional embedded variance-model definition.
- :class:`WeightedExtractedOrderSpectrum` – per-order 1-D spectrum container.
- :class:`WeightedExtractedSpectrumSet` – full result container.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from .rectified_orders import RectifiedOrderSet
from .variance_model import (
    VarianceModelDefinition,
    build_unit_variance_image,
    build_variance_image,
)

__all__ = [
    "WeightedExtractionDefinition",
    "WeightedExtractedOrderSpectrum",
    "WeightedExtractedSpectrumSet",
    "extract_weighted_optimal",
]

# Supported profile estimation modes.
_VALID_PROFILE_MODES = {"global_median", "columnwise"}

# Supported profile source identifiers (Stage 20).
_VALID_PROFILE_SOURCES = {"empirical", "smoothed_empirical", "external"}


# ---------------------------------------------------------------------------
# Extraction definition
# ---------------------------------------------------------------------------


@dataclass
class WeightedExtractionDefinition:
    """Definition of the aperture, profile, and variance-model parameters.

    All spatial coordinates are expressed in **fractional slit coordinates**
    in ``[0, 1]``, consistent with the ``spatial_frac`` axis stored in each
    :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrder`.

    Parameters
    ----------
    center_frac : float
        Fractional slit position of the aperture center.  Must be in
        ``[0, 1]``.
    radius_frac : float
        Half-width of the object aperture in fractional slit units.  Must
        be > 0.  Spatial pixels with
        ``|spatial_frac - center_frac| <= radius_frac`` are included in
        the aperture.
    background_inner : float or None, optional
        Inner edge of the background annulus in fractional slit distance
        from *center_frac*.  If ``None`` (default), background subtraction
        is disabled.  Must satisfy ``background_inner > radius_frac`` when
        provided.
    background_outer : float or None, optional
        Outer edge of the background annulus in fractional slit distance
        from *center_frac*.  Must satisfy
        ``background_outer > background_inner`` when provided.  Must be
        ``None`` if *background_inner* is ``None``.
    profile_mode : str, optional
        How to estimate the spatial profile inside the aperture when
        *profile_source* is ``"empirical"`` or ``"smoothed_empirical"``.
        One of:

        ``"global_median"`` (default)
            The profile is the per-row median of the aperture sub-image
            across all spectral columns.  Robust to spectral features but
            ignores wavelength-dependent profile variation.

        ``"columnwise"``
            The profile is estimated independently for each spectral
            column.  Tracks profile variation with wavelength but is noisier.

        Ignored when *profile_source* is ``"external"``.
    normalize_profile : bool, optional
        If ``True`` (default), normalize the profile so that its spatial
        sum equals 1.0 per column before computing the weighted extraction.
        Applied to all profile sources, including ``"external"``.
    use_variance_model : bool, optional
        If ``True``, the *variance_model* embedded in this definition will
        be used as a fallback variance source when no *variance_image* or
        function-level *variance_model* is supplied to
        :func:`extract_weighted_optimal`.  If ``True``, *variance_model*
        must be provided.  Defaults to ``False``.
    variance_model : :class:`~pyspextool.instruments.ishell.variance_model.VarianceModelDefinition` or None, optional
        Embedded noise-model definition used when *use_variance_model* is
        ``True``.  Must be ``None`` when *use_variance_model* is ``False``.
    reject_outliers : bool, optional
        If ``True``, apply an iterative sigma-clipping outlier-rejection
        loop after the initial profile estimation.  Defaults to ``False``.
        When ``False``, behavior is identical to Stage 17 (single extraction
        pass, no rejection).
    max_iterations : int, optional
        Maximum number of sigma-clipping iterations.  Must be >= 1.
        Defaults to 3.  The loop may stop earlier if no new pixels are
        rejected.
    sigma_clip : float, optional
        Rejection threshold in units of normalized residuals
        (``|residual| / sqrt(variance)``).  Must be > 0.  Defaults to 5.0.
        Pixels with normalized residuals exceeding *sigma_clip* are marked
        as bad and excluded from subsequent iterations.
    min_valid_pixels : int, optional
        Minimum number of valid (unmasked) aperture pixels that must remain
        in a spectral column after rejection.  Must be >= 1.  Defaults to
        2.  If rejecting all outlier candidates in a column would leave
        fewer than *min_valid_pixels* valid pixels, no pixels are rejected
        in that column for that iteration.
    reestimate_profile : bool, optional
        If ``True``, re-estimate the spatial profile from the surviving
        good pixels after each rejection iteration.  Defaults to ``False``.
        Has no effect when *reject_outliers* is ``False``.  When ``True``,
        the profile used for the next iteration's model and for the final
        extraction is updated using the same *profile_mode* and
        *normalize_profile* settings.  This moves the scaffold closer to
        the Horne (1986) iterative scheme but is still a scaffold
        simplification (no full convergence criterion).

        When *profile_source* is ``"external"``, this flag is **ignored**:
        the external profile is always kept fixed regardless of rejections.
        When *profile_source* is ``"smoothed_empirical"``, re-estimation
        re-estimates the empirical profile and re-applies smoothing.
    profile_source : str, optional
        Controls where the spatial profile comes from.  One of:

        ``"empirical"`` (default)
            Estimate the profile from the current aperture data using
            *profile_mode*.  This is the original Stage 17–19 behavior.

        ``"smoothed_empirical"``
            Estimate the empirical profile then apply a Gaussian smoothing
            kernel along the **spatial axis** with standard deviation
            *profile_smooth_sigma* pixels.  Smoothing reduces sensitivity
            to single-pixel outliers in the profile.  If
            *profile_smooth_sigma* is 0, the result is identical to
            ``"empirical"``.

        ``"external"``
            Use the caller-supplied *external_profile* array instead of
            estimating from the data.  The profile must have shape
            ``(n_ap_spatial, n_spectral)``, ``(n_ap_spatial, 1)``, or
            ``(n_ap_spatial,)``; the latter two are broadcast across all
            spectral columns.  *profile_mode* is ignored.  The profile is
            normalized when *normalize_profile* is ``True`` but is never
            re-estimated inside the iterative rejection loop.
    external_profile : ndarray or None, optional
        Caller-supplied spatial profile array.  **Required** when
        *profile_source* is ``"external"``; must be ``None`` for all other
        profile sources.  Accepted shapes:

        - ``(n_ap_spatial, n_spectral)`` — full 2-D profile.
        - ``(n_ap_spatial, 1)`` — same profile for all columns (broadcast).
        - ``(n_ap_spatial,)`` — 1-D profile, same for all columns.

        Here *n_ap_spatial* is the number of spatial pixels selected by
        the aperture mask (determined at extraction time).  Shape is
        validated when :func:`extract_weighted_optimal` is called, not
        here, because the aperture size is unknown at definition time.
    profile_smooth_sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel applied along
        the **spatial axis** when *profile_source* is
        ``"smoothed_empirical"``.  Must be >= 0.  Defaults to 0.0 (no
        smoothing, identical to ``"empirical"``).  Smoothing is performed
        with ``scipy.ndimage.gaussian_filter1d(..., axis=0,
        mode='reflect')``.  Ignored for other profile sources.

    Notes
    -----
    The aperture region is symmetric about *center_frac*.  The background
    annulus is also symmetric; pixels on both sides of the object contribute
    provided they satisfy the distance criterion.

    This is a scaffold simplification.  Real Spextool supports independent
    background apertures on either side of the object and uses more
    sophisticated sky-subtraction algorithms.
    """

    center_frac: float
    radius_frac: float
    background_inner: Optional[float] = None
    background_outer: Optional[float] = None
    profile_mode: str = "global_median"
    normalize_profile: bool = True
    use_variance_model: bool = False
    variance_model: Optional[VarianceModelDefinition] = None
    reject_outliers: bool = False
    max_iterations: int = 3
    sigma_clip: float = 5.0
    min_valid_pixels: int = 2
    reestimate_profile: bool = False
    profile_source: str = "empirical"
    external_profile: Optional[npt.NDArray] = None
    profile_smooth_sigma: float = 0.0

    def __post_init__(self) -> None:
        """Validate extraction-definition parameters."""
        if not (0.0 <= self.center_frac <= 1.0):
            raise ValueError(
                f"center_frac must be in [0, 1]; got {self.center_frac!r}."
            )
        if self.radius_frac <= 0.0:
            raise ValueError(
                f"radius_frac must be > 0; got {self.radius_frac!r}."
            )
        if (self.background_inner is None) != (self.background_outer is None):
            raise ValueError(
                "background_inner and background_outer must both be provided "
                "or both be None."
            )
        if self.background_inner is not None:
            if self.background_inner <= self.radius_frac:
                raise ValueError(
                    f"background_inner ({self.background_inner!r}) must be "
                    f"> radius_frac ({self.radius_frac!r})."
                )
            if self.background_outer <= self.background_inner:
                raise ValueError(
                    f"background_outer ({self.background_outer!r}) must be "
                    f"> background_inner ({self.background_inner!r})."
                )
        if self.profile_mode not in _VALID_PROFILE_MODES:
            raise ValueError(
                f"profile_mode must be one of "
                f"{sorted(_VALID_PROFILE_MODES)!r}; "
                f"got {self.profile_mode!r}."
            )
        if self.use_variance_model and self.variance_model is None:
            raise ValueError(
                "use_variance_model=True requires variance_model to be "
                "provided; got None."
            )
        if not self.use_variance_model and self.variance_model is not None:
            raise ValueError(
                "variance_model is provided but use_variance_model=False; "
                "set use_variance_model=True to use the embedded model, "
                "or set variance_model=None."
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1; got {self.max_iterations!r}."
            )
        if self.sigma_clip <= 0.0:
            raise ValueError(
                f"sigma_clip must be > 0; got {self.sigma_clip!r}."
            )
        if self.min_valid_pixels < 1:
            raise ValueError(
                f"min_valid_pixels must be >= 1; got {self.min_valid_pixels!r}."
            )
        # Stage 20: validate profile source fields.
        if self.profile_source not in _VALID_PROFILE_SOURCES:
            raise ValueError(
                f"profile_source must be one of "
                f"{sorted(_VALID_PROFILE_SOURCES)!r}; "
                f"got {self.profile_source!r}."
            )
        if self.profile_smooth_sigma < 0.0:
            raise ValueError(
                f"profile_smooth_sigma must be >= 0; "
                f"got {self.profile_smooth_sigma!r}."
            )
        if self.profile_source == "external" and self.external_profile is None:
            raise ValueError(
                "profile_source='external' requires external_profile to be "
                "provided; got None."
            )
        if self.profile_source != "external" and self.external_profile is not None:
            raise ValueError(
                "external_profile is provided but profile_source is "
                f"{self.profile_source!r}; set profile_source='external' "
                "to use the supplied profile, or set external_profile=None."
            )

    @property
    def has_background(self) -> bool:
        """True if a background region is defined."""
        return self.background_inner is not None


# ---------------------------------------------------------------------------
# Per-order 1-D spectrum
# ---------------------------------------------------------------------------


@dataclass
class WeightedExtractedOrderSpectrum:
    """Proto-Horne variance-weighted 1-D spectrum for one echelle order.

    Parameters
    ----------
    order : int
        Echelle order number.
    wavelength_um : ndarray, shape (n_spectral,)
        Wavelength axis in µm.
    flux : ndarray, shape (n_spectral,)
        Inverse-variance weighted 1-D flux::

            flux_1d[col] = sum(P * F / V, axis=0) / sum(P^2 / V, axis=0)

        Spectral columns with no valid aperture support are NaN.
    variance : ndarray, shape (n_spectral,)
        Propagated variance of *flux*::

            var_1d[col] = 1 / sum(P^2 / V, axis=0)

        This is the Cramér-Rao lower bound for the unbiased linear estimator.
        Columns with no valid support are NaN.
    profile : ndarray, shape (n_ap_spatial, n_spectral)
        Spatial-profile array used as weights.  If *normalize_profile* was
        ``True``, the profile columns sum to 1.0 (modulo NaN exclusion).
    aperture : :class:`WeightedExtractionDefinition`
        The extraction definition used for this order.
    method : str
        Always ``"horne_weighted"`` for this module.
    n_pixels_used : int
        Number of spatial pixels inside the aperture that have at least one
        finite value across spectral columns.
    weights : ndarray or None, shape (n_ap_spatial, n_spectral)
        The 2-D weight array ``P^2 / V`` used in the extraction, or ``None``
        if weights were not stored.  Pixels excluded by the bad-pixel mask
        have weight zero.
    n_rejected_pixels : int
        Number of aperture pixels rejected by the iterative outlier-rejection
        loop.  Zero if *reject_outliers* was ``False`` in the extraction
        definition or no rejections occurred.
    final_mask : ndarray of bool or None, shape (n_ap_spatial, n_spectral)
        Boolean array marking pixels rejected by the iterative rejection loop
        (``True`` = rejected).  ``None`` when *reject_outliers* is ``False``
        or the aperture is empty.  Does not include pixels that were already
        invalid before the rejection loop (non-finite data / variance, user
        mask).
    n_iterations_used : int
        Number of rejection iterations actually performed.  Zero if
        *reject_outliers* was ``False``.  May be less than *max_iterations*
        if the loop stopped early due to no new rejections.
    profile_reestimated : bool
        ``True`` if the spatial profile was actually re-estimated from the
        surviving pixels during at least one rejection iteration.  Always
        ``False`` when *reject_outliers* is ``False`` or *reestimate_profile*
        is ``False``, and also ``False`` if no pixels were rejected (so
        re-estimation never triggered).  Always ``False`` when
        *profile_source* is ``"external"`` (the external profile is kept
        fixed regardless of rejections).
    initial_profile : ndarray or None, shape (n_ap_spatial, n_spectral)
        The spatial profile computed from the original data before any
        rejection-driven re-estimation.  Only populated (non-``None``) when
        *profile_reestimated* is ``True``.  ``None`` otherwise.  Provided so
        callers can compare the initial and final profiles when re-estimation
        is active.
    profile_source_used : str
        The value of *profile_source* that was actually used for this order's
        extraction.  One of ``"empirical"``, ``"smoothed_empirical"``, or
        ``"external"``.  Useful when inspecting results from a batch of
        extractions with different definitions.
    profile_smoothed : bool
        ``True`` if Gaussian smoothing was applied to the profile
        (i.e. *profile_source* was ``"smoothed_empirical"`` and
        *profile_smooth_sigma* > 0).  ``False`` for all other profile
        sources or when *profile_smooth_sigma* == 0.

    Notes
    -----
    Unlike :class:`~pyspextool.instruments.ishell.optimal_extraction.OptimalExtractedOrderSpectrum`,
    the *variance* field is always present (never ``None``) because this
    module requires a variance source.  If no explicit variance source was
    provided, unit variance was used and *variance* reflects that choice.
    """

    order: int
    wavelength_um: npt.NDArray  # shape (n_spectral,)
    flux: npt.NDArray  # shape (n_spectral,)
    variance: npt.NDArray  # shape (n_spectral,)
    profile: npt.NDArray  # shape (n_ap_spatial, n_spectral)
    aperture: WeightedExtractionDefinition
    method: str
    n_pixels_used: int
    weights: Optional[npt.NDArray] = None  # shape (n_ap_spatial, n_spectral)
    n_rejected_pixels: int = 0
    final_mask: Optional[npt.NDArray] = None  # shape (n_ap_spatial, n_spectral)
    n_iterations_used: int = 0
    profile_reestimated: bool = False
    initial_profile: Optional[npt.NDArray] = None  # shape (n_ap_spatial, n_spectral)
    profile_source_used: str = "empirical"
    profile_smoothed: bool = False

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels in the extracted spectrum."""
        return len(self.wavelength_um)


# ---------------------------------------------------------------------------
# Full-set container
# ---------------------------------------------------------------------------


@dataclass
class WeightedExtractedSpectrumSet:
    """Collection of proto-Horne weighted 1-D spectra for all orders.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    spectra : list of :class:`WeightedExtractedOrderSpectrum`
        Per-order extracted spectra, in the same order as the input
        :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`.
    """

    mode: str
    spectra: list[WeightedExtractedOrderSpectrum] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    @property
    def orders(self) -> list[int]:
        """List of order numbers in the set (in storage order)."""
        return [sp.order for sp in self.spectra]

    @property
    def n_orders(self) -> int:
        """Number of orders in the set."""
        return len(self.spectra)

    def get_order(self, order_number: int) -> WeightedExtractedOrderSpectrum:
        """Return the :class:`WeightedExtractedOrderSpectrum` for a given order.

        Parameters
        ----------
        order_number : int
            Echelle order number to look up.

        Returns
        -------
        :class:`WeightedExtractedOrderSpectrum`

        Raises
        ------
        KeyError
            If *order_number* is not present in the set.
        """
        for sp in self.spectra:
            if sp.order == order_number:
                return sp
        raise KeyError(
            f"Order {order_number} not found in WeightedExtractedSpectrumSet "
            f"for mode {self.mode!r}.  Available orders: {self.orders}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_aperture_pixels(flux_ap: npt.NDArray) -> int:
    """Return the number of spatial rows in the aperture that are not fully NaN.

    Parameters
    ----------
    flux_ap : ndarray, shape (n_ap_spatial, n_spectral)
        Flux sub-image restricted to aperture rows.

    Returns
    -------
    int
        Number of rows that have at least one finite value.
    """
    any_finite = np.any(np.isfinite(flux_ap), axis=1)
    return int(np.sum(any_finite))


def _estimate_profile(
    flux_ap: npt.NDArray,
    profile_mode: str,
    normalize_profile: bool,
) -> npt.NDArray:
    """Estimate the spatial profile from the aperture sub-image.

    Parameters
    ----------
    flux_ap : ndarray, shape (n_ap_spatial, n_spectral)
        Background-subtracted (if applicable) aperture flux sub-image.
    profile_mode : str
        One of ``"global_median"`` or ``"columnwise"``.
    normalize_profile : bool
        If ``True``, normalize the profile so its spatial sum equals 1.0
        per column.

    Returns
    -------
    ndarray, shape (n_ap_spatial, n_spectral)
        Profile array.  Columns with no valid (finite) pixels are NaN.

    Notes
    -----
    ``"global_median"`` computes ``nanmedian(flux_ap, axis=1)``, giving a
    1-D profile vector that is broadcast across all spectral columns.
    Negative values are clipped to zero.

    ``"columnwise"`` uses each column of *flux_ap* directly as the profile
    for that column.  Negative values are likewise clipped to zero.

    When *normalize_profile* is ``True``, each column is divided by its
    spatial sum.  Columns where the sum is zero or NaN are set to NaN.
    """
    n_ap, n_spectral = flux_ap.shape

    if profile_mode == "global_median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profile_1d = np.nanmedian(flux_ap, axis=1)  # (n_ap,)
        finite_1d = np.isfinite(profile_1d)
        profile_1d = np.where(finite_1d, np.maximum(profile_1d, 0.0), np.nan)
        profile = np.broadcast_to(
            profile_1d[:, np.newaxis], (n_ap, n_spectral)
        ).copy()
    else:
        # profile_mode == "columnwise"
        profile = flux_ap.copy()
        finite_mask = np.isfinite(profile)
        profile = np.where(finite_mask, np.maximum(profile, 0.0), np.nan)

    if normalize_profile:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_sums = np.nansum(profile, axis=0)  # (n_spectral,)
        zero_or_nan = (col_sums == 0.0) | (~np.isfinite(col_sums))
        safe_sums = np.where(zero_or_nan, 1.0, col_sums)
        profile = profile / safe_sums[np.newaxis, :]
        profile[:, zero_or_nan] = np.nan

    return profile


def _smooth_profile(
    profile: npt.NDArray,
    sigma: float,
    normalize_profile: bool,
) -> npt.NDArray:
    """Smooth the spatial profile with a Gaussian kernel along the spatial axis.

    Smoothing is applied along **axis 0** (the spatial axis) using
    ``scipy.ndimage.gaussian_filter1d`` with ``mode='reflect'``.  NaN values
    are filled with zero before filtering and restored afterward; this means
    that fully-NaN spectral columns remain NaN after smoothing.

    Parameters
    ----------
    profile : ndarray, shape (n_ap_spatial, n_spectral)
        Estimated (possibly unnormalized) profile to smooth.  May contain
        NaN where the aperture is empty.
    sigma : float
        Gaussian kernel standard deviation in spatial pixels.  When <= 0,
        smoothing is skipped (only normalization is applied).
    normalize_profile : bool
        If ``True``, renormalize the smoothed profile so that each column
        sums to 1.0.  Columns with zero or NaN sum are set to NaN.

    Returns
    -------
    ndarray, shape (n_ap_spatial, n_spectral)
        Smoothed (and optionally renormalized) profile.  Negative values
        introduced by the Gaussian kernel are clipped to zero.

    Notes
    -----
    Smoothing is performed along the **spatial axis only** (axis 0).  This
    choice preserves spectral information: each wavelength column is smoothed
    independently, so spectral features are not mixed across columns.
    """
    nan_mask = ~np.isfinite(profile)
    if sigma > 0.0:
        # Replace NaN with 0 before filtering to avoid NaN propagation.
        filled = np.where(nan_mask, 0.0, profile)
        smoothed = gaussian_filter1d(filled, sigma=sigma, axis=0, mode="reflect")
        # Restore NaN where the original was NaN.
        smoothed[nan_mask] = np.nan
    else:
        smoothed = profile.copy()

    # Clip any negatives introduced by the Gaussian kernel at boundaries.
    smoothed = np.where(np.isfinite(smoothed), np.maximum(smoothed, 0.0), np.nan)

    if normalize_profile:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_sums = np.nansum(smoothed, axis=0)  # (n_spectral,)
        zero_or_nan = (col_sums == 0.0) | (~np.isfinite(col_sums))
        safe_sums = np.where(zero_or_nan, 1.0, col_sums)
        smoothed = smoothed / safe_sums[np.newaxis, :]
        smoothed[:, zero_or_nan] = np.nan

    return smoothed


def _prepare_external_profile(
    ext_profile: npt.NDArray,
    n_ap: int,
    n_spectral: int,
    normalize_profile: bool,
) -> npt.NDArray:
    """Validate, broadcast, and normalize a caller-supplied external profile.

    Parameters
    ----------
    ext_profile : ndarray
        Caller-supplied profile.  Accepted shapes:

        - ``(n_ap, n_spectral)`` — full 2-D profile.
        - ``(n_ap, 1)`` — same profile for all columns (broadcast).
        - ``(n_ap,)`` — 1-D profile, same for all columns.
    n_ap : int
        Expected number of aperture spatial pixels (first dimension).
    n_spectral : int
        Number of spectral columns (second dimension after broadcast).
    normalize_profile : bool
        If ``True``, normalize each column to sum to 1.0.  Columns with
        zero or NaN sum are set to NaN.

    Returns
    -------
    ndarray, shape (n_ap, n_spectral)
        Validated, broadcast, and optionally normalized profile.  Negative
        values are clipped to zero.

    Raises
    ------
    ValueError
        If *ext_profile* has an incompatible shape.
    """
    p = np.asarray(ext_profile, dtype=float)

    # Handle 1-D input: (n_ap,) → (n_ap, 1) for broadcasting.
    if p.ndim == 1:
        if p.shape[0] != n_ap:
            raise ValueError(
                f"external_profile 1-D shape {p.shape} is incompatible with "
                f"aperture size n_ap={n_ap}; expected ({n_ap},)."
            )
        p = p[:, np.newaxis]  # (n_ap, 1)

    if p.ndim != 2:
        raise ValueError(
            f"external_profile must be 1-D or 2-D; got shape {p.shape}."
        )
    if p.shape[0] != n_ap:
        raise ValueError(
            f"external_profile first dimension {p.shape[0]} does not match "
            f"aperture size n_ap={n_ap}."
        )
    if p.shape[1] not in (1, n_spectral):
        raise ValueError(
            f"external_profile second dimension {p.shape[1]} is incompatible "
            f"with n_spectral={n_spectral}; expected 1 or {n_spectral}."
        )

    # Broadcast to full (n_ap, n_spectral) shape and take a writable copy.
    profile = np.broadcast_to(p, (n_ap, n_spectral)).copy()

    # Clip negatives to zero; preserve NaN.
    finite_mask = np.isfinite(profile)
    profile = np.where(finite_mask, np.maximum(profile, 0.0), np.nan)

    if normalize_profile:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_sums = np.nansum(profile, axis=0)  # (n_spectral,)
        zero_or_nan = (col_sums == 0.0) | (~np.isfinite(col_sums))
        safe_sums = np.where(zero_or_nan, 1.0, col_sums)
        profile = profile / safe_sums[np.newaxis, :]
        profile[:, zero_or_nan] = np.nan

    return profile


def _horne_weighted_extract(
    profile: npt.NDArray,
    flux_ap: npt.NDArray,
    var_ap: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Apply proto-Horne inverse-variance weighted extraction.

    Computes::

        weights[i, col] = P[i, col]^2 / V[i, col]

        flux_1d[col] = sum_i( P[i,col] * F[i,col] / V[i,col] )
                     / sum_i( P[i,col]^2  / V[i,col] )

        var_1d[col]  = 1 / sum_i( P[i,col]^2 / V[i,col] )

    Only pixels that satisfy all of the following conditions contribute:

    - ``P[i, col]`` is finite and ≥ 0.
    - ``F[i, col]`` is finite.
    - ``V[i, col]`` is finite and > 0.

    Parameters
    ----------
    profile : ndarray, shape (n_ap_spatial, n_spectral)
        Spatial profile array (optionally normalized).  Non-negative values
        only; any negative values are treated as invalid (zero weight).
    flux_ap : ndarray, shape (n_ap_spatial, n_spectral)
        Background-subtracted aperture flux sub-image.
    var_ap : ndarray, shape (n_ap_spatial, n_spectral)
        Per-pixel variance within the aperture.  Pixels with variance ≤ 0
        are excluded (zero weight).

    Returns
    -------
    flux_1d : ndarray, shape (n_spectral,)
        Weighted 1-D spectrum.  Columns with no valid support are NaN.
    var_1d : ndarray, shape (n_spectral,)
        Propagated 1-D variance.  Columns with no valid support are NaN.
        Always ≥ 0 where finite.
    weights : ndarray, shape (n_ap_spatial, n_spectral)
        The 2-D weight array ``P^2 / V``.  Invalid pixels have weight 0.

    Notes
    -----
    Pixels are zeroed (rather than NaN) before summing so that
    :func:`numpy.sum` can be used directly without NaN propagation.
    This avoids masking valid pixels in the same column.
    """
    # Build pixel-validity mask: finite data, finite positive variance,
    # finite non-negative profile.
    valid = (
        np.isfinite(profile)
        & np.isfinite(flux_ap)
        & np.isfinite(var_ap)
        & (var_ap > 0.0)
        & (profile >= 0.0)
    )

    # Weight array: P^2 / V (zero for invalid pixels).
    weights = np.where(valid, profile ** 2 / var_ap, 0.0)

    # Numerator: P * F / V (zero for invalid pixels).
    numerator_vals = np.where(valid, profile * flux_ap / var_ap, 0.0)

    sum_weights = np.sum(weights, axis=0)       # (n_spectral,)
    sum_numerator = np.sum(numerator_vals, axis=0)  # (n_spectral,)

    # Columns with valid support (at least one contributing pixel).
    valid_cols = np.isfinite(sum_weights) & (sum_weights > 0.0)

    flux_1d = np.full(sum_weights.shape, np.nan)
    flux_1d[valid_cols] = sum_numerator[valid_cols] / sum_weights[valid_cols]

    var_1d = np.full(sum_weights.shape, np.nan)
    var_1d[valid_cols] = 1.0 / sum_weights[valid_cols]

    return flux_1d, var_1d, weights


def _iterative_outlier_rejection(
    profile: npt.NDArray,
    flux_ap: npt.NDArray,
    var_ap: npt.NDArray,
    *,
    sigma_clip: float,
    max_iterations: int,
    min_valid_pixels: int,
    reestimate_profile: bool = False,
    profile_mode: str = "global_median",
    normalize_profile: bool = True,
    profile_source: str = "empirical",
    profile_smooth_sigma: float = 0.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, int, int, npt.NDArray, bool]:
    """Apply iterative sigma-clipping outlier rejection within the aperture.

    For each iteration:

    1. Compute the weighted extracted flux for each spectral column using the
       current pixel mask.
    2. Build a model image: ``model = profile * flux_1d``.
    3. Compute residuals: ``resid = data - model``.
    4. Normalize residuals: ``norm_resid = resid / sqrt(variance)``.
    5. Mark pixels with ``|norm_resid| > sigma_clip`` as bad, subject to:

       - Only within aperture pixels that are not already bad.
       - Only where variance is finite and > 0.
       - Per-column: if rejecting all candidates would leave fewer than
         *min_valid_pixels* valid pixels, skip the entire column.

    6. If *reestimate_profile* is ``True`` **and** *profile_source* is not
       ``"external"``, re-estimate the spatial profile from the currently
       surviving (non-rejected) pixels after committing rejections.  For
       ``"smoothed_empirical"``, the re-estimated profile is additionally
       smoothed with the same *profile_smooth_sigma*.  When *profile_source*
       is ``"external"``, the profile is **always kept fixed** regardless of
       *reestimate_profile*.

    7. Stop early if no new pixels were rejected in the last iteration.

    Parameters
    ----------
    profile : ndarray, shape (n_ap, n_spectral)
        Spatial profile array (optionally normalized).  Used as the initial
        profile for the first iteration.  Updated after each rejection step
        when *reestimate_profile* is ``True`` and *profile_source* is not
        ``"external"``.
    flux_ap : ndarray, shape (n_ap, n_spectral)
        Aperture flux sub-image.  Pixels already bad (NaN) are treated as
        initially masked and are never re-introduced.
    var_ap : ndarray, shape (n_ap, n_spectral)
        Per-pixel variance within the aperture.  Must be positive where valid.
    sigma_clip : float
        Rejection threshold in units of normalized residuals.
    max_iterations : int
        Maximum number of iterations.  The loop may stop earlier.
    min_valid_pixels : int
        Minimum valid pixels that must remain per column after rejection.
    reestimate_profile : bool, optional
        If ``True``, re-estimate the spatial profile from the surviving pixels
        after each rejection step.  Ignored when *profile_source* is
        ``"external"`` (the external profile is always kept fixed).  Defaults
        to ``False`` (Stage 18 behavior).
    profile_mode : str, optional
        Profile estimation mode (``"global_median"`` or ``"columnwise"``).
        Only used when *reestimate_profile* is ``True`` and *profile_source*
        is not ``"external"``.
    normalize_profile : bool, optional
        Whether to normalize the re-estimated profile.  Only used when
        *reestimate_profile* is ``True`` and *profile_source* is not
        ``"external"``.
    profile_source : str, optional
        Profile source identifier (``"empirical"``, ``"smoothed_empirical"``,
        or ``"external"``).  When ``"external"``, *reestimate_profile* is
        ignored and the profile is never updated.  When
        ``"smoothed_empirical"``, re-estimation additionally applies Gaussian
        smoothing with *profile_smooth_sigma*.  Defaults to ``"empirical"``.
    profile_smooth_sigma : float, optional
        Gaussian smoothing sigma (in spatial pixels) applied during
        re-estimation when *profile_source* is ``"smoothed_empirical"``.
        Defaults to 0.0 (no smoothing).

    Returns
    -------
    flux_ap_out : ndarray, shape (n_ap, n_spectral)
        Updated aperture flux with newly rejected pixels replaced by NaN.
    var_ap_out : ndarray, shape (n_ap, n_spectral)
        Updated aperture variance with newly rejected pixels replaced by NaN.
    reject_mask : ndarray of bool, shape (n_ap, n_spectral)
        Boolean array where ``True`` indicates a pixel was rejected by this
        function.  Does not include pixels that were already NaN on input.
    n_rejected : int
        Total number of pixels rejected across all iterations.
    n_iterations_used : int
        Number of iterations performed (1 even if no rejections occurred).
    final_profile : ndarray, shape (n_ap, n_spectral)
        The spatial profile used at the end of the rejection loop.  Equal to
        the input *profile* if *reestimate_profile* is ``False``, if
        *profile_source* is ``"external"``, or if no rejections occurred.
        Otherwise the re-estimated (and optionally smoothed) profile.
    profile_was_reestimated : bool
        ``True`` if the profile was actually re-estimated at least once
        (i.e. *reestimate_profile* was ``True``, *profile_source* is not
        ``"external"``, and at least one rejection was committed).  Always
        ``False`` when *profile_source* is ``"external"``.

    Notes
    -----
    When *reestimate_profile* is ``False``, the spatial profile is **not**
    re-estimated after each rejection.  This is the Stage 18 scaffold
    behavior.  Setting *reestimate_profile* to ``True`` moves the scaffold
    one step closer to the Horne (1986) iterative scheme.

    When *profile_source* is ``"external"``, the caller-supplied profile is
    treated as fixed ground truth and is never updated inside the loop.
    This is the intended behavior: the external profile is assumed to come
    from a higher-S/N independent source.
    """
    n_ap, n_spectral = flux_ap.shape

    # Track which pixels were already invalid on entry (NaN/non-positive var).
    # These are never counted as new rejections.
    initially_bad = ~np.isfinite(flux_ap) | ~np.isfinite(var_ap) | (var_ap <= 0.0)

    # reject_mask accumulates pixels rejected by this function only.
    reject_mask = np.zeros((n_ap, n_spectral), dtype=bool)

    # Working copies — we NaN-out newly rejected pixels here.
    flux_work = np.where(initially_bad, np.nan, flux_ap)
    var_work = np.where(initially_bad, np.nan, var_ap)

    # Working profile — may be updated when reestimate_profile=True and
    # profile_source != "external".
    profile_work = profile.copy()
    profile_was_reestimated = False

    n_rejected = 0
    n_iterations_used = 0

    for iteration in range(max_iterations):
        n_iterations_used += 1

        # Extract current best-estimate 1-D spectrum.
        flux_1d, _, _ = _horne_weighted_extract(profile_work, flux_work, var_work)

        # Build model image: profile[i,col] * flux_1d[col].
        model = profile_work * flux_1d[np.newaxis, :]  # (n_ap, n_spectral)

        # Residuals (NaN where flux_work is NaN or flux_1d is NaN).
        resid = flux_work - model  # (n_ap, n_spectral)

        # Normalized residuals: only where variance is finite and positive.
        valid_var = np.isfinite(var_work) & (var_work > 0.0)
        safe_var = np.where(valid_var, var_work, 1.0)  # avoid divide-by-zero
        norm_resid = np.where(
            valid_var & np.isfinite(resid),
            resid / np.sqrt(safe_var),
            np.nan,
        )

        # Candidate new bad pixels: not already bad, finite norm_resid, exceeds threshold.
        currently_bad = initially_bad | reject_mask
        new_bad_candidate = (
            ~currently_bad
            & np.isfinite(norm_resid)
            & (np.abs(norm_resid) > sigma_clip)
        )

        if not np.any(new_bad_candidate):
            break  # Early stop: nothing to reject.

        # Apply min_valid_pixels constraint column by column.
        new_bad = np.zeros((n_ap, n_spectral), dtype=bool)
        for col in range(n_spectral):
            if not np.any(new_bad_candidate[:, col]):
                continue
            n_valid_before = int(np.sum(~currently_bad[:, col]))
            n_new = int(np.sum(new_bad_candidate[:, col]))
            if n_valid_before - n_new >= min_valid_pixels:
                new_bad[:, col] = new_bad_candidate[:, col]
            # else: no pixels rejected in this column this iteration
            # to preserve the min_valid_pixels requirement.

        if not np.any(new_bad):
            break  # All rejections blocked by min_valid_pixels.

        # Commit the new rejections.
        reject_mask |= new_bad
        flux_work = np.where(new_bad, np.nan, flux_work)
        var_work = np.where(new_bad, np.nan, var_work)
        n_rejected += int(np.sum(new_bad))

        # Re-estimate profile from surviving pixels (Stage 19 / Stage 20).
        # When profile_source="external", the caller-supplied profile is
        # kept fixed regardless of reestimate_profile.
        if reestimate_profile and profile_source != "external":
            profile_work = _estimate_profile(
                flux_work, profile_mode, normalize_profile
            )
            if profile_source == "smoothed_empirical":
                profile_work = _smooth_profile(
                    profile_work, profile_smooth_sigma, normalize_profile
                )
            profile_was_reestimated = True

    return (
        flux_work,
        var_work,
        reject_mask,
        n_rejected,
        n_iterations_used,
        profile_work,
        profile_was_reestimated,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_weighted_optimal(
    rectified_orders: RectifiedOrderSet,
    extraction_def: WeightedExtractionDefinition,
    *,
    variance_image: Optional[npt.NDArray] = None,
    variance_model: Optional[VarianceModelDefinition] = None,
    subtract_background: bool = True,
    mask: Optional[npt.NDArray] = None,
) -> WeightedExtractedSpectrumSet:
    """Extract 1-D spectra using proto-Horne inverse-variance weighted extraction.

    For each order in *rectified_orders*, selects aperture pixels, optionally
    subtracts a per-column background, resolves a variance image, estimates a
    provisional spatial profile, and computes an inverse-variance weighted 1-D
    extraction.

    Non-finite pixels (NaN or inf) in *data* or the variance image, pixels
    with non-positive variance, and any pixels flagged by *mask*, are excluded
    from both the flux sum and the denominator.  Spectral columns where every
    aperture pixel is excluded return NaN flux and NaN variance.

    Parameters
    ----------
    rectified_orders : :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
        Stage-7 output.  Must contain at least one order.
    extraction_def : :class:`WeightedExtractionDefinition`
        Aperture, background region, profile-mode, and optional embedded
        variance-model parameters.
    variance_image : ndarray or None, optional
        Per-pixel variance image with the same shape as each order's
        ``flux`` array (``n_spatial × n_spectral``).  Takes highest priority.
        If ``None`` (default), the next source in the priority chain is tried.
    variance_model : :class:`~pyspextool.instruments.ishell.variance_model.VarianceModelDefinition` or None, optional
        Stage-14 variance model definition.  If *variance_image* is ``None``
        and *variance_model* is provided, a variance image is built internally
        for each order using
        :func:`~pyspextool.instruments.ishell.variance_model.build_variance_image`.
        Ignored when *variance_image* is provided.
    subtract_background : bool, optional
        If ``True`` (default) **and** *extraction_def* has a background region
        defined, estimate a per-column background as the median of
        background-annulus pixels and subtract it before extraction.
    mask : ndarray of bool or None, optional
        Optional bad-pixel mask with the same shape as each order's ``flux``
        array.  Pixels where *mask* is ``True`` are treated as bad.  Combined
        with the internal mask derived from non-finite data and variance values.

    Returns
    -------
    :class:`WeightedExtractedSpectrumSet`
        One :class:`WeightedExtractedOrderSpectrum` per order.  Each spectrum's
        ``variance`` field is always present and reflects the inverse-variance
        propagation formula.

    Raises
    ------
    ValueError
        If *rectified_orders* is empty (``n_orders == 0``).
    ValueError
        If any order's ``flux`` field is not a 2-D array.
    ValueError
        If *variance_image* is provided but its shape does not match the
        order's flux shape.
    ValueError
        If *mask* is provided but its shape does not match the order's flux
        shape.

    Notes
    -----
    **Variance source priority**

    ``explicit variance_image > function's variance_model >
    definition's variance_model (when use_variance_model=True) >
    unit variance (fallback)``

    If no variance source is available from any of the above, **unit
    variance (V=1 everywhere) is used automatically**.  In that case the
    extracted flux is identical to the Stage-12 profile-weighted estimator,
    and the variance is ``1/sum(P^2)``.

    **Algorithm (per order)**

    1. Compute ``dist = |spatial_frac - center_frac|``.
    2. Select aperture pixels: ``dist <= radius_frac``.
    3. If background subtraction is enabled, select background pixels:
       ``background_inner < dist <= background_outer``.
       Subtract the per-column median from each aperture pixel.
    4. Resolve the variance image from the priority chain above.
    5. Apply the combined bad-pixel mask (non-finite data, non-finite or
       non-positive variance, user mask).
    6. Estimate the spatial profile from the background-subtracted aperture
       sub-image.
    7. Normalize the profile column-by-column.
    8. Compute the proto-Horne weighted extraction and 1/sum(P^2/V) variance.

    **Background variance**

    When background subtraction is active, the background variance per column
    is approximated as ``nanmedian(var_image[bg_mask], axis=0)`` and added to
    the aperture variance before weighting.  This is a first-order
    approximation (median ≠ mean for skewed distributions).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if rectified_orders.n_orders == 0:
        raise ValueError(
            "rectified_orders is empty (n_orders == 0); "
            "cannot extract spectra."
        )

    do_background = subtract_background and extraction_def.has_background

    # ------------------------------------------------------------------
    # Extract one spectrum per order
    # ------------------------------------------------------------------
    spectra: list[WeightedExtractedOrderSpectrum] = []

    for ro in rectified_orders.rectified_orders:
        flux_2d = np.asarray(ro.flux, dtype=float)

        if flux_2d.ndim != 2:
            raise ValueError(
                f"Order {ro.order}: flux must be a 2-D array; "
                f"got shape {flux_2d.shape}."
            )

        n_spectral = flux_2d.shape[1]

        # -- Variance image: resolve from priority chain --
        if variance_image is not None:
            var_2d = np.asarray(variance_image, dtype=float)
            if var_2d.shape != flux_2d.shape:
                raise ValueError(
                    f"Order {ro.order}: variance_image shape {var_2d.shape} "
                    f"does not match flux shape {flux_2d.shape}."
                )
        elif variance_model is not None:
            var_2d = build_variance_image(flux_2d, variance_model).variance_image
        elif extraction_def.use_variance_model and extraction_def.variance_model is not None:
            var_2d = build_variance_image(
                flux_2d, extraction_def.variance_model
            ).variance_image
        else:
            # Fallback: unit variance.  Documented behaviour; not an error.
            var_2d = build_unit_variance_image(flux_2d).variance_image

        # -- Build combined bad-pixel mask --
        # Exclude: non-finite data, non-finite variance, user mask.
        bad = ~np.isfinite(flux_2d) | ~np.isfinite(var_2d)
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != flux_2d.shape:
                raise ValueError(
                    f"Order {ro.order}: mask shape {mask_arr.shape} "
                    f"does not match flux shape {flux_2d.shape}."
                )
            bad |= mask_arr
        if np.any(bad):
            flux_2d = flux_2d.copy()
            flux_2d[bad] = np.nan
            var_2d = var_2d.copy()
            var_2d[bad] = np.nan

        spatial_frac = np.asarray(ro.spatial_frac, dtype=float)
        dist = np.abs(spatial_frac - extraction_def.center_frac)

        # -- Aperture mask --
        ap_mask = dist <= extraction_def.radius_frac
        flux_ap = flux_2d[ap_mask, :]   # (n_ap, n_spectral)
        var_ap = var_2d[ap_mask, :]     # (n_ap, n_spectral)
        n_ap = flux_ap.shape[0]

        if n_ap == 0:
            n_spectral_out = n_spectral
            spectra.append(
                WeightedExtractedOrderSpectrum(
                    order=ro.order,
                    wavelength_um=ro.wavelength_um.copy(),
                    flux=np.full(n_spectral_out, np.nan),
                    variance=np.full(n_spectral_out, np.nan),
                    profile=np.full((0, n_spectral_out), np.nan),
                    aperture=extraction_def,
                    method="horne_weighted",
                    n_pixels_used=0,
                    weights=np.full((0, n_spectral_out), 0.0),
                )
            )
            continue

        # -- Background estimation --
        var_bg_col: Optional[npt.NDArray] = None
        if do_background:
            bg_mask = (dist > extraction_def.background_inner) & (
                dist <= extraction_def.background_outer
            )
            flux_bg = flux_2d[bg_mask, :]  # (n_bg, n_spectral)

            if flux_bg.shape[0] == 0:
                bg_1d = np.full(n_spectral, np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bg_1d = np.nanmedian(flux_bg, axis=0)  # (n_spectral,)

            flux_ap = flux_ap - bg_1d[np.newaxis, :]

            # Background variance: approximate as median of variance in the
            # background annulus (first-order approximation).
            var_bg_pixels = var_2d[bg_mask, :]  # (n_bg, n_spectral)
            if var_bg_pixels.shape[0] == 0:
                var_bg_col = np.zeros(n_spectral)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    var_bg_col = np.nanmedian(var_bg_pixels, axis=0)

        # Add background variance to aperture variance when applicable.
        if var_bg_col is not None:
            var_ap = var_ap + var_bg_col[np.newaxis, :]

        # -- Profile estimation (Stage 20: dispatch on profile_source) --
        p_source = extraction_def.profile_source
        p_sigma = extraction_def.profile_smooth_sigma
        profile_smoothed_flag = False

        if p_source == "external":
            # Use caller-supplied external profile; normalize if requested.
            # If the profile covers the full order spatial extent (n_spatial),
            # slice it to the aperture using ap_mask so that callers can
            # supply a full-order template (e.g. from profile_templates.py)
            # without pre-slicing.
            ext_p = np.asarray(extraction_def.external_profile)
            n_spatial_full = flux_2d.shape[0]
            if (
                ext_p.ndim >= 1
                and ext_p.shape[0] == n_spatial_full
                and n_spatial_full != n_ap
            ):
                ext_p = ext_p[ap_mask]
            profile = _prepare_external_profile(
                ext_p,
                n_ap,
                n_spectral,
                extraction_def.normalize_profile,
            )
        elif p_source == "smoothed_empirical":
            # Estimate empirical profile (without normalization), then smooth
            # (which also handles normalization).
            raw_profile = _estimate_profile(
                flux_ap,
                profile_mode=extraction_def.profile_mode,
                normalize_profile=False,
            )
            profile = _smooth_profile(
                raw_profile, p_sigma, extraction_def.normalize_profile
            )
            profile_smoothed_flag = p_sigma > 0.0
        else:
            # "empirical" (default): existing behavior.
            profile = _estimate_profile(
                flux_ap,
                profile_mode=extraction_def.profile_mode,
                normalize_profile=extraction_def.normalize_profile,
            )

        # -- Iterative outlier rejection (Stage 18/19/20, optional) --
        n_rejected = 0
        n_iter_used = 0
        final_mask_ap: Optional[npt.NDArray] = None
        profile_reestimated = False
        initial_profile_ap: Optional[npt.NDArray] = None
        if extraction_def.reject_outliers:
            if extraction_def.reestimate_profile:
                initial_profile_ap = profile.copy()
            (
                flux_ap,
                var_ap,
                rejection_mask,
                n_rejected,
                n_iter_used,
                profile,
                profile_reestimated,
            ) = _iterative_outlier_rejection(
                profile,
                flux_ap,
                var_ap,
                sigma_clip=extraction_def.sigma_clip,
                max_iterations=extraction_def.max_iterations,
                min_valid_pixels=extraction_def.min_valid_pixels,
                reestimate_profile=extraction_def.reestimate_profile,
                profile_mode=extraction_def.profile_mode,
                normalize_profile=extraction_def.normalize_profile,
                profile_source=p_source,
                profile_smooth_sigma=p_sigma,
            )
            final_mask_ap = rejection_mask
            # Only keep initial_profile when re-estimation actually happened.
            if not profile_reestimated:
                initial_profile_ap = None

        # -- Proto-Horne weighted extraction --
        n_pixels_used = _count_aperture_pixels(flux_ap)
        flux_1d, var_1d, weights_2d = _horne_weighted_extract(
            profile, flux_ap, var_ap
        )

        spectra.append(
            WeightedExtractedOrderSpectrum(
                order=ro.order,
                wavelength_um=ro.wavelength_um.copy(),
                flux=flux_1d,
                variance=var_1d,
                profile=profile,
                aperture=extraction_def,
                method="horne_weighted",
                n_pixels_used=n_pixels_used,
                weights=weights_2d,
                n_rejected_pixels=n_rejected,
                final_mask=final_mask_ap,
                n_iterations_used=n_iter_used,
                profile_reestimated=profile_reestimated,
                initial_profile=initial_profile_ap,
                profile_source_used=p_source,
                profile_smoothed=profile_smoothed_flag,
            )
        )

    return WeightedExtractedSpectrumSet(
        mode=rectified_orders.mode,
        spectra=spectra,
    )
