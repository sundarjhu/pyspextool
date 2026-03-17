"""
Provisional optimal-extraction scaffold for iSHELL 2DXD reduction.

This module implements the **twelfth stage** of the iSHELL 2DXD reduction
scaffold: a first provisional optimal-extraction pass based on a simple
spatial-profile weighting scheme.

.. note::
    This is a **scaffold implementation**.  It is intentionally simple and
    does **not** represent final science-quality iSHELL or Spextool optimal
    extraction.  See ``docs/ishell_optimal_extraction.md`` for a full list
    of limitations and for a description of what real optimal extraction
    would require.

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
   * - **12**
     - **``optimal_extraction.py``**
     - **Provisional optimal extraction (this module)**

What this module does
---------------------
* Accepts a
  :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
  (the Stage-7 output) and an :class:`OptimalExtractionDefinition`.

* For each order:

  1. Selects aperture pixels defined by *center_frac* ± *radius_frac* in
     fractional slit coordinates ``[0, 1]``.

  2. Optionally estimates a per-column background as the median of pixels
     in a background annulus, and subtracts it from the aperture pixels
     before profile estimation or extraction.

  3. Estimates a provisional spatial profile from the aperture sub-image
     using the method specified by *profile_mode*:

     - ``"global_median"``: the profile is the per-row median of the
       aperture sub-image across all spectral columns.  This produces a
       1-D profile vector that is broadcast across all columns.  This is
       robust to spectral features but ignores spatial-profile variation
       along the dispersion axis.

     - ``"columnwise"``: the profile is estimated independently for each
       spectral column from the column's aperture pixels.  This tracks
       spatial-profile variation with wavelength but is noisier.

  4. If *normalize_profile* is ``True``, the profile is normalized so that
     its spatial sum equals 1.0 per column.  Columns with no valid profile
     support are set to NaN.

  5. Computes a weighted 1-D extraction using::

         flux_1d[col] = sum(P[:, col] * F[:, col]) / sum(P[:, col] ** 2)

     where ``P`` is the (optionally normalized) profile array with shape
     ``(n_ap_spatial, n_spectral)`` and ``F`` is the (optionally
     background-subtracted) aperture flux sub-image with the same shape.

     Columns where the profile denominator is zero or NaN are set to NaN.

  6. *(Optional)* If a *variance_image* is provided, propagates variance
     through background subtraction and profile-weighted extraction::

         var_flux[col] = sum(P[:, col]**2 * (var_pixel + var_bg))

     where *var_bg* is the per-column background variance approximated as
     the median of the variance image within the background annulus.

     .. note::
         Background variance is approximated using the median of the
         variance image within the background annulus.  This is a
         first-order approximation and does not account for correlated
         noise or detailed detector characteristics.

* Returns an :class:`OptimalExtractedSpectrumSet` collecting one
  :class:`OptimalExtractedOrderSpectrum` per echelle order.

Weighting formula
-----------------
The weighted extraction formula is::

    flux_1d[col] = sum(P * F, axis=0) / sum(P^2, axis=0)

This is equivalent to a least-squares estimate of the flux scalar
``a`` for the linear model ``F[:, col] = a * P[:, col]``.  It is
**not** Horne (1986) optimal extraction because:

- no per-pixel variance weighting (all pixels treated equally),
- no iterative bad-pixel rejection,
- no PSF modelling.

For the scaffold it is an acceptable first approximation.  See
``docs/ishell_optimal_extraction.md`` for details.

What this module does NOT do (by design)
-----------------------------------------
* **No Horne-style optimal extraction** – variance-weighted extraction
  belongs to a later stage that requires a reliable variance model.

* **No iterative cosmic-ray rejection** – profile-based outlier rejection
  is not implemented.

* **No profile fitting with a PSF model** – the profile is a simple
  empirical estimate from the data itself.

* **No aperture finding / centroiding** – the aperture center and radius
  must be supplied explicitly via :class:`OptimalExtractionDefinition`.

* **No sky modelling beyond simple median subtraction** – background is
  estimated as the median of background-annulus pixels per column.

* **No telluric correction** – that belongs to a later stage.

* **No order merging** – each order is extracted independently.

NaN handling
------------
:func:`numpy.nanmedian`, :func:`numpy.nansum`, and :func:`numpy.nanmean`
are used throughout so that NaN values (e.g. out-of-bounds rectification
pixels) are excluded automatically.  Spectral columns with no valid
aperture pixels or no valid profile support return NaN in the output flux.

Public API
----------
- :func:`extract_optimal` – main entry point.
- :class:`OptimalExtractionDefinition` – extraction aperture and
  profile-mode definition.
- :class:`OptimalExtractedOrderSpectrum` – per-order 1-D spectrum
  container.
- :class:`OptimalExtractedSpectrumSet` – full result container.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from .rectified_orders import RectifiedOrderSet
from .variance_model import VarianceModelDefinition, build_variance_image

__all__ = [
    "OptimalExtractionDefinition",
    "OptimalExtractedOrderSpectrum",
    "OptimalExtractedSpectrumSet",
    "extract_optimal",
]

# Supported profile estimation modes.
_VALID_PROFILE_MODES = {"global_median", "columnwise"}


# ---------------------------------------------------------------------------
# Extraction definition
# ---------------------------------------------------------------------------


@dataclass
class OptimalExtractionDefinition:
    """Definition of the aperture and profile-estimation parameters.

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
        How to estimate the spatial profile inside the aperture.  One of:

        ``"global_median"`` (default)
            The profile is the per-row median of the aperture sub-image
            across all spectral columns.  This 1-D profile is then
            broadcast across wavelength.  Robust to spectral features but
            ignores wavelength-dependent profile variation.

        ``"columnwise"``
            The profile is estimated independently for each spectral
            column from the column's aperture pixels.  Tracks profile
            variation with wavelength but is noisier.
    normalize_profile : bool, optional
        If ``True`` (default), normalize the profile so that its spatial
        sum equals 1.0 per column before computing the weighted extraction.
        If ``False``, the raw (un-normalized) profile is used; in this case
        the extracted flux is **not** in physical units relative to the
        total aperture flux.

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
        # Background region consistency checks.
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

    @property
    def has_background(self) -> bool:
        """True if a background region is defined."""
        return self.background_inner is not None


# ---------------------------------------------------------------------------
# Per-order 1-D spectrum
# ---------------------------------------------------------------------------


@dataclass
class OptimalExtractedOrderSpectrum:
    """Provisional optimally-extracted 1-D spectrum for one echelle order.

    Parameters
    ----------
    order : int
        Echelle order number.
    wavelength_um : ndarray, shape (n_spectral,)
        Wavelength axis in µm.
    flux : ndarray, shape (n_spectral,)
        Weighted 1-D flux extracted using the spatial profile as weights.
        Spectral columns with no valid aperture pixels or no valid profile
        support are NaN.
    profile : ndarray, shape (n_ap_spatial, n_spectral)
        Spatial-profile array used as weights.  Shape is
        ``(n_ap_spatial, n_spectral)`` where *n_ap_spatial* is the number
        of spatial pixels inside the aperture for this order.  If
        *normalize_profile* was ``True``, the profile columns sum to 1.0
        (modulo NaN exclusion).
    aperture : :class:`OptimalExtractionDefinition`
        The extraction definition used for this order.
    method : str
        Always ``"optimal_weighted"`` for this module.
    n_pixels_used : int
        Number of spatial pixels inside the aperture that have at least one
        finite value across spectral columns.
    variance : ndarray or None, shape (n_spectral,)
        Propagated variance of *flux*, or ``None`` if no *variance_image*
        was supplied to :func:`extract_optimal`.  When provided, this is
        computed as ``nansum(weights**2 * variance, axis=0)`` over the
        aperture pixels (after adding the per-column background variance
        contribution when background subtraction is active).

    Notes
    -----
    Background variance is approximated using the median of the variance
    image within the background annulus; this is a first-order
    approximation.
    """

    order: int
    wavelength_um: npt.NDArray  # shape (n_spectral,)
    flux: npt.NDArray  # shape (n_spectral,)
    profile: npt.NDArray  # shape (n_ap_spatial, n_spectral)
    aperture: OptimalExtractionDefinition
    method: str
    n_pixels_used: int
    variance: Optional[npt.NDArray] = None  # None when no variance_image supplied

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels in the extracted spectrum."""
        return len(self.wavelength_um)


# ---------------------------------------------------------------------------
# Full-set container
# ---------------------------------------------------------------------------


@dataclass
class OptimalExtractedSpectrumSet:
    """Collection of optimally-extracted 1-D spectra for all orders.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    spectra : list of :class:`OptimalExtractedOrderSpectrum`
        Per-order extracted spectra, in the same order as the input
        :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`.
    """

    mode: str
    spectra: list[OptimalExtractedOrderSpectrum] = field(default_factory=list)

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

    def get_order(self, order_number: int) -> OptimalExtractedOrderSpectrum:
        """Return the :class:`OptimalExtractedOrderSpectrum` for a given order.

        Parameters
        ----------
        order_number : int
            Echelle order number to look up.

        Returns
        -------
        :class:`OptimalExtractedOrderSpectrum`

        Raises
        ------
        KeyError
            If *order_number* is not present in the set.
        """
        for sp in self.spectra:
            if sp.order == order_number:
                return sp
        raise KeyError(
            f"Order {order_number} not found in OptimalExtractedSpectrumSet "
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
    ``"global_median"`` computes ``nanmedian(flux_ap, axis=1)``, which
    gives a 1-D profile vector ``(n_ap_spatial,)`` that is then broadcast
    across all spectral columns.  Negative values are clipped to zero
    before normalization; this prevents negative profile weights from
    inverting the extracted flux sign.

    ``"columnwise"`` uses each column of *flux_ap* directly as the profile
    for that column.  Negative values are likewise clipped to zero.

    In both cases, if *normalize_profile* is ``True``, each column of the
    resulting profile is divided by its spatial sum (computed with
    ``nansum``).  Columns where the sum is zero (or the sum is NaN) are
    set to NaN.
    """
    n_ap, n_spectral = flux_ap.shape

    if profile_mode == "global_median":
        # Median across spectral axis → shape (n_ap,)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profile_1d = np.nanmedian(flux_ap, axis=1)  # (n_ap,)
        # Clip negatives: negative profile weights can invert extracted flux.
        finite_1d = np.isfinite(profile_1d)
        profile_1d = np.where(finite_1d, np.maximum(profile_1d, 0.0), np.nan)
        # Broadcast to (n_ap, n_spectral).
        profile = np.broadcast_to(profile_1d[:, np.newaxis], (n_ap, n_spectral)).copy()
    else:
        # profile_mode == "columnwise"
        # Use the flux values themselves as the per-column profile.
        profile = flux_ap.copy()
        # Clip negatives.
        finite_mask = np.isfinite(profile)
        profile = np.where(finite_mask, np.maximum(profile, 0.0), np.nan)

    if normalize_profile:
        # Normalize each column so its spatial sum is 1.0.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_sums = np.nansum(profile, axis=0)  # (n_spectral,)
        # Columns where the sum is zero or NaN should yield NaN profile.
        zero_or_nan = (col_sums == 0.0) | (~np.isfinite(col_sums))
        safe_sums = np.where(zero_or_nan, 1.0, col_sums)  # avoid div-by-zero
        profile = profile / safe_sums[np.newaxis, :]
        # Mark columns with no valid support as NaN.
        profile[:, zero_or_nan] = np.nan

    return profile


def _weighted_extract(
    profile: npt.NDArray,
    flux_ap: npt.NDArray,
) -> npt.NDArray:
    """Apply profile-weighted extraction along the spatial axis.

    Computes::

        flux_1d[col] = sum(P[:, col] * F[:, col]) / sum(P[:, col] ** 2)

    where ``P`` is the profile and ``F`` is the aperture flux.

    Parameters
    ----------
    profile : ndarray, shape (n_ap_spatial, n_spectral)
        Spatial profile array (optionally normalized).
    flux_ap : ndarray, shape (n_ap_spatial, n_spectral)
        Background-subtracted aperture flux sub-image.

    Returns
    -------
    ndarray, shape (n_spectral,)
        Weighted 1-D spectrum.  Columns where the denominator is zero,
        NaN, or where there are no valid pixels are set to NaN.

    Notes
    -----
    Using NaN-safe reductions ensures that individual NaN pixels (e.g.
    from out-of-bounds rectification) are silently excluded.  Spectral
    columns with no valid pixels in either the profile or the flux
    become NaN in the output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        numerator = np.nansum(profile * flux_ap, axis=0)  # (n_spectral,)
        denominator = np.nansum(profile**2, axis=0)      # (n_spectral,)

    flux_1d = np.full(numerator.shape, np.nan)
    valid = np.isfinite(denominator) & (denominator > 0.0)
    flux_1d[valid] = numerator[valid] / denominator[valid]

    # Also NaN out columns where numerator was not finite (all-NaN input).
    all_nan_cols = np.all(~np.isfinite(flux_ap), axis=0)
    flux_1d[all_nan_cols] = np.nan

    return flux_1d


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_optimal(
    rectified_orders: RectifiedOrderSet,
    extraction_def: OptimalExtractionDefinition,
    *,
    subtract_background: bool = True,
    variance_image: Optional[npt.NDArray] = None,
    variance_model: Optional[VarianceModelDefinition] = None,
    mask: Optional[npt.NDArray] = None,
) -> OptimalExtractedSpectrumSet:
    """Extract 1-D spectra from rectified orders using profile-weighted extraction.

    For each order in *rectified_orders*, selects aperture pixels, optionally
    subtracts a per-column background, estimates a provisional spatial profile,
    and computes a weighted 1-D extraction.

    Non-finite pixels (NaN or inf) in *data* or *variance_image*, plus any
    pixels flagged by *mask*, are excluded from both the flux sum and the
    denominator (weights/normalization) so they do not bias the result.
    Spectral columns where every aperture pixel is masked return NaN flux
    and NaN variance (when variance is computed).

    Parameters
    ----------
    rectified_orders : :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
        Stage-7 output.  Must contain at least one order.
    extraction_def : :class:`OptimalExtractionDefinition`
        Aperture, background region, and profile-mode parameters.
    subtract_background : bool, optional
        If ``True`` (default) **and** *extraction_def* has a background region
        defined (``extraction_def.has_background``), estimate a per-column
        background as the median of background-annulus pixels and subtract it
        from each aperture pixel before profile estimation and extraction.  If
        ``False``, or if *extraction_def* has no background region, no
        subtraction is performed.
    variance_image : ndarray or None, optional
        Per-pixel variance image with the same shape as each order's
        ``flux`` array (``n_spatial × n_spectral``).  If provided, it
        takes priority over *variance_model*.  If ``None`` (default) and
        *variance_model* is also ``None``, no variance propagation is
        performed.
    variance_model : :class:`~pyspextool.instruments.ishell.variance_model.VarianceModelDefinition` or None, optional
        Stage-14 variance model definition.  If *variance_image* is
        ``None`` and *variance_model* is provided, a variance image is
        built internally for each order using
        :func:`~pyspextool.instruments.ishell.variance_model.build_variance_image`.
        If both *variance_image* and *variance_model* are provided,
        *variance_image* is used and *variance_model* is ignored.
    mask : ndarray of bool or None, optional
        Optional bad-pixel mask with the same shape as each order's
        ``flux`` array.  Pixels where *mask* is ``True`` are treated as
        bad and excluded from extraction and variance propagation.
        Combined with the internal mask derived from non-finite (NaN/inf)
        values in *data* and *variance_image*.  If ``None`` (default), no
        additional masking beyond non-finite detection is applied.

    Returns
    -------
    :class:`OptimalExtractedSpectrumSet`
        One :class:`OptimalExtractedOrderSpectrum` per order in
        *rectified_orders*.  Each spectrum's ``variance`` field contains
        the propagated variance when a variance source is provided, or
        ``None`` otherwise.

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
        If *mask* is provided but its shape does not match the order's
        flux shape.

    Notes
    -----
    **Variance source priority**

    ``explicit variance_image > variance_model > None``

    **Algorithm**

    For each order:

    1. Compute ``dist = |spatial_frac - center_frac|`` for every spatial pixel.
    2. Select aperture pixels: ``dist <= radius_frac``.
    3. If background subtraction is enabled and the extraction definition
       has a background region, select background pixels:
       ``background_inner < dist <= background_outer``.
       Compute a per-column background estimate as
       :func:`numpy.nanmedian` of those pixels (shape ``(n_spectral,)``).
       Subtract the background from every aperture pixel.
    4. Estimate the spatial profile from the (background-subtracted) aperture
       sub-image using the method specified by *extraction_def.profile_mode*.
    5. Optionally normalize the profile column-by-column.
    6. Compute the weighted extraction:
       ``flux_1d[col] = sum(P * F, axis=0) / sum(P^2, axis=0)``
    7. Set columns with no valid aperture pixels or no valid profile
       support to NaN.

    **Variance propagation**

    When a variance source is available (either *variance_image* or built
    from *variance_model*):

    - Background variance per column is approximated as
      ``nanmedian(variance_image[bg_mask], axis=0)``.  This is a
      first-order approximation (median ≠ mean for skewed distributions).
    - The total per-pixel variance including background is
      ``variance_image[ap_mask] + var_bg_col``.
    - The propagated flux variance is
      ``nansum(weights**2 * variance_total, axis=0)`` where *weights*
      are the profile values used for extraction.

    **Scaffold simplifications**

    - Background is a simple per-column median of annulus pixels; no
      polynomial fitting or interpolation is performed.
    - No per-pixel variance weighting; all aperture pixels are treated
      equally by the profile.
    - No iterative bad-pixel rejection.
    - The aperture must be supplied explicitly; no centroiding or automatic
      aperture finding is performed.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if rectified_orders.n_orders == 0:
        raise ValueError(
            "rectified_orders is empty (n_orders == 0); "
            "cannot extract spectra."
        )

    # Whether background subtraction will actually be applied.
    do_background = subtract_background and extraction_def.has_background

    # ------------------------------------------------------------------
    # Extract one spectrum per order
    # ------------------------------------------------------------------
    spectra: list[OptimalExtractedOrderSpectrum] = []

    for ro in rectified_orders.rectified_orders:
        flux_2d = np.asarray(ro.flux, dtype=float)

        if flux_2d.ndim != 2:
            raise ValueError(
                f"Order {ro.order}: flux must be a 2-D array; "
                f"got shape {flux_2d.shape}."
            )

        n_spectral = flux_2d.shape[1]

        # -- Variance image handling --
        if variance_image is not None:
            var_2d = np.asarray(variance_image, dtype=float)
            if var_2d.shape != flux_2d.shape:
                raise ValueError(
                    f"Order {ro.order}: variance_image shape {var_2d.shape} "
                    f"does not match flux shape {flux_2d.shape}."
                )
        elif variance_model is not None:
            # Build variance image internally from the Stage-14 model.
            var_2d = build_variance_image(flux_2d, variance_model).variance_image
        else:
            var_2d = None

        # -- Build combined bad-pixel mask and apply it --
        # Exclude NaN/inf from data and variance, plus any user-supplied mask.
        bad = ~np.isfinite(flux_2d)
        if var_2d is not None:
            bad |= ~np.isfinite(var_2d)
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
            if var_2d is not None:
                var_2d = var_2d.copy()
                var_2d[bad] = np.nan

        spatial_frac = np.asarray(ro.spatial_frac, dtype=float)
        # shape (n_spatial,)
        dist = np.abs(spatial_frac - extraction_def.center_frac)

        # -- Aperture mask: shape (n_spatial,) boolean --
        ap_mask = dist <= extraction_def.radius_frac

        # Sub-image for aperture pixels: shape (n_ap, n_spectral)
        flux_ap = flux_2d[ap_mask, :]
        n_ap = flux_ap.shape[0]

        if n_ap == 0:
            # No spatial pixels inside the aperture for this order.
            spectra.append(
                OptimalExtractedOrderSpectrum(
                    order=ro.order,
                    wavelength_um=ro.wavelength_um.copy(),
                    flux=np.full(n_spectral, np.nan),
                    profile=np.full((0, n_spectral), np.nan),
                    aperture=extraction_def,
                    method="optimal_weighted",
                    n_pixels_used=0,
                    variance=None,
                )
            )
            continue

        # -- Background estimation --
        var_bg_col: Optional[npt.NDArray] = None
        if do_background:
            bg_mask = (dist > extraction_def.background_inner) & (
                dist <= extraction_def.background_outer
            )
            flux_bg = flux_2d[bg_mask, :]  # shape (n_bg, n_spectral)

            if flux_bg.shape[0] == 0:
                # No background pixels; use NaN background (no subtraction).
                bg_1d = np.full(n_spectral, np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bg_1d = np.nanmedian(flux_bg, axis=0)  # (n_spectral,)

            # Subtract per-column background from each aperture row.
            flux_ap = flux_ap - bg_1d[np.newaxis, :]  # broadcast

            # Background variance: approximate as median of variance within
            # the background annulus (first-order approximation).
            if var_2d is not None:
                var_bg_pixels = var_2d[bg_mask, :]  # (n_bg, n_spectral)
                if var_bg_pixels.shape[0] == 0:
                    var_bg_col = np.zeros(n_spectral)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        var_bg_col = np.nanmedian(
                            var_bg_pixels, axis=0
                        )  # (n_spectral,)

        # -- Profile estimation --
        profile = _estimate_profile(
            flux_ap,
            profile_mode=extraction_def.profile_mode,
            normalize_profile=extraction_def.normalize_profile,
        )

        # -- Weighted extraction --
        n_pixels_used = _count_aperture_pixels(flux_ap)
        flux_1d = _weighted_extract(profile, flux_ap)

        # -- Variance propagation --
        variance_1d: Optional[npt.NDArray] = None
        if var_2d is not None:
            var_ap = var_2d[ap_mask, :]  # (n_ap, n_spectral)
            if var_bg_col is not None:
                var_total = var_ap + var_bg_col[np.newaxis, :]
            else:
                var_total = var_ap
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                variance_1d = np.nansum(
                    profile**2 * var_total, axis=0
                )  # (n_spectral,)
            # Columns where all aperture pixels are NaN → NaN variance.
            all_nan_cols = np.all(~np.isfinite(flux_ap), axis=0)
            variance_1d[all_nan_cols] = np.nan

        spectra.append(
            OptimalExtractedOrderSpectrum(
                order=ro.order,
                wavelength_um=ro.wavelength_um.copy(),
                flux=flux_1d,
                profile=profile,
                aperture=extraction_def,
                method="optimal_weighted",
                n_pixels_used=n_pixels_used,
                variance=variance_1d,
            )
        )

    return OptimalExtractedSpectrumSet(
        mode=rectified_orders.mode,
        spectra=spectra,
    )
