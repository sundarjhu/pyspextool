"""
External spatial-profile template builder for iSHELL 2DXD weighted extraction.

This module implements **Stage 21** of the iSHELL 2DXD reduction scaffold:
constructing an external spatial-profile template from one or more
:class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
inputs.  The resulting templates are ready for direct use with
``profile_source="external"`` in
:func:`~pyspextool.instruments.ishell.weighted_optimal_extraction.extract_weighted_optimal`.

.. note::
    This is a **scaffold implementation**.  It is intentionally simple and
    does **not** represent science-quality PSF fitting, order-to-order
    profile transfer, or wavelength-dependent profile modeling.  See
    ``docs/ishell_profile_templates.md`` for a full description of the
    algorithm, its limitations, and what remains unimplemented.

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
   * - 17–20
     - ``weighted_optimal_extraction.py``
     - Proto-Horne variance-weighted extraction
   * - **21**
     - **``profile_templates.py``**
     - **External profile template builder (this module)**

What this module does
---------------------
* Accepts one or more
  :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
  objects (all from the same iSHELL mode).

* For each echelle order that appears in all provided sets:

  1. Stacks the rectified flux images along a new axis (one image per frame).

  2. Optionally applies a pixel mask.

  3. Combines the stacked images across frames using either the per-pixel
     **median** (default) or **mean**, treating NaN values robustly.

  4. Derives a 2-D spatial profile from the combined image using a
     per-row nanmedian across the wavelength axis, which is then broadcast
     across all spectral columns (equivalent to the ``"global_median"``
     profile mode used in Stage 12 / Stage 17).

  5. Optionally applies Gaussian smoothing along the spatial axis (same
     kernel as the ``"smoothed_empirical"`` profile source in Stage 20).

  6. Optionally normalizes the profile column-by-column so that each
     column sums to 1.0.

* Returns an :class:`ExternalProfileTemplateSet` containing one
  :class:`ExternalProfileTemplate` per order.

What this module does NOT do (by design)
-----------------------------------------
* **No PSF fitting** – the profile is derived from a simple median, not
  from a parametric PSF model.
* **No order-to-order profile transfer** – each order's profile is
  built independently from the frames supplied for that order.
* **No wavelength-dependent centroid alignment** – the spatial axis
  remains in fractional slit-position units; no centroid tracking.
* **No cross-night calibration logic** – all input frames are treated
  as equally valid without temporal weighting.
* **No science-quality profile modeling** – this is a scaffold intended
  for pipeline plumbing and early testing only.
* **No bad-pixel interpolation** – bad pixels are masked (set to NaN)
  rather than replaced by a model value.
* **No telluric correction or order merging** – downstream steps.

Compatibility target
--------------------
The per-order ``profile`` arrays returned by
:class:`ExternalProfileTemplate` have shape ``(n_spatial, n_spectral)``
and can be supplied directly as:

.. code-block:: python

    from pyspextool.instruments.ishell.weighted_optimal_extraction import (
        WeightedExtractionDefinition,
        extract_weighted_optimal,
    )

    ext_def = WeightedExtractionDefinition(
        center_frac=0.5,
        radius_frac=0.2,
        profile_source="external",
        external_profile=template.profile,
    )
    result = extract_weighted_optimal(rectified_orders, ext_def)

Public API
----------
- :class:`ProfileTemplateDefinition` – template-builder parameters.
- :class:`ExternalProfileTemplate` – per-order profile template container.
- :class:`ExternalProfileTemplateSet` – full template set container.
- :func:`build_external_profile_template` – main entry point.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from .rectified_orders import RectifiedOrderSet

__all__ = [
    "ProfileTemplateDefinition",
    "ExternalProfileTemplate",
    "ExternalProfileTemplateSet",
    "build_external_profile_template",
]


# ---------------------------------------------------------------------------
# Template-builder definition / parameters
# ---------------------------------------------------------------------------


@dataclass
class ProfileTemplateDefinition:
    """Parameters controlling the external profile template builder.

    Parameters
    ----------
    combine_method : str
        How to combine flux images across multiple input frames.
        ``"median"`` (default) uses the per-pixel nanmedian;
        ``"mean"`` uses the per-pixel nanmean.
    normalize_profile : bool
        If ``True`` (default), normalize the derived profile so that the
        spatial sum equals 1.0 per spectral column.  Columns whose spatial
        sum is zero or NaN are set to NaN.
    smooth_sigma : float
        Standard deviation (in spatial pixels) of the Gaussian smoothing
        kernel applied along the spatial axis after profile estimation.
        Set to ``0.0`` (default) to skip smoothing.
    min_finite_fraction : float
        Minimum fraction of finite input pixels required across frames for
        a given image pixel to be included in the combined image.  Pixels
        that do not meet this threshold are set to NaN.  Must be in
        ``(0.0, 1.0]``; default is ``0.5``.
    mask_sources : bool
        Reserved for future use.  Currently has no effect on the output.
        When ``True`` (default), the intent is to mask bright point sources
        before combining; this is not yet implemented.
    """

    combine_method: str = "median"
    normalize_profile: bool = True
    smooth_sigma: float = 0.0
    min_finite_fraction: float = 0.5
    mask_sources: bool = True

    def __post_init__(self) -> None:
        if self.combine_method not in ("median", "mean"):
            raise ValueError(
                f"combine_method must be 'median' or 'mean'; "
                f"got {self.combine_method!r}."
            )
        if self.smooth_sigma < 0.0:
            raise ValueError(
                f"smooth_sigma must be >= 0.0; got {self.smooth_sigma}."
            )
        if not (0.0 < self.min_finite_fraction <= 1.0):
            raise ValueError(
                f"min_finite_fraction must be in (0.0, 1.0]; "
                f"got {self.min_finite_fraction}."
            )


# ---------------------------------------------------------------------------
# Per-order template container
# ---------------------------------------------------------------------------


@dataclass
class ExternalProfileTemplate:
    """Spatial-profile template for one echelle order.

    Parameters
    ----------
    order : int
        Echelle order number.
    wavelength_um : ndarray, shape (n_spectral,)
        Wavelength axis in µm, taken from the first input
        :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrder`.
    spatial_frac : ndarray, shape (n_spatial,)
        Spatial axis as a fractional slit position in ``[0.0, 1.0]``.
    profile : ndarray, shape (n_spatial, n_spectral)
        Derived spatial-profile array.  Each column sums to 1.0 when
        *normalize_profile* is ``True``.  Columns with insufficient finite
        support are NaN.
    n_frames_used : int
        Number of input frames that contained this order.
    source_mode : str
        iSHELL observing mode of the contributing frames (e.g. ``"H1"``).
    profile_smoothed : bool
        Whether Gaussian smoothing was applied to this profile.
    finite_fraction : float
        Fraction of pixels in the combined image that were finite before
        profile estimation.  Useful for quality control.

    Notes
    -----
    The ``profile`` array is suitable for direct use as
    ``external_profile`` in :class:`WeightedExtractionDefinition` with
    ``profile_source="external"``.  If the aperture in the extraction
    definition is smaller than the full spatial extent of the template,
    the extraction code will slice the profile accordingly.
    """

    order: int
    wavelength_um: npt.NDArray  # shape (n_spectral,)
    spatial_frac: npt.NDArray   # shape (n_spatial,)
    profile: npt.NDArray         # shape (n_spatial, n_spectral)
    n_frames_used: int
    source_mode: str
    profile_smoothed: bool
    finite_fraction: float

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels."""
        return int(self.wavelength_um.shape[0])

    @property
    def n_spatial(self) -> int:
        """Number of spatial pixels."""
        return int(self.spatial_frac.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Profile array shape ``(n_spatial, n_spectral)``."""
        return (self.n_spatial, self.n_spectral)


# ---------------------------------------------------------------------------
# Template-set container
# ---------------------------------------------------------------------------


@dataclass
class ExternalProfileTemplateSet:
    """Collection of per-order external profile templates.

    Parameters
    ----------
    mode : str
        iSHELL observing mode shared by all contributing frames
        (e.g. ``"H1"``).
    templates : list of ExternalProfileTemplate
        One entry per echelle order.  In the same order as the input
        :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`.
    """

    mode: str
    templates: List[ExternalProfileTemplate] = field(default_factory=list)

    @property
    def orders(self) -> list[int]:
        """List of order numbers present in this set (in storage order)."""
        return [t.order for t in self.templates]

    @property
    def n_orders(self) -> int:
        """Number of orders in this set."""
        return len(self.templates)

    def get_order(self, order: int) -> ExternalProfileTemplate:
        """Return the :class:`ExternalProfileTemplate` for *order*.

        Parameters
        ----------
        order : int
            Echelle order number to look up.

        Returns
        -------
        ExternalProfileTemplate

        Raises
        ------
        KeyError
            If *order* is not present in this set.
        """
        for t in self.templates:
            if t.order == order:
                return t
        raise KeyError(
            f"Order {order} not found in ExternalProfileTemplateSet "
            f"(available: {self.orders})."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _combine_frames(
    stack: npt.NDArray,
    method: str,
    min_finite_fraction: float,
) -> npt.NDArray:
    """Combine a stack of images along axis 0 using median or mean.

    Parameters
    ----------
    stack : ndarray, shape (n_frames, n_spatial, n_spectral)
        Stack of aligned flux images (NaN for bad/missing pixels).
    method : str
        ``"median"`` or ``"mean"``.
    min_finite_fraction : float
        Pixels that are finite in fewer than this fraction of frames are
        set to NaN in the output.

    Returns
    -------
    ndarray, shape (n_spatial, n_spectral)
        Combined image.
    """
    n_frames = stack.shape[0]

    # Count finite values per pixel across frames.
    finite_count = np.sum(np.isfinite(stack), axis=0)  # (n_spatial, n_spectral)
    finite_frac = finite_count / n_frames

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "median":
            combined = np.nanmedian(stack, axis=0)
        else:  # "mean"
            combined = np.nanmean(stack, axis=0)

    # Blank pixels that don't meet the finite-fraction threshold.
    combined[finite_frac < min_finite_fraction] = np.nan

    return combined


def _estimate_profile_from_image(
    combined: npt.NDArray,
    normalize_profile: bool,
) -> npt.NDArray:
    """Derive a 2-D spatial profile from a combined image.

    Uses a per-row nanmedian across the wavelength axis (equivalent to the
    ``"global_median"`` profile mode in Stage 12 / Stage 17), then
    broadcasts the 1-D profile vector across all spectral columns.

    Parameters
    ----------
    combined : ndarray, shape (n_spatial, n_spectral)
        Combined (multi-frame) rectified flux image.
    normalize_profile : bool
        If ``True``, normalize each column to sum to 1.0.  Columns with
        zero or NaN sum are set to NaN.

    Returns
    -------
    ndarray, shape (n_spatial, n_spectral)
        Estimated spatial profile.
    """
    n_spatial, n_spectral = combined.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Per-row median across wavelength → 1-D profile vector.
        profile_1d = np.nanmedian(combined, axis=1)  # (n_spatial,)

    finite_1d = np.isfinite(profile_1d)
    # Clip negatives to zero; preserve NaN.
    profile_1d = np.where(finite_1d, np.maximum(profile_1d, 0.0), np.nan)

    # Broadcast to full (n_spatial, n_spectral) shape.
    profile = np.broadcast_to(
        profile_1d[:, np.newaxis], (n_spatial, n_spectral)
    ).copy()

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
    """Smooth a spatial profile with a Gaussian kernel along the spatial axis.

    Parameters
    ----------
    profile : ndarray, shape (n_spatial, n_spectral)
        Profile to smooth.  May contain NaN.
    sigma : float
        Gaussian standard deviation in spatial pixels.  When ``<= 0``
        smoothing is skipped.
    normalize_profile : bool
        If ``True``, renormalize each column to sum to 1.0 after smoothing.

    Returns
    -------
    ndarray, shape (n_spatial, n_spectral)
        Smoothed (and optionally renormalized) profile.
    """
    nan_mask = ~np.isfinite(profile)
    if sigma > 0.0:
        filled = np.where(nan_mask, 0.0, profile)
        smoothed = gaussian_filter1d(filled, sigma=sigma, axis=0, mode="reflect")
        smoothed[nan_mask] = np.nan
    else:
        smoothed = profile.copy()

    # Clip negatives introduced by the Gaussian kernel at boundaries.
    smoothed = np.where(np.isfinite(smoothed), np.maximum(smoothed, 0.0), np.nan)

    if normalize_profile:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_sums = np.nansum(smoothed, axis=0)
        zero_or_nan = (col_sums == 0.0) | (~np.isfinite(col_sums))
        safe_sums = np.where(zero_or_nan, 1.0, col_sums)
        smoothed = smoothed / safe_sums[np.newaxis, :]
        smoothed[:, zero_or_nan] = np.nan

    return smoothed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_external_profile_template(
    rectified_order_sets: "RectifiedOrderSet | list[RectifiedOrderSet]",
    definition: ProfileTemplateDefinition,
    *,
    mask: Optional[npt.NDArray] = None,
) -> ExternalProfileTemplateSet:
    """Build an external spatial-profile template from rectified order images.

    Combines one or more rectified-order sets into a single set of
    spatial-profile templates suitable for use with
    ``profile_source="external"`` in
    :func:`~pyspextool.instruments.ishell.weighted_optimal_extraction.extract_weighted_optimal`.

    Parameters
    ----------
    rectified_order_sets : RectifiedOrderSet or list of RectifiedOrderSet
        Input rectified order set(s).  All sets must share the same iSHELL
        mode.  When multiple sets are supplied, orders are matched by order
        number; only orders present in *all* sets are included in the output.
    definition : ProfileTemplateDefinition
        Template-builder parameters (combine method, smoothing, normalization,
        etc.).
    mask : ndarray, shape (n_spatial, n_spectral) or None
        Optional boolean bad-pixel mask for the rectified order images.
        ``True`` marks pixels to exclude (set to NaN before combining).
        If supplied, the mask must match the shape of each per-order
        rectified flux image ``(n_spatial, n_spectral)``.  Currently the
        mask is applied uniformly to every rectified pixel (no
        sub-pixel interpolation of the mask boundary); masked pixels
        become NaN in the combined image.

        .. note::
            The mask is applied in *rectified* space: a ``True`` entry at
            ``mask[j, i]`` sets the corresponding pixel in every input flux
            image to NaN before combining.  There is no inverse-rectification
            of the mask from detector to rectified coordinates at this stage.

    Returns
    -------
    ExternalProfileTemplateSet
        One :class:`ExternalProfileTemplate` per order that appears in all
        input sets.

    Raises
    ------
    ValueError
        If *rectified_order_sets* is empty.
    ValueError
        If the input sets do not all share the same mode.
    ValueError
        If no orders are common to all input sets.
    ValueError
        If *mask* shape is incompatible with the input data shapes.

    Notes
    -----
    **Single-frame input** – if only one
    :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
    is provided, the profile is derived directly from that single frame.
    The ``combine_method`` parameter has no effect in this case (there is
    nothing to combine), though the interface remains consistent.

    **Multi-frame stacking** – when multiple sets are provided, their flux
    images are stacked along a new leading axis and combined pixel-by-pixel
    using the chosen method.  Only pixels that are finite in at least
    ``definition.min_finite_fraction`` of the frames contribute to the
    combined image.

    **Profile estimation** – after combining, the profile is estimated by
    taking the per-row nanmedian across the wavelength axis (same as the
    ``"global_median"`` mode in the Stage-17 extractor).  Negative values
    are clipped to zero.

    **Smoothing** – if ``definition.smooth_sigma > 0``, the profile is
    smoothed along the spatial axis with a Gaussian kernel.

    **Normalization** – if ``definition.normalize_profile`` is ``True``
    (default), each spectral column of the profile is rescaled so that
    the spatial sum equals 1.0.  Columns with zero or NaN spatial sum are
    set to NaN.

    **Insufficient finite support** – if an entire order image is NaN
    (e.g. all input frames have NaN for that order's pixels), the returned
    template will contain a NaN profile and a ``finite_fraction`` of 0.0.
    The template is still included in the returned set rather than silently
    dropped, so callers can detect and handle such pathological cases.
    """
    # ------------------------------------------------------------------
    # Normalise input to a list.
    # ------------------------------------------------------------------
    if isinstance(rectified_order_sets, RectifiedOrderSet):
        sets: list[RectifiedOrderSet] = [rectified_order_sets]
    else:
        sets = list(rectified_order_sets)

    if len(sets) == 0:
        raise ValueError(
            "rectified_order_sets must contain at least one RectifiedOrderSet."
        )

    # ------------------------------------------------------------------
    # Validate that all sets share the same mode.
    # ------------------------------------------------------------------
    modes = [s.mode for s in sets]
    if len(set(modes)) != 1:
        raise ValueError(
            f"All input RectifiedOrderSets must share the same mode; "
            f"got modes: {modes}."
        )
    mode = modes[0]

    # ------------------------------------------------------------------
    # Find the intersection of order numbers across all sets.
    # ------------------------------------------------------------------
    order_sets_per_input = [set(s.orders) for s in sets]
    common_orders = order_sets_per_input[0]
    for order_set in order_sets_per_input[1:]:
        common_orders = common_orders & order_set
    common_orders = sorted(common_orders)

    if len(common_orders) == 0:
        raise ValueError(
            "No common orders found across the provided RectifiedOrderSets."
        )

    # ------------------------------------------------------------------
    # Validate mask (if provided) against first-set source_image_shape.
    # ------------------------------------------------------------------
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        # We can only validate the mask if it is 2-D and matches the
        # rectified-order image shapes (n_spatial, n_spectral), which
        # may differ per order.  We defer per-order shape validation to
        # the per-order loop below.
        if mask_arr.ndim not in (2,):
            raise ValueError(
                f"mask must be a 2-D boolean array; got shape {mask_arr.shape}."
            )
    else:
        mask_arr = None

    # ------------------------------------------------------------------
    # Build per-order templates.
    # ------------------------------------------------------------------
    templates: list[ExternalProfileTemplate] = []

    for order in common_orders:
        # Collect the rectified flux images for this order from all sets.
        flux_images: list[npt.NDArray] = []
        ref_order = sets[0].get_order(order)
        wavelength_um = ref_order.wavelength_um.copy()
        spatial_frac = ref_order.spatial_frac.copy()
        expected_shape = ref_order.flux.shape  # (n_spatial, n_spectral)

        for ros in sets:
            ro = ros.get_order(order)
            flux = ro.flux.astype(float).copy()

            # Apply user mask in rectified space if provided.
            if mask_arr is not None:
                if mask_arr.shape != flux.shape:
                    raise ValueError(
                        f"mask shape {mask_arr.shape} is incompatible with "
                        f"order {order} flux shape {flux.shape}."
                    )
                flux[mask_arr] = np.nan

            flux_images.append(flux)

        # Stack to (n_frames, n_spatial, n_spectral).
        n_frames = len(flux_images)
        stack = np.stack(flux_images, axis=0)

        # Compute finite fraction before combining.
        finite_frac_total = float(np.mean(np.isfinite(stack)))

        if n_frames == 1:
            # Single-frame: no need to call _combine_frames.
            combined = stack[0].copy()
            # Still apply min_finite_fraction: pixels that are NaN in
            # the single frame stay NaN (no change needed).
        else:
            combined = _combine_frames(
                stack,
                method=definition.combine_method,
                min_finite_fraction=definition.min_finite_fraction,
            )

        # Estimate spatial profile from the combined image.
        profile = _estimate_profile_from_image(
            combined, normalize_profile=definition.normalize_profile
        )

        # Optionally smooth.
        profile_smoothed_flag = definition.smooth_sigma > 0.0
        if profile_smoothed_flag:
            profile = _smooth_profile(
                profile,
                sigma=definition.smooth_sigma,
                normalize_profile=definition.normalize_profile,
            )

        templates.append(
            ExternalProfileTemplate(
                order=order,
                wavelength_um=wavelength_um,
                spatial_frac=spatial_frac,
                profile=profile,
                n_frames_used=n_frames,
                source_mode=mode,
                profile_smoothed=profile_smoothed_flag,
                finite_fraction=finite_frac_total,
            )
        )

    return ExternalProfileTemplateSet(mode=mode, templates=templates)
