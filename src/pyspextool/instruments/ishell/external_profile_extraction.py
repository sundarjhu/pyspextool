"""
External-profile extraction workflow for iSHELL 2DXD weighted extraction.

This module implements **Stage 22** of the iSHELL 2DXD reduction scaffold:
a clean, reusable interface for applying external profile templates — built
by :mod:`~pyspextool.instruments.ishell.profile_templates` (Stage 21) — to
weighted optimal extraction across all orders.

.. note::
    This is a **scaffold implementation**.  It is intentionally simple and
    does **not** represent a science-quality PSF-based extraction workflow.
    See ``docs/ishell_external_profile_extraction.md`` for a full description
    of the algorithm, its limitations, and what remains unimplemented.

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
   * - 21
     - ``profile_templates.py``
     - External profile template builder
   * - **22**
     - **``external_profile_extraction.py``**
     - **External-profile extraction workflow (this module)**

What this module does
---------------------
* Accepts a
  :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
  (Stage-7 output), a :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractionDefinition`,
  and an
  :class:`~pyspextool.instruments.ishell.profile_templates.ExternalProfileTemplateSet`
  (Stage-21 output).

* For each order in *rectified_orders*:

  1. Looks up the matching
     :class:`~pyspextool.instruments.ishell.profile_templates.ExternalProfileTemplate`
     via order number.

  2. If a matching template is found, overrides *extraction_def* with
     ``profile_source="external"`` and
     ``external_profile=template.profile``, then calls
     :func:`~pyspextool.instruments.ishell.weighted_optimal_extraction.extract_weighted_optimal`.

  3. If a matching template is *not* found:

     - If *fallback_profile_source* is ``None`` (default), raises a
       :exc:`ValueError` listing the missing orders.
     - If *fallback_profile_source* is a valid profile-source string
       (``"empirical"`` or ``"smoothed_empirical"``), uses that source
       instead without an external profile.

* Returns an :class:`ExternalProfileExtractionResult` containing the full
  :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractedSpectrumSet`
  plus lightweight per-order bookkeeping.

Validation
----------
Before extraction begins this module checks:

* *profile_templates* mode matches *rectified_orders* mode.
* Every order in *rectified_orders* has a template (or a fallback is
  configured).
* For orders with a template, the template's spatial dimension matches the
  rectified order's spatial dimension.
* For orders with a template, the template's spectral dimension matches the
  rectified order's spectral dimension (when the template is 2-D).

Fallback behaviour
------------------
Pass ``fallback_profile_source="empirical"`` (or
``"smoothed_empirical"``) to allow orders without a template to be
extracted using that profile source instead.  The per-order
``external_profile_applied`` bookkeeping field records which orders used
the external template and which fell back.

What this module does NOT do (by design)
-----------------------------------------
* **No PSF fitting** – profiles are flat empirical estimates.
* **No centroid alignment** – spatial centering is not adjusted between
  the template frame and the science frame.
* **No wavelength-dependent profile warping** – the template is assumed
  to be spectrally constant (or the wavelength axis is ignored).
* **No cross-order interpolation** – each order uses its own independent
  template.
* **No science-quality validation** – signal-to-noise, centring accuracy,
  and template staleness are not checked.

Public API
----------
- :func:`extract_with_external_profile` – main entry point.
- :class:`ExternalProfileExtractionResult` – result container.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy.typing as npt

from .profile_templates import ExternalProfileTemplateSet
from .rectified_orders import RectifiedOrderSet
from .variance_model import VarianceModelDefinition
from .weighted_optimal_extraction import (
    WeightedExtractionDefinition,
    WeightedExtractedSpectrumSet,
    extract_weighted_optimal,
)

__all__ = [
    "ExternalProfileExtractionResult",
    "extract_with_external_profile",
]

# Profile-source values that are accepted as fallback_profile_source.
_VALID_FALLBACK_SOURCES = {"empirical", "smoothed_empirical"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ExternalProfileExtractionResult:
    """Result of :func:`extract_with_external_profile`.

    Wraps the standard :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractedSpectrumSet`
    produced by :func:`~pyspextool.instruments.ishell.weighted_optimal_extraction.extract_weighted_optimal`
    and adds lightweight per-order bookkeeping about how the external
    profile template was applied.

    Parameters
    ----------
    spectra : :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractedSpectrumSet`
        Full extraction result; one spectrum per order.
    external_profile_applied : dict mapping int → bool
        ``True`` for each order that was extracted with an external profile
        template; ``False`` for orders that used the fallback profile
        source.
    template_n_frames_used : dict mapping int → int
        For each order where ``external_profile_applied`` is ``True``,
        the value of
        :attr:`~pyspextool.instruments.ishell.profile_templates.ExternalProfileTemplate.n_frames_used`
        from the template.  Orders that used the fallback source have
        value ``0``.
    """

    spectra: WeightedExtractedSpectrumSet
    external_profile_applied: dict[int, bool] = field(default_factory=dict)
    template_n_frames_used: dict[int, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience pass-throughs
    # ------------------------------------------------------------------

    @property
    def orders(self) -> list[int]:
        """List of order numbers in the set (in storage order)."""
        return self.spectra.orders

    @property
    def n_orders(self) -> int:
        """Number of orders in the result."""
        return self.spectra.n_orders


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_with_external_profile(
    rectified_orders: RectifiedOrderSet,
    extraction_def: WeightedExtractionDefinition,
    profile_templates: ExternalProfileTemplateSet,
    *,
    variance_image: Optional[npt.NDArray] = None,
    variance_model: Optional[VarianceModelDefinition] = None,
    mask: Optional[npt.NDArray] = None,
    fallback_profile_source: Optional[str] = None,
) -> ExternalProfileExtractionResult:
    """Extract 1-D spectra using an external spatial profile template.

    For each order in *rectified_orders*, retrieves the matching external
    profile template from *profile_templates* and calls
    :func:`~pyspextool.instruments.ishell.weighted_optimal_extraction.extract_weighted_optimal`
    with ``profile_source="external"`` and
    ``external_profile=template.profile``.

    Orders without a matching template either raise a :exc:`ValueError` (if
    *fallback_profile_source* is ``None``) or are extracted using the
    *fallback_profile_source* profile mode.

    Parameters
    ----------
    rectified_orders : :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
        Stage-7 output.  Must contain at least one order.
    extraction_def : :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractionDefinition`
        Aperture, background region, and optional variance-model parameters.
        ``profile_source`` and ``external_profile`` fields in this definition
        are overridden per order — **do not** set them manually for
        external-profile extraction (they will be ignored).
    profile_templates : :class:`~pyspextool.instruments.ishell.profile_templates.ExternalProfileTemplateSet`
        Stage-21 output.  One template per echelle order.
    variance_image : ndarray or None, optional
        Per-pixel variance image with the same shape as each order's
        ``flux`` array.  Highest-priority variance source.
    variance_model : :class:`~pyspextool.instruments.ishell.variance_model.VarianceModelDefinition` or None, optional
        Stage-14 variance model definition.  Used when *variance_image* is
        ``None``.
    mask : ndarray of bool or None, optional
        Optional bad-pixel mask with the same shape as each order's ``flux``
        array.  Pixels where *mask* is ``True`` are treated as bad.
    fallback_profile_source : str or None, optional
        If ``None`` (default), every order in *rectified_orders* must have a
        matching template; a :exc:`ValueError` is raised otherwise.

        If set to ``"empirical"`` or ``"smoothed_empirical"``, orders without
        a template are extracted using that profile source.  This is useful
        when the template set covers only a subset of the observed orders.

    Returns
    -------
    :class:`ExternalProfileExtractionResult`
        Contains the full :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractedSpectrumSet`
        plus ``external_profile_applied`` and ``template_n_frames_used``
        dictionaries for per-order bookkeeping.

    Raises
    ------
    ValueError
        If *rectified_orders* is empty.
    ValueError
        If the modes of *rectified_orders* and *profile_templates* differ.
    ValueError
        If one or more orders in *rectified_orders* have no matching template
        and *fallback_profile_source* is ``None``.
    ValueError
        If *fallback_profile_source* is not ``None`` and is not a valid
        profile-source string (``"empirical"`` or ``"smoothed_empirical"``).
    ValueError
        If a template's spatial dimension does not match the corresponding
        order's spatial dimension.
    ValueError
        If a template's spectral dimension (for 2-D templates) does not match
        the corresponding order's spectral dimension.

    Notes
    -----
    **Profile source override**

    The *extraction_def.profile_source* and *extraction_def.external_profile*
    fields are ignored.  This function always builds a fresh per-order
    :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractionDefinition`
    with ``profile_source`` set appropriately for each order.

    **Single-order extraction**

    Internally this function wraps each individual order in a temporary
    single-order :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
    and calls :func:`~pyspextool.instruments.ishell.weighted_optimal_extraction.extract_weighted_optimal`
    once per order, then assembles the results into the final
    :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractedSpectrumSet`.

    **Fallback behaviour**

    When *fallback_profile_source* is set and an order has no matching
    template, the order-level :class:`~pyspextool.instruments.ishell.weighted_optimal_extraction.WeightedExtractionDefinition`
    uses ``profile_source=fallback_profile_source`` with no external profile.
    The ``external_profile_applied`` bookkeeping entry for that order is set
    to ``False`` and ``template_n_frames_used`` is set to ``0``.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if rectified_orders.n_orders == 0:
        raise ValueError(
            "rectified_orders is empty (n_orders == 0); "
            "cannot extract spectra."
        )

    if rectified_orders.mode != profile_templates.mode:
        raise ValueError(
            f"Mode mismatch: rectified_orders.mode={rectified_orders.mode!r} "
            f"but profile_templates.mode={profile_templates.mode!r}.  "
            "All inputs must share the same iSHELL observing mode."
        )

    if fallback_profile_source is not None:
        if fallback_profile_source not in _VALID_FALLBACK_SOURCES:
            raise ValueError(
                f"fallback_profile_source={fallback_profile_source!r} is not "
                f"valid.  Must be one of {sorted(_VALID_FALLBACK_SOURCES)} or "
                "None."
            )

    # Identify which orders have templates and which do not.
    template_orders = set(profile_templates.orders)
    science_orders = rectified_orders.orders
    missing = [o for o in science_orders if o not in template_orders]

    if missing and fallback_profile_source is None:
        raise ValueError(
            f"No template found for order(s) {missing} in "
            f"profile_templates (available: {profile_templates.orders}).  "
            "Provide a template for every science order or set "
            "fallback_profile_source to 'empirical' / 'smoothed_empirical'."
        )

    # Per-order spatial and spectral dimension compatibility check.
    for ro in rectified_orders.rectified_orders:
        if ro.order not in template_orders:
            continue  # will use fallback; no template to validate
        template = profile_templates.get_order(ro.order)
        _validate_template_shape(template, ro)

    # ------------------------------------------------------------------
    # Per-order extraction
    # ------------------------------------------------------------------
    all_spectra = []
    applied: dict[int, bool] = {}
    n_frames: dict[int, int] = {}

    for ro in rectified_orders.rectified_orders:
        order = ro.order

        if order in template_orders:
            template = profile_templates.get_order(order)
            # Build an overridden extraction definition for this order.
            order_def = _override_profile_source(
                extraction_def,
                profile_source="external",
                external_profile=template.profile,
            )
            applied[order] = True
            n_frames[order] = template.n_frames_used
        else:
            # Use the fallback profile source (already validated above).
            order_def = _override_profile_source(
                extraction_def,
                profile_source=fallback_profile_source,
                external_profile=None,
            )
            applied[order] = False
            n_frames[order] = 0

        # Wrap single order in a temporary RectifiedOrderSet.
        single_order_set = RectifiedOrderSet(
            mode=rectified_orders.mode,
            rectified_orders=[ro],
            source_image_shape=rectified_orders.source_image_shape,
        )

        order_result = extract_weighted_optimal(
            single_order_set,
            order_def,
            variance_image=variance_image,
            variance_model=variance_model,
            mask=mask,
        )
        # extract_weighted_optimal returns one spectrum for the single order.
        all_spectra.extend(order_result.spectra)

    spectrum_set = WeightedExtractedSpectrumSet(
        mode=rectified_orders.mode,
        spectra=all_spectra,
    )

    return ExternalProfileExtractionResult(
        spectra=spectrum_set,
        external_profile_applied=applied,
        template_n_frames_used=n_frames,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_template_shape(template, rectified_order) -> None:
    """Check that *template* is compatible with *rectified_order*.

    Parameters
    ----------
    template : ExternalProfileTemplate
        Template to check.
    rectified_order : RectifiedOrder
        Corresponding rectified order image.

    Raises
    ------
    ValueError
        If the spatial dimensions do not match.
    ValueError
        If the spectral dimensions do not match (for 2-D templates).
    """
    n_spatial_order = rectified_order.n_spatial
    profile = template.profile  # shape (n_spatial,) or (n_spatial, n_spectral)

    if profile.ndim == 1:
        n_spatial_tmpl = profile.shape[0]
    else:
        n_spatial_tmpl, n_spectral_tmpl = profile.shape[:2]
        # Spectral check only for 2-D templates.
        n_spectral_order = rectified_order.n_spectral
        if n_spectral_tmpl != n_spectral_order:
            raise ValueError(
                f"Order {rectified_order.order}: template spectral dimension "
                f"({n_spectral_tmpl}) does not match rectified order spectral "
                f"dimension ({n_spectral_order})."
            )

    if n_spatial_tmpl != n_spatial_order:
        raise ValueError(
            f"Order {rectified_order.order}: template spatial dimension "
            f"({n_spatial_tmpl}) does not match rectified order spatial "
            f"dimension ({n_spatial_order})."
        )


def _override_profile_source(
    extraction_def: WeightedExtractionDefinition,
    profile_source: str,
    external_profile: Optional[npt.NDArray],
) -> WeightedExtractionDefinition:
    """Return a copy of *extraction_def* with overridden profile fields.

    Parameters
    ----------
    extraction_def : WeightedExtractionDefinition
        Template extraction definition to copy.
    profile_source : str
        New value for ``profile_source``.
    external_profile : ndarray or None
        New value for ``external_profile``.

    Returns
    -------
    WeightedExtractionDefinition
        A new instance with the profile fields replaced.
    """
    return dataclasses.replace(
        extraction_def,
        profile_source=profile_source,
        external_profile=external_profile,
    )
