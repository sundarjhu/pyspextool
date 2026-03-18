"""
Profile diagnostics and quality metrics for iSHELL 2DXD external-profile extraction.

This module implements **Stage 23** of the iSHELL 2DXD reduction scaffold:
a diagnostic-only layer that characterises external spatial-profile template
quality, compares external vs empirical extraction behaviour, and detects
potential template leakage (template built from the same data as the science
frame).

.. important::
    **This module is diagnostic only.**

    It does **not** change extraction behaviour, reject templates
    automatically, switch fallback sources, or make any pipeline decisions.
    All flags and thresholds are heuristic and must *not* be used for
    autonomous science-quality decisions.

    "These diagnostics are not sufficient for science validation."

Pipeline stage
--------------
.. list-table::
   :header-rows: 1

   * - Stage
     - Module
     - Purpose
   * - 21
     - ``profile_templates.py``
     - External profile template builder
   * - 22
     - ``external_profile_extraction.py``
     - External-profile extraction workflow
   * - **23**
     - **``profile_diagnostics.py``**
     - **Template quality diagnostics (this module)**

Public API
----------
- :class:`ProfileDiagnostics`
- :class:`ProfileDiagnosticsSet`
- :func:`compute_profile_diagnostics`
- :class:`ExternalVsEmpiricalDiagnostics`
- :class:`ExternalVsEmpiricalDiagnosticsSet`
- :func:`compare_external_vs_empirical`
- :class:`TemplateLeakageDiagnostics`
- :class:`TemplateLeakageDiagnosticsSet`
- :func:`compute_leakage_diagnostics`
- :func:`run_full_profile_diagnostics`
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from .external_profile_extraction import extract_with_external_profile
from .profile_templates import ExternalProfileTemplate, ExternalProfileTemplateSet
from .rectified_orders import RectifiedOrderSet
from .variance_model import VarianceModelDefinition
from .weighted_optimal_extraction import (
    WeightedExtractionDefinition,
    extract_weighted_optimal,
)

__all__ = [
    "ProfileDiagnostics",
    "ProfileDiagnosticsSet",
    "compute_profile_diagnostics",
    "ExternalVsEmpiricalDiagnostics",
    "ExternalVsEmpiricalDiagnosticsSet",
    "compare_external_vs_empirical",
    "TemplateLeakageDiagnostics",
    "TemplateLeakageDiagnosticsSet",
    "compute_leakage_diagnostics",
    "run_full_profile_diagnostics",
]

# ---------------------------------------------------------------------------
# Leakage detection thresholds (heuristic, diagnostic only)
# ---------------------------------------------------------------------------

#: Pearson correlation coefficient above which two profiles are considered
#: suspiciously similar.  This is a heuristic value; it does not imply
#: science-quality template validation.
LEAKAGE_CORRELATION_THRESHOLD: float = 0.98

#: Normalised L2 difference below which two profiles are considered
#: suspiciously similar.  Both profiles are normalised to unit L2 norm
#: before comparison so that scale differences do not inflate the metric.
LEAKAGE_L2_THRESHOLD: float = 0.05

#: Column-sum tolerance used by :attr:`ProfileDiagnostics.is_normalized_like`.
#: A template is considered "normalised-like" when all finite column sums
#: are within this absolute tolerance of 1.0.
_NORMALIZED_ATOL: float = 1e-2


# ===========================================================================
# 1. Template quality diagnostics
# ===========================================================================


@dataclass
class ProfileDiagnostics:
    """Per-order quality metrics for an :class:`~.profile_templates.ExternalProfileTemplate`.

    All metrics are computed directly from the stored profile array and are
    intended as a first-pass quality screen — not as science-quality
    validation.

    Parameters
    ----------
    order : int
        Echelle order number.
    finite_fraction : float
        Fraction of finite values in the profile array (range ``[0, 1]``).
    peak_spatial_index : int
        Row index of the spatial maximum in the collapsed (per-column median)
        profile.  ``-1`` when no finite values exist.
    peak_spatial_frac : float
        Fractional slit position (``spatial_frac[peak_spatial_index]``) of
        the peak.  ``float('nan')`` when no finite values exist.
    colsum_min : float
        Minimum finite column sum across all spectral columns.
        ``float('nan')`` when the profile is entirely non-finite.
    colsum_max : float
        Maximum finite column sum.  ``float('nan')`` when entirely non-finite.
    colsum_median : float
        Median finite column sum.  ``float('nan')`` when entirely non-finite.
    roughness : float
        Mean absolute difference between adjacent finite values along the
        spatial axis of the collapsed 1-D profile.  A rough proxy for
        high-spatial-frequency noise.  ``float('nan')`` when fewer than two
        finite values are available.
    n_frames_used : int
        Number of calibration frames that contributed to this template
        (copied from :attr:`~.profile_templates.ExternalProfileTemplate.n_frames_used`).
    is_normalized_like : bool
        ``True`` when all finite column sums are within
        :data:`_NORMALIZED_ATOL` of 1.0.
    """

    order: int
    finite_fraction: float
    peak_spatial_index: int
    peak_spatial_frac: float
    colsum_min: float
    colsum_max: float
    colsum_median: float
    roughness: float
    n_frames_used: int
    is_normalized_like: bool


@dataclass
class ProfileDiagnosticsSet:
    """Collection of :class:`ProfileDiagnostics` for all orders.

    Parameters
    ----------
    mode : str
        iSHELL observing mode string (e.g. ``"H1"``).
    diagnostics : list of :class:`ProfileDiagnostics`
        One entry per order, in the order they appear in the template set.
    """

    mode: str
    diagnostics: List[ProfileDiagnostics] = field(default_factory=list)

    @property
    def orders(self) -> list[int]:
        """List of order numbers in this set."""
        return [d.order for d in self.diagnostics]

    @property
    def n_orders(self) -> int:
        """Number of orders."""
        return len(self.diagnostics)

    def get_order(self, order: int) -> ProfileDiagnostics:
        """Return the :class:`ProfileDiagnostics` for a specific order.

        Parameters
        ----------
        order : int
            Echelle order number.

        Raises
        ------
        KeyError
            If *order* is not present in this set.
        """
        for d in self.diagnostics:
            if d.order == order:
                return d
        raise KeyError(
            f"Order {order} not found in ProfileDiagnosticsSet "
            f"(available: {self.orders})."
        )


def _compute_single_profile_diagnostics(
    template: ExternalProfileTemplate,
) -> ProfileDiagnostics:
    """Compute :class:`ProfileDiagnostics` from a single template.

    Parameters
    ----------
    template : ExternalProfileTemplate
        The template to analyse.

    Returns
    -------
    ProfileDiagnostics
    """
    profile = template.profile  # shape (n_spatial, n_spectral) or (n_spatial,)
    spatial_frac = template.spatial_frac

    # Ensure 2-D.
    if profile.ndim == 1:
        profile = profile[:, np.newaxis]

    n_spatial, n_spectral = profile.shape
    n_total = profile.size

    # ------------------------------------------------------------------
    # finite_fraction
    # ------------------------------------------------------------------
    finite_mask = np.isfinite(profile)
    n_finite = int(np.sum(finite_mask))
    finite_fraction = n_finite / n_total if n_total > 0 else 0.0

    # ------------------------------------------------------------------
    # Collapsed 1-D profile: per-row median across spectral axis.
    # Used for peak detection and roughness.
    # ------------------------------------------------------------------
    with np.errstate(all="ignore"):
        collapsed = np.nanmedian(profile, axis=1)  # shape (n_spatial,)

    # ------------------------------------------------------------------
    # peak_spatial_index / peak_spatial_frac
    # ------------------------------------------------------------------
    collapsed_finite = np.where(np.isfinite(collapsed), collapsed, np.nan)
    if np.any(np.isfinite(collapsed_finite)):
        peak_idx = int(np.nanargmax(collapsed_finite))
        peak_frac = float(spatial_frac[peak_idx])
    else:
        peak_idx = -1
        peak_frac = float("nan")

    # ------------------------------------------------------------------
    # Column sums
    # ------------------------------------------------------------------
    with np.errstate(all="ignore"):
        colsums = np.nansum(profile, axis=0).astype(float)
    # Only consider columns that have at least one finite value.
    finite_col_mask = np.any(finite_mask, axis=0)
    if np.any(finite_col_mask):
        valid_colsums = colsums[finite_col_mask]
        colsum_min = float(np.min(valid_colsums))
        colsum_max = float(np.max(valid_colsums))
        colsum_median = float(np.median(valid_colsums))
    else:
        colsum_min = float("nan")
        colsum_max = float("nan")
        colsum_median = float("nan")

    # ------------------------------------------------------------------
    # roughness: mean |diff| along spatial axis of collapsed profile
    # ------------------------------------------------------------------
    finite_vals = collapsed_finite[np.isfinite(collapsed_finite)]
    if finite_vals.size >= 2:
        roughness = float(np.mean(np.abs(np.diff(finite_vals))))
    else:
        roughness = float("nan")

    # ------------------------------------------------------------------
    # is_normalized_like
    # ------------------------------------------------------------------
    if np.any(finite_col_mask):
        is_normalized_like = bool(
            np.all(np.abs(valid_colsums - 1.0) <= _NORMALIZED_ATOL)
        )
    else:
        is_normalized_like = False

    return ProfileDiagnostics(
        order=template.order,
        finite_fraction=finite_fraction,
        peak_spatial_index=peak_idx,
        peak_spatial_frac=peak_frac,
        colsum_min=colsum_min,
        colsum_max=colsum_max,
        colsum_median=colsum_median,
        roughness=roughness,
        n_frames_used=template.n_frames_used,
        is_normalized_like=is_normalized_like,
    )


def compute_profile_diagnostics(
    profile_templates: ExternalProfileTemplateSet,
) -> ProfileDiagnosticsSet:
    """Compute per-order quality metrics for an :class:`~.profile_templates.ExternalProfileTemplateSet`.

    Parameters
    ----------
    profile_templates : ExternalProfileTemplateSet
        Template set produced by Stage 21.

    Returns
    -------
    ProfileDiagnosticsSet
        One :class:`ProfileDiagnostics` entry per order in *profile_templates*.
    """
    diags = [
        _compute_single_profile_diagnostics(t)
        for t in profile_templates.templates
    ]
    return ProfileDiagnosticsSet(mode=profile_templates.mode, diagnostics=diags)


# ===========================================================================
# 2. External vs empirical extraction comparison
# ===========================================================================


@dataclass
class ExternalVsEmpiricalDiagnostics:
    """Per-order comparison metrics between external and empirical extraction.

    Parameters
    ----------
    order : int
        Echelle order number.
    flux_l2_difference : float
        Normalised L2 (RMS) difference between the external and empirical
        extracted flux vectors:
        ``||flux_ext - flux_emp|| / max(||flux_emp||, 1e-30)``.
        ``float('nan')`` when neither vector has finite values.
    median_abs_flux_difference : float
        Median absolute difference between finite pairs of the two flux
        vectors.  ``float('nan')`` when no finite pairs exist.
    finite_fraction_flux : float
        Fraction of wavelength bins where the *external* flux is finite.
    finite_fraction_variance : float
        Fraction of wavelength bins where the *external* variance is finite.
    external_profile_applied : bool
        ``True`` when the external profile template was actually used for
        this order (as reported by
        :attr:`~.external_profile_extraction.ExternalProfileExtractionResult.external_profile_applied`).
    template_n_frames_used : int
        Number of calibration frames used to build the template (``0`` if
        fallback was used).
    """

    order: int
    flux_l2_difference: float
    median_abs_flux_difference: float
    finite_fraction_flux: float
    finite_fraction_variance: float
    external_profile_applied: bool
    template_n_frames_used: int


@dataclass
class ExternalVsEmpiricalDiagnosticsSet:
    """Collection of :class:`ExternalVsEmpiricalDiagnostics` for all orders.

    Parameters
    ----------
    mode : str
        iSHELL observing mode string.
    diagnostics : list of :class:`ExternalVsEmpiricalDiagnostics`
        One entry per order.
    """

    mode: str
    diagnostics: List[ExternalVsEmpiricalDiagnostics] = field(
        default_factory=list
    )

    @property
    def orders(self) -> list[int]:
        """List of order numbers."""
        return [d.order for d in self.diagnostics]

    @property
    def n_orders(self) -> int:
        """Number of orders."""
        return len(self.diagnostics)

    def get_order(self, order: int) -> ExternalVsEmpiricalDiagnostics:
        """Return the :class:`ExternalVsEmpiricalDiagnostics` for *order*.

        Raises
        ------
        KeyError
            If *order* is not present.
        """
        for d in self.diagnostics:
            if d.order == order:
                return d
        raise KeyError(
            f"Order {order} not found in ExternalVsEmpiricalDiagnosticsSet "
            f"(available: {self.orders})."
        )


def compare_external_vs_empirical(
    rectified_orders: RectifiedOrderSet,
    extraction_def: WeightedExtractionDefinition,
    profile_templates: ExternalProfileTemplateSet,
    *,
    fallback_profile_source: Optional[str] = None,
    variance_image: Optional[npt.NDArray] = None,
    variance_model: Optional[VarianceModelDefinition] = None,
    mask: Optional[npt.NDArray] = None,
) -> ExternalVsEmpiricalDiagnosticsSet:
    """Compare external-profile extraction against empirical extraction.

    Runs extraction twice — once using external profile templates (Stage 22)
    and once with ``profile_source="empirical"`` — and returns per-order
    comparison metrics.

    This function does **not** change any extraction behaviour; its sole
    purpose is to produce diagnostic numbers.

    Parameters
    ----------
    rectified_orders : RectifiedOrderSet
        Science-frame rectified orders (Stage 7 output).
    extraction_def : WeightedExtractionDefinition
        Aperture / background / variance parameters.  ``profile_source``
        and ``external_profile`` are overridden internally.
    profile_templates : ExternalProfileTemplateSet
        Stage-21 template set to use for the external extraction.
    fallback_profile_source : str or None, optional
        Passed to :func:`~.external_profile_extraction.extract_with_external_profile`
        for orders without a template.
    variance_image : ndarray or None, optional
        Per-pixel variance array.
    variance_model : VarianceModelDefinition or None, optional
        Stage-14 variance model.
    mask : ndarray of bool or None, optional
        Bad-pixel mask.

    Returns
    -------
    ExternalVsEmpiricalDiagnosticsSet
        One :class:`ExternalVsEmpiricalDiagnostics` entry per order in
        *rectified_orders*.
    """
    # ------------------------------------------------------------------
    # Run external-profile extraction (Stage 22).
    # ------------------------------------------------------------------
    ext_result = extract_with_external_profile(
        rectified_orders,
        extraction_def,
        profile_templates,
        fallback_profile_source=fallback_profile_source,
        variance_image=variance_image,
        variance_model=variance_model,
        mask=mask,
    )

    # ------------------------------------------------------------------
    # Run empirical extraction for comparison.
    # ------------------------------------------------------------------
    emp_def = dataclasses.replace(
        extraction_def,
        profile_source="empirical",
        external_profile=None,
    )
    emp_result = extract_weighted_optimal(
        rectified_orders,
        emp_def,
        variance_image=variance_image,
        variance_model=variance_model,
        mask=mask,
    )

    # ------------------------------------------------------------------
    # Build per-order comparison diagnostics.
    # ------------------------------------------------------------------
    diags = []
    for order in rectified_orders.orders:
        ext_sp = ext_result.spectra.get_order(order)
        emp_sp = emp_result.get_order(order)

        flux_ext = np.asarray(ext_sp.flux, dtype=float)
        flux_emp = np.asarray(emp_sp.flux, dtype=float)
        var_ext = np.asarray(ext_sp.variance, dtype=float)

        n = flux_ext.size

        # finite fraction of external flux / variance
        ff_flux = float(np.sum(np.isfinite(flux_ext))) / n if n > 0 else 0.0
        ff_var = float(np.sum(np.isfinite(var_ext))) / n if n > 0 else 0.0

        # pairs that are finite in both arrays
        both_finite = np.isfinite(flux_ext) & np.isfinite(flux_emp)
        if np.any(both_finite):
            diff = flux_ext[both_finite] - flux_emp[both_finite]
            norm = max(float(np.linalg.norm(flux_emp[both_finite])), 1e-30)
            l2_diff = float(np.linalg.norm(diff)) / norm
            med_abs = float(np.median(np.abs(diff)))
        else:
            l2_diff = float("nan")
            med_abs = float("nan")

        diags.append(
            ExternalVsEmpiricalDiagnostics(
                order=order,
                flux_l2_difference=l2_diff,
                median_abs_flux_difference=med_abs,
                finite_fraction_flux=ff_flux,
                finite_fraction_variance=ff_var,
                external_profile_applied=ext_result.external_profile_applied[order],
                template_n_frames_used=ext_result.template_n_frames_used[order],
            )
        )

    return ExternalVsEmpiricalDiagnosticsSet(
        mode=rectified_orders.mode, diagnostics=diags
    )


# ===========================================================================
# 3. Template leakage detection
# ===========================================================================


@dataclass
class TemplateLeakageDiagnostics:
    """Per-order heuristic metrics for detecting template leakage.

    Template leakage occurs when a profile template was inadvertently
    built from the *same* data being extracted, rather than from an
    independent calibration frame.  These metrics quantify the similarity
    between the empirical profile (derived from the science frame) and the
    external template.

    .. warning::
        All fields are **heuristic**.  A positive leakage flag does not
        prove leakage; it indicates that the two profiles are unusually
        similar.  A negative flag does not certify the template is
        independent.

    Parameters
    ----------
    order : int
        Echelle order number.
    profile_correlation : float
        Pearson correlation coefficient between the empirical profile
        (from the science frame) and the external template profile,
        computed on the collapsed 1-D (per-row median) representations.
        ``float('nan')`` when fewer than two finite paired values exist.
    profile_l2_difference : float
        Shape-normalised L2 difference between the two collapsed profiles.
        Both profiles are normalised to unit L2 norm before subtraction so
        that scale differences (e.g. column-sum normalisation of the template
        vs. raw science-frame flux units) do not inflate the metric.  A value
        near 0 indicates identical profile shapes.
        ``float('nan')`` when no finite paired values exist.
    flux_image_correlation : float
        Pearson correlation coefficient between the science aperture flux
        image (flattened) and the template-derived model image, restricted
        to the aperture rows defined in *extraction_def*.

        The model is built as ``outer(profile, colspec)`` where
        ``colspec[j] = nansum(flux_ap[:, j])`` is the per-column summed
        flux — a minimal but column-varying proxy for the extracted 1-D
        spectrum.  This makes the model physically faithful in the sense
        that each spectral column is scaled independently rather than by
        a single scalar.

        ``float('nan')`` when fewer than two finite pairs exist.
    possible_template_leakage : bool
        Heuristic flag set to ``True`` when *both* of the following
        hold:

        - ``profile_correlation > LEAKAGE_CORRELATION_THRESHOLD``
          (default :data:`LEAKAGE_CORRELATION_THRESHOLD`)
        - ``profile_l2_difference < LEAKAGE_L2_THRESHOLD``
          (default :data:`LEAKAGE_L2_THRESHOLD`)

        This flag is **informational only** and must not be used to
        automatically reject templates or change extraction behaviour.
    """

    order: int
    profile_correlation: float
    profile_l2_difference: float
    flux_image_correlation: float
    possible_template_leakage: bool


@dataclass
class TemplateLeakageDiagnosticsSet:
    """Collection of :class:`TemplateLeakageDiagnostics` for all orders.

    Parameters
    ----------
    mode : str
        iSHELL observing mode string.
    diagnostics : list of :class:`TemplateLeakageDiagnostics`
        One entry per order.
    """

    mode: str
    diagnostics: List[TemplateLeakageDiagnostics] = field(default_factory=list)

    @property
    def orders(self) -> list[int]:
        """List of order numbers."""
        return [d.order for d in self.diagnostics]

    @property
    def n_orders(self) -> int:
        """Number of orders."""
        return len(self.diagnostics)

    def get_order(self, order: int) -> TemplateLeakageDiagnostics:
        """Return the :class:`TemplateLeakageDiagnostics` for *order*.

        Raises
        ------
        KeyError
            If *order* is not present.
        """
        for d in self.diagnostics:
            if d.order == order:
                return d
        raise KeyError(
            f"Order {order} not found in TemplateLeakageDiagnosticsSet "
            f"(available: {self.orders})."
        )


def _pearson_correlation(a: npt.NDArray, b: npt.NDArray) -> float:
    """Compute Pearson correlation between finite pairs in *a* and *b*.

    Returns ``float('nan')`` when fewer than two finite paired values exist
    or when either vector has zero variance.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return float("nan")
    a_m = a[mask]
    b_m = b[mask]
    a_std = float(np.std(a_m))
    b_std = float(np.std(b_m))
    if a_std < 1e-30 or b_std < 1e-30:
        return float("nan")
    return float(np.corrcoef(a_m, b_m)[0, 1])


def _collapsed_profile(profile: npt.NDArray) -> npt.NDArray:
    """Return per-row median of a 2-D profile, or the array itself if 1-D."""
    if profile.ndim == 1:
        return np.asarray(profile, dtype=float)
    with np.errstate(all="ignore"):
        return np.nanmedian(profile, axis=1).astype(float)


def _compute_aperture_rows(
    n_spatial: int,
    spatial_frac: npt.NDArray,
    center_frac: float,
    radius_frac: float,
) -> npt.NDArray:
    """Return boolean mask for aperture rows.

    Parameters
    ----------
    n_spatial : int
        Total number of spatial rows.
    spatial_frac : ndarray, shape (n_spatial,)
        Fractional slit positions.
    center_frac : float
        Aperture centre in fractional slit coordinates.
    radius_frac : float
        Aperture half-width in fractional slit coordinates.

    Returns
    -------
    ndarray of bool, shape (n_spatial,)
    """
    lo = center_frac - radius_frac
    hi = center_frac + radius_frac
    return (spatial_frac >= lo) & (spatial_frac <= hi)


def _compute_single_leakage_diagnostics(
    order: int,
    science_flux: npt.NDArray,
    science_spatial_frac: npt.NDArray,
    template: ExternalProfileTemplate,
    extraction_def: WeightedExtractionDefinition,
) -> TemplateLeakageDiagnostics:
    """Compute leakage metrics for a single order.

    Parameters
    ----------
    order : int
        Echelle order number.
    science_flux : ndarray, shape (n_spatial, n_spectral)
        Science-frame rectified flux image for this order.
    science_spatial_frac : ndarray, shape (n_spatial,)
        Fractional slit positions for this order.
    template : ExternalProfileTemplate
        External profile template for this order.
    extraction_def : WeightedExtractionDefinition
        Used to determine the aperture region.

    Returns
    -------
    TemplateLeakageDiagnostics
    """
    # ------------------------------------------------------------------ #
    # Build empirical profile from science frame (per-row median).
    # ------------------------------------------------------------------ #
    science_flux = np.asarray(science_flux, dtype=float)
    n_spatial = science_flux.shape[0]
    spatial_frac = np.asarray(science_spatial_frac, dtype=float)

    # Aperture row mask.
    ap_mask = _compute_aperture_rows(
        n_spatial,
        spatial_frac,
        extraction_def.center_frac,
        extraction_def.radius_frac,
    )
    flux_ap = science_flux[ap_mask, :]

    with np.errstate(all="ignore"):
        emp_profile_ap = np.nanmedian(flux_ap, axis=1)  # shape (n_ap_rows,)

    # Collapsed external template restricted to aperture rows.
    ext_profile_full = _collapsed_profile(template.profile)
    ext_profile_ap = ext_profile_full[ap_mask]

    # ------------------------------------------------------------------ #
    # profile_correlation
    # ------------------------------------------------------------------ #
    profile_corr = _pearson_correlation(emp_profile_ap, ext_profile_ap)

    # ------------------------------------------------------------------ #
    # profile_l2_difference (shape comparison after unit-L2 normalisation)
    #
    # Both profiles are normalised to unit L2 norm before computing the
    # difference so that scale differences (e.g. between a raw empirical
    # profile and a column-sum-normalised template) do not inflate the
    # metric.  A value near 0 means the two profiles have the same shape.
    # ------------------------------------------------------------------ #
    both_fin = np.isfinite(emp_profile_ap) & np.isfinite(ext_profile_ap)
    if np.any(both_fin):
        emp_vec = emp_profile_ap[both_fin]
        ext_vec = ext_profile_ap[both_fin]
        norm_emp = max(float(np.linalg.norm(emp_vec)), 1e-30)
        norm_ext = max(float(np.linalg.norm(ext_vec)), 1e-30)
        emp_unit = emp_vec / norm_emp
        ext_unit = ext_vec / norm_ext
        l2_diff = float(np.linalg.norm(ext_unit - emp_unit))
    else:
        l2_diff = float("nan")

    # ------------------------------------------------------------------ #
    # flux_image_correlation
    # ------------------------------------------------------------------ #
    # Build a per-column extracted-spectrum estimate (shape (n_spectral,))
    # as the nansum of the aperture rows for each spectral column.  This
    # is a minimal but physically faithful proxy for the 1-D extracted
    # spectrum: the full model image is outer(profile, colspec), which
    # varies column-by-column rather than being constant across wavelength.
    with np.errstate(all="ignore"):
        colspec = np.nansum(flux_ap, axis=0)  # shape (n_spectral,)
    model_2d = np.outer(ext_profile_ap, colspec)  # (n_ap_rows, n_spectral)
    flux_img_corr = _pearson_correlation(flux_ap.ravel(), model_2d.ravel())

    # ------------------------------------------------------------------ #
    # possible_template_leakage flag
    # ------------------------------------------------------------------ #
    if np.isfinite(profile_corr) and np.isfinite(l2_diff):
        possible_leakage = bool(
            profile_corr > LEAKAGE_CORRELATION_THRESHOLD
            and l2_diff < LEAKAGE_L2_THRESHOLD
        )
    else:
        possible_leakage = False

    return TemplateLeakageDiagnostics(
        order=order,
        profile_correlation=profile_corr,
        profile_l2_difference=l2_diff,
        flux_image_correlation=flux_img_corr,
        possible_template_leakage=possible_leakage,
    )


def compute_leakage_diagnostics(
    rectified_orders: RectifiedOrderSet,
    profile_templates: ExternalProfileTemplateSet,
    extraction_def: WeightedExtractionDefinition,
) -> TemplateLeakageDiagnosticsSet:
    """Compute heuristic template-leakage metrics for all orders.

    Compares the empirical spatial profile of the science frame against
    the external template and flags orders where the two are suspiciously
    similar.

    .. warning::
        This is a **heuristic diagnostic only**.  A positive
        ``possible_template_leakage`` flag does not prove that leakage
        occurred.  A negative flag does not certify independence.

    Parameters
    ----------
    rectified_orders : RectifiedOrderSet
        Science-frame rectified orders.
    profile_templates : ExternalProfileTemplateSet
        External template set to compare against.
    extraction_def : WeightedExtractionDefinition
        Defines the aperture region used for profile comparison.

    Returns
    -------
    TemplateLeakageDiagnosticsSet
        One :class:`TemplateLeakageDiagnostics` per order that appears
        in *both* ``rectified_orders`` and ``profile_templates``.
        Orders present in ``rectified_orders`` but missing from
        ``profile_templates`` are silently skipped.
    """
    template_orders = set(profile_templates.orders)
    diags: list[TemplateLeakageDiagnostics] = []

    for ro in rectified_orders.rectified_orders:
        if ro.order not in template_orders:
            continue
        template = profile_templates.get_order(ro.order)
        diag = _compute_single_leakage_diagnostics(
            order=ro.order,
            science_flux=ro.flux,
            science_spatial_frac=ro.spatial_frac,
            template=template,
            extraction_def=extraction_def,
        )
        diags.append(diag)

    return TemplateLeakageDiagnosticsSet(
        mode=rectified_orders.mode, diagnostics=diags
    )


# ===========================================================================
# 4. Combined diagnostics wrapper
# ===========================================================================


@dataclass
class FullProfileDiagnosticsResult:
    """Result of :func:`run_full_profile_diagnostics`.

    Parameters
    ----------
    template_diagnostics : ProfileDiagnosticsSet
        Per-order template quality metrics.
    comparison_diagnostics : ExternalVsEmpiricalDiagnosticsSet
        Per-order comparison between external and empirical extraction.
    leakage_diagnostics : TemplateLeakageDiagnosticsSet
        Per-order heuristic template-leakage metrics.
    """

    template_diagnostics: ProfileDiagnosticsSet
    comparison_diagnostics: ExternalVsEmpiricalDiagnosticsSet
    leakage_diagnostics: TemplateLeakageDiagnosticsSet


def run_full_profile_diagnostics(
    rectified_orders: RectifiedOrderSet,
    extraction_def: WeightedExtractionDefinition,
    profile_templates: ExternalProfileTemplateSet,
    *,
    fallback_profile_source: Optional[str] = None,
    variance_image: Optional[npt.NDArray] = None,
    variance_model: Optional[VarianceModelDefinition] = None,
    mask: Optional[npt.NDArray] = None,
) -> FullProfileDiagnosticsResult:
    """Run all profile diagnostics and return a combined result.

    Convenience wrapper that calls:

    1. :func:`compute_profile_diagnostics` — template quality metrics.
    2. :func:`compare_external_vs_empirical` — external vs empirical
       comparison metrics.
    3. :func:`compute_leakage_diagnostics` — heuristic leakage metrics.

    Parameters
    ----------
    rectified_orders : RectifiedOrderSet
        Science-frame rectified orders.
    extraction_def : WeightedExtractionDefinition
        Aperture / background / variance parameters.
    profile_templates : ExternalProfileTemplateSet
        Stage-21 template set.
    fallback_profile_source : str or None, optional
        Fallback profile source for orders without a template (passed
        through to :func:`compare_external_vs_empirical`).
    variance_image : ndarray or None, optional
        Per-pixel variance array.
    variance_model : VarianceModelDefinition or None, optional
        Stage-14 variance model.
    mask : ndarray of bool or None, optional
        Bad-pixel mask.

    Returns
    -------
    FullProfileDiagnosticsResult
        Contains all three diagnostic sets.
    """
    tmpl_diags = compute_profile_diagnostics(profile_templates)

    cmp_diags = compare_external_vs_empirical(
        rectified_orders,
        extraction_def,
        profile_templates,
        fallback_profile_source=fallback_profile_source,
        variance_image=variance_image,
        variance_model=variance_model,
        mask=mask,
    )

    leak_diags = compute_leakage_diagnostics(
        rectified_orders,
        profile_templates,
        extraction_def,
    )

    return FullProfileDiagnosticsResult(
        template_diagnostics=tmpl_diags,
        comparison_diagnostics=cmp_diags,
        leakage_diagnostics=leak_diags,
    )
