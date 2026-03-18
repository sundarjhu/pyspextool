"""
Diagnostic reporting and warning layer for iSHELL 2DXD profile diagnostics.

This module implements **Stage 24** of the iSHELL 2DXD reduction scaffold:
a human-readable reporting and warning layer that converts the numeric metrics
produced by Stage 23 (``profile_diagnostics.py``) into structured warnings.

.. important::
    **Warnings are informational and do not imply automatic failure.**

    This module does **not** change extraction behaviour, reject templates,
    switch fallback profile sources, or introduce any decision logic.
    All thresholds are heuristic constants documented below and are
    *not* tuned for science use.

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
   * - 23
     - ``profile_diagnostics.py``
     - Template quality diagnostics
   * - **24**
     - **``profile_warnings.py``**
     - **Diagnostic reporting and warning layer (this module)**

Public API
----------
- :class:`ProfileWarning`
- :class:`ProfileWarningSet`
- :func:`generate_profile_warnings`
- :func:`format_profile_warnings`
- :class:`DiagnosticsWithWarnings`
- :func:`run_diagnostics_with_warnings`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .profile_diagnostics import (
    ExternalVsEmpiricalDiagnosticsSet,
    FullProfileDiagnosticsResult,
    ProfileDiagnosticsSet,
    RectifiedOrderSet,
    TemplateLeakageDiagnosticsSet,
    run_full_profile_diagnostics,
)
from .profile_templates import ExternalProfileTemplateSet
from .variance_model import VarianceModelDefinition
from .weighted_optimal_extraction import WeightedExtractionDefinition

import numpy.typing as npt

__all__ = [
    "ProfileWarning",
    "ProfileWarningSet",
    "generate_profile_warnings",
    "format_profile_warnings",
    "DiagnosticsWithWarnings",
    "run_diagnostics_with_warnings",
]

# ---------------------------------------------------------------------------
# Warning thresholds (heuristic, diagnostic only)
#
# These are simple scalar constants intended for human-inspection use only.
# They are NOT calibrated for science quality and must NOT be used for
# autonomous pipeline decisions.
# ---------------------------------------------------------------------------

#: Minimum acceptable finite-pixel fraction for a template profile.
#: Templates below this threshold may have significant NaN contamination.
THRESHOLD_LOW_FINITE_FRACTION: float = 0.9

#: Roughness value above which a template is considered spatially noisy.
#: Roughness is the mean absolute difference between adjacent spatial values
#: in the collapsed 1-D profile.
THRESHOLD_HIGH_ROUGHNESS: float = 0.05

#: Normalised L2 difference between external and empirical flux vectors
#: above which the two extractions are considered substantially different.
THRESHOLD_LARGE_FLUX_DIFFERENCE: float = 0.2

#: Minimum acceptable finite-flux fraction from external extraction.
#: Values below this suggest many bins failed in the external extraction.
THRESHOLD_LOW_FINITE_FLUX: float = 0.8


# ===========================================================================
# 1. Warning data model
# ===========================================================================


@dataclass
class ProfileWarning:
    """A single diagnostic warning for one echelle order.

    Parameters
    ----------
    order : int
        Echelle order number this warning applies to.
    code : str
        Machine-readable warning code.  One of:

        - ``"LOW_FINITE_FRACTION"``
        - ``"NOT_NORMALIZED"``
        - ``"HIGH_ROUGHNESS"``
        - ``"LARGE_FLUX_DIFFERENCE"``
        - ``"LOW_FINITE_FLUX"``
        - ``"POSSIBLE_TEMPLATE_LEAKAGE"``

    severity : str
        Severity level.  One of ``"info"``, ``"warning"``, ``"severe"``.
    message : str
        Human-readable description of the issue.
    metric_value : float or None
        The numeric metric that triggered this warning.  ``None`` when not
        applicable (e.g. boolean flags).
    threshold : float or None
        The threshold value compared against ``metric_value``.  ``None``
        when the trigger is not threshold-based.
    context : dict
        Additional key/value pairs providing context (e.g. order metadata).
    """

    order: int
    code: str
    severity: str
    message: str
    metric_value: Optional[float]
    threshold: Optional[float]
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_severities = {"info", "warning", "severe"}
        if self.severity not in valid_severities:
            raise ValueError(
                f"severity must be one of {valid_severities!r}, "
                f"got {self.severity!r}"
            )


class ProfileWarningSet:
    """Ordered collection of :class:`ProfileWarning` objects.

    Parameters
    ----------
    warnings : list of :class:`ProfileWarning`
        All warnings to include in this set.
    """

    def __init__(self, warnings: List[ProfileWarning]) -> None:
        self._warnings: List[ProfileWarning] = list(warnings)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def warnings(self) -> List[ProfileWarning]:
        """All warnings in insertion order."""
        return list(self._warnings)

    @property
    def orders(self) -> List[int]:
        """Sorted list of unique order numbers that have at least one warning."""
        return sorted({w.order for w in self._warnings})

    @property
    def n_orders(self) -> int:
        """Number of distinct orders that have at least one warning."""
        return len(self.orders)

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def get_order(self, order: int) -> List[ProfileWarning]:
        """Return all warnings for *order*.

        Parameters
        ----------
        order : int
            Echelle order number.

        Returns
        -------
        list of :class:`ProfileWarning`
            May be empty when there are no warnings for that order.
        """
        return [w for w in self._warnings if w.order == order]

    def filter_by_severity(self, severity: str) -> "ProfileWarningSet":
        """Return a new :class:`ProfileWarningSet` containing only warnings
        at *severity*.

        Parameters
        ----------
        severity : str
            One of ``"info"``, ``"warning"``, ``"severe"``.

        Returns
        -------
        ProfileWarningSet
        """
        return ProfileWarningSet([w for w in self._warnings if w.severity == severity])

    def has_severe(self) -> bool:
        """Return ``True`` if any warning has severity ``"severe"``."""
        return any(w.severity == "severe" for w in self._warnings)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._warnings)

    def __repr__(self) -> str:
        return (
            f"ProfileWarningSet("
            f"n_warnings={len(self)}, "
            f"n_orders={self.n_orders}, "
            f"has_severe={self.has_severe()})"
        )


# ===========================================================================
# 2. Warning generation from diagnostics
# ===========================================================================


def generate_profile_warnings(
    profile_diag_set: ProfileDiagnosticsSet,
    comparison_diag_set: Optional[ExternalVsEmpiricalDiagnosticsSet] = None,
    leakage_diag_set: Optional[TemplateLeakageDiagnosticsSet] = None,
) -> ProfileWarningSet:
    """Convert diagnostic metrics into structured :class:`ProfileWarning` objects.

    Rules applied per order (heuristic, documented):

    **Template quality warnings** (from *profile_diag_set*):

    - ``LOW_FINITE_FRACTION`` — ``finite_fraction < 0.9``
    - ``NOT_NORMALIZED`` — ``is_normalized_like is False``
    - ``HIGH_ROUGHNESS`` — ``roughness > 0.05``

    **External vs empirical warnings** (from *comparison_diag_set*):

    - ``LARGE_FLUX_DIFFERENCE`` — ``flux_l2_difference > 0.2``
    - ``LOW_FINITE_FLUX`` — ``finite_fraction_flux < 0.8``

    **Leakage warnings** (from *leakage_diag_set*):

    - ``POSSIBLE_TEMPLATE_LEAKAGE`` — ``possible_template_leakage is True``

    Parameters
    ----------
    profile_diag_set : ProfileDiagnosticsSet
        Per-order template quality metrics from Stage 23.
    comparison_diag_set : ExternalVsEmpiricalDiagnosticsSet or None, optional
        Per-order external vs empirical comparison metrics.  When ``None``
        the corresponding warnings are skipped.
    leakage_diag_set : TemplateLeakageDiagnosticsSet or None, optional
        Per-order leakage heuristic metrics.  When ``None`` the corresponding
        warnings are skipped.

    Returns
    -------
    ProfileWarningSet
        All generated warnings across all orders.
    """
    all_warnings: List[ProfileWarning] = []

    # ------------------------------------------------------------------
    # Template quality warnings
    # ------------------------------------------------------------------
    for diag in profile_diag_set.diagnostics:
        order = diag.order

        # LOW_FINITE_FRACTION
        if diag.finite_fraction < THRESHOLD_LOW_FINITE_FRACTION:
            all_warnings.append(
                ProfileWarning(
                    order=order,
                    code="LOW_FINITE_FRACTION",
                    severity="warning",
                    message=(
                        f"Template has low finite-pixel fraction "
                        f"(finite_fraction={diag.finite_fraction:.3f}, "
                        f"threshold={THRESHOLD_LOW_FINITE_FRACTION}). "
                        "Significant NaN contamination may degrade extraction."
                    ),
                    metric_value=diag.finite_fraction,
                    threshold=THRESHOLD_LOW_FINITE_FRACTION,
                    context={"n_frames_used": diag.n_frames_used},
                )
            )

        # NOT_NORMALIZED
        if not diag.is_normalized_like:
            all_warnings.append(
                ProfileWarning(
                    order=order,
                    code="NOT_NORMALIZED",
                    severity="warning",
                    message=(
                        f"Template column sums deviate from 1 "
                        f"(colsum_median={diag.colsum_median:.4f}). "
                        "Profile may not be column-normalized."
                    ),
                    metric_value=diag.colsum_median,
                    threshold=None,
                    context={
                        "colsum_min": diag.colsum_min,
                        "colsum_max": diag.colsum_max,
                        "colsum_median": diag.colsum_median,
                    },
                )
            )

        # HIGH_ROUGHNESS
        if not math.isnan(diag.roughness) and diag.roughness > THRESHOLD_HIGH_ROUGHNESS:
            all_warnings.append(
                ProfileWarning(
                    order=order,
                    code="HIGH_ROUGHNESS",
                    severity="info",
                    message=(
                        f"Template spatial profile is rough "
                        f"(roughness={diag.roughness:.4f}, "
                        f"threshold={THRESHOLD_HIGH_ROUGHNESS}). "
                        "Consider using more calibration frames."
                    ),
                    metric_value=diag.roughness,
                    threshold=THRESHOLD_HIGH_ROUGHNESS,
                    context={"n_frames_used": diag.n_frames_used},
                )
            )

    # ------------------------------------------------------------------
    # External vs empirical warnings
    # ------------------------------------------------------------------
    if comparison_diag_set is not None:
        for diag in comparison_diag_set.diagnostics:
            order = diag.order

            # LARGE_FLUX_DIFFERENCE
            if diag.flux_l2_difference > THRESHOLD_LARGE_FLUX_DIFFERENCE:
                all_warnings.append(
                    ProfileWarning(
                        order=order,
                        code="LARGE_FLUX_DIFFERENCE",
                        severity="warning",
                        message=(
                            f"External and empirical extractions differ substantially "
                            f"(flux_l2_difference={diag.flux_l2_difference:.4f}, "
                            f"threshold={THRESHOLD_LARGE_FLUX_DIFFERENCE}). "
                            "Template may be miscentred or from a different mode."
                        ),
                        metric_value=diag.flux_l2_difference,
                        threshold=THRESHOLD_LARGE_FLUX_DIFFERENCE,
                        context={
                            "external_profile_applied": diag.external_profile_applied,
                            "template_n_frames_used": diag.template_n_frames_used,
                        },
                    )
                )

            # LOW_FINITE_FLUX
            if diag.finite_fraction_flux < THRESHOLD_LOW_FINITE_FLUX:
                all_warnings.append(
                    ProfileWarning(
                        order=order,
                        code="LOW_FINITE_FLUX",
                        severity="warning",
                        message=(
                            f"External extraction produced few finite flux values "
                            f"(finite_fraction_flux={diag.finite_fraction_flux:.3f}, "
                            f"threshold={THRESHOLD_LOW_FINITE_FLUX}). "
                            "Many wavelength bins may have failed."
                        ),
                        metric_value=diag.finite_fraction_flux,
                        threshold=THRESHOLD_LOW_FINITE_FLUX,
                        context={
                            "external_profile_applied": diag.external_profile_applied,
                        },
                    )
                )

    # ------------------------------------------------------------------
    # Leakage warnings
    # ------------------------------------------------------------------
    if leakage_diag_set is not None:
        for diag in leakage_diag_set.diagnostics:
            order = diag.order

            # POSSIBLE_TEMPLATE_LEAKAGE
            if diag.possible_template_leakage:
                all_warnings.append(
                    ProfileWarning(
                        order=order,
                        code="POSSIBLE_TEMPLATE_LEAKAGE",
                        severity="severe",
                        message=(
                            f"Template profile is suspiciously similar to the "
                            f"science frame profile "
                            f"(correlation={diag.profile_correlation:.4f}, "
                            f"l2_diff={diag.profile_l2_difference:.4f}). "
                            "Verify that the template was not built from "
                            "the science data."
                        ),
                        metric_value=diag.profile_correlation,
                        threshold=None,
                        context={
                            "profile_correlation": diag.profile_correlation,
                            "profile_l2_difference": diag.profile_l2_difference,
                        },
                    )
                )

    return ProfileWarningSet(all_warnings)


# ===========================================================================
# 3. Human-readable reporting
# ===========================================================================

#: Mapping from severity string to display prefix.
_SEVERITY_LABELS = {
    "info": "[INFO]",
    "warning": "[WARNING]",
    "severe": "[SEVERE]",
}

#: Ordering for severity (lowest to highest) used when sorting within an order.
_SEVERITY_ORDER = {"info": 0, "warning": 1, "severe": 2}


def format_profile_warnings(warning_set: ProfileWarningSet) -> str:
    """Format a :class:`ProfileWarningSet` as a human-readable multi-line string.

    Output is grouped by order number and, within each order, sorted from
    most severe to least severe.

    Parameters
    ----------
    warning_set : ProfileWarningSet
        The warnings to format.

    Returns
    -------
    str
        Multi-line text.  An empty string is returned when there are no
        warnings.

    Examples
    --------
    ::

        Order 311:
          [WARNING] NOT_NORMALIZED: Template column sums deviate from 1 (colsum_median=1.12)
          [SEVERE] POSSIBLE_TEMPLATE_LEAKAGE: Template profile is suspiciously similar ...

        Order 315:
          [INFO] HIGH_ROUGHNESS: Template spatial profile is rough (roughness=0.0623)
    """
    if not warning_set:
        return ""

    lines: List[str] = []
    for order in warning_set.orders:
        order_warnings = warning_set.get_order(order)
        # Sort: severe first, then warning, then info
        order_warnings_sorted = sorted(
            order_warnings,
            key=lambda w: _SEVERITY_ORDER.get(w.severity, 0),
            reverse=True,
        )
        lines.append(f"Order {order}:")
        for w in order_warnings_sorted:
            label = _SEVERITY_LABELS.get(w.severity, f"[{w.severity.upper()}]")
            lines.append(f"  {label} {w.code}: {w.message}")

    return "\n".join(lines)


# ===========================================================================
# 4. Integration helper
# ===========================================================================


@dataclass
class DiagnosticsWithWarnings:
    """Combined result of :func:`run_diagnostics_with_warnings`.

    Parameters
    ----------
    profile_diagnostics : ProfileDiagnosticsSet
        Per-order template quality metrics (Stage 23).
    comparison_diagnostics : ExternalVsEmpiricalDiagnosticsSet
        Per-order external vs empirical comparison metrics (Stage 23).
    leakage_diagnostics : TemplateLeakageDiagnosticsSet
        Per-order heuristic template-leakage metrics (Stage 23).
    warnings : ProfileWarningSet
        Human-readable warnings generated from the diagnostics (Stage 24).
    """

    profile_diagnostics: ProfileDiagnosticsSet
    comparison_diagnostics: ExternalVsEmpiricalDiagnosticsSet
    leakage_diagnostics: TemplateLeakageDiagnosticsSet
    warnings: ProfileWarningSet


def run_diagnostics_with_warnings(
    rectified_orders: RectifiedOrderSet,
    extraction_def: WeightedExtractionDefinition,
    profile_templates: ExternalProfileTemplateSet,
    *,
    fallback_profile_source: Optional[str] = None,
    variance_image: Optional[npt.NDArray] = None,
    variance_model: Optional[VarianceModelDefinition] = None,
    mask: Optional[npt.NDArray] = None,
) -> DiagnosticsWithWarnings:
    """Run all profile diagnostics and generate warnings.

    This is the **recommended user entry point** for Stage 24.  It calls
    :func:`~.profile_diagnostics.run_full_profile_diagnostics` (Stage 23)
    and then passes the results to :func:`generate_profile_warnings` to
    produce a :class:`ProfileWarningSet`.

    Parameters
    ----------
    rectified_orders : RectifiedOrderSet
        Science-frame rectified orders.
    extraction_def : WeightedExtractionDefinition
        Aperture / background / variance parameters.
    profile_templates : ExternalProfileTemplateSet
        Stage-21 template set.
    fallback_profile_source : str or None, optional
        Fallback profile source for orders without a template.
    variance_image : ndarray or None, optional
        Per-pixel variance array.
    variance_model : VarianceModelDefinition or None, optional
        Stage-14 variance model.
    mask : ndarray of bool or None, optional
        Bad-pixel mask.

    Returns
    -------
    DiagnosticsWithWarnings
        Contains all three diagnostic sets and the generated warnings.
    """
    full_result: FullProfileDiagnosticsResult = run_full_profile_diagnostics(
        rectified_orders,
        extraction_def,
        profile_templates,
        fallback_profile_source=fallback_profile_source,
        variance_image=variance_image,
        variance_model=variance_model,
        mask=mask,
    )

    warnings = generate_profile_warnings(
        full_result.template_diagnostics,
        comparison_diag_set=full_result.comparison_diagnostics,
        leakage_diag_set=full_result.leakage_diagnostics,
    )

    return DiagnosticsWithWarnings(
        profile_diagnostics=full_result.template_diagnostics,
        comparison_diagnostics=full_result.comparison_diagnostics,
        leakage_diagnostics=full_result.leakage_diagnostics,
        warnings=warnings,
    )
