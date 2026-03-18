"""
Tests for profile_warnings.py (Stage 24).

Coverage:

Warning data model (ProfileWarning / ProfileWarningSet):
  - ProfileWarning stores fields correctly
  - ProfileWarning rejects invalid severity
  - ProfileWarningSet.orders returns sorted unique orders
  - ProfileWarningSet.n_orders counts distinct orders
  - ProfileWarningSet.get_order returns correct subset
  - ProfileWarningSet.filter_by_severity returns subset
  - ProfileWarningSet.has_severe() behaves correctly
  - ProfileWarningSet.__len__ returns total warning count

Warning generation — each code triggers correctly:
  - LOW_FINITE_FRACTION triggers on low finite_fraction
  - LOW_FINITE_FRACTION does NOT trigger on clean data
  - NOT_NORMALIZED triggers when is_normalized_like is False
  - NOT_NORMALIZED does NOT trigger for normalized template
  - HIGH_ROUGHNESS triggers on rough profile
  - HIGH_ROUGHNESS does NOT trigger on smooth profile
  - LARGE_FLUX_DIFFERENCE triggers on large l2 difference
  - LARGE_FLUX_DIFFERENCE does NOT trigger on close data
  - LOW_FINITE_FLUX triggers on low finite_fraction_flux
  - LOW_FINITE_FLUX does NOT trigger on high finite_fraction_flux
  - POSSIBLE_TEMPLATE_LEAKAGE triggers on leakage flag
  - POSSIBLE_TEMPLATE_LEAKAGE does NOT trigger when flag is False

Edge cases:
  - empty diagnostics sets → empty warnings
  - multiple warnings per order
  - None comparison / leakage sets → only template warnings generated

Severity tests:
  - has_severe() False when no severe warnings
  - has_severe() True when severe warning present
  - filter_by_severity returns correct subset
  - severity filtering on empty set returns empty set

Formatting:
  - format_profile_warnings produces non-empty string
  - output contains order number
  - output contains warning code
  - output contains severity label
  - format_profile_warnings on empty set returns empty string

Integration test:
  - diagnostics → warnings pipeline via run_diagnostics_with_warnings
  - DiagnosticsWithWarnings fields are all populated
  - result warnings match generate_profile_warnings output
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from pyspextool.instruments.ishell.profile_diagnostics import (
    ExternalVsEmpiricalDiagnostics,
    ExternalVsEmpiricalDiagnosticsSet,
    ProfileDiagnostics,
    ProfileDiagnosticsSet,
    TemplateLeakageDiagnostics,
    TemplateLeakageDiagnosticsSet,
    run_full_profile_diagnostics,
)
from pyspextool.instruments.ishell.profile_templates import (
    ExternalProfileTemplateSet,
    ProfileTemplateDefinition,
    build_external_profile_template,
)
from pyspextool.instruments.ishell.profile_warnings import (
    THRESHOLD_HIGH_ROUGHNESS,
    THRESHOLD_LARGE_FLUX_DIFFERENCE,
    THRESHOLD_LOW_FINITE_FRACTION,
    THRESHOLD_LOW_FINITE_FLUX,
    DiagnosticsWithWarnings,
    ProfileWarning,
    ProfileWarningSet,
    format_profile_warnings,
    generate_profile_warnings,
    run_diagnostics_with_warnings,
)
from pyspextool.instruments.ishell.rectified_orders import (
    RectifiedOrder,
    RectifiedOrderSet,
)
from pyspextool.instruments.ishell.weighted_optimal_extraction import (
    WeightedExtractionDefinition,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers (same structure as test_ishell_profile_diagnostics)
# ---------------------------------------------------------------------------

_N_SPECTRAL = 32
_N_SPATIAL = 20
_MODE = "H1_test"

_ORDER_PARAMS = [
    {"order_number": 311},
    {"order_number": 315},
    {"order_number": 320},
]


def _make_spatial_frac(n_spatial: int = _N_SPATIAL) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_spatial)


def _make_wavelength(idx: int, n_spectral: int = _N_SPECTRAL) -> np.ndarray:
    wav_start = 1.55 + idx * 0.05
    return np.linspace(wav_start, wav_start + 0.04, n_spectral)


def _make_rectified_order_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
    fill_value: float = 2.0,
    mode: str = _MODE,
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.0,
    gaussian_profile: bool = False,
    center_frac: float = 0.5,
    sigma_frac: float = 0.1,
) -> RectifiedOrderSet:
    spatial = _make_spatial_frac(n_spatial)
    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        if gaussian_profile:
            profile_1d = np.exp(
                -0.5 * ((spatial - center_frac) / sigma_frac) ** 2
            )
            profile_1d /= profile_1d.sum()
            flux = np.outer(profile_1d, np.ones(n_spectral)) * fill_value
        else:
            flux = np.full((n_spatial, n_spectral), fill_value)
        if noise_scale > 0.0 and rng is not None:
            flux = flux + rng.normal(0.0, noise_scale, size=(n_spatial, n_spectral))
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=_make_wavelength(idx, n_spectral),
                spatial_frac=spatial.copy(),
                flux=flux,
                source_image_shape=(200, 800),
            )
        )
    return RectifiedOrderSet(
        mode=mode,
        rectified_orders=orders,
        source_image_shape=(200, 800),
    )


def _make_template_set(
    rectified_set: RectifiedOrderSet,
    normalize: bool = True,
    smooth_sigma: float = 0.0,
) -> ExternalProfileTemplateSet:
    definition = ProfileTemplateDefinition(
        combine_method="median",
        normalize_profile=normalize,
        smooth_sigma=smooth_sigma,
    )
    return build_external_profile_template(rectified_set, definition)


def _make_extraction_def(
    center_frac: float = 0.5,
    radius_frac: float = 0.4,
) -> WeightedExtractionDefinition:
    return WeightedExtractionDefinition(
        center_frac=center_frac,
        radius_frac=radius_frac,
    )


# ---------------------------------------------------------------------------
# Factories for mock diagnostic sets
# ---------------------------------------------------------------------------


def _make_profile_diag(
    order: int = 311,
    finite_fraction: float = 1.0,
    is_normalized_like: bool = True,
    roughness: float = 0.01,
    colsum_median: float = 1.0,
    colsum_min: float = 0.99,
    colsum_max: float = 1.01,
    n_frames_used: int = 3,
    peak_spatial_index: int = 10,
    peak_spatial_frac: float = 0.5,
) -> ProfileDiagnostics:
    return ProfileDiagnostics(
        order=order,
        finite_fraction=finite_fraction,
        peak_spatial_index=peak_spatial_index,
        peak_spatial_frac=peak_spatial_frac,
        colsum_min=colsum_min,
        colsum_max=colsum_max,
        colsum_median=colsum_median,
        roughness=roughness,
        n_frames_used=n_frames_used,
        is_normalized_like=is_normalized_like,
    )


def _make_profile_diag_set(diags: list) -> ProfileDiagnosticsSet:
    return ProfileDiagnosticsSet(mode=_MODE, diagnostics=diags)


def _make_comparison_diag(
    order: int = 311,
    flux_l2_difference: float = 0.01,
    median_abs_flux_difference: float = 0.005,
    finite_fraction_flux: float = 1.0,
    finite_fraction_variance: float = 1.0,
    external_profile_applied: bool = True,
    template_n_frames_used: int = 3,
) -> ExternalVsEmpiricalDiagnostics:
    return ExternalVsEmpiricalDiagnostics(
        order=order,
        flux_l2_difference=flux_l2_difference,
        median_abs_flux_difference=median_abs_flux_difference,
        finite_fraction_flux=finite_fraction_flux,
        finite_fraction_variance=finite_fraction_variance,
        external_profile_applied=external_profile_applied,
        template_n_frames_used=template_n_frames_used,
    )


def _make_comparison_diag_set(diags: list) -> ExternalVsEmpiricalDiagnosticsSet:
    return ExternalVsEmpiricalDiagnosticsSet(mode=_MODE, diagnostics=diags)


def _make_leakage_diag(
    order: int = 311,
    profile_correlation: float = 0.5,
    profile_l2_difference: float = 0.3,
    flux_image_correlation: float = 0.5,
    possible_template_leakage: bool = False,
) -> TemplateLeakageDiagnostics:
    return TemplateLeakageDiagnostics(
        order=order,
        profile_correlation=profile_correlation,
        profile_l2_difference=profile_l2_difference,
        flux_image_correlation=flux_image_correlation,
        possible_template_leakage=possible_template_leakage,
    )


def _make_leakage_diag_set(diags: list) -> TemplateLeakageDiagnosticsSet:
    return TemplateLeakageDiagnosticsSet(mode=_MODE, diagnostics=diags)


# ===========================================================================
# 1. Warning data model
# ===========================================================================


class TestProfileWarning:
    """Tests for ProfileWarning dataclass."""

    def test_stores_fields_correctly(self):
        w = ProfileWarning(
            order=42,
            code="NOT_NORMALIZED",
            severity="warning",
            message="test message",
            metric_value=1.5,
            threshold=1.0,
            context={"key": "val"},
        )
        assert w.order == 42
        assert w.code == "NOT_NORMALIZED"
        assert w.severity == "warning"
        assert w.message == "test message"
        assert w.metric_value == 1.5
        assert w.threshold == 1.0
        assert w.context == {"key": "val"}

    def test_none_metric_and_threshold_allowed(self):
        w = ProfileWarning(
            order=10,
            code="NOT_NORMALIZED",
            severity="info",
            message="msg",
            metric_value=None,
            threshold=None,
        )
        assert w.metric_value is None
        assert w.threshold is None

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            ProfileWarning(
                order=10,
                code="X",
                severity="critical",  # invalid
                message="msg",
                metric_value=None,
                threshold=None,
            )

    def test_all_valid_severities_accepted(self):
        for sev in ("info", "warning", "severe"):
            w = ProfileWarning(
                order=1, code="X", severity=sev, message="m",
                metric_value=None, threshold=None,
            )
            assert w.severity == sev


class TestProfileWarningSet:
    """Tests for ProfileWarningSet collection class."""

    def _make_set(self) -> ProfileWarningSet:
        return ProfileWarningSet([
            ProfileWarning(order=311, code="A", severity="info", message="m1",
                           metric_value=None, threshold=None),
            ProfileWarning(order=311, code="B", severity="warning", message="m2",
                           metric_value=1.0, threshold=0.5),
            ProfileWarning(order=315, code="C", severity="severe", message="m3",
                           metric_value=0.99, threshold=None),
        ])

    def test_len(self):
        ws = self._make_set()
        assert len(ws) == 3

    def test_orders_sorted_unique(self):
        ws = self._make_set()
        assert ws.orders == [311, 315]

    def test_n_orders(self):
        ws = self._make_set()
        assert ws.n_orders == 2

    def test_get_order_returns_correct_warnings(self):
        ws = self._make_set()
        order311 = ws.get_order(311)
        assert len(order311) == 2
        codes = {w.code for w in order311}
        assert codes == {"A", "B"}

    def test_get_order_empty_for_missing_order(self):
        ws = self._make_set()
        assert ws.get_order(999) == []

    def test_filter_by_severity(self):
        ws = self._make_set()
        info_ws = ws.filter_by_severity("info")
        assert len(info_ws) == 1
        assert all(w.severity == "info" for w in info_ws.warnings)

        warning_ws = ws.filter_by_severity("warning")
        assert len(warning_ws) == 1

        severe_ws = ws.filter_by_severity("severe")
        assert len(severe_ws) == 1

    def test_filter_by_severity_empty_result(self):
        ws = ProfileWarningSet([
            ProfileWarning(order=1, code="X", severity="info", message="m",
                           metric_value=None, threshold=None),
        ])
        result = ws.filter_by_severity("severe")
        assert len(result) == 0
        assert result.n_orders == 0

    def test_has_severe_true(self):
        ws = self._make_set()
        assert ws.has_severe() is True

    def test_has_severe_false(self):
        ws = ProfileWarningSet([
            ProfileWarning(order=1, code="X", severity="info", message="m",
                           metric_value=None, threshold=None),
            ProfileWarning(order=2, code="Y", severity="warning", message="m",
                           metric_value=None, threshold=None),
        ])
        assert ws.has_severe() is False

    def test_has_severe_empty_set(self):
        ws = ProfileWarningSet([])
        assert ws.has_severe() is False

    def test_warnings_property_returns_copy(self):
        ws = self._make_set()
        lst = ws.warnings
        lst.append(None)  # mutating returned list should not affect internal state
        assert len(ws) == 3

    def test_empty_set(self):
        ws = ProfileWarningSet([])
        assert len(ws) == 0
        assert ws.orders == []
        assert ws.n_orders == 0
        assert ws.has_severe() is False

    def test_repr_contains_counts(self):
        ws = self._make_set()
        r = repr(ws)
        assert "n_warnings=3" in r
        assert "n_orders=2" in r


# ===========================================================================
# 2. Warning generation — individual codes
# ===========================================================================


class TestLowFiniteFraction:
    """LOW_FINITE_FRACTION warning tests."""

    def test_triggers_below_threshold(self):
        diag = _make_profile_diag(order=311, finite_fraction=0.5)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FRACTION" in codes

    def test_does_not_trigger_above_threshold(self):
        diag = _make_profile_diag(order=311, finite_fraction=0.95)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FRACTION" not in codes

    def test_at_threshold_boundary(self):
        # Exactly equal to threshold should NOT trigger (threshold is strict <)
        diag = _make_profile_diag(order=311, finite_fraction=THRESHOLD_LOW_FINITE_FRACTION)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FRACTION" not in codes

    def test_metric_value_matches(self):
        ff = 0.75
        diag = _make_profile_diag(order=311, finite_fraction=ff)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        low_ff = [w for w in ws.warnings if w.code == "LOW_FINITE_FRACTION"]
        assert len(low_ff) == 1
        assert low_ff[0].metric_value == pytest.approx(ff)
        assert low_ff[0].threshold == THRESHOLD_LOW_FINITE_FRACTION
        assert low_ff[0].order == 311
        assert low_ff[0].severity == "warning"


class TestNotNormalized:
    """NOT_NORMALIZED warning tests."""

    def test_triggers_when_not_normalized(self):
        diag = _make_profile_diag(order=311, is_normalized_like=False, colsum_median=5.0)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "NOT_NORMALIZED" in codes

    def test_does_not_trigger_when_normalized(self):
        diag = _make_profile_diag(order=311, is_normalized_like=True)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "NOT_NORMALIZED" not in codes

    def test_warning_contains_colsum_median(self):
        diag = _make_profile_diag(order=311, is_normalized_like=False, colsum_median=2.5)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        not_norm = [w for w in ws.warnings if w.code == "NOT_NORMALIZED"]
        assert len(not_norm) == 1
        assert not_norm[0].metric_value == pytest.approx(2.5)
        assert not_norm[0].severity == "warning"
        assert "colsum_median" in not_norm[0].context


class TestHighRoughness:
    """HIGH_ROUGHNESS warning tests."""

    def test_triggers_above_threshold(self):
        diag = _make_profile_diag(order=311, roughness=0.1)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "HIGH_ROUGHNESS" in codes

    def test_does_not_trigger_below_threshold(self):
        diag = _make_profile_diag(order=311, roughness=0.01)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "HIGH_ROUGHNESS" not in codes

    def test_does_not_trigger_on_nan_roughness(self):
        diag = _make_profile_diag(order=311, roughness=float("nan"))
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "HIGH_ROUGHNESS" not in codes

    def test_metric_value_and_threshold(self):
        roughness_val = 0.15
        diag = _make_profile_diag(order=311, roughness=roughness_val)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        high_rough = [w for w in ws.warnings if w.code == "HIGH_ROUGHNESS"]
        assert len(high_rough) == 1
        assert high_rough[0].metric_value == pytest.approx(roughness_val)
        assert high_rough[0].threshold == THRESHOLD_HIGH_ROUGHNESS
        assert high_rough[0].severity == "info"


class TestLargeFluxDifference:
    """LARGE_FLUX_DIFFERENCE warning tests."""

    def test_triggers_above_threshold(self):
        cmp_diag = _make_comparison_diag(order=311, flux_l2_difference=0.5)
        cds = _make_comparison_diag_set([cmp_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, comparison_diag_set=cds)
        codes = {w.code for w in ws.warnings}
        assert "LARGE_FLUX_DIFFERENCE" in codes

    def test_does_not_trigger_below_threshold(self):
        cmp_diag = _make_comparison_diag(order=311, flux_l2_difference=0.05)
        cds = _make_comparison_diag_set([cmp_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, comparison_diag_set=cds)
        codes = {w.code for w in ws.warnings}
        assert "LARGE_FLUX_DIFFERENCE" not in codes

    def test_skipped_when_comparison_is_none(self):
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, comparison_diag_set=None)
        codes = {w.code for w in ws.warnings}
        assert "LARGE_FLUX_DIFFERENCE" not in codes

    def test_metric_value_and_threshold(self):
        l2 = 0.45
        cmp_diag = _make_comparison_diag(order=311, flux_l2_difference=l2)
        cds = _make_comparison_diag_set([cmp_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, comparison_diag_set=cds)
        large_diff = [w for w in ws.warnings if w.code == "LARGE_FLUX_DIFFERENCE"]
        assert len(large_diff) == 1
        assert large_diff[0].metric_value == pytest.approx(l2)
        assert large_diff[0].threshold == THRESHOLD_LARGE_FLUX_DIFFERENCE
        assert large_diff[0].severity == "warning"


class TestLowFiniteFlux:
    """LOW_FINITE_FLUX warning tests."""

    def test_triggers_below_threshold(self):
        cmp_diag = _make_comparison_diag(order=311, finite_fraction_flux=0.5)
        cds = _make_comparison_diag_set([cmp_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, comparison_diag_set=cds)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FLUX" in codes

    def test_does_not_trigger_above_threshold(self):
        cmp_diag = _make_comparison_diag(order=311, finite_fraction_flux=0.95)
        cds = _make_comparison_diag_set([cmp_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, comparison_diag_set=cds)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FLUX" not in codes

    def test_skipped_when_comparison_is_none(self):
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, comparison_diag_set=None)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FLUX" not in codes

    def test_metric_value_and_threshold(self):
        ff = 0.6
        cmp_diag = _make_comparison_diag(order=311, finite_fraction_flux=ff)
        cds = _make_comparison_diag_set([cmp_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, comparison_diag_set=cds)
        low_ff = [w for w in ws.warnings if w.code == "LOW_FINITE_FLUX"]
        assert len(low_ff) == 1
        assert low_ff[0].metric_value == pytest.approx(ff)
        assert low_ff[0].threshold == THRESHOLD_LOW_FINITE_FLUX
        assert low_ff[0].severity == "warning"


class TestPossibleTemplateLeakage:
    """POSSIBLE_TEMPLATE_LEAKAGE warning tests."""

    def test_triggers_when_leakage_flag_set(self):
        leak_diag = _make_leakage_diag(order=311, possible_template_leakage=True,
                                        profile_correlation=0.99, profile_l2_difference=0.01)
        lds = _make_leakage_diag_set([leak_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, leakage_diag_set=lds)
        codes = {w.code for w in ws.warnings}
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in codes

    def test_does_not_trigger_when_flag_is_false(self):
        leak_diag = _make_leakage_diag(order=311, possible_template_leakage=False)
        lds = _make_leakage_diag_set([leak_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, leakage_diag_set=lds)
        codes = {w.code for w in ws.warnings}
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in codes

    def test_skipped_when_leakage_set_is_none(self):
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, leakage_diag_set=None)
        codes = {w.code for w in ws.warnings}
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in codes

    def test_severity_is_severe(self):
        leak_diag = _make_leakage_diag(order=311, possible_template_leakage=True,
                                        profile_correlation=0.99, profile_l2_difference=0.01)
        lds = _make_leakage_diag_set([leak_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, leakage_diag_set=lds)
        leakage_warnings = [w for w in ws.warnings if w.code == "POSSIBLE_TEMPLATE_LEAKAGE"]
        assert len(leakage_warnings) == 1
        assert leakage_warnings[0].severity == "severe"
        assert leakage_warnings[0].order == 311

    def test_context_contains_correlation_and_l2(self):
        corr = 0.992
        l2 = 0.008
        leak_diag = _make_leakage_diag(order=311, possible_template_leakage=True,
                                        profile_correlation=corr, profile_l2_difference=l2)
        lds = _make_leakage_diag_set([leak_diag])
        pds = _make_profile_diag_set([_make_profile_diag(order=311)])
        ws = generate_profile_warnings(pds, leakage_diag_set=lds)
        leakage_warnings = [w for w in ws.warnings if w.code == "POSSIBLE_TEMPLATE_LEAKAGE"]
        ctx = leakage_warnings[0].context
        assert "profile_correlation" in ctx
        assert "profile_l2_difference" in ctx


# ===========================================================================
# 3. No false positives on clean data
# ===========================================================================


class TestNoFalsePositivesCleanData:
    """Verify that perfectly clean synthetic data produces no warnings."""

    def _clean_profile_diag_set(self) -> ProfileDiagnosticsSet:
        diags = [
            _make_profile_diag(
                order=p["order_number"],
                finite_fraction=1.0,
                is_normalized_like=True,
                roughness=0.001,
                colsum_median=1.0,
            )
            for p in _ORDER_PARAMS
        ]
        return _make_profile_diag_set(diags)

    def _clean_comparison_diag_set(self) -> ExternalVsEmpiricalDiagnosticsSet:
        diags = [
            _make_comparison_diag(
                order=p["order_number"],
                flux_l2_difference=0.01,
                finite_fraction_flux=1.0,
            )
            for p in _ORDER_PARAMS
        ]
        return _make_comparison_diag_set(diags)

    def _clean_leakage_diag_set(self) -> TemplateLeakageDiagnosticsSet:
        diags = [
            _make_leakage_diag(
                order=p["order_number"],
                possible_template_leakage=False,
            )
            for p in _ORDER_PARAMS
        ]
        return _make_leakage_diag_set(diags)

    def test_no_warnings_on_clean_data(self):
        pds = self._clean_profile_diag_set()
        cds = self._clean_comparison_diag_set()
        lds = self._clean_leakage_diag_set()
        ws = generate_profile_warnings(pds, comparison_diag_set=cds, leakage_diag_set=lds)
        assert len(ws) == 0
        assert ws.has_severe() is False


# ===========================================================================
# 4. Multiple warnings per order
# ===========================================================================


class TestMultipleWarningsPerOrder:
    """Verify that multiple issues on the same order produce multiple warnings."""

    def test_multiple_warnings_same_order(self):
        # Order 311 has: low finite_fraction AND not normalized AND high roughness
        diag = _make_profile_diag(
            order=311,
            finite_fraction=0.5,        # triggers LOW_FINITE_FRACTION
            is_normalized_like=False,   # triggers NOT_NORMALIZED
            roughness=0.2,              # triggers HIGH_ROUGHNESS
            colsum_median=3.0,
        )
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        order_311_warnings = ws.get_order(311)
        codes = {w.code for w in order_311_warnings}
        assert "LOW_FINITE_FRACTION" in codes
        assert "NOT_NORMALIZED" in codes
        assert "HIGH_ROUGHNESS" in codes
        assert len(order_311_warnings) == 3

    def test_warnings_across_multiple_orders(self):
        diags = [
            _make_profile_diag(order=311, finite_fraction=0.5),
            _make_profile_diag(order=315, is_normalized_like=False, colsum_median=2.0),
            _make_profile_diag(order=320),  # clean
        ]
        ds = _make_profile_diag_set(diags)
        ws = generate_profile_warnings(ds)
        assert 311 in ws.orders
        assert 315 in ws.orders
        # Order 320 is clean, may or may not be in orders
        order_311 = ws.get_order(311)
        order_315 = ws.get_order(315)
        assert any(w.code == "LOW_FINITE_FRACTION" for w in order_311)
        assert any(w.code == "NOT_NORMALIZED" for w in order_315)


# ===========================================================================
# 5. Severity filtering
# ===========================================================================


class TestSeverityFiltering:
    """Tests for severity filtering behaviour."""

    def _make_mixed_set(self) -> ProfileWarningSet:
        return ProfileWarningSet([
            ProfileWarning(order=311, code="HIGH_ROUGHNESS", severity="info",
                           message="rough", metric_value=0.1, threshold=0.05),
            ProfileWarning(order=315, code="LOW_FINITE_FRACTION", severity="warning",
                           message="low ff", metric_value=0.7, threshold=0.9),
            ProfileWarning(order=320, code="POSSIBLE_TEMPLATE_LEAKAGE", severity="severe",
                           message="leak", metric_value=0.99, threshold=None),
        ])

    def test_filter_info(self):
        ws = self._make_mixed_set()
        info = ws.filter_by_severity("info")
        assert len(info) == 1
        assert info.warnings[0].code == "HIGH_ROUGHNESS"

    def test_filter_warning(self):
        ws = self._make_mixed_set()
        warn = ws.filter_by_severity("warning")
        assert len(warn) == 1
        assert warn.warnings[0].code == "LOW_FINITE_FRACTION"

    def test_filter_severe(self):
        ws = self._make_mixed_set()
        sev = ws.filter_by_severity("severe")
        assert len(sev) == 1
        assert sev.warnings[0].code == "POSSIBLE_TEMPLATE_LEAKAGE"

    def test_has_severe_mixed(self):
        ws = self._make_mixed_set()
        assert ws.has_severe() is True

    def test_has_severe_without_severe(self):
        ws = ProfileWarningSet([
            ProfileWarning(order=1, code="X", severity="info", message="m",
                           metric_value=None, threshold=None),
        ])
        assert ws.has_severe() is False


# ===========================================================================
# 6. Empty diagnostics → empty warnings
# ===========================================================================


class TestEmptyDiagnostics:
    """Empty diagnostic sets should produce empty warning sets."""

    def test_empty_profile_diag_set(self):
        ds = _make_profile_diag_set([])
        ws = generate_profile_warnings(ds)
        assert len(ws) == 0
        assert ws.n_orders == 0

    def test_empty_all_sets(self):
        pds = _make_profile_diag_set([])
        cds = _make_comparison_diag_set([])
        lds = _make_leakage_diag_set([])
        ws = generate_profile_warnings(pds, comparison_diag_set=cds, leakage_diag_set=lds)
        assert len(ws) == 0
        assert ws.has_severe() is False

    def test_only_profile_set_given(self):
        """Comparison and leakage sets are optional; only template warnings generated."""
        diag = _make_profile_diag(order=311, finite_fraction=0.5)
        ds = _make_profile_diag_set([diag])
        ws = generate_profile_warnings(ds)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FRACTION" in codes
        assert "LARGE_FLUX_DIFFERENCE" not in codes
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in codes


# ===========================================================================
# 7. Formatting output
# ===========================================================================


class TestFormatProfileWarnings:
    """Tests for format_profile_warnings."""

    def _make_sample_set(self) -> ProfileWarningSet:
        return ProfileWarningSet([
            ProfileWarning(order=311, code="NOT_NORMALIZED", severity="warning",
                           message="col sums deviate from 1 (median=1.12)",
                           metric_value=1.12, threshold=None),
            ProfileWarning(order=311, code="POSSIBLE_TEMPLATE_LEAKAGE", severity="severe",
                           message="correlation=0.995",
                           metric_value=0.995, threshold=None),
            ProfileWarning(order=315, code="HIGH_ROUGHNESS", severity="info",
                           message="roughness=0.065",
                           metric_value=0.065, threshold=0.05),
        ])

    def test_nonempty_output(self):
        ws = self._make_sample_set()
        text = format_profile_warnings(ws)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_empty_set_returns_empty_string(self):
        ws = ProfileWarningSet([])
        text = format_profile_warnings(ws)
        assert text == ""

    def test_output_contains_order_number(self):
        ws = self._make_sample_set()
        text = format_profile_warnings(ws)
        assert "311" in text
        assert "315" in text

    def test_output_contains_warning_codes(self):
        ws = self._make_sample_set()
        text = format_profile_warnings(ws)
        assert "NOT_NORMALIZED" in text
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in text
        assert "HIGH_ROUGHNESS" in text

    def test_output_contains_severity_labels(self):
        ws = self._make_sample_set()
        text = format_profile_warnings(ws)
        assert "[WARNING]" in text
        assert "[SEVERE]" in text
        assert "[INFO]" in text

    def test_severe_appears_before_warning_within_order(self):
        """Within each order, severe warnings should appear before less severe ones."""
        ws = self._make_sample_set()
        text = format_profile_warnings(ws)
        # For order 311: SEVERE should appear before WARNING in the text
        severe_pos = text.find("[SEVERE]")
        warning_pos = text.find("[WARNING]")
        assert severe_pos < warning_pos, (
            "Expected [SEVERE] to appear before [WARNING] within order 311"
        )

    def test_orders_appear_in_sorted_order(self):
        ws = self._make_sample_set()
        text = format_profile_warnings(ws)
        pos_311 = text.find("Order 311")
        pos_315 = text.find("Order 315")
        assert pos_311 < pos_315

    def test_multiline_output(self):
        ws = self._make_sample_set()
        text = format_profile_warnings(ws)
        lines = text.split("\n")
        assert len(lines) > 1


# ===========================================================================
# 8. Integration test: diagnostics → warnings pipeline
# ===========================================================================


class TestIntegrationDiagnosticsToWarnings:
    """End-to-end tests using real diagnostic infrastructure."""

    def _make_science_ros(self, gaussian: bool = True) -> RectifiedOrderSet:
        return _make_rectified_order_set(gaussian_profile=gaussian)

    def _make_template(self, ros: RectifiedOrderSet) -> ExternalProfileTemplateSet:
        definition = ProfileTemplateDefinition(
            combine_method="median",
            normalize_profile=True,
            smooth_sigma=0.0,
        )
        return build_external_profile_template(ros, definition)

    def test_run_diagnostics_with_warnings_returns_correct_type(self):
        ros = self._make_science_ros()
        tmpl = self._make_template(ros)
        extraction_def = _make_extraction_def()
        result = run_diagnostics_with_warnings(ros, extraction_def, tmpl)
        assert isinstance(result, DiagnosticsWithWarnings)

    def test_all_fields_populated(self):
        ros = self._make_science_ros()
        tmpl = self._make_template(ros)
        extraction_def = _make_extraction_def()
        result = run_diagnostics_with_warnings(ros, extraction_def, tmpl)
        assert result.profile_diagnostics is not None
        assert result.comparison_diagnostics is not None
        assert result.leakage_diagnostics is not None
        assert result.warnings is not None
        assert isinstance(result.warnings, ProfileWarningSet)

    def test_diagnostics_have_correct_order_count(self):
        ros = self._make_science_ros()
        tmpl = self._make_template(ros)
        extraction_def = _make_extraction_def()
        result = run_diagnostics_with_warnings(ros, extraction_def, tmpl)
        assert result.profile_diagnostics.n_orders == len(_ORDER_PARAMS)

    def test_warnings_match_generate_profile_warnings(self):
        """run_diagnostics_with_warnings warnings should match
        generate_profile_warnings called with the same diagnostics."""
        ros = self._make_science_ros()
        tmpl = self._make_template(ros)
        extraction_def = _make_extraction_def()
        result = run_diagnostics_with_warnings(ros, extraction_def, tmpl)

        # Regenerate warnings independently
        expected_ws = generate_profile_warnings(
            result.profile_diagnostics,
            comparison_diag_set=result.comparison_diagnostics,
            leakage_diag_set=result.leakage_diagnostics,
        )
        # Should have the same number of warnings
        assert len(result.warnings) == len(expected_ws)

    def test_clean_gaussian_data_produces_no_false_positives(self):
        """A clean normalized Gaussian template with science from the same data
        may trigger leakage (expected) but should not trigger quality warnings."""
        ros = self._make_science_ros(gaussian=True)
        tmpl = self._make_template(ros)
        extraction_def = _make_extraction_def()
        result = run_diagnostics_with_warnings(ros, extraction_def, tmpl)
        ws = result.warnings

        # No LOW_FINITE_FRACTION, NOT_NORMALIZED, or HIGH_ROUGHNESS on clean data
        template_quality_codes = {"LOW_FINITE_FRACTION", "NOT_NORMALIZED", "HIGH_ROUGHNESS"}
        triggered_codes = {w.code for w in ws.warnings}
        quality_issues = triggered_codes & template_quality_codes
        assert len(quality_issues) == 0, (
            f"Unexpected template quality warnings on clean data: {quality_issues}"
        )

    def test_leakage_flagged_when_same_data(self):
        """When the template is built from the science data itself, leakage
        should be flagged."""
        ros = self._make_science_ros(gaussian=True)
        tmpl = self._make_template(ros)
        extraction_def = _make_extraction_def()
        result = run_diagnostics_with_warnings(ros, extraction_def, tmpl)
        ws = result.warnings

        # At least one leakage warning is expected since template == science
        leakage_codes = {w.code for w in ws.warnings}
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in leakage_codes, (
            "Expected leakage warning when template is built from science data"
        )

    def test_format_warnings_on_integration_result(self):
        """format_profile_warnings should not crash on a real result."""
        ros = self._make_science_ros(gaussian=True)
        tmpl = self._make_template(ros)
        extraction_def = _make_extraction_def()
        result = run_diagnostics_with_warnings(ros, extraction_def, tmpl)
        text = format_profile_warnings(result.warnings)
        assert isinstance(text, str)
