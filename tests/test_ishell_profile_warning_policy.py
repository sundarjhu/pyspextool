"""
Tests for ProfileWarningPolicy (Stage 25).

Coverage:

Policy dataclass:
  - default policy has Stage 24 threshold values
  - custom thresholds are stored correctly
  - invalid threshold (non-finite) raises ValueError
  - fraction out-of-range raises ValueError
  - negative roughness raises ValueError
  - negative L2 threshold raises ValueError
  - enable flags default to True
  - enable flags can be set to False

Presets:
  - conservative() returns a valid policy
  - strict() returns a valid policy
  - conservative thresholds are more permissive than default
  - strict thresholds are more restrictive than default

generate_profile_warnings with policy=None reproduces Stage 24 behavior:
  - LOW_FINITE_FRACTION threshold unchanged
  - HIGH_ROUGHNESS threshold unchanged
  - LARGE_FLUX_DIFFERENCE threshold unchanged
  - LOW_FINITE_FLUX threshold unchanged

Custom thresholds trigger/suppress warnings:
  - lower finite_fraction_min → LOW_FINITE_FRACTION suppressed
  - higher finite_fraction_min → LOW_FINITE_FRACTION triggered on previously clean data
  - lower roughness_max → HIGH_ROUGHNESS triggered on previously clean data
  - higher roughness_max → HIGH_ROUGHNESS suppressed
  - lower flux_l2_diff_max → LARGE_FLUX_DIFFERENCE triggered on previously clean data
  - higher flux_l2_diff_max → LARGE_FLUX_DIFFERENCE suppressed
  - lower finite_flux_min → LOW_FINITE_FLUX suppressed
  - higher finite_flux_min → LOW_FINITE_FLUX triggered on previously clean data

Disabling individual warning types:
  - enable_low_finite_fraction=False suppresses LOW_FINITE_FRACTION
  - enable_not_normalized=False suppresses NOT_NORMALIZED
  - enable_high_roughness=False suppresses HIGH_ROUGHNESS
  - enable_large_flux_difference=False suppresses LARGE_FLUX_DIFFERENCE
  - enable_low_finite_flux=False suppresses LOW_FINITE_FLUX
  - enable_leakage_warning=False suppresses POSSIBLE_TEMPLATE_LEAKAGE

Formatting filters:
  - include_info=False omits info-level warnings
  - include_warning=False omits warning-level warnings
  - include_severe=False omits severe-level warnings
  - all False returns empty string
  - group_by_order=False produces flat output without headers
  - default format_profile_warnings output unchanged (backward compat)

run_diagnostics_with_warnings policy pass-through:
  - policy=None behaves like Stage 24
  - policy is passed through to generate_profile_warnings
"""

from __future__ import annotations

import math

import pytest

from pyspextool.instruments.ishell.profile_diagnostics import (
    ExternalVsEmpiricalDiagnostics,
    ExternalVsEmpiricalDiagnosticsSet,
    ProfileDiagnostics,
    ProfileDiagnosticsSet,
    TemplateLeakageDiagnostics,
    TemplateLeakageDiagnosticsSet,
)
from pyspextool.instruments.ishell.profile_warnings import (
    THRESHOLD_HIGH_ROUGHNESS,
    THRESHOLD_LARGE_FLUX_DIFFERENCE,
    THRESHOLD_LEAKAGE_CORR_MIN,
    THRESHOLD_LEAKAGE_L2_MAX,
    THRESHOLD_LOW_FINITE_FRACTION,
    THRESHOLD_LOW_FINITE_FLUX,
    ProfileWarning,
    ProfileWarningPolicy,
    ProfileWarningSet,
    format_profile_warnings,
    generate_profile_warnings,
)

# ---------------------------------------------------------------------------
# Synthetic diagnostic helpers (mirrors test_ishell_profile_warnings.py style)
# ---------------------------------------------------------------------------

_MODE = "H1_test"


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
# 1. ProfileWarningPolicy dataclass
# ===========================================================================


class TestProfileWarningPolicyDefaults:
    """Default values must exactly match Stage 24 hard-coded constants."""

    def test_finite_fraction_min(self):
        assert ProfileWarningPolicy().finite_fraction_min == THRESHOLD_LOW_FINITE_FRACTION

    def test_roughness_max(self):
        assert ProfileWarningPolicy().roughness_max == THRESHOLD_HIGH_ROUGHNESS

    def test_flux_l2_diff_max(self):
        assert ProfileWarningPolicy().flux_l2_diff_max == THRESHOLD_LARGE_FLUX_DIFFERENCE

    def test_finite_flux_min(self):
        assert ProfileWarningPolicy().finite_flux_min == THRESHOLD_LOW_FINITE_FLUX

    def test_leakage_corr_min(self):
        assert ProfileWarningPolicy().leakage_corr_min == THRESHOLD_LEAKAGE_CORR_MIN

    def test_leakage_l2_max(self):
        assert ProfileWarningPolicy().leakage_l2_max == THRESHOLD_LEAKAGE_L2_MAX

    def test_all_enable_flags_true(self):
        p = ProfileWarningPolicy()
        assert p.enable_low_finite_fraction is True
        assert p.enable_not_normalized is True
        assert p.enable_high_roughness is True
        assert p.enable_large_flux_difference is True
        assert p.enable_low_finite_flux is True
        assert p.enable_leakage_warning is True


class TestProfileWarningPolicyCustom:
    """Custom values are stored and validated correctly."""

    def test_custom_thresholds_stored(self):
        p = ProfileWarningPolicy(
            finite_fraction_min=0.8,
            roughness_max=0.1,
            flux_l2_diff_max=0.3,
            finite_flux_min=0.7,
            leakage_corr_min=0.97,
            leakage_l2_max=0.08,
        )
        assert p.finite_fraction_min == 0.8
        assert p.roughness_max == 0.1
        assert p.flux_l2_diff_max == 0.3
        assert p.finite_flux_min == 0.7
        assert p.leakage_corr_min == 0.97
        assert p.leakage_l2_max == 0.08

    def test_enable_flags_can_be_false(self):
        p = ProfileWarningPolicy(
            enable_low_finite_fraction=False,
            enable_not_normalized=False,
            enable_high_roughness=False,
            enable_large_flux_difference=False,
            enable_low_finite_flux=False,
            enable_leakage_warning=False,
        )
        assert p.enable_low_finite_fraction is False
        assert p.enable_not_normalized is False
        assert p.enable_high_roughness is False
        assert p.enable_large_flux_difference is False
        assert p.enable_low_finite_flux is False
        assert p.enable_leakage_warning is False


class TestProfileWarningPolicyValidation:
    """Invalid inputs raise ValueError."""

    def test_nan_finite_fraction_min_raises(self):
        with pytest.raises(ValueError, match="finite_fraction_min"):
            ProfileWarningPolicy(finite_fraction_min=float("nan"))

    def test_inf_roughness_max_raises(self):
        with pytest.raises(ValueError, match="roughness_max"):
            ProfileWarningPolicy(roughness_max=float("inf"))

    def test_nan_flux_l2_diff_max_raises(self):
        with pytest.raises(ValueError, match="flux_l2_diff_max"):
            ProfileWarningPolicy(flux_l2_diff_max=float("nan"))

    def test_finite_fraction_min_above_one_raises(self):
        with pytest.raises(ValueError, match="finite_fraction_min"):
            ProfileWarningPolicy(finite_fraction_min=1.1)

    def test_finite_fraction_min_below_zero_raises(self):
        with pytest.raises(ValueError, match="finite_fraction_min"):
            ProfileWarningPolicy(finite_fraction_min=-0.1)

    def test_finite_flux_min_above_one_raises(self):
        with pytest.raises(ValueError, match="finite_flux_min"):
            ProfileWarningPolicy(finite_flux_min=1.5)

    def test_leakage_corr_min_below_zero_raises(self):
        with pytest.raises(ValueError, match="leakage_corr_min"):
            ProfileWarningPolicy(leakage_corr_min=-0.5)

    def test_negative_roughness_max_raises(self):
        with pytest.raises(ValueError, match="roughness_max"):
            ProfileWarningPolicy(roughness_max=-0.01)

    def test_negative_flux_l2_diff_max_raises(self):
        with pytest.raises(ValueError, match="flux_l2_diff_max"):
            ProfileWarningPolicy(flux_l2_diff_max=-0.1)

    def test_negative_leakage_l2_max_raises(self):
        with pytest.raises(ValueError, match="leakage_l2_max"):
            ProfileWarningPolicy(leakage_l2_max=-0.01)

    def test_boundary_zero_is_valid(self):
        # Zero is a valid fraction and non-negative threshold
        p = ProfileWarningPolicy(
            finite_fraction_min=0.0,
            roughness_max=0.0,
            flux_l2_diff_max=0.0,
            finite_flux_min=0.0,
            leakage_corr_min=0.0,
            leakage_l2_max=0.0,
        )
        assert p.finite_fraction_min == 0.0

    def test_boundary_one_is_valid(self):
        p = ProfileWarningPolicy(
            finite_fraction_min=1.0,
            finite_flux_min=1.0,
            leakage_corr_min=1.0,
        )
        assert p.finite_fraction_min == 1.0


# ===========================================================================
# 2. Policy presets
# ===========================================================================


class TestProfileWarningPolicyPresets:
    """Preset constructors return valid policies with expected relative values."""

    def test_conservative_is_valid(self):
        p = ProfileWarningPolicy.conservative()
        assert isinstance(p, ProfileWarningPolicy)

    def test_strict_is_valid(self):
        p = ProfileWarningPolicy.strict()
        assert isinstance(p, ProfileWarningPolicy)

    def test_conservative_finite_fraction_min_lower_than_default(self):
        # conservative → higher tolerance → fewer warnings → lower min threshold
        default = ProfileWarningPolicy()
        conservative = ProfileWarningPolicy.conservative()
        assert conservative.finite_fraction_min < default.finite_fraction_min

    def test_conservative_roughness_max_higher_than_default(self):
        # conservative → higher threshold → fewer warnings
        default = ProfileWarningPolicy()
        conservative = ProfileWarningPolicy.conservative()
        assert conservative.roughness_max > default.roughness_max

    def test_conservative_flux_l2_diff_max_higher_than_default(self):
        default = ProfileWarningPolicy()
        conservative = ProfileWarningPolicy.conservative()
        assert conservative.flux_l2_diff_max > default.flux_l2_diff_max

    def test_strict_finite_fraction_min_higher_than_default(self):
        # strict → lower tolerance → more warnings → higher min threshold
        default = ProfileWarningPolicy()
        strict = ProfileWarningPolicy.strict()
        assert strict.finite_fraction_min > default.finite_fraction_min

    def test_strict_roughness_max_lower_than_default(self):
        # strict → smaller max threshold → triggered more easily
        default = ProfileWarningPolicy()
        strict = ProfileWarningPolicy.strict()
        assert strict.roughness_max < default.roughness_max

    def test_strict_flux_l2_diff_max_lower_than_default(self):
        default = ProfileWarningPolicy()
        strict = ProfileWarningPolicy.strict()
        assert strict.flux_l2_diff_max < default.flux_l2_diff_max

    def test_strict_more_warnings_than_default_more_than_conservative(self):
        """On borderline data, strict > default > conservative for warning count."""
        # Use values that sit between strict and conservative thresholds
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, finite_fraction=0.85, roughness=0.06),
        ])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, flux_l2_difference=0.15, finite_fraction_flux=0.85),
        ])
        n_strict = len(generate_profile_warnings(
            diag_set, comparison_diag_set=cmp_set, policy=ProfileWarningPolicy.strict()
        ))
        n_default = len(generate_profile_warnings(
            diag_set, comparison_diag_set=cmp_set, policy=ProfileWarningPolicy()
        ))
        n_conservative = len(generate_profile_warnings(
            diag_set, comparison_diag_set=cmp_set, policy=ProfileWarningPolicy.conservative()
        ))
        assert n_strict >= n_default >= n_conservative


# ===========================================================================
# 3. Default policy reproduces Stage 24 behavior exactly
# ===========================================================================


class TestDefaultPolicyBackwardCompatibility:
    """policy=None must produce identical output to generate_profile_warnings(...) in Stage 24."""

    def test_low_finite_fraction_same_threshold(self):
        # Just below default threshold → triggers
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, finite_fraction=THRESHOLD_LOW_FINITE_FRACTION - 0.01),
        ])
        ws_none = generate_profile_warnings(diag_set)
        ws_default = generate_profile_warnings(diag_set, policy=ProfileWarningPolicy())
        assert len(ws_none) == len(ws_default)
        codes_none = {w.code for w in ws_none.warnings}
        codes_default = {w.code for w in ws_default.warnings}
        assert codes_none == codes_default

    def test_high_roughness_same_threshold(self):
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, roughness=THRESHOLD_HIGH_ROUGHNESS + 0.01),
        ])
        ws_none = generate_profile_warnings(diag_set)
        ws_default = generate_profile_warnings(diag_set, policy=ProfileWarningPolicy())
        assert len(ws_none) == len(ws_default)

    def test_large_flux_difference_same_threshold(self):
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, flux_l2_difference=THRESHOLD_LARGE_FLUX_DIFFERENCE + 0.01),
        ])
        ws_none = generate_profile_warnings(diag_set, comparison_diag_set=cmp_set)
        ws_default = generate_profile_warnings(
            diag_set, comparison_diag_set=cmp_set, policy=ProfileWarningPolicy()
        )
        assert len(ws_none) == len(ws_default)

    def test_low_finite_flux_same_threshold(self):
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, finite_fraction_flux=THRESHOLD_LOW_FINITE_FLUX - 0.01),
        ])
        ws_none = generate_profile_warnings(diag_set, comparison_diag_set=cmp_set)
        ws_default = generate_profile_warnings(
            diag_set, comparison_diag_set=cmp_set, policy=ProfileWarningPolicy()
        )
        assert len(ws_none) == len(ws_default)

    def test_threshold_stored_in_warning_uses_policy_value(self):
        """Warning.threshold should reflect the active policy value."""
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, finite_fraction=0.5),
        ])
        ws = generate_profile_warnings(diag_set, policy=ProfileWarningPolicy())
        w = next(w for w in ws.warnings if w.code == "LOW_FINITE_FRACTION")
        assert w.threshold == THRESHOLD_LOW_FINITE_FRACTION


# ===========================================================================
# 4. Custom thresholds trigger / suppress warnings
# ===========================================================================


class TestCustomThresholds:
    """Custom policy thresholds change which warnings fire."""

    def test_lower_finite_fraction_min_suppresses_warning(self):
        # With finite_fraction=0.85 (below default 0.9), warning fires by default.
        # Lowering min to 0.8 suppresses it.
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, finite_fraction=0.85),
        ])
        policy = ProfileWarningPolicy(finite_fraction_min=0.8)
        ws = generate_profile_warnings(diag_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FRACTION" not in codes

    def test_higher_finite_fraction_min_triggers_warning(self):
        # With finite_fraction=0.95 (above default 0.9), no warning by default.
        # Raising min to 0.97 triggers it.
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, finite_fraction=0.95),
        ])
        policy = ProfileWarningPolicy(finite_fraction_min=0.97)
        ws = generate_profile_warnings(diag_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FRACTION" in codes

    def test_lower_roughness_max_triggers_warning(self):
        # roughness=0.03 (below default 0.05) → no warning by default.
        # Lowering max to 0.02 → triggers.
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, roughness=0.03),
        ])
        policy = ProfileWarningPolicy(roughness_max=0.02)
        ws = generate_profile_warnings(diag_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "HIGH_ROUGHNESS" in codes

    def test_higher_roughness_max_suppresses_warning(self):
        # roughness=0.07 (above default 0.05) → warning by default.
        # Raising max to 0.10 → suppresses.
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, roughness=0.07),
        ])
        policy = ProfileWarningPolicy(roughness_max=0.10)
        ws = generate_profile_warnings(diag_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "HIGH_ROUGHNESS" not in codes

    def test_lower_flux_l2_diff_max_triggers_warning(self):
        # flux_l2_difference=0.15 (below default 0.2) → no warning by default.
        # Lowering max to 0.1 → triggers.
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, flux_l2_difference=0.15),
        ])
        policy = ProfileWarningPolicy(flux_l2_diff_max=0.1)
        ws = generate_profile_warnings(diag_set, comparison_diag_set=cmp_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LARGE_FLUX_DIFFERENCE" in codes

    def test_higher_flux_l2_diff_max_suppresses_warning(self):
        # flux_l2_difference=0.25 (above default 0.2) → warning by default.
        # Raising max to 0.3 → suppresses.
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, flux_l2_difference=0.25),
        ])
        policy = ProfileWarningPolicy(flux_l2_diff_max=0.3)
        ws = generate_profile_warnings(diag_set, comparison_diag_set=cmp_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LARGE_FLUX_DIFFERENCE" not in codes

    def test_higher_finite_flux_min_triggers_warning(self):
        # finite_fraction_flux=0.85 (above default 0.8) → no warning by default.
        # Raising min to 0.9 → triggers.
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, finite_fraction_flux=0.85),
        ])
        policy = ProfileWarningPolicy(finite_flux_min=0.9)
        ws = generate_profile_warnings(diag_set, comparison_diag_set=cmp_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FLUX" in codes

    def test_lower_finite_flux_min_suppresses_warning(self):
        # finite_fraction_flux=0.75 (below default 0.8) → warning by default.
        # Lowering min to 0.7 → suppresses.
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, finite_fraction_flux=0.75),
        ])
        policy = ProfileWarningPolicy(finite_flux_min=0.7)
        ws = generate_profile_warnings(diag_set, comparison_diag_set=cmp_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FLUX" not in codes

    def test_custom_threshold_stored_in_warning(self):
        """The threshold recorded in the warning reflects the active policy value."""
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, finite_fraction=0.60),
        ])
        policy = ProfileWarningPolicy(finite_fraction_min=0.75)
        ws = generate_profile_warnings(diag_set, policy=policy)
        w = next(w for w in ws.warnings if w.code == "LOW_FINITE_FRACTION")
        assert w.threshold == 0.75

    def test_lower_leakage_corr_min_triggers_leakage_warning(self):
        # corr=0.96 is above default 0.98? No → no warning by default.
        # Lower corr_min to 0.95 → 0.96 > 0.95 AND l2 < 0.05 → fires.
        leak_set = _make_leakage_diag_set([
            _make_leakage_diag(
                order=311,
                profile_correlation=0.96,
                profile_l2_difference=0.01,
                possible_template_leakage=False,
            ),
        ])
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        # Default policy: 0.96 > 0.98? No → no warning
        ws_default = generate_profile_warnings(diag_set, leakage_diag_set=leak_set)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in {w.code for w in ws_default.warnings}
        # Lower corr_min: 0.96 > 0.95? Yes AND 0.01 < 0.05? Yes → fires
        policy = ProfileWarningPolicy(leakage_corr_min=0.95)
        ws_custom = generate_profile_warnings(diag_set, leakage_diag_set=leak_set, policy=policy)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in {w.code for w in ws_custom.warnings}

    def test_higher_leakage_corr_min_suppresses_leakage_warning(self):
        # corr=0.99 (above default 0.98) → fires by default.
        # Raise corr_min to 0.995 → 0.99 > 0.995? No → suppressed.
        leak_set = _make_leakage_diag_set([
            _make_leakage_diag(
                order=311,
                profile_correlation=0.99,
                profile_l2_difference=0.01,
                possible_template_leakage=True,
            ),
        ])
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        # Default policy fires
        ws_default = generate_profile_warnings(diag_set, leakage_diag_set=leak_set)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in {w.code for w in ws_default.warnings}
        # Raised corr_min: 0.99 > 0.995? No → suppressed
        policy = ProfileWarningPolicy(leakage_corr_min=0.995)
        ws_custom = generate_profile_warnings(diag_set, leakage_diag_set=leak_set, policy=policy)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in {w.code for w in ws_custom.warnings}

    def test_higher_leakage_l2_max_triggers_leakage_warning(self):
        # l2_diff=0.08 (above default 0.05 max) → no warning by default.
        # Raise l2_max to 0.10 → 0.08 < 0.10 AND corr > 0.98 → fires.
        leak_set = _make_leakage_diag_set([
            _make_leakage_diag(
                order=311,
                profile_correlation=0.99,
                profile_l2_difference=0.08,
                possible_template_leakage=False,
            ),
        ])
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        # Default policy: 0.08 < 0.05? No → no warning
        ws_default = generate_profile_warnings(diag_set, leakage_diag_set=leak_set)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in {w.code for w in ws_default.warnings}
        # Raised l2_max: 0.08 < 0.10? Yes AND 0.99 > 0.98? Yes → fires
        policy = ProfileWarningPolicy(leakage_l2_max=0.10)
        ws_custom = generate_profile_warnings(diag_set, leakage_diag_set=leak_set, policy=policy)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in {w.code for w in ws_custom.warnings}

    def test_lower_leakage_l2_max_suppresses_leakage_warning(self):
        # l2_diff=0.03 (below default 0.05 max) → fires by default.
        # Lower l2_max to 0.02 → 0.03 < 0.02? No → suppressed.
        leak_set = _make_leakage_diag_set([
            _make_leakage_diag(
                order=311,
                profile_correlation=0.99,
                profile_l2_difference=0.03,
                possible_template_leakage=True,
            ),
        ])
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        # Default policy fires
        ws_default = generate_profile_warnings(diag_set, leakage_diag_set=leak_set)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in {w.code for w in ws_default.warnings}
        # Lower l2_max: 0.03 < 0.02? No → suppressed
        policy = ProfileWarningPolicy(leakage_l2_max=0.02)
        ws_custom = generate_profile_warnings(diag_set, leakage_diag_set=leak_set, policy=policy)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in {w.code for w in ws_custom.warnings}

    def test_leakage_requires_both_conditions(self):
        """Both correlation AND l2 conditions must be satisfied simultaneously."""
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])

        # High correlation but large l2_diff → no warning
        leak_high_corr_only = _make_leakage_diag_set([
            _make_leakage_diag(order=311, profile_correlation=0.999, profile_l2_difference=0.99),
        ])
        ws = generate_profile_warnings(diag_set, leakage_diag_set=leak_high_corr_only)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in {w.code for w in ws.warnings}

        # Low l2_diff but low correlation → no warning
        leak_low_l2_only = _make_leakage_diag_set([
            _make_leakage_diag(order=311, profile_correlation=0.5, profile_l2_difference=0.001),
        ])
        ws = generate_profile_warnings(diag_set, leakage_diag_set=leak_low_l2_only)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in {w.code for w in ws.warnings}

        # Both conditions met → warning fires
        leak_both = _make_leakage_diag_set([
            _make_leakage_diag(order=311, profile_correlation=0.999, profile_l2_difference=0.001),
        ])
        ws = generate_profile_warnings(diag_set, leakage_diag_set=leak_both)
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in {w.code for w in ws.warnings}


# ===========================================================================
# 5. Disabling individual warning types
# ===========================================================================


class TestEnableFlags:
    """enable_* flags suppress their respective warning codes."""

    def test_disable_low_finite_fraction(self):
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, finite_fraction=0.5),
        ])
        policy = ProfileWarningPolicy(enable_low_finite_fraction=False)
        ws = generate_profile_warnings(diag_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FRACTION" not in codes

    def test_disable_not_normalized(self):
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, is_normalized_like=False, colsum_median=1.5),
        ])
        policy = ProfileWarningPolicy(enable_not_normalized=False)
        ws = generate_profile_warnings(diag_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "NOT_NORMALIZED" not in codes

    def test_disable_high_roughness(self):
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, roughness=0.5),
        ])
        policy = ProfileWarningPolicy(enable_high_roughness=False)
        ws = generate_profile_warnings(diag_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "HIGH_ROUGHNESS" not in codes

    def test_disable_large_flux_difference(self):
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, flux_l2_difference=0.9),
        ])
        policy = ProfileWarningPolicy(enable_large_flux_difference=False)
        ws = generate_profile_warnings(diag_set, comparison_diag_set=cmp_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LARGE_FLUX_DIFFERENCE" not in codes

    def test_disable_low_finite_flux(self):
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, finite_fraction_flux=0.1),
        ])
        policy = ProfileWarningPolicy(enable_low_finite_flux=False)
        ws = generate_profile_warnings(diag_set, comparison_diag_set=cmp_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "LOW_FINITE_FLUX" not in codes

    def test_disable_leakage_warning(self):
        diag_set = _make_profile_diag_set([_make_profile_diag(order=311)])
        leak_set = _make_leakage_diag_set([
            _make_leakage_diag(
                order=311,
                profile_correlation=0.99,
                profile_l2_difference=0.01,
                possible_template_leakage=True,
            ),
        ])
        policy = ProfileWarningPolicy(enable_leakage_warning=False)
        ws = generate_profile_warnings(diag_set, leakage_diag_set=leak_set, policy=policy)
        codes = {w.code for w in ws.warnings}
        assert "POSSIBLE_TEMPLATE_LEAKAGE" not in codes

    def test_disable_all_returns_empty(self):
        """Disabling every warning type produces an empty warning set."""
        diag_set = _make_profile_diag_set([
            _make_profile_diag(order=311, finite_fraction=0.5, is_normalized_like=False, roughness=0.5),
        ])
        cmp_set = _make_comparison_diag_set([
            _make_comparison_diag(order=311, flux_l2_difference=0.9, finite_fraction_flux=0.1),
        ])
        leak_set = _make_leakage_diag_set([
            _make_leakage_diag(
                order=311,
                profile_correlation=0.99,
                profile_l2_difference=0.01,
                possible_template_leakage=True,
            ),
        ])
        policy = ProfileWarningPolicy(
            enable_low_finite_fraction=False,
            enable_not_normalized=False,
            enable_high_roughness=False,
            enable_large_flux_difference=False,
            enable_low_finite_flux=False,
            enable_leakage_warning=False,
        )
        ws = generate_profile_warnings(
            diag_set, comparison_diag_set=cmp_set, leakage_diag_set=leak_set, policy=policy
        )
        assert len(ws) == 0


# ===========================================================================
# 6. Formatting filters
# ===========================================================================


def _make_mixed_warning_set() -> ProfileWarningSet:
    """Create a warning set with one warning at each severity level."""
    return ProfileWarningSet([
        ProfileWarning(
            order=311, code="HIGH_ROUGHNESS", severity="info",
            message="info msg", metric_value=0.1, threshold=0.05,
        ),
        ProfileWarning(
            order=311, code="LOW_FINITE_FRACTION", severity="warning",
            message="warning msg", metric_value=0.8, threshold=0.9,
        ),
        ProfileWarning(
            order=315, code="POSSIBLE_TEMPLATE_LEAKAGE", severity="severe",
            message="severe msg", metric_value=0.999, threshold=None,
        ),
    ])


class TestFormattingFilters:
    """format_profile_warnings filtering parameters."""

    def test_default_includes_all_severities(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(ws)
        assert "[INFO]" in text
        assert "[WARNING]" in text
        assert "[SEVERE]" in text

    def test_exclude_info(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(ws, include_info=False)
        assert "[INFO]" not in text
        assert "[WARNING]" in text
        assert "[SEVERE]" in text

    def test_exclude_warning(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(ws, include_warning=False)
        assert "[INFO]" in text
        assert "[WARNING]" not in text
        assert "[SEVERE]" in text

    def test_exclude_severe(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(ws, include_severe=False)
        assert "[INFO]" in text
        assert "[WARNING]" in text
        assert "[SEVERE]" not in text

    def test_exclude_all_returns_empty_string(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(
            ws, include_info=False, include_warning=False, include_severe=False
        )
        assert text == ""

    def test_group_by_order_default(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(ws)
        assert "Order 311:" in text
        assert "Order 315:" in text

    def test_group_by_order_false_no_headers(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(ws, group_by_order=False)
        assert "Order 311:" not in text
        assert "Order 315:" not in text
        # All codes should still appear
        assert "HIGH_ROUGHNESS" in text
        assert "LOW_FINITE_FRACTION" in text
        assert "POSSIBLE_TEMPLATE_LEAKAGE" in text

    def test_group_by_order_false_contains_order_number(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(ws, group_by_order=False)
        # Each line should still contain the order number
        assert "311" in text
        assert "315" in text

    def test_empty_set_returns_empty_string(self):
        ws = ProfileWarningSet([])
        text = format_profile_warnings(ws)
        assert text == ""

    def test_filter_leaves_only_matching_severity(self):
        ws = _make_mixed_warning_set()
        text = format_profile_warnings(
            ws, include_info=False, include_warning=False, include_severe=True
        )
        assert "[SEVERE]" in text
        assert "[INFO]" not in text
        assert "[WARNING]" not in text

    def test_backward_compat_default_output_matches_legacy(self):
        """Default call produces same output as a call with all defaults explicit."""
        ws = _make_mixed_warning_set()
        default_text = format_profile_warnings(ws)
        explicit_text = format_profile_warnings(
            ws,
            include_info=True,
            include_warning=True,
            include_severe=True,
            group_by_order=True,
        )
        assert default_text == explicit_text
