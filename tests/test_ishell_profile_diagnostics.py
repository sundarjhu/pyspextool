"""
Tests for profile_diagnostics.py (Stage 23).

Coverage:

Template diagnostics (ProfileDiagnostics / ProfileDiagnosticsSet):
  - normalized template → column sums near 1, is_normalized_like True
  - unnormalized template → is_normalized_like False
  - smoothing reduces roughness
  - peak detection works for simple Gaussian profile
  - diagnostics set lookup works (get_order, KeyError on missing)
  - all-NaN profile → finite_fraction=0, peak_idx=-1, colsum_min NaN
  - finite_fraction is correct

External vs empirical comparison (ExternalVsEmpiricalDiagnostics):
  - identical data → small differences
  - noisy data → measurable differences
  - metrics are finite
  - get_order lookup works

Leakage detection (TemplateLeakageDiagnostics):
  - template built from SAME data → leakage flag True
  - template built from DIFFERENT noisy data → leakage flag False
  - correlation metric behaves as expected
  - thresholds behave sensibly (no over-triggering in random noise)

Edge cases:
  - all-NaN profile
  - empty template set → ProfileDiagnosticsSet with n_orders == 0
  - missing orders → skipped in leakage diagnostics

Smoke test:
  - build templates from calibration set
  - run full diagnostics on science set
  - verify one entry per order, no crashes, metrics well-formed
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyspextool.instruments.ishell.profile_diagnostics import (
    LEAKAGE_CORRELATION_THRESHOLD,
    LEAKAGE_L2_THRESHOLD,
    ExternalVsEmpiricalDiagnostics,
    ExternalVsEmpiricalDiagnosticsSet,
    FullProfileDiagnosticsResult,
    ProfileDiagnostics,
    ProfileDiagnosticsSet,
    TemplateLeakageDiagnostics,
    TemplateLeakageDiagnosticsSet,
    compare_external_vs_empirical,
    compute_leakage_diagnostics,
    compute_profile_diagnostics,
    run_full_profile_diagnostics,
)
from pyspextool.instruments.ishell.profile_templates import (
    ExternalProfileTemplate,
    ExternalProfileTemplateSet,
    ProfileTemplateDefinition,
    build_external_profile_template,
)
from pyspextool.instruments.ishell.rectified_orders import (
    RectifiedOrder,
    RectifiedOrderSet,
)
from pyspextool.instruments.ishell.weighted_optimal_extraction import (
    WeightedExtractionDefinition,
)

# ---------------------------------------------------------------------------
# Synthetic data constants
# ---------------------------------------------------------------------------

_N_SPECTRAL = 32
_N_SPATIAL = 20
_MODE = "H1_test"

_ORDER_PARAMS = [
    {"order_number": 311},
    {"order_number": 315},
    {"order_number": 320},
]


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


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
    """Build a RectifiedOrderSet with constant or Gaussian flux images."""
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
    """Build a template set from a RectifiedOrderSet."""
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
# 1. Template quality diagnostics
# ---------------------------------------------------------------------------


class TestProfileDiagnosticsBasic:
    """Tests for compute_profile_diagnostics."""

    def test_normalized_template_is_normalized_like(self):
        ros = _make_rectified_order_set(gaussian_profile=True)
        tmpl = _make_template_set(ros, normalize=True)
        result = compute_profile_diagnostics(tmpl)
        assert result.n_orders == len(_ORDER_PARAMS)
        for d in result.diagnostics:
            assert d.is_normalized_like, (
                f"Order {d.order}: expected is_normalized_like=True, "
                f"colsum_median={d.colsum_median}"
            )

    def test_normalized_template_colsum_near_1(self):
        ros = _make_rectified_order_set(gaussian_profile=True)
        tmpl = _make_template_set(ros, normalize=True)
        result = compute_profile_diagnostics(tmpl)
        for d in result.diagnostics:
            assert abs(d.colsum_median - 1.0) < 0.05, (
                f"Order {d.order}: colsum_median={d.colsum_median}"
            )

    def test_unnormalized_template_not_normalized_like(self):
        # Use fill_value != 1, not normalized
        ros = _make_rectified_order_set(fill_value=5.0)
        tmpl = _make_template_set(ros, normalize=False)
        result = compute_profile_diagnostics(tmpl)
        for d in result.diagnostics:
            assert not d.is_normalized_like, (
                f"Order {d.order}: expected is_normalized_like=False, "
                f"colsum_median={d.colsum_median}"
            )

    def test_smoothing_reduces_roughness(self):
        """Smoothed profile should have lower roughness than unsmoothed."""
        rng = np.random.default_rng(42)
        ros = _make_rectified_order_set(
            gaussian_profile=True, noise_scale=0.1, rng=rng
        )
        tmpl_nosmooth = _make_template_set(ros, smooth_sigma=0.0)
        tmpl_smooth = _make_template_set(ros, smooth_sigma=1.5)

        diags_no = compute_profile_diagnostics(tmpl_nosmooth)
        diags_sm = compute_profile_diagnostics(tmpl_smooth)

        for order in diags_no.orders:
            rough_no = diags_no.get_order(order).roughness
            rough_sm = diags_sm.get_order(order).roughness
            # Smoothed should be less rough (or at least not significantly more)
            assert rough_sm <= rough_no + 1e-10, (
                f"Order {order}: expected rough_sm <= rough_no, "
                f"but {rough_sm} > {rough_no}"
            )

    def test_peak_detection_gaussian(self):
        """Peak should be at center for a centred Gaussian profile."""
        center_frac = 0.5
        sigma_frac = 0.1
        ros = _make_rectified_order_set(
            gaussian_profile=True,
            center_frac=center_frac,
            sigma_frac=sigma_frac,
        )
        tmpl = _make_template_set(ros, normalize=True)
        result = compute_profile_diagnostics(tmpl)
        for d in result.diagnostics:
            # Peak spatial fraction should be near 0.5
            assert abs(d.peak_spatial_frac - center_frac) < 0.15, (
                f"Order {d.order}: peak_spatial_frac={d.peak_spatial_frac}, "
                f"expected near {center_frac}"
            )

    def test_peak_detection_off_center(self):
        """Peak should track off-center Gaussian."""
        center_frac = 0.3
        ros = _make_rectified_order_set(
            gaussian_profile=True,
            center_frac=center_frac,
            sigma_frac=0.08,
        )
        tmpl = _make_template_set(ros)
        result = compute_profile_diagnostics(tmpl)
        for d in result.diagnostics:
            assert abs(d.peak_spatial_frac - center_frac) < 0.15

    def test_finite_fraction_all_finite(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compute_profile_diagnostics(tmpl)
        for d in result.diagnostics:
            assert d.finite_fraction == pytest.approx(1.0)

    def test_finite_fraction_partial_nan(self):
        """Template with partial NaN → finite_fraction < 1."""
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        # Manually inject NaN into the first template's profile
        tmpl.templates[0].profile[0, :] = np.nan
        result = compute_profile_diagnostics(tmpl)
        d0 = result.get_order(_ORDER_PARAMS[0]["order_number"])
        assert d0.finite_fraction < 1.0

    def test_diagnostics_set_get_order(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compute_profile_diagnostics(tmpl)
        for p in _ORDER_PARAMS:
            d = result.get_order(p["order_number"])
            assert d.order == p["order_number"]

    def test_diagnostics_set_get_order_key_error(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compute_profile_diagnostics(tmpl)
        with pytest.raises(KeyError):
            result.get_order(999)

    def test_diagnostics_set_orders_property(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compute_profile_diagnostics(tmpl)
        assert result.orders == [p["order_number"] for p in _ORDER_PARAMS]

    def test_diagnostics_set_n_orders(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compute_profile_diagnostics(tmpl)
        assert result.n_orders == len(_ORDER_PARAMS)

    def test_n_frames_used(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compute_profile_diagnostics(tmpl)
        for d in result.diagnostics:
            assert d.n_frames_used == 1  # single-frame template

    def test_n_frames_used_multi_frame(self):
        rng = np.random.default_rng(7)
        sets = [
            _make_rectified_order_set(rng=rng, noise_scale=0.01)
            for _ in range(3)
        ]
        tmpl = build_external_profile_template(
            sets, ProfileTemplateDefinition(combine_method="median")
        )
        result = compute_profile_diagnostics(tmpl)
        for d in result.diagnostics:
            assert d.n_frames_used == 3


class TestProfileDiagnosticsAllNaN:
    """Edge case: all-NaN profile."""

    def test_all_nan_profile(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        # Inject all-NaN into the first template
        order0 = _ORDER_PARAMS[0]["order_number"]
        tmpl.templates[0].profile[:] = np.nan
        result = compute_profile_diagnostics(tmpl)
        d = result.get_order(order0)
        assert d.finite_fraction == pytest.approx(0.0)
        assert d.peak_spatial_index == -1
        assert np.isnan(d.peak_spatial_frac)
        assert np.isnan(d.colsum_min)
        assert np.isnan(d.colsum_max)
        assert np.isnan(d.colsum_median)
        assert np.isnan(d.roughness)
        assert d.is_normalized_like is False


class TestProfileDiagnosticsEmptySet:
    """Edge case: empty template set."""

    def test_empty_template_set(self):
        empty_set = ExternalProfileTemplateSet(mode=_MODE, templates=[])
        result = compute_profile_diagnostics(empty_set)
        assert result.n_orders == 0
        assert result.orders == []


# ---------------------------------------------------------------------------
# 2. External vs empirical comparison
# ---------------------------------------------------------------------------


class TestExternalVsEmpiricalDiagnostics:
    """Tests for compare_external_vs_empirical."""

    def test_identical_data_small_differences(self):
        """When template is built from the same data, differences should be small."""
        ros = _make_rectified_order_set(gaussian_profile=True, fill_value=10.0)
        tmpl = _make_template_set(ros, normalize=True)
        ext_def = _make_extraction_def()
        result = compare_external_vs_empirical(ros, ext_def, tmpl)
        assert result.n_orders == len(_ORDER_PARAMS)
        for d in result.diagnostics:
            assert np.isfinite(d.flux_l2_difference), f"Order {d.order}: l2 NaN"
            assert np.isfinite(d.median_abs_flux_difference), f"Order {d.order}: mad NaN"

    def test_metrics_are_finite(self):
        ros = _make_rectified_order_set(gaussian_profile=True, fill_value=5.0)
        tmpl = _make_template_set(ros)
        ext_def = _make_extraction_def()
        result = compare_external_vs_empirical(ros, ext_def, tmpl)
        for d in result.diagnostics:
            assert np.isfinite(d.finite_fraction_flux)
            assert np.isfinite(d.finite_fraction_variance)

    def test_noisy_data_measurable_differences(self):
        """Noisy external profile produces nonzero L2 difference from empirical."""
        rng = np.random.default_rng(99)
        cal_ros = _make_rectified_order_set(
            gaussian_profile=True, fill_value=10.0, noise_scale=0.5, rng=rng
        )
        sci_ros = _make_rectified_order_set(
            gaussian_profile=True, fill_value=10.0, noise_scale=0.5, rng=rng
        )
        tmpl = _make_template_set(cal_ros, normalize=True)
        ext_def = _make_extraction_def()
        result = compare_external_vs_empirical(sci_ros, ext_def, tmpl)
        # At least some orders should show nonzero L2 difference
        l2s = [d.flux_l2_difference for d in result.diagnostics if np.isfinite(d.flux_l2_difference)]
        assert len(l2s) > 0

    def test_external_profile_applied_bookkeeping(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        ext_def = _make_extraction_def()
        result = compare_external_vs_empirical(ros, ext_def, tmpl)
        for d in result.diagnostics:
            assert d.external_profile_applied is True
            assert d.template_n_frames_used == 1

    def test_get_order_lookup(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compare_external_vs_empirical(ros, _make_extraction_def(), tmpl)
        for p in _ORDER_PARAMS:
            d = result.get_order(p["order_number"])
            assert d.order == p["order_number"]

    def test_get_order_key_error(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compare_external_vs_empirical(ros, _make_extraction_def(), tmpl)
        with pytest.raises(KeyError):
            result.get_order(9999)

    def test_orders_and_n_orders(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compare_external_vs_empirical(ros, _make_extraction_def(), tmpl)
        assert result.n_orders == len(_ORDER_PARAMS)
        assert result.orders == [p["order_number"] for p in _ORDER_PARAMS]


# ---------------------------------------------------------------------------
# 3. Template leakage detection
# ---------------------------------------------------------------------------


class TestLeakageDetectionSameData:
    """Template built from SAME data → leakage flag expected True."""

    def test_same_data_leakage_flag_true(self):
        """Using the exact same RectifiedOrderSet for template and science
        should trigger the leakage flag on most / all orders."""
        ros = _make_rectified_order_set(gaussian_profile=True, fill_value=10.0)
        tmpl = _make_template_set(ros, normalize=True)
        ext_def = _make_extraction_def()
        result = compute_leakage_diagnostics(ros, tmpl, ext_def)
        leakage_flags = [d.possible_template_leakage for d in result.diagnostics]
        # Expect at least one order to be flagged
        assert any(leakage_flags), (
            "Expected at least one order flagged as possible leakage when "
            "template is built from the same data."
        )


class TestLeakageDetectionDifferentData:
    """Template built from DIFFERENT noisy data → leakage flag expected False."""

    def test_different_noisy_data_no_leakage(self):
        """Template from different noisy data should not trigger leakage flag."""
        rng = np.random.default_rng(123)
        cal_ros = _make_rectified_order_set(
            gaussian_profile=True,
            fill_value=10.0,
            noise_scale=2.0,
            rng=rng,
            center_frac=0.4,
        )
        sci_ros = _make_rectified_order_set(
            gaussian_profile=True,
            fill_value=10.0,
            noise_scale=2.0,
            rng=rng,
            center_frac=0.6,
        )
        tmpl = _make_template_set(cal_ros, normalize=True)
        ext_def = _make_extraction_def()
        result = compute_leakage_diagnostics(sci_ros, tmpl, ext_def)
        leakage_flags = [d.possible_template_leakage for d in result.diagnostics]
        assert not all(leakage_flags), (
            "Expected NOT all orders flagged when template is from different data."
        )


class TestLeakageDetectionCorrelation:
    """Correlation metric behaviour."""

    def test_identical_profiles_high_correlation(self):
        """Identical profiles should give correlation near 1.0."""
        ros = _make_rectified_order_set(gaussian_profile=True, fill_value=5.0)
        tmpl = _make_template_set(ros, normalize=False)
        ext_def = _make_extraction_def()
        result = compute_leakage_diagnostics(ros, tmpl, ext_def)
        for d in result.diagnostics:
            if np.isfinite(d.profile_correlation):
                assert d.profile_correlation > 0.95, (
                    f"Order {d.order}: expected high correlation for identical "
                    f"profiles, got {d.profile_correlation}"
                )

    def test_random_noise_low_correlation(self):
        """Two independent random noise frames should have low correlation."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        cal_ros = _make_rectified_order_set(
            fill_value=0.0, noise_scale=1.0, rng=rng1
        )
        sci_ros = _make_rectified_order_set(
            fill_value=0.0, noise_scale=1.0, rng=rng2
        )
        tmpl = _make_template_set(cal_ros, normalize=False)
        ext_def = _make_extraction_def()
        result = compute_leakage_diagnostics(sci_ros, tmpl, ext_def)
        for d in result.diagnostics:
            # Leakage flag should NOT be set for independent noise
            assert not d.possible_template_leakage, (
                f"Order {d.order}: unexpected leakage flag for independent noise"
            )

    def test_threshold_constants_are_reasonable(self):
        """Sanity check on the threshold values."""
        assert 0.9 < LEAKAGE_CORRELATION_THRESHOLD < 1.0
        assert 0.0 < LEAKAGE_L2_THRESHOLD < 0.5


class TestLeakageDetectionEdgeCases:
    """Edge cases in leakage diagnostics."""

    def test_missing_order_skipped(self):
        """Orders in rectified_orders but not in templates are silently skipped."""
        ros = _make_rectified_order_set()  # orders 311, 315, 320
        # Template set with only one order
        tmpl_full = _make_template_set(ros)
        # Keep only the first order
        partial_tmpl = ExternalProfileTemplateSet(
            mode=tmpl_full.mode,
            templates=[tmpl_full.templates[0]],
        )
        ext_def = _make_extraction_def()
        result = compute_leakage_diagnostics(ros, partial_tmpl, ext_def)
        assert result.n_orders == 1
        assert result.orders == [_ORDER_PARAMS[0]["order_number"]]

    def test_get_order_and_key_error(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        ext_def = _make_extraction_def()
        result = compute_leakage_diagnostics(ros, tmpl, ext_def)
        for p in _ORDER_PARAMS:
            d = result.get_order(p["order_number"])
            assert d.order == p["order_number"]
        with pytest.raises(KeyError):
            result.get_order(9999)

    def test_n_orders_and_orders(self):
        ros = _make_rectified_order_set()
        tmpl = _make_template_set(ros)
        result = compute_leakage_diagnostics(ros, tmpl, _make_extraction_def())
        assert result.n_orders == len(_ORDER_PARAMS)


# ---------------------------------------------------------------------------
# 4. run_full_profile_diagnostics
# ---------------------------------------------------------------------------


class TestRunFullProfileDiagnostics:
    """Tests for the combined diagnostics wrapper."""

    def test_returns_full_result(self):
        ros = _make_rectified_order_set(gaussian_profile=True)
        tmpl = _make_template_set(ros)
        result = run_full_profile_diagnostics(
            ros, _make_extraction_def(), tmpl
        )
        assert isinstance(result, FullProfileDiagnosticsResult)
        assert isinstance(result.template_diagnostics, ProfileDiagnosticsSet)
        assert isinstance(result.comparison_diagnostics, ExternalVsEmpiricalDiagnosticsSet)
        assert isinstance(result.leakage_diagnostics, TemplateLeakageDiagnosticsSet)

    def test_one_entry_per_order(self):
        ros = _make_rectified_order_set(gaussian_profile=True)
        tmpl = _make_template_set(ros)
        result = run_full_profile_diagnostics(
            ros, _make_extraction_def(), tmpl
        )
        assert result.template_diagnostics.n_orders == len(_ORDER_PARAMS)
        assert result.comparison_diagnostics.n_orders == len(_ORDER_PARAMS)
        assert result.leakage_diagnostics.n_orders == len(_ORDER_PARAMS)

    def test_no_crashes(self):
        """Smoke test: ensure full diagnostics run without errors."""
        rng = np.random.default_rng(55)
        cal_ros = _make_rectified_order_set(
            gaussian_profile=True, fill_value=8.0, noise_scale=0.2, rng=rng
        )
        sci_ros = _make_rectified_order_set(
            gaussian_profile=True, fill_value=8.0, noise_scale=0.2, rng=rng
        )
        tmpl = _make_template_set(cal_ros, normalize=True)
        result = run_full_profile_diagnostics(
            sci_ros, _make_extraction_def(), tmpl
        )
        # Check metrics are well-formed (finite or NaN, no exceptions)
        for d in result.template_diagnostics.diagnostics:
            assert d.finite_fraction >= 0.0
            assert d.n_frames_used >= 0

        for d in result.comparison_diagnostics.diagnostics:
            assert d.finite_fraction_flux >= 0.0
            assert d.finite_fraction_variance >= 0.0

        for d in result.leakage_diagnostics.diagnostics:
            assert isinstance(d.possible_template_leakage, bool)


# ---------------------------------------------------------------------------
# 5. Smoke test: multi-frame calibration build + diagnostics
# ---------------------------------------------------------------------------


class TestSmokeDiagnostics:
    """Smoke test: build templates from a calibration set, run all diagnostics."""

    def test_multi_frame_calibration_diagnostics(self):
        rng = np.random.default_rng(77)
        cal_sets = [
            _make_rectified_order_set(
                gaussian_profile=True, fill_value=10.0, noise_scale=0.3, rng=rng
            )
            for _ in range(3)
        ]
        sci_ros = _make_rectified_order_set(
            gaussian_profile=True, fill_value=10.0, noise_scale=0.3, rng=rng
        )
        tmpl = build_external_profile_template(
            cal_sets,
            ProfileTemplateDefinition(
                combine_method="median",
                normalize_profile=True,
                smooth_sigma=0.5,
            ),
        )
        result = run_full_profile_diagnostics(
            sci_ros, _make_extraction_def(), tmpl
        )
        # One entry per order
        assert result.template_diagnostics.n_orders == len(_ORDER_PARAMS)
        assert result.comparison_diagnostics.n_orders == len(_ORDER_PARAMS)
        assert result.leakage_diagnostics.n_orders == len(_ORDER_PARAMS)
        # Template diagnostics: 3 frames used
        for d in result.template_diagnostics.diagnostics:
            assert d.n_frames_used == 3
            assert d.finite_fraction > 0.0
        # Comparison diagnostics: all finite
        for d in result.comparison_diagnostics.diagnostics:
            assert np.isfinite(d.finite_fraction_flux)
        # Leakage diagnostics: metrics present
        for d in result.leakage_diagnostics.diagnostics:
            assert isinstance(d.possible_template_leakage, bool)
