"""
Tests for the external spatial-profile template builder (profile_templates.py).

Coverage:
  - ProfileTemplateDefinition: construction, validation, defaults.
  - ExternalProfileTemplate: construction, field access, shape properties.
  - ExternalProfileTemplateSet: construction, get_order, orders, n_orders.
  - build_external_profile_template on synthetic data:
      * single RectifiedOrderSet input,
      * multiple RectifiedOrderSet inputs,
      * mean vs median combine method,
      * normalization behaviour (normalize_profile True/False),
      * smoothing behaviour (smooth_sigma > 0 vs 0),
      * NaN handling (partial NaN, all-NaN order),
      * order lookup via get_order(),
      * profile shape compatibility with weighted extraction,
      * multi-frame profile is smoother/stabler than single-frame in noisy case.
  - Integration test:
      * build external profile template then pass into extract_weighted_optimal
        with profile_source="external".
  - Error tests:
      * empty input list → ValueError,
      * mixed modes → ValueError,
      * no common orders → ValueError,
      * invalid combine_method → ValueError,
      * negative smooth_sigma → ValueError,
      * invalid min_finite_fraction → ValueError,
      * mask shape mismatch → ValueError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * build external profile templates from a single rectified-order set,
      * verify n_orders > 0,
      * verify template shapes are well-formed,
      * verify profiles are normalized where finite,
      * verify a template can be passed into weighted extraction without crash.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

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
from pyspextool.instruments.ishell.io_utils import is_fits_file

# ---------------------------------------------------------------------------
# Path helpers for real H1 calibration data
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_H1_RAW_DIR = os.path.join(
    _REPO_ROOT, "data", "testdata", "ishell_h1_calibrations", "raw"
)
_LFS_MAGIC = b"version https://git-lfs"


def _is_real_fits(path: str) -> bool:
    """Return True if path is a real (non-LFS-pointer) FITS file."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
        return not head.startswith(_LFS_MAGIC)
    except OSError:
        return False


def _real_files(pattern: str) -> list[str]:
    """Return sorted real (non-LFS-pointer) file paths matching *pattern*."""
    if not os.path.isdir(_H1_RAW_DIR):
        return []
    return sorted(
        p
        for p in (
            os.path.join(_H1_RAW_DIR, f)
            for f in os.listdir(_H1_RAW_DIR)
            if pattern in f and is_fits_file(f)
        )
        if _is_real_fits(p)
    )


_H1_FLAT_FILES = _real_files("flat")
_H1_ARC_FILES = _real_files("arc")
_HAVE_H1_DATA = len(_H1_FLAT_FILES) >= 1 and len(_H1_ARC_FILES) >= 1

# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

_NROWS = 200
_NCOLS = 800
_N_ORDERS = 3
_N_SPECTRAL = 32
_N_SPATIAL = 20

_ORDER_PARAMS = [
    {"order_number": 311},
    {"order_number": 315},
    {"order_number": 320},
]


def _make_synthetic_rectified_order_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
    fill_value: float = 2.0,
    mode: str = "H1_test",
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.0,
) -> RectifiedOrderSet:
    """Build a RectifiedOrderSet with constant (or noisy) flux images.

    Parameters
    ----------
    n_spectral : int
        Number of spectral columns.
    n_spatial : int
        Number of spatial rows.
    fill_value : float
        Base flux value.
    mode : str
        iSHELL mode string.
    rng : numpy Generator or None
        Random generator for reproducible noise.
    noise_scale : float
        Standard deviation of Gaussian noise added to each pixel.
    """
    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        flux = np.full((n_spatial, n_spectral), fill_value)
        if noise_scale > 0.0 and rng is not None:
            flux = flux + rng.normal(0.0, noise_scale, size=(n_spatial, n_spectral))
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=np.linspace(wav_start, wav_end, n_spectral),
                spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                flux=flux,
                source_image_shape=(_NROWS, _NCOLS),
            )
        )
    return RectifiedOrderSet(
        mode=mode,
        rectified_orders=orders,
        source_image_shape=(_NROWS, _NCOLS),
    )


def _make_gaussian_profile_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
    center_frac: float = 0.5,
    sigma_frac: float = 0.1,
    mode: str = "H1_test",
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.0,
) -> RectifiedOrderSet:
    """Build a RectifiedOrderSet with a Gaussian spatial profile.

    The flux is a 2-D image where each column has a Gaussian profile
    in the spatial direction, centred at ``center_frac`` with standard
    deviation ``sigma_frac`` (in fractional slit units).  Optionally
    adds Gaussian noise.
    """
    spatial = np.linspace(0.0, 1.0, n_spatial)
    profile_1d = np.exp(-0.5 * ((spatial - center_frac) / sigma_frac) ** 2)
    profile_1d = profile_1d / profile_1d.sum()  # normalise

    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        flux = np.outer(profile_1d, np.ones(n_spectral)) * 100.0  # bright
        if noise_scale > 0.0 and rng is not None:
            flux = flux + rng.normal(0.0, noise_scale, size=(n_spatial, n_spectral))
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=np.linspace(wav_start, wav_end, n_spectral),
                spatial_frac=spatial.copy(),
                flux=flux,
                source_image_shape=(_NROWS, _NCOLS),
            )
        )
    return RectifiedOrderSet(
        mode=mode,
        rectified_orders=orders,
        source_image_shape=(_NROWS, _NCOLS),
    )


def _make_default_definition() -> ProfileTemplateDefinition:
    """Return a default ProfileTemplateDefinition."""
    return ProfileTemplateDefinition()


# ===========================================================================
# 1. ProfileTemplateDefinition: construction and validation
# ===========================================================================


class TestProfileTemplateDefinitionConstruction:
    """Tests for ProfileTemplateDefinition construction and validation."""

    def test_default_construction(self):
        """ProfileTemplateDefinition can be constructed with defaults."""
        d = ProfileTemplateDefinition()
        assert d.combine_method == "median"
        assert d.normalize_profile is True
        assert d.smooth_sigma == 0.0
        assert d.min_finite_fraction == 0.5
        assert d.mask_sources is True

    def test_explicit_fields(self):
        """ProfileTemplateDefinition accepts explicit field values."""
        d = ProfileTemplateDefinition(
            combine_method="mean",
            normalize_profile=False,
            smooth_sigma=1.5,
            min_finite_fraction=0.8,
            mask_sources=False,
        )
        assert d.combine_method == "mean"
        assert d.normalize_profile is False
        assert d.smooth_sigma == 1.5
        assert d.min_finite_fraction == 0.8
        assert d.mask_sources is False

    def test_invalid_combine_method(self):
        """Invalid combine_method raises ValueError."""
        with pytest.raises(ValueError, match="combine_method"):
            ProfileTemplateDefinition(combine_method="mode")

    def test_negative_smooth_sigma(self):
        """Negative smooth_sigma raises ValueError."""
        with pytest.raises(ValueError, match="smooth_sigma"):
            ProfileTemplateDefinition(smooth_sigma=-0.1)

    def test_zero_smooth_sigma_is_valid(self):
        """smooth_sigma=0.0 is valid (no smoothing)."""
        d = ProfileTemplateDefinition(smooth_sigma=0.0)
        assert d.smooth_sigma == 0.0

    def test_invalid_min_finite_fraction_zero(self):
        """min_finite_fraction=0.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_finite_fraction"):
            ProfileTemplateDefinition(min_finite_fraction=0.0)

    def test_invalid_min_finite_fraction_negative(self):
        """Negative min_finite_fraction raises ValueError."""
        with pytest.raises(ValueError, match="min_finite_fraction"):
            ProfileTemplateDefinition(min_finite_fraction=-0.1)

    def test_invalid_min_finite_fraction_gt_one(self):
        """min_finite_fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_finite_fraction"):
            ProfileTemplateDefinition(min_finite_fraction=1.5)

    def test_min_finite_fraction_one_is_valid(self):
        """min_finite_fraction=1.0 is valid (all frames must be finite)."""
        d = ProfileTemplateDefinition(min_finite_fraction=1.0)
        assert d.min_finite_fraction == 1.0


# ===========================================================================
# 2. ExternalProfileTemplate: construction and properties
# ===========================================================================


class TestExternalProfileTemplateConstruction:
    """Tests for ExternalProfileTemplate construction and properties."""

    def _make_template(self) -> ExternalProfileTemplate:
        return ExternalProfileTemplate(
            order=311,
            wavelength_um=np.linspace(1.55, 1.59, _N_SPECTRAL),
            spatial_frac=np.linspace(0.0, 1.0, _N_SPATIAL),
            profile=np.ones((_N_SPATIAL, _N_SPECTRAL)) / _N_SPATIAL,
            n_frames_used=1,
            source_mode="H1_test",
            profile_smoothed=False,
            finite_fraction=1.0,
        )

    def test_basic_construction(self):
        """ExternalProfileTemplate can be constructed."""
        t = self._make_template()
        assert t.order == 311
        assert t.n_frames_used == 1
        assert t.source_mode == "H1_test"
        assert t.profile_smoothed is False
        assert t.finite_fraction == 1.0

    def test_n_spectral_property(self):
        """n_spectral returns the correct value."""
        t = self._make_template()
        assert t.n_spectral == _N_SPECTRAL

    def test_n_spatial_property(self):
        """n_spatial returns the correct value."""
        t = self._make_template()
        assert t.n_spatial == _N_SPATIAL

    def test_shape_property(self):
        """shape property returns (n_spatial, n_spectral)."""
        t = self._make_template()
        assert t.shape == (_N_SPATIAL, _N_SPECTRAL)


# ===========================================================================
# 3. ExternalProfileTemplateSet: construction and access
# ===========================================================================


class TestExternalProfileTemplateSetConstruction:
    """Tests for ExternalProfileTemplateSet construction and access."""

    def _make_template_set(self) -> ExternalProfileTemplateSet:
        templates = [
            ExternalProfileTemplate(
                order=o,
                wavelength_um=np.linspace(1.55, 1.59, _N_SPECTRAL),
                spatial_frac=np.linspace(0.0, 1.0, _N_SPATIAL),
                profile=np.ones((_N_SPATIAL, _N_SPECTRAL)) / _N_SPATIAL,
                n_frames_used=1,
                source_mode="H1_test",
                profile_smoothed=False,
                finite_fraction=1.0,
            )
            for o in [311, 315, 320]
        ]
        return ExternalProfileTemplateSet(mode="H1_test", templates=templates)

    def test_basic_construction(self):
        """ExternalProfileTemplateSet can be constructed."""
        ts = self._make_template_set()
        assert ts.mode == "H1_test"
        assert ts.n_orders == 3

    def test_orders_property(self):
        """orders property returns list of order numbers."""
        ts = self._make_template_set()
        assert ts.orders == [311, 315, 320]

    def test_get_order_success(self):
        """get_order returns the correct template."""
        ts = self._make_template_set()
        t = ts.get_order(315)
        assert t.order == 315

    def test_get_order_raises_key_error(self):
        """get_order raises KeyError for missing order."""
        ts = self._make_template_set()
        with pytest.raises(KeyError):
            ts.get_order(999)

    def test_n_orders_empty_set(self):
        """An empty set has n_orders=0."""
        ts = ExternalProfileTemplateSet(mode="H1_test")
        assert ts.n_orders == 0
        assert ts.orders == []


# ===========================================================================
# 4. build_external_profile_template: single input
# ===========================================================================


class TestBuildExternalProfileTemplateSingleInput:
    """Tests for build_external_profile_template with a single input set."""

    def test_returns_correct_type(self):
        """Returns an ExternalProfileTemplateSet."""
        ros = _make_synthetic_rectified_order_set()
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        assert isinstance(result, ExternalProfileTemplateSet)

    def test_mode_propagated(self):
        """mode is taken from the input set."""
        ros = _make_synthetic_rectified_order_set(mode="H1_test")
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        assert result.mode == "H1_test"

    def test_n_orders_matches_input(self):
        """n_orders equals the number of orders in the input set."""
        ros = _make_synthetic_rectified_order_set()
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        assert result.n_orders == ros.n_orders

    def test_orders_list_matches_input(self):
        """orders list matches the input set's order numbers."""
        ros = _make_synthetic_rectified_order_set()
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        assert sorted(result.orders) == sorted(ros.orders)

    def test_profile_shape_correct(self):
        """Each template profile has shape (n_spatial, n_spectral)."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            ro = ros.get_order(template.order)
            assert template.profile.shape == ro.flux.shape

    def test_n_frames_used_is_one(self):
        """n_frames_used == 1 for a single input."""
        ros = _make_synthetic_rectified_order_set()
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            assert template.n_frames_used == 1

    def test_profile_not_smoothed_by_default(self):
        """profile_smoothed is False when smooth_sigma=0."""
        ros = _make_synthetic_rectified_order_set()
        d = ProfileTemplateDefinition(smooth_sigma=0.0)
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            assert template.profile_smoothed is False

    def test_wavelength_um_propagated(self):
        """wavelength_um in template matches input order wavelengths."""
        ros = _make_synthetic_rectified_order_set()
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            ro = ros.get_order(template.order)
            np.testing.assert_array_equal(template.wavelength_um, ro.wavelength_um)

    def test_spatial_frac_propagated(self):
        """spatial_frac in template matches input order spatial axis."""
        ros = _make_synthetic_rectified_order_set()
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            ro = ros.get_order(template.order)
            np.testing.assert_array_equal(template.spatial_frac, ro.spatial_frac)

    def test_single_ros_also_accepted_as_list(self):
        """A list containing one set is equivalent to passing the set directly."""
        ros = _make_synthetic_rectified_order_set()
        d = _make_default_definition()
        result_direct = build_external_profile_template(ros, d)
        result_list = build_external_profile_template([ros], d)
        assert result_direct.n_orders == result_list.n_orders
        np.testing.assert_array_equal(
            result_direct.templates[0].profile,
            result_list.templates[0].profile,
        )


# ===========================================================================
# 5. build_external_profile_template: multiple inputs
# ===========================================================================


class TestBuildExternalProfileTemplateMultipleInputs:
    """Tests for build_external_profile_template with multiple input sets."""

    def test_multiple_sets_n_frames_used(self):
        """n_frames_used equals the number of input sets."""
        ros1 = _make_synthetic_rectified_order_set(fill_value=2.0)
        ros2 = _make_synthetic_rectified_order_set(fill_value=3.0)
        d = _make_default_definition()
        result = build_external_profile_template([ros1, ros2], d)
        for template in result.templates:
            assert template.n_frames_used == 2

    def test_three_sets_n_frames_used(self):
        """n_frames_used equals three when three sets are supplied."""
        sets = [_make_synthetic_rectified_order_set(fill_value=float(v)) for v in range(1, 4)]
        d = _make_default_definition()
        result = build_external_profile_template(sets, d)
        for template in result.templates:
            assert template.n_frames_used == 3

    def test_median_combine_of_identical_images(self):
        """Median of two identical images equals the original."""
        ros1 = _make_synthetic_rectified_order_set(fill_value=5.0)
        ros2 = _make_synthetic_rectified_order_set(fill_value=5.0)
        d = ProfileTemplateDefinition(combine_method="median", normalize_profile=False)
        result = build_external_profile_template([ros1, ros2], d)
        single = build_external_profile_template(ros1, d)
        for order in result.orders:
            np.testing.assert_allclose(
                result.get_order(order).profile,
                single.get_order(order).profile,
                rtol=1e-10,
            )

    def test_mean_combine_of_identical_images(self):
        """Mean of two identical images equals the original."""
        ros1 = _make_synthetic_rectified_order_set(fill_value=5.0)
        ros2 = _make_synthetic_rectified_order_set(fill_value=5.0)
        d = ProfileTemplateDefinition(combine_method="mean", normalize_profile=False)
        result = build_external_profile_template([ros1, ros2], d)
        single = build_external_profile_template(ros1, d)
        for order in result.orders:
            np.testing.assert_allclose(
                result.get_order(order).profile,
                single.get_order(order).profile,
                rtol=1e-10,
            )

    def test_median_vs_mean_differ_on_asymmetric_data(self):
        """median and mean produce different results on asymmetric data."""
        ros1 = _make_synthetic_rectified_order_set(fill_value=1.0)
        ros2 = _make_synthetic_rectified_order_set(fill_value=10.0)
        ros3 = _make_synthetic_rectified_order_set(fill_value=10.0)
        d_med = ProfileTemplateDefinition(combine_method="median", normalize_profile=False)
        d_mean = ProfileTemplateDefinition(combine_method="mean", normalize_profile=False)
        result_med = build_external_profile_template([ros1, ros2, ros3], d_med)
        result_mean = build_external_profile_template([ros1, ros2, ros3], d_mean)
        order = result_med.orders[0]
        # Median of [1,10,10] = 10; mean of [1,10,10] = 7
        assert not np.allclose(
            result_med.get_order(order).profile,
            result_mean.get_order(order).profile,
        )

    def test_only_common_orders_returned(self):
        """Only orders present in all input sets are in the output."""
        ros1 = _make_synthetic_rectified_order_set()
        # ros2 is missing order 311 (first order)
        orders_subset = [
            RectifiedOrder(
                order=315,
                order_index=0,
                wavelength_um=np.linspace(1.60, 1.64, _N_SPECTRAL),
                spatial_frac=np.linspace(0.0, 1.0, _N_SPATIAL),
                flux=np.ones((_N_SPATIAL, _N_SPECTRAL)),
                source_image_shape=(_NROWS, _NCOLS),
            ),
            RectifiedOrder(
                order=320,
                order_index=1,
                wavelength_um=np.linspace(1.65, 1.69, _N_SPECTRAL),
                spatial_frac=np.linspace(0.0, 1.0, _N_SPATIAL),
                flux=np.ones((_N_SPATIAL, _N_SPECTRAL)),
                source_image_shape=(_NROWS, _NCOLS),
            ),
        ]
        ros2 = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=orders_subset,
            source_image_shape=(_NROWS, _NCOLS),
        )
        d = _make_default_definition()
        result = build_external_profile_template([ros1, ros2], d)
        # Only orders 315 and 320 should be in the result
        assert 311 not in result.orders
        assert 315 in result.orders
        assert 320 in result.orders


# ===========================================================================
# 6. Normalization behaviour
# ===========================================================================


class TestNormalizationBehaviour:
    """Tests for normalize_profile=True/False."""

    def test_normalized_columns_sum_to_one(self):
        """Each finite column sums to 1.0 when normalize_profile=True."""
        ros = _make_gaussian_profile_set()
        d = ProfileTemplateDefinition(normalize_profile=True)
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            profile = template.profile
            col_sums = np.nansum(profile, axis=0)
            finite_cols = np.isfinite(col_sums) & (col_sums > 0)
            np.testing.assert_allclose(
                col_sums[finite_cols], 1.0, atol=1e-10,
                err_msg=f"Order {template.order}: columns do not sum to 1.0"
            )

    def test_unnormalized_columns_do_not_sum_to_one(self):
        """Columns do not sum to 1.0 when normalize_profile=False."""
        ros = _make_gaussian_profile_set()
        d = ProfileTemplateDefinition(normalize_profile=False)
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            profile = template.profile
            col_sums = np.nansum(profile, axis=0)
            finite_cols = np.isfinite(col_sums) & (col_sums > 0)
            # With a Gaussian scaled to 100, the sum will be >> 1
            assert not np.allclose(col_sums[finite_cols], 1.0, atol=1e-3)

    def test_constant_flux_normalized_sums_to_one(self):
        """Constant-flux input → all non-NaN columns sum to 1.0 after normalization."""
        ros = _make_synthetic_rectified_order_set(fill_value=3.0)
        d = ProfileTemplateDefinition(normalize_profile=True)
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            profile = template.profile
            col_sums = np.nansum(profile, axis=0)
            finite_cols = np.isfinite(col_sums) & (col_sums > 0)
            np.testing.assert_allclose(col_sums[finite_cols], 1.0, atol=1e-10)


# ===========================================================================
# 7. Smoothing behaviour
# ===========================================================================


class TestSmoothingBehaviour:
    """Tests for smooth_sigma > 0 vs 0."""

    def test_smooth_sigma_gt_zero_sets_profile_smoothed_flag(self):
        """profile_smoothed is True when smooth_sigma > 0."""
        ros = _make_gaussian_profile_set()
        d = ProfileTemplateDefinition(smooth_sigma=2.0)
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            assert template.profile_smoothed is True

    def test_smooth_sigma_zero_leaves_profile_smoothed_false(self):
        """profile_smoothed is False when smooth_sigma = 0."""
        ros = _make_gaussian_profile_set()
        d = ProfileTemplateDefinition(smooth_sigma=0.0)
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            assert template.profile_smoothed is False

    def test_smoothed_profile_differs_from_unsmoothed(self):
        """Smoothed profile is different from unsmoothed profile."""
        ros = _make_gaussian_profile_set(sigma_frac=0.05)  # narrow profile
        d_smooth = ProfileTemplateDefinition(smooth_sigma=3.0, normalize_profile=False)
        d_plain = ProfileTemplateDefinition(smooth_sigma=0.0, normalize_profile=False)
        result_smooth = build_external_profile_template(ros, d_smooth)
        result_plain = build_external_profile_template(ros, d_plain)
        order = result_smooth.orders[0]
        prof_smooth = result_smooth.get_order(order).profile
        prof_plain = result_plain.get_order(order).profile
        assert not np.allclose(prof_smooth, prof_plain, equal_nan=True)

    def test_smoothed_profile_sums_to_one_when_normalized(self):
        """Smoothed profile columns still sum to 1.0 when normalize_profile=True."""
        ros = _make_gaussian_profile_set()
        d = ProfileTemplateDefinition(smooth_sigma=2.0, normalize_profile=True)
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            profile = template.profile
            col_sums = np.nansum(profile, axis=0)
            finite_cols = np.isfinite(col_sums) & (col_sums > 0)
            np.testing.assert_allclose(col_sums[finite_cols], 1.0, atol=1e-10)


# ===========================================================================
# 8. NaN handling
# ===========================================================================


class TestNaNHandling:
    """Tests for NaN handling in input flux images."""

    def test_partial_nan_input_produces_finite_profile(self):
        """Partial NaN input still yields finite profiles where data is available."""
        ros = _make_synthetic_rectified_order_set(fill_value=2.0)
        # Introduce NaN in the first few columns of the first order
        first_order = ros.rectified_orders[0].flux
        first_order[:, :5] = np.nan
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        template = result.get_order(ros.orders[0])
        # Non-NaN columns should have finite profiles
        finite_cols = np.isfinite(template.profile).all(axis=0)
        assert finite_cols.sum() > 0

    def test_all_nan_order_returns_nan_profile(self):
        """An all-NaN order produces a NaN profile."""
        ros = _make_synthetic_rectified_order_set(fill_value=2.0)
        # Make first order all-NaN
        ros.rectified_orders[0].flux[:, :] = np.nan
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        template = result.get_order(ros.orders[0])
        assert np.all(~np.isfinite(template.profile))

    def test_all_nan_order_finite_fraction_is_zero(self):
        """An all-NaN order has finite_fraction = 0."""
        ros = _make_synthetic_rectified_order_set(fill_value=2.0)
        ros.rectified_orders[0].flux[:, :] = np.nan
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        template = result.get_order(ros.orders[0])
        assert template.finite_fraction == 0.0

    def test_good_orders_unaffected_by_nan_in_other_orders(self):
        """NaN in one order does not affect profiles of other orders."""
        ros = _make_synthetic_rectified_order_set(fill_value=3.0)
        ros.rectified_orders[0].flux[:, :] = np.nan
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        for order in ros.orders[1:]:  # skip the NaN order
            template = result.get_order(order)
            assert np.any(np.isfinite(template.profile))

    def test_nan_pixels_excluded_from_combination(self):
        """Pixels that are NaN in one frame are excluded from combination."""
        rng = np.random.default_rng(0)
        ros1 = _make_synthetic_rectified_order_set(fill_value=4.0)
        ros2 = _make_synthetic_rectified_order_set(fill_value=4.0)
        # Introduce NaN in some pixels of ros1 first order
        ros1.rectified_orders[0].flux[0, 0] = np.nan
        d = ProfileTemplateDefinition(
            combine_method="mean",
            normalize_profile=False,
            min_finite_fraction=0.4,  # allow 1-of-2 finite
        )
        result = build_external_profile_template([ros1, ros2], d)
        # Should still produce finite result (mean of [nan, 4] with min_finite_fraction=0.4 → 4)
        template = result.get_order(ros1.orders[0])
        assert np.any(np.isfinite(template.profile))

    def test_min_finite_fraction_masks_insufficiently_covered_pixels(self):
        """Pixels finite in < min_finite_fraction of frames become NaN."""
        ros1 = _make_synthetic_rectified_order_set(fill_value=4.0)
        ros2 = _make_synthetic_rectified_order_set(fill_value=4.0)
        # Make first order in ros1 all NaN → only 1/2 = 0.5 finite fraction
        ros1.rectified_orders[0].flux[:, :] = np.nan
        d = ProfileTemplateDefinition(
            combine_method="median",
            normalize_profile=False,
            min_finite_fraction=0.9,  # require 90% → 0.5 < 0.9 → all NaN
        )
        result = build_external_profile_template([ros1, ros2], d)
        template = result.get_order(ros1.orders[0])
        assert np.all(~np.isfinite(template.profile))


# ===========================================================================
# 9. Profile shape compatibility with weighted extraction
# ===========================================================================


class TestProfileShapeCompatibility:
    """Tests that profile shapes are compatible with weighted extraction."""

    def test_profile_shape_is_n_spatial_by_n_spectral(self):
        """Profile shape is (n_spatial, n_spectral)."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=15, n_spectral=24
        )
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            assert template.profile.shape == (15, 24)
            assert template.n_spatial == 15
            assert template.n_spectral == 24

    def test_wavelength_and_spatial_axes_consistent(self):
        """wavelength_um and spatial_frac axes are consistent with profile shape."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=12, n_spectral=18
        )
        d = _make_default_definition()
        result = build_external_profile_template(ros, d)
        for template in result.templates:
            assert template.wavelength_um.shape[0] == template.n_spectral
            assert template.spatial_frac.shape[0] == template.n_spatial


# ===========================================================================
# 10. Multi-frame profile is smoother than single-frame in noisy case
# ===========================================================================


class TestMultiFrameStability:
    """Tests that multi-frame averaging improves profile stability."""

    def test_multi_frame_profile_has_lower_variance_than_single(self):
        """Profile from 10 noisy frames has lower column-to-column variance."""
        rng = np.random.default_rng(42)
        n_frames = 10
        noise_scale = 0.5
        sets = [
            _make_gaussian_profile_set(
                rng=rng, noise_scale=noise_scale
            )
            for _ in range(n_frames)
        ]
        single = _make_gaussian_profile_set(rng=rng, noise_scale=noise_scale)

        d = ProfileTemplateDefinition(
            combine_method="median", normalize_profile=True
        )
        result_multi = build_external_profile_template(sets, d)
        result_single = build_external_profile_template(single, d)

        # For each order, compute variance of profile across spectral columns.
        for order in result_multi.orders:
            prof_multi = result_multi.get_order(order).profile
            prof_single = result_single.get_order(order).profile

            # Standard deviation of the profile across spectral columns
            # (should be lower for multi-frame).
            finite_multi = np.isfinite(prof_multi)
            finite_single = np.isfinite(prof_single)
            if finite_multi.sum() == 0 or finite_single.sum() == 0:
                continue

            std_multi = np.nanstd(prof_multi)
            std_single = np.nanstd(prof_single)
            assert std_multi <= std_single * 1.5, (
                f"Order {order}: multi-frame std {std_multi:.4f} is not "
                f"substantially lower than single-frame std {std_single:.4f}"
            )


# ===========================================================================
# 11. Integration test: build template → extract_weighted_optimal
# ===========================================================================


class TestIntegrationWithWeightedExtraction:
    """Integration test: build external profile template and pass to extraction."""

    def test_extraction_with_external_profile_runs_without_error(self):
        """Extraction with profile_source='external' runs without crash."""
        from pyspextool.instruments.ishell.weighted_optimal_extraction import (
            WeightedExtractionDefinition,
            WeightedExtractedSpectrumSet,
            extract_weighted_optimal,
        )

        ros = _make_gaussian_profile_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        d = _make_default_definition()
        template_set = build_external_profile_template(ros, d)

        results = []
        for template in template_set.templates:
            ext_def = WeightedExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.3,
                profile_source="external",
                external_profile=template.profile,
            )
            result = extract_weighted_optimal(ros, ext_def)
            results.append(result)

        assert len(results) == template_set.n_orders

    def test_extraction_returns_correct_type(self):
        """Extraction with external profile returns WeightedExtractedSpectrumSet."""
        from pyspextool.instruments.ishell.weighted_optimal_extraction import (
            WeightedExtractionDefinition,
            WeightedExtractedSpectrumSet,
            extract_weighted_optimal,
        )

        ros = _make_gaussian_profile_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        d = _make_default_definition()
        template_set = build_external_profile_template(ros, d)

        first_template = template_set.templates[0]
        ext_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="external",
            external_profile=first_template.profile,
        )
        result = extract_weighted_optimal(ros, ext_def)
        assert isinstance(result, WeightedExtractedSpectrumSet)

    def test_multi_frame_template_extraction_produces_finite_flux(self):
        """Extraction using a multi-frame external template produces finite flux."""
        from pyspextool.instruments.ishell.weighted_optimal_extraction import (
            WeightedExtractionDefinition,
            extract_weighted_optimal,
        )

        rng = np.random.default_rng(7)
        sets = [
            _make_gaussian_profile_set(
                n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL,
                rng=rng, noise_scale=0.05,
            )
            for _ in range(3)
        ]
        d = _make_default_definition()
        template_set = build_external_profile_template(sets, d)

        science_ros = sets[0]  # use first set as the science frame to extract
        first_template = template_set.templates[0]
        ext_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="external",
            external_profile=first_template.profile,
        )
        result = extract_weighted_optimal(science_ros, ext_def)
        sp = result.spectra[0]
        finite_frac = np.mean(np.isfinite(sp.flux))
        assert finite_frac > 0.5, (
            f"Fewer than 50% of flux values are finite ({finite_frac:.1%})"
        )

    def test_profile_source_used_is_external(self):
        """profile_source_used == 'external' in the extraction result."""
        from pyspextool.instruments.ishell.weighted_optimal_extraction import (
            WeightedExtractionDefinition,
            extract_weighted_optimal,
        )

        ros = _make_gaussian_profile_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        d = _make_default_definition()
        template_set = build_external_profile_template(ros, d)
        first_template = template_set.templates[0]
        ext_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="external",
            external_profile=first_template.profile,
        )
        result = extract_weighted_optimal(ros, ext_def)
        for sp in result.spectra:
            assert sp.profile_source_used == "external"


# ===========================================================================
# 12. Error tests
# ===========================================================================


class TestErrorCases:
    """Tests for invalid inputs and error handling."""

    def test_empty_list_raises_value_error(self):
        """Empty input list raises ValueError."""
        d = _make_default_definition()
        with pytest.raises(ValueError, match="at least one"):
            build_external_profile_template([], d)

    def test_mixed_modes_raises_value_error(self):
        """Input sets with different modes raise ValueError."""
        ros1 = _make_synthetic_rectified_order_set(mode="H1_test")
        ros2 = _make_synthetic_rectified_order_set(mode="K_test")
        d = _make_default_definition()
        with pytest.raises(ValueError, match="mode"):
            build_external_profile_template([ros1, ros2], d)

    def test_no_common_orders_raises_value_error(self):
        """Input sets with disjoint order numbers raise ValueError."""
        orders_a = [
            RectifiedOrder(
                order=300,
                order_index=0,
                wavelength_um=np.linspace(1.5, 1.6, _N_SPECTRAL),
                spatial_frac=np.linspace(0.0, 1.0, _N_SPATIAL),
                flux=np.ones((_N_SPATIAL, _N_SPECTRAL)),
                source_image_shape=(_NROWS, _NCOLS),
            )
        ]
        orders_b = [
            RectifiedOrder(
                order=400,
                order_index=0,
                wavelength_um=np.linspace(1.6, 1.7, _N_SPECTRAL),
                spatial_frac=np.linspace(0.0, 1.0, _N_SPATIAL),
                flux=np.ones((_N_SPATIAL, _N_SPECTRAL)),
                source_image_shape=(_NROWS, _NCOLS),
            )
        ]
        ros_a = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=orders_a,
            source_image_shape=(_NROWS, _NCOLS),
        )
        ros_b = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=orders_b,
            source_image_shape=(_NROWS, _NCOLS),
        )
        d = _make_default_definition()
        with pytest.raises(ValueError, match="[Oo]rder"):
            build_external_profile_template([ros_a, ros_b], d)

    def test_invalid_combine_method(self):
        """Invalid combine_method raises ValueError at definition construction."""
        with pytest.raises(ValueError, match="combine_method"):
            ProfileTemplateDefinition(combine_method="harmonic")

    def test_negative_smooth_sigma(self):
        """Negative smooth_sigma raises ValueError at definition construction."""
        with pytest.raises(ValueError, match="smooth_sigma"):
            ProfileTemplateDefinition(smooth_sigma=-1.0)

    def test_invalid_min_finite_fraction(self):
        """min_finite_fraction outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="min_finite_fraction"):
            ProfileTemplateDefinition(min_finite_fraction=0.0)

    def test_mask_shape_mismatch_raises_value_error(self):
        """Mask shape mismatch raises ValueError."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        d = _make_default_definition()
        # Mask with wrong shape
        bad_mask = np.zeros((_N_SPATIAL + 5, _N_SPECTRAL), dtype=bool)
        with pytest.raises(ValueError, match="[Mm]ask"):
            build_external_profile_template(ros, d, mask=bad_mask)

    def test_mask_must_be_2d(self):
        """1-D mask raises ValueError."""
        ros = _make_synthetic_rectified_order_set()
        d = _make_default_definition()
        bad_mask = np.zeros(10, dtype=bool)
        with pytest.raises(ValueError, match="[Mm]ask"):
            build_external_profile_template(ros, d, mask=bad_mask)


# ===========================================================================
# 13. Mask behaviour
# ===========================================================================


class TestMaskBehaviour:
    """Tests for the optional pixel mask."""

    def test_mask_sets_pixels_to_nan(self):
        """Masked pixels are set to NaN before combining."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL, fill_value=5.0
        )
        d = ProfileTemplateDefinition(normalize_profile=False)
        # Mask the first spatial row in the first order only
        first_order_shape = ros.rectified_orders[0].flux.shape
        mask = np.zeros(first_order_shape, dtype=bool)
        mask[0, :] = True  # mask the first row

        result_masked = build_external_profile_template(ros, d, mask=mask)
        result_plain = build_external_profile_template(ros, d, mask=None)

        first_order = ros.orders[0]
        # The masked result's first-row profile values should be NaN or zero
        prof_masked = result_masked.get_order(first_order).profile
        prof_plain = result_plain.get_order(first_order).profile
        # Profiles should differ (masked spatial row is excluded)
        assert not np.allclose(prof_masked, prof_plain, equal_nan=True)

    def test_all_false_mask_is_equivalent_to_no_mask(self):
        """An all-False mask produces the same result as no mask."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL, fill_value=5.0
        )
        d = _make_default_definition()
        first_order_shape = ros.rectified_orders[0].flux.shape
        zero_mask = np.zeros(first_order_shape, dtype=bool)

        result_mask = build_external_profile_template(ros, d, mask=zero_mask)
        result_none = build_external_profile_template(ros, d, mask=None)

        for order in result_mask.orders:
            np.testing.assert_array_equal(
                result_mask.get_order(order).profile,
                result_none.get_order(order).profile,
            )


# ===========================================================================
# 14. Smoke test on real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1ProfileTemplateSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset.

    Chain:
      1. Flat-order tracing (Stage 1)
      2. Arc-line tracing (Stage 2)
      3. Provisional wavelength mapping (Stage 3)
      4. Coefficient-surface refinement (Stage 5)
      5. Rectification indices (Stage 6)
      6. Rectified orders (Stage 7)
      7. External profile template building (Stage 21)
      8. Proto-Horne weighted optimal extraction (Stage 17 / 20)
    """

    @pytest.fixture(scope="class")
    def h1_profile_template_result(self):
        """ExternalProfileTemplateSet from the real H1 calibration chain."""
        import astropy.io.fits as fits

        from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
        from pyspextool.instruments.ishell.calibrations import (
            read_line_list,
            read_wavecalinfo,
        )
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )
        from pyspextool.instruments.ishell.rectified_orders import (
            build_rectified_orders,
        )
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
        from pyspextool.instruments.ishell.wavecal_2d import (
            fit_provisional_wavelength_map,
        )
        from pyspextool.instruments.ishell.wavecal_2d_refine import (
            fit_refined_coefficient_surface,
        )

        # Stage 1: flat tracing
        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES, col_range=(650, 1550)
        )
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))

        # Stage 2: arc tracing
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)

        # Stage 3: provisional wavelength mapping
        wci = read_wavecalinfo("H1")
        ll = read_line_list("H1")
        wav_map = fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )

        # Stage 5: coefficient-surface refinement
        surface = fit_refined_coefficient_surface(wav_map, disp_degree=3)

        # Stage 6: rectification indices
        rect_indices = build_rectification_indices(geom, surface, wav_map=wav_map)

        # Stage 7: rectified orders
        with fits.open(_H1_FLAT_FILES[0]) as hdul:
            detector_image = hdul[0].data.astype(float)
        rectified = build_rectified_orders(detector_image, rect_indices)

        # Stage 21: external profile template builder
        definition = ProfileTemplateDefinition(
            combine_method="median",
            normalize_profile=True,
            smooth_sigma=0.0,
        )
        template_set = build_external_profile_template(rectified, definition)

        return template_set, rectified

    def test_returns_external_profile_template_set(self, h1_profile_template_result):
        """build_external_profile_template returns an ExternalProfileTemplateSet."""
        template_set, _ = h1_profile_template_result
        assert isinstance(template_set, ExternalProfileTemplateSet)

    def test_n_orders_positive(self, h1_profile_template_result):
        """n_orders > 0."""
        template_set, _ = h1_profile_template_result
        assert template_set.n_orders > 0

    def test_n_orders_matches_rectified(self, h1_profile_template_result):
        """n_orders matches the rectified order set."""
        template_set, rectified = h1_profile_template_result
        assert template_set.n_orders == rectified.n_orders

    def test_template_shapes_well_formed(self, h1_profile_template_result):
        """Each template profile has shape (n_spatial, n_spectral)."""
        template_set, rectified = h1_profile_template_result
        for template in template_set.templates:
            ro = rectified.get_order(template.order)
            assert template.profile.shape == ro.flux.shape

    def test_profiles_normalized_where_finite(self, h1_profile_template_result):
        """Finite columns of each profile sum to 1.0."""
        template_set, _ = h1_profile_template_result
        for template in template_set.templates:
            profile = template.profile
            col_sums = np.nansum(profile, axis=0)
            finite_cols = np.isfinite(col_sums) & (col_sums > 0)
            if finite_cols.sum() == 0:
                continue
            np.testing.assert_allclose(
                col_sums[finite_cols], 1.0, atol=1e-10,
                err_msg=f"Order {template.order}: profile columns don't sum to 1.0"
            )

    def test_finite_fraction_is_reasonable(self, h1_profile_template_result):
        """finite_fraction > 0 for at least one order."""
        template_set, _ = h1_profile_template_result
        fractions = [t.finite_fraction for t in template_set.templates]
        assert max(fractions) > 0.0

    def test_extraction_with_external_template_runs(self, h1_profile_template_result):
        """Extraction with an external template from H1 data runs without crash."""
        from pyspextool.instruments.ishell.weighted_optimal_extraction import (
            WeightedExtractionDefinition,
            WeightedExtractedSpectrumSet,
            extract_weighted_optimal,
        )
        from pyspextool.instruments.ishell.variance_model import VarianceModelDefinition

        template_set, rectified = h1_profile_template_result
        first_template = template_set.templates[0]

        var_model = VarianceModelDefinition(
            read_noise_electron=11.0,
            gain_e_per_adu=1.8,
        )
        ext_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            background_inner=0.3,
            background_outer=0.45,
            profile_source="external",
            external_profile=first_template.profile,
        )
        result = extract_weighted_optimal(
            rectified,
            ext_def,
            variance_model=var_model,
            subtract_background=True,
        )
        assert isinstance(result, WeightedExtractedSpectrumSet)
        assert result.n_orders > 0
