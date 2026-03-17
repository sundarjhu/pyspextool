"""
Tests for the proto-Horne variance-weighted optimal-extraction scaffold
(weighted_optimal_extraction.py).

Coverage:
  - WeightedExtractionDefinition: construction, validation, properties.
  - WeightedExtractedOrderSpectrum: construction, field access, n_spectral.
  - WeightedExtractedSpectrumSet: construction, get_order, orders, n_orders.
  - extract_weighted_optimal on synthetic data:
      * known spatial profile with unit variance,
      * known spatial profile with non-uniform variance,
      * unit variance case recovers same flux as Stage-12 unweighted estimator,
      * non-uniform variance changes result relative to unit variance,
      * background subtraction correctness,
      * NaN / inf / mask handling,
      * shape consistency,
      * wavelength-axis propagation (independent copy),
      * variance positivity,
      * columns with no valid support return NaN,
      * weights 2-D array stored and has correct shape.
  - Error tests:
      * invalid WeightedExtractionDefinition (bad center, radius, background,
        profile_mode, use_variance_model inconsistency),
      * empty rectified order set → ValueError,
      * non-2-D flux image → ValueError,
      * variance_image shape mismatch → ValueError,
      * mask shape mismatch → ValueError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * full chain stages 1–7 + weighted optimal extraction (stage 17),
      * number of spectra matches number of rectified orders,
      * wavelength axes are monotonic,
      * flux arrays have correct shape,
      * variance arrays have correct shape and are non-negative where finite,
      * outputs are finite where expected.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyspextool.instruments.ishell.rectified_orders import (
    RectifiedOrder,
    RectifiedOrderSet,
)
from pyspextool.instruments.ishell.variance_model import VarianceModelDefinition
from pyspextool.instruments.ishell.weighted_optimal_extraction import (
    WeightedExtractionDefinition,
    WeightedExtractedOrderSpectrum,
    WeightedExtractedSpectrumSet,
    extract_weighted_optimal,
)

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
            if pattern in f and f.endswith(".fits")
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
) -> RectifiedOrderSet:
    """Build a RectifiedOrderSet with constant-value flux images."""
    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=np.linspace(wav_start, wav_end, n_spectral),
                spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                flux=np.full((n_spatial, n_spectral), fill_value),
                source_image_shape=(_NROWS, _NCOLS),
            )
        )
    return RectifiedOrderSet(
        mode="H1_test",
        rectified_orders=orders,
        source_image_shape=(_NROWS, _NCOLS),
    )


def _make_center_extraction_def(
    radius: float = 0.2,
    profile_mode: str = "global_median",
    normalize_profile: bool = True,
) -> WeightedExtractionDefinition:
    """Extraction definition centered at slit midpoint, no background."""
    return WeightedExtractionDefinition(
        center_frac=0.5,
        radius_frac=radius,
        profile_mode=profile_mode,
        normalize_profile=normalize_profile,
    )


def _make_center_extraction_def_with_bg(
    radius: float = 0.2,
    bg_inner: float = 0.3,
    bg_outer: float = 0.45,
    profile_mode: str = "global_median",
) -> WeightedExtractionDefinition:
    """Extraction definition centered at slit midpoint with background annulus."""
    return WeightedExtractionDefinition(
        center_frac=0.5,
        radius_frac=radius,
        background_inner=bg_inner,
        background_outer=bg_outer,
        profile_mode=profile_mode,
    )


# ===========================================================================
# 1. WeightedExtractionDefinition: construction and validation
# ===========================================================================


class TestWeightedExtractionDefinitionConstruction:
    """Tests for WeightedExtractionDefinition construction and properties."""

    def test_basic_construction(self):
        """WeightedExtractionDefinition can be constructed with minimal args."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert ed.center_frac == 0.5
        assert ed.radius_frac == 0.1
        assert ed.background_inner is None
        assert ed.background_outer is None
        assert ed.profile_mode == "global_median"
        assert ed.normalize_profile is True
        assert ed.use_variance_model is False
        assert ed.variance_model is None

    def test_has_background_false_by_default(self):
        """has_background is False when no background region is defined."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert not ed.has_background

    def test_has_background_true_when_defined(self):
        """has_background is True when background region is defined."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.1,
            background_inner=0.2,
            background_outer=0.4,
        )
        assert ed.has_background

    def test_columnwise_profile_mode(self):
        """profile_mode='columnwise' is accepted."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.1, profile_mode="columnwise"
        )
        assert ed.profile_mode == "columnwise"

    def test_normalize_profile_false(self):
        """normalize_profile=False is accepted."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.1, normalize_profile=False
        )
        assert ed.normalize_profile is False

    def test_use_variance_model_with_model(self):
        """use_variance_model=True with a model is accepted."""
        vmodel = VarianceModelDefinition(
            read_noise_electron=10.0, gain_e_per_adu=2.0
        )
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.1,
            use_variance_model=True,
            variance_model=vmodel,
        )
        assert ed.use_variance_model is True
        assert ed.variance_model is vmodel

    def test_invalid_center_below_zero(self):
        """ValueError if center_frac < 0."""
        with pytest.raises(ValueError, match="center_frac"):
            WeightedExtractionDefinition(center_frac=-0.1, radius_frac=0.1)

    def test_invalid_center_above_one(self):
        """ValueError if center_frac > 1."""
        with pytest.raises(ValueError, match="center_frac"):
            WeightedExtractionDefinition(center_frac=1.1, radius_frac=0.1)

    def test_invalid_radius_zero(self):
        """ValueError if radius_frac == 0."""
        with pytest.raises(ValueError, match="radius_frac"):
            WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.0)

    def test_invalid_radius_negative(self):
        """ValueError if radius_frac < 0."""
        with pytest.raises(ValueError, match="radius_frac"):
            WeightedExtractionDefinition(center_frac=0.5, radius_frac=-0.1)

    def test_background_inner_only_raises(self):
        """ValueError if only background_inner is provided."""
        with pytest.raises(ValueError, match="both"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, background_inner=0.2
            )

    def test_background_outer_only_raises(self):
        """ValueError if only background_outer is provided."""
        with pytest.raises(ValueError, match="both"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, background_outer=0.4
            )

    def test_background_inner_le_radius_raises(self):
        """ValueError if background_inner <= radius_frac."""
        with pytest.raises(ValueError, match="background_inner"):
            WeightedExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.2,
                background_inner=0.2,
                background_outer=0.4,
            )

    def test_background_outer_le_inner_raises(self):
        """ValueError if background_outer <= background_inner."""
        with pytest.raises(ValueError, match="background_outer"):
            WeightedExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.1,
                background_inner=0.3,
                background_outer=0.2,
            )

    def test_invalid_profile_mode_raises(self):
        """ValueError if profile_mode is not a recognized string."""
        with pytest.raises(ValueError, match="profile_mode"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, profile_mode="magic_psf"
            )

    def test_use_variance_model_true_without_model_raises(self):
        """ValueError if use_variance_model=True but variance_model is None."""
        with pytest.raises(ValueError, match="use_variance_model"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, use_variance_model=True
            )

    def test_variance_model_without_flag_raises(self):
        """ValueError if variance_model provided but use_variance_model=False."""
        vmodel = VarianceModelDefinition(
            read_noise_electron=10.0, gain_e_per_adu=2.0
        )
        with pytest.raises(ValueError, match="use_variance_model"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, variance_model=vmodel
            )


# ===========================================================================
# 2. WeightedExtractedOrderSpectrum: construction
# ===========================================================================


class TestWeightedExtractedOrderSpectrumConstruction:
    """Unit tests for WeightedExtractedOrderSpectrum construction and access."""

    def _make_minimal(self) -> WeightedExtractedOrderSpectrum:
        n = 32
        n_ap = 5
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.2)
        return WeightedExtractedOrderSpectrum(
            order=311,
            wavelength_um=np.linspace(1.64, 1.69, n),
            flux=np.ones(n),
            variance=np.ones(n) * 0.1,
            profile=np.ones((n_ap, n)) / n_ap,
            aperture=ed,
            method="horne_weighted",
            n_pixels_used=n_ap,
        )

    def test_construction(self):
        """WeightedExtractedOrderSpectrum can be constructed."""
        sp = self._make_minimal()
        assert sp.order == 311

    def test_n_spectral(self):
        """n_spectral matches length of wavelength_um."""
        sp = self._make_minimal()
        assert sp.n_spectral == 32

    def test_variance_present(self):
        """variance is always present (not None)."""
        sp = self._make_minimal()
        assert sp.variance is not None

    def test_method_stored(self):
        """method is stored correctly."""
        sp = self._make_minimal()
        assert sp.method == "horne_weighted"

    def test_n_pixels_used_stored(self):
        """n_pixels_used is stored correctly."""
        sp = self._make_minimal()
        assert sp.n_pixels_used == 5

    def test_weights_default_none(self):
        """weights is None by default."""
        sp = self._make_minimal()
        assert sp.weights is None

    def test_weights_stored_when_provided(self):
        """weights can be stored when provided."""
        n = 32
        n_ap = 5
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.2)
        sp = WeightedExtractedOrderSpectrum(
            order=311,
            wavelength_um=np.linspace(1.64, 1.69, n),
            flux=np.ones(n),
            variance=np.ones(n) * 0.1,
            profile=np.ones((n_ap, n)) / n_ap,
            aperture=ed,
            method="horne_weighted",
            n_pixels_used=n_ap,
            weights=np.ones((n_ap, n)) * 0.5,
        )
        assert sp.weights is not None
        assert sp.weights.shape == (n_ap, n)


# ===========================================================================
# 3. WeightedExtractedSpectrumSet: construction and interface
# ===========================================================================


class TestWeightedExtractedSpectrumSetConstruction:
    """Tests for WeightedExtractedSpectrumSet interface."""

    def _make_set(self) -> WeightedExtractedSpectrumSet:
        n = 32
        n_ap = 5
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.2)
        spectra = [
            WeightedExtractedOrderSpectrum(
                order=311 + i * 4,
                wavelength_um=np.linspace(1.55 + i * 0.05, 1.59 + i * 0.05, n),
                flux=np.ones(n) * (i + 1),
                variance=np.ones(n) * 0.1,
                profile=np.ones((n_ap, n)) / n_ap,
                aperture=ed,
                method="horne_weighted",
                n_pixels_used=n_ap,
            )
            for i in range(3)
        ]
        return WeightedExtractedSpectrumSet(mode="H1_test", spectra=spectra)

    def test_mode_stored(self):
        """mode is stored correctly."""
        ss = self._make_set()
        assert ss.mode == "H1_test"

    def test_n_orders(self):
        """n_orders matches the number of spectra."""
        ss = self._make_set()
        assert ss.n_orders == 3

    def test_orders_list(self):
        """orders returns the list of order numbers."""
        ss = self._make_set()
        assert ss.orders == [311, 315, 319]

    def test_get_order_existing(self):
        """get_order returns the correct spectrum for a known order."""
        ss = self._make_set()
        sp = ss.get_order(311)
        assert sp.order == 311

    def test_get_order_missing_raises(self):
        """get_order raises KeyError for an unknown order."""
        ss = self._make_set()
        with pytest.raises(KeyError):
            ss.get_order(999)

    def test_empty_set(self):
        """An empty WeightedExtractedSpectrumSet is valid."""
        ss = WeightedExtractedSpectrumSet(mode="H1_test")
        assert ss.n_orders == 0
        assert ss.orders == []


# ===========================================================================
# 4. extract_weighted_optimal: basic behaviour
# ===========================================================================


class TestExtractWeightedOptimalBasic:
    """Basic behaviour tests for extract_weighted_optimal."""

    def test_returns_weighted_extracted_spectrum_set(self):
        """extract_weighted_optimal returns a WeightedExtractedSpectrumSet."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        assert isinstance(result, WeightedExtractedSpectrumSet)

    def test_mode_propagated(self):
        """mode is propagated from the rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        assert result.mode == ros.mode

    def test_n_orders_matches(self):
        """n_orders matches the input rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        assert result.n_orders == ros.n_orders

    def test_orders_list_matches(self):
        """orders list matches the input rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        assert result.orders == ros.orders

    def test_output_flux_shape(self):
        """Extracted flux has shape (n_spectral,)."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.flux.shape == (ro.n_spectral,)

    def test_output_variance_shape(self):
        """Extracted variance has shape (n_spectral,)."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.variance.shape == (ro.n_spectral,)

    def test_variance_always_present(self):
        """variance is always present (never None)."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.variance is not None

    def test_variance_nonnegative(self):
        """Variance is non-negative everywhere it is finite."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result = extract_weighted_optimal(ros, ed, variance_image=var_image)
        for sp in result.spectra:
            finite = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite] >= 0.0)

    def test_profile_shape(self):
        """Profile has shape (n_ap_spatial, n_spectral)."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(radius=0.2)
        result = extract_weighted_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.profile.shape[1] == ro.n_spectral
            dist = np.abs(ro.spatial_frac - ed.center_frac)
            n_ap = int(np.sum(dist <= ed.radius_frac))
            assert sp.profile.shape[0] == n_ap

    def test_wavelength_axis_propagated(self):
        """wavelength_um is identical to the input rectified order wavelength."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            np.testing.assert_array_equal(sp.wavelength_um, ro.wavelength_um)

    def test_wavelength_axis_is_copy(self):
        """wavelength_um is a copy, not a reference to the input array."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            original = ro.wavelength_um.copy()
            ro.wavelength_um[0] += 999.0
            np.testing.assert_array_equal(sp.wavelength_um, original)

    def test_method_is_horne_weighted(self):
        """method field is always 'horne_weighted'."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.method == "horne_weighted"

    def test_n_pixels_used_positive(self):
        """n_pixels_used > 0 for a reasonable aperture."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(radius=0.2)
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.n_pixels_used > 0

    def test_weights_stored(self):
        """weights 2-D array is stored and has correct shape."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(radius=0.2)
        result = extract_weighted_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.weights is not None
            # Same shape as profile
            assert sp.weights.shape == sp.profile.shape


# ===========================================================================
# 5. extract_weighted_optimal: unit variance — formula verification
# ===========================================================================


class TestExtractWeightedUnitVariance:
    """Tests verifying the Horne formula with unit variance."""

    def test_unit_variance_flux_matches_stage12_formula(self):
        """With unit variance, flux equals the Stage-12 formula sum(P*F)/sum(P^2).

        For unit variance V=1:
            flux_1d = sum(P*F/1) / sum(P^2/1) = sum(P*F) / sum(P^2)
        This is identical to the Stage-12 unweighted estimator.
        """
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL
        fill = 3.0

        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=fill
        )
        ed = _make_center_extraction_def(radius=0.2, normalize_profile=False)
        var_image = np.ones((n_spatial, n_spectral))

        result = extract_weighted_optimal(
            ros, ed, variance_image=var_image, subtract_background=False
        )

        # For constant F=c and profile P=F=c (not normalized):
        # Stage-12: sum(P*F)/sum(P^2) = sum(c^2)/sum(c^2) = 1.0
        # Horne with V=1: sum(P*F/1)/sum(P^2/1) = same = 1.0
        for sp in result.spectra:
            np.testing.assert_allclose(sp.flux, 1.0, rtol=1e-12)

    def test_unit_variance_variance_is_inverse_sum_p_squared(self):
        """With unit variance, var_1d = 1 / sum(P^2).

        For unit variance V=1:
            var_1d = 1 / sum(P^2 / 1) = 1 / sum(P^2)
        """
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL

        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=2.0
        )
        ed = _make_center_extraction_def(radius=0.2, normalize_profile=True)
        var_image = np.ones((n_spatial, n_spectral))

        result = extract_weighted_optimal(
            ros, ed, variance_image=var_image, subtract_background=False
        )
        for sp in result.spectra:
            expected_var = 1.0 / np.nansum(sp.profile ** 2, axis=0)
            np.testing.assert_allclose(sp.variance, expected_var, rtol=1e-12)

    def test_no_variance_source_uses_unit_variance(self):
        """When no variance source is provided, unit variance is used as fallback.

        The fallback is documented behaviour: V=1 everywhere.
        """
        ros = _make_synthetic_rectified_order_set(fill_value=2.0)
        ed = _make_center_extraction_def(radius=0.2, normalize_profile=True)

        result_no_var = extract_weighted_optimal(
            ros, ed, subtract_background=False
        )
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result_unit_var = extract_weighted_optimal(
            ros, ed, variance_image=var_image, subtract_background=False
        )

        for sp_no, sp_unit in zip(result_no_var.spectra, result_unit_var.spectra):
            np.testing.assert_allclose(sp_no.flux, sp_unit.flux, rtol=1e-12)
            np.testing.assert_allclose(
                sp_no.variance, sp_unit.variance, rtol=1e-12
            )


# ===========================================================================
# 6. extract_weighted_optimal: non-uniform variance
# ===========================================================================


class TestExtractWeightedNonUniformVariance:
    """Tests for non-uniform variance weighting."""

    def test_non_uniform_variance_changes_flux(self):
        """Non-uniform variance changes the extracted flux compared to unit variance.

        Use a 2-row aperture with anti-correlated spectral patterns so the
        global_median profile (estimated once from all columns) does NOT
        match the per-column flux values.  In that case, shifting V between
        rows shifts the effective weighting between rows, producing a
        different extracted spectrum.

        Setup:
          Row 0: flux = 10 in the first half of spectral columns, 1 in second.
          Row 1: flux =  1 in the first half,                    10 in second.

        Global-median profile (both rows have the same median ≈ 5.5):
          P[0] = P[1] ≈ 0.5 (after normalization).

        Unit variance — col 0 example:
          flux = (0.5·10 + 0.5·1) / (0.25 + 0.25) = 11

        Low variance for row 0 (V[0]=0.01, V[1]=100) — col 0:
          flux = (0.5·10/0.01 + 0.5·1/100) / (0.25/0.01 + 0.25/100) ≈ 20

        The two variance choices produce measurably different spectra.
        """
        n_spectral = 16
        spatial_frac = np.array([0.4, 0.6])  # two aperture rows

        # Anti-correlated spectral patterns.
        flux_2d = np.ones((2, n_spectral))
        flux_2d[0, :n_spectral // 2] = 10.0   # row 0 high in first half
        flux_2d[0, n_spectral // 2:] = 1.0
        flux_2d[1, :n_spectral // 2] = 1.0    # row 1 high in second half
        flux_2d[1, n_spectral // 2:] = 10.0

        ro = RectifiedOrder(
            order=311,
            order_index=0,
            wavelength_um=np.linspace(1.55, 1.59, n_spectral),
            spatial_frac=spatial_frac,
            flux=flux_2d.copy(),
            source_image_shape=(_NROWS, _NCOLS),
        )
        ros = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=[ro],
            source_image_shape=(_NROWS, _NCOLS),
        )

        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.15,
            profile_mode="global_median",
            normalize_profile=True,
        )

        # Unit variance: equal weights for both rows.
        var_unit = np.ones((2, n_spectral))
        result_unit = extract_weighted_optimal(
            ros, ed, variance_image=var_unit, subtract_background=False
        )

        # Row 0 gets much lower variance → dominates the extraction.
        var_nonuniform = np.ones((2, n_spectral))
        var_nonuniform[0, :] = 0.01   # row 0: low variance
        var_nonuniform[1, :] = 100.0  # row 1: high variance
        result_nonuniform = extract_weighted_optimal(
            ros, ed, variance_image=var_nonuniform, subtract_background=False
        )

        # With non-uniform variance, row 0 dominates: first-half columns
        # (row-0 has flux=10) should yield higher flux than unit-V result.
        flux_unit = result_unit.spectra[0].flux
        flux_nonu = result_nonuniform.spectra[0].flux

        assert not np.allclose(flux_unit, flux_nonu, rtol=1e-3), (
            "Non-uniform variance should change the extracted flux."
        )
        # First-half columns: row 0 has high flux (10), row 0 has low V → higher.
        assert np.all(flux_nonu[:n_spectral // 2] > flux_unit[:n_spectral // 2]), (
            "Low-variance row 0 (high-flux) should raise extraction in first half"
        )

    def test_lower_variance_pixels_weighted_higher(self):
        """Pixels with lower variance get higher weight in extraction.

        Set up an aperture with two rows: one with low variance (good pixel)
        and one with high variance (noisy pixel).  The extracted flux should
        be closer to the low-variance pixel's value.
        """
        n_spectral = 16
        spatial_frac = np.array([0.3, 0.7])  # two aperture rows

        # Row 0 (low variance): flux = 10
        # Row 1 (high variance): flux = 1
        flux_2d = np.array([[10.0] * n_spectral, [1.0] * n_spectral])

        ro = RectifiedOrder(
            order=311,
            order_index=0,
            wavelength_um=np.linspace(1.55, 1.59, n_spectral),
            spatial_frac=spatial_frac,
            flux=flux_2d,
            source_image_shape=(_NROWS, _NCOLS),
        )
        ros = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=[ro],
            source_image_shape=(_NROWS, _NCOLS),
        )

        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.25,
            profile_mode="global_median",
            normalize_profile=True,
        )

        # Low variance for row 0 (high flux), high variance for row 1 (low flux).
        var_image = np.ones((2, n_spectral))
        var_image[0, :] = 0.01   # row 0: low variance
        var_image[1, :] = 100.0  # row 1: high variance

        result = extract_weighted_optimal(
            ros, ed, variance_image=var_image, subtract_background=False
        )
        sp = result.spectra[0]

        # With V[0] << V[1], the low-variance row dominates.
        # The normalized profile P = [P0, P1] with P0 + P1 = 1.
        # weights = [P0^2/0.01, P1^2/100]
        # Since P0 ≈ P1 ≈ 0.5, weight0 ≈ 25*weight1 >> weight1.
        # flux ≈ F[0] = 10.
        # Check that flux > 5 (closer to 10 than to 1).
        assert np.all(sp.flux[np.isfinite(sp.flux)] > 5.0), (
            "Low-variance pixels should dominate: flux should be closer to "
            f"high-flux row value of 10; got {sp.flux}"
        )

    def test_variance_positivity_with_nonuniform_variance(self):
        """Variance is non-negative with non-uniform variance image."""
        ros = _make_synthetic_rectified_order_set(fill_value=3.0)
        ed = _make_center_extraction_def(radius=0.2)
        var_image = np.random.default_rng(42).uniform(
            0.1, 5.0, (_N_SPATIAL, _N_SPECTRAL)
        )
        result = extract_weighted_optimal(ros, ed, variance_image=var_image)
        for sp in result.spectra:
            finite = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite] >= 0.0)


# ===========================================================================
# 7. extract_weighted_optimal: background subtraction
# ===========================================================================


class TestExtractWeightedBackgroundSubtraction:
    """Tests for background subtraction in extract_weighted_optimal."""

    def test_background_subtracted_flux(self):
        """Background-subtracted extraction gives expected result.

        Aperture pixels: object_fill.  Background pixels: bg_fill.
        After subtraction, aperture pixels = object_fill - bg_fill.
        With normalize_profile=False, P = F (constant), so
        flux_1d = sum(P*F/V) / sum(P^2/V) = 1.0 for unit variance.
        """
        n_spatial = 20
        n_spectral = 16
        center = 0.5
        radius = 0.2
        bg_inner = radius + 0.05
        bg_outer = radius + 0.25

        spatial_frac = np.linspace(0.0, 1.0, n_spatial)
        dist = np.abs(spatial_frac - center)

        object_fill = 10.0
        bg_fill = 3.0
        flux_2d = np.full((n_spatial, n_spectral), bg_fill)
        ap_mask = dist <= radius
        flux_2d[ap_mask, :] = object_fill

        ro = RectifiedOrder(
            order=311,
            order_index=0,
            wavelength_um=np.linspace(1.55, 1.59, n_spectral),
            spatial_frac=spatial_frac,
            flux=flux_2d,
            source_image_shape=(_NROWS, _NCOLS),
        )
        ros = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=[ro],
            source_image_shape=(_NROWS, _NCOLS),
        )

        ed = WeightedExtractionDefinition(
            center_frac=center,
            radius_frac=radius,
            background_inner=bg_inner,
            background_outer=bg_outer,
            profile_mode="global_median",
            normalize_profile=False,
        )
        var_image = np.ones((n_spatial, n_spectral))

        result = extract_weighted_optimal(
            ros, ed, variance_image=var_image, subtract_background=True
        )
        sp = result.spectra[0]

        # After subtraction: aperture = object_fill - bg_fill = constant.
        # P = F = constant → flux_1d = 1.0 for unit variance.
        np.testing.assert_allclose(sp.flux, 1.0, rtol=1e-12)

    def test_no_subtraction_when_flag_false(self):
        """Setting subtract_background=False does not subtract background."""
        fill = 7.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ed = _make_center_extraction_def_with_bg()
        result = extract_weighted_optimal(ros, ed, subtract_background=False)
        for sp in result.spectra:
            assert np.all(np.isfinite(sp.flux))

    def test_background_subtraction_variance_nonnegative(self):
        """Variance is non-negative when background subtraction is active."""
        ros = _make_synthetic_rectified_order_set(fill_value=5.0)
        ed = _make_center_extraction_def_with_bg()
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result = extract_weighted_optimal(
            ros, ed, variance_image=var_image, subtract_background=True
        )
        for sp in result.spectra:
            finite = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite] >= 0.0)


# ===========================================================================
# 8. extract_weighted_optimal: NaN / inf / mask handling
# ===========================================================================


class TestExtractWeightedNaNHandling:
    """Tests for NaN / inf / mask handling in extract_weighted_optimal."""

    def test_all_nan_aperture_column_produces_nan(self):
        """A spectral column where all aperture pixels are NaN → NaN output."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(radius=0.2)
        ros.rectified_orders[0].flux[:, 0] = np.nan
        result = extract_weighted_optimal(ros, ed)
        assert np.isnan(result.spectra[0].flux[0])
        assert np.isnan(result.spectra[0].variance[0])

    def test_all_nan_aperture_column_variance_is_nan(self):
        """Columns with no valid support yield NaN variance."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(radius=0.2)
        ros.rectified_orders[0].flux[:, 0] = np.nan
        result = extract_weighted_optimal(ros, ed)
        assert np.isnan(result.spectra[0].variance[0])

    def test_partial_nan_row_excluded(self):
        """A single NaN aperture row does not invalidate the entire column."""
        n_spatial = 20
        n_spectral = _N_SPECTRAL
        fill = 5.0
        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=fill
        )
        ed = _make_center_extraction_def(radius=0.2)
        ro = ros.rectified_orders[0]
        dist = np.abs(ro.spatial_frac - ed.center_frac)
        ap_indices = np.where(dist <= ed.radius_frac)[0]
        ro.flux[ap_indices[0], :] = np.nan
        result = extract_weighted_optimal(
            ros, ed, subtract_background=False
        )
        assert np.all(np.isfinite(result.spectra[0].flux))

    def test_inf_in_data_excluded(self):
        """Inf pixels in data are treated as bad and excluded."""
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL
        fill = 3.0
        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=fill
        )
        ed = _make_center_extraction_def(radius=0.2)
        ro = ros.rectified_orders[0]
        dist = np.abs(ro.spatial_frac - ed.center_frac)
        ap_indices = np.where(dist <= ed.radius_frac)[0]
        ro.flux[ap_indices[0], :] = np.inf
        result = extract_weighted_optimal(
            ros, ed, subtract_background=False
        )
        assert np.all(np.isfinite(result.spectra[0].flux))

    def test_empty_aperture_gives_all_nan(self):
        """An aperture too narrow to capture any pixels → all-NaN output."""
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=1e-6)
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert np.all(np.isnan(sp.flux))
            assert np.all(np.isnan(sp.variance))

    def test_user_mask_excludes_pixels(self):
        """User-supplied mask excludes marked pixels.

        Masking one aperture row should give the same result as setting that
        row to NaN in the data.
        """
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL
        fill = 6.0
        ros_mask = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=fill
        )
        ros_nan = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=fill
        )
        ed = _make_center_extraction_def(radius=0.2)

        dist = np.abs(ros_mask.rectified_orders[0].spatial_frac - ed.center_frac)
        ap_indices = np.where(dist <= ed.radius_frac)[0]

        bad_mask = np.zeros((n_spatial, n_spectral), dtype=bool)
        bad_mask[ap_indices[0], :] = True

        ros_nan.rectified_orders[0].flux[ap_indices[0], :] = np.nan

        result_mask = extract_weighted_optimal(
            ros_mask, ed, mask=bad_mask, subtract_background=False
        )
        result_nan = extract_weighted_optimal(
            ros_nan, ed, subtract_background=False
        )

        np.testing.assert_allclose(
            result_mask.spectra[0].flux,
            result_nan.spectra[0].flux,
            rtol=1e-12,
        )

    def test_all_aperture_masked_column_returns_nan(self):
        """Entire aperture column masked → NaN flux and NaN variance."""
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL
        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=5.0
        )
        ed = _make_center_extraction_def(radius=0.2)

        bad_mask = np.zeros((n_spatial, n_spectral), dtype=bool)
        dist = np.abs(ros.rectified_orders[0].spatial_frac - ed.center_frac)
        ap_indices = np.where(dist <= ed.radius_frac)[0]
        bad_mask[np.ix_(ap_indices, [3])] = True

        var_image = np.ones((n_spatial, n_spectral))
        result = extract_weighted_optimal(
            ros, ed, mask=bad_mask, variance_image=var_image
        )
        sp = result.spectra[0]
        assert np.isnan(sp.flux[3])
        assert np.isnan(sp.variance[3])
        assert np.all(np.isfinite(sp.flux[np.arange(n_spectral) != 3]))

    def test_nan_in_variance_image_excluded(self):
        """NaN in variance image causes that pixel to be excluded."""
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL
        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=2.0
        )
        ed = _make_center_extraction_def(radius=0.2)

        var_image = np.ones((n_spatial, n_spectral))
        dist = np.abs(ros.rectified_orders[0].spatial_frac - ed.center_frac)
        ap_indices = np.where(dist <= ed.radius_frac)[0]
        # NaN variance for all aperture pixels in column 0.
        var_image[np.ix_(ap_indices, [0])] = np.nan

        result = extract_weighted_optimal(
            ros, ed, variance_image=var_image
        )
        sp = result.spectra[0]
        # All aperture variance is NaN for col 0 → no valid support → NaN.
        assert np.isnan(sp.flux[0])
        assert np.isnan(sp.variance[0])
        # Other columns should be finite.
        assert np.all(np.isfinite(sp.flux[1:]))


# ===========================================================================
# 9. Parametric shape-consistency tests
# ===========================================================================


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 10), (64, 40), (128, 50)])
def test_output_shapes_parametric(n_spectral, n_spatial):
    """Extracted spectra have correct shapes for various (n_spectral, n_spatial)."""
    ros = _make_synthetic_rectified_order_set(
        n_spectral=n_spectral, n_spatial=n_spatial
    )
    for profile_mode in ("global_median", "columnwise"):
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            profile_mode=profile_mode,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.flux.shape == (n_spectral,)
            assert sp.variance.shape == (n_spectral,)
            assert sp.wavelength_um.shape == (n_spectral,)
            assert sp.profile.shape[1] == n_spectral


# ===========================================================================
# 10. Error tests
# ===========================================================================


class TestExtractWeightedErrors:
    """Error paths for extract_weighted_optimal."""

    def test_raises_on_empty_rectified_order_set(self):
        """ValueError if rectified_orders has no orders."""
        empty_ros = RectifiedOrderSet(mode="test")
        ed = _make_center_extraction_def()
        with pytest.raises(ValueError, match="empty"):
            extract_weighted_optimal(empty_ros, ed)

    def test_raises_on_non_2d_flux(self):
        """ValueError if a flux array is not 2-D."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        ros.rectified_orders[0].flux = np.ones(_N_SPECTRAL)
        with pytest.raises(ValueError, match="2-D"):
            extract_weighted_optimal(ros, ed)

    def test_raises_on_variance_shape_mismatch(self):
        """ValueError if variance_image shape does not match flux shape."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        bad_variance = np.ones((_N_SPATIAL + 1, _N_SPECTRAL))
        with pytest.raises(ValueError, match="variance_image shape"):
            extract_weighted_optimal(ros, ed, variance_image=bad_variance)

    def test_raises_on_mask_shape_mismatch(self):
        """ValueError if mask shape does not match flux shape."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        bad_mask = np.zeros((_N_SPATIAL + 1, _N_SPECTRAL), dtype=bool)
        with pytest.raises(ValueError, match="mask shape"):
            extract_weighted_optimal(ros, ed, mask=bad_mask)

    def test_invalid_profile_mode_raises_at_definition_time(self):
        """ValueError if profile_mode is invalid (caught at definition time)."""
        with pytest.raises(ValueError, match="profile_mode"):
            WeightedExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.1,
                profile_mode="bad_mode",
            )

    def test_use_variance_model_true_without_model_raises(self):
        """ValueError if use_variance_model=True but variance_model is None."""
        with pytest.raises(ValueError, match="use_variance_model"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, use_variance_model=True
            )


# ===========================================================================
# 11. Variance source priority tests
# ===========================================================================


class TestExtractWeightedVariancePriority:
    """Tests for the variance source priority chain."""

    def _make_variance_model(self) -> VarianceModelDefinition:
        return VarianceModelDefinition(
            read_noise_electron=10.0, gain_e_per_adu=2.0
        )

    def test_variance_image_overrides_variance_model(self):
        """Explicit variance_image takes priority over variance_model."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL, fill_value=1.0
        )
        ed = _make_center_extraction_def(normalize_profile=True)
        vmodel = self._make_variance_model()
        var_image = np.full((_N_SPATIAL, _N_SPECTRAL), 5.0)

        result_model = extract_weighted_optimal(ros, ed, variance_model=vmodel)
        result_image = extract_weighted_optimal(
            ros, ed, variance_image=var_image, variance_model=vmodel
        )
        result_image_only = extract_weighted_optimal(
            ros, ed, variance_image=var_image
        )

        # variance_image + model should equal variance_image only (model ignored).
        for sp_img, sp_both in zip(
            result_image_only.spectra, result_image.spectra
        ):
            np.testing.assert_allclose(
                sp_both.variance, sp_img.variance, rtol=1e-12
            )

        # Model-only result should differ from image-only result (different V).
        for sp_model, sp_img in zip(result_model.spectra, result_image_only.spectra):
            assert not np.allclose(
                sp_model.variance, sp_img.variance, rtol=1e-6
            ), "variance_model should give different result from variance_image"

    def test_variance_model_function_arg_produces_variance(self):
        """variance_model function argument produces non-trivial variance."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        vmodel = self._make_variance_model()
        result = extract_weighted_optimal(ros, ed, variance_model=vmodel)
        for sp in result.spectra:
            assert sp.variance is not None
            finite = np.isfinite(sp.variance)
            assert np.any(finite)

    def test_definition_variance_model_used_as_fallback(self):
        """Definition-embedded variance model is used when no other source."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        vmodel = self._make_variance_model()
        ed_with_model = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            use_variance_model=True,
            variance_model=vmodel,
        )
        ed_no_model = _make_center_extraction_def()

        result_with = extract_weighted_optimal(ros, ed_with_model)
        result_without = extract_weighted_optimal(ros, ed_no_model)

        # The embedded variance model should differ from unit variance.
        for sp_with, sp_without in zip(result_with.spectra, result_without.spectra):
            assert not np.allclose(
                sp_with.variance, sp_without.variance, rtol=1e-6
            ), "Definition-embedded model should differ from unit variance fallback"

    def test_function_variance_model_overrides_definition_model(self):
        """Function-level variance_model takes priority over definition's model."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        vmodel_def = VarianceModelDefinition(
            read_noise_electron=5.0, gain_e_per_adu=1.0
        )
        vmodel_fn = VarianceModelDefinition(
            read_noise_electron=50.0, gain_e_per_adu=1.0
        )
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            use_variance_model=True,
            variance_model=vmodel_def,
        )

        # Same definition, but function-level model has very different read noise.
        result_def = extract_weighted_optimal(ros, ed)
        result_fn = extract_weighted_optimal(ros, ed, variance_model=vmodel_fn)

        # Function-level model (read_noise=50) should dominate over def's (5).
        for sp_def, sp_fn in zip(result_def.spectra, result_fn.spectra):
            assert not np.allclose(
                sp_def.variance, sp_fn.variance, rtol=1e-3
            ), "Function-level variance_model should override definition's model"


# ===========================================================================
# 12. Profile mode tests
# ===========================================================================


class TestExtractWeightedProfileModes:
    """Tests for global_median vs columnwise profile modes."""

    def test_global_median_constant_flux(self):
        """For constant flux, global_median and columnwise give same flux."""
        fill = 4.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))

        ed_gm = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.2, profile_mode="global_median"
        )
        ed_cw = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.2, profile_mode="columnwise"
        )

        result_gm = extract_weighted_optimal(
            ros, ed_gm, variance_image=var_image, subtract_background=False
        )
        result_cw = extract_weighted_optimal(
            ros, ed_cw, variance_image=var_image, subtract_background=False
        )

        for sp_gm, sp_cw in zip(result_gm.spectra, result_cw.spectra):
            np.testing.assert_allclose(sp_gm.flux, sp_cw.flux, rtol=1e-12)

    def test_global_median_profile_constant_across_columns(self):
        """global_median produces a profile that is the same in every column."""
        ros = _make_synthetic_rectified_order_set(fill_value=5.0)
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.2, profile_mode="global_median"
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            first_col = sp.profile[:, 0]
            for j in range(1, sp.n_spectral):
                np.testing.assert_allclose(
                    sp.profile[:, j], first_col, rtol=1e-12
                )


# ===========================================================================
# 13. Smoke test on real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1WeightedExtractionSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset.

    Chain:
      1. Flat-order tracing (Stage 1)
      2. Arc-line tracing (Stage 2)
      3. Provisional wavelength mapping (Stage 3)
      4. Coefficient-surface refinement (Stage 5)
      5. Rectification indices (Stage 6)
      6. Rectified orders (Stage 7)
      7. Proto-Horne weighted optimal extraction (Stage 17)
    """

    @pytest.fixture(scope="class")
    def h1_weighted_result(self):
        """WeightedExtractedSpectrumSet from the real H1 calibration chain."""
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

        # Stage 17: proto-Horne weighted optimal extraction
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            background_inner=0.3,
            background_outer=0.45,
            profile_mode="global_median",
        )
        var_model = VarianceModelDefinition(
            read_noise_electron=11.0,  # approximate iSHELL H1 read noise
            gain_e_per_adu=1.8,        # approximate iSHELL gain
        )
        extracted = extract_weighted_optimal(
            rectified,
            ed,
            variance_model=var_model,
            subtract_background=True,
        )

        return extracted, rectified

    def test_returns_weighted_extracted_spectrum_set(self, h1_weighted_result):
        """extract_weighted_optimal returns a WeightedExtractedSpectrumSet."""
        extracted, _ = h1_weighted_result
        assert isinstance(extracted, WeightedExtractedSpectrumSet)

    def test_n_orders_matches_rectified(self, h1_weighted_result):
        """n_orders matches the rectified order set."""
        extracted, rectified = h1_weighted_result
        assert extracted.n_orders == rectified.n_orders

    def test_n_orders_positive(self, h1_weighted_result):
        """n_orders > 0."""
        extracted, _ = h1_weighted_result
        assert extracted.n_orders > 0

    def test_wavelength_axes_monotonic(self, h1_weighted_result):
        """wavelength_um is monotonic for every extracted spectrum."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            diffs = np.diff(sp.wavelength_um)
            assert np.all(diffs > 0) or np.all(diffs < 0), (
                f"Order {sp.order}: wavelength axis is not monotonic"
            )

    def test_flux_shape_matches_wavelength(self, h1_weighted_result):
        """flux shape equals wavelength_um shape for every spectrum."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            assert sp.flux.shape == sp.wavelength_um.shape

    def test_variance_shape_matches_wavelength(self, h1_weighted_result):
        """variance shape equals wavelength_um shape for every spectrum."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            assert sp.variance.shape == sp.wavelength_um.shape

    def test_variance_nonnegative(self, h1_weighted_result):
        """Variance is non-negative everywhere it is finite."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            finite = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite] >= 0.0)

    def test_flux_finite_for_most_columns(self, h1_weighted_result):
        """Most flux values are finite (some NaN from rectification is OK)."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            finite_frac = np.mean(np.isfinite(sp.flux))
            assert finite_frac > 0.5, (
                f"Order {sp.order}: fewer than 50% of flux values are finite "
                f"({finite_frac:.1%})"
            )

    def test_variance_finite_for_most_columns(self, h1_weighted_result):
        """Most variance values are finite."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            finite_frac = np.mean(np.isfinite(sp.variance))
            assert finite_frac > 0.5, (
                f"Order {sp.order}: fewer than 50% of variance values are "
                f"finite ({finite_frac:.1%})"
            )

    def test_orders_list_matches_rectified(self, h1_weighted_result):
        """orders list matches the rectified order set."""
        extracted, rectified = h1_weighted_result
        assert extracted.orders == rectified.orders

    def test_get_order_works_for_all_orders(self, h1_weighted_result):
        """get_order succeeds for every order in the set."""
        extracted, _ = h1_weighted_result
        for order in extracted.orders:
            sp = extracted.get_order(order)
            assert sp.order == order

    def test_n_pixels_used_positive(self, h1_weighted_result):
        """n_pixels_used > 0 for every order."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            assert sp.n_pixels_used > 0

    def test_profile_has_correct_n_spectral(self, h1_weighted_result):
        """profile has the correct number of spectral columns."""
        extracted, rectified = h1_weighted_result
        for sp, ro in zip(extracted.spectra, rectified.rectified_orders):
            assert sp.profile.shape[1] == ro.n_spectral

    def test_weights_shape_matches_profile(self, h1_weighted_result):
        """weights 2-D array shape matches profile shape."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            assert sp.weights is not None
            assert sp.weights.shape == sp.profile.shape

    def test_method_is_horne_weighted(self, h1_weighted_result):
        """method is 'horne_weighted' for every order."""
        extracted, _ = h1_weighted_result
        for sp in extracted.spectra:
            assert sp.method == "horne_weighted"


# ===========================================================================
# 14. Iterative outlier rejection: definition validation (error tests)
# ===========================================================================


class TestIterativeRejectionDefinitionValidation:
    """Validation tests for the outlier-rejection fields in WeightedExtractionDefinition."""

    def test_default_reject_outliers_is_false(self):
        """reject_outliers defaults to False."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert ed.reject_outliers is False

    def test_default_max_iterations(self):
        """max_iterations defaults to 3."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert ed.max_iterations == 3

    def test_default_sigma_clip(self):
        """sigma_clip defaults to 5.0."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert ed.sigma_clip == 5.0

    def test_default_min_valid_pixels(self):
        """min_valid_pixels defaults to 2."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert ed.min_valid_pixels == 2

    def test_reject_outliers_enabled(self):
        """reject_outliers=True with valid parameters is accepted."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.1,
            reject_outliers=True,
            max_iterations=5,
            sigma_clip=3.0,
            min_valid_pixels=3,
        )
        assert ed.reject_outliers is True
        assert ed.max_iterations == 5
        assert ed.sigma_clip == 3.0
        assert ed.min_valid_pixels == 3

    def test_invalid_sigma_clip_zero(self):
        """ValueError if sigma_clip == 0."""
        with pytest.raises(ValueError, match="sigma_clip"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, sigma_clip=0.0
            )

    def test_invalid_sigma_clip_negative(self):
        """ValueError if sigma_clip < 0."""
        with pytest.raises(ValueError, match="sigma_clip"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, sigma_clip=-1.0
            )

    def test_invalid_max_iterations_zero(self):
        """ValueError if max_iterations == 0."""
        with pytest.raises(ValueError, match="max_iterations"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, max_iterations=0
            )

    def test_invalid_max_iterations_negative(self):
        """ValueError if max_iterations < 0."""
        with pytest.raises(ValueError, match="max_iterations"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, max_iterations=-1
            )

    def test_invalid_min_valid_pixels_zero(self):
        """ValueError if min_valid_pixels == 0."""
        with pytest.raises(ValueError, match="min_valid_pixels"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, min_valid_pixels=0
            )

    def test_invalid_min_valid_pixels_negative(self):
        """ValueError if min_valid_pixels < 0."""
        with pytest.raises(ValueError, match="min_valid_pixels"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, min_valid_pixels=-1
            )

    def test_max_iterations_one_is_valid(self):
        """max_iterations=1 is the minimum valid value."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.1, max_iterations=1
        )
        assert ed.max_iterations == 1

    def test_min_valid_pixels_one_is_valid(self):
        """min_valid_pixels=1 is the minimum valid value."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.1, min_valid_pixels=1
        )
        assert ed.min_valid_pixels == 1


# ===========================================================================
# 15. Iterative outlier rejection: synthetic behaviour tests
# ===========================================================================

# Outlier test parameters: all rows in aperture so n_ap == _N_AP_OUTLIER.
_N_AP_OUTLIER = 20
_N_SPEC_OUTLIER = 32
_OUTLIER_ROW = 10
_OUTLIER_COL = 5
_TRUE_FLUX = 100.0
_OUTLIER_AMP = 50.0


def _make_outlier_ros(
    n_spatial: int = _N_AP_OUTLIER,
    n_spectral: int = _N_SPEC_OUTLIER,
    true_flux: float = _TRUE_FLUX,
    outlier_amp: float = _OUTLIER_AMP,
    outlier_row: int = _OUTLIER_ROW,
    outlier_col: int = _OUTLIER_COL,
):
    """Synthetic uniform-profile data with one injected outlier pixel.

    Returns (ros, variance) where variance is a unit-variance image.
    With a box profile of n_spatial pixels and radius_frac=0.5, the
    extracted flux at clean columns equals true_flux.
    """
    spatial_frac = np.linspace(0.0, 1.0, n_spatial)
    pixel_flux = true_flux / n_spatial
    flux_2d = np.full((n_spatial, n_spectral), pixel_flux)
    flux_2d[outlier_row, outlier_col] += outlier_amp
    variance = np.ones((n_spatial, n_spectral))

    order = RectifiedOrder(
        order=311,
        order_index=0,
        wavelength_um=np.linspace(1.55, 1.60, n_spectral),
        spatial_frac=spatial_frac,
        flux=flux_2d,
        source_image_shape=(200, 800),
    )
    ros = RectifiedOrderSet(
        mode="H1_test",
        rectified_orders=[order],
        source_image_shape=(200, 800),
    )
    return ros, variance


def _ed_outlier(sigma_clip: float = 5.0, **kwargs) -> WeightedExtractionDefinition:
    """Full-aperture definition with rejection enabled."""
    return WeightedExtractionDefinition(
        center_frac=0.5,
        radius_frac=0.5,
        reject_outliers=True,
        sigma_clip=sigma_clip,
        **kwargs,
    )


def _ed_no_outlier() -> WeightedExtractionDefinition:
    """Full-aperture definition with rejection disabled."""
    return WeightedExtractionDefinition(
        center_frac=0.5,
        radius_frac=0.5,
        reject_outliers=False,
    )


class TestIterativeOutlierRejection:
    """Synthetic-data tests for the iterative outlier-rejection scaffold."""

    def test_outlier_rejected_when_enabled(self):
        """A single extreme outlier pixel is rejected when reject_outliers=True."""
        ros, variance = _make_outlier_ros()
        result = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance, subtract_background=False
        )
        sp = result.spectra[0]
        assert sp.n_rejected_pixels >= 1
        assert sp.final_mask is not None
        assert sp.final_mask[_OUTLIER_ROW, _OUTLIER_COL], (
            f"Outlier pixel ({_OUTLIER_ROW}, {_OUTLIER_COL}) should be in final_mask."
        )

    def test_outlier_not_rejected_when_disabled(self):
        """The same outlier pixel is not rejected when reject_outliers=False."""
        ros, variance = _make_outlier_ros()
        result = extract_weighted_optimal(
            ros, _ed_no_outlier(), variance_image=variance, subtract_background=False
        )
        sp = result.spectra[0]
        assert sp.n_rejected_pixels == 0
        assert sp.final_mask is None
        assert sp.n_iterations_used == 0

    def test_rejection_reduces_outlier_influence(self):
        """Rejection reduces the influence of a cosmic-ray-like pixel on extracted flux.

        With a box profile of n=20 pixels and unit variance, the extracted flux
        at the outlier column is true_flux + outlier_amp without rejection and
        approximately true_flux after rejection.
        """
        ros, variance = _make_outlier_ros()

        r_rej = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance, subtract_background=False
        )
        r_no = extract_weighted_optimal(
            ros, _ed_no_outlier(), variance_image=variance, subtract_background=False
        )

        flux_rej = r_rej.spectra[0].flux[_OUTLIER_COL]
        flux_no = r_no.spectra[0].flux[_OUTLIER_COL]

        assert flux_rej < flux_no, (
            f"Rejected flux ({flux_rej:.2f}) should be less than "
            f"non-rejected flux ({flux_no:.2f})."
        )
        np.testing.assert_allclose(flux_rej, _TRUE_FLUX, atol=1.0)

    def test_output_shapes_identical(self):
        """reject_outliers=True does not change any output array shapes."""
        ros, variance = _make_outlier_ros()
        r_rej = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance, subtract_background=False
        )
        r_no = extract_weighted_optimal(
            ros, _ed_no_outlier(), variance_image=variance, subtract_background=False
        )
        sp_rej = r_rej.spectra[0]
        sp_no = r_no.spectra[0]
        assert sp_rej.flux.shape == sp_no.flux.shape
        assert sp_rej.variance.shape == sp_no.variance.shape
        assert sp_rej.profile.shape == sp_no.profile.shape

    def test_non_outlier_columns_unchanged(self):
        """Columns without outliers produce the same flux regardless of rejection flag."""
        ros, variance = _make_outlier_ros()
        r_rej = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance, subtract_background=False
        )
        r_no = extract_weighted_optimal(
            ros, _ed_no_outlier(), variance_image=variance, subtract_background=False
        )
        sp_rej = r_rej.spectra[0]
        sp_no = r_no.spectra[0]
        for col in range(sp_rej.n_spectral):
            if col == _OUTLIER_COL:
                continue
            np.testing.assert_allclose(
                sp_rej.flux[col], sp_no.flux[col], rtol=1e-10,
                err_msg=f"Column {col} should be unaffected by rejection.",
            )

    def test_n_iterations_used_positive_when_rejection_enabled(self):
        """n_iterations_used >= 1 when reject_outliers=True."""
        ros, variance = _make_outlier_ros()
        result = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance, subtract_background=False
        )
        assert result.spectra[0].n_iterations_used >= 1

    def test_final_mask_shape_matches_aperture(self):
        """final_mask shape is (n_ap_spatial, n_spectral)."""
        ros, variance = _make_outlier_ros(
            n_spatial=_N_AP_OUTLIER, n_spectral=_N_SPEC_OUTLIER
        )
        result = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance, subtract_background=False
        )
        sp = result.spectra[0]
        assert sp.final_mask is not None
        assert sp.final_mask.shape == (_N_AP_OUTLIER, _N_SPEC_OUTLIER)

    def test_early_stop_when_no_rejections(self):
        """Iteration stops early when no outliers exist in the data.

        For clean data, the first iteration finds no candidates and sets
        n_iterations_used=1 (the check pass itself counts as one iteration).
        """
        ros = _make_synthetic_rectified_order_set(fill_value=2.0)
        variance = np.ones((_N_SPATIAL, _N_SPECTRAL))
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            reject_outliers=True,
            max_iterations=10,
            sigma_clip=5.0,
        )
        result = extract_weighted_optimal(
            ros, ed, variance_image=variance, subtract_background=False
        )
        for sp in result.spectra:
            assert sp.n_rejected_pixels == 0
            assert sp.n_iterations_used == 1, (
                "Expected early stop after 1 iteration for clean data."
            )
            assert sp.final_mask is not None and not np.any(sp.final_mask), (
                "final_mask should exist but have no True entries for clean data."
            )

    def test_n_iterations_bounded_by_max_iterations(self):
        """n_iterations_used never exceeds max_iterations."""
        ros, variance = _make_outlier_ros()
        max_iter = 2
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.5,
            reject_outliers=True,
            max_iterations=max_iter,
            sigma_clip=5.0,
        )
        result = extract_weighted_optimal(
            ros, ed, variance_image=variance, subtract_background=False
        )
        assert result.spectra[0].n_iterations_used <= max_iter

    def test_min_valid_pixels_prevents_rejection(self):
        """Rejection is blocked when it would leave fewer than min_valid_pixels."""
        ros, variance = _make_outlier_ros(n_spatial=_N_AP_OUTLIER)
        # Require ALL pixels to remain valid → no pixel can ever be rejected.
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.5,
            reject_outliers=True,
            sigma_clip=5.0,
            min_valid_pixels=_N_AP_OUTLIER,
        )
        result = extract_weighted_optimal(
            ros, ed, variance_image=variance, subtract_background=False
        )
        assert result.spectra[0].n_rejected_pixels == 0, (
            "min_valid_pixels == n_ap should prevent any rejection."
        )

    def test_min_valid_pixels_one_allows_rejection(self):
        """With min_valid_pixels=1 the outlier pixel is still rejected."""
        ros, variance = _make_outlier_ros(n_spatial=_N_AP_OUTLIER)
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.5,
            reject_outliers=True,
            sigma_clip=5.0,
            min_valid_pixels=1,
        )
        result = extract_weighted_optimal(
            ros, ed, variance_image=variance, subtract_background=False
        )
        assert result.spectra[0].n_rejected_pixels >= 1

    def test_user_mask_and_iterative_rejection_combine(self):
        """User-masked pixels and iteratively rejected pixels are both excluded."""
        ros, variance = _make_outlier_ros()
        user_mask = np.zeros((_N_AP_OUTLIER, _N_SPEC_OUTLIER), dtype=bool)
        user_mask[0, :] = True  # mask entire first row

        result = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance,
            mask=user_mask, subtract_background=False
        )
        sp = result.spectra[0]

        # Row 0 was user-masked, not iteratively rejected → not in final_mask.
        assert sp.final_mask is not None
        assert not np.any(sp.final_mask[0, :]), (
            "Row 0 was user-masked, not iteratively rejected; "
            "final_mask should be False for it."
        )
        # The injected outlier should appear in final_mask.
        assert sp.final_mask[_OUTLIER_ROW, _OUTLIER_COL], (
            "Outlier pixel should appear in final_mask."
        )

    def test_user_mask_does_not_count_as_rejected(self):
        """n_rejected_pixels counts only iterative rejections, not user-masked pixels."""
        ros, variance = _make_outlier_ros()
        user_mask = np.zeros((_N_AP_OUTLIER, _N_SPEC_OUTLIER), dtype=bool)
        user_mask[1, :] = True  # mask entire row 1

        r_no = extract_weighted_optimal(
            ros, _ed_no_outlier(), variance_image=variance,
            mask=user_mask, subtract_background=False
        )
        assert r_no.spectra[0].n_rejected_pixels == 0

    def test_no_rejection_bookkeeping_zero_when_disabled(self):
        """When reject_outliers=False, all bookkeeping fields are 0 or None."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        assert ed.reject_outliers is False
        result = extract_weighted_optimal(ros, ed, subtract_background=False)
        for sp in result.spectra:
            assert sp.n_rejected_pixels == 0
            assert sp.final_mask is None
            assert sp.n_iterations_used == 0

    def test_spectra_finite_at_outlier_column_after_rejection(self):
        """Extracted flux/variance at the outlier column remain finite after rejection."""
        ros, variance = _make_outlier_ros()
        result = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance, subtract_background=False
        )
        sp = result.spectra[0]
        assert np.isfinite(sp.flux[_OUTLIER_COL]), (
            "flux at the outlier column should still be finite after rejection."
        )
        assert np.isfinite(sp.variance[_OUTLIER_COL]), (
            "variance at the outlier column should still be finite after rejection."
        )

    def test_n_rejected_pixels_consistent_with_final_mask(self):
        """n_rejected_pixels equals the number of True entries in final_mask."""
        ros, variance = _make_outlier_ros()
        result = extract_weighted_optimal(
            ros, _ed_outlier(), variance_image=variance, subtract_background=False
        )
        sp = result.spectra[0]
        assert sp.final_mask is not None
        assert sp.n_rejected_pixels == int(np.sum(sp.final_mask))


# ===========================================================================
# 16. H1 smoke tests for iterative rejection
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1IterativeRejectionSmokeTest:
    """Smoke tests for reject_outliers=True/False using real H1 calibration data."""

    @pytest.fixture(scope="class")
    def h1_rejection_results(self):
        """Run weighted extraction with and without rejection on real H1 data."""
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

        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES, col_range=(650, 1550)
        )
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)
        wci = read_wavecalinfo("H1")
        ll = read_line_list("H1")
        wav_map = fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )
        surface = fit_refined_coefficient_surface(wav_map, disp_degree=3)
        rect_indices = build_rectification_indices(geom, surface, wav_map=wav_map)
        with fits.open(_H1_FLAT_FILES[0]) as hdul:
            detector_image = hdul[0].data.astype(float)
        rectified = build_rectified_orders(detector_image, rect_indices)

        var_model = VarianceModelDefinition(
            read_noise_electron=11.0,
            gain_e_per_adu=1.8,
        )
        base = dict(
            center_frac=0.5,
            radius_frac=0.2,
            background_inner=0.3,
            background_outer=0.45,
            profile_mode="global_median",
        )
        ed_no_rej = WeightedExtractionDefinition(**base, reject_outliers=False)
        ed_with_rej = WeightedExtractionDefinition(
            **base, reject_outliers=True, sigma_clip=5.0, max_iterations=3
        )
        r_no = extract_weighted_optimal(
            rectified, ed_no_rej, variance_model=var_model, subtract_background=True
        )
        r_yes = extract_weighted_optimal(
            rectified, ed_with_rej, variance_model=var_model, subtract_background=True
        )
        return r_no, r_yes, rectified

    def test_no_crashes(self, h1_rejection_results):
        """Both reject_outliers=False and True complete without error."""
        r_no, r_yes, _ = h1_rejection_results
        assert r_no is not None
        assert r_yes is not None

    def test_output_shapes_identical(self, h1_rejection_results):
        """Output array shapes are unchanged by reject_outliers=True."""
        r_no, r_yes, _ = h1_rejection_results
        for sp_no, sp_yes in zip(r_no.spectra, r_yes.spectra):
            assert sp_no.flux.shape == sp_yes.flux.shape
            assert sp_no.variance.shape == sp_yes.variance.shape
            assert sp_no.profile.shape == sp_yes.profile.shape

    def test_extracted_spectra_finite_where_expected(self, h1_rejection_results):
        """Extracted spectra remain finite for most columns when rejection is on."""
        _, r_yes, _ = h1_rejection_results
        for sp in r_yes.spectra:
            finite_frac = np.mean(np.isfinite(sp.flux))
            assert finite_frac > 0.5, (
                f"Order {sp.order}: fewer than 50% of flux values are finite "
                f"({finite_frac:.1%}) with rejection enabled."
            )

    def test_n_rejected_pixels_reported_consistently(self, h1_rejection_results):
        """n_rejected_pixels is a non-negative integer for every order."""
        r_no, r_yes, _ = h1_rejection_results
        for sp in r_no.spectra:
            assert sp.n_rejected_pixels == 0
        for sp in r_yes.spectra:
            assert isinstance(sp.n_rejected_pixels, int)
            assert sp.n_rejected_pixels >= 0

    def test_final_mask_none_without_rejection(self, h1_rejection_results):
        """final_mask is None when reject_outliers=False."""
        r_no, _, _ = h1_rejection_results
        for sp in r_no.spectra:
            assert sp.final_mask is None

    def test_n_iterations_zero_without_rejection(self, h1_rejection_results):
        """n_iterations_used is 0 when reject_outliers=False."""
        r_no, _, _ = h1_rejection_results
        for sp in r_no.spectra:
            assert sp.n_iterations_used == 0

    def test_n_iterations_positive_with_rejection(self, h1_rejection_results):
        """n_iterations_used >= 1 for every order when reject_outliers=True."""
        _, r_yes, _ = h1_rejection_results
        for sp in r_yes.spectra:
            assert sp.n_iterations_used >= 1

    def test_final_mask_shape_with_rejection(self, h1_rejection_results):
        """final_mask shape is (n_ap_spatial, n_spectral) when reject_outliers=True."""
        _, r_yes, rectified = h1_rejection_results
        for sp, ro in zip(r_yes.spectra, rectified.rectified_orders):
            assert sp.final_mask is not None
            assert sp.final_mask.shape[1] == ro.n_spectral

    def test_final_mask_consistent_with_n_rejected(self, h1_rejection_results):
        """n_rejected_pixels equals the number of True entries in final_mask."""
        _, r_yes, _ = h1_rejection_results
        for sp in r_yes.spectra:
            assert sp.final_mask is not None
            assert sp.n_rejected_pixels == int(np.sum(sp.final_mask))


# ===========================================================================


# ===========================================================================
# 17. Stage 19: Profile re-estimation — definition-level tests
# ===========================================================================


class TestReestimateProfileDefinition:
    """Tests for the reestimate_profile field on WeightedExtractionDefinition."""

    def test_reestimate_profile_default_false(self):
        """reestimate_profile defaults to False."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert ed.reestimate_profile is False

    def test_reestimate_profile_can_be_set_true(self):
        """reestimate_profile=True is accepted."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.1,
            reject_outliers=True, reestimate_profile=True,
        )
        assert ed.reestimate_profile is True

    def test_reestimate_profile_true_without_reject_outliers_accepted(self):
        """reestimate_profile=True with reject_outliers=False is silently accepted.

        The flag is simply ignored when reject_outliers=False.
        """
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.1,
            reject_outliers=False, reestimate_profile=True,
        )
        assert ed.reestimate_profile is True


# ===========================================================================
# 18. Stage 19: Profile re-estimation — bookkeeping field tests
# ===========================================================================


class TestReestimateProfileBookkeeping:
    """Tests for profile_reestimated / initial_profile on WeightedExtractedOrderSpectrum."""

    def test_profile_reestimated_false_no_rejection(self):
        """profile_reestimated is False when reject_outliers=False."""
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5, reject_outliers=False,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        for sp in result.spectra:
            assert sp.profile_reestimated is False

    def test_initial_profile_none_no_rejection(self):
        """initial_profile is None when reject_outliers=False."""
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5, reject_outliers=False,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        for sp in result.spectra:
            assert sp.initial_profile is None

    def test_profile_reestimated_false_when_reestimate_false(self):
        """profile_reestimated is False when reestimate_profile=False."""
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=False,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        for sp in result.spectra:
            assert sp.profile_reestimated is False

    def test_initial_profile_none_when_reestimate_false(self):
        """initial_profile is None when reestimate_profile=False."""
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=False,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        for sp in result.spectra:
            assert sp.initial_profile is None

    def test_profile_reestimated_true_when_outlier_rejected(self):
        """profile_reestimated is True after an outlier is actually rejected.

        Uses the default _make_outlier_ros() outlier amplitude (50.0) which
        produces a single-pixel outlier detectable by sigma_clip=3.0 with a
        global_median profile (the outlier doesn't dominate the row median,
        so the profile is unbiased, and the residual at the outlier pixel is
        large).
        """
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True, sigma_clip=3.0,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        sp = result.spectra[0]
        # The large outlier must be rejected for re-estimation to happen.
        assert sp.n_rejected_pixels > 0, "Expected outlier to be rejected."
        assert sp.profile_reestimated is True

    def test_initial_profile_stored_when_reestimated(self):
        """initial_profile is a 2-D array when profile_reestimated is True."""
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True, sigma_clip=3.0,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        sp = result.spectra[0]
        assert sp.n_rejected_pixels > 0, "Expected outlier to be rejected."
        assert sp.initial_profile is not None
        assert sp.initial_profile.ndim == 2
        assert sp.initial_profile.shape == sp.profile.shape

    def test_initial_profile_is_independent_copy(self):
        """initial_profile is a separate array, not aliased to profile."""
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True, sigma_clip=3.0,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        sp = result.spectra[0]
        assert sp.n_rejected_pixels > 0, "Expected outlier to be rejected."
        assert sp.initial_profile is not None
        # Must be distinct Python objects (not aliases of each other).
        assert sp.initial_profile is not sp.profile

    def test_initial_profile_and_profile_identical_for_robust_global_median(self):
        """With global_median mode and a single-pixel outlier, initial and final
        profile values are numerically identical.

        global_median estimates the profile via per-row nanmedian.  A single
        outlier pixel (one column out of n_spectral) does not affect the
        row median, so re-estimation after rejection produces the same values.
        This is expected and correct scaffold behavior: re-estimation is
        conservative and does no harm.
        """
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True, sigma_clip=3.0,
            profile_mode="global_median",
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        sp = result.spectra[0]
        assert sp.n_rejected_pixels > 0, "Expected outlier to be rejected."
        assert sp.initial_profile is not None
        # Values should be numerically identical (global_median is robust
        # to a single-pixel outlier out of n_spectral columns).
        np.testing.assert_allclose(
            sp.initial_profile, sp.profile, rtol=1e-12, equal_nan=True
        )

    def test_profile_reestimated_false_when_no_pixels_rejected(self):
        """profile_reestimated is False when no outlier pixels are rejected (clean data)."""
        ros = _make_synthetic_rectified_order_set()
        variance = np.ones((_N_SPATIAL, _N_SPECTRAL))
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.2,
            reject_outliers=True, reestimate_profile=True,
            sigma_clip=100.0,  # very loose: nothing will be rejected
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        for sp in result.spectra:
            assert sp.n_rejected_pixels == 0
            assert sp.profile_reestimated is False
            assert sp.initial_profile is None


# ===========================================================================
# 19. Stage 19: Profile re-estimation — behavioral correctness tests
# ===========================================================================


class TestReestimateProfileBehavior:
    """Behavioral tests for profile re-estimation in the rejection loop."""

    def test_reestimate_false_preserves_stage18_behavior(self):
        """reestimate_profile=False gives identical results to Stage 18."""
        ros, variance = _make_outlier_ros()

        ed_stage18 = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=False,
        )
        ed_stage19 = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=False,
        )
        r18 = extract_weighted_optimal(ros, ed_stage18, variance_image=variance)
        r19 = extract_weighted_optimal(ros, ed_stage19, variance_image=variance)

        sp18 = r18.spectra[0]
        sp19 = r19.spectra[0]
        np.testing.assert_array_equal(sp18.flux, sp19.flux)
        np.testing.assert_array_equal(sp18.variance, sp19.variance)
        assert sp18.n_rejected_pixels == sp19.n_rejected_pixels

    def test_clean_data_identical_with_and_without_reestimation(self):
        """Clean data gives identical flux/variance whether or not re-estimation is active."""
        ros = _make_synthetic_rectified_order_set()
        variance = np.ones((_N_SPATIAL, _N_SPECTRAL))

        ed_no_re = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.2,
            reject_outliers=True, reestimate_profile=False, sigma_clip=5.0,
        )
        ed_with_re = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.2,
            reject_outliers=True, reestimate_profile=True, sigma_clip=5.0,
        )
        r_no = extract_weighted_optimal(ros, ed_no_re, variance_image=variance)
        r_with = extract_weighted_optimal(ros, ed_with_re, variance_image=variance)

        for sp_no, sp_with in zip(r_no.spectra, r_with.spectra):
            np.testing.assert_allclose(
                sp_no.flux, sp_with.flux, rtol=1e-10, equal_nan=True
            )
            np.testing.assert_allclose(
                sp_no.variance, sp_with.variance, rtol=1e-10, equal_nan=True
            )

    def test_reestimation_produces_valid_output_with_contaminated_data(self):
        """Profile re-estimation runs without error and produces finite output
        in the presence of a single-pixel outlier.

        With global_median profile mode, a single-pixel outlier is detected
        (not absorbed by the robust median), rejected, and then the profile is
        re-estimated.  The re-estimated profile equals the initial profile
        numerically (global_median is robust), but the outlier pixel is
        correctly excluded from the final extraction.
        """
        ros, variance = _make_outlier_ros()

        ed_reest = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True,
            sigma_clip=3.0,
        )
        result = extract_weighted_optimal(ros, ed_reest, variance_image=variance)
        sp = result.spectra[0]

        # Must have rejected the outlier pixel.
        assert sp.n_rejected_pixels > 0

        # Flux at clean columns must be finite.
        clean_cols = [c for c in range(_N_SPEC_OUTLIER) if c != _OUTLIER_COL]
        for c in clean_cols:
            assert np.isfinite(sp.flux[c]), f"Flux at clean column {c} should be finite."

        # Flux at the outlier column should be finite and close to the true value
        # (outlier pixel was rejected so it no longer inflates the result).
        assert np.isfinite(sp.flux[_OUTLIER_COL])
        assert abs(sp.flux[_OUTLIER_COL] - _TRUE_FLUX) < 5.0, (
            f"Outlier-column flux {sp.flux[_OUTLIER_COL]:.2f} should be close to "
            f"true flux {_TRUE_FLUX} after rejection."
        )

    def test_reestimation_corrects_outlier_column_flux(self):
        """After rejection with re-estimation, the outlier-column flux improves
        compared to the no-rejection baseline.

        When reject_outliers=False, the extracted flux at the outlier column is
        biased upward by the outlier pixel.  When reject_outliers=True (with or
        without re-estimation), the outlier is removed and the flux recovers.
        """
        ros, variance = _make_outlier_ros()

        ed_no_rej = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5, reject_outliers=False,
        )
        ed_with_reest = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True, sigma_clip=3.0,
        )
        r_no_rej = extract_weighted_optimal(ros, ed_no_rej, variance_image=variance)
        r_with_reest = extract_weighted_optimal(ros, ed_with_reest, variance_image=variance)

        sp_no = r_no_rej.spectra[0]
        sp_re = r_with_reest.spectra[0]

        # Without rejection, outlier column flux is inflated.
        assert sp_no.flux[_OUTLIER_COL] > _TRUE_FLUX + 10.0, (
            "Expected inflated flux at outlier column without rejection."
        )

        # With rejection+re-estimation, flux is closer to the true value.
        assert abs(sp_re.flux[_OUTLIER_COL] - _TRUE_FLUX) < 5.0, (
            f"Flux at outlier column should recover after rejection. "
            f"Got {sp_re.flux[_OUTLIER_COL]:.2f}, expected ~{_TRUE_FLUX}."
        )

    def test_profile_normalization_holds_after_reestimation(self):
        """Profile columns sum to 1.0 (approximately) after re-estimation."""
        ros, variance = _make_outlier_ros()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True,
            sigma_clip=3.0, normalize_profile=True,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        sp = result.spectra[0]

        # For each finite column of the final profile, sum should be ≈ 1.
        profile = sp.profile
        for col in range(profile.shape[1]):
            col_vals = profile[:, col]
            finite_vals = col_vals[np.isfinite(col_vals)]
            if len(finite_vals) > 0:
                col_sum = float(np.sum(finite_vals))
                assert abs(col_sum - 1.0) < 1e-9, (
                    f"Profile column {col} sum = {col_sum:.6f} != 1.0 "
                    f"after re-estimation."
                )

    def test_output_shapes_unchanged_with_reestimation(self):
        """Output shapes are identical with and without re-estimation."""
        ros, variance = _make_outlier_ros()
        ed_no = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=False, sigma_clip=3.0,
        )
        ed_yes = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True, sigma_clip=3.0,
        )
        r_no = extract_weighted_optimal(ros, ed_no, variance_image=variance)
        r_yes = extract_weighted_optimal(ros, ed_yes, variance_image=variance)

        sp_no = r_no.spectra[0]
        sp_yes = r_yes.spectra[0]
        assert sp_no.flux.shape == sp_yes.flux.shape
        assert sp_no.variance.shape == sp_yes.variance.shape
        assert sp_no.profile.shape == sp_yes.profile.shape


# ===========================================================================
# 20. Stage 19: Profile re-estimation — stability / edge-case tests
# ===========================================================================


class TestReestimateProfileStability:
    """Stability and edge-case tests for profile re-estimation."""

    def test_nearly_empty_aperture_no_crash(self):
        """Re-estimation with a nearly empty aperture does not crash.

        When the aperture contains only 1 or 2 spatial pixels (radius very
        small), re-estimation should not raise.
        """
        n_spatial = 20
        n_spectral = 16
        spatial_frac = np.linspace(0.0, 1.0, n_spatial)
        flux_2d = np.full((n_spatial, n_spectral), 2.0)
        # Inject a mild outlier near the center.
        center_idx = n_spatial // 2
        flux_2d[center_idx, 3] += 50.0
        variance = np.ones((n_spatial, n_spectral))

        order = RectifiedOrder(
            order=311, order_index=0,
            wavelength_um=np.linspace(1.55, 1.60, n_spectral),
            spatial_frac=spatial_frac,
            flux=flux_2d,
            source_image_shape=(200, 800),
        )
        ros = RectifiedOrderSet(
            mode="H1_test", rectified_orders=[order],
            source_image_shape=(200, 800),
        )

        # radius_frac=0.01 means only ~1 spatial pixel is in the aperture.
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.01,
            reject_outliers=True, reestimate_profile=True, sigma_clip=3.0,
        )
        # Must not raise.
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        sp = result.spectra[0]
        assert sp.flux.shape == (n_spectral,)

    def test_min_valid_pixels_guard_with_reestimation(self):
        """min_valid_pixels guard still works when re-estimation is active.

        With min_valid_pixels equal to the aperture size, no pixels can
        ever be rejected (it would leave too few valid pixels).
        """
        n_spatial = 5  # small aperture for control
        n_spectral = 16
        spatial_frac = np.linspace(0.0, 1.0, n_spatial)
        flux_2d = np.full((n_spatial, n_spectral), 2.0)
        # Inject a large outlier.
        flux_2d[2, 4] += 1000.0
        variance = np.ones((n_spatial, n_spectral))

        order = RectifiedOrder(
            order=311, order_index=0,
            wavelength_um=np.linspace(1.55, 1.60, n_spectral),
            spatial_frac=spatial_frac,
            flux=flux_2d,
            source_image_shape=(200, 800),
        )
        ros = RectifiedOrderSet(
            mode="H1_test", rectified_orders=[order],
            source_image_shape=(200, 800),
        )

        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True,
            sigma_clip=1.0,  # very aggressive threshold
            min_valid_pixels=n_spatial,  # can't reject any pixel
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        sp = result.spectra[0]
        # Nothing should have been rejected (min_valid_pixels guard).
        assert sp.n_rejected_pixels == 0
        assert sp.profile_reestimated is False

    def test_early_stopping_still_works_with_reestimation(self):
        """Early stopping halts when no new pixels are rejected, even with re-estimation."""
        # Clean data: no pixels should be rejected.
        ros = _make_synthetic_rectified_order_set()
        variance = np.ones((_N_SPATIAL, _N_SPECTRAL))

        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.2,
            reject_outliers=True, reestimate_profile=True,
            sigma_clip=100.0,  # nothing will be rejected
            max_iterations=5,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        for sp in result.spectra:
            # Should stop after 1 iteration (nothing to reject in iteration 1).
            assert sp.n_iterations_used == 1
            assert sp.n_rejected_pixels == 0

    def test_max_iterations_bound_with_reestimation(self):
        """n_iterations_used does not exceed max_iterations with re-estimation active."""
        ros, variance = _make_outlier_ros()
        max_iter = 2
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True,
            sigma_clip=3.0, max_iterations=max_iter,
        )
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        for sp in result.spectra:
            assert sp.n_iterations_used <= max_iter

    def test_mask_handling_with_reestimation(self):
        """User mask is still respected when re-estimation is active."""
        ros, variance = _make_outlier_ros()
        n_spatial = _N_AP_OUTLIER
        n_spectral = _N_SPEC_OUTLIER

        # Mask entire column 0 in the aperture.
        mask = np.zeros((n_spatial, n_spectral), dtype=bool)
        mask[:, 0] = True

        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.5,
            reject_outliers=True, reestimate_profile=True, sigma_clip=5.0,
        )
        result = extract_weighted_optimal(
            ros, ed, variance_image=variance, mask=mask
        )
        sp = result.spectra[0]
        # Column 0 should have NaN flux (all pixels masked).
        assert np.isnan(sp.flux[0]), "Masked column should produce NaN flux."
        # Other columns should be finite.
        assert np.isfinite(sp.flux[1])

    def test_reestimation_no_crash_with_single_pixel_aperture(self):
        """Re-estimation does not crash when only 1 spatial pixel is in the aperture."""
        n_spatial = 10
        n_spectral = 8
        spatial_frac = np.linspace(0.0, 1.0, n_spatial)
        flux_2d = np.full((n_spatial, n_spectral), 5.0)
        variance = np.ones((n_spatial, n_spectral))

        order = RectifiedOrder(
            order=311, order_index=0,
            wavelength_um=np.linspace(1.55, 1.60, n_spectral),
            spatial_frac=spatial_frac,
            flux=flux_2d,
            source_image_shape=(200, 800),
        )
        ros = RectifiedOrderSet(
            mode="H1_test", rectified_orders=[order],
            source_image_shape=(200, 800),
        )

        # Very small radius: only 1 spatial pixel in the aperture.
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=1.0 / (2 * n_spatial),
            reject_outliers=True, reestimate_profile=True,
            sigma_clip=3.0, min_valid_pixels=1,
        )
        # Must not raise.
        result = extract_weighted_optimal(ros, ed, variance_image=variance)
        assert result is not None


# ===========================================================================
# 21. Stage 19: Profile re-estimation — H1 smoke tests
# ===========================================================================


@pytest.mark.skipif(not _HAVE_H1_DATA, reason="Real H1 LFS data not available")
class TestH1ProfileReestimationSmokeTest:
    """Smoke tests for reject_outliers=True with reestimate_profile True/False."""

    @pytest.fixture(scope="class")
    def h1_reestimation_results(self):
        """Run weighted extraction with fixed profile and re-estimated profile."""
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

        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES, col_range=(650, 1550)
        )
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)
        wci = read_wavecalinfo("H1")
        ll = read_line_list("H1")
        wav_map = fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )
        surface = fit_refined_coefficient_surface(wav_map, disp_degree=3)
        rect_indices = build_rectification_indices(geom, surface, wav_map=wav_map)
        with fits.open(_H1_FLAT_FILES[0]) as hdul:
            detector_image = hdul[0].data.astype(float)
        rectified = build_rectified_orders(detector_image, rect_indices)

        var_model = VarianceModelDefinition(
            read_noise_electron=11.0,
            gain_e_per_adu=1.8,
        )
        base = dict(
            center_frac=0.5,
            radius_frac=0.2,
            background_inner=0.3,
            background_outer=0.45,
            profile_mode="global_median",
            reject_outliers=True,
            sigma_clip=5.0,
            max_iterations=3,
        )
        ed_fixed = WeightedExtractionDefinition(**base, reestimate_profile=False)
        ed_reest = WeightedExtractionDefinition(**base, reestimate_profile=True)

        r_fixed = extract_weighted_optimal(
            rectified, ed_fixed, variance_model=var_model, subtract_background=True
        )
        r_reest = extract_weighted_optimal(
            rectified, ed_reest, variance_model=var_model, subtract_background=True
        )
        return r_fixed, r_reest, rectified

    def test_no_crashes(self, h1_reestimation_results):
        """Both reestimate_profile=False and True complete without error."""
        r_fixed, r_reest, _ = h1_reestimation_results
        assert r_fixed is not None
        assert r_reest is not None

    def test_output_shapes_identical(self, h1_reestimation_results):
        """Output array shapes are unchanged by reestimate_profile=True."""
        r_fixed, r_reest, _ = h1_reestimation_results
        for sp_f, sp_r in zip(r_fixed.spectra, r_reest.spectra):
            assert sp_f.flux.shape == sp_r.flux.shape
            assert sp_f.variance.shape == sp_r.variance.shape
            assert sp_f.profile.shape == sp_r.profile.shape

    def test_spectra_well_formed(self, h1_reestimation_results):
        """Spectra and variance arrays are well-formed with re-estimation."""
        _, r_reest, _ = h1_reestimation_results
        for sp in r_reest.spectra:
            finite_frac = np.mean(np.isfinite(sp.flux))
            assert finite_frac > 0.5, (
                f"Order {sp.order}: fewer than 50% of flux values are finite "
                f"with reestimate_profile=True ({finite_frac:.1%})."
            )
            # Variance non-negative where finite.
            fin_var = sp.variance[np.isfinite(sp.variance)]
            if len(fin_var) > 0:
                assert np.all(fin_var >= 0.0), (
                    f"Order {sp.order}: negative variance with reestimate_profile=True."
                )

    def test_bookkeeping_consistent(self, h1_reestimation_results):
        """profile_reestimated and initial_profile are consistently populated."""
        r_fixed, r_reest, _ = h1_reestimation_results
        for sp in r_fixed.spectra:
            # Fixed profile: no re-estimation.
            assert sp.profile_reestimated is False
            assert sp.initial_profile is None
        for sp in r_reest.spectra:
            # Re-estimation enabled.  If any pixels were rejected,
            # re-estimation should have happened.
            if sp.n_rejected_pixels > 0:
                assert sp.profile_reestimated is True
                assert sp.initial_profile is not None
                assert sp.initial_profile.shape == sp.profile.shape
            else:
                # No rejections → no re-estimation.
                assert sp.profile_reestimated is False
                assert sp.initial_profile is None

    def test_n_iterations_consistent(self, h1_reestimation_results):
        """n_iterations_used >= 1 for every order when reject_outliers=True."""
        _, r_reest, _ = h1_reestimation_results
        for sp in r_reest.spectra:
            assert sp.n_iterations_used >= 1


# ===========================================================================
# 22. WeightedExtractionDefinition: profile_source validation (Stage 20)
# ===========================================================================


class TestProfileSourceDefinitionValidation:
    """Tests for Stage-20 profile_source fields in WeightedExtractionDefinition."""

    def test_default_profile_source_is_empirical(self):
        """profile_source defaults to 'empirical'."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.2)
        assert ed.profile_source == "empirical"

    def test_smoothed_empirical_accepted(self):
        """profile_source='smoothed_empirical' is accepted."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=1.5,
        )
        assert ed.profile_source == "smoothed_empirical"
        assert ed.profile_smooth_sigma == 1.5

    def test_external_accepted_with_profile(self):
        """profile_source='external' with external_profile is accepted."""
        ext = np.ones((5, 32)) / 5.0
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            profile_source="external",
            external_profile=ext,
        )
        assert ed.profile_source == "external"
        assert ed.external_profile is ext

    def test_invalid_profile_source_raises(self):
        """ValueError if profile_source is not a recognized value."""
        with pytest.raises(ValueError, match="profile_source"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.2, profile_source="magic_psf"
            )

    def test_negative_profile_smooth_sigma_raises(self):
        """ValueError if profile_smooth_sigma < 0."""
        with pytest.raises(ValueError, match="profile_smooth_sigma"):
            WeightedExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.2,
                profile_source="smoothed_empirical",
                profile_smooth_sigma=-1.0,
            )

    def test_external_without_profile_raises(self):
        """ValueError if profile_source='external' but external_profile is None."""
        with pytest.raises(ValueError, match="external_profile"):
            WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.2, profile_source="external"
            )

    def test_external_profile_with_wrong_source_raises(self):
        """ValueError if external_profile provided but profile_source != 'external'."""
        ext = np.ones((5, 32)) / 5.0
        with pytest.raises(ValueError, match="profile_source"):
            WeightedExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.2,
                profile_source="empirical",
                external_profile=ext,
            )

    def test_default_profile_smooth_sigma_zero(self):
        """profile_smooth_sigma defaults to 0.0."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.2)
        assert ed.profile_smooth_sigma == 0.0

    def test_default_external_profile_none(self):
        """external_profile defaults to None."""
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.2)
        assert ed.external_profile is None

    def test_profile_smooth_sigma_zero_accepted(self):
        """profile_smooth_sigma=0 is accepted (no smoothing)."""
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=0.0,
        )
        assert ed.profile_smooth_sigma == 0.0


# ===========================================================================
# 23. profile_source='empirical' backward compatibility (Stage 20)
# ===========================================================================


class TestProfileSourceEmpirical:
    """profile_source='empirical' produces the same result as default behavior."""

    def test_empirical_matches_default(self):
        """Explicit profile_source='empirical' gives identical result to default."""
        ros = _make_synthetic_rectified_order_set()
        ed_default = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.3
        )
        ed_explicit = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.3, profile_source="empirical"
        )
        r_default = extract_weighted_optimal(ros, ed_default)
        r_explicit = extract_weighted_optimal(ros, ed_explicit)
        for sp_d, sp_e in zip(r_default.spectra, r_explicit.spectra):
            np.testing.assert_array_equal(sp_d.flux, sp_e.flux)
            np.testing.assert_array_equal(sp_d.variance, sp_e.variance)

    def test_empirical_profile_source_used_field(self):
        """profile_source_used is 'empirical' for default extraction."""
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.3)
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile_source_used == "empirical"

    def test_empirical_profile_smoothed_false(self):
        """profile_smoothed is False for empirical extraction."""
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.3)
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile_smoothed is False


# ===========================================================================
# 24. profile_source='smoothed_empirical' behavior (Stage 20)
# ===========================================================================


def _make_noisy_rectified_order_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
    rng=None,
) -> "RectifiedOrderSet":
    """Build a RectifiedOrderSet with Gaussian-noisy flux images."""
    if rng is None:
        rng = np.random.default_rng(42)
    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        noisy_flux = 10.0 + rng.standard_normal((n_spatial, n_spectral))
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=np.linspace(wav_start, wav_end, n_spectral),
                spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                flux=noisy_flux,
                source_image_shape=(_NROWS, _NCOLS),
            )
        )
    return RectifiedOrderSet(
        mode="H1_test",
        rectified_orders=orders,
        source_image_shape=(_NROWS, _NCOLS),
    )


class TestProfileSourceSmoothedEmpirical:
    """Tests for profile_source='smoothed_empirical' behavior."""

    def test_smoothed_empirical_output_shapes(self):
        """smoothed_empirical produces correct output shapes."""
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=1.0,
        )
        result = extract_weighted_optimal(ros, ed)
        assert result.n_orders == _N_ORDERS
        for sp in result.spectra:
            assert sp.flux.shape == (_N_SPECTRAL,)
            assert sp.variance.shape == (_N_SPECTRAL,)

    def test_smoothed_empirical_profile_source_used(self):
        """profile_source_used is 'smoothed_empirical'."""
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=1.0,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile_source_used == "smoothed_empirical"

    def test_smoothed_empirical_profile_smoothed_flag(self):
        """profile_smoothed is True when sigma > 0."""
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=1.0,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile_smoothed is True

    def test_smoothed_empirical_sigma_zero_matches_empirical(self):
        """smoothed_empirical with sigma=0 matches empirical exactly."""
        ros = _make_noisy_rectified_order_set()
        ed_emp = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.3, profile_source="empirical"
        )
        ed_smooth0 = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=0.0,
        )
        r_emp = extract_weighted_optimal(ros, ed_emp)
        r_sm0 = extract_weighted_optimal(ros, ed_smooth0)
        for sp_e, sp_s in zip(r_emp.spectra, r_sm0.spectra):
            np.testing.assert_allclose(sp_e.flux, sp_s.flux)
            np.testing.assert_allclose(sp_e.variance, sp_s.variance)

    def test_smoothed_profile_differs_from_empirical_in_noisy_case(self):
        """smoothed_empirical produces a different profile from empirical on noisy data."""
        rng = np.random.default_rng(17)
        ros = _make_noisy_rectified_order_set(rng=rng)
        ed_emp = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.3, profile_source="empirical"
        )
        ed_sm = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=2.0,
        )
        r_emp = extract_weighted_optimal(ros, ed_emp)
        r_sm = extract_weighted_optimal(ros, ed_sm)
        for sp_e, sp_s in zip(r_emp.spectra, r_sm.spectra):
            assert not np.allclose(sp_e.profile, sp_s.profile), \
                "Expected smoothed and empirical profiles to differ for noisy data."

    def test_smoothed_profile_reduces_single_pixel_outlier_sensitivity(self):
        """Smoothing reduces the effect of a single-pixel profile spike."""
        n_ap = 15
        n_spec = 32
        rng = np.random.default_rng(99)
        flux = np.ones((n_ap, n_spec)) * 10.0 + rng.standard_normal((n_ap, n_spec)) * 0.1
        spike_row = n_ap // 2
        flux[spike_row, :] += 50.0

        orders = [
            RectifiedOrder(
                order=311,
                order_index=0,
                wavelength_um=np.linspace(1.55, 1.59, n_spec),
                spatial_frac=np.linspace(0.0, 1.0, n_ap),
                flux=flux,
                source_image_shape=(_NROWS, _NCOLS),
            )
        ]
        ros = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=orders,
            source_image_shape=(_NROWS, _NCOLS),
        )
        ed_emp = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49
        )
        ed_sm = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.49,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=1.5,
        )
        r_emp = extract_weighted_optimal(ros, ed_emp)
        r_sm = extract_weighted_optimal(ros, ed_sm)
        sp_emp = r_emp.spectra[0]
        sp_sm = r_sm.spectra[0]
        assert sp_sm.profile.max() < sp_emp.profile.max(), (
            "Smoothing should reduce the spike in the profile."
        )

    def test_smoothed_empirical_no_smoothing_when_sigma_zero(self):
        """profile_smoothed=False when profile_smooth_sigma=0."""
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=0.0,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile_smoothed is False

    def test_smoothed_empirical_profile_normalized(self):
        """Smoothed profile columns sum to 1.0 where finite."""
        ros = _make_noisy_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.3,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=1.0,
            normalize_profile=True,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            col_sums = np.nansum(sp.profile, axis=0)
            finite_cols = np.isfinite(col_sums)
            np.testing.assert_allclose(
                col_sums[finite_cols], 1.0, atol=1e-12,
                err_msg=f"Order {sp.order}: smoothed profile columns should sum to 1.0",
            )

    def test_smoothed_empirical_reestimate_runs_without_crash(self):
        """smoothed_empirical + reestimate_profile + reject_outliers runs without crash."""
        ros = _make_synthetic_rectified_order_set(fill_value=5.0)
        ro0 = ros.rectified_orders[0]
        flux_with_outlier = ro0.flux.copy()
        flux_with_outlier[5, 10] = 1000.0
        orders_mod = [
            RectifiedOrder(
                order=ro0.order,
                order_index=ro0.order_index,
                wavelength_um=ro0.wavelength_um.copy(),
                spatial_frac=ro0.spatial_frac.copy(),
                flux=flux_with_outlier,
                source_image_shape=ro0.source_image_shape,
            )
        ] + [
            RectifiedOrder(
                order=ro.order,
                order_index=ro.order_index,
                wavelength_um=ro.wavelength_um.copy(),
                spatial_frac=ro.spatial_frac.copy(),
                flux=ro.flux.copy(),
                source_image_shape=ro.source_image_shape,
            )
            for ro in ros.rectified_orders[1:]
        ]
        ros_mod = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=orders_mod,
            source_image_shape=ros.source_image_shape,
        )
        ed = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.49,
            profile_source="smoothed_empirical",
            profile_smooth_sigma=1.0,
            reject_outliers=True,
            sigma_clip=3.0,
            max_iterations=3,
            reestimate_profile=True,
        )
        result = extract_weighted_optimal(ros_mod, ed)
        sp0 = result.spectra[0]
        assert sp0.flux.shape == (_N_SPECTRAL,)
        assert sp0.variance.shape == (_N_SPECTRAL,)
        assert sp0.profile_source_used == "smoothed_empirical"


# ===========================================================================
# 25. profile_source='external' behavior (Stage 20)
# ===========================================================================


class TestProfileSourceExternal:
    """Tests for profile_source='external' behavior."""

    @staticmethod
    def _make_ext_profile(n_ap: int, n_spectral: int) -> np.ndarray:
        """Create a normalized Gaussian-shaped external profile."""
        rows = np.arange(n_ap, dtype=float)
        center = n_ap / 2.0
        sigma = max(n_ap / 6.0, 1.0)
        p1d = np.exp(-0.5 * ((rows - center) / sigma) ** 2)
        p1d /= p1d.sum()
        return np.broadcast_to(p1d[:, np.newaxis], (n_ap, n_spectral)).copy()

    @staticmethod
    def _aperture_size(n_spatial: int = _N_SPATIAL, radius: float = 0.49) -> int:
        dist = np.abs(np.linspace(0.0, 1.0, n_spatial) - 0.5)
        return int(np.sum(dist <= radius))

    def test_external_output_shapes(self):
        """external profile produces correct output shapes."""
        n_ap = self._aperture_size()
        ext = self._make_ext_profile(n_ap, _N_SPECTRAL)
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=ext,
        )
        result = extract_weighted_optimal(ros, ed)
        assert result.n_orders == _N_ORDERS
        for sp in result.spectra:
            assert sp.flux.shape == (_N_SPECTRAL,)
            assert sp.variance.shape == (_N_SPECTRAL,)

    def test_external_profile_source_used_field(self):
        """profile_source_used is 'external'."""
        n_ap = self._aperture_size()
        ext = self._make_ext_profile(n_ap, _N_SPECTRAL)
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=ext,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile_source_used == "external"

    def test_external_profile_smoothed_false(self):
        """profile_smoothed is False for external extraction."""
        n_ap = self._aperture_size()
        ext = self._make_ext_profile(n_ap, _N_SPECTRAL)
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=ext,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile_smoothed is False

    def test_external_profile_used_exactly(self):
        """The supplied external profile (already normalized) is stored unchanged."""
        n_ap = self._aperture_size()
        ext = self._make_ext_profile(n_ap, _N_SPECTRAL)  # already normalized
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=ext,
            normalize_profile=True,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            np.testing.assert_allclose(sp.profile, ext, atol=1e-12)

    def test_external_1d_profile_broadcast(self):
        """External profile supplied as 1-D (n_ap,) is broadcast correctly."""
        n_ap = self._aperture_size()
        profile_1d = np.ones(n_ap) / n_ap
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=profile_1d,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile.shape == (n_ap, _N_SPECTRAL)

    def test_external_2d_one_col_profile_broadcast(self):
        """External profile supplied as (n_ap, 1) is broadcast correctly."""
        n_ap = self._aperture_size()
        profile_col = np.ones((n_ap, 1)) / n_ap
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=profile_col,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile.shape == (n_ap, _N_SPECTRAL)

    def test_external_incompatible_n_ap_raises(self):
        """ValueError if external_profile first dim != aperture size."""
        n_ap = self._aperture_size()
        wrong_profile = np.ones((n_ap + 5, _N_SPECTRAL)) / (n_ap + 5)
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=wrong_profile,
        )
        with pytest.raises(ValueError, match="n_ap"):
            extract_weighted_optimal(ros, ed)

    def test_external_incompatible_n_spectral_raises(self):
        """ValueError if external_profile second dim != n_spectral and != 1."""
        n_ap = self._aperture_size()
        wrong_profile = np.ones((n_ap, _N_SPECTRAL + 3)) / n_ap
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=wrong_profile,
        )
        with pytest.raises(ValueError, match="n_spectral"):
            extract_weighted_optimal(ros, ed)

    def test_external_with_rejection_keeps_profile_fixed(self):
        """External profile is not re-estimated during iterative rejection."""
        n_ap = self._aperture_size()
        ext = self._make_ext_profile(n_ap, _N_SPECTRAL)
        ros = _make_synthetic_rectified_order_set(fill_value=5.0)
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=ext,
            reject_outliers=True,
            reestimate_profile=True,  # Ignored for external.
            sigma_clip=3.0,
            max_iterations=3,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.profile_reestimated is False
            assert sp.initial_profile is None

    def test_external_profile_stored_normalized(self):
        """Unnormalized external profile is normalized before storage."""
        n_ap = self._aperture_size()
        ext_unnorm = np.full((n_ap, _N_SPECTRAL), 2.0)
        ros = _make_synthetic_rectified_order_set()
        ed = WeightedExtractionDefinition(
            center_frac=0.5, radius_frac=0.49,
            profile_source="external", external_profile=ext_unnorm,
            normalize_profile=True,
        )
        result = extract_weighted_optimal(ros, ed)
        for sp in result.spectra:
            col_sums = np.nansum(sp.profile, axis=0)
            finite_cols = np.isfinite(col_sums)
            np.testing.assert_allclose(
                col_sums[finite_cols], 1.0, atol=1e-12,
                err_msg="Normalized external profile should sum to 1.0 per column.",
            )


# ===========================================================================
# 26. Bookkeeping fields: profile_source_used and profile_smoothed (Stage 20)
# ===========================================================================


class TestProfileSourceBookkeeping:
    """Tests for profile_source_used and profile_smoothed bookkeeping fields."""

    def test_all_sources_populate_bookkeeping(self):
        """profile_source_used and profile_smoothed are set for all profile sources."""
        dist = np.abs(np.linspace(0.0, 1.0, _N_SPATIAL) - 0.5)
        n_ap = int(np.sum(dist <= 0.49))
        ext = np.ones((n_ap, _N_SPECTRAL)) / n_ap
        ros = _make_synthetic_rectified_order_set()

        cases = [
            ("empirical", {}, False),
            ("smoothed_empirical", {"profile_smooth_sigma": 1.5}, True),
            ("smoothed_empirical", {"profile_smooth_sigma": 0.0}, False),
            ("external", {"external_profile": ext}, False),
        ]
        for source, kwargs, expected_smoothed in cases:
            ed = WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.49,
                profile_source=source, **kwargs,
            )
            result = extract_weighted_optimal(ros, ed)
            for sp in result.spectra:
                assert sp.profile_source_used == source, (
                    f"Expected profile_source_used={source!r}, "
                    f"got {sp.profile_source_used!r}"
                )
                assert sp.profile_smoothed is expected_smoothed, (
                    f"source={source!r}: expected profile_smoothed="
                    f"{expected_smoothed}, got {sp.profile_smoothed}"
                )

    def test_weighted_extraction_shapes_all_sources(self):
        """All profile sources produce well-formed flux and variance arrays."""
        dist = np.abs(np.linspace(0.0, 1.0, _N_SPATIAL) - 0.5)
        n_ap = int(np.sum(dist <= 0.49))
        ext = np.ones((n_ap, _N_SPECTRAL)) / n_ap
        ros = _make_synthetic_rectified_order_set()

        for source, kwargs in [
            ("empirical", {}),
            ("smoothed_empirical", {"profile_smooth_sigma": 1.0}),
            ("external", {"external_profile": ext}),
        ]:
            ed = WeightedExtractionDefinition(
                center_frac=0.5, radius_frac=0.49,
                profile_source=source, **kwargs,
            )
            result = extract_weighted_optimal(ros, ed)
            for sp in result.spectra:
                assert sp.flux.shape == (_N_SPECTRAL,), (
                    f"source={source!r}: wrong flux shape"
                )
                assert sp.variance.shape == (_N_SPECTRAL,), (
                    f"source={source!r}: wrong variance shape"
                )
                fin_var = sp.variance[np.isfinite(sp.variance)]
                assert np.all(fin_var >= 0.0), (
                    f"source={source!r}: negative variance"
                )


# ===========================================================================
# 27. H1 smoke test: smoothed_empirical and empirical (Stage 20)
# ===========================================================================

_h1_profile_source_skip = pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 FITS data not available (LFS pointer files only).",
)


@_h1_profile_source_skip
class TestH1ProfileSourceSmokeTest:
    """Smoke tests for Stage 20 profile sources on real H1 calibration data."""

    @pytest.fixture(scope="class")
    def h1_profile_source_results(self):
        """Run H1 extraction with empirical and smoothed_empirical."""
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

        flat_trace = trace_orders_from_flat(_H1_FLAT_FILES, col_range=(650, 1550))
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)
        wci = read_wavecalinfo("H1")
        ll = read_line_list("H1")
        wav_map = fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )
        surface = fit_refined_coefficient_surface(wav_map, disp_degree=3)
        rect_indices = build_rectification_indices(geom, surface, wav_map=wav_map)
        with fits.open(_H1_FLAT_FILES[0]) as hdul:
            detector_image = hdul[0].data.astype(float)
        rectified = build_rectified_orders(detector_image, rect_indices)

        var_model = VarianceModelDefinition(
            read_noise_electron=11.0, gain_e_per_adu=1.8
        )
        ed_emp = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            background_inner=0.3,
            background_outer=0.45,
            profile_mode="global_median",
            profile_source="empirical",
        )
        ed_sm = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            background_inner=0.3,
            background_outer=0.45,
            profile_mode="global_median",
            profile_source="smoothed_empirical",
            profile_smooth_sigma=1.0,
        )
        r_emp = extract_weighted_optimal(
            rectified, ed_emp,
            variance_model=var_model, subtract_background=True,
        )
        r_sm = extract_weighted_optimal(
            rectified, ed_sm,
            variance_model=var_model, subtract_background=True,
        )
        return r_emp, r_sm, rectified

    def test_no_crashes(self, h1_profile_source_results):
        """Empirical and smoothed_empirical both complete without error."""
        r_emp, r_sm, _ = h1_profile_source_results
        assert r_emp is not None
        assert r_sm is not None

    def test_output_shapes_identical(self, h1_profile_source_results):
        """Output array shapes are identical for both profile sources."""
        r_emp, r_sm, _ = h1_profile_source_results
        for sp_e, sp_s in zip(r_emp.spectra, r_sm.spectra):
            assert sp_e.flux.shape == sp_s.flux.shape
            assert sp_e.variance.shape == sp_s.variance.shape
            assert sp_e.profile.shape == sp_s.profile.shape

    def test_spectra_well_formed(self, h1_profile_source_results):
        """Spectra and variance arrays are well-formed for both profile sources."""
        r_emp, r_sm, _ = h1_profile_source_results
        for result in (r_emp, r_sm):
            for sp in result.spectra:
                finite_frac = np.mean(np.isfinite(sp.flux))
                assert finite_frac > 0.5, (
                    f"Order {sp.order}: fewer than 50% of flux values are finite "
                    f"(profile_source={sp.profile_source_used!r})."
                )
                fin_var = sp.variance[np.isfinite(sp.variance)]
                if len(fin_var) > 0:
                    assert np.all(fin_var >= 0.0), (
                        f"Order {sp.order}: negative variance "
                        f"(profile_source={sp.profile_source_used!r})."
                    )

    def test_bookkeeping_consistent(self, h1_profile_source_results):
        """profile_source_used and profile_smoothed are correctly populated."""
        r_emp, r_sm, _ = h1_profile_source_results
        for sp in r_emp.spectra:
            assert sp.profile_source_used == "empirical"
            assert sp.profile_smoothed is False
        for sp in r_sm.spectra:
            assert sp.profile_source_used == "smoothed_empirical"
            assert sp.profile_smoothed is True
