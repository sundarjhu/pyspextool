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
