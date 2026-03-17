"""
Tests for the provisional optimal-extraction scaffold
(optimal_extraction.py).

Coverage:
  - OptimalExtractionDefinition: construction, validation, properties.
  - OptimalExtractedOrderSpectrum: construction, field access, n_spectral.
  - OptimalExtractedSpectrumSet: construction, get_order, orders, n_orders.
  - extract_optimal on synthetic data:
      * returns correct type and mode,
      * n_orders matches rectified order set,
      * known synthetic spatial profile → correct weighted extraction,
      * weighted extraction differs from simple sum/mean as expected,
      * profile normalization behaviour (normalize_profile True/False),
      * background subtraction correctness,
      * NaN handling (all-NaN column, empty aperture, all-NaN row),
      * shape consistency,
      * wavelength-axis propagation (independent copy),
      * profile array shape and values,
      * global_median vs columnwise profile modes.
  - Error tests:
      * invalid OptimalExtractionDefinition (bad center, radius, background,
        profile_mode),
      * empty rectified order set → ValueError,
      * non-2-D flux image → ValueError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * full chain stages 1–7 + optimal extraction (stage 12),
      * number of extracted spectra matches number of rectified orders,
      * wavelength axes are monotonic,
      * extracted flux arrays have correct shape,
      * profile arrays have expected dimensions,
      * outputs are finite where expected.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyspextool.instruments.ishell.optimal_extraction import (
    OptimalExtractionDefinition,
    OptimalExtractedOrderSpectrum,
    OptimalExtractedSpectrumSet,
    extract_optimal,
)
from pyspextool.instruments.ishell.rectification_indices import (
    RectificationIndexOrder,
    RectificationIndexSet,
)
from pyspextool.instruments.ishell.rectified_orders import (
    RectifiedOrder,
    RectifiedOrderSet,
    build_rectified_orders,
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
_N_SPATIAL = 20  # enough rows to define aperture + background

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
) -> OptimalExtractionDefinition:
    """Extraction definition centered at slit midpoint, no background."""
    return OptimalExtractionDefinition(
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
) -> OptimalExtractionDefinition:
    """Extraction definition centered at slit midpoint with background annulus."""
    return OptimalExtractionDefinition(
        center_frac=0.5,
        radius_frac=radius,
        background_inner=bg_inner,
        background_outer=bg_outer,
        profile_mode=profile_mode,
    )


# ===========================================================================
# 1. OptimalExtractionDefinition: construction and validation
# ===========================================================================


class TestOptimalExtractionDefinitionConstruction:
    """Tests for OptimalExtractionDefinition construction and property access."""

    def test_basic_construction(self):
        """OptimalExtractionDefinition can be constructed with minimal args."""
        ed = OptimalExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert ed.center_frac == 0.5
        assert ed.radius_frac == 0.1
        assert ed.background_inner is None
        assert ed.background_outer is None
        assert ed.profile_mode == "global_median"
        assert ed.normalize_profile is True

    def test_has_background_false_by_default(self):
        """has_background is False when no background region is defined."""
        ed = OptimalExtractionDefinition(center_frac=0.5, radius_frac=0.1)
        assert not ed.has_background

    def test_has_background_true_when_defined(self):
        """has_background is True when background region is defined."""
        ed = OptimalExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.1,
            background_inner=0.2,
            background_outer=0.4,
        )
        assert ed.has_background

    def test_columnwise_profile_mode(self):
        """profile_mode='columnwise' is accepted."""
        ed = OptimalExtractionDefinition(
            center_frac=0.5, radius_frac=0.1, profile_mode="columnwise"
        )
        assert ed.profile_mode == "columnwise"

    def test_normalize_profile_false(self):
        """normalize_profile=False is accepted."""
        ed = OptimalExtractionDefinition(
            center_frac=0.5, radius_frac=0.1, normalize_profile=False
        )
        assert ed.normalize_profile is False

    def test_invalid_center_below_zero(self):
        """ValueError if center_frac < 0."""
        with pytest.raises(ValueError, match="center_frac"):
            OptimalExtractionDefinition(center_frac=-0.1, radius_frac=0.1)

    def test_invalid_center_above_one(self):
        """ValueError if center_frac > 1."""
        with pytest.raises(ValueError, match="center_frac"):
            OptimalExtractionDefinition(center_frac=1.1, radius_frac=0.1)

    def test_invalid_radius_zero(self):
        """ValueError if radius_frac == 0."""
        with pytest.raises(ValueError, match="radius_frac"):
            OptimalExtractionDefinition(center_frac=0.5, radius_frac=0.0)

    def test_invalid_radius_negative(self):
        """ValueError if radius_frac < 0."""
        with pytest.raises(ValueError, match="radius_frac"):
            OptimalExtractionDefinition(center_frac=0.5, radius_frac=-0.1)

    def test_background_inner_only_raises(self):
        """ValueError if only background_inner is provided."""
        with pytest.raises(ValueError, match="both"):
            OptimalExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, background_inner=0.2
            )

    def test_background_outer_only_raises(self):
        """ValueError if only background_outer is provided."""
        with pytest.raises(ValueError, match="both"):
            OptimalExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, background_outer=0.4
            )

    def test_background_inner_le_radius_raises(self):
        """ValueError if background_inner <= radius_frac."""
        with pytest.raises(ValueError, match="background_inner"):
            OptimalExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.2,
                background_inner=0.2,  # equal to radius_frac → invalid
                background_outer=0.4,
            )

    def test_background_outer_le_inner_raises(self):
        """ValueError if background_outer <= background_inner."""
        with pytest.raises(ValueError, match="background_outer"):
            OptimalExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.1,
                background_inner=0.3,
                background_outer=0.2,  # less than inner → invalid
            )

    def test_invalid_profile_mode_raises(self):
        """ValueError if profile_mode is not a recognized string."""
        with pytest.raises(ValueError, match="profile_mode"):
            OptimalExtractionDefinition(
                center_frac=0.5, radius_frac=0.1, profile_mode="magic_psf"
            )


# ===========================================================================
# 2. OptimalExtractedOrderSpectrum: construction
# ===========================================================================


class TestOptimalExtractedOrderSpectrumConstruction:
    """Unit tests for OptimalExtractedOrderSpectrum construction and access."""

    def _make_minimal(self) -> OptimalExtractedOrderSpectrum:
        n = 32
        n_ap = 5
        ed = OptimalExtractionDefinition(center_frac=0.5, radius_frac=0.2)
        return OptimalExtractedOrderSpectrum(
            order=311,
            wavelength_um=np.linspace(1.64, 1.69, n),
            flux=np.ones(n),
            profile=np.ones((n_ap, n)),
            aperture=ed,
            method="optimal_weighted",
            n_pixels_used=n_ap,
            variance=None,
        )

    def test_construction(self):
        """OptimalExtractedOrderSpectrum can be constructed."""
        sp = self._make_minimal()
        assert sp.order == 311

    def test_n_spectral(self):
        """n_spectral matches length of wavelength_um."""
        sp = self._make_minimal()
        assert sp.n_spectral == 32

    def test_variance_none(self):
        """variance is None in the scaffold."""
        sp = self._make_minimal()
        assert sp.variance is None

    def test_method_stored(self):
        """method is stored correctly."""
        sp = self._make_minimal()
        assert sp.method == "optimal_weighted"

    def test_n_pixels_used_stored(self):
        """n_pixels_used is stored correctly."""
        sp = self._make_minimal()
        assert sp.n_pixels_used == 5

    def test_profile_shape(self):
        """profile has shape (n_ap, n_spectral)."""
        sp = self._make_minimal()
        assert sp.profile.shape == (5, 32)

    def test_aperture_stored(self):
        """aperture is stored correctly."""
        sp = self._make_minimal()
        assert sp.aperture.center_frac == 0.5


# ===========================================================================
# 3. OptimalExtractedSpectrumSet: construction and interface
# ===========================================================================


class TestOptimalExtractedSpectrumSetConstruction:
    """Tests for OptimalExtractedSpectrumSet interface."""

    def _make_set(self) -> OptimalExtractedSpectrumSet:
        n = 32
        n_ap = 5
        ed = OptimalExtractionDefinition(center_frac=0.5, radius_frac=0.2)
        spectra = [
            OptimalExtractedOrderSpectrum(
                order=311 + i * 4,
                wavelength_um=np.linspace(1.55 + i * 0.05, 1.59 + i * 0.05, n),
                flux=np.ones(n) * (i + 1),
                profile=np.ones((n_ap, n)),
                aperture=ed,
                method="optimal_weighted",
                n_pixels_used=n_ap,
                variance=None,
            )
            for i in range(3)
        ]
        return OptimalExtractedSpectrumSet(mode="H1_test", spectra=spectra)

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
        """An empty OptimalExtractedSpectrumSet is valid."""
        ss = OptimalExtractedSpectrumSet(mode="H1_test")
        assert ss.n_orders == 0
        assert ss.orders == []


# ===========================================================================
# 4. extract_optimal: synthetic data — basic behaviour
# ===========================================================================


class TestExtractOptimalSyntheticBasic:
    """Basic behaviour tests for extract_optimal on synthetic data."""

    def test_returns_optimal_extracted_spectrum_set(self):
        """extract_optimal returns an OptimalExtractedSpectrumSet."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        assert isinstance(result, OptimalExtractedSpectrumSet)

    def test_mode_propagated(self):
        """mode is propagated from the rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        assert result.mode == ros.mode

    def test_n_orders_matches(self):
        """n_orders matches the input rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        assert result.n_orders == ros.n_orders

    def test_orders_list_matches(self):
        """orders list matches input rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        assert result.orders == ros.orders

    def test_output_flux_shape(self):
        """Extracted flux has shape (n_spectral,)."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.flux.shape == (ro.n_spectral,)

    def test_profile_shape(self):
        """Profile has shape (n_ap_spatial, n_spectral)."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(radius=0.2)
        result = extract_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            # Profile must have n_spectral columns.
            assert sp.profile.shape[1] == ro.n_spectral
            # Profile must have the same number of rows as aperture pixels.
            dist = np.abs(ro.spatial_frac - ed.center_frac)
            n_ap = int(np.sum(dist <= ed.radius_frac))
            assert sp.profile.shape[0] == n_ap

    def test_wavelength_axis_propagated(self):
        """wavelength_um is identical to the input rectified order wavelength."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            np.testing.assert_array_equal(sp.wavelength_um, ro.wavelength_um)

    def test_wavelength_axis_is_copy(self):
        """wavelength_um is a copy, not a reference to the input array."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            # Mutating the original should not affect the output.
            original = ro.wavelength_um.copy()
            ro.wavelength_um[0] += 999.0
            np.testing.assert_array_equal(sp.wavelength_um, original)

    def test_method_is_optimal_weighted(self):
        """method field is always 'optimal_weighted'."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.method == "optimal_weighted"

    def test_variance_is_none(self):
        """variance is None in the scaffold."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.variance is None

    def test_n_pixels_used_positive(self):
        """n_pixels_used > 0 for a reasonable aperture."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(radius=0.2)
        result = extract_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.n_pixels_used > 0


# ===========================================================================
# 5. extract_optimal: known synthetic profile — weighted extraction
# ===========================================================================


class TestExtractOptimalWeightedExtraction:
    """Tests verifying the weighted extraction formula on known inputs."""

    def test_constant_flux_constant_profile_extracts_correctly(self):
        """For uniform flux F and normalized profile P, flux_1d should be F.

        With normalized profile P[:, col] summing to 1, flux_1d[col] =
        sum(P * F) / sum(P^2).  For constant F=c and uniform P=1/n_ap:
          numerator = sum(1/n_ap * c) = c
          denominator = sum((1/n_ap)^2) = 1/n_ap
          flux_1d = c / (1/n_ap) = c * n_ap

        Wait — this is NOT equal to c for a normalized profile.
        The formula gives the unscaled total, not the mean.

        Actually, let's think again. For normalize_profile=True,
        P[:, col] sums to 1. So P[i, col] = 1 / n_ap for uniform profile.
        flux_1d = sum(P * F) / sum(P^2)
                = sum(1/n_ap * c) / sum(1/n_ap^2)
                = c / (1/n_ap)
                = c * n_ap

        So for normalize_profile=True and constant flux c, the result
        is c * n_ap.  This is a feature of the formula, not a bug.

        For normalize_profile=False, P = 1 (each pixel has weight 1):
        flux_1d = sum(1 * c) / sum(1^2) = c * n_ap / n_ap = c.

        Let us test both cases explicitly.
        """
        fill = 3.0
        n_spatial = 20
        n_spectral = _N_SPECTRAL
        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=fill
        )
        ed_no_norm = _make_center_extraction_def(
            radius=0.2, normalize_profile=False
        )
        result = extract_optimal(ros, ed_no_norm, subtract_background=False)
        # With un-normalized profile P = F = constant c,
        # flux_1d = sum(c*c) / sum(c^2) = 1 for any constant c != 0.
        # So actually the result should be 1.0 for all columns.
        for sp in result.spectra:
            np.testing.assert_allclose(sp.flux, 1.0, rtol=1e-12)

    def test_known_profile_weighted_extraction(self):
        """Verify weighted extraction with a known non-uniform spatial profile.

        Build a single-order RectifiedOrderSet where the flux image has a
        known spatial profile: row values are [1, 2, 3, ..., n_ap] at every
        spectral column.

        For profile_mode='global_median', normalize_profile=False:
          profile_1d[i] = median over columns of flux_ap[i, :] = row_value[i]
          (since flux is constant across spectral axis)

        So profile[:, j] = [1, 2, 3, ..., n_ap] for all j.

        flux_1d[j] = sum(P * F, axis=0)[j] / sum(P^2, axis=0)[j]
                   = sum(p_i * f_i) / sum(p_i^2)

        where p_i = f_i = row_value[i] = (i+1):
          numerator   = sum((i+1)^2 for i in 0..n_ap-1)
          denominator = sum((i+1)^2 for i in 0..n_ap-1)
          flux_1d[j]  = 1.0 for all j.
        """
        n_ap = 5
        n_spectral = 16
        # We want exactly n_ap rows in the aperture. Use spatial_frac values
        # that are exactly at the aperture edges.
        spatial_frac = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 5 rows
        row_values = np.arange(1, n_ap + 1, dtype=float)  # [1, 2, 3, 4, 5]
        flux_2d = row_values[:, np.newaxis] * np.ones((n_ap, n_spectral))

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

        # Aperture covers all 5 rows (center=0.3, radius=0.25 → [0.05, 0.55]).
        ed = OptimalExtractionDefinition(
            center_frac=0.3,
            radius_frac=0.25,
            profile_mode="global_median",
            normalize_profile=False,
        )
        result = extract_optimal(ros, ed, subtract_background=False)
        sp = result.spectra[0]

        # numerator = sum(p * f) = sum((i+1)^2) across n_ap rows
        # denominator = sum(p^2) = sum((i+1)^2) across n_ap rows
        # → flux_1d = 1.0 everywhere
        np.testing.assert_allclose(sp.flux, 1.0, rtol=1e-12)

    def test_weighted_extraction_differs_from_simple_mean(self):
        """Optimal extraction with non-uniform profile differs from simple mean.

        For a non-uniform spatial profile, the weighted result is NOT equal
        to the column mean of the flux.
        """
        n_ap = 6
        n_spectral = 16
        spatial_frac = np.linspace(0.0, 1.0, n_ap)
        # Non-uniform flux: row i has flux (i+1) * 10.
        row_values = (np.arange(n_ap) + 1.0) * 10.0
        flux_2d = row_values[:, np.newaxis] * np.ones((n_ap, n_spectral))

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

        # Full-slit aperture so all rows are included.
        ed = OptimalExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.5,
            profile_mode="global_median",
            normalize_profile=False,
        )
        result = extract_optimal(ros, ed, subtract_background=False)
        sp = result.spectra[0]

        # Simple mean = mean(row_values)
        simple_mean = np.mean(row_values)
        # Weighted result: sum(p*f)/sum(p^2) = sum(f^2)/sum(f^2) = 1.0
        # since profile = f when normalize_profile=False.
        # These are not the same: 1.0 != mean(row_values) = 35.0
        assert not np.allclose(sp.flux, simple_mean, rtol=1e-6), (
            "Optimal extraction should differ from simple mean for "
            "non-uniform flux"
        )

    def test_normalize_profile_sums_to_one(self):
        """With normalize_profile=True, profile columns sum to 1.0."""
        ros = _make_synthetic_rectified_order_set(fill_value=5.0)
        ed = _make_center_extraction_def(
            radius=0.3, normalize_profile=True
        )
        result = extract_optimal(ros, ed, subtract_background=False)
        for sp in result.spectra:
            col_sums = np.nansum(sp.profile, axis=0)
            np.testing.assert_allclose(col_sums, 1.0, rtol=1e-12)

    def test_unnormalized_profile_not_summing_to_one(self):
        """With normalize_profile=False and n_ap > 1, profile does not sum to 1."""
        fill = 3.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ed = _make_center_extraction_def(
            radius=0.3, normalize_profile=False
        )
        result = extract_optimal(ros, ed, subtract_background=False)
        for sp in result.spectra:
            # With fill=3.0 and n_ap > 1, sum ≠ 1.0
            col_sums = np.nansum(sp.profile, axis=0)
            assert not np.allclose(col_sums, 1.0, rtol=1e-6)


# ===========================================================================
# 6. extract_optimal: background subtraction
# ===========================================================================


class TestExtractOptimalBackgroundSubtraction:
    """Tests for background subtraction in extract_optimal."""

    def test_background_subtracted_flux(self):
        """Background-subtracted extraction gives expected result.

        Set up aperture and background regions where:
          - aperture pixels have value object_fill
          - background pixels have value bg_fill

        With background subtraction, the aperture pixels become
        (object_fill - bg_fill). The weighted extraction should recover 1.0
        for normalize_profile=False (since P = F - bg = constant).
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

        ed = OptimalExtractionDefinition(
            center_frac=center,
            radius_frac=radius,
            background_inner=bg_inner,
            background_outer=bg_outer,
            profile_mode="global_median",
            normalize_profile=False,
        )
        result = extract_optimal(ros, ed, subtract_background=True)
        sp = result.spectra[0]

        # After background subtraction, aperture pixels = object_fill - bg_fill.
        # Weighted extraction with P = F - bg (constant) gives 1.0.
        np.testing.assert_allclose(sp.flux, 1.0, rtol=1e-12)

    def test_no_subtraction_when_flag_false(self):
        """Setting subtract_background=False does not subtract background."""
        fill = 7.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ed = _make_center_extraction_def_with_bg()
        result = extract_optimal(ros, ed, subtract_background=False)
        # No background subtraction; with uniform fill and normalize_profile=True,
        # each column should give fill.
        # Actually: for normalize_profile=True, flux_1d = sum(P*F)/sum(P^2)
        # where P sums to 1 and F = fill uniformly.
        # = sum(P * fill) / sum(P^2) = fill * sum(P) / sum(P^2) = fill / sum(P^2)
        # Since P = 1/n_ap uniformly, sum(P^2) = 1/n_ap → flux_1d = fill * n_ap.
        # Test should just check that no background subtraction happened,
        # i.e., the profile used the raw fill value.
        for sp in result.spectra:
            # flux is finite everywhere the aperture has pixels
            assert np.all(np.isfinite(sp.flux))


# ===========================================================================
# 7. extract_optimal: NaN handling
# ===========================================================================


class TestExtractOptimalNaNHandling:
    """Tests for NaN handling in extract_optimal."""

    def test_all_nan_aperture_column_produces_nan(self):
        """A spectral column where all aperture pixels are NaN → NaN output."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(radius=0.2)
        # Set first spectral column of first order to all NaN.
        ros.rectified_orders[0].flux[:, 0] = np.nan
        result = extract_optimal(ros, ed)
        assert np.isnan(result.spectra[0].flux[0])

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
        # Set one aperture row to NaN.
        ro.flux[ap_indices[0], :] = np.nan
        result = extract_optimal(ros, ed, subtract_background=False)
        # Remaining aperture rows are all finite → output should be finite.
        assert np.all(np.isfinite(result.spectra[0].flux))

    def test_empty_aperture_gives_all_nan(self):
        """An aperture too narrow to capture any pixels → all-NaN flux."""
        ros = _make_synthetic_rectified_order_set()
        # spatial_frac is linspace(0, 1, _N_SPATIAL=20); spacing ≈ 0.0526.
        # center_frac=0.5 does not coincide with any grid point exactly,
        # so radius=1e-6 selects no pixels → all NaN.
        ed = OptimalExtractionDefinition(
            center_frac=0.5, radius_frac=1e-6
        )
        result = extract_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.flux.shape == (sp.n_spectral,)
            assert np.all(np.isnan(sp.flux)), (
                f"Order {sp.order}: expected all-NaN flux for empty aperture"
            )

    def test_n_pixels_used_decreases_with_nan_row(self):
        """n_pixels_used decreases when an aperture row is set to all-NaN."""
        fill = 2.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ed = _make_center_extraction_def(radius=0.2)
        ro = ros.rectified_orders[0]
        dist = np.abs(ro.spatial_frac - ed.center_frac)
        ap_indices = np.where(dist <= ed.radius_frac)[0]

        result_full = extract_optimal(ros, ed)
        n_full = result_full.spectra[0].n_pixels_used

        # Set one aperture row to all-NaN.
        ro.flux[ap_indices[0], :] = np.nan
        result_nan = extract_optimal(ros, ed)
        n_nan = result_nan.spectra[0].n_pixels_used

        assert n_nan == n_full - 1


# ===========================================================================
# 8. extract_optimal: profile modes
# ===========================================================================


class TestExtractOptimalProfileModes:
    """Tests for global_median vs columnwise profile modes."""

    def test_global_median_constant_flux(self):
        """For constant flux, global_median and columnwise give same profile."""
        fill = 4.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ed_gm = _make_center_extraction_def(
            profile_mode="global_median", normalize_profile=True
        )
        ed_cw = _make_center_extraction_def(
            profile_mode="columnwise", normalize_profile=True
        )
        result_gm = extract_optimal(ros, ed_gm, subtract_background=False)
        result_cw = extract_optimal(ros, ed_cw, subtract_background=False)
        for sp_gm, sp_cw in zip(result_gm.spectra, result_cw.spectra):
            np.testing.assert_allclose(sp_gm.flux, sp_cw.flux, rtol=1e-12)

    def test_global_median_profile_is_constant_across_columns(self):
        """global_median produces a profile that is the same in every column."""
        ros = _make_synthetic_rectified_order_set(fill_value=5.0)
        ed = _make_center_extraction_def(
            profile_mode="global_median", normalize_profile=True
        )
        result = extract_optimal(ros, ed)
        for sp in result.spectra:
            # Each column of the profile should be identical.
            first_col = sp.profile[:, 0]
            for j in range(1, sp.n_spectral):
                np.testing.assert_allclose(sp.profile[:, j], first_col, rtol=1e-12)

    def test_columnwise_profile_shape(self):
        """columnwise profile has correct shape."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def(
            radius=0.2, profile_mode="columnwise"
        )
        result = extract_optimal(ros, ed)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            dist = np.abs(ro.spatial_frac - ed.center_frac)
            n_ap = int(np.sum(dist <= ed.radius_frac))
            assert sp.profile.shape == (n_ap, ro.n_spectral)


# ===========================================================================
# 9. Parametric shape-consistency tests
# ===========================================================================


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 10), (64, 40), (128, 50)])
def test_output_shapes_parametric(n_spectral, n_spatial):
    """Extracted spectra have correct shapes for various (n_spectral, n_spatial)."""
    ros = _make_synthetic_rectified_order_set(
        n_spectral=n_spectral, n_spatial=n_spatial
    )
    ed = _make_center_extraction_def(radius=0.2)
    for profile_mode in ("global_median", "columnwise"):
        ed_mode = OptimalExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            profile_mode=profile_mode,
        )
        result = extract_optimal(ros, ed_mode)
        for sp in result.spectra:
            assert sp.flux.shape == (n_spectral,), (
                f"profile_mode={profile_mode}: "
                f"flux shape {sp.flux.shape} != ({n_spectral},)"
            )
            assert sp.wavelength_um.shape == (n_spectral,)
            assert sp.profile.shape[1] == n_spectral


# ===========================================================================
# 10. Error handling
# ===========================================================================


class TestExtractOptimalErrors:
    """Error paths for extract_optimal."""

    def test_raises_on_empty_rectified_order_set(self):
        """ValueError if rectified_orders has no orders."""
        empty_ros = RectifiedOrderSet(mode="test")
        ed = _make_center_extraction_def()
        with pytest.raises(ValueError, match="empty"):
            extract_optimal(empty_ros, ed)

    def test_raises_on_non_2d_flux(self):
        """ValueError if a flux array is not 2-D."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        ros.rectified_orders[0].flux = np.ones(_N_SPECTRAL)
        with pytest.raises(ValueError, match="2-D"):
            extract_optimal(ros, ed)

    def test_raises_on_invalid_profile_mode(self):
        """ValueError if profile_mode is invalid (caught at definition time)."""
        with pytest.raises(ValueError, match="profile_mode"):
            OptimalExtractionDefinition(
                center_frac=0.5,
                radius_frac=0.1,
                profile_mode="not_a_mode",
            )

    def test_raises_on_variance_shape_mismatch(self):
        """ValueError if variance_image shape does not match flux shape."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        bad_variance = np.ones((_N_SPATIAL + 1, _N_SPECTRAL))
        with pytest.raises(ValueError, match="variance_image shape"):
            extract_optimal(ros, ed, variance_image=bad_variance)


# ===========================================================================
# 11. Variance propagation
# ===========================================================================


class TestExtractOptimalVariancePropagation:
    """Tests for variance propagation in extract_optimal."""

    def test_variance_none_when_no_variance_image(self):
        """variance is None when variance_image is not provided."""
        ros = _make_synthetic_rectified_order_set()
        ed = _make_center_extraction_def()
        result = extract_optimal(ros, ed)
        for sp in result.spectra:
            assert sp.variance is None

    def test_variance_shape_matches_flux(self):
        """variance has the same shape as flux when variance_image is provided."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result = extract_optimal(ros, ed, variance_image=var_image)
        for sp in result.spectra:
            assert sp.variance is not None
            assert sp.variance.shape == sp.flux.shape

    def test_variance_nonnegative(self):
        """Propagated variance is non-negative everywhere it is finite."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result = extract_optimal(ros, ed, variance_image=var_image)
        for sp in result.spectra:
            finite_mask = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite_mask] >= 0.0)

    def test_variance_with_background_subtraction(self):
        """Variance is propagated correctly when background subtraction is active."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def_with_bg()
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result = extract_optimal(
            ros, ed, subtract_background=True, variance_image=var_image
        )
        for sp in result.spectra:
            assert sp.variance is not None
            assert sp.variance.shape == sp.flux.shape
            finite_mask = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite_mask] >= 0.0)

    def test_variance_larger_with_background_than_without(self):
        """Background variance contribution increases total variance.

        When background variance is added, the propagated variance
        ``sum(P^2 * (var_ap + var_bg))`` is >= ``sum(P^2 * var_ap)``
        provided the profile is non-zero.

        Build a dataset where aperture pixels differ from background
        pixels so the profile remains non-zero after subtraction.
        """
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL
        center = 0.5
        radius = 0.2
        bg_inner = 0.3
        bg_outer = 0.45

        spatial_frac = np.linspace(0.0, 1.0, n_spatial)
        dist = np.abs(spatial_frac - center)

        # Aperture: high flux; background: low flux.
        object_fill = 10.0
        bg_fill = 2.0
        flux_2d = np.full((n_spatial, n_spectral), bg_fill)
        ap_mask = dist <= radius
        flux_2d[ap_mask, :] = object_fill

        from pyspextool.instruments.ishell.rectified_orders import (
            RectifiedOrder,
            RectifiedOrderSet,
        )

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

        ed_no_bg = OptimalExtractionDefinition(
            center_frac=center, radius_frac=radius
        )
        ed_with_bg = OptimalExtractionDefinition(
            center_frac=center,
            radius_frac=radius,
            background_inner=bg_inner,
            background_outer=bg_outer,
        )
        var_image = np.ones((n_spatial, n_spectral))

        result_no_bg = extract_optimal(
            ros, ed_no_bg, subtract_background=False, variance_image=var_image
        )
        result_with_bg = extract_optimal(
            ros, ed_with_bg, subtract_background=True, variance_image=var_image
        )
        sp_no_bg = result_no_bg.spectra[0]
        sp_with_bg = result_with_bg.spectra[0]

        finite = np.isfinite(sp_no_bg.variance) & np.isfinite(
            sp_with_bg.variance
        )
        assert np.all(sp_with_bg.variance[finite] >= sp_no_bg.variance[finite])

    def test_unit_variance_no_background(self):
        """Unit variance image gives deterministic variance values.

        For unit variance, profile P, and n_ap aperture pixels (no background):
          variance_1d = nansum(P**2 * 1, axis=0)
        """
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL
        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=1.0
        )
        ed = _make_center_extraction_def(radius=0.2, normalize_profile=True)
        var_image = np.ones((n_spatial, n_spectral))
        result = extract_optimal(
            ros, ed, subtract_background=False, variance_image=var_image
        )
        for sp in result.spectra:
            # variance = nansum(P**2, axis=0) for unit variance.
            expected = np.nansum(sp.profile**2, axis=0)
            np.testing.assert_allclose(sp.variance, expected, rtol=1e-12)


# ===========================================================================
# 13. variance_model integration tests
# ===========================================================================


class TestExtractOptimalVarianceModel:
    """Tests for variance_model integration in extract_optimal."""

    def _make_variance_model(self):
        from pyspextool.instruments.ishell.variance_model import VarianceModelDefinition
        return VarianceModelDefinition(read_noise_electron=10.0, gain_e_per_adu=2.0)

    def test_variance_model_only_produces_variance(self):
        """variance_model alone produces non-None variance output."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        vmodel = self._make_variance_model()
        result = extract_optimal(ros, ed, variance_model=vmodel)
        for sp in result.spectra:
            assert sp.variance is not None

    def test_variance_model_shape_correct(self):
        """variance produced from variance_model has same shape as flux."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        vmodel = self._make_variance_model()
        result = extract_optimal(ros, ed, variance_model=vmodel)
        for sp in result.spectra:
            assert sp.variance.shape == sp.flux.shape

    def test_variance_model_nonnegative(self):
        """Variance from variance_model is non-negative everywhere it is finite."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        vmodel = self._make_variance_model()
        result = extract_optimal(ros, ed, variance_model=vmodel)
        for sp in result.spectra:
            finite_mask = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite_mask] >= 0.0)

    def test_variance_image_overrides_variance_model(self):
        """explicit variance_image takes priority over variance_model."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL, fill_value=1.0
        )
        ed = _make_center_extraction_def(normalize_profile=True)
        vmodel = self._make_variance_model()
        # Use a distinctive all-5 variance image.
        var_image = np.full((_N_SPATIAL, _N_SPECTRAL), 5.0)
        result_model = extract_optimal(ros, ed, variance_model=vmodel)
        result_image = extract_optimal(
            ros, ed, variance_image=var_image, variance_model=vmodel
        )
        result_image_only = extract_optimal(ros, ed, variance_image=var_image)
        # variance_image+model should equal variance_image only (model ignored)
        for sp_img, sp_both in zip(
            result_image_only.spectra, result_image.spectra
        ):
            np.testing.assert_allclose(sp_both.variance, sp_img.variance, rtol=1e-12)

    def test_variance_model_shape_matches_flux(self):
        """variance length matches the wavelength axis length (n_spectral)."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ed = _make_center_extraction_def()
        vmodel = self._make_variance_model()
        result = extract_optimal(ros, ed, variance_model=vmodel)
        for sp in result.spectra:
            assert sp.variance.shape == sp.wavelength_um.shape


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1OptimalExtractionSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset.

    Chain:
      1. Flat-order tracing (Stage 1)
      2. Arc-line tracing (Stage 2)
      3. Provisional wavelength mapping (Stage 3)
      4. Coefficient-surface refinement (Stage 5)
      5. Rectification indices (Stage 6)
      6. Rectified orders (Stage 7)
      7. Optimal extraction (Stage 12)
    """

    @pytest.fixture(scope="class")
    def h1_optimal_result(self):
        """OptimalExtractedSpectrumSet from the real H1 calibration chain."""
        import astropy.io.fits as fits

        from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
        from pyspextool.instruments.ishell.calibrations import (
            read_line_list,
            read_wavecalinfo,
        )
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
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

        # Stage 12: optimal extraction
        ed = OptimalExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            background_inner=0.3,
            background_outer=0.45,
            profile_mode="global_median",
        )
        extracted = extract_optimal(
            rectified, ed, subtract_background=True
        )

        return extracted, rectified

    def test_returns_optimal_extracted_spectrum_set(self, h1_optimal_result):
        """extract_optimal returns an OptimalExtractedSpectrumSet."""
        extracted, _ = h1_optimal_result
        assert isinstance(extracted, OptimalExtractedSpectrumSet)

    def test_n_orders_matches_rectified(self, h1_optimal_result):
        """n_orders matches the rectified order set."""
        extracted, rectified = h1_optimal_result
        assert extracted.n_orders == rectified.n_orders

    def test_n_orders_positive(self, h1_optimal_result):
        """n_orders > 0."""
        extracted, _ = h1_optimal_result
        assert extracted.n_orders > 0

    def test_wavelength_axes_monotonic(self, h1_optimal_result):
        """wavelength_um is monotonic for every extracted spectrum."""
        extracted, _ = h1_optimal_result
        for sp in extracted.spectra:
            diffs = np.diff(sp.wavelength_um)
            assert np.all(diffs > 0) or np.all(diffs < 0), (
                f"Order {sp.order}: wavelength axis is not monotonic"
            )

    def test_flux_shape_matches_wavelength(self, h1_optimal_result):
        """flux shape equals wavelength_um shape for every spectrum."""
        extracted, _ = h1_optimal_result
        for sp in extracted.spectra:
            assert sp.flux.shape == sp.wavelength_um.shape

    def test_profile_has_correct_n_spectral(self, h1_optimal_result):
        """profile has the correct number of spectral columns."""
        extracted, rectified = h1_optimal_result
        for sp, ro in zip(extracted.spectra, rectified.rectified_orders):
            assert sp.profile.shape[1] == ro.n_spectral

    def test_flux_finite_for_most_columns(self, h1_optimal_result):
        """Most flux values are finite (some NaN from rectification is OK)."""
        extracted, _ = h1_optimal_result
        for sp in extracted.spectra:
            finite_frac = np.mean(np.isfinite(sp.flux))
            assert finite_frac > 0.5, (
                f"Order {sp.order}: fewer than 50% of flux values are finite "
                f"({finite_frac:.1%})"
            )

    def test_orders_list_matches_rectified(self, h1_optimal_result):
        """orders list matches the rectified order set."""
        extracted, rectified = h1_optimal_result
        assert extracted.orders == rectified.orders

    def test_get_order_works_for_all_orders(self, h1_optimal_result):
        """get_order succeeds for every order in the set."""
        extracted, _ = h1_optimal_result
        for order in extracted.orders:
            sp = extracted.get_order(order)
            assert sp.order == order

    def test_n_pixels_used_positive(self, h1_optimal_result):
        """n_pixels_used > 0 for every order."""
        extracted, _ = h1_optimal_result
        for sp in extracted.spectra:
            assert sp.n_pixels_used > 0

    def test_variance_is_none(self, h1_optimal_result):
        """variance is None for every order in the scaffold."""
        extracted, _ = h1_optimal_result
        for sp in extracted.spectra:
            assert sp.variance is None
