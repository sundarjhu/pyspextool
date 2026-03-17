"""
Tests for the provisional variance-image scaffold (variance_model.py).

Coverage:
  - VarianceModelDefinition: construction, validation, defaults.
  - VarianceImageProduct: construction, shape property, finite_fraction.
  - build_variance_image on synthetic data:
      * read-noise-only case,
      * Poisson-only case,
      * Poisson + read-noise case,
      * negative flux handling (clip_negative_flux=True/False),
      * minimum_variance floor,
      * shape preservation,
      * NaN propagation.
  - build_unit_variance_image:
      * returns all-ones variance,
      * shape matches input.
  - Error handling:
      * non-2-D image → ValueError,
      * invalid VarianceModelDefinition values → ValueError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * variance image shape matches detector image shape,
      * all values finite,
      * all values non-negative,
      * minimum_variance floor respected.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyspextool.instruments.ishell.variance_model import (
    VarianceImageProduct,
    VarianceModelDefinition,
    build_unit_variance_image,
    build_variance_image,
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
_HAVE_H1_DATA = len(_H1_FLAT_FILES) >= 1

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

_NROWS = 50
_NCOLS = 80


# ===========================================================================
# 1. VarianceModelDefinition: construction and validation
# ===========================================================================


class TestVarianceModelDefinitionConstruction:
    """Tests for VarianceModelDefinition construction and validation."""

    def test_basic_construction(self):
        """VarianceModelDefinition can be constructed with required args."""
        defn = VarianceModelDefinition(
            read_noise_electron=10.0,
            gain_e_per_adu=2.0,
        )
        assert defn.read_noise_electron == 10.0
        assert defn.gain_e_per_adu == 2.0
        assert defn.include_poisson is True
        assert defn.include_read_noise is True
        assert defn.minimum_variance > 0.0
        assert defn.clip_negative_flux is True

    def test_read_noise_zero_is_valid(self):
        """read_noise_electron=0 is valid (noiseless read-out)."""
        defn = VarianceModelDefinition(
            read_noise_electron=0.0, gain_e_per_adu=1.0
        )
        assert defn.read_noise_electron == 0.0

    def test_defaults_include_both_noise_terms(self):
        """Default construction includes both Poisson and read-noise terms."""
        defn = VarianceModelDefinition(
            read_noise_electron=5.0, gain_e_per_adu=1.5
        )
        assert defn.include_poisson is True
        assert defn.include_read_noise is True

    def test_override_flags(self):
        """Boolean flags can be overridden at construction."""
        defn = VarianceModelDefinition(
            read_noise_electron=5.0,
            gain_e_per_adu=1.5,
            include_poisson=False,
            include_read_noise=False,
        )
        assert defn.include_poisson is False
        assert defn.include_read_noise is False

    def test_invalid_read_noise_negative(self):
        """ValueError if read_noise_electron < 0."""
        with pytest.raises(ValueError, match="read_noise_electron"):
            VarianceModelDefinition(
                read_noise_electron=-1.0, gain_e_per_adu=2.0
            )

    def test_invalid_gain_zero(self):
        """ValueError if gain_e_per_adu == 0."""
        with pytest.raises(ValueError, match="gain_e_per_adu"):
            VarianceModelDefinition(
                read_noise_electron=5.0, gain_e_per_adu=0.0
            )

    def test_invalid_gain_negative(self):
        """ValueError if gain_e_per_adu < 0."""
        with pytest.raises(ValueError, match="gain_e_per_adu"):
            VarianceModelDefinition(
                read_noise_electron=5.0, gain_e_per_adu=-1.0
            )

    def test_invalid_minimum_variance_zero(self):
        """ValueError if minimum_variance == 0."""
        with pytest.raises(ValueError, match="minimum_variance"):
            VarianceModelDefinition(
                read_noise_electron=5.0,
                gain_e_per_adu=2.0,
                minimum_variance=0.0,
            )

    def test_invalid_minimum_variance_negative(self):
        """ValueError if minimum_variance < 0."""
        with pytest.raises(ValueError, match="minimum_variance"):
            VarianceModelDefinition(
                read_noise_electron=5.0,
                gain_e_per_adu=2.0,
                minimum_variance=-1e-5,
            )


# ===========================================================================
# 2. VarianceImageProduct: construction and properties
# ===========================================================================


class TestVarianceImageProductConstruction:
    """Tests for VarianceImageProduct construction and property access."""

    def _make_product(self) -> VarianceImageProduct:
        defn = VarianceModelDefinition(
            read_noise_electron=10.0, gain_e_per_adu=2.0
        )
        img = np.ones((_NROWS, _NCOLS), dtype=float)
        return build_variance_image(img, defn, mode="H1")

    def test_returns_variance_image_product(self):
        """build_variance_image returns a VarianceImageProduct."""
        assert isinstance(self._make_product(), VarianceImageProduct)

    def test_shape_property(self):
        """shape property returns (n_rows, n_cols)."""
        prod = self._make_product()
        assert prod.shape == (_NROWS, _NCOLS)

    def test_source_image_shape(self):
        """source_image_shape matches input image shape."""
        prod = self._make_product()
        assert prod.source_image_shape == (_NROWS, _NCOLS)

    def test_mode_stored(self):
        """mode is stored in the product."""
        prod = self._make_product()
        assert prod.mode == "H1"

    def test_mode_none_by_default(self):
        """mode defaults to None when not provided."""
        defn = VarianceModelDefinition(
            read_noise_electron=5.0, gain_e_per_adu=1.0
        )
        img = np.ones((_NROWS, _NCOLS))
        prod = build_variance_image(img, defn)
        assert prod.mode is None

    def test_finite_fraction_all_ones(self):
        """finite_fraction is 1.0 for a fully finite variance image."""
        prod = self._make_product()
        assert prod.finite_fraction == pytest.approx(1.0)

    def test_finite_fraction_with_nan(self):
        """finite_fraction is < 1.0 when variance image has NaN pixels."""
        defn = VarianceModelDefinition(
            read_noise_electron=10.0, gain_e_per_adu=2.0
        )
        img = np.ones((_NROWS, _NCOLS), dtype=float)
        img[0, 0] = np.nan
        prod = build_variance_image(img, defn)
        assert prod.finite_fraction < 1.0


# ===========================================================================
# 3. build_variance_image: synthetic correctness tests
# ===========================================================================


class TestBuildVarianceImageReadNoiseOnly:
    """Tests for read-noise-only variance (include_poisson=False)."""

    def test_variance_equals_read_noise_term(self):
        """With include_poisson=False, variance = (rn/gain)**2 everywhere."""
        rn = 10.0
        gain = 2.0
        defn = VarianceModelDefinition(
            read_noise_electron=rn,
            gain_e_per_adu=gain,
            include_poisson=False,
            include_read_noise=True,
            minimum_variance=1e-10,
        )
        img = np.full((_NROWS, _NCOLS), 100.0)
        prod = build_variance_image(img, defn)

        expected = (rn / gain) ** 2
        np.testing.assert_allclose(
            prod.variance_image, expected, rtol=1e-12
        )

    def test_uniform_for_varying_signal(self):
        """Read-noise variance is constant regardless of signal level."""
        rn = 8.0
        gain = 4.0
        defn = VarianceModelDefinition(
            read_noise_electron=rn,
            gain_e_per_adu=gain,
            include_poisson=False,
            include_read_noise=True,
            minimum_variance=1e-10,
        )
        rng = np.random.default_rng(42)
        img = rng.uniform(0, 1000, (_NROWS, _NCOLS))
        prod = build_variance_image(img, defn)

        expected = (rn / gain) ** 2
        np.testing.assert_allclose(
            prod.variance_image, expected, rtol=1e-12
        )


class TestBuildVarianceImagePoissonOnly:
    """Tests for Poisson-only variance (include_read_noise=False)."""

    def test_variance_equals_poisson_term(self):
        """With include_read_noise=False, variance = image / gain everywhere."""
        gain = 3.0
        defn = VarianceModelDefinition(
            read_noise_electron=0.0,
            gain_e_per_adu=gain,
            include_poisson=True,
            include_read_noise=False,
            minimum_variance=1e-10,
        )
        signal = 300.0
        img = np.full((_NROWS, _NCOLS), signal)
        prod = build_variance_image(img, defn)

        expected = signal / gain
        np.testing.assert_allclose(
            prod.variance_image, expected, rtol=1e-12
        )

    def test_varies_with_signal(self):
        """Poisson variance scales linearly with signal."""
        gain = 2.0
        defn = VarianceModelDefinition(
            read_noise_electron=0.0,
            gain_e_per_adu=gain,
            include_poisson=True,
            include_read_noise=False,
            minimum_variance=1e-30,
        )
        # Two-column image with different signal levels
        img = np.array([[100.0, 200.0]])
        prod = build_variance_image(img, defn)

        np.testing.assert_allclose(
            prod.variance_image[0, 0], 100.0 / gain, rtol=1e-12
        )
        np.testing.assert_allclose(
            prod.variance_image[0, 1], 200.0 / gain, rtol=1e-12
        )


class TestBuildVarianceImageCombined:
    """Tests for Poisson + read-noise variance."""

    def test_combined_variance(self):
        """variance = image/gain + (rn/gain)**2 for positive signal."""
        rn = 10.0
        gain = 2.0
        signal = 500.0
        defn = VarianceModelDefinition(
            read_noise_electron=rn,
            gain_e_per_adu=gain,
            include_poisson=True,
            include_read_noise=True,
            minimum_variance=1e-10,
        )
        img = np.full((_NROWS, _NCOLS), signal)
        prod = build_variance_image(img, defn)

        expected = signal / gain + (rn / gain) ** 2
        np.testing.assert_allclose(
            prod.variance_image, expected, rtol=1e-12
        )


class TestBuildVarianceImageNegativeFlux:
    """Tests for negative flux handling."""

    def test_clip_negative_flux_true(self):
        """With clip_negative_flux=True, negative pixels do not reduce variance."""
        rn = 5.0
        gain = 2.0
        defn = VarianceModelDefinition(
            read_noise_electron=rn,
            gain_e_per_adu=gain,
            include_poisson=True,
            include_read_noise=True,
            minimum_variance=1e-10,
            clip_negative_flux=True,
        )
        # All-negative image: Poisson contribution should be 0.
        img = np.full((_NROWS, _NCOLS), -100.0)
        prod = build_variance_image(img, defn)

        # Expected: only read-noise term (Poisson clipped to 0).
        expected = (rn / gain) ** 2
        np.testing.assert_allclose(
            prod.variance_image, expected, rtol=1e-12
        )

    def test_clip_negative_flux_false(self):
        """With clip_negative_flux=False, negative pixels reduce variance."""
        rn = 5.0
        gain = 2.0
        signal = -100.0
        defn = VarianceModelDefinition(
            read_noise_electron=rn,
            gain_e_per_adu=gain,
            include_poisson=True,
            include_read_noise=True,
            minimum_variance=1e-10,
            clip_negative_flux=False,
        )
        img = np.full((_NROWS, _NCOLS), signal)
        prod = build_variance_image(img, defn)

        # Expected: signal/gain + (rn/gain)**2, but floored at minimum_variance.
        expected_raw = signal / gain + (rn / gain) ** 2
        expected = max(expected_raw, defn.minimum_variance)
        np.testing.assert_allclose(
            prod.variance_image, expected, rtol=1e-12
        )

    def test_positive_pixels_unaffected_by_clip(self):
        """Positive pixels are unaffected regardless of clip_negative_flux."""
        rn = 5.0
        gain = 2.0
        signal = 200.0
        defn_clip = VarianceModelDefinition(
            read_noise_electron=rn,
            gain_e_per_adu=gain,
            include_poisson=True,
            include_read_noise=True,
            minimum_variance=1e-10,
            clip_negative_flux=True,
        )
        defn_noclip = VarianceModelDefinition(
            read_noise_electron=rn,
            gain_e_per_adu=gain,
            include_poisson=True,
            include_read_noise=True,
            minimum_variance=1e-10,
            clip_negative_flux=False,
        )
        img = np.full((_NROWS, _NCOLS), signal)
        prod_clip = build_variance_image(img, defn_clip)
        prod_noclip = build_variance_image(img, defn_noclip)

        np.testing.assert_allclose(
            prod_clip.variance_image, prod_noclip.variance_image, rtol=1e-12
        )


class TestBuildVarianceImageMinimumFloor:
    """Tests for the minimum_variance floor."""

    def test_floor_applied_everywhere(self):
        """variance >= minimum_variance for every pixel."""
        floor = 1e-4
        defn = VarianceModelDefinition(
            read_noise_electron=0.0,
            gain_e_per_adu=1.0,
            include_poisson=True,
            include_read_noise=False,
            minimum_variance=floor,
            clip_negative_flux=True,
        )
        # All-negative image → Poisson clipped to 0 → floor applied.
        img = np.full((_NROWS, _NCOLS), -50.0)
        prod = build_variance_image(img, defn)

        assert np.all(prod.variance_image >= floor)

    def test_floor_not_applied_to_high_variance(self):
        """minimum_variance floor does not affect high-variance pixels."""
        floor = 1e-10
        signal = 1000.0
        gain = 1.0
        defn = VarianceModelDefinition(
            read_noise_electron=0.0,
            gain_e_per_adu=gain,
            include_poisson=True,
            include_read_noise=False,
            minimum_variance=floor,
        )
        img = np.full((_NROWS, _NCOLS), signal)
        prod = build_variance_image(img, defn)

        # signal/gain >> floor, so floor has no effect.
        np.testing.assert_allclose(
            prod.variance_image, signal / gain, rtol=1e-12
        )


class TestBuildVarianceImageShapePreservation:
    """Tests that output shape matches input image shape."""

    @pytest.mark.parametrize("shape", [(1, 1), (10, 20), (200, 800)])
    def test_shape_preserved(self, shape: tuple[int, int]):
        """variance_image shape matches input image shape."""
        defn = VarianceModelDefinition(
            read_noise_electron=5.0, gain_e_per_adu=2.0
        )
        img = np.ones(shape)
        prod = build_variance_image(img, defn)
        assert prod.shape == shape
        assert prod.source_image_shape == shape

    def test_float_output_dtype(self):
        """variance_image has floating-point dtype."""
        defn = VarianceModelDefinition(
            read_noise_electron=5.0, gain_e_per_adu=2.0
        )
        img = np.ones((_NROWS, _NCOLS), dtype=np.int32)
        prod = build_variance_image(img, defn)
        assert np.issubdtype(prod.variance_image.dtype, np.floating)


class TestBuildVarianceImageNaNPropagation:
    """Tests for NaN propagation behaviour."""

    def test_nan_pixel_gives_nan_variance(self):
        """A NaN pixel in the image produces a NaN in the Poisson term."""
        defn = VarianceModelDefinition(
            read_noise_electron=0.0,
            gain_e_per_adu=1.0,
            include_poisson=True,
            include_read_noise=False,
            minimum_variance=1e-10,
        )
        img = np.ones((_NROWS, _NCOLS), dtype=float)
        img[5, 10] = np.nan
        prod = build_variance_image(img, defn)

        assert np.isnan(prod.variance_image[5, 10])

    def test_non_nan_pixels_unaffected(self):
        """NaN in one pixel does not affect other pixels."""
        defn = VarianceModelDefinition(
            read_noise_electron=0.0,
            gain_e_per_adu=1.0,
            include_poisson=True,
            include_read_noise=False,
            minimum_variance=1e-10,
        )
        img = np.full((_NROWS, _NCOLS), 100.0)
        img[5, 10] = np.nan
        prod = build_variance_image(img, defn)

        # All other pixels should equal 100 / 1.0 = 100.
        mask = np.ones((_NROWS, _NCOLS), dtype=bool)
        mask[5, 10] = False
        np.testing.assert_allclose(
            prod.variance_image[mask], 100.0, rtol=1e-12
        )


# ===========================================================================
# 4. build_unit_variance_image
# ===========================================================================


class TestBuildUnitVarianceImage:
    """Tests for build_unit_variance_image."""

    def test_returns_variance_image_product(self):
        """build_unit_variance_image returns a VarianceImageProduct."""
        img = np.ones((_NROWS, _NCOLS))
        prod = build_unit_variance_image(img)
        assert isinstance(prod, VarianceImageProduct)

    def test_all_ones(self):
        """variance_image is all-ones."""
        img = np.zeros((_NROWS, _NCOLS))
        prod = build_unit_variance_image(img)
        np.testing.assert_array_equal(prod.variance_image, 1.0)

    def test_shape_matches_input(self):
        """shape matches input image."""
        shape = (30, 60)
        img = np.zeros(shape)
        prod = build_unit_variance_image(img)
        assert prod.shape == shape

    def test_mode_stored(self):
        """mode is stored in the product."""
        img = np.ones((_NROWS, _NCOLS))
        prod = build_unit_variance_image(img, mode="H1")
        assert prod.mode == "H1"

    def test_non_2d_raises(self):
        """ValueError if image is 1-D."""
        img = np.ones(100)
        with pytest.raises(ValueError, match="2-D"):
            build_unit_variance_image(img)


# ===========================================================================
# 5. Error handling
# ===========================================================================


class TestErrorHandling:
    """Tests for ValueError on invalid inputs."""

    def test_non_2d_image_1d_raises(self):
        """ValueError if image is 1-D."""
        defn = VarianceModelDefinition(
            read_noise_electron=5.0, gain_e_per_adu=2.0
        )
        img = np.ones(100)
        with pytest.raises(ValueError, match="2-D"):
            build_variance_image(img, defn)

    def test_non_2d_image_3d_raises(self):
        """ValueError if image is 3-D."""
        defn = VarianceModelDefinition(
            read_noise_electron=5.0, gain_e_per_adu=2.0
        )
        img = np.ones((5, 10, 20))
        with pytest.raises(ValueError, match="2-D"):
            build_variance_image(img, defn)

    def test_invalid_read_noise_raises(self):
        """ValueError for negative read_noise_electron."""
        with pytest.raises(ValueError, match="read_noise_electron"):
            VarianceModelDefinition(
                read_noise_electron=-5.0, gain_e_per_adu=2.0
            )

    def test_invalid_gain_raises(self):
        """ValueError for non-positive gain_e_per_adu."""
        with pytest.raises(ValueError, match="gain_e_per_adu"):
            VarianceModelDefinition(
                read_noise_electron=5.0, gain_e_per_adu=0.0
            )

    def test_invalid_minimum_variance_raises(self):
        """ValueError for non-positive minimum_variance."""
        with pytest.raises(ValueError, match="minimum_variance"):
            VarianceModelDefinition(
                read_noise_electron=5.0,
                gain_e_per_adu=2.0,
                minimum_variance=0.0,
            )


# ===========================================================================
# 6. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1VarianceModelSmokeTest:
    """Smoke test using the real H1 iSHELL flat-field detector image.

    Verifies that the variance model scaffold runs without error on a real
    detector image and produces a well-formed output.
    """

    @pytest.fixture(scope="class")
    def h1_variance_result(self):
        """VarianceImageProduct from a real H1 detector image."""
        import astropy.io.fits as fits

        with fits.open(_H1_FLAT_FILES[0]) as hdul:
            detector_image = hdul[0].data.astype(float)

        defn = VarianceModelDefinition(
            read_noise_electron=10.0,
            gain_e_per_adu=2.0,
        )
        prod = build_variance_image(detector_image, defn, mode="H1")
        return prod, detector_image

    def test_returns_variance_image_product(self, h1_variance_result):
        """build_variance_image returns a VarianceImageProduct."""
        prod, _ = h1_variance_result
        assert isinstance(prod, VarianceImageProduct)

    def test_shape_matches_detector(self, h1_variance_result):
        """variance_image shape matches detector image shape."""
        prod, img = h1_variance_result
        assert prod.shape == img.shape

    def test_source_image_shape_matches(self, h1_variance_result):
        """source_image_shape matches detector image shape."""
        prod, img = h1_variance_result
        assert prod.source_image_shape == img.shape

    def test_all_finite(self, h1_variance_result):
        """All variance values are finite (no NaN/Inf from real data)."""
        prod, img = h1_variance_result
        # Only test pixels where image itself is finite.
        mask = np.isfinite(img)
        assert np.all(np.isfinite(prod.variance_image[mask]))

    def test_all_non_negative(self, h1_variance_result):
        """All variance values are >= 0."""
        prod, _ = h1_variance_result
        finite_mask = np.isfinite(prod.variance_image)
        assert np.all(prod.variance_image[finite_mask] >= 0.0)

    def test_minimum_variance_floor_respected(self, h1_variance_result):
        """All finite variance values respect the minimum_variance floor."""
        prod, _ = h1_variance_result
        floor = prod.definition.minimum_variance
        finite_mask = np.isfinite(prod.variance_image)
        assert np.all(prod.variance_image[finite_mask] >= floor)

    def test_finite_fraction_high(self, h1_variance_result):
        """finite_fraction is close to 1.0 for a real flat image."""
        prod, _ = h1_variance_result
        assert prod.finite_fraction > 0.9
