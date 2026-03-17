"""
Tests for the aperture-aware spectral-extraction scaffold
(aperture_extraction.py).

Coverage:
  - ApertureDefinition: construction, validation, has_background property.
  - ExtractedApertureSpectrum: construction and field access.
  - ExtractedApertureSpectrumSet: construction, get_order, orders, n_orders.
  - extract_with_aperture on synthetic data:
      * returns correct type and mode,
      * n_orders matches rectified order set,
      * sum vs mean behaviour with known aperture,
      * background subtraction correctness,
      * sum vs mean difference,
      * NaN handling (all-NaN column, empty aperture),
      * aperture edge cases (full slit, single pixel, edge aperture),
      * wavelength-axis propagation and copy.
  - Error handling:
      * invalid ApertureDefinition (bad center, bad radius, bad background),
      * empty rectified order set → ValueError,
      * invalid extraction method → ValueError,
      * non-2-D flux image → ValueError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * full chain stages 1-8,
      * number of spectra matches number of orders,
      * wavelength axes are monotonic,
      * extraction output shapes are correct.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyspextool.instruments.ishell.aperture_extraction import (
    ApertureDefinition,
    ExtractedApertureSpectrum,
    ExtractedApertureSpectrumSet,
    extract_with_aperture,
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
    {"order_number": 311, "col_start": 50,  "col_end": 350, "center_row": 40,  "half_width": 8},
    {"order_number": 315, "col_start": 50,  "col_end": 350, "center_row": 100, "half_width": 8},
    {"order_number": 320, "col_start": 50,  "col_end": 350, "center_row": 160, "half_width": 8},
]


def _make_synthetic_rectification_index_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
) -> RectificationIndexSet:
    """Build a RectificationIndexSet from the synthetic order parameters."""
    index_orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        cr = float(p["center_row"])
        hw = float(p["half_width"])
        col_start = float(p["col_start"])
        col_end = float(p["col_end"])

        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        output_wavs = np.linspace(wav_start, wav_end, n_spectral)
        output_spatial_frac = np.linspace(0.0, 1.0, n_spatial)

        src_cols = np.linspace(col_start, col_end, n_spectral)
        bottom = cr - hw
        top = cr + hw
        src_rows = (
            bottom + output_spatial_frac[:, np.newaxis] * (top - bottom)
        ) * np.ones((1, n_spectral))

        index_orders.append(
            RectificationIndexOrder(
                order=p["order_number"],
                order_index=idx,
                output_wavelengths_um=output_wavs,
                output_spatial_frac=output_spatial_frac,
                src_cols=src_cols,
                src_rows=src_rows,
            )
        )
    return RectificationIndexSet(mode="H1_test", index_orders=index_orders)


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


def _make_center_aperture(radius: float = 0.2) -> ApertureDefinition:
    """Aperture centered at slit midpoint with no background."""
    return ApertureDefinition(center_frac=0.5, radius_frac=radius)


def _make_center_aperture_with_bg(
    radius: float = 0.2,
    bg_inner: float = 0.3,
    bg_outer: float = 0.45,
) -> ApertureDefinition:
    """Aperture centered at slit midpoint with background annulus."""
    return ApertureDefinition(
        center_frac=0.5,
        radius_frac=radius,
        background_inner=bg_inner,
        background_outer=bg_outer,
    )


# ===========================================================================
# 1. ApertureDefinition: construction and validation
# ===========================================================================


class TestApertureDefinitionConstruction:
    """Tests for ApertureDefinition construction and property access."""

    def test_basic_construction(self):
        """ApertureDefinition can be constructed with minimal args."""
        ap = ApertureDefinition(center_frac=0.5, radius_frac=0.1)
        assert ap.center_frac == 0.5
        assert ap.radius_frac == 0.1
        assert ap.background_inner is None
        assert ap.background_outer is None

    def test_has_background_false_by_default(self):
        """has_background is False when no background region is defined."""
        ap = ApertureDefinition(center_frac=0.5, radius_frac=0.1)
        assert not ap.has_background

    def test_has_background_true_when_defined(self):
        """has_background is True when background region is defined."""
        ap = ApertureDefinition(
            center_frac=0.5,
            radius_frac=0.1,
            background_inner=0.2,
            background_outer=0.4,
        )
        assert ap.has_background

    def test_center_at_zero(self):
        """center_frac=0 is valid."""
        ap = ApertureDefinition(center_frac=0.0, radius_frac=0.05)
        assert ap.center_frac == 0.0

    def test_center_at_one(self):
        """center_frac=1 is valid."""
        ap = ApertureDefinition(center_frac=1.0, radius_frac=0.05)
        assert ap.center_frac == 1.0

    def test_invalid_center_below_zero(self):
        """ValueError if center_frac < 0."""
        with pytest.raises(ValueError, match="center_frac"):
            ApertureDefinition(center_frac=-0.1, radius_frac=0.1)

    def test_invalid_center_above_one(self):
        """ValueError if center_frac > 1."""
        with pytest.raises(ValueError, match="center_frac"):
            ApertureDefinition(center_frac=1.1, radius_frac=0.1)

    def test_invalid_radius_zero(self):
        """ValueError if radius_frac == 0."""
        with pytest.raises(ValueError, match="radius_frac"):
            ApertureDefinition(center_frac=0.5, radius_frac=0.0)

    def test_invalid_radius_negative(self):
        """ValueError if radius_frac < 0."""
        with pytest.raises(ValueError, match="radius_frac"):
            ApertureDefinition(center_frac=0.5, radius_frac=-0.1)

    def test_background_inner_only_raises(self):
        """ValueError if only background_inner is provided."""
        with pytest.raises(ValueError, match="both"):
            ApertureDefinition(
                center_frac=0.5, radius_frac=0.1, background_inner=0.2
            )

    def test_background_outer_only_raises(self):
        """ValueError if only background_outer is provided."""
        with pytest.raises(ValueError, match="both"):
            ApertureDefinition(
                center_frac=0.5, radius_frac=0.1, background_outer=0.4
            )

    def test_background_inner_le_radius_raises(self):
        """ValueError if background_inner <= radius_frac."""
        with pytest.raises(ValueError, match="background_inner"):
            ApertureDefinition(
                center_frac=0.5,
                radius_frac=0.2,
                background_inner=0.2,  # equal to radius_frac → invalid
                background_outer=0.4,
            )

    def test_background_outer_le_inner_raises(self):
        """ValueError if background_outer <= background_inner."""
        with pytest.raises(ValueError, match="background_outer"):
            ApertureDefinition(
                center_frac=0.5,
                radius_frac=0.1,
                background_inner=0.3,
                background_outer=0.2,  # less than inner → invalid
            )


# ===========================================================================
# 2. ExtractedApertureSpectrum: construction
# ===========================================================================


class TestExtractedApertureSpectrumConstruction:
    """Unit tests for ExtractedApertureSpectrum construction and field access."""

    def _make_minimal(self) -> ExtractedApertureSpectrum:
        n = 32
        ap = ApertureDefinition(center_frac=0.5, radius_frac=0.2)
        return ExtractedApertureSpectrum(
            order=311,
            wavelength_um=np.linspace(1.64, 1.69, n),
            flux=np.ones(n),
            background_flux=None,
            aperture=ap,
            method="sum",
            n_pixels_used=8,
        )

    def test_construction(self):
        """ExtractedApertureSpectrum can be constructed."""
        sp = self._make_minimal()
        assert sp.order == 311

    def test_n_spectral(self):
        """n_spectral matches length of wavelength_um."""
        sp = self._make_minimal()
        assert sp.n_spectral == 32

    def test_background_flux_none(self):
        """background_flux can be None."""
        sp = self._make_minimal()
        assert sp.background_flux is None

    def test_method_stored(self):
        """method is stored correctly."""
        sp = self._make_minimal()
        assert sp.method == "sum"

    def test_n_pixels_used_stored(self):
        """n_pixels_used is stored correctly."""
        sp = self._make_minimal()
        assert sp.n_pixels_used == 8

    def test_aperture_stored(self):
        """aperture is stored correctly."""
        sp = self._make_minimal()
        assert sp.aperture.center_frac == 0.5


# ===========================================================================
# 3. ExtractedApertureSpectrumSet: construction and interface
# ===========================================================================


class TestExtractedApertureSpectrumSetConstruction:
    """Tests for ExtractedApertureSpectrumSet interface."""

    def _make_set(self) -> ExtractedApertureSpectrumSet:
        n = 32
        ap = ApertureDefinition(center_frac=0.5, radius_frac=0.2)
        spectra = [
            ExtractedApertureSpectrum(
                order=311 + i * 4,
                wavelength_um=np.linspace(1.55 + i * 0.05, 1.59 + i * 0.05, n),
                flux=np.ones(n) * (i + 1),
                background_flux=None,
                aperture=ap,
                method="sum",
                n_pixels_used=5,
            )
            for i in range(3)
        ]
        return ExtractedApertureSpectrumSet(mode="H1_test", spectra=spectra)

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
        """An empty ExtractedApertureSpectrumSet is valid."""
        ss = ExtractedApertureSpectrumSet(mode="H1_test")
        assert ss.n_orders == 0
        assert ss.orders == []


# ===========================================================================
# 4. extract_with_aperture: synthetic data
# ===========================================================================


class TestExtractWithApertureSynthetic:
    """Synthetic-data tests for extract_with_aperture."""

    def test_returns_extracted_aperture_spectrum_set(self):
        """extract_with_aperture returns an ExtractedApertureSpectrumSet."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap)
        assert isinstance(result, ExtractedApertureSpectrumSet)

    def test_mode_propagated(self):
        """mode is propagated from the rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap)
        assert result.mode == ros.mode

    def test_n_orders_matches(self):
        """n_orders matches the input rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap)
        assert result.n_orders == ros.n_orders

    def test_orders_list_matches(self):
        """orders list matches input rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap)
        assert result.orders == ros.orders

    def test_sum_output_shape(self):
        """Extracted flux has shape (n_spectral,) for sum method."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap, method="sum")
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.flux.shape == (ro.n_spectral,)

    def test_mean_output_shape(self):
        """Extracted flux has shape (n_spectral,) for mean method."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap, method="mean")
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.flux.shape == (ro.n_spectral,)

    def test_sum_known_aperture_value(self):
        """sum extraction with known fill equals fill * n_ap_pixels."""
        fill = 3.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ap = _make_center_aperture(radius=0.2)
        result = extract_with_aperture(ros, ap, method="sum", subtract_background=False)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            # Count spatial pixels inside aperture.
            dist = np.abs(ro.spatial_frac - ap.center_frac)
            n_ap = int(np.sum(dist <= ap.radius_frac))
            expected = fill * n_ap
            np.testing.assert_allclose(
                sp.flux,
                expected,
                rtol=1e-12,
                err_msg=f"Order {sp.order}: sum value incorrect",
            )

    def test_mean_known_aperture_value(self):
        """mean extraction of constant flux equals the fill value."""
        fill = 5.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ap = _make_center_aperture(radius=0.2)
        result = extract_with_aperture(ros, ap, method="mean", subtract_background=False)
        for sp in result.spectra:
            np.testing.assert_allclose(
                sp.flux,
                fill,
                rtol=1e-12,
                err_msg=f"Order {sp.order}: mean value incorrect",
            )

    def test_sum_vs_mean_ratio(self):
        """sum == mean * n_ap_pixels for uniform flux."""
        fill = 4.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ap = _make_center_aperture(radius=0.2)
        sum_r = extract_with_aperture(ros, ap, method="sum", subtract_background=False)
        mean_r = extract_with_aperture(ros, ap, method="mean", subtract_background=False)
        for sp_sum, sp_mean, ro in zip(sum_r.spectra, mean_r.spectra, ros.rectified_orders):
            dist = np.abs(ro.spatial_frac - ap.center_frac)
            n_ap = int(np.sum(dist <= ap.radius_frac))
            np.testing.assert_allclose(
                sp_sum.flux,
                sp_mean.flux * n_ap,
                rtol=1e-12,
                err_msg=f"Order {sp_sum.order}: sum != mean * n_ap",
            )

    def test_wavelength_axis_propagated(self):
        """wavelength_um is copied from the rectified order."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            np.testing.assert_array_equal(
                sp.wavelength_um,
                ro.wavelength_um,
                err_msg=f"Order {sp.order}: wavelength_um mismatch",
            )

    def test_wavelength_axis_is_copy(self):
        """wavelength_um in the output is a copy, not an alias."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap)
        original = ros.rectified_orders[0].wavelength_um.copy()
        ros.rectified_orders[0].wavelength_um[:] = 0.0
        np.testing.assert_array_equal(
            result.spectra[0].wavelength_um,
            original,
            err_msg="wavelength_um was not copied; mutation of source affected output",
        )

    def test_method_stored_sum(self):
        """method field is 'sum' when sum is used."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap, method="sum")
        for sp in result.spectra:
            assert sp.method == "sum"

    def test_method_stored_mean(self):
        """method field is 'mean' when mean is used."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap, method="mean")
        for sp in result.spectra:
            assert sp.method == "mean"

    def test_aperture_stored_in_spectrum(self):
        """aperture field is stored in each ExtractedApertureSpectrum."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap)
        for sp in result.spectra:
            assert sp.aperture is ap


# ===========================================================================
# 5. Background subtraction correctness
# ===========================================================================


class TestBackgroundSubtraction:
    """Tests verifying background subtraction behaviour."""

    def test_background_flux_none_when_no_bg_region(self):
        """background_flux is None when aperture has no background region."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap, subtract_background=True)
        for sp in result.spectra:
            assert sp.background_flux is None

    def test_background_flux_none_when_disabled(self):
        """background_flux is None when subtract_background=False."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture_with_bg()
        result = extract_with_aperture(ros, ap, subtract_background=False)
        for sp in result.spectra:
            assert sp.background_flux is None

    def test_background_flux_array_when_enabled(self):
        """background_flux is an ndarray when subtract_background=True and bg is defined."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture_with_bg()
        result = extract_with_aperture(ros, ap, subtract_background=True)
        for sp in result.spectra:
            assert sp.background_flux is not None
            assert sp.background_flux.shape == (sp.n_spectral,)

    def test_background_subtraction_result(self):
        """Background-subtracted mean flux == object_fill - bg_fill for flat images."""
        object_fill = 10.0
        bg_fill = 3.0
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL

        # Build a single order with different fill in aperture and background.
        spatial_frac = np.linspace(0.0, 1.0, n_spatial)
        flux = np.full((n_spatial, n_spectral), bg_fill)

        # Overwrite aperture region (center ± 0.2) with object_fill.
        center = 0.5
        radius = 0.2
        ap_mask = np.abs(spatial_frac - center) <= radius
        flux[ap_mask, :] = object_fill

        ro = RectifiedOrder(
            order=311,
            order_index=0,
            wavelength_um=np.linspace(1.55, 1.59, n_spectral),
            spatial_frac=spatial_frac,
            flux=flux,
            source_image_shape=(_NROWS, _NCOLS),
        )
        ros = RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=[ro],
            source_image_shape=(_NROWS, _NCOLS),
        )

        ap = ApertureDefinition(
            center_frac=center,
            radius_frac=radius,
            background_inner=radius + 0.05,
            background_outer=radius + 0.25,
        )

        result = extract_with_aperture(ros, ap, method="mean", subtract_background=True)
        sp = result.spectra[0]

        # background estimate should equal bg_fill.
        np.testing.assert_allclose(sp.background_flux, bg_fill, rtol=1e-12)
        # background-subtracted mean should equal object_fill - bg_fill.
        np.testing.assert_allclose(
            sp.flux, object_fill - bg_fill, rtol=1e-12
        )

    def test_no_subtraction_when_flag_false(self):
        """Setting subtract_background=False returns raw aperture sum."""
        fill = 7.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ap = _make_center_aperture_with_bg()
        result_raw = extract_with_aperture(ros, ap, method="mean", subtract_background=False)
        for sp in result_raw.spectra:
            np.testing.assert_allclose(sp.flux, fill, rtol=1e-12)


# ===========================================================================
# 6. NaN handling
# ===========================================================================


class TestNaNHandling:
    """Tests for NaN handling in extract_with_aperture."""

    def test_all_nan_aperture_column_produces_nan(self):
        """A spectral column where all aperture pixels are NaN → NaN output."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture(radius=0.2)
        # Set first spectral column of first order to all NaN.
        ros.rectified_orders[0].flux[:, 0] = np.nan
        result = extract_with_aperture(ros, ap, method="sum")
        assert np.isnan(result.spectra[0].flux[0])

    def test_partial_nan_row_excluded_from_mean(self):
        """A single NaN aperture row is excluded from the mean."""
        fill = 4.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ap = _make_center_aperture(radius=0.2)
        # Find an aperture row and set it to NaN.
        ro = ros.rectified_orders[0]
        dist = np.abs(ro.spatial_frac - ap.center_frac)
        ap_indices = np.where(dist <= ap.radius_frac)[0]
        ro.flux[ap_indices[0], :] = np.nan
        result = extract_with_aperture(ros, ap, method="mean", subtract_background=False)
        # Mean of remaining aperture rows (all fill) is still fill.
        np.testing.assert_allclose(result.spectra[0].flux, fill, rtol=1e-12)

    def test_empty_aperture_gives_all_nan(self):
        """An aperture too narrow to capture any pixels → all-NaN flux."""
        ros = _make_synthetic_rectified_order_set()
        # spatial_frac is linspace(0, 1, _N_SPATIAL=20); spacing ≈ 0.0526.
        # center_frac=0.5 does not coincide with any grid point (9/19 ≈ 0.4737,
        # 10/19 ≈ 0.5263), so radius=1e-6 selects no pixels → all NaN.
        ap = ApertureDefinition(center_frac=0.5, radius_frac=1e-6)
        result = extract_with_aperture(ros, ap, method="sum")
        for sp in result.spectra:
            assert sp.flux.shape == (sp.n_spectral,)
            assert np.all(np.isnan(sp.flux)), (
                f"Order {sp.order}: expected all-NaN flux for empty aperture"
            )


# ===========================================================================
# 7. Aperture edge cases
# ===========================================================================


class TestApertureEdgeCases:
    """Tests for edge cases in aperture definitions."""

    def test_full_slit_aperture(self):
        """An aperture covering the full slit includes all spatial pixels."""
        fill = 2.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        # Aperture from 0 to 1 (full slit).
        ap = ApertureDefinition(center_frac=0.5, radius_frac=0.5)
        result = extract_with_aperture(ros, ap, method="sum", subtract_background=False)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            expected = fill * ro.n_spatial
            np.testing.assert_allclose(sp.flux, expected, rtol=1e-12)

    def test_aperture_at_slit_edge(self):
        """An aperture near the slit edge (center_frac near 0 or 1) is valid."""
        ros = _make_synthetic_rectified_order_set()
        ap = ApertureDefinition(center_frac=0.05, radius_frac=0.1)
        result = extract_with_aperture(ros, ap, method="sum")
        assert result.n_orders == ros.n_orders

    def test_very_narrow_aperture(self):
        """A very narrow aperture still produces correct shape output."""
        ros = _make_synthetic_rectified_order_set()
        ap = ApertureDefinition(center_frac=0.5, radius_frac=0.01)
        result = extract_with_aperture(ros, ap, method="sum")
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.flux.shape == (ro.n_spectral,)

    def test_n_pixels_used_positive(self):
        """n_pixels_used > 0 for a reasonable aperture."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture(radius=0.2)
        result = extract_with_aperture(ros, ap)
        for sp in result.spectra:
            assert sp.n_pixels_used > 0

    def test_n_pixels_used_decreases_with_nan_row(self):
        """n_pixels_used decreases when an aperture row is set to all-NaN."""
        fill = 2.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ap = _make_center_aperture(radius=0.2)
        ro = ros.rectified_orders[0]
        dist = np.abs(ro.spatial_frac - ap.center_frac)
        ap_indices = np.where(dist <= ap.radius_frac)[0]

        # Baseline n_pixels_used.
        result_full = extract_with_aperture(ros, ap)
        n_full = result_full.spectra[0].n_pixels_used

        # Set one aperture row to all-NaN.
        ro.flux[ap_indices[0], :] = np.nan
        result_nan = extract_with_aperture(ros, ap)
        n_nan = result_nan.spectra[0].n_pixels_used

        assert n_nan == n_full - 1


# ===========================================================================
# 8. Parametric shape-consistency tests
# ===========================================================================


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 10), (64, 40), (128, 50)])
def test_output_shapes_parametric(n_spectral, n_spatial):
    """Extracted spectra have correct shapes for various (n_spectral, n_spatial)."""
    ros = _make_synthetic_rectified_order_set(
        n_spectral=n_spectral, n_spatial=n_spatial
    )
    ap = _make_center_aperture(radius=0.2)
    for method in ("sum", "mean"):
        result = extract_with_aperture(ros, ap, method=method)
        for sp in result.spectra:
            assert sp.flux.shape == (n_spectral,), (
                f"method={method}: flux shape {sp.flux.shape} != ({n_spectral},)"
            )
            assert sp.wavelength_um.shape == (n_spectral,)


# ===========================================================================
# 9. Error handling
# ===========================================================================


class TestExtractWithApertureErrors:
    """Error paths for extract_with_aperture."""

    def test_raises_on_empty_rectified_order_set(self):
        """ValueError if rectified_orders has no orders."""
        empty_ros = RectifiedOrderSet(mode="test")
        ap = _make_center_aperture()
        with pytest.raises(ValueError, match="empty"):
            extract_with_aperture(empty_ros, ap)

    def test_raises_on_invalid_method(self):
        """ValueError if method is not 'sum' or 'mean'."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        with pytest.raises(ValueError, match="method"):
            extract_with_aperture(ros, ap, method="optimal")

    def test_raises_on_non_2d_flux(self):
        """ValueError if a flux array is not 2-D."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        ros.rectified_orders[0].flux = np.ones(_N_SPECTRAL)
        with pytest.raises(ValueError, match="2-D"):
            extract_with_aperture(ros, ap)

    def test_raises_on_invalid_method_empty_string(self):
        """ValueError if method is an empty string."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        with pytest.raises(ValueError, match="method"):
            extract_with_aperture(ros, ap, method="")

    def test_raises_on_variance_shape_mismatch(self):
        """ValueError if variance_image shape does not match flux shape."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ap = _make_center_aperture()
        bad_variance = np.ones((_N_SPATIAL + 1, _N_SPECTRAL))
        with pytest.raises(ValueError, match="variance_image shape"):
            extract_with_aperture(ros, ap, variance_image=bad_variance)


# ===========================================================================
# 10. Variance propagation
# ===========================================================================


class TestExtractWithApertureVariancePropagation:
    """Tests for variance propagation in extract_with_aperture."""

    def test_variance_none_when_no_variance_image(self):
        """variance is None when variance_image is not provided."""
        ros = _make_synthetic_rectified_order_set()
        ap = _make_center_aperture()
        result = extract_with_aperture(ros, ap)
        for sp in result.spectra:
            assert sp.variance is None

    def test_variance_shape_matches_flux(self):
        """variance has the same shape as flux when variance_image is provided."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ap = _make_center_aperture()
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result = extract_with_aperture(ros, ap, variance_image=var_image)
        for sp in result.spectra:
            assert sp.variance is not None
            assert sp.variance.shape == sp.flux.shape

    def test_variance_nonnegative(self):
        """Propagated variance is non-negative everywhere it is finite."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ap = _make_center_aperture()
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result = extract_with_aperture(ros, ap, variance_image=var_image)
        for sp in result.spectra:
            finite_mask = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite_mask] >= 0.0)

    def test_variance_with_background_subtraction(self):
        """Variance is propagated correctly when background subtraction is active."""
        ros = _make_synthetic_rectified_order_set(
            n_spatial=_N_SPATIAL, n_spectral=_N_SPECTRAL
        )
        ap = _make_center_aperture_with_bg()
        var_image = np.ones((_N_SPATIAL, _N_SPECTRAL))
        result = extract_with_aperture(
            ros, ap, subtract_background=True, variance_image=var_image
        )
        for sp in result.spectra:
            assert sp.variance is not None
            assert sp.variance.shape == sp.flux.shape
            finite_mask = np.isfinite(sp.variance)
            assert np.all(sp.variance[finite_mask] >= 0.0)

    def test_variance_unit_image_no_background(self):
        """Unit variance image gives deterministic variance values.

        For unit variance, n_ap aperture pixels, and method='sum':
          variance_1d = nansum(1 * n_ap) = n_ap for each column.
        """
        n_spatial = _N_SPATIAL
        n_spectral = _N_SPECTRAL
        ros = _make_synthetic_rectified_order_set(
            n_spatial=n_spatial, n_spectral=n_spectral, fill_value=1.0
        )
        ap = _make_center_aperture(radius=0.2)
        var_image = np.ones((n_spatial, n_spectral))
        result = extract_with_aperture(
            ros, ap,
            method="sum",
            subtract_background=False,
            variance_image=var_image,
        )
        for sp in result.spectra:
            # variance = sum of n_ap unit variances = n_ap for each column.
            ro = next(
                r for r in ros.rectified_orders if r.order == sp.order
            )
            dist = np.abs(ro.spatial_frac - ap.center_frac)
            n_ap = int(np.sum(dist <= ap.radius_frac))
            np.testing.assert_allclose(
                sp.variance, float(n_ap), rtol=1e-12
            )


# ===========================================================================
# 11. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1ApertureExtractionSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset.

    Chain:
      1. Flat-order tracing (Stage 1)
      2. Arc-line tracing (Stage 2)
      3. Provisional wavelength mapping (Stage 3)
      4. Coefficient-surface refinement (Stage 5)
      5. Rectification indices (Stage 6)
      6. Rectified orders (Stage 7)
      7. Aperture extraction (Stage 11)
    """

    @pytest.fixture(scope="class")
    def h1_aperture_result(self):
        """ExtractedApertureSpectrumSet from the real H1 calibration chain."""
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

        # Stage 11: aperture extraction
        ap = ApertureDefinition(
            center_frac=0.5,
            radius_frac=0.2,
            background_inner=0.3,
            background_outer=0.45,
        )
        extracted = extract_with_aperture(
            rectified, ap, method="sum", subtract_background=True
        )

        return extracted, rectified

    def test_returns_extracted_aperture_spectrum_set(self, h1_aperture_result):
        """extract_with_aperture returns an ExtractedApertureSpectrumSet."""
        extracted, _ = h1_aperture_result
        assert isinstance(extracted, ExtractedApertureSpectrumSet)

    def test_n_orders_matches_rectified(self, h1_aperture_result):
        """n_orders matches the rectified order set."""
        extracted, rectified = h1_aperture_result
        assert extracted.n_orders == rectified.n_orders

    def test_n_orders_positive(self, h1_aperture_result):
        """n_orders > 0."""
        extracted, _ = h1_aperture_result
        assert extracted.n_orders > 0

    def test_wavelength_axes_monotonic(self, h1_aperture_result):
        """wavelength_um is monotonic for every extracted spectrum."""
        extracted, _ = h1_aperture_result
        for sp in extracted.spectra:
            diffs = np.diff(sp.wavelength_um)
            assert np.all(diffs > 0) or np.all(diffs < 0), (
                f"Order {sp.order}: wavelength axis is not monotonic"
            )

    def test_output_shapes_correct(self, h1_aperture_result):
        """flux shape equals wavelength_um shape for every spectrum."""
        extracted, _ = h1_aperture_result
        for sp in extracted.spectra:
            assert sp.flux.shape == sp.wavelength_um.shape

    def test_background_flux_shape(self, h1_aperture_result):
        """background_flux has same shape as flux when subtraction was applied."""
        extracted, _ = h1_aperture_result
        for sp in extracted.spectra:
            assert sp.background_flux is not None
            assert sp.background_flux.shape == sp.flux.shape

    def test_orders_list_matches_rectified(self, h1_aperture_result):
        """orders list matches the rectified order set."""
        extracted, rectified = h1_aperture_result
        assert extracted.orders == rectified.orders

    def test_get_order_works_for_all_orders(self, h1_aperture_result):
        """get_order succeeds for every order in the set."""
        extracted, _ = h1_aperture_result
        for order in extracted.orders:
            sp = extracted.get_order(order)
            assert sp.order == order

    def test_n_pixels_used_positive(self, h1_aperture_result):
        """n_pixels_used > 0 for every order."""
        extracted, _ = h1_aperture_result
        for sp in extracted.spectra:
            assert sp.n_pixels_used > 0
