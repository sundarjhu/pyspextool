"""
Tests for the iSHELL spectral extraction module (extract.py).

Coverage
--------
* Successful extraction for representative J, H, and K mode configurations
  using synthetic science frames and calibration objects.
* Dimensional consistency of extracted spectra ((4, nwave) per aperture per order).
* Variance and bitmask propagation through extraction.
* Single-aperture (A mode) and default parameter behaviour.
* Failure cases: malformed preprocess_result dict, missing geometry
  wavelength solution, inconsistent aperture parameter shapes, empty geometry.

Provisional tilt model
----------------------
Tests that exercise extraction on rectified data are smoke tests: they verify
structural properties (shape, finiteness, flag propagation) rather than exact
pixel values, because the provisional zero-tilt placeholder is not
scientifically meaningful.

No real iSHELL science data is required.  All synthetic data is created
in temporary directories using the same helper conventions as
``test_ishell_preprocess.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.extract import (
    build_extraction_arrays,
    extract_spectra,
    _DEFAULT_APERTURE_RADIUS_ARCSEC,
)
from pyspextool.instruments.ishell.calibrations import FlatInfo
from pyspextool.instruments.ishell.geometry import (
    OrderGeometry,
    OrderGeometrySet,
)
from pyspextool.instruments.ishell.preprocess import preprocess_science_frames
from pyspextool.pyspextoolerror import pySpextoolError


# ---------------------------------------------------------------------------
# Constants shared across all tests
# ---------------------------------------------------------------------------

_NROWS = 64
_NCOLS = 64

# One representative mode per band (J, H, K) for parametric tests
REPRESENTATIVE_MODES = [
    ("J0", 262),   # (mode_name, a representative order number)
    ("H1", 201),
    ("K1", 233),
]


# ---------------------------------------------------------------------------
# Fixture / helper factory functions
# ---------------------------------------------------------------------------


def _make_ishell_header(mode="K1", **extra):
    """Return a minimal valid iSHELL primary FITS header."""
    hdr = fits.Header()
    hdr["INSTRUME"] = "iSHELL"
    hdr["DATE_OBS"] = "2024-06-15"
    hdr["TIME_OBS"] = "08:30:00.0"
    hdr["MJD_OBS"] = 60476.354167
    hdr["ITIME"] = 30.0
    hdr["CO_ADDS"] = 1
    hdr["NDR"] = 16
    hdr["TCS_RA"] = "05:35:17.3"
    hdr["TCS_DEC"] = "-05:23:28"
    hdr["TCS_HA"] = "01:22:00"
    hdr["TCS_AM"] = 1.12
    hdr["POSANGLE"] = 45.0
    hdr["IRAFNAME"] = "ishell_0042.fits"
    hdr["PASSBAND"] = mode
    hdr["SLIT"] = "0.375_K1"
    hdr["OBJECT"] = "HD12345"
    for k, v in extra.items():
        hdr[k] = v
    return hdr


def _make_synthetic_mef(
    path,
    sig_value=5000.0,
    ped_value=500.0,
    sig_sum_value=800.0,
    shape=(_NROWS, _NCOLS),
    mode="K1",
    **header_extra,
):
    """Write a 3-extension iSHELL MEF FITS file with known signal values."""
    hdr = _make_ishell_header(mode=mode, **header_extra)
    ext0 = fits.PrimaryHDU(
        data=np.full(shape, sig_value, dtype=np.float32), header=hdr
    )
    ext1 = fits.ImageHDU(data=np.full(shape, ped_value, dtype=np.float32))
    ext2 = fits.ImageHDU(data=np.full(shape, sig_sum_value, dtype=np.float32))
    hdul = fits.HDUList([ext0, ext1, ext2])
    hdul.writeto(str(path), overwrite=True)
    return str(path)


def _make_flat_info(
    mode="K1",
    shape=(_NROWS, _NCOLS),
    flat_value=1.0,
    n_orders=1,
    plate_scale=0.125,
    order_start=233,
):
    """Build a minimal FlatInfo with a constant flat image.

    When n_orders > 1 the detector rows are divided into equal non-overlapping
    strips (matching _make_geometry) so that edge evaluation stays valid.
    """
    nrows, ncols = shape
    orders = list(range(order_start, order_start + n_orders))
    xranges = np.array([[0, ncols - 1]] * n_orders, dtype=int)
    strip_height = nrows // (n_orders + 1)
    edge_coeffs_list = []
    slit_height_arcsec = strip_height * plate_scale
    for idx in range(n_orders):
        row_bot = float(idx * strip_height + 1)
        row_top = float((idx + 1) * strip_height - 1)
        edge_coeffs_list.append([[row_bot, 0.0], [row_top, 0.0]])
    edge_coeffs = np.array(edge_coeffs_list, dtype=float)
    image = np.full(shape, flat_value, dtype=np.float32)
    return FlatInfo(
        mode=mode,
        orders=orders,
        rotation=5,
        plate_scale_arcsec=plate_scale,
        slit_height_pixels=slit_height_arcsec / plate_scale,
        slit_height_arcsec=slit_height_arcsec,
        slit_range_pixels=(int(edge_coeffs[0, 0, 0]), int(edge_coeffs[-1, 1, 0])),
        resolving_power_pixel=23000.0,
        step=5,
        flat_fraction=0.85,
        comm_window=5,
        image=image,
        xranges=xranges,
        edge_coeffs=edge_coeffs,
        edge_degree=1,
    )


def _make_geometry(
    mode="K1",
    n_orders=1,
    shape=(_NROWS, _NCOLS),
    plate_scale=0.125,
    order_start=233,
):
    """Build a minimal OrderGeometrySet with wavelength and tilt solutions.

    When n_orders > 1 the detector rows are divided into equal non-overlapping
    strips so that each order has a distinct footprint.

    .. note::
        Tilt coefficients are set to zero (provisional placeholder), which
        is the current iSHELL behaviour.
    """
    nrows, ncols = shape
    orders = list(range(order_start, order_start + n_orders))
    # Divide rows into equal strips; leave a 1-row gap between orders
    strip_height = nrows // (n_orders + 1)
    geoms = []
    for idx, order in enumerate(orders):
        row_bot = float((idx + 0) * strip_height + 1)
        row_top = float((idx + 1) * strip_height - 1)
        g = OrderGeometry(
            order=order,
            x_start=0,
            x_end=ncols - 1,
            bottom_edge_coeffs=np.array([row_bot, 0.0]),
            top_edge_coeffs=np.array([row_top, 0.0]),
        )
        g.wave_coeffs = np.array([2.0, 1e-4])       # wave_um = 2.0 + 1e-4 * col
        g.spatcal_coeffs = np.array([0.0, plate_scale])
        g.tilt_coeffs = np.array([0.0])              # provisional zero-tilt
        geoms.append(g)
    return OrderGeometrySet(mode=mode, geometries=geoms)


def _make_preprocess_result(
    image_value=100.0,
    variance_value=10.0,
    bitmask_value=0,
    shape=(_NROWS, _NCOLS),
    subtraction_mode="A",
    rectified=False,
    tilt_provisional=False,
):
    """Return a minimal preprocess_result dict with constant-value arrays."""
    image = np.full(shape, image_value, dtype=float)
    variance = np.full(shape, variance_value, dtype=float)
    bitmask = np.full(shape, bitmask_value, dtype=np.uint8)
    return {
        "image": image,
        "variance": variance,
        "bitmask": bitmask,
        "hdrinfo": {"FILENAME": ("ishell_0042.fits", "filename")},
        "subtraction_mode": subtraction_mode,
        "flat_applied": True,
        "rectified": rectified,
        "tilt_provisional": tilt_provisional,
    }


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def flat_info():
    """FlatInfo with flat_value=1.0 for the default K1 mode."""
    return _make_flat_info(mode="K1", flat_value=1.0)


@pytest.fixture()
def geometry():
    """Minimal K1 OrderGeometrySet with one order and wavelength solution."""
    return _make_geometry(mode="K1")


@pytest.fixture()
def preprocess_result():
    """Minimal preprocess_result with uniform signal."""
    return _make_preprocess_result()


@pytest.fixture()
def a_file(tmp_path):
    """Single A-beam synthetic MEF file."""
    return _make_synthetic_mef(tmp_path / "ishell_A.fits", sig_value=5000.0)


@pytest.fixture()
def ab_files(tmp_path):
    """A-beam and B-beam synthetic MEF files."""
    fa = _make_synthetic_mef(tmp_path / "ishell_A.fits", sig_value=6000.0)
    fb = _make_synthetic_mef(tmp_path / "ishell_B.fits", sig_value=2000.0)
    return [fa, fb]


# ===========================================================================
# 1. Module-level checks
# ===========================================================================


class TestModuleAPI:
    """Public API is importable and callable."""

    def test_build_extraction_arrays_callable(self):
        assert callable(build_extraction_arrays)

    def test_extract_spectra_callable(self):
        assert callable(extract_spectra)

    def test_default_aperture_radius_positive(self):
        assert _DEFAULT_APERTURE_RADIUS_ARCSEC > 0


# ===========================================================================
# 2. build_extraction_arrays
# ===========================================================================


class TestBuildExtractionArrays:
    """Tests for build_extraction_arrays()."""

    def test_returns_three_arrays(self, geometry):
        result = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        assert len(result) == 3

    def test_ordermask_shape(self, geometry):
        ordermask, _, _ = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        assert ordermask.shape == (_NROWS, _NCOLS)

    def test_wavecal_shape(self, geometry):
        _, wavecal, _ = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        assert wavecal.shape == (_NROWS, _NCOLS)

    def test_spatcal_shape(self, geometry):
        _, _, spatcal = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        assert spatcal.shape == (_NROWS, _NCOLS)

    def test_ordermask_inside_order(self, geometry):
        """Pixels between the bottom and top edge should carry the order number."""
        ordermask, _, _ = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        expected_order = geometry.orders[0]
        assert (ordermask == expected_order).any()

    def test_ordermask_outside_order_zero(self, geometry):
        """Pixels outside all orders should be zero."""
        ordermask, _, _ = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        # Row 0 is above the top edge (rows 0..15 for a 64-row image)
        assert ordermask[0, 0] == 0

    def test_wavecal_nan_outside_order(self, geometry):
        """Pixels outside order footprints should be NaN in wavecal."""
        _, wavecal, _ = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        assert not np.isfinite(wavecal[0, 0])

    def test_wavecal_values_inside_order(self, geometry):
        """Wavelength should equal wave_coeffs evaluated at the column."""
        _, wavecal, _ = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        geom = geometry.geometries[0]
        col = _NCOLS // 2
        # Use actual order boundaries
        row_bot = int(np.ceil(geom.eval_bottom_edge(float(col))))
        row_top = int(np.floor(geom.eval_top_edge(float(col))))
        row_mid = (row_bot + row_top) // 2
        expected_wave = np.polynomial.polynomial.polyval(
            float(col), geom.wave_coeffs
        )
        assert np.isfinite(wavecal[row_mid, col])
        np.testing.assert_allclose(wavecal[row_mid, col], expected_wave,
                                   rtol=1e-6)

    def test_spatcal_nan_outside_order(self, geometry):
        """Pixels outside order footprints should be NaN in spatcal."""
        _, _, spatcal = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        assert not np.isfinite(spatcal[0, 0])

    def test_spatcal_zero_at_bottom_edge(self, geometry):
        """spatcal should be ~0 at the bottom edge row of the order."""
        _, _, spatcal = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        geom = geometry.geometries[0]
        col = _NCOLS // 2
        row_bot = int(np.floor(geom.eval_bottom_edge(float(col))))
        # The bottom edge row should have spatcal close to 0
        val = spatcal[row_bot, col]
        if np.isfinite(val):
            assert val >= 0.0
            assert val < 0.5  # definitely not at the top

    def test_spatcal_positive_at_top(self, geometry):
        """spatcal should be > 0 near the top edge of the order."""
        _, _, spatcal = build_extraction_arrays(geometry, _NROWS, _NCOLS)
        geom = geometry.geometries[0]
        col = _NCOLS // 2
        row_top = int(np.ceil(geom.eval_top_edge(float(col)))) - 1
        val = spatcal[row_top, col]
        if np.isfinite(val):
            assert val > 0.0

    def test_multiple_orders(self):
        """Two orders should both appear in ordermask with correct values."""
        geom = _make_geometry(mode="K1", n_orders=2, order_start=233)
        ordermask, wavecal, _ = build_extraction_arrays(geom, _NROWS, _NCOLS)
        for order in geom.orders:
            assert (ordermask == order).any()

    def test_raises_on_missing_wavecal(self):
        """Should raise ValueError if wave_coeffs not populated."""
        geom = OrderGeometrySet(
            mode="K1",
            geometries=[
                OrderGeometry(
                    order=233,
                    x_start=0,
                    x_end=_NCOLS - 1,
                    bottom_edge_coeffs=np.array([16.0, 0.0]),
                    top_edge_coeffs=np.array([48.0, 0.0]),
                )
            ],
        )
        with pytest.raises(ValueError, match="wavelength solution"):
            build_extraction_arrays(geom, _NROWS, _NCOLS)

    def test_raises_on_empty_geometry(self):
        """Should raise ValueError if OrderGeometrySet has no orders."""
        geom = OrderGeometrySet(mode="K1", geometries=[])
        with pytest.raises(ValueError, match="empty"):
            build_extraction_arrays(geom, _NROWS, _NCOLS)


# ===========================================================================
# 3. extract_spectra — basic output structure
# ===========================================================================


class TestExtractSpectraOutput:
    """Basic output structure tests for extract_spectra()."""

    def test_returns_tuple(self, preprocess_result, geometry):
        result = extract_spectra(preprocess_result, geometry)
        assert isinstance(result, tuple) and len(result) == 2

    def test_spectra_is_list(self, preprocess_result, geometry):
        spectra, _ = extract_spectra(preprocess_result, geometry)
        assert isinstance(spectra, list)

    def test_metadata_is_dict(self, preprocess_result, geometry):
        _, meta = extract_spectra(preprocess_result, geometry)
        assert isinstance(meta, dict)

    def test_spectra_length_one_order_one_ap(self, preprocess_result, geometry):
        """With 1 order and 1 aperture, spectra should have 1 element."""
        spectra, _ = extract_spectra(preprocess_result, geometry)
        assert len(spectra) == 1

    def test_spectrum_has_4_rows(self, preprocess_result, geometry):
        """Each extracted spectrum should be (4, nwave)."""
        spectra, _ = extract_spectra(preprocess_result, geometry)
        assert spectra[0].ndim == 2
        assert spectra[0].shape[0] == 4

    def test_spectrum_nwave_positive(self, preprocess_result, geometry):
        """The spectral axis must have at least one wavelength sample."""
        spectra, _ = extract_spectra(preprocess_result, geometry)
        assert spectra[0].shape[1] > 0

    def test_wavelength_row_finite(self, preprocess_result, geometry):
        """Wavelength row (row 0) should be finite."""
        spectra, _ = extract_spectra(preprocess_result, geometry)
        waves = spectra[0][0, :]
        assert np.isfinite(waves).any()

    def test_metadata_keys_present(self, preprocess_result, geometry):
        """Required metadata keys should be present."""
        _, meta = extract_spectra(preprocess_result, geometry)
        required = {
            "orders",
            "n_apertures",
            "aperture_positions_arcsec",
            "aperture_radii_arcsec",
            "aperture_signs",
            "subtraction_mode",
            "rectified",
            "tilt_provisional",
            "plate_scale_arcsec",
        }
        assert required.issubset(set(meta.keys()))

    def test_metadata_orders_match_geometry(self, preprocess_result, geometry):
        """meta['orders'] should match geometry.orders."""
        _, meta = extract_spectra(preprocess_result, geometry)
        assert meta["orders"] == geometry.orders

    def test_metadata_n_apertures_default_one(self, preprocess_result, geometry):
        _, meta = extract_spectra(preprocess_result, geometry)
        assert meta["n_apertures"] == 1

    def test_metadata_subtraction_mode_propagated(self, geometry):
        pre = _make_preprocess_result(subtraction_mode="A-B")
        _, meta = extract_spectra(pre, geometry)
        assert meta["subtraction_mode"] == "A-B"

    def test_metadata_rectified_propagated(self, geometry):
        pre = _make_preprocess_result(rectified=True, tilt_provisional=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, meta = extract_spectra(pre, geometry)
        assert meta["rectified"] is True

    def test_metadata_tilt_provisional_propagated(self, geometry):
        pre = _make_preprocess_result(rectified=True, tilt_provisional=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, meta = extract_spectra(pre, geometry)
        assert meta["tilt_provisional"] is True


# ===========================================================================
# 4. Dimensional consistency
# ===========================================================================


class TestDimensionalConsistency:
    """Verify that array shapes are self-consistent."""

    def test_all_spectrum_rows_same_length(self, preprocess_result, geometry):
        """All four rows of a spectrum should have the same length."""
        spectra, _ = extract_spectra(preprocess_result, geometry)
        sp = spectra[0]
        assert sp[0].shape == sp[1].shape == sp[2].shape == sp[3].shape

    def test_aperture_radii_shape_in_metadata(self, preprocess_result, geometry):
        """meta['aperture_radii_arcsec'] should be (norders, naps)."""
        _, meta = extract_spectra(preprocess_result, geometry)
        norders = len(geometry.orders)
        naps = meta["n_apertures"]
        assert meta["aperture_radii_arcsec"].shape == (norders, naps)

    def test_two_orders_gives_two_spectra(self):
        """With 2 orders and 1 aperture, we should get 2 spectra."""
        geom = _make_geometry(mode="K1", n_orders=2, order_start=233)
        pre = _make_preprocess_result()
        spectra, meta = extract_spectra(pre, geom)
        assert len(spectra) == 2
        assert meta["n_apertures"] == 1

    def test_two_apertures_gives_two_spectra_per_order(self):
        """With 1 order and 2 apertures, we should get 2 spectra."""
        geom = _make_geometry(mode="K1", n_orders=1)
        pre = _make_preprocess_result()
        nrows, ncols = pre["image"].shape
        geom_g = geom.geometries[0]
        col_mid = 0.5 * (geom_g.x_start + geom_g.x_end)
        row_bot = float(geom_g.eval_bottom_edge(col_mid))
        row_top = float(geom_g.eval_top_edge(col_mid))
        slit_arcsec = (row_top - row_bot) * 0.125
        pos = np.array([slit_arcsec * 0.25, slit_arcsec * 0.75])
        signs = np.array([1, -1])
        spectra, meta = extract_spectra(
            pre, geom, aperture_positions_arcsec=pos, aperture_signs=signs
        )
        assert len(spectra) == 2
        assert meta["n_apertures"] == 2

    @pytest.mark.parametrize("mode,order_start", REPRESENTATIVE_MODES)
    def test_representative_modes_extract(self, mode, order_start):
        """Successful extraction for J, H, K representative cases."""
        geom = _make_geometry(mode=mode, n_orders=1, order_start=order_start)
        pre = _make_preprocess_result()
        spectra, meta = extract_spectra(pre, geom)
        assert len(spectra) == 1
        assert spectra[0].shape[0] == 4
        assert spectra[0].shape[1] > 0
        assert meta["orders"] == geom.orders


# ===========================================================================
# 5. Variance propagation
# ===========================================================================


class TestVariancePropagation:
    """Verify that the variance array is used and propagated."""

    def test_uncertainty_row_nonnegative(self, preprocess_result, geometry):
        """Uncertainty (row 2) should be non-negative everywhere it is finite."""
        spectra, _ = extract_spectra(preprocess_result, geometry)
        unc = spectra[0][2, :]
        finite = np.isfinite(unc)
        if finite.any():
            assert (unc[finite] >= 0).all()

    def test_higher_variance_gives_higher_uncertainty(self, geometry):
        """Doubling the variance should increase (or maintain) uncertainty."""
        pre_low = _make_preprocess_result(variance_value=1.0)
        pre_high = _make_preprocess_result(variance_value=4.0)
        sp_low, _ = extract_spectra(pre_low, geometry)
        sp_high, _ = extract_spectra(pre_high, geometry)
        unc_low = sp_low[0][2, :]
        unc_high = sp_high[0][2, :]
        finite = np.isfinite(unc_low) & np.isfinite(unc_high)
        if finite.any():
            # Uncertainty should be at least as large for higher variance
            assert (unc_high[finite] >= unc_low[finite] - 1e-10).all()

    def test_zero_variance_gives_zero_uncertainty(self, geometry):
        """If variance is zero, uncertainty should be zero (or NaN)."""
        pre = _make_preprocess_result(variance_value=0.0)
        spectra, _ = extract_spectra(pre, geometry)
        unc = spectra[0][2, :]
        finite = np.isfinite(unc)
        if finite.any():
            np.testing.assert_allclose(unc[finite], 0.0, atol=1e-12)


# ===========================================================================
# 6. Bitmask propagation
# ===========================================================================


class TestBitmaskPropagation:
    """Verify that bitmask flags are propagated to the output spectral flags."""

    def test_clean_bitmask_flags_are_zero(self, geometry):
        """With a zero bitmask, output flags should all be zero."""
        pre = _make_preprocess_result(bitmask_value=0)
        spectra, _ = extract_spectra(pre, geometry)
        flags = spectra[0][3, :]
        assert (flags == 0).all()

    def test_nonzero_bitmask_flags_some_wavelengths(self, geometry):
        """With a non-zero bitmask inside the aperture, output flags should be set."""
        pre = _make_preprocess_result(bitmask_value=1)
        spectra, _ = extract_spectra(pre, geometry)
        flags = spectra[0][3, :]
        # All wavelengths should be flagged because the entire image is masked
        assert (flags > 0).any()

    def test_linearity_bit_set_flags_spectrum(self, geometry):
        """Linearity-flagged pixels (bitmask bit 0) should set spectral flags."""
        pre = _make_preprocess_result(bitmask_value=1)  # bit 0 = linearity
        spectra, _ = extract_spectra(pre, geometry)
        flags = spectra[0][3, :]
        assert (flags > 0).any()

    def test_flat_zero_bit_flags_spectrum(self, geometry):
        """Flat-zero-flagged pixels (bitmask bit 4) should set spectral flags."""
        pre = _make_preprocess_result(bitmask_value=16)  # bit 4 = flat zero
        spectra, _ = extract_spectra(pre, geometry)
        flags = spectra[0][3, :]
        assert (flags > 0).any()

    def test_nan_pixels_flagged(self, geometry):
        """NaN pixels in the image should be treated as bad and flag the spectrum."""
        pre = _make_preprocess_result()
        pre["image"] = np.full((_NROWS, _NCOLS), np.nan)
        spectra, _ = extract_spectra(pre, geometry)
        # NaN image → zero signal and flagged pixels
        flags = spectra[0][3, :]
        assert (flags > 0).any()


# ===========================================================================
# 7. Aperture parameter handling
# ===========================================================================


class TestApertureParameters:
    """Verify aperture radius/position/sign parameter handling."""

    def test_scalar_aperture_radius(self, preprocess_result, geometry):
        """Scalar aperture_radii_arcsec should work without error."""
        spectra, meta = extract_spectra(
            preprocess_result, geometry, aperture_radii_arcsec=0.3
        )
        assert meta["aperture_radii_arcsec"].shape == (1, 1)

    def test_array_aperture_radius_per_order(self, geometry):
        """1-D aperture_radii_arcsec with length == norders should work."""
        geom = _make_geometry(mode="K1", n_orders=2, order_start=233)
        pre = _make_preprocess_result()
        spectra, meta = extract_spectra(
            pre, geom, aperture_radii_arcsec=np.array([0.3, 0.4])
        )
        assert meta["aperture_radii_arcsec"].shape == (2, 1)

    def test_2d_aperture_radius(self, geometry):
        """2-D aperture_radii_arcsec with shape (norders, naps) should work."""
        radii = np.array([[0.5]])
        spectra, meta = extract_spectra(
            _make_preprocess_result(), geometry, aperture_radii_arcsec=radii
        )
        assert meta["aperture_radii_arcsec"].shape == (1, 1)

    def test_custom_aperture_position(self, preprocess_result, geometry):
        """A user-specified aperture position should be used."""
        geom_g = geometry.geometries[0]
        col_mid = 0.5 * (geom_g.x_start + geom_g.x_end)
        row_bot = float(geom_g.eval_bottom_edge(col_mid))
        row_top = float(geom_g.eval_top_edge(col_mid))
        slit_arcsec = (row_top - row_bot) * 0.125
        pos = np.array([slit_arcsec / 3.0])
        spectra, meta = extract_spectra(
            preprocess_result, geometry, aperture_positions_arcsec=pos
        )
        np.testing.assert_allclose(meta["aperture_positions_arcsec"], pos)

    def test_signs_positive_and_negative(self, geometry):
        """Extract with one positive and one negative aperture (A-B nodding)."""
        geom_g = geometry.geometries[0]
        col_mid = 0.5 * (geom_g.x_start + geom_g.x_end)
        row_bot = float(geom_g.eval_bottom_edge(col_mid))
        row_top = float(geom_g.eval_top_edge(col_mid))
        slit_arcsec = (row_top - row_bot) * 0.125
        pos = np.array([slit_arcsec * 0.3, slit_arcsec * 0.7])
        signs = np.array([1, -1])
        spectra, meta = extract_spectra(
            _make_preprocess_result(subtraction_mode="A-B"),
            geometry,
            aperture_positions_arcsec=pos,
            aperture_signs=signs,
        )
        assert len(spectra) == 2
        np.testing.assert_array_equal(meta["aperture_signs"], signs)


# ===========================================================================
# 8. Failure cases
# ===========================================================================


class TestFailureCases:
    """Failure cases for malformed inputs."""

    def test_missing_image_key(self, geometry):
        """preprocess_result without 'image' should raise pySpextoolError."""
        bad = {"variance": np.zeros((_NROWS, _NCOLS)),
               "bitmask": np.zeros((_NROWS, _NCOLS), dtype=np.uint8)}
        with pytest.raises(pySpextoolError, match="missing required keys"):
            extract_spectra(bad, geometry)

    def test_missing_variance_key(self, geometry):
        """preprocess_result without 'variance' should raise pySpextoolError."""
        bad = {"image": np.zeros((_NROWS, _NCOLS)),
               "bitmask": np.zeros((_NROWS, _NCOLS), dtype=np.uint8)}
        with pytest.raises(pySpextoolError, match="missing required keys"):
            extract_spectra(bad, geometry)

    def test_missing_bitmask_key(self, geometry):
        """preprocess_result without 'bitmask' should raise pySpextoolError."""
        bad = {"image": np.zeros((_NROWS, _NCOLS)),
               "variance": np.zeros((_NROWS, _NCOLS))}
        with pytest.raises(pySpextoolError, match="missing required keys"):
            extract_spectra(bad, geometry)

    def test_not_a_dict(self, geometry):
        """Passing a non-dict preprocess_result should raise pySpextoolError."""
        with pytest.raises(pySpextoolError):
            extract_spectra("not_a_dict", geometry)

    def test_geometry_without_wavecal(self):
        """OrderGeometrySet without wavelength solution should raise ValueError."""
        geom = OrderGeometrySet(
            mode="K1",
            geometries=[
                OrderGeometry(
                    order=233,
                    x_start=0,
                    x_end=_NCOLS - 1,
                    bottom_edge_coeffs=np.array([16.0, 0.0]),
                    top_edge_coeffs=np.array([48.0, 0.0]),
                )
            ],
        )
        pre = _make_preprocess_result()
        with pytest.raises(ValueError, match="wavelength solution"):
            extract_spectra(pre, geom)

    def test_geometry_empty(self):
        """Empty OrderGeometrySet should raise ValueError."""
        geom = OrderGeometrySet(mode="K1", geometries=[])
        pre = _make_preprocess_result()
        with pytest.raises(ValueError, match="empty"):
            extract_spectra(pre, geom)

    def test_mismatched_aperture_positions_and_signs(self, geometry):
        """Mismatched lengths for positions and signs should raise pySpextoolError."""
        pos = np.array([1.0, 2.0])      # length 2
        signs = np.array([1, -1, 1])   # length 3 — mismatch
        pre = _make_preprocess_result()
        with pytest.raises(pySpextoolError, match="same length"):
            extract_spectra(pre, geometry,
                            aperture_positions_arcsec=pos,
                            aperture_signs=signs)

    def test_wrong_aperture_radii_shape(self, geometry):
        """Wrong 2-D shape for aperture_radii_arcsec should raise pySpextoolError."""
        # norders=1, naps=1 → expect (1,1); supply (2,1)
        radii = np.ones((2, 1))
        pre = _make_preprocess_result()
        with pytest.raises(pySpextoolError, match="norders"):
            extract_spectra(pre, geometry, aperture_radii_arcsec=radii)

    def test_naps_gt_1_without_positions_raises(self, geometry):
        """naps > 1 without aperture_positions_arcsec should raise pySpextoolError."""
        pre = _make_preprocess_result()
        with pytest.raises(pySpextoolError, match="aperture_positions_arcsec"):
            extract_spectra(pre, geometry, aperture_signs=np.array([1, -1]))

    def test_tilt_provisional_emits_warning(self, geometry):
        """tilt_provisional=True in preprocess_result should emit RuntimeWarning."""
        pre = _make_preprocess_result(tilt_provisional=True)
        with pytest.warns(RuntimeWarning, match="tilt_provisional"):
            extract_spectra(pre, geometry)


# ===========================================================================
# 9. Integration with preprocess_science_frames
# ===========================================================================


class TestIntegrationWithPreprocess:
    """Integration tests combining preprocessing and extraction."""

    def test_a_mode_pipeline(self, tmp_path):
        """Full pipeline: single A-beam → preprocess → extract."""
        fi = _make_flat_info(mode="K1", flat_value=1.0)
        geom = _make_geometry(mode="K1")
        fa = _make_synthetic_mef(tmp_path / "a.fits", sig_value=5000.0)
        pre = preprocess_science_frames([fa], fi, subtraction_mode="A")
        spectra, meta = extract_spectra(pre, geom)
        assert len(spectra) == 1
        assert spectra[0].shape[0] == 4
        assert spectra[0].shape[1] > 0

    def test_ab_mode_pipeline(self, tmp_path):
        """Full pipeline: A-B pair → preprocess → extract."""
        fi = _make_flat_info(mode="K1", flat_value=1.0)
        geom = _make_geometry(mode="K1")
        fa = _make_synthetic_mef(tmp_path / "a.fits", sig_value=6000.0)
        fb = _make_synthetic_mef(tmp_path / "b.fits", sig_value=2000.0)
        pre = preprocess_science_frames([fa, fb], fi, subtraction_mode="A-B")
        spectra, meta = extract_spectra(pre, geom)
        assert len(spectra) == 1
        assert meta["subtraction_mode"] == "A-B"

    def test_rectified_pipeline(self, tmp_path):
        """Full pipeline with rectification: preprocess → extract (smoke test)."""
        fi = _make_flat_info(mode="K1", flat_value=1.0)
        geom = _make_geometry(mode="K1")
        fa = _make_synthetic_mef(tmp_path / "a.fits", sig_value=5000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pre = preprocess_science_frames(
                [fa], fi, geometry=geom, subtraction_mode="A"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spectra, meta = extract_spectra(pre, geom)
        assert len(spectra) == 1
        assert meta["tilt_provisional"] is True
        # Spectral axis should still have sensible length
        assert spectra[0].shape[1] > 0

    @pytest.mark.parametrize("mode,order_start", REPRESENTATIVE_MODES)
    def test_representative_mode_full_pipeline(self, tmp_path, mode, order_start):
        """End-to-end pipeline for J, H, K representative modes."""
        fi = _make_flat_info(mode=mode, flat_value=1.0,
                             order_start=order_start)
        geom = _make_geometry(mode=mode, n_orders=1,
                              order_start=order_start)
        fa = _make_synthetic_mef(tmp_path / f"a_{mode}.fits",
                                 sig_value=5000.0, mode=mode)
        pre = preprocess_science_frames([fa], fi, subtraction_mode="A")
        spectra, meta = extract_spectra(pre, geom)
        assert len(spectra) == 1
        assert spectra[0].shape == (4, spectra[0].shape[1])
        assert meta["orders"] == geom.orders
