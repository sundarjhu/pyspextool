"""
Tests for the iSHELL science-frame preprocessing pipeline (preprocess.py).

Coverage
--------
* Successful preprocessing for representative J, H, and K modes using
  synthetic MEF fixtures and minimal calibration objects.
* Dimensional consistency of all output arrays.
* Correct handling of A-B and A subtraction modes.
* Flat-field division (correct quotient, bad-pixel flagging).
* Rectification with provisional geometry (smoke test only — actual pixel
  values depend on the zero-tilt placeholder mapping).
* Failure cases for malformed or inconsistent inputs.

No real iSHELL science data is required.  All synthetic FITS files are
created in temporary directories using helpers adapted from
``test_ishell_ingestion.py``.

Provisional tilt model
----------------------
The current order-rectification path uses a zero-tilt placeholder.  Tests
that exercise rectification are deliberately written as smoke tests (shape,
NaN structure, flag propagation) rather than exact pixel-value assertions,
because the provisional mapping is not scientifically meaningful.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.preprocess import (
    SUPPORTED_SUBTRACTION_MODES,
    preprocess_science_frames,
)
from pyspextool.instruments.ishell.calibrations import FlatInfo
from pyspextool.instruments.ishell.geometry import (
    OrderGeometry,
    OrderGeometrySet,
)
from pyspextool.pyspextoolerror import pySpextoolError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Small detector size to keep tests fast
_NROWS = 64
_NCOLS = 64

# Representative one mode per band
REPRESENTATIVE_MODES = ["J0", "H1", "K1"]


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
    sig_value=1000.0,
    ped_value=500.0,
    sig_sum_value=800.0,
    shape=(_NROWS, _NCOLS),
    mode="K1",
    **header_extra,
):
    """
    Write a 3-extension iSHELL MEF FITS file to *path* with known values.

    * Extension 0 (PRIMARY): signal difference ``sig_value`` (DN total).
    * Extension 1 (IMAGE):   pedestal sum ``ped_value``.
    * Extension 2 (IMAGE):   signal sum ``sig_sum_value``.
    """
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
):
    """
    Build a minimal :class:`FlatInfo` with a constant flat image.

    One order is placed in the middle rows of the image so that edge-
    polynomial evaluation stays within the valid column range.
    """
    # Place orders in the middle half of the image
    orders = list(range(233, 233 + n_orders))
    nrows, ncols = shape
    # x_ranges: each order spans the full column range
    xranges = np.array([[0, ncols - 1]] * n_orders, dtype=int)
    # Edge polynomials: flat bottom=nrows//4, top=3*nrows//4 (constant rows)
    row_bot = nrows // 4
    row_top = 3 * nrows // 4
    edge_coeffs = np.array(
        [[[float(row_bot), 0.0], [float(row_top), 0.0]]] * n_orders,
        dtype=float,
    )
    image = np.full(shape, flat_value, dtype=np.float32)
    slit_height_arcsec = 5.0
    return FlatInfo(
        mode=mode,
        orders=orders,
        rotation=5,
        plate_scale_arcsec=plate_scale,
        slit_height_pixels=slit_height_arcsec / plate_scale,
        slit_height_arcsec=slit_height_arcsec,
        slit_range_pixels=(row_bot, row_top),
        resolving_power_pixel=23000.0,
        step=5,
        flat_fraction=0.85,
        comm_window=5,
        image=image,
        xranges=xranges,
        edge_coeffs=edge_coeffs,
        edge_degree=1,
    )


def _make_geometry(mode="K1", n_orders=1, shape=(_NROWS, _NCOLS)):
    """
    Build a minimal :class:`OrderGeometrySet` with a wavelength solution.

    .. note::
        Tilt coefficients are set to zero (provisional placeholder), which
        is the current iSHELL behaviour.

    The wavelength solution is a linear mapping
    ``wave_um = 2.0 + 1e-4 * col`` so that the output wavelength grid
    contains sensible values.
    """
    nrows, ncols = shape
    orders = list(range(233, 233 + n_orders))
    row_bot = float(nrows // 4)
    row_top = float(3 * nrows // 4)
    geoms = []
    for order in orders:
        g = OrderGeometry(
            order=order,
            x_start=0,
            x_end=ncols - 1,
            bottom_edge_coeffs=np.array([row_bot, 0.0]),
            top_edge_coeffs=np.array([row_top, 0.0]),
        )
        # Populate wavelength solution and spatial calibration
        g.wave_coeffs = np.array([2.0, 1e-4])   # wave_um = 2 + 1e-4 * col
        g.spatcal_coeffs = np.array([0.0, 0.125])  # arcsec = 0.125 * row_offset
        # Provisional zero-tilt placeholder
        g.tilt_coeffs = np.array([0.0])
        geoms.append(g)
    return OrderGeometrySet(mode=mode, geometries=geoms)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def a_file(tmp_path):
    """Single A-beam synthetic MEF file."""
    return _make_synthetic_mef(
        tmp_path / "ishell_A.fits", sig_value=6000.0, mode="K1"
    )


@pytest.fixture()
def ab_files(tmp_path):
    """A-beam and B-beam synthetic MEF files with different signal values."""
    fa = _make_synthetic_mef(
        tmp_path / "ishell_A.fits", sig_value=6000.0, mode="K1"
    )
    fb = _make_synthetic_mef(
        tmp_path / "ishell_B.fits", sig_value=2000.0, mode="K1"
    )
    return [fa, fb]


@pytest.fixture()
def flat_info():
    """Minimal FlatInfo with a flat-value=2.0 image."""
    return _make_flat_info(mode="K1", flat_value=2.0)


@pytest.fixture()
def geometry():
    """Minimal OrderGeometrySet for K1 with one order and a wavelength solution."""
    return _make_geometry(mode="K1")


# ===========================================================================
# 1. Module-level checks
# ===========================================================================


class TestModuleAPI:
    """Public API constants and function are importable."""

    def test_supported_subtraction_modes_exists(self):
        assert SUPPORTED_SUBTRACTION_MODES is not None

    def test_a_mode_in_supported(self):
        assert "A" in SUPPORTED_SUBTRACTION_MODES

    def test_ab_mode_in_supported(self):
        assert "A-B" in SUPPORTED_SUBTRACTION_MODES

    def test_preprocess_callable(self):
        assert callable(preprocess_science_frames)


# ===========================================================================
# 2. A mode — single-frame preprocessing
# ===========================================================================


class TestAMode:
    """Single A-beam (no sky subtraction) preprocessing."""

    def test_returns_dict(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert isinstance(result, dict)

    def test_required_keys_present(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        required = {
            "image", "variance", "bitmask", "hdrinfo",
            "subtraction_mode", "flat_applied", "rectified", "tilt_provisional",
        }
        assert required.issubset(set(result.keys()))

    def test_subtraction_mode_recorded(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert result["subtraction_mode"] == "A"

    def test_flat_applied_true(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert result["flat_applied"] is True

    def test_rectified_false_without_geometry(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert result["rectified"] is False

    def test_tilt_provisional_false_without_geometry(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert result["tilt_provisional"] is False

    def test_image_shape(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert result["image"].shape == (_NROWS, _NCOLS)

    def test_variance_shape(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert result["variance"].shape == (_NROWS, _NCOLS)

    def test_bitmask_shape(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert result["bitmask"].shape == (_NROWS, _NCOLS)

    def test_hdrinfo_is_dict(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        assert isinstance(result["hdrinfo"], dict)

    def test_image_values_flat_divided(self, tmp_path, flat_info):
        """Image pixel values should equal raw_signal / exptime / flat."""
        # sig = 6000, itime = 30, coadds = 1 → img_before_flat = 200 DN/s
        # flat = 2.0 → expected = 100 DN/s
        fa = _make_synthetic_mef(
            tmp_path / "a.fits", sig_value=6000.0, mode="K1"
        )
        result = preprocess_science_frames([fa], flat_info, subtraction_mode="A")
        img = result["image"]
        # All finite pixels should be 100 DN/s
        finite = np.isfinite(img)
        assert finite.any(), "No finite pixels found"
        np.testing.assert_allclose(img[finite], 100.0, rtol=1e-5)

    def test_variance_positive(self, a_file, flat_info):
        result = preprocess_science_frames([a_file], flat_info, subtraction_mode="A")
        finite_var = result["variance"][np.isfinite(result["variance"])]
        assert (finite_var > 0).all()


# ===========================================================================
# 3. A-B mode — pair-subtracted preprocessing
# ===========================================================================


class TestABMode:
    """A-B nod-pair subtraction preprocessing."""

    def test_returns_dict(self, ab_files, flat_info):
        result = preprocess_science_frames(
            ab_files, flat_info, subtraction_mode="A-B"
        )
        assert isinstance(result, dict)

    def test_subtraction_mode_recorded(self, ab_files, flat_info):
        result = preprocess_science_frames(
            ab_files, flat_info, subtraction_mode="A-B"
        )
        assert result["subtraction_mode"] == "A-B"

    def test_image_shape(self, ab_files, flat_info):
        result = preprocess_science_frames(
            ab_files, flat_info, subtraction_mode="A-B"
        )
        assert result["image"].shape == (_NROWS, _NCOLS)

    def test_variance_shape(self, ab_files, flat_info):
        result = preprocess_science_frames(
            ab_files, flat_info, subtraction_mode="A-B"
        )
        assert result["variance"].shape == (_NROWS, _NCOLS)

    def test_bitmask_shape(self, ab_files, flat_info):
        result = preprocess_science_frames(
            ab_files, flat_info, subtraction_mode="A-B"
        )
        assert result["bitmask"].shape == (_NROWS, _NCOLS)

    def test_pair_subtraction_value(self, tmp_path):
        """
        After A-B subtraction and flat division, pixel values should be
        (sig_A - sig_B) / exptime / flat.

        sig_A = 9000, sig_B = 3000, itime = 30, coadds = 1, flat = 3.0
        img_A = 9000 / 30 = 300 DN/s
        img_B = 3000 / 30 = 100 DN/s
        A-B  = 200 DN/s
        /flat= 200 / 3.0 = 66.667 DN/s
        """
        fi = _make_flat_info(mode="K1", flat_value=3.0)
        fa = _make_synthetic_mef(
            tmp_path / "a.fits", sig_value=9000.0, mode="K1"
        )
        fb = _make_synthetic_mef(
            tmp_path / "b.fits", sig_value=3000.0, mode="K1"
        )
        result = preprocess_science_frames(
            [fa, fb], fi, subtraction_mode="A-B"
        )
        img = result["image"]
        finite = np.isfinite(img)
        assert finite.any()
        expected = (9000.0 - 3000.0) / 30.0 / 3.0   # = 66.667 DN/s
        np.testing.assert_allclose(
            img[finite], expected, rtol=1e-4,
            err_msg="A-B subtracted and flat-divided value is incorrect"
        )

    def test_variance_is_sum(self, tmp_path):
        """Variance from A-B mode should be var_A + var_B (propagated through flat)."""
        fi = _make_flat_info(mode="K1", flat_value=1.0)
        fa = _make_synthetic_mef(tmp_path / "a.fits", sig_value=1000.0)
        fb = _make_synthetic_mef(tmp_path / "b.fits", sig_value=1000.0)
        result_ab = preprocess_science_frames([fa, fb], fi, subtraction_mode="A-B")
        result_a = preprocess_science_frames([fa], fi, subtraction_mode="A")
        # var(A-B) ≈ var_A + var_B = 2 * var_A (since both frames are identical)
        var_ab = result_ab["variance"]
        var_a = result_a["variance"]
        finite = np.isfinite(var_ab) & np.isfinite(var_a)
        assert finite.any()
        np.testing.assert_allclose(
            var_ab[finite], 2.0 * var_a[finite], rtol=1e-5
        )

    def test_bitmask_is_union(self, tmp_path):
        """Bitmask in A-B mode should be the bitwise OR of both frames' masks."""
        # Create a frame with a non-linear pixel using a high ped+sig sum
        fi = _make_flat_info(mode="K1", flat_value=1.0, shape=(_NROWS, _NCOLS))
        # Normal A-beam
        fa = _make_synthetic_mef(
            tmp_path / "a.fits",
            sig_value=100.0, ped_value=50.0, sig_sum_value=80.0,
        )
        # B-beam with non-linear pixels (ped + sig > 30000)
        fb = _make_synthetic_mef(
            tmp_path / "b.fits",
            sig_value=100.0, ped_value=20000.0, sig_sum_value=15000.0,
        )
        result = preprocess_science_frames([fa, fb], fi, subtraction_mode="A-B")
        # Linearity bit (bit 0) should be set for all pixels in bitmask
        # because B-beam has ped+sig = 35000 > 30000
        lin_bit = 0
        assert np.all(result["bitmask"] & np.uint8(2 ** lin_bit) > 0)

    def test_hdrinfo_from_a_beam(self, tmp_path):
        """Header info must come from the A-beam, not the B-beam."""
        fi = _make_flat_info(mode="K1", flat_value=1.0)
        fa = _make_synthetic_mef(
            tmp_path / "a.fits", mode="K1", OBJECT="StarA"
        )
        fb = _make_synthetic_mef(
            tmp_path / "b.fits", mode="K1", OBJECT="StarB"
        )
        result = preprocess_science_frames([fa, fb], fi, subtraction_mode="A-B")
        # IRAFNAME should come from A-beam file (ishell_0042.fits from helper)
        assert result["hdrinfo"]["FILENAME"][0] == "ishell_0042.fits"


# ===========================================================================
# 4. Rectification
# ===========================================================================


class TestRectification:
    """
    Smoke tests for order rectification with the provisional zero-tilt model.

    These tests verify structural properties (shape, NaN masking, flag bits)
    rather than exact pixel values, because the provisional zero-tilt model
    does not produce scientifically meaningful rectification.
    """

    def test_rectified_true_with_geometry(self, a_file, flat_info, geometry):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = preprocess_science_frames(
                [a_file], flat_info, geometry=geometry, subtraction_mode="A"
            )
        assert result["rectified"] is True

    def test_tilt_provisional_true_with_geometry(
        self, a_file, flat_info, geometry
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = preprocess_science_frames(
                [a_file], flat_info, geometry=geometry, subtraction_mode="A"
            )
        assert result["tilt_provisional"] is True

    def test_image_shape_after_rectification(self, a_file, flat_info, geometry):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = preprocess_science_frames(
                [a_file], flat_info, geometry=geometry, subtraction_mode="A"
            )
        assert result["image"].shape == (_NROWS, _NCOLS)

    def test_variance_shape_after_rectification(
        self, a_file, flat_info, geometry
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = preprocess_science_frames(
                [a_file], flat_info, geometry=geometry, subtraction_mode="A"
            )
        assert result["variance"].shape == (_NROWS, _NCOLS)

    def test_nan_outside_order_footprint(self, a_file, flat_info, geometry):
        """Pixels outside the order footprint should be NaN after rectification."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = preprocess_science_frames(
                [a_file], flat_info, geometry=geometry, subtraction_mode="A"
            )
        img = result["image"]
        # The geometry has one order spanning rows [16, 48] of a 64-row array.
        # Row 0 is outside all orders → should be NaN.
        assert np.isnan(img[0, :]).all()

    def test_runtime_warning_emitted(self, a_file, flat_info, geometry):
        """RuntimeWarning must be emitted when rectification is applied."""
        with pytest.warns(RuntimeWarning, match="provisional zero-tilt"):
            preprocess_science_frames(
                [a_file], flat_info, geometry=geometry,
                subtraction_mode="A", verbose=True,
            )

    def test_no_warning_without_geometry(self, a_file, flat_info):
        """No RuntimeWarning about tilt when rectification is skipped."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Should not raise
            preprocess_science_frames([a_file], flat_info, subtraction_mode="A")

    def test_ab_mode_with_geometry(self, ab_files, flat_info, geometry):
        """A-B mode with geometry should also rectify correctly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = preprocess_science_frames(
                ab_files, flat_info, geometry=geometry, subtraction_mode="A-B"
            )
        assert result["rectified"] is True
        assert result["image"].shape == (_NROWS, _NCOLS)


# ===========================================================================
# 5. Flat-field edge cases
# ===========================================================================


class TestFlatFieldEdgeCases:
    """Flat-field division edge cases and bad-pixel flagging."""

    def test_flat_zero_pixels_become_nan(self, tmp_path):
        """Science pixels where flat == 0 must become NaN."""
        shape = (_NROWS, _NCOLS)
        flat_img = np.ones(shape, dtype=np.float32)
        flat_img[:, 0] = 0.0  # first column is zero

        fi = FlatInfo(
            mode="K1",
            orders=[233],
            rotation=5,
            plate_scale_arcsec=0.125,
            slit_height_pixels=40.0,
            slit_height_arcsec=5.0,
            slit_range_pixels=(16, 48),
            resolving_power_pixel=23000.0,
            step=5,
            flat_fraction=0.85,
            comm_window=5,
            image=flat_img,
            xranges=np.array([[0, _NCOLS - 1]]),
            edge_coeffs=np.array([[[16.0, 0.0], [48.0, 0.0]]]),
            edge_degree=1,
        )
        fa = _make_synthetic_mef(tmp_path / "a.fits", sig_value=1000.0)
        result = preprocess_science_frames([fa], fi, subtraction_mode="A")
        # First column should be NaN
        assert np.isnan(result["image"][:, 0]).all()
        # Other columns should be finite
        assert np.isfinite(result["image"][:, 1]).all()

    def test_flat_zero_sets_bitmask_bit(self, tmp_path):
        """Flat-zero pixels must set bit 4 in the bitmask."""
        shape = (_NROWS, _NCOLS)
        flat_img = np.ones(shape, dtype=np.float32)
        flat_img[0, 0] = 0.0  # one bad pixel

        fi = FlatInfo(
            mode="K1",
            orders=[233],
            rotation=5,
            plate_scale_arcsec=0.125,
            slit_height_pixels=40.0,
            slit_height_arcsec=5.0,
            slit_range_pixels=(16, 48),
            resolving_power_pixel=23000.0,
            step=5,
            flat_fraction=0.85,
            comm_window=5,
            image=flat_img,
            xranges=np.array([[0, _NCOLS - 1]]),
            edge_coeffs=np.array([[[16.0, 0.0], [48.0, 0.0]]]),
            edge_degree=1,
        )
        fa = _make_synthetic_mef(tmp_path / "a.fits", sig_value=1000.0)
        result = preprocess_science_frames([fa], fi, subtraction_mode="A")
        # Bit 4 should be set at position (0, 0)
        assert result["bitmask"][0, 0] & np.uint8(2 ** 4) != 0
        # Bit 4 should not be set elsewhere
        assert result["bitmask"][1, 1] & np.uint8(2 ** 4) == 0


# ===========================================================================
# 6. Representative J / H / K bands — shape and result key checks
# ===========================================================================


class TestRepresentativeBands:
    """
    Parametric smoke tests across J, H, K representative modes.

    These tests use a minimal but consistent FlatInfo / geometry for each
    mode.  They verify that the function runs without error and that all
    output arrays have the expected shape.
    """

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_a_mode_succeeds(self, tmp_path, mode):
        fi = _make_flat_info(mode=mode, flat_value=1.5)
        fa = _make_synthetic_mef(tmp_path / f"a_{mode}.fits", mode=mode)
        result = preprocess_science_frames([fa], fi, subtraction_mode="A")
        assert isinstance(result, dict)
        assert result["image"].shape == (_NROWS, _NCOLS)

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_ab_mode_succeeds(self, tmp_path, mode):
        fi = _make_flat_info(mode=mode, flat_value=1.5)
        fa = _make_synthetic_mef(
            tmp_path / f"a_{mode}.fits", sig_value=5000.0, mode=mode
        )
        fb = _make_synthetic_mef(
            tmp_path / f"b_{mode}.fits", sig_value=2000.0, mode=mode
        )
        result = preprocess_science_frames([fa, fb], fi, subtraction_mode="A-B")
        assert isinstance(result, dict)
        assert result["image"].shape == (_NROWS, _NCOLS)

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_image_variance_bitmask_same_shape(self, tmp_path, mode):
        fi = _make_flat_info(mode=mode, flat_value=1.0)
        fa = _make_synthetic_mef(tmp_path / f"a_{mode}.fits", mode=mode)
        result = preprocess_science_frames([fa], fi, subtraction_mode="A")
        assert result["image"].shape == result["variance"].shape
        assert result["image"].shape == result["bitmask"].shape

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_with_geometry(self, tmp_path, mode):
        """Rectification smoke test for J, H, K representative modes."""
        fi = _make_flat_info(mode=mode, flat_value=1.0)
        geom = _make_geometry(mode=mode)
        fa = _make_synthetic_mef(tmp_path / f"a_{mode}.fits", mode=mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = preprocess_science_frames(
                [fa], fi, geometry=geom, subtraction_mode="A"
            )
        assert result["rectified"] is True
        assert result["image"].shape == (_NROWS, _NCOLS)

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_hdrinfo_instr_is_ishell(self, tmp_path, mode):
        fi = _make_flat_info(mode=mode, flat_value=1.0)
        fa = _make_synthetic_mef(tmp_path / f"a_{mode}.fits", mode=mode)
        result = preprocess_science_frames([fa], fi, subtraction_mode="A")
        assert result["hdrinfo"]["INSTR"][0] == "iSHELL"


# ===========================================================================
# 7. Failure / malformed-input cases
# ===========================================================================


class TestFailureCases:
    """Error handling for malformed or inconsistent inputs."""

    def test_unsupported_subtraction_mode_raises(self, a_file, flat_info):
        with pytest.raises(pySpextoolError, match="not supported"):
            preprocess_science_frames(
                [a_file], flat_info, subtraction_mode="A-Sky"
            )

    def test_ab_mode_one_file_raises(self, a_file, flat_info):
        with pytest.raises(pySpextoolError, match="2 files"):
            preprocess_science_frames([a_file], flat_info, subtraction_mode="A-B")

    def test_a_mode_two_files_raises(self, ab_files, flat_info):
        with pytest.raises(pySpextoolError, match="1 file"):
            preprocess_science_frames(ab_files, flat_info, subtraction_mode="A")

    def test_ab_mode_zero_files_raises(self, flat_info):
        with pytest.raises(pySpextoolError):
            preprocess_science_frames([], flat_info, subtraction_mode="A-B")

    def test_none_flat_image_raises(self, a_file):
        fi = FlatInfo(
            mode="K1",
            orders=[233],
            rotation=5,
            plate_scale_arcsec=0.125,
            slit_height_pixels=40.0,
            slit_height_arcsec=5.0,
            slit_range_pixels=(16, 48),
            resolving_power_pixel=23000.0,
            step=5,
            flat_fraction=0.85,
            comm_window=5,
            image=None,
            xranges=np.array([[0, 63]]),
            edge_coeffs=np.array([[[16.0, 0.0], [48.0, 0.0]]]),
            edge_degree=1,
        )
        with pytest.raises(ValueError, match="flat_info.image is None"):
            preprocess_science_frames([a_file], fi, subtraction_mode="A")

    def test_all_zero_flat_raises(self, a_file):
        fi = FlatInfo(
            mode="K1",
            orders=[233],
            rotation=5,
            plate_scale_arcsec=0.125,
            slit_height_pixels=40.0,
            slit_height_arcsec=5.0,
            slit_range_pixels=(16, 48),
            resolving_power_pixel=23000.0,
            step=5,
            flat_fraction=0.85,
            comm_window=5,
            image=np.zeros((_NROWS, _NCOLS), dtype=np.float32),
            xranges=np.array([[0, 63]]),
            edge_coeffs=np.array([[[16.0, 0.0], [48.0, 0.0]]]),
            edge_degree=1,
        )
        with pytest.raises(ValueError, match="all-zero or all-NaN"):
            preprocess_science_frames([a_file], fi, subtraction_mode="A")

    def test_geometry_no_wavelength_solution_raises(self, a_file, flat_info):
        """geometry without wavelength solution must raise ValueError."""
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
        assert not geom.has_wavelength_solution()
        with pytest.raises(ValueError, match="wavelength solution"):
            preprocess_science_frames(
                [a_file], flat_info, geometry=geom, subtraction_mode="A"
            )

    def test_geometry_mode_mismatch_raises(self, a_file, flat_info):
        """geometry.mode != flat_info.mode must raise ValueError."""
        geom = _make_geometry(mode="J0")  # different from flat_info.mode="K1"
        with pytest.raises(ValueError, match="does not match"):
            preprocess_science_frames(
                [a_file], flat_info, geometry=geom, subtraction_mode="A"
            )

    def test_nonexistent_file_raises(self, flat_info, tmp_path):
        """Missing input file must raise an error (from load_data)."""
        missing = str(tmp_path / "does_not_exist.fits")
        with pytest.raises(Exception):  # FileNotFoundError or pySpextoolError
            preprocess_science_frames(
                [missing], flat_info, subtraction_mode="A"
            )

    def test_missing_itime_raises(self, tmp_path, flat_info):
        """FITS file missing ITIME must raise pySpextoolError."""
        hdr = _make_ishell_header(mode="K1")
        del hdr["ITIME"]
        shape = (_NROWS, _NCOLS)
        ext0 = fits.PrimaryHDU(
            data=np.zeros(shape, dtype=np.float32), header=hdr
        )
        ext1 = fits.ImageHDU(data=np.zeros(shape, dtype=np.float32))
        ext2 = fits.ImageHDU(data=np.zeros(shape, dtype=np.float32))
        path = str(tmp_path / "no_itime.fits")
        fits.HDUList([ext0, ext1, ext2]).writeto(path, overwrite=True)
        with pytest.raises(pySpextoolError, match="ITIME"):
            preprocess_science_frames([path], flat_info, subtraction_mode="A")

    def test_wrong_extension_count_raises(self, tmp_path, flat_info):
        """FITS file with != 3 extensions must raise pySpextoolError."""
        hdr = _make_ishell_header(mode="K1")
        shape = (_NROWS, _NCOLS)
        ext0 = fits.PrimaryHDU(
            data=np.zeros(shape, dtype=np.float32), header=hdr
        )
        path = str(tmp_path / "one_ext.fits")
        fits.HDUList([ext0]).writeto(path, overwrite=True)
        with pytest.raises(pySpextoolError, match="3 FITS extensions"):
            preprocess_science_frames([path], flat_info, subtraction_mode="A")

    def test_nonlinear_pixels_flagged(self, tmp_path, flat_info):
        """Pixels with pedestal+signal > 30000 must be flagged in bitmask (bit 0)."""
        # ped_value + sig_sum_value = 20000 + 12000 = 32000 > 30000
        fa = _make_synthetic_mef(
            tmp_path / "a.fits",
            sig_value=1000.0,
            ped_value=20000.0,
            sig_sum_value=12000.0,
            mode="K1",
        )
        result = preprocess_science_frames([fa], flat_info, subtraction_mode="A")
        lin_bit = 0
        assert np.all(result["bitmask"] & np.uint8(2 ** lin_bit) > 0)

    def test_linear_pixels_not_flagged(self, tmp_path, flat_info):
        """Pixels with pedestal+signal <= 30000 must NOT be flagged."""
        # ped_value + sig_sum_value = 500 + 800 = 1300 (well below 30000)
        fa = _make_synthetic_mef(
            tmp_path / "a.fits",
            sig_value=1000.0,
            ped_value=500.0,
            sig_sum_value=800.0,
            mode="K1",
        )
        result = preprocess_science_frames([fa], flat_info, subtraction_mode="A")
        lin_bit = 0
        assert np.all(result["bitmask"] & np.uint8(2 ** lin_bit) == 0)


# ===========================================================================
# 8. Linearity_info customization
# ===========================================================================


class TestLinearityInfo:
    """Custom linearity_info dict is respected."""

    def test_custom_max_threshold(self, tmp_path, flat_info):
        """Pixels above a custom threshold must be flagged."""
        # ped + sig = 500 + 800 = 1300; set threshold to 1000 → should flag
        fa = _make_synthetic_mef(
            tmp_path / "a.fits",
            sig_value=1000.0,
            ped_value=500.0,
            sig_sum_value=800.0,
        )
        lin_info = {"max": 1000, "bit": 1}
        result = preprocess_science_frames(
            [fa], flat_info, subtraction_mode="A", linearity_info=lin_info
        )
        # All pixels have ped+sig = 1300 > 1000 → bit 1 should be set
        assert np.all(result["bitmask"] & np.uint8(2 ** 1) > 0)

    def test_default_linearity_info_used_when_none(self, a_file, flat_info):
        """Not providing linearity_info must default to LINCORMAX=30000."""
        # Default run (no linearity_info) should not raise
        result = preprocess_science_frames(
            [a_file], flat_info, subtraction_mode="A"
        )
        assert isinstance(result, dict)
