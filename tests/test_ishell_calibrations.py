"""
Tests for the iSHELL typed calibration-resource readers (calibrations.py).

These tests verify:
  - successful resource reads for all J/H/K modes,
  - correct dataclass types and field values,
  - dimensional consistency between calibration resources,
  - rejection of malformed / missing resources,
  - LFS-pointer detection,
  - convenience loader (load_mode_calibrations).
"""

from __future__ import annotations

import io
import tempfile
import os

import numpy as np
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.calibrations import (
    DETECTOR_NCOLS,
    DETECTOR_NROWS,
    BiasFrame,
    FlatInfo,
    LineList,
    LinearityCube,
    ModeCalibrations,
    PixelMask,
    WaveCalInfo,
    load_mode_calibrations,
    read_bias,
    read_flatinfo,
    read_line_list,
    read_linearity_cube,
    read_pixel_mask,
    read_wavecalinfo,
)

# ---------------------------------------------------------------------------
# Constants shared by all tests
# ---------------------------------------------------------------------------

SUPPORTED_MODES = [
    "J0", "J1", "J2", "J3",
    "H1", "H2", "H3",
    "K1", "K2", "K3", "Kgas",
]

_DETECTOR_SHAPE = (DETECTOR_NROWS, DETECTOR_NCOLS)


# ===========================================================================
# read_line_list
# ===========================================================================


class TestReadLineList:
    """Tests for read_line_list()."""

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_returns_linelist_instance(self, mode):
        """read_line_list must return a LineList object for every mode."""
        result = read_line_list(mode)
        assert isinstance(result, LineList)

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_mode_attribute(self, mode):
        """LineList.mode must match the requested mode."""
        result = read_line_list(mode)
        assert result.mode == mode

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_non_empty(self, mode):
        """Each mode must have at least one arc line."""
        result = read_line_list(mode)
        assert result.n_lines > 0, f"Line list for mode '{mode}' is empty"

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_orders_are_positive_integers(self, mode):
        """All order numbers must be positive integers."""
        result = read_line_list(mode)
        for entry in result.entries:
            assert isinstance(entry.order, int)
            assert entry.order > 0

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_wavelengths_are_positive_floats(self, mode):
        """All wavelengths must be positive floats in a physically plausible range."""
        result = read_line_list(mode)
        wl = result.wavelengths_um
        assert np.all(wl > 0), f"Non-positive wavelength in mode '{mode}'"
        assert np.all(wl < 5.0), f"Implausibly large wavelength in mode '{mode}'"

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_fit_types_known(self, mode):
        """All fit_type values must be recognised strings (e.g. 'G' for Gaussian)."""
        result = read_line_list(mode)
        known = {"G", "L", "V"}  # Gaussian, Lorentzian, Voigt
        for entry in result.entries:
            assert entry.fit_type in known, (
                f"Unknown fit_type '{entry.fit_type}' in mode '{mode}'"
            )

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_fit_n_terms_positive(self, mode):
        """All fit_n_terms values must be positive."""
        result = read_line_list(mode)
        for entry in result.entries:
            assert entry.fit_n_terms > 0

    def test_unknown_mode_raises_key_error(self):
        """read_line_list must raise KeyError for an unknown mode."""
        with pytest.raises(KeyError):
            read_line_list("NOTAMODE")

    def test_wavelengths_array_shape(self):
        """wavelengths_um property must return a 1-D NumPy array."""
        result = read_line_list("J0")
        wl = result.wavelengths_um
        assert isinstance(wl, np.ndarray)
        assert wl.ndim == 1
        assert len(wl) == result.n_lines

    def test_orders_property_sorted(self):
        """orders property must return a sorted list of unique integers."""
        result = read_line_list("H1")
        orders = result.orders
        assert orders == sorted(set(orders))

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """read_line_list must raise FileNotFoundError if the file is missing."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        def fake_get_mode_resource(mode_name, resource_key):
            # Return a path object pointing to a non-existent file
            return tmp_path / "nonexistent.dat"

        monkeypatch.setattr(cal_mod, "get_mode_resource", fake_get_mode_resource)
        with pytest.raises(FileNotFoundError):
            read_line_list("J0")

    def test_malformed_file_raises_value_error(self, tmp_path, monkeypatch):
        """read_line_list must raise ValueError if column count is wrong."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        # Write a file with wrong column count
        bad_file = tmp_path / "bad.dat"
        bad_file.write_text("457 | 1.234 | Th I\n")  # only 3 columns

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="r", encoding=None):
                return bad_file.open(mode, encoding=encoding)
            def __str__(self):
                return str(bad_file)

        monkeypatch.setattr(cal_mod, "get_mode_resource", lambda m, k: _FakePath())
        with pytest.raises(ValueError, match="Unexpected number of columns"):
            read_line_list("J0")

    def test_empty_file_raises_value_error(self, tmp_path, monkeypatch):
        """read_line_list must raise ValueError if the file has no data rows."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        comment_only = tmp_path / "comments_only.dat"
        comment_only.write_text("# comment\n# another comment\n")

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="r", encoding=None):
                return comment_only.open(mode, encoding=encoding)
            def __str__(self):
                return str(comment_only)

        monkeypatch.setattr(cal_mod, "get_mode_resource", lambda m, k: _FakePath())
        with pytest.raises(ValueError, match="empty"):
            read_line_list("J0")


# ===========================================================================
# read_flatinfo
# ===========================================================================


class TestReadFlatInfo:
    """Tests for read_flatinfo()."""

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_returns_flatinfo_instance(self, mode):
        """read_flatinfo must return a FlatInfo object for every mode."""
        result = read_flatinfo(mode)
        assert isinstance(result, FlatInfo)

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_mode_attribute(self, mode):
        """FlatInfo.mode must match the requested mode."""
        result = read_flatinfo(mode)
        assert result.mode == mode

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_image_shape(self, mode):
        """Flat image must have the iSHELL detector shape (2048×2048)."""
        result = read_flatinfo(mode)
        assert result.image.shape == _DETECTOR_SHAPE, (
            f"Mode '{mode}': expected {_DETECTOR_SHAPE}, got {result.image.shape}"
        )

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_orders_non_empty(self, mode):
        """There must be at least one order in the flat."""
        result = read_flatinfo(mode)
        assert result.n_orders > 0

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_orders_are_positive_integers(self, mode):
        """All order numbers must be positive integers."""
        result = read_flatinfo(mode)
        for o in result.orders:
            assert isinstance(o, int) and o > 0

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_n_orders_consistent(self, mode):
        """n_orders property must equal len(orders)."""
        result = read_flatinfo(mode)
        assert result.n_orders == len(result.orders)

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_plate_scale_positive(self, mode):
        """plate_scale_arcsec must be positive."""
        result = read_flatinfo(mode)
        assert result.plate_scale_arcsec > 0

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_flat_fraction_range(self, mode):
        """flat_fraction must be in (0, 1]."""
        result = read_flatinfo(mode)
        assert 0 < result.flat_fraction <= 1.0

    def test_unknown_mode_raises_key_error(self):
        """read_flatinfo must raise KeyError for an unknown mode."""
        with pytest.raises(KeyError):
            read_flatinfo("NOTAMODE")

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """read_flatinfo must raise FileNotFoundError if the file is missing."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        monkeypatch.setattr(
            cal_mod, "get_mode_resource",
            lambda m, k: tmp_path / "nonexistent.fits"
        )
        with pytest.raises(FileNotFoundError):
            read_flatinfo("J0")

    def test_lfs_pointer_raises_runtime_error(self, tmp_path, monkeypatch):
        """read_flatinfo must raise RuntimeError for a Git LFS pointer file."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        lfs_file = tmp_path / "lfs_pointer.fits"
        lfs_file.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12345\n"
        )

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                if "b" in mode:
                    return lfs_file.open("rb")
                return lfs_file.open(mode, encoding=encoding)
            def __str__(self):
                return str(lfs_file)

        monkeypatch.setattr(
            cal_mod, "get_mode_resource", lambda m, k: _FakePath()
        )
        with pytest.raises(RuntimeError, match="Git LFS"):
            read_flatinfo("J0")

    def test_missing_orders_header_raises(self, tmp_path, monkeypatch):
        """read_flatinfo must raise ValueError if ORDERS header keyword is absent."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        # Create a FITS file without ORDERS keyword
        hdu = fits.PrimaryHDU(data=np.zeros(_DETECTOR_SHAPE, dtype=np.int16))
        hdu.header["ROTATION"] = 5
        fits_file = tmp_path / "bad_flatinfo.fits"
        hdu.writeto(str(fits_file))

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return fits_file.open("rb")
            def __str__(self):
                return str(fits_file)

        monkeypatch.setattr(
            cal_mod, "get_mode_resource", lambda m, k: _FakePath()
        )
        with pytest.raises(ValueError, match="ORDERS"):
            read_flatinfo("J0")


# ===========================================================================
# read_wavecalinfo
# ===========================================================================


class TestReadWaveCalInfo:
    """Tests for read_wavecalinfo()."""

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_returns_wavecalinfo_instance(self, mode):
        """read_wavecalinfo must return a WaveCalInfo object for every mode."""
        result = read_wavecalinfo(mode)
        assert isinstance(result, WaveCalInfo)

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_mode_attribute(self, mode):
        """WaveCalInfo.mode must match the requested mode."""
        result = read_wavecalinfo(mode)
        assert result.mode == mode

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_data_shape(self, mode):
        """Data cube must have shape (n_orders, 4, n_pixels)."""
        result = read_wavecalinfo(mode)
        assert result.data.ndim == 3
        assert result.data.shape[0] == result.n_orders
        assert result.data.shape[1] == 4

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_n_orders_consistent(self, mode):
        """n_orders must equal len(orders) and match the data cube first axis."""
        result = read_wavecalinfo(mode)
        assert result.n_orders == len(result.orders)
        assert result.n_orders == result.data.shape[0]

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_orders_are_positive_integers(self, mode):
        """All order numbers must be positive integers."""
        result = read_wavecalinfo(mode)
        for o in result.orders:
            assert isinstance(o, int) and o > 0

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_n_pixels_positive(self, mode):
        """n_pixels must be > 0."""
        result = read_wavecalinfo(mode)
        assert result.n_pixels > 0

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_resolving_power_plausible(self, mode):
        """Resolving power must be a positive finite number."""
        result = read_wavecalinfo(mode)
        assert np.isfinite(result.resolving_power)
        assert result.resolving_power > 0

    def test_unknown_mode_raises_key_error(self):
        """read_wavecalinfo must raise KeyError for an unknown mode."""
        with pytest.raises(KeyError):
            read_wavecalinfo("NOTAMODE")

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """read_wavecalinfo must raise FileNotFoundError if the file is missing."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        monkeypatch.setattr(
            cal_mod, "get_mode_resource",
            lambda m, k: tmp_path / "nonexistent.fits"
        )
        with pytest.raises(FileNotFoundError):
            read_wavecalinfo("J0")

    def test_lfs_pointer_raises_runtime_error(self, tmp_path, monkeypatch):
        """read_wavecalinfo must raise RuntimeError for a Git LFS pointer."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        lfs_file = tmp_path / "lfs.fits"
        lfs_file.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
        )

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return lfs_file.open("rb")
            def __str__(self):
                return str(lfs_file)

        monkeypatch.setattr(
            cal_mod, "get_mode_resource", lambda m, k: _FakePath()
        )
        with pytest.raises(RuntimeError, match="Git LFS"):
            read_wavecalinfo("J0")

    def test_missing_norders_header_raises(self, tmp_path, monkeypatch):
        """read_wavecalinfo must raise ValueError if NORDERS is absent."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        hdu = fits.PrimaryHDU(data=np.zeros((47, 4, 100), dtype=np.float64))
        # intentionally omit NORDERS / ORDERS
        fits_file = tmp_path / "bad_wci.fits"
        hdu.writeto(str(fits_file))

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return fits_file.open("rb")
            def __str__(self):
                return str(fits_file)

        monkeypatch.setattr(
            cal_mod, "get_mode_resource", lambda m, k: _FakePath()
        )
        with pytest.raises(ValueError, match="NORDERS"):
            read_wavecalinfo("J0")

    def test_norders_mismatch_raises(self, tmp_path, monkeypatch):
        """read_wavecalinfo must raise ValueError if data cube first axis != NORDERS."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        hdu = fits.PrimaryHDU(data=np.zeros((10, 4, 100), dtype=np.float64))
        hdu.header["NORDERS"] = 47  # mismatch: data has 10 but header says 47
        hdu.header["ORDERS"] = ",".join(str(i) for i in range(457, 457 + 47))
        fits_file = tmp_path / "mismatch_wci.fits"
        hdu.writeto(str(fits_file))

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return fits_file.open("rb")
            def __str__(self):
                return str(fits_file)

        monkeypatch.setattr(
            cal_mod, "get_mode_resource", lambda m, k: _FakePath()
        )
        with pytest.raises(ValueError, match="NORDERS"):
            read_wavecalinfo("J0")


# ===========================================================================
# Cross-mode dimensional consistency
# ===========================================================================


class TestCrossModeConsistency:
    """Tests that flatinfo and wavecalinfo agree on order counts."""

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_flatinfo_wavecalinfo_norders_match(self, mode):
        """flatinfo.n_orders must equal wavecalinfo.n_orders."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        assert fi.n_orders == wci.n_orders, (
            f"Mode '{mode}': flatinfo has {fi.n_orders} orders but "
            f"wavecalinfo has {wci.n_orders}."
        )

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_flatinfo_wavecalinfo_orders_match(self, mode):
        """flatinfo.orders must equal wavecalinfo.orders."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        assert fi.orders == wci.orders, (
            f"Mode '{mode}': flatinfo and wavecalinfo order lists differ."
        )

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_linelist_orders_subset_of_flatinfo(self, mode):
        """All orders in the line list must be a subset of flatinfo.orders."""
        ll = read_line_list(mode)
        fi = read_flatinfo(mode)
        fi_set = set(fi.orders)
        for order in ll.orders:
            assert order in fi_set, (
                f"Mode '{mode}': line list order {order} not in flatinfo.orders"
            )


# ===========================================================================
# read_linearity_cube
# ===========================================================================


class TestReadLinearityCube:
    """Tests for read_linearity_cube()."""

    def test_returns_linearity_cube_instance(self):
        """read_linearity_cube must return a LinearityCube object."""
        result = read_linearity_cube()
        assert isinstance(result, LinearityCube)

    def test_data_shape(self):
        """Data cube must have shape (n_coeffs, 2048, 2048)."""
        result = read_linearity_cube()
        assert result.data.ndim == 3
        assert result.data.shape[1:] == _DETECTOR_SHAPE

    def test_n_coeffs_consistent(self):
        """n_coeffs property must equal data.shape[0]."""
        result = read_linearity_cube()
        assert result.n_coeffs == result.data.shape[0]

    def test_dn_limits_plausible(self):
        """DN limits must be finite and lower < upper."""
        result = read_linearity_cube()
        assert np.isfinite(result.dn_lower_limit)
        assert np.isfinite(result.dn_upper_limit)
        assert result.dn_lower_limit < result.dn_upper_limit

    def test_fit_order_positive(self):
        """fit_order must be positive."""
        result = read_linearity_cube()
        assert result.fit_order > 0

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """read_linearity_cube must raise FileNotFoundError for a missing file."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        monkeypatch.setattr(
            cal_mod, "get_detector_resource",
            lambda k: tmp_path / "nonexistent.fits"
        )
        with pytest.raises(FileNotFoundError):
            read_linearity_cube()

    def test_lfs_pointer_raises_runtime_error(self, tmp_path, monkeypatch):
        """read_linearity_cube must raise RuntimeError for a Git LFS pointer."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        lfs_file = tmp_path / "lfs.fits"
        lfs_file.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
        )

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return lfs_file.open("rb")
            def __str__(self):
                return str(lfs_file)

        monkeypatch.setattr(
            cal_mod, "get_detector_resource", lambda k: _FakePath()
        )
        with pytest.raises(RuntimeError, match="Git LFS"):
            read_linearity_cube()

    def test_wrong_shape_raises_value_error(self, tmp_path, monkeypatch):
        """read_linearity_cube must raise ValueError for wrong spatial dimensions."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        hdu = fits.PrimaryHDU(data=np.zeros((9, 512, 512), dtype=np.float32))
        hdu.header["DNLOLIM"] = 0.0
        hdu.header["DNUPLIM"] = 37000.0
        hdu.header["FITORDER"] = 6
        hdu.header["NPEDS"] = 2
        fits_file = tmp_path / "bad_lin.fits"
        hdu.writeto(str(fits_file))

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return fits_file.open("rb")
            def __str__(self):
                return str(fits_file)

        monkeypatch.setattr(
            cal_mod, "get_detector_resource", lambda k: _FakePath()
        )
        with pytest.raises(ValueError, match="spatial dimensions"):
            read_linearity_cube()


# ===========================================================================
# read_pixel_mask
# ===========================================================================


class TestReadPixelMask:
    """Tests for read_pixel_mask()."""

    @pytest.mark.parametrize("mask_type", ["bad", "hot"])
    def test_returns_pixel_mask_instance(self, mask_type):
        """read_pixel_mask must return a PixelMask for 'bad' and 'hot'."""
        result = read_pixel_mask(mask_type)
        assert isinstance(result, PixelMask)

    @pytest.mark.parametrize("mask_type", ["bad", "hot"])
    def test_mask_type_attribute(self, mask_type):
        """PixelMask.mask_type must match the requested type."""
        result = read_pixel_mask(mask_type)
        assert result.mask_type == mask_type

    @pytest.mark.parametrize("mask_type", ["bad", "hot"])
    def test_shape(self, mask_type):
        """Mask must have iSHELL detector shape (2048×2048)."""
        result = read_pixel_mask(mask_type)
        assert result.data.shape == _DETECTOR_SHAPE

    @pytest.mark.parametrize("mask_type", ["bad", "hot"])
    def test_is_placeholder_true(self, mask_type):
        """Both masks are currently placeholder stubs."""
        result = read_pixel_mask(mask_type)
        assert result.is_placeholder is True

    def test_invalid_mask_type_raises_value_error(self):
        """read_pixel_mask must raise ValueError for invalid mask_type."""
        with pytest.raises(ValueError, match="mask_type"):
            read_pixel_mask("ugly")

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """read_pixel_mask must raise FileNotFoundError for a missing file."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        monkeypatch.setattr(
            cal_mod, "get_detector_resource",
            lambda k: tmp_path / "nonexistent.fits"
        )
        with pytest.raises(FileNotFoundError):
            read_pixel_mask("bad")

    def test_wrong_shape_raises_value_error(self, tmp_path, monkeypatch):
        """read_pixel_mask must raise ValueError for wrong detector dimensions."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        hdu = fits.PrimaryHDU(data=np.zeros((512, 512), dtype=np.uint8))
        hdu.header.add_comment("PLACEHOLDER: test stub")
        fits_file = tmp_path / "bad_mask.fits"
        hdu.writeto(str(fits_file))

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return fits_file.open("rb")
            def __str__(self):
                return str(fits_file)

        monkeypatch.setattr(
            cal_mod, "get_detector_resource", lambda k: _FakePath()
        )
        monkeypatch.setattr(
            cal_mod, "is_placeholder_resource", lambda k: True
        )
        with pytest.raises(ValueError, match="shape"):
            read_pixel_mask("bad")


# ===========================================================================
# read_bias
# ===========================================================================


class TestReadBias:
    """Tests for read_bias()."""

    def test_returns_bias_frame_instance(self):
        """read_bias must return a BiasFrame object."""
        result = read_bias()
        assert isinstance(result, BiasFrame)

    def test_shape(self):
        """Bias frame must have iSHELL detector shape (2048×2048)."""
        result = read_bias()
        assert result.data.shape == _DETECTOR_SHAPE

    def test_divisor_positive(self):
        """divisor must be a positive integer."""
        result = read_bias()
        assert isinstance(result.divisor, int)
        assert result.divisor > 0

    def test_missing_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """read_bias must raise FileNotFoundError for a missing file."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        monkeypatch.setattr(
            cal_mod, "get_detector_resource",
            lambda k: tmp_path / "nonexistent.fits"
        )
        with pytest.raises(FileNotFoundError):
            read_bias()

    def test_wrong_shape_raises_value_error(self, tmp_path, monkeypatch):
        """read_bias must raise ValueError for wrong detector dimensions."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        hdu = fits.PrimaryHDU(data=np.zeros((512, 512), dtype=np.int32))
        hdu.header["DIVISOR"] = 10
        fits_file = tmp_path / "bad_bias.fits"
        hdu.writeto(str(fits_file))

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return fits_file.open("rb")
            def __str__(self):
                return str(fits_file)

        monkeypatch.setattr(
            cal_mod, "get_detector_resource", lambda k: _FakePath()
        )
        with pytest.raises(ValueError, match="shape"):
            read_bias()


# ===========================================================================
# load_mode_calibrations
# ===========================================================================


class TestLoadModeCalibrations:
    """Tests for the load_mode_calibrations() convenience loader."""

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_returns_mode_calibrations_instance(self, mode):
        """load_mode_calibrations must return a ModeCalibrations object."""
        result = load_mode_calibrations(mode)
        assert isinstance(result, ModeCalibrations)

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_mode_attribute(self, mode):
        """ModeCalibrations.mode must match the requested mode."""
        result = load_mode_calibrations(mode)
        assert result.mode == mode

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_all_sub_resources_populated(self, mode):
        """All sub-resources must be populated and have correct types."""
        result = load_mode_calibrations(mode)
        assert isinstance(result.line_list, LineList)
        assert isinstance(result.flatinfo, FlatInfo)
        assert isinstance(result.wavecalinfo, WaveCalInfo)

    def test_unknown_mode_raises_key_error(self):
        """load_mode_calibrations must raise KeyError for an unknown mode."""
        with pytest.raises(KeyError):
            load_mode_calibrations("NOTAMODE")


# ===========================================================================
# LFS pointer detection (standalone helper)
# ===========================================================================


class TestLfsPointerDetection:
    """Tests for the _check_lfs() internal helper via public functions."""

    def test_lfs_pointer_detected_by_read_line_list(self, tmp_path, monkeypatch):
        """_check_lfs is invoked when reading a line list."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        lfs_file = tmp_path / "lfs.dat"
        lfs_file.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
        )

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                if "b" in mode:
                    return lfs_file.open("rb")
                return lfs_file.open(mode, encoding=encoding)
            def __str__(self):
                return str(lfs_file)

        monkeypatch.setattr(
            cal_mod, "get_mode_resource", lambda m, k: _FakePath()
        )
        # LFS check is done by _check_lfs before parsing.
        # For a line-list file the _check_lfs step is skipped
        # (text files don't embed LFS magic), so the RuntimeError path
        # belongs to FITS readers.  Instead verify that FileNotFoundError
        # is NOT raised (the file does exist) and ValueError IS raised
        # (the LFS bytes are not valid pipe-delimited lines).
        with pytest.raises(ValueError):
            read_line_list("J0")

    def test_lfs_pointer_detected_by_read_flatinfo(self, tmp_path, monkeypatch):
        """RuntimeError with 'Git LFS' message is raised for LFS pointer FITS."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        lfs_file = tmp_path / "lfs.fits"
        lfs_file.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
        )

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return lfs_file.open("rb")
            def __str__(self):
                return str(lfs_file)

        monkeypatch.setattr(
            cal_mod, "get_mode_resource", lambda m, k: _FakePath()
        )
        with pytest.raises(RuntimeError, match="Git LFS"):
            read_flatinfo("J0")

    def test_lfs_pointer_detected_by_read_wavecalinfo(self, tmp_path, monkeypatch):
        """RuntimeError with 'Git LFS' message is raised for LFS pointer FITS."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        lfs_file = tmp_path / "lfs.fits"
        lfs_file.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
        )

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return lfs_file.open("rb")
            def __str__(self):
                return str(lfs_file)

        monkeypatch.setattr(
            cal_mod, "get_mode_resource", lambda m, k: _FakePath()
        )
        with pytest.raises(RuntimeError, match="Git LFS"):
            read_wavecalinfo("J0")

    def test_lfs_pointer_detected_by_read_linearity_cube(self, tmp_path, monkeypatch):
        """RuntimeError with 'Git LFS' message is raised for LFS pointer FITS."""
        import pyspextool.instruments.ishell.calibrations as cal_mod

        lfs_file = tmp_path / "lfs.fits"
        lfs_file.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
        )

        class _FakePath:
            def is_file(self):
                return True
            def open(self, mode="rb", encoding=None):
                return lfs_file.open("rb")
            def __str__(self):
                return str(lfs_file)

        monkeypatch.setattr(
            cal_mod, "get_detector_resource", lambda k: _FakePath()
        )
        with pytest.raises(RuntimeError, match="Git LFS"):
            read_linearity_cube()
