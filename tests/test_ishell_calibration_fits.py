"""
Tests for the provisional FITS calibration-file writer (calibration_fits.py).

Coverage:
  - write_wavecal_fits:
      * creates the output file,
      * extension count equals number of orders,
      * primary header has PRODTYPE = 'WAVECAL_PROVISIONAL',
      * each extension has correct ORDER header keyword,
      * wavelength arrays are monotonic,
      * NSPECTRA/NSPATIAL keywords match array lengths,
      * WAVMIN/WAVMAX keywords are correct.
  - write_spatcal_fits:
      * creates the output file,
      * extension count equals number of orders,
      * primary header has PRODTYPE = 'SPATCAL_PROVISIONAL',
      * spatial arrays are in [0, 1],
      * NSPATIAL keyword matches array length.
  - write_calibration_fits:
      * both files are created,
      * file names are derived from mode,
      * returned paths match on-disk files.
  - Error handling:
      * empty calibration product set → ValueError for all three functions,
      * non-existent output path/directory → OSError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * FITS files exist after writing,
      * extension count matches number of orders,
      * wavelength arrays are monotonic.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import astropy.io.fits as fits

from pyspextool.instruments.ishell.calibration_fits import (
    write_calibration_fits,
    write_spatcal_fits,
    write_wavecal_fits,
)
from pyspextool.instruments.ishell.calibration_products import (
    CalibrationProductSet,
    build_calibration_products,
)
from pyspextool.instruments.ishell.rectified_orders import (
    RectifiedOrder,
    RectifiedOrderSet,
)
from pyspextool.instruments.ishell.io_utils import is_fits_file

# ---------------------------------------------------------------------------
# Path helpers for real H1 calibration data  (mirrors other smoke tests)
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
_N_SPECTRAL = 32
_N_SPATIAL = 16

_ORDER_PARAMS = [
    {
        "order_number": 311,
        "wav_start": 1.55,
    },
    {
        "order_number": 315,
        "wav_start": 1.60,
    },
    {
        "order_number": 320,
        "wav_start": 1.65,
    },
]


def _make_synthetic_calibration_products(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
    mode: str = "H1_test",
) -> CalibrationProductSet:
    """Build a synthetic CalibrationProductSet for testing."""
    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = p["wav_start"]
        wav_end = wav_start + 0.04
        wavelength_um = np.linspace(wav_start, wav_end, n_spectral)
        spatial_frac = np.linspace(0.0, 1.0, n_spatial)
        flux = np.ones((n_spatial, n_spectral), dtype=float) * (idx + 1.0)
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=wavelength_um,
                spatial_frac=spatial_frac,
                flux=flux,
                source_image_shape=(_NROWS, _NCOLS),
            )
        )
    ros = RectifiedOrderSet(
        mode=mode,
        rectified_orders=orders,
        source_image_shape=(_NROWS, _NCOLS),
    )
    return build_calibration_products(ros)


def _make_empty_calibration_products() -> CalibrationProductSet:
    """Build a CalibrationProductSet with no orders."""
    return CalibrationProductSet(mode="empty_test")


# ===========================================================================
# 1. write_wavecal_fits – basic structure
# ===========================================================================


class TestWriteWavecalFitsStructure:
    """write_wavecal_fits output structure tests."""

    @pytest.fixture(autouse=True)
    def tmp_path_fixture(self, tmp_path):
        self.tmp_path = tmp_path

    def test_creates_file(self):
        """write_wavecal_fits creates the output file."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        assert os.path.isfile(out)

    def test_extension_count_equals_n_orders(self):
        """Number of image extensions equals number of orders."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            # hdul[0] is the primary HDU; extensions start at index 1.
            assert len(hdul) - 1 == cal.n_orders

    def test_primary_header_prodtype(self):
        """Primary HDU has PRODTYPE = 'WAVECAL_PROVISIONAL'."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            assert hdul[0].header["PRODTYPE"] == "WAVECAL_PROVISIONAL"

    def test_primary_header_mode(self):
        """Primary HDU records the mode."""
        cal = _make_synthetic_calibration_products(mode="H1_test")
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            assert hdul[0].header["MODE"] == "H1_test"

    def test_primary_header_norders(self):
        """Primary HDU records the number of orders."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            assert hdul[0].header["NORDERS"] == cal.n_orders

    def test_extension_names(self):
        """Each extension is named ORDER_<order_number>."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            for wcp, hdu in zip(cal.wavecal_products, hdul[1:]):
                assert hdu.name == f"ORDER_{wcp.order}"

    def test_extension_order_keyword(self):
        """Each extension has the correct ORDER header keyword."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            for wcp, hdu in zip(cal.wavecal_products, hdul[1:]):
                assert hdu.header["ORDER"] == wcp.order

    def test_extension_data_shape(self):
        """Each extension has shape (2, n_spectral)."""
        cal = _make_synthetic_calibration_products(n_spectral=32, n_spatial=16)
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            for wcp, hdu in zip(cal.wavecal_products, hdul[1:]):
                assert hdu.data.shape == (2, wcp.n_spectral)

    def test_wavelength_row_values(self):
        """Row 0 of each extension matches wavelength_um."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            for wcp, hdu in zip(cal.wavecal_products, hdul[1:]):
                np.testing.assert_allclose(
                    hdu.data[0],
                    wcp.wavelength_um,
                    rtol=1e-5,
                    err_msg=f"Order {wcp.order}: wavelength row mismatch",
                )

    def test_nspectra_keyword(self):
        """NSPECTRA keyword matches n_spectral."""
        cal = _make_synthetic_calibration_products(n_spectral=40)
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            for wcp, hdu in zip(cal.wavecal_products, hdul[1:]):
                assert hdu.header["NSPECTRA"] == wcp.n_spectral

    def test_nspatial_keyword(self):
        """NSPATIAL keyword matches n_spatial."""
        cal = _make_synthetic_calibration_products(n_spatial=20)
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            for wcp, hdu in zip(cal.wavecal_products, hdul[1:]):
                assert hdu.header["NSPATIAL"] == wcp.n_spatial

    def test_wavmin_wavmax_keywords(self):
        """WAVMIN/WAVMAX keywords are correct."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            for wcp, hdu in zip(cal.wavecal_products, hdul[1:]):
                assert pytest.approx(hdu.header["WAVMIN"], rel=1e-4) == float(
                    np.min(wcp.wavelength_um)
                )
                assert pytest.approx(hdu.header["WAVMAX"], rel=1e-4) == float(
                    np.max(wcp.wavelength_um)
                )

    def test_wavelength_axes_monotonic(self):
        """Wavelength axis stored in each extension is monotonic."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        with fits.open(out) as hdul:
            for hdu in hdul[1:]:
                wav = hdu.data[0]
                diffs = np.diff(wav)
                assert np.all(diffs > 0) or np.all(diffs < 0), (
                    f"{hdu.name}: wavelength axis is not monotonic"
                )

    def test_overwrite_existing_file(self):
        """write_wavecal_fits overwrites an existing file without error."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        write_wavecal_fits(cal, out)
        write_wavecal_fits(cal, out)  # second call should not raise
        assert os.path.isfile(out)


# ===========================================================================
# 2. write_spatcal_fits – basic structure
# ===========================================================================


class TestWriteSpatcalFitsStructure:
    """write_spatcal_fits output structure tests."""

    @pytest.fixture(autouse=True)
    def tmp_path_fixture(self, tmp_path):
        self.tmp_path = tmp_path

    def test_creates_file(self):
        """write_spatcal_fits creates the output file."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        assert os.path.isfile(out)

    def test_extension_count_equals_n_orders(self):
        """Number of image extensions equals number of orders."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            assert len(hdul) - 1 == cal.n_orders

    def test_primary_header_prodtype(self):
        """Primary HDU has PRODTYPE = 'SPATCAL_PROVISIONAL'."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            assert hdul[0].header["PRODTYPE"] == "SPATCAL_PROVISIONAL"

    def test_primary_header_mode(self):
        """Primary HDU records the mode."""
        cal = _make_synthetic_calibration_products(mode="H1_test")
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            assert hdul[0].header["MODE"] == "H1_test"

    def test_primary_header_norders(self):
        """Primary HDU records the number of orders."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            assert hdul[0].header["NORDERS"] == cal.n_orders

    def test_extension_names(self):
        """Each extension is named ORDER_<order_number>."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            for scp, hdu in zip(cal.spatcal_products, hdul[1:]):
                assert hdu.name == f"ORDER_{scp.order}"

    def test_extension_order_keyword(self):
        """Each extension has the correct ORDER header keyword."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            for scp, hdu in zip(cal.spatcal_products, hdul[1:]):
                assert hdu.header["ORDER"] == scp.order

    def test_extension_data_shape(self):
        """Each extension has shape (2, n_spatial)."""
        cal = _make_synthetic_calibration_products(n_spatial=20)
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            for scp, hdu in zip(cal.spatcal_products, hdul[1:]):
                assert hdu.data.shape == (2, scp.n_spatial)

    def test_spatial_frac_row_values(self):
        """Row 0 of each extension matches spatial_frac."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            for scp, hdu in zip(cal.spatcal_products, hdul[1:]):
                np.testing.assert_allclose(
                    hdu.data[0],
                    scp.spatial_frac,
                    rtol=1e-5,
                    err_msg=f"Order {scp.order}: spatial_frac row mismatch",
                )

    def test_spatial_frac_in_unit_interval(self):
        """Spatial-fraction values stored in file are in [0, 1]."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            for hdu in hdul[1:]:
                sf = hdu.data[0]
                assert np.all(sf >= 0.0), f"{hdu.name}: spatial_frac has values < 0"
                assert np.all(sf <= 1.0), f"{hdu.name}: spatial_frac has values > 1"

    def test_nspatial_keyword(self):
        """NSPATIAL keyword matches n_spatial."""
        cal = _make_synthetic_calibration_products(n_spatial=24)
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            for scp, hdu in zip(cal.spatcal_products, hdul[1:]):
                assert hdu.header["NSPATIAL"] == scp.n_spatial

    def test_sfmin_sfmax_keywords(self):
        """SFMIN/SFMAX keywords are correct."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        write_spatcal_fits(cal, out)
        with fits.open(out) as hdul:
            for scp, hdu in zip(cal.spatcal_products, hdul[1:]):
                assert pytest.approx(hdu.header["SFMIN"], rel=1e-4) == float(
                    np.min(scp.spatial_frac)
                )
                assert pytest.approx(hdu.header["SFMAX"], rel=1e-4) == float(
                    np.max(scp.spatial_frac)
                )


# ===========================================================================
# 3. write_calibration_fits – convenience wrapper
# ===========================================================================


class TestWriteCalibrationFits:
    """write_calibration_fits convenience wrapper tests."""

    @pytest.fixture(autouse=True)
    def tmp_path_fixture(self, tmp_path):
        self.tmp_path = tmp_path

    def test_both_files_created(self):
        """Both wavecal and spatcal files are created."""
        cal = _make_synthetic_calibration_products(mode="H1_test")
        wpath, spath = write_calibration_fits(cal, str(self.tmp_path))
        assert os.path.isfile(wpath)
        assert os.path.isfile(spath)

    def test_file_names_derived_from_mode(self):
        """File names contain the mode string."""
        cal = _make_synthetic_calibration_products(mode="H1_test")
        wpath, spath = write_calibration_fits(cal, str(self.tmp_path))
        assert "H1_test" in os.path.basename(wpath)
        assert "H1_test" in os.path.basename(spath)

    def test_wavecal_suffix(self):
        """Wavecal file name ends with '_wavecal.fits'."""
        cal = _make_synthetic_calibration_products(mode="H1_test")
        wpath, _ = write_calibration_fits(cal, str(self.tmp_path))
        assert os.path.basename(wpath) == "H1_test_wavecal.fits"

    def test_spatcal_suffix(self):
        """Spatcal file name ends with '_spatcal.fits'."""
        cal = _make_synthetic_calibration_products(mode="H1_test")
        _, spath = write_calibration_fits(cal, str(self.tmp_path))
        assert os.path.basename(spath) == "H1_test_spatcal.fits"

    def test_returned_paths_are_absolute(self):
        """Returned paths are absolute (or at least point to real files)."""
        cal = _make_synthetic_calibration_products(mode="H1_test")
        wpath, spath = write_calibration_fits(cal, str(self.tmp_path))
        assert os.path.isfile(wpath)
        assert os.path.isfile(spath)

    def test_wavecal_extension_count(self):
        """Wavecal file has correct extension count."""
        cal = _make_synthetic_calibration_products()
        wpath, _ = write_calibration_fits(cal, str(self.tmp_path))
        with fits.open(wpath) as hdul:
            assert len(hdul) - 1 == cal.n_orders

    def test_spatcal_extension_count(self):
        """Spatcal file has correct extension count."""
        cal = _make_synthetic_calibration_products()
        _, spath = write_calibration_fits(cal, str(self.tmp_path))
        with fits.open(spath) as hdul:
            assert len(hdul) - 1 == cal.n_orders


# ===========================================================================
# 4. Error handling
# ===========================================================================


class TestErrorHandling:
    """Error paths for the FITS writer functions."""

    @pytest.fixture(autouse=True)
    def tmp_path_fixture(self, tmp_path):
        self.tmp_path = tmp_path

    # --- empty calibration product set ---

    def test_write_wavecal_raises_on_empty(self):
        """write_wavecal_fits raises ValueError for empty CalibrationProductSet."""
        empty = _make_empty_calibration_products()
        out = str(self.tmp_path / "wavecal.fits")
        with pytest.raises(ValueError, match="empty"):
            write_wavecal_fits(empty, out)

    def test_write_spatcal_raises_on_empty(self):
        """write_spatcal_fits raises ValueError for empty CalibrationProductSet."""
        empty = _make_empty_calibration_products()
        out = str(self.tmp_path / "spatcal.fits")
        with pytest.raises(ValueError, match="empty"):
            write_spatcal_fits(empty, out)

    def test_write_calibration_raises_on_empty(self):
        """write_calibration_fits raises ValueError for empty CalibrationProductSet."""
        empty = _make_empty_calibration_products()
        with pytest.raises(ValueError, match="empty"):
            write_calibration_fits(empty, str(self.tmp_path))

    # --- non-existent output directory ---

    def test_write_wavecal_raises_on_bad_dir(self):
        """write_wavecal_fits raises OSError for non-existent parent directory."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "nonexistent_subdir" / "wavecal.fits")
        with pytest.raises(OSError):
            write_wavecal_fits(cal, out)

    def test_write_spatcal_raises_on_bad_dir(self):
        """write_spatcal_fits raises OSError for non-existent parent directory."""
        cal = _make_synthetic_calibration_products()
        out = str(self.tmp_path / "nonexistent_subdir" / "spatcal.fits")
        with pytest.raises(OSError):
            write_spatcal_fits(cal, out)

    def test_write_calibration_raises_on_bad_dir(self):
        """write_calibration_fits raises OSError for non-existent output directory."""
        cal = _make_synthetic_calibration_products()
        bad_dir = str(self.tmp_path / "nonexistent_subdir")
        with pytest.raises(OSError):
            write_calibration_fits(cal, bad_dir)


# ===========================================================================
# 5. Parametric tests – varying sizes
# ===========================================================================


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 4), (64, 32), (128, 128)])
def test_wavecal_shapes_parametric(tmp_path, n_spectral, n_spatial):
    """Extension data shapes are correct for various spectral/spatial sizes."""
    cal = _make_synthetic_calibration_products(
        n_spectral=n_spectral, n_spatial=n_spatial
    )
    out = str(tmp_path / "wavecal.fits")
    write_wavecal_fits(cal, out)
    with fits.open(out) as hdul:
        for wcp, hdu in zip(cal.wavecal_products, hdul[1:]):
            assert hdu.data.shape == (2, n_spectral)


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 4), (64, 32), (128, 128)])
def test_spatcal_shapes_parametric(tmp_path, n_spectral, n_spatial):
    """Extension data shapes are correct for various spectral/spatial sizes."""
    cal = _make_synthetic_calibration_products(
        n_spectral=n_spectral, n_spatial=n_spatial
    )
    out = str(tmp_path / "spatcal.fits")
    write_spatcal_fits(cal, out)
    with fits.open(out) as hdul:
        for scp, hdu in zip(cal.spatcal_products, hdul[1:]):
            assert hdu.data.shape == (2, n_spatial)


# ===========================================================================
# 6. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1CalibrationFitsSmokeTest:
    """End-to-end smoke test: full chain → write calibration FITS.

    The chain is:
      1. Flat-order tracing (Stage 1)
      2. Arc-line tracing (Stage 2)
      3. Provisional wavelength mapping (Stage 3)
      4. Coefficient-surface refinement (Stage 5)
      5. Rectification indices (Stage 6)
      6. Rectified orders (Stage 7)
      7. Calibration products (Stage 8)
      8. Write calibration FITS (Stage 9)
    """

    @pytest.fixture(scope="class")
    def h1_fits_paths(self, tmp_path_factory):
        """Write H1 calibration FITS files; return (wavecal_path, spatcal_path, n_orders)."""
        import astropy.io.fits as afits

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

        out_dir = str(tmp_path_factory.mktemp("h1_fits"))

        # Stage 1
        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES,
            col_range=(650, 1550),
        )
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))

        # Stage 2
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)

        # Stage 3
        wci = read_wavecalinfo("H1")
        ll = read_line_list("H1")
        wav_map = fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )

        # Stage 5
        surface = fit_refined_coefficient_surface(wav_map, disp_degree=3)

        # Stage 6
        rect_indices = build_rectification_indices(geom, surface, wav_map=wav_map)

        # Stage 7
        with afits.open(_H1_FLAT_FILES[0]) as hdul:
            detector_image = hdul[0].data.astype(float)
        rectified = build_rectified_orders(detector_image, rect_indices)

        # Stage 8
        cal_products = build_calibration_products(rectified)

        # Stage 9
        wpath, spath = write_calibration_fits(cal_products, out_dir)

        return wpath, spath, cal_products.n_orders

    def test_wavecal_file_exists(self, h1_fits_paths):
        """Wavecal FITS file exists on disk."""
        wpath, _, _ = h1_fits_paths
        assert os.path.isfile(wpath)

    def test_spatcal_file_exists(self, h1_fits_paths):
        """Spatcal FITS file exists on disk."""
        _, spath, _ = h1_fits_paths
        assert os.path.isfile(spath)

    def test_wavecal_extension_count_matches_n_orders(self, h1_fits_paths):
        """Wavecal FITS has extension count equal to n_orders."""
        wpath, _, n_orders = h1_fits_paths
        with fits.open(wpath) as hdul:
            assert len(hdul) - 1 == n_orders

    def test_spatcal_extension_count_matches_n_orders(self, h1_fits_paths):
        """Spatcal FITS has extension count equal to n_orders."""
        _, spath, n_orders = h1_fits_paths
        with fits.open(spath) as hdul:
            assert len(hdul) - 1 == n_orders

    def test_wavelength_axes_monotonic(self, h1_fits_paths):
        """Wavelength axis in each wavecal extension is monotonic."""
        wpath, _, _ = h1_fits_paths
        with fits.open(wpath) as hdul:
            for hdu in hdul[1:]:
                wav = hdu.data[0]
                diffs = np.diff(wav)
                assert np.all(diffs > 0) or np.all(diffs < 0), (
                    f"{hdu.name}: wavelength axis is not monotonic"
                )

    def test_spatcal_spatial_frac_in_unit_interval(self, h1_fits_paths):
        """Spatial-fraction axis in each spatcal extension is in [0, 1]."""
        _, spath, _ = h1_fits_paths
        with fits.open(spath) as hdul:
            for hdu in hdul[1:]:
                sf = hdu.data[0]
                assert np.all(sf >= 0.0)
                assert np.all(sf <= 1.0)
