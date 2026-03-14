"""
Tests for the iSHELL flat-field generation path.

These tests verify:
  - read_flatcal_file() works correctly with the packaged iSHELL flatinfo
    files for all supported J, H, and K modes,
  - the returned modeinfo dict has the correct structure and dimensional
    consistency (orders, edgecoeffs, xranges, guesspos shapes all agree),
  - the SLIT keyword parsing in make_flat handles iSHELL format correctly,
  - the flatinfo path fallback (data/ subdirectory) in make_flat works,
  - make_flat runs end-to-end for iSHELL using synthetic flat frames with
    the order-tracing step mocked to use the packaged edge coefficients,
  - appropriate errors are raised for malformed or missing inputs.

No real iSHELL science data is required: synthetic MEF FITS files are
created in temporary directories using the same fixture helpers as in
test_ishell_ingestion.py.
"""

from __future__ import annotations

import os
import re
import tempfile
from os.path import join
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

import pyspextool as ps
from pyspextool.extract.flat import read_flatcal_file
from pyspextool.setup_utils import set_instrument

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODES = [
    "J0", "J1", "J2", "J3",
    "H1", "H2", "H3",
    "K1", "K2", "K3", "Kgas",
]

# Representative one mode per band for faster testing
REPRESENTATIVE_MODES = ["J0", "H1", "K1"]

_ISHELL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "src", "pyspextool", "instruments", "ishell", "data",
)

# Detector shape for iSHELL
_NROWS = 2048
_NCOLS = 2048


# ---------------------------------------------------------------------------
# Helper to build the expected flatinfo path
# ---------------------------------------------------------------------------

def _flatinfo_path(mode):
    return join(_ISHELL_DATA_DIR, f"{mode}_flatinfo.fits")


# ---------------------------------------------------------------------------
# Helper to create a synthetic iSHELL MEF FITS file
# ---------------------------------------------------------------------------

def _make_ishell_header(mode="K1", slit="0.375_K1", **extra):
    hdr = fits.Header()
    hdr['INSTRUME'] = 'iSHELL'
    hdr['DATE_OBS'] = '2024-06-15'
    hdr['TIME_OBS'] = '08:30:00.0'
    hdr['MJD_OBS'] = 60476.354167
    hdr['ITIME'] = 30.0
    hdr['CO_ADDS'] = 1
    hdr['NDR'] = 16
    hdr['TCS_RA'] = '05:35:17.3'
    hdr['TCS_DEC'] = '-05:23:28'
    hdr['TCS_HA'] = '01:22:00'
    hdr['TCS_AM'] = 1.12
    hdr['POSANGLE'] = 45.0
    hdr['IRAFNAME'] = 'ishell_0042.fits'
    hdr['PASSBAND'] = mode
    hdr['SLIT'] = slit
    hdr['OBJECT'] = 'flat'
    for k, v in extra.items():
        hdr[k] = v
    return hdr


def _make_synthetic_mef(path, mode="K1", slit="0.375_K1",
                         shape=(_NROWS, _NCOLS), flat_value=10000.0):
    """
    Write a synthetic 3-extension iSHELL MEF FITS file to *path*.

    The science image (ext 0) is filled with *flat_value* to simulate an
    illuminated flat-field frame.  Pedestal / signal sums (ext 1, ext 2) are
    filled with half the flat value so that total counts remain below LINCORMAX.
    """
    hdr = _make_ishell_header(mode=mode, slit=slit)
    ext0 = fits.PrimaryHDU(
        data=np.full(shape, flat_value, dtype=np.float32),
        header=hdr,
    )
    ext1 = fits.ImageHDU(data=np.full(shape, flat_value / 2, dtype=np.float32))
    ext2 = fits.ImageHDU(data=np.full(shape, flat_value / 2, dtype=np.float32))
    hdul = fits.HDUList([ext0, ext1, ext2])
    hdul.writeto(str(path), overwrite=True)
    return path


# ===========================================================================
# 1. read_flatcal_file with iSHELL flatinfo files
# ===========================================================================


class TestReadFlatcalFileIshell:
    """read_flatcal_file must parse every iSHELL flatinfo FITS correctly."""

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_returns_dict_for_all_modes(self, mode):
        """read_flatcal_file must return a dict for every supported mode."""
        path = _flatinfo_path(mode)
        result = read_flatcal_file(path)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_required_keys_present(self, mode):
        """All keys required by make_flat must be present."""
        required = {
            'rotation', 'slith_arc', 'slith_pix', 'slith_range',
            'orders', 'rpppix', 'ps', 'step', 'flatfrac', 'comwidth',
            'edgedeg', 'nxgrid', 'nygrid', 'oversamp', 'ybuffer',
            'ycororder', 'xranges', 'edgecoeffs', 'guesspos',
        }
        path = _flatinfo_path(mode)
        result = read_flatcal_file(path)
        missing = required - set(result.keys())
        assert not missing, f"Mode {mode}: missing keys {missing}"

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_orders_is_nonempty_array(self, mode):
        result = read_flatcal_file(_flatinfo_path(mode))
        orders = result['orders']
        assert isinstance(orders, np.ndarray)
        assert len(orders) > 0

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_rotation_is_int(self, mode):
        result = read_flatcal_file(_flatinfo_path(mode))
        assert isinstance(result['rotation'], (int, np.integer))

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_slith_arc_positive(self, mode):
        result = read_flatcal_file(_flatinfo_path(mode))
        assert result['slith_arc'] > 0.0

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_ps_positive(self, mode):
        """Plate scale must be positive arcsec/pixel."""
        result = read_flatcal_file(_flatinfo_path(mode))
        assert result['ps'] > 0.0


# ===========================================================================
# 2. Dimensional consistency of order metadata
# ===========================================================================


class TestFlatinfoOrderMetadata:
    """Shapes of per-order arrays must be consistent with the order list."""

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_xranges_shape(self, mode):
        result = read_flatcal_file(_flatinfo_path(mode))
        norders = len(result['orders'])
        assert result['xranges'].shape == (norders, 2), (
            f"Mode {mode}: xranges shape {result['xranges'].shape} "
            f"!= ({norders}, 2)")

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_edgecoeffs_shape(self, mode):
        result = read_flatcal_file(_flatinfo_path(mode))
        norders = len(result['orders'])
        edgedeg = result['edgedeg']
        expected = (norders, 2, edgedeg + 1)
        assert result['edgecoeffs'].shape == expected, (
            f"Mode {mode}: edgecoeffs shape {result['edgecoeffs'].shape} "
            f"!= {expected}")

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_guesspos_shape(self, mode):
        result = read_flatcal_file(_flatinfo_path(mode))
        norders = len(result['orders'])
        assert result['guesspos'].shape == (norders, 2), (
            f"Mode {mode}: guesspos shape {result['guesspos'].shape} "
            f"!= ({norders}, 2)")

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_xranges_within_detector(self, mode):
        """xranges column bounds must lie within [0, ncols-1]."""
        result = read_flatcal_file(_flatinfo_path(mode))
        assert np.all(result['xranges'] >= 0), (
            f"Mode {mode}: xranges contain negative values")
        assert np.all(result['xranges'] < _NCOLS), (
            f"Mode {mode}: xranges exceed detector width {_NCOLS}")

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_xranges_start_before_end(self, mode):
        """xranges[:,0] (start) must be < xranges[:,1] (end) for each order."""
        result = read_flatcal_file(_flatinfo_path(mode))
        assert np.all(result['xranges'][:, 0] < result['xranges'][:, 1]), (
            f"Mode {mode}: some orders have start >= end column in xranges")


# ===========================================================================
# 3. SLIT keyword parsing robustness
# ===========================================================================


class TestSlitParsing:
    """The regex-based slit-width extractor must handle all known formats."""

    @staticmethod
    def _parse_slit(slit_str):
        """Reproduce the slit-width extraction logic from make_flat.py."""
        m = re.match(r'^([0-9]+(?:\.[0-9]+)?)', str(slit_str))
        if m:
            return float(m.group(1))
        return float(str(slit_str)[0:3])

    @pytest.mark.parametrize("slit_str, expected", [
        # iSHELL formats
        ('0.375_K1', 0.375),
        ('0.75_K1',  0.75),
        ('1.5_K1',   1.5),
        ('3.0_K1',   3.0),
        ('4.5_K1',   4.5),
        # SpeX / uSpeX formats (must still work after the change)
        ('0.3x15',   0.3),
        ('0.5x15',   0.5),
        ('0.8x60',   0.8),
        ('1.6x60',   1.6),
        ('3.0x60',   3.0),
    ])
    def test_parse_slit_value(self, slit_str, expected):
        result = self._parse_slit(slit_str)
        assert abs(result - expected) < 1e-9, (
            f"_parse_slit({slit_str!r}) = {result} != {expected}")


# ===========================================================================
# 4. Flatinfo path fallback (data/ subdirectory)
# ===========================================================================


class TestFlatinfoPathFallback:
    """make_flat must find flatinfo in data/ when absent at instrument root."""

    def test_flatinfo_found_in_data_subdir(self):
        """The flatinfo file is stored in data/ for iSHELL; must be found."""
        # The top-level instrument_path does NOT contain J0_flatinfo.fits;
        # it lives in instrument_path/data/.
        import pyspextool.config as _cfg
        set_instrument('ishell')
        instrument_path = _cfg.state['instrument_path']

        root_file = join(instrument_path, 'J0_flatinfo.fits')
        data_file = join(instrument_path, 'data', 'J0_flatinfo.fits')

        assert not os.path.isfile(root_file), (
            "J0_flatinfo.fits unexpectedly found at instrument root; "
            "the path-fallback test is no longer needed.")
        assert os.path.isfile(data_file), (
            f"J0_flatinfo.fits not found at expected data/ path: {data_file}")

    def test_read_flatcal_file_via_data_path(self):
        """read_flatcal_file must succeed when given the data/ path."""
        path = _flatinfo_path('K1')
        result = read_flatcal_file(path)
        assert len(result['orders']) > 0


# ===========================================================================
# 5. End-to-end make_flat with synthetic iSHELL flat frames
# ===========================================================================


class TestMakeFlatIshellEndToEnd:
    """
    End-to-end make_flat tests for iSHELL using synthetic flat frames.

    locate_orders() is mocked with the pre-set edge coefficients from the
    packaged flatinfo so that the test does not require a realistic
    illuminated flat image.  normalize=False is used to skip the
    computationally expensive per-order normalisation on the full-sized
    2048×2048 detector – that step's correctness is tested separately.
    All other pipeline steps (scaling, medianing, writing) execute on
    real code paths.
    """

    @pytest.fixture(autouse=True)
    def setup_ishell(self, tmp_path):
        """Configure pySpextool for iSHELL and set up per-test temp dirs."""
        self.raw_path = str(tmp_path / 'raw')
        self.cal_path = str(tmp_path / 'cal')
        self.qa_path = str(tmp_path / 'qa')
        os.makedirs(self.raw_path)
        os.makedirs(self.cal_path)
        os.makedirs(self.qa_path)

        ps.pyspextool_setup(
            instrument='ishell',
            raw_path=self.raw_path,
            cal_path=self.cal_path,
            proc_path=self.cal_path,
            qa_path=self.qa_path,
            verbose=False,
            qa_show=False,
            qa_write=False,
        )

    def _write_flat_files(self, mode, n=3, slit='0.375_K1'):
        """
        Write *n* synthetic iSHELL flat FITS files.

        Returns the files parameter suitable for passing to make_flat, i.e.
        a comma-separated string of bare file names (without the raw_path
        prefix, which make_flat prepends automatically).
        """
        names = []
        for i in range(1, n + 1):
            # Name files to match the iSHELL naming convention used in
            # practice: <prefix><NNNN>.<a|b>.fits
            fname = f'ishell_flat{i:04d}.a.fits'
            fpath = join(self.raw_path, fname)
            _make_synthetic_mef(fpath, mode=mode, slit=slit)
            names.append(fname)
        # Return as a comma-separated string – the 'filename' read mode
        return ','.join(names)

    def _mock_locate_orders(self, mode):
        """
        Return a mock for locate_orders that uses the pre-packaged edge
        coefficients from the iSHELL flatinfo file.
        """
        modeinfo = read_flatcal_file(_flatinfo_path(mode))

        def _locate_orders_stub(img, guess_positions, search_ranges,
                                step_size, slit_height_range, poly_degree,
                                ybuffer, intensity_fraction, com_width,
                                **kwargs):
            return modeinfo['edgecoeffs'], modeinfo['xranges']

        return _locate_orders_stub

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_make_flat_produces_fits_file(self, mode):
        """make_flat must write a FITS output file for each J/H/K mode."""
        files_arg = self._write_flat_files(mode)
        output_name = f'flat_{mode}'

        with patch('pyspextool.extract.make_flat.locate_orders',
                   side_effect=self._mock_locate_orders(mode)):
            ps.extract.make_flat(files_arg, output_name,
                                 normalize=False, verbose=False)

        output_fits = join(self.cal_path, output_name + '.fits')
        assert os.path.isfile(output_fits), (
            f"make_flat did not write {output_fits}")

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_flat_output_has_required_extensions(self, mode):
        """The output flat FITS must contain exactly 5 extensions."""
        files_arg = self._write_flat_files(mode)
        output_name = f'flat_{mode}'

        with patch('pyspextool.extract.make_flat.locate_orders',
                   side_effect=self._mock_locate_orders(mode)):
            ps.extract.make_flat(files_arg, output_name,
                                 normalize=False, verbose=False)

        output_fits = join(self.cal_path, output_name + '.fits')
        with fits.open(output_fits) as hdul:
            # write_flat writes:
            #   [0] PrimaryHDU (header-only, no data)
            #   [1] flat image
            #   [2] variance image
            #   [3] flag image
            #   [4] order mask
            assert len(hdul) == 5, (
                f"Expected 5 extensions, got {len(hdul)}")

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_flat_image_shape_matches_detector(self, mode):
        """The flat image (extension 1) must have the iSHELL detector shape."""
        files_arg = self._write_flat_files(mode)
        output_name = f'flat_{mode}'

        with patch('pyspextool.extract.make_flat.locate_orders',
                   side_effect=self._mock_locate_orders(mode)):
            ps.extract.make_flat(files_arg, output_name,
                                 normalize=False, verbose=False)

        output_fits = join(self.cal_path, output_name + '.fits')
        with fits.open(output_fits) as hdul:
            # Extension 1 carries the flat image data
            flat_shape = hdul[1].data.shape
        assert flat_shape == (_NROWS, _NCOLS), (
            f"Flat shape {flat_shape} != ({_NROWS}, {_NCOLS})")

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_flat_header_records_mode(self, mode):
        """The flat FITS header must record the observing mode."""
        files_arg = self._write_flat_files(mode)
        output_name = f'flat_{mode}'

        with patch('pyspextool.extract.make_flat.locate_orders',
                   side_effect=self._mock_locate_orders(mode)):
            ps.extract.make_flat(files_arg, output_name,
                                 normalize=False, verbose=False)

        output_fits = join(self.cal_path, output_name + '.fits')
        with fits.open(output_fits) as hdul:
            hdr = hdul[0].header
        assert 'MODE' in hdr, "MODE keyword missing from flat header"
        assert hdr['MODE'] == mode, (
            f"MODE header value '{hdr['MODE']}' != '{mode}'")

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_flat_header_records_orders(self, mode):
        """The ORDERS header keyword must list the expected echelle orders."""
        files_arg = self._write_flat_files(mode)
        output_name = f'flat_{mode}'

        expected_orders = read_flatcal_file(_flatinfo_path(mode))['orders']

        with patch('pyspextool.extract.make_flat.locate_orders',
                   side_effect=self._mock_locate_orders(mode)):
            ps.extract.make_flat(files_arg, output_name,
                                 normalize=False, verbose=False)

        output_fits = join(self.cal_path, output_name + '.fits')
        with fits.open(output_fits) as hdul:
            hdr = hdul[0].header
        assert 'ORDERS' in hdr, "ORDERS keyword missing from flat header"
        written_orders = [int(o) for o in hdr['ORDERS'].split(',')]
        assert written_orders == list(expected_orders), (
            f"ORDERS in header {written_orders} != expected {list(expected_orders)}")

    @pytest.mark.parametrize("mode, slit_str, expected_width", [
        ("J0", "0.375_K1", 0.375),
        ("H1", "0.75_K1",  0.75),
        ("K1", "1.5_K1",   1.5),
    ])
    def test_slit_width_parsed_correctly(self, mode, slit_str, expected_width):
        """SLTW_ARC in the output flat must match the raw SLIT keyword."""
        files_arg = self._write_flat_files(mode, slit=slit_str)
        output_name = f'flat_{mode}_slit'

        with patch('pyspextool.extract.make_flat.locate_orders',
                   side_effect=self._mock_locate_orders(mode)):
            ps.extract.make_flat(files_arg, output_name,
                                 normalize=False, verbose=False)

        output_fits = join(self.cal_path, output_name + '.fits')
        with fits.open(output_fits) as hdul:
            hdr = hdul[0].header
        assert 'SLTW_ARC' in hdr, "SLTW_ARC keyword missing from flat header"
        assert abs(float(hdr['SLTW_ARC']) - expected_width) < 1e-4, (
            f"SLTW_ARC {hdr['SLTW_ARC']} != {expected_width}")


# ===========================================================================
# 6. Failure modes for malformed or missing inputs
# ===========================================================================


class TestMakeFlatIshellFailureModes:
    """make_flat must raise meaningful errors for bad inputs."""

    @pytest.fixture(autouse=True)
    def setup_ishell(self, tmp_path):
        self.raw_path = str(tmp_path / 'raw')
        self.cal_path = str(tmp_path / 'cal')
        os.makedirs(self.raw_path)
        os.makedirs(self.cal_path)

        ps.pyspextool_setup(
            instrument='ishell',
            raw_path=self.raw_path,
            cal_path=self.cal_path,
            proc_path=self.cal_path,
            qa_path=self.cal_path,
            verbose=False,
            qa_show=False,
            qa_write=False,
        )

    def test_missing_flatinfo_raises(self, tmp_path):
        """read_flatcal_file must raise FileNotFoundError for a missing file."""
        with pytest.raises((FileNotFoundError, OSError)):
            read_flatcal_file(str(tmp_path / 'nonexistent_flatinfo.fits'))

    def test_wrong_nint_raises(self, tmp_path):
        """load_data must raise pySpextoolError for a non-3-extension file."""
        from pyspextool.instruments.ishell.ishell import load_data
        from pyspextool.pyspextoolerror import pySpextoolError

        # Write a 2-extension MEF (invalid for iSHELL)
        bad_file = str(tmp_path / 'bad.fits')
        hdr = _make_ishell_header()
        hdul = fits.HDUList([
            fits.PrimaryHDU(data=np.zeros((32, 32), dtype=np.float32),
                            header=hdr),
            fits.ImageHDU(data=np.zeros((32, 32), dtype=np.float32)),
        ])
        hdul.writeto(bad_file, overwrite=True)

        with pytest.raises(pySpextoolError):
            load_data(bad_file, {'max': 30000, 'bit': 0}, [], None, False)

    def test_missing_itime_raises(self, tmp_path):
        """load_data must raise pySpextoolError when ITIME is absent."""
        from pyspextool.instruments.ishell.ishell import load_data
        from pyspextool.pyspextoolerror import pySpextoolError

        bad_file = str(tmp_path / 'no_itime.fits')
        hdr = _make_ishell_header()
        del hdr['ITIME']
        hdul = fits.HDUList([
            fits.PrimaryHDU(data=np.zeros((32, 32), dtype=np.float32),
                            header=hdr),
            fits.ImageHDU(data=np.zeros((32, 32), dtype=np.float32)),
            fits.ImageHDU(data=np.zeros((32, 32), dtype=np.float32)),
        ])
        hdul.writeto(bad_file, overwrite=True)

        with pytest.raises(pySpextoolError):
            load_data(bad_file, {'max': 30000, 'bit': 0}, [], None, False)

    def test_zero_itime_raises(self, tmp_path):
        """load_data must raise pySpextoolError when ITIME <= 0."""
        from pyspextool.instruments.ishell.ishell import load_data
        from pyspextool.pyspextoolerror import pySpextoolError

        bad_file = str(tmp_path / 'zero_itime.fits')
        hdr = _make_ishell_header()
        hdr['ITIME'] = 0.0
        hdul = fits.HDUList([
            fits.PrimaryHDU(data=np.zeros((32, 32), dtype=np.float32),
                            header=hdr),
            fits.ImageHDU(data=np.zeros((32, 32), dtype=np.float32)),
            fits.ImageHDU(data=np.zeros((32, 32), dtype=np.float32)),
        ])
        hdul.writeto(bad_file, overwrite=True)

        with pytest.raises(pySpextoolError):
            load_data(bad_file, {'max': 30000, 'bit': 0}, [], None, False)
