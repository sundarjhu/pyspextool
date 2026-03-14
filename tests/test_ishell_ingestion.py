"""
Tests for the iSHELL raw FITS ingestion layer (Phase 1).

These tests exercise read_fits(), get_header(), and load_data() using
synthetic multi-extension FITS (MEF) fixtures created in a tmp directory.
No real iSHELL data files are required.

Synthetic MEF structure mirrors the documented raw iSHELL format
(iSHELL Spextool Manual §2.3):

  Extension 0 (PRIMARY): signal difference S = Σ pedestal − Σ signal
  Extension 1 (IMAGE):   pedestal sum
  Extension 2 (IMAGE):   signal sum

The primary header carries all standard iSHELL FITS keywords
(DATE_OBS, TIME_OBS, MJD_OBS, ITIME, CO_ADDS, TCS_RA, TCS_DEC, TCS_HA,
TCS_AM, POSANGLE, IRAFNAME, PASSBAND).
"""

import numpy as np
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.ishell import (
    GAIN_ELECTRONS_PER_DN,
    READNOISE_ELECTRONS,
    get_header,
    load_data,
    read_fits,
)
from pyspextool.pyspextoolerror import pySpextoolError


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_NROWS = 32   # deliberately small to keep tests fast
_NCOLS = 32


def _make_ishell_header(**extra):
    """Return a minimal but valid iSHELL-like primary FITS header."""
    hdr = fits.Header()
    hdr['INSTRUME'] = 'iSHELL'
    hdr['DATE_OBS'] = '2024-06-15'
    hdr['TIME_OBS'] = '08:30:00.0'
    hdr['MJD_OBS'] = 60476.354167
    hdr['ITIME'] = 30.0
    hdr['CO_ADDS'] = 2
    hdr['NDR'] = 16
    hdr['TCS_RA'] = '05:35:17.3'
    hdr['TCS_DEC'] = '-05:23:28'
    hdr['TCS_HA'] = '01:22:00'
    hdr['TCS_AM'] = 1.12
    hdr['POSANGLE'] = 45.0
    hdr['IRAFNAME'] = 'ishell_0042.fits'
    hdr['PASSBAND'] = 'K1'
    hdr['SLIT'] = '0.375_K1'
    hdr['OBJECT'] = 'HD12345'
    for k, v in extra.items():
        hdr[k] = v
    return hdr


def _make_synthetic_mef(path, sig_value=1000.0, ped_value=500.0,
                         sig_sum_value=800.0, shape=(_NROWS, _NCOLS),
                         **header_extra):
    """
    Write a 3-extension MEF FITS file to *path* with known data values.

    Parameters
    ----------
    sig_value : float
        Pixel value used for extension 0 (signal difference).
    ped_value : float
        Pixel value used for extension 1 (pedestal sum).
    sig_sum_value : float
        Pixel value used for extension 2 (signal sum).
    shape : tuple
        (nrows, ncols) of the data arrays.
    header_extra : keyword arguments
        Extra FITS keywords to set in the primary header.
    """
    hdr = _make_ishell_header(**header_extra)
    ext0 = fits.PrimaryHDU(
        data=np.full(shape, sig_value, dtype=np.float32),
        header=hdr,
    )
    ext1 = fits.ImageHDU(data=np.full(shape, ped_value, dtype=np.float32))
    ext2 = fits.ImageHDU(data=np.full(shape, sig_sum_value, dtype=np.float32))
    hdul = fits.HDUList([ext0, ext1, ext2])
    hdul.writeto(str(path), overwrite=True)
    return path


@pytest.fixture()
def synthetic_mef(tmp_path):
    """Return the path of a standard synthetic iSHELL MEF FITS file."""
    return _make_synthetic_mef(tmp_path / 'ishell_0042.fits')


@pytest.fixture()
def synthetic_mef_pair(tmp_path):
    """Return paths to two synthetic MEF files for pair-subtraction tests."""
    a = _make_synthetic_mef(tmp_path / 'a.fits', sig_value=5000.0)
    b = _make_synthetic_mef(tmp_path / 'b.fits', sig_value=3000.0)
    return str(a), str(b)


# ---------------------------------------------------------------------------
# get_header() – keyword mapping
# ---------------------------------------------------------------------------


class TestGetHeader:
    """Tests for get_header() keyword extraction and normalisation."""

    def _header(self, **extra):
        return _make_ishell_header(**extra)

    def test_required_output_keys_present(self):
        """All pySpextool-standard output keys must be present."""
        result = get_header(self._header(), [])
        for key in ('AM', 'HA', 'PA', 'RA', 'DEC', 'ITIME', 'NCOADDS',
                    'IMGITIME', 'TIME', 'DATE', 'MJD', 'FILENAME',
                    'MODE', 'INSTR'):
            assert key in result, f"missing output key '{key}'"

    def test_each_value_is_two_element_list(self):
        """Every entry must be a [value, comment] list."""
        result = get_header(self._header(), [])
        for key, val in result.items():
            assert isinstance(val, list) and len(val) == 2, (
                f"get_header['{key}'] = {val!r} is not a [value, comment] list")

    def test_airmass_value(self):
        result = get_header(self._header(), [])
        assert abs(result['AM'][0] - 1.12) < 1e-9

    def test_posangle_value(self):
        result = get_header(self._header(), [])
        assert abs(result['PA'][0] - 45.0) < 1e-9

    def test_itime_value(self):
        result = get_header(self._header(), [])
        assert result['ITIME'][0] == 30.0

    def test_ncoadds_value(self):
        result = get_header(self._header(), [])
        assert result['NCOADDS'][0] == 2

    def test_imgitime_is_itime_times_ncoadds(self):
        result = get_header(self._header(), [])
        assert result['IMGITIME'][0] == pytest.approx(60.0)

    def test_mjd_passthrough(self):
        result = get_header(self._header(), [])
        assert abs(result['MJD'][0] - 60476.354167) < 1e-5

    def test_date_value(self):
        result = get_header(self._header(), [])
        assert result['DATE'][0] == '2024-06-15'

    def test_time_value(self):
        result = get_header(self._header(), [])
        assert result['TIME'][0] == '08:30:00.0'

    def test_filename_from_irafname(self):
        result = get_header(self._header(), [])
        assert result['FILENAME'][0] == 'ishell_0042.fits'

    def test_mode_from_passband(self):
        result = get_header(self._header(), [])
        assert result['MODE'][0] == 'K1'

    def test_instr_always_ishell(self):
        result = get_header(self._header(), [])
        assert result['INSTR'][0] == 'iSHELL'

    def test_ha_has_sign_prefix(self):
        """HA string must always start with + or -."""
        result = get_header(self._header(), [])
        ha = result['HA'][0]
        assert ha[0] in ('+', '-'), f"HA '{ha}' lacks sign prefix"

    def test_dec_has_sign_prefix(self):
        """DEC string must start with + or -."""
        result = get_header(self._header(), [])
        dec = result['DEC'][0]
        assert dec[0] in ('+', '-'), f"DEC '{dec}' lacks sign prefix"

    def test_dec_negative_preserved(self):
        hdr = self._header()
        hdr['TCS_DEC'] = '-45:00:00'
        result = get_header(hdr, [])
        assert result['DEC'][0].startswith('-')

    def test_ra_value(self):
        result = get_header(self._header(), [])
        assert result['RA'][0] == '05:35:17.3'

    def test_missing_airmass_is_nan(self):
        import math
        hdr = self._header()
        del hdr['TCS_AM']
        result = get_header(hdr, [])
        assert math.isnan(result['AM'][0])

    def test_missing_mjd_computed_from_date_time(self):
        hdr = self._header()
        del hdr['MJD_OBS']
        result = get_header(hdr, [])
        # Should be close to the fixture value
        assert abs(result['MJD'][0] - 60476.354167) < 0.01

    def test_completely_empty_header_no_exception(self):
        """An empty header must not raise; all missing keys get nan/unknown."""
        result = get_header(fits.Header(), [])
        assert result['INSTR'][0] == 'iSHELL'
        assert result['FILENAME'][0] == 'unknown'
        assert result['MODE'][0] == 'unknown'

    def test_extra_keywords_forwarded(self):
        """Extra user keywords are passed through."""
        hdr = self._header()
        hdr['OBJECT'] = 'MyTarget'
        result = get_header(hdr, ['OBJECT'])
        assert 'OBJECT' in result
        assert result['OBJECT'][0] == 'MyTarget'

    def test_fallback_ra_without_tcs_prefix(self):
        hdr = self._header()
        del hdr['TCS_RA']
        hdr['RA'] = '10:20:30.0'
        result = get_header(hdr, [])
        assert result['RA'][0] == '10:20:30.0'

    def test_mode_fallback_to_grat(self):
        hdr = self._header()
        del hdr['PASSBAND']
        hdr['GRAT'] = 'H2'
        result = get_header(hdr, [])
        assert result['MODE'][0] == 'H2'


# ---------------------------------------------------------------------------
# load_data() – single-file ingestion
# ---------------------------------------------------------------------------


class TestLoadData:
    """Tests for load_data() with synthetic MEF files."""

    _linearity_info = {'max': 30000, 'bit': 0}

    def test_returns_four_element_tuple(self, synthetic_mef):
        result = load_data(str(synthetic_mef), self._linearity_info, [])
        assert len(result) == 4

    def test_img_shape(self, synthetic_mef):
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        assert img.shape == (_NROWS, _NCOLS)

    def test_var_shape(self, synthetic_mef):
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        assert var.shape == (_NROWS, _NCOLS)

    def test_bitmask_shape(self, synthetic_mef):
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        assert mask.shape == (_NROWS, _NCOLS)

    def test_bitmask_dtype_uint8(self, synthetic_mef):
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        assert mask.dtype == np.uint8

    def test_hdrinfo_is_dict(self, synthetic_mef):
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        assert isinstance(hdr, dict)

    def test_hdrinfo_has_required_keys(self, synthetic_mef):
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        for key in ('AM', 'ITIME', 'NCOADDS', 'DATE', 'TIME', 'MJD',
                    'INSTR', 'MODE'):
            assert key in hdr, f"hdrinfo missing key '{key}'"

    def test_img_normalised_to_dn_per_sec(self, synthetic_mef):
        """img = ext0 / (ITIME * CO_ADDS) = 1000 / (30 * 2) = 16.667 DN/s."""
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        expected = 1000.0 / (30.0 * 2)
        assert np.allclose(img, expected), (
            f"img not normalised correctly; got mean={img.mean():.4f}, "
            f"expected {expected:.4f}")

    def test_var_is_positive(self, synthetic_mef):
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        assert np.all(var >= 0.0), "variance must be non-negative"

    def test_var_finite(self, synthetic_mef):
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        assert np.all(np.isfinite(var)), "variance must be finite everywhere"

    def test_no_nonlinear_pixels_below_threshold(self, synthetic_mef):
        """With ped+sig = 500+800 = 1300 << 30000, all pixels should be clean."""
        img, var, hdr, mask = load_data(
            str(synthetic_mef), self._linearity_info, [])
        assert np.all(mask == 0), (
            "Expected no flagged pixels when counts well below threshold")

    def test_nonlinear_pixels_flagged(self, tmp_path):
        """All pixels should be flagged when ped+sig exceeds threshold."""
        path = _make_synthetic_mef(
            tmp_path / 'nonlinear.fits',
            sig_value=1000.0,
            ped_value=20000.0,   # ped + sig_sum = 20000 + 20000 = 40000 > 30000
            sig_sum_value=20000.0,
        )
        linfo = {'max': 30000, 'bit': 0}
        img, var, hdr, mask = load_data(str(path), linfo, [])
        assert np.all(mask > 0), "Expected all pixels flagged as non-linear"

    def test_bit_number_respected(self, tmp_path):
        """The bit in linearity_info['bit'] must be the one that is set."""
        path = _make_synthetic_mef(
            tmp_path / 'bit2.fits',
            ped_value=20000.0,
            sig_sum_value=20000.0,
        )
        linfo = {'max': 30000, 'bit': 2}
        img, var, hdr, mask = load_data(str(path), linfo, [])
        expected_flag = np.uint8(2 ** 2)
        assert np.all(mask == expected_flag), (
            f"Expected mask value {expected_flag}, got {np.unique(mask)}")

    def test_wrong_nint_raises_pyspextoolerror(self, tmp_path):
        """A FITS file with 4 extensions must raise pySpextoolError."""
        path = tmp_path / 'four_ext.fits'
        ext0 = fits.PrimaryHDU(data=np.zeros((_NROWS, _NCOLS)))
        ext0.header['ITIME'] = 1.0
        ext0.header['CO_ADDS'] = 1
        hdul = fits.HDUList([
            ext0,
            fits.ImageHDU(data=np.zeros((_NROWS, _NCOLS))),
            fits.ImageHDU(data=np.zeros((_NROWS, _NCOLS))),
            fits.ImageHDU(data=np.zeros((_NROWS, _NCOLS))),
        ])
        hdul.writeto(str(path), overwrite=True)
        with pytest.raises(pySpextoolError, match='3 FITS extensions'):
            load_data(str(path), {'max': 30000, 'bit': 0}, [])

    def test_single_extension_raises_pyspextoolerror(self, tmp_path):
        """A non-MEF (single extension) file must raise pySpextoolError."""
        path = tmp_path / 'single.fits'
        hdu = fits.PrimaryHDU(data=np.zeros((_NROWS, _NCOLS)))
        hdu.header['ITIME'] = 1.0
        hdu.header['CO_ADDS'] = 1
        hdu.writeto(str(path), overwrite=True)
        with pytest.raises(pySpextoolError, match='3 FITS extensions'):
            load_data(str(path), {'max': 30000, 'bit': 0}, [])

    def test_missing_itime_raises_pyspextoolerror(self, tmp_path):
        """Missing ITIME in header must raise pySpextoolError."""
        hdr = _make_ishell_header()
        del hdr['ITIME']
        path = tmp_path / 'no_itime.fits'
        ext0 = fits.PrimaryHDU(data=np.zeros((_NROWS, _NCOLS)), header=hdr)
        hdul = fits.HDUList([
            ext0,
            fits.ImageHDU(data=np.zeros((_NROWS, _NCOLS))),
            fits.ImageHDU(data=np.zeros((_NROWS, _NCOLS))),
        ])
        hdul.writeto(str(path), overwrite=True)
        with pytest.raises(pySpextoolError, match='ITIME'):
            load_data(str(path), {'max': 30000, 'bit': 0}, [])

    def test_extra_keywords_in_hdrinfo(self, tmp_path):
        """Keywords passed in 'keywords' list must appear in hdrinfo."""
        path = _make_synthetic_mef(tmp_path / 'extra.fits', OBJECT='MyTarget')
        img, var, hdr, mask = load_data(str(path), self._linearity_info,
                                        ['OBJECT'])
        assert 'OBJECT' in hdr
        assert hdr['OBJECT'][0] == 'MyTarget'


# ---------------------------------------------------------------------------
# read_fits() – multi-file orchestration
# ---------------------------------------------------------------------------


class TestReadFits:
    """Tests for read_fits() with synthetic MEF fixtures."""

    _linearity_info = {'max': 30000, 'bit': 0}

    def test_single_file_returns_squeezed_2d(self, synthetic_mef):
        """Single-file result must be squeezed to (nrows, ncols)."""
        data, var, hdrinfo, mask = read_fits(
            [str(synthetic_mef)], self._linearity_info)
        assert data.ndim == 2
        assert data.shape == (_NROWS, _NCOLS)

    def test_single_file_hdrinfo_length(self, synthetic_mef):
        data, var, hdrinfo, mask = read_fits(
            [str(synthetic_mef)], self._linearity_info)
        assert len(hdrinfo) == 1

    def test_two_files_returns_3d(self, tmp_path):
        a = _make_synthetic_mef(tmp_path / 'a.fits')
        b = _make_synthetic_mef(tmp_path / 'b.fits')
        data, var, hdrinfo, mask = read_fits(
            [str(a), str(b)], self._linearity_info)
        assert data.ndim == 3
        assert data.shape[0] == 2
        assert len(hdrinfo) == 2

    def test_pair_subtract_returns_2d(self, synthetic_mef_pair):
        a, b = synthetic_mef_pair
        data, var, hdrinfo, mask = read_fits(
            [a, b], self._linearity_info, pair_subtract=True)
        assert data.ndim == 2

    def test_pair_subtract_hdrinfo_has_one_header_per_image(self, synthetic_mef_pair):
        """pair_subtract=True must store exactly 1 header per output image (A-beam)."""
        a, b = synthetic_mef_pair
        data, var, hdrinfo, mask = read_fits(
            [a, b], self._linearity_info, pair_subtract=True)
        assert len(hdrinfo) == 1, (
            f'Expected 1 header for 1 pair-subtracted image; got {len(hdrinfo)}')

    def test_pair_subtract_values(self, synthetic_mef_pair):
        """A - B must equal (5000 - 3000) / (30 * 2) = 33.333… DN/s."""
        a, b = synthetic_mef_pair
        data, var, hdrinfo, mask = read_fits(
            [a, b], self._linearity_info, pair_subtract=True)
        expected = (5000.0 - 3000.0) / (30.0 * 2)
        assert np.allclose(data, expected), (
            f"pair subtraction wrong: mean={data.mean():.4f}, "
            f"expected {expected:.4f}")

    def test_pair_subtract_var_adds(self, synthetic_mef_pair):
        """Variance of A-B must equal varA + varB."""
        a, b = synthetic_mef_pair
        # individual variances
        _, var_a, _, _ = read_fits([a], self._linearity_info)
        _, var_b, _, _ = read_fits([b], self._linearity_info)
        _, var_ab, _, _ = read_fits([a, b], self._linearity_info,
                                    pair_subtract=True)
        assert np.allclose(var_ab, var_a + var_b)

    def test_odd_number_of_files_with_pair_subtract_raises(self, tmp_path):
        a = _make_synthetic_mef(tmp_path / 'a.fits')
        b = _make_synthetic_mef(tmp_path / 'b.fits')
        c = _make_synthetic_mef(tmp_path / 'c.fits')
        with pytest.raises(pySpextoolError):
            read_fits([str(a), str(b), str(c)], self._linearity_info,
                      pair_subtract=True)

    def test_rotate_changes_shape(self, synthetic_mef):
        """Rotation by 1 (90°) should transpose nrows and ncols."""
        data_r0, _, _, _ = read_fits(
            [str(synthetic_mef)], self._linearity_info, rotate=0)
        data_r1, _, _, _ = read_fits(
            [str(synthetic_mef)], self._linearity_info, rotate=1)
        assert data_r0.shape == (_NROWS, _NCOLS)
        assert data_r1.shape == (_NCOLS, _NROWS)

    def test_bitmask_dtype(self, synthetic_mef):
        data, var, hdrinfo, mask = read_fits(
            [str(synthetic_mef)], self._linearity_info)
        assert mask.dtype in (np.uint8, np.int8)

    def test_verbose_does_not_raise(self, synthetic_mef):
        """verbose=True must not raise an exception."""
        read_fits([str(synthetic_mef)], self._linearity_info, verbose=True)


# ---------------------------------------------------------------------------
# Supported-mode validation (get_header + load_data)
# ---------------------------------------------------------------------------


class TestModeValidation:
    """Tests for the SUPPORTED_MODES guard added to get_header()."""

    _linearity_info = {'max': 30000, 'bit': 0}

    def test_supported_mode_does_not_raise(self):
        """All J/H/K modes must be accepted without error."""
        from pyspextool.instruments.ishell.ishell import SUPPORTED_MODES
        for mode in SUPPORTED_MODES:
            hdr = _make_ishell_header(PASSBAND=mode)
            result = get_header(hdr, [])
            assert result['MODE'][0] == mode

    @pytest.mark.parametrize('bad_mode', ['L', 'Lp', 'M', 'Lc', 'M_short'])
    def test_unsupported_mode_raises_pyspextoolerror(self, bad_mode):
        """Explicitly unsupported modes (L / Lp / M family) must raise."""
        hdr = _make_ishell_header(PASSBAND=bad_mode)
        with pytest.raises(pySpextoolError, match='not supported'):
            get_header(hdr, [])

    def test_unknown_mode_does_not_raise(self):
        """A header with no PASSBAND/GRAT ('unknown') must not raise."""
        hdr = fits.Header()
        hdr['ITIME'] = 10.0
        hdr['CO_ADDS'] = 1
        result = get_header(hdr, [])
        assert result['MODE'][0] == 'unknown'

    def test_unsupported_mode_in_load_data_raises(self, tmp_path):
        """load_data must propagate the mode error from get_header."""
        path = _make_synthetic_mef(tmp_path / 'lmode.fits', PASSBAND='M')
        with pytest.raises(pySpextoolError, match='not supported'):
            load_data(str(path), self._linearity_info, [])


# ---------------------------------------------------------------------------
# Metadata sanity checks (load_data)
# ---------------------------------------------------------------------------


class TestMetadataSanityChecks:
    """Tests for the ITIME > 0 and CO_ADDS >= 1 guards in load_data()."""

    _linearity_info = {'max': 30000, 'bit': 0}

    def _write_mef(self, path, *, itime, coadds):
        """Write a minimal 3-ext MEF with the given ITIME and CO_ADDS."""
        hdr = _make_ishell_header(ITIME=itime, CO_ADDS=coadds)
        ext0 = fits.PrimaryHDU(
            data=np.ones((_NROWS, _NCOLS), dtype=np.float32), header=hdr)
        hdul = fits.HDUList([
            ext0,
            fits.ImageHDU(data=np.ones((_NROWS, _NCOLS), dtype=np.float32)),
            fits.ImageHDU(data=np.ones((_NROWS, _NCOLS), dtype=np.float32)),
        ])
        hdul.writeto(str(path), overwrite=True)
        return str(path)

    def test_positive_itime_and_valid_coadds_ok(self, tmp_path):
        """Normal values must not raise."""
        path = self._write_mef(tmp_path / 'ok.fits', itime=30.0, coadds=4)
        img, var, hdr, mask = load_data(path, self._linearity_info, [])
        assert img.shape == (_NROWS, _NCOLS)

    def test_zero_itime_raises(self, tmp_path):
        """ITIME = 0 must raise pySpextoolError."""
        path = self._write_mef(tmp_path / 'zero_itime.fits',
                               itime=0.0, coadds=1)
        with pytest.raises(pySpextoolError, match='ITIME'):
            load_data(path, self._linearity_info, [])

    def test_negative_itime_raises(self, tmp_path):
        """ITIME < 0 must raise pySpextoolError."""
        path = self._write_mef(tmp_path / 'neg_itime.fits',
                               itime=-5.0, coadds=1)
        with pytest.raises(pySpextoolError, match='ITIME'):
            load_data(path, self._linearity_info, [])

    def test_zero_coadds_raises(self, tmp_path):
        """CO_ADDS = 0 must raise pySpextoolError."""
        path = self._write_mef(tmp_path / 'zero_coadds.fits',
                               itime=10.0, coadds=0)
        with pytest.raises(pySpextoolError, match='CO_ADDS'):
            load_data(path, self._linearity_info, [])

    def test_negative_coadds_raises(self, tmp_path):
        """CO_ADDS < 0 must raise pySpextoolError."""
        path = self._write_mef(tmp_path / 'neg_coadds.fits',
                               itime=10.0, coadds=-2)
        with pytest.raises(pySpextoolError, match='CO_ADDS'):
            load_data(path, self._linearity_info, [])

    def test_pair_subtract_hdrinfo_is_abeam(self, tmp_path):
        """pair_subtract=True: hdrinfo must contain the A-beam header."""
        a = _make_synthetic_mef(tmp_path / 'a.fits', IRAFNAME='abeam.fits',
                                sig_value=5000.0)
        b = _make_synthetic_mef(tmp_path / 'b.fits', IRAFNAME='bbeam.fits',
                                sig_value=3000.0)
        _, _, hdrinfo, _ = read_fits(
            [str(a), str(b)], {'max': 30000, 'bit': 0}, pair_subtract=True)
        assert len(hdrinfo) == 1
        assert hdrinfo[0]['FILENAME'][0] == 'abeam.fits', (
            f"Expected A-beam filename; got {hdrinfo[0]['FILENAME'][0]!r}")
