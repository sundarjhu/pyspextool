"""
Tests for the iSHELL instrument backend (Phase 0 scaffolding).

These tests verify:
  - 'ishell' is registered in the pySpextool instrument list
  - The iSHELL instrument module is importable
  - set_instrument('ishell') populates the expected state variables
  - The required public callables are present with the correct signatures
  - The placeholder data files exist and are parseable
"""

import importlib
import inspect
import os

import pytest

import pyspextool.config as setup
from pyspextool.setup_utils import set_instrument


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def test_ishell_in_instruments_list():
    """'ishell' must appear in config.state['instruments']."""
    assert 'ishell' in setup.state['instruments']


# ---------------------------------------------------------------------------
# Module importability
# ---------------------------------------------------------------------------


def test_ishell_module_importable():
    """The iSHELL instrument module must be importable."""
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    assert mod is not None


def test_ishell_module_has_read_fits():
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    assert hasattr(mod, 'read_fits')
    assert callable(mod.read_fits)


def test_ishell_module_has_get_header():
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    assert hasattr(mod, 'get_header')
    assert callable(mod.get_header)


def test_ishell_module_has_load_data():
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    assert hasattr(mod, 'load_data')
    assert callable(mod.load_data)


# ---------------------------------------------------------------------------
# Signature compatibility with SpeX / uSpeX interface contract
# ---------------------------------------------------------------------------


def test_read_fits_signature():
    """read_fits must accept the parameters defined in the interface."""
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    sig = inspect.signature(mod.read_fits)
    params = list(sig.parameters)
    for required in ('files', 'linearity_info', 'keywords', 'pair_subtract',
                     'rotate', 'linearity_correction', 'extra', 'verbose'):
        assert required in params, f"read_fits missing parameter '{required}'"


def test_load_data_signature():
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    sig = inspect.signature(mod.load_data)
    params = list(sig.parameters)
    for required in ('file', 'linearity_info', 'keywords', 'coefficients',
                     'linearity_correction'):
        assert required in params, f"load_data missing parameter '{required}'"


def test_get_header_signature():
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    sig = inspect.signature(mod.get_header)
    params = list(sig.parameters)
    for required in ('header', 'keywords'):
        assert required in params, f"get_header missing parameter '{required}'"


# ---------------------------------------------------------------------------
# set_instrument() integration
# ---------------------------------------------------------------------------


def test_set_instrument_ishell():
    """set_instrument('ishell') must set state without raising an exception."""
    set_instrument('ishell')
    assert setup.state['instrument'] == 'ishell'


def test_set_instrument_ishell_lincormax():
    set_instrument('ishell')
    # LINCORMAX = 30000 DN (iSHELL Spextool Manual Table 4, LINCRMAX example)
    assert setup.state['lincormax'] == 30000


def test_set_instrument_ishell_irtf_flag():
    set_instrument('ishell')
    assert setup.state.get('irtf') is True


def test_set_instrument_ishell_suffix():
    set_instrument('ishell')
    assert setup.state['suffix'] == '.[ab]'


def test_set_instrument_ishell_nint():
    set_instrument('ishell')
    # NINT = 3: raw MEF has 3 extensions (signal diff, pedestal sum, signal sum)
    # (iSHELL Spextool Manual §2.3, equation 1)
    assert setup.state['nint'] == 3


def test_set_instrument_ishell_bad_pixel_mask_loaded():
    """The bad-pixel mask must be loadable and have the correct shape."""
    import numpy as np
    set_instrument('ishell')
    mask = setup.state['raw_bad_pixel_mask']
    assert mask is not None
    assert mask.shape == (2048, 2048)


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------


def _ishell_data_path():
    pkg_path = setup.state.get('package_path')
    if pkg_path is None:
        # Resolve from installed package
        from importlib.resources import files as _files
        pkg_path = str(_files('pyspextool'))
    return os.path.join(pkg_path, 'instruments', 'ishell')


@pytest.fixture(autouse=False)
def ishell_instrument(tmp_path):
    """Ensure iSHELL instrument is set before data-file tests."""
    set_instrument('ishell')


def test_ishell_dat_exists():
    set_instrument('ishell')
    dat = os.path.join(_ishell_data_path(), 'ishell.dat')
    assert os.path.isfile(dat), f"ishell.dat not found at {dat}"


def test_telluric_modeinfo_exists():
    set_instrument('ishell')
    f = os.path.join(_ishell_data_path(), 'telluric_modeinfo.dat')
    assert os.path.isfile(f), f"telluric_modeinfo.dat not found at {f}"


def test_telluric_ewadjustments_exists():
    set_instrument('ishell')
    f = os.path.join(_ishell_data_path(), 'telluric_ewadjustments.dat')
    assert os.path.isfile(f), f"telluric_ewadjustments.dat not found at {f}"


def test_telluric_shiftinfo_exists():
    set_instrument('ishell')
    f = os.path.join(_ishell_data_path(), 'telluric_shiftinfo.dat')
    assert os.path.isfile(f), f"telluric_shiftinfo.dat not found at {f}"


def test_ip_coefficients_exists():
    set_instrument('ishell')
    f = os.path.join(_ishell_data_path(), 'IP_coefficients.dat')
    assert os.path.isfile(f), f"IP_coefficients.dat not found at {f}"


@pytest.mark.parametrize('band', ['J', 'H', 'K'])
def test_band_lines_dat_exists(band):
    set_instrument('ishell')
    f = os.path.join(_ishell_data_path(), f'{band}_lines.dat')
    assert os.path.isfile(f), f"{band}_lines.dat not found at {f}"


def test_bad_pixel_mask_fits_exists():
    set_instrument('ishell')
    f = os.path.join(_ishell_data_path(), 'ishell_bdpxmk.fits')
    assert os.path.isfile(f), f"ishell_bdpxmk.fits not found at {f}"


# ---------------------------------------------------------------------------
# Phase 1 behaviour: get_header and load_data are now implemented
# ---------------------------------------------------------------------------


def test_get_header_returns_required_keys():
    """get_header must return all pySpextool-standard output keys."""
    from astropy.io.fits import Header
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')

    hdr = Header()
    hdr['ITIME'] = 30.0
    hdr['CO_ADDS'] = 1
    hdr['DATE_OBS'] = '2024-01-01'
    hdr['TIME_OBS'] = '05:00:00.0'
    hdr['MJD_OBS'] = 60310.208333
    hdr['TCS_RA'] = '05:35:17.3'
    hdr['TCS_DEC'] = '-05:23:28'
    hdr['TCS_HA'] = '00:00:00'
    hdr['TCS_AM'] = 1.05
    hdr['POSANGLE'] = 90.0
    hdr['IRAFNAME'] = 'ishell_0001.fits'
    hdr['PASSBAND'] = 'K1'

    result = mod.get_header(hdr, [])

    required_keys = ['AM', 'HA', 'PA', 'RA', 'DEC', 'ITIME', 'NCOADDS',
                     'IMGITIME', 'TIME', 'DATE', 'MJD', 'FILENAME', 'MODE',
                     'INSTR']
    for key in required_keys:
        assert key in result, f"get_header missing required output key '{key}'"
        assert isinstance(result[key], list) and len(result[key]) == 2, (
            f"get_header['{key}'] must be a [value, comment] list")


def test_get_header_instr_is_ishell():
    """get_header must always set INSTR to 'iSHELL'."""
    from astropy.io.fits import Header
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    result = mod.get_header(Header(), [])
    assert result['INSTR'][0] == 'iSHELL'


def test_get_header_missing_keywords_produce_nan():
    """get_header must substitute nan/nan for missing numeric/string keys."""
    import math as _math
    from astropy.io.fits import Header
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')

    # Completely empty header
    result = mod.get_header(Header(), [])
    assert _math.isnan(result['AM'][0])
    assert _math.isnan(result['PA'][0])
    assert _math.isnan(result['ITIME'][0])
    assert result['HA'][0] == 'nan'
    assert result['RA'][0] == 'nan'
    assert result['DEC'][0] == 'nan'
    assert result['TIME'][0] == 'nan'
    assert result['DATE'][0] == 'nan'
    assert result['FILENAME'][0] == 'unknown'
    assert result['MODE'][0] == 'unknown'
