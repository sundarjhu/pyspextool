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
# NotImplementedError is raised (Phase 0 behaviour)
# ---------------------------------------------------------------------------


def test_get_header_raises_not_implemented():
    """get_header must raise NotImplementedError until Phase 1 is complete."""
    from astropy.io.fits import Header
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    with pytest.raises(NotImplementedError):
        mod.get_header(Header(), [])


def test_load_data_raises_not_implemented():
    """load_data must raise NotImplementedError until Phase 1 is complete."""
    mod = importlib.import_module('pyspextool.instruments.ishell.ishell')
    with pytest.raises(NotImplementedError):
        mod.load_data('dummy.fits', {'max': 55000, 'bit': 0}, [])
