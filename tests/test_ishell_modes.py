"""
Tests for the iSHELL mode registry (modes.yaml) and resources.py helpers.

These tests verify:
  - the registry loads successfully,
  - every supported mode is present,
  - each mode has the required keys,
  - every declared resource path resolves to an existing packaged file,
  - detector-level resource paths resolve correctly,
  - detector FITS loaders return valid HDUList objects,
  - stray legacy files have been removed from data/,
  - placeholder FITS files carry the expected PLACEHOLDER FITS comment.
"""

import pytest
from astropy.io import fits
from importlib.resources import files as _res_files

from pyspextool.instruments.ishell.resources import (
    DATA_PACKAGE,
    get_bad_pixel_mask,
    get_bias,
    get_detector_resource,
    get_hot_pixel_mask,
    get_mode_config,
    get_mode_resource,
    get_registry,
    is_placeholder_resource,
)


SUPPORTED_MODES = [
    "J0", "J1", "J2", "J3",
    "H1", "H2", "H3",
    "K1", "K2", "K3", "Kgas",
]

REQUIRED_KEYS = {"family", "wavecal_method", "line_list", "flatinfo", "wavecalinfo"}

RESOURCE_KEYS = ("line_list", "flatinfo", "wavecalinfo")


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def test_registry_loads():
    """The modes registry must load without error and return a non-empty dict."""
    registry = get_registry()
    assert isinstance(registry, dict)
    assert len(registry) > 0


def test_registry_contains_all_modes():
    """Every supported mode must be present in the registry."""
    registry = get_registry()
    for mode in SUPPORTED_MODES:
        assert mode in registry, f"Mode '{mode}' missing from registry"


def test_registry_has_no_extra_modes():
    """Registry must contain exactly the supported modes (no stray entries)."""
    registry = get_registry()
    assert set(registry) == set(SUPPORTED_MODES)


# ---------------------------------------------------------------------------
# Per-mode configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_mode_config_has_required_keys(mode):
    """Each mode must expose all required configuration keys."""
    config = get_mode_config(mode)
    for key in REQUIRED_KEYS:
        assert key in config, f"Mode '{mode}' missing key '{key}'"


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_mode_family_is_valid(mode):
    """Each mode's family must be one of J, H, K."""
    config = get_mode_config(mode)
    assert config["family"] in {"J", "H", "K"}, (
        f"Mode '{mode}' has unexpected family '{config['family']}'"
    )


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_mode_wavecal_method(mode):
    """All J/H/K modes must use wavecal_method 'thar'."""
    config = get_mode_config(mode)
    assert config["wavecal_method"] == "thar", (
        f"Mode '{mode}' has unexpected wavecal_method '{config['wavecal_method']}'"
    )


def test_get_mode_config_unknown_mode():
    """get_mode_config must raise KeyError for an unrecognised mode name."""
    with pytest.raises(KeyError, match="Unknown iSHELL mode"):
        get_mode_config("NOTAMODE")


def test_get_mode_resource_unknown_key():
    """get_mode_resource must raise KeyError for an unrecognised resource key."""
    with pytest.raises(KeyError):
        get_mode_resource("J0", "nonexistent_key")


# ---------------------------------------------------------------------------
# Packaged resource resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
@pytest.mark.parametrize("resource_key", RESOURCE_KEYS)
def test_mode_resource_file_exists(mode, resource_key):
    """Every resource declared in the registry must resolve to a real file."""
    resource_path = get_mode_resource(mode, resource_key)
    assert resource_path.is_file(), (
        f"Resource '{resource_key}' for mode '{mode}' not found at {resource_path}"
    )


# ---------------------------------------------------------------------------
# Family groupings
# ---------------------------------------------------------------------------


def test_j_modes_have_j_family():
    for mode in ("J0", "J1", "J2", "J3"):
        assert get_mode_config(mode)["family"] == "J"


def test_h_modes_have_h_family():
    for mode in ("H1", "H2", "H3"):
        assert get_mode_config(mode)["family"] == "H"


def test_k_modes_have_k_family():
    for mode in ("K1", "K2", "K3", "Kgas"):
        assert get_mode_config(mode)["family"] == "K"


# ---------------------------------------------------------------------------
# Detector-level resource paths
# ---------------------------------------------------------------------------

DETECTOR_RESOURCE_KEYS = ("linearity", "bad_pixel_mask", "hot_pixel_mask", "bias")


@pytest.mark.parametrize("resource_key", DETECTOR_RESOURCE_KEYS)
def test_detector_resource_file_exists(resource_key):
    """Every detector resource must resolve to a real packaged file."""
    path = get_detector_resource(resource_key)
    assert path.is_file(), (
        f"Detector resource '{resource_key}' not found at {path}"
    )


def test_get_detector_resource_unknown_key():
    """get_detector_resource must raise KeyError for an unknown resource key."""
    with pytest.raises(KeyError, match="Unknown detector resource"):
        get_detector_resource("nonexistent_resource")


# ---------------------------------------------------------------------------
# Detector FITS loaders
# ---------------------------------------------------------------------------


def test_get_bad_pixel_mask_returns_hdulist():
    """get_bad_pixel_mask must return a valid HDUList."""
    with get_bad_pixel_mask() as hdul:
        assert isinstance(hdul, fits.HDUList)


def test_get_hot_pixel_mask_returns_hdulist():
    """get_hot_pixel_mask must return a valid HDUList."""
    with get_hot_pixel_mask() as hdul:
        assert isinstance(hdul, fits.HDUList)


def test_get_bias_returns_hdulist():
    """get_bias must return a valid HDUList."""
    with get_bias() as hdul:
        assert isinstance(hdul, fits.HDUList)


def test_bad_pixel_mask_shape():
    """Bad-pixel mask must match the iSHELL detector size (2048×2048)."""
    with get_bad_pixel_mask() as hdul:
        assert hdul[0].data.shape == (2048, 2048)


def test_hot_pixel_mask_shape():
    """Hot-pixel mask must match the iSHELL detector size (2048×2048)."""
    with get_hot_pixel_mask() as hdul:
        assert hdul[0].data.shape == (2048, 2048)


def test_bias_shape():
    """Bias frame must match the iSHELL detector size (2048×2048)."""
    with get_bias() as hdul:
        assert hdul[0].data.shape == (2048, 2048)


# ---------------------------------------------------------------------------
# Stray legacy file removal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy_file", [
    "ishell.dat",
    "xtellcor_modeinfo.dat",
])
def test_no_stray_legacy_files_in_data(legacy_file):
    """Stray legacy IDL/xSpextool files must not exist in the data/ sub-package.

    These files used IDL-era formats (NINT=5, XSPEXTOOL_KEYWORD, old xtellcor
    column layout) incompatible with pySpextool and were removed in the Phase 0
    scaffold cleanup.  See docs/ishell_resource_layout.md §4.
    """
    path = _res_files(DATA_PACKAGE) / legacy_file
    assert not path.is_file(), (
        f"Stray legacy file '{legacy_file}' found in data/.  "
        "It should have been removed during the Phase 0 scaffold cleanup."
    )


# ---------------------------------------------------------------------------
# Placeholder FITS file header validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resource_key", ["bad_pixel_mask", "hot_pixel_mask"])
def test_placeholder_fits_has_placeholder_comment(resource_key):
    """Placeholder detector FITS files must carry a PLACEHOLDER FITS COMMENT.

    This ensures placeholder stubs cannot be silently mistaken for real
    calibration data.  The COMMENT must start with the string 'PLACEHOLDER:'.
    See docs/ishell_resource_layout.md §3.
    """
    path = get_detector_resource(resource_key)
    with fits.open(str(path)) as hdul:
        comments = [str(c) for c in hdul[0].header.get("COMMENT", [])]
    assert any(c.startswith("PLACEHOLDER:") for c in comments), (
        f"Detector resource '{resource_key}' has no 'PLACEHOLDER:' COMMENT in its "
        "FITS header.  Add one so users know this is not real calibration data."
    )


@pytest.mark.parametrize("resource_key", ["bad_pixel_mask", "hot_pixel_mask"])
def test_placeholder_fits_is_all_zeros(resource_key):
    """Placeholder detector FITS files must have all-zero data arrays.

    A non-zero placeholder mask could cause incorrect bad-pixel flagging
    if accidentally used in production.
    """
    import numpy as np

    path = get_detector_resource(resource_key)
    with fits.open(str(path)) as hdul:
        data = hdul[0].data
    assert np.all(data == 0), (
        f"Placeholder '{resource_key}' has non-zero pixels "
        f"({np.sum(data != 0)} bad/hot pixels).  "
        "Replace the corrupt placeholder with an all-zeros stub."
    )


# ---------------------------------------------------------------------------
# is_placeholder_resource() API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resource_key", ["bad_pixel_mask", "hot_pixel_mask"])
def test_is_placeholder_returns_true_for_stubs(resource_key):
    """is_placeholder_resource() must return True for known placeholder stubs."""
    assert is_placeholder_resource(resource_key) is True


@pytest.mark.parametrize("resource_key", ["linearity", "bias"])
def test_is_placeholder_returns_false_for_real_data(resource_key):
    """is_placeholder_resource() must return False for non-placeholder resources."""
    assert is_placeholder_resource(resource_key) is False


def test_is_placeholder_raises_for_unknown_key():
    """is_placeholder_resource() must raise KeyError for unknown keys."""
    with pytest.raises(KeyError):
        is_placeholder_resource("nonexistent_resource")
