"""
Tests for the iSHELL mode registry (modes.yaml) and resources.py helpers.

These tests verify:
  - the registry loads successfully,
  - every supported mode is present,
  - each mode has the required keys,
  - every declared resource path resolves to an existing packaged file.
"""

import pytest

from pyspextool.instruments.ishell.resources import (
    get_mode_config,
    get_mode_resource,
    get_registry,
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
