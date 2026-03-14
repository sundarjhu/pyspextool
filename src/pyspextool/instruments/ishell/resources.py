from functools import lru_cache
from importlib.resources import files
from astropy.io import fits
import yaml


_ISHELL_PACKAGE = "pyspextool.instruments.ishell"
DATA_PACKAGE = "pyspextool.instruments.ishell.data"

# Detector-level calibration files (all stored in the data/ sub-package).
#
# Placeholder status (as of Phase 0):
#   linearity     – real data (IDL-era cube from Dec 2016; format TBC for Phase 1)
#   bad_pixel_mask – PLACEHOLDER: all-zeros (no real bad-pixel map available yet)
#   hot_pixel_mask – PLACEHOLDER: all-zeros (no real hot-pixel map available yet)
#   bias          – candidate real data (iSHELL arc observation); format TBC
#
# Files that DO NOT belong here (removed):
#   ishell.dat          – top-level pySpextool config; read by set_instrument()
#   xtellcor_modeinfo.dat – legacy IDL xSpextool format; superseded by
#                           ishell/telluric_modeinfo.dat
#
# See docs/ishell_resource_layout.md for a full authoritative-vs-placeholder inventory.
_DETECTOR_FILES = {
    "linearity": "ishell_lincorr_CDS.fits",
    "bad_pixel_mask": "ishell_bdpxmk.fits",
    "hot_pixel_mask": "ishell_htpxmk.fits",
    "bias": "ishell_bias.fits",
}

# Files in data/ that are placeholder stubs awaiting real IRTF calibration data.
# A PLACEHOLDER FITS file contains a COMMENT header card starting with
# "PLACEHOLDER:" and has all-zero data arrays.
_PLACEHOLDER_FILES = frozenset({
    "bad_pixel_mask",
    "hot_pixel_mask",
})


@lru_cache(maxsize=None)
def _load_modes_registry():
    """Parse modes.yaml and return a dict keyed by mode name.

    The result is cached after the first call (thread-safe via lru_cache).
    """
    path = files(_ISHELL_PACKAGE) / "modes.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_registry():
    """Return the iSHELL mode registry as a dict keyed by mode name.

    The registry is loaded from ``modes.yaml`` on first access and cached
    for all subsequent calls.

    Returns
    -------
    dict
        Mapping of mode name → config dict.
    """
    return _load_modes_registry()


def get_mode_config(mode_name):
    """
    Return the configuration dict for the named iSHELL mode.

    Parameters
    ----------
    mode_name : str
        One of the supported iSHELL modes (e.g. ``"J0"``, ``"K3"``).

    Returns
    -------
    dict
        Keys include ``family``, ``wavecal_method``, ``line_list``,
        ``flatinfo``, and ``wavecalinfo``.

    Raises
    ------
    KeyError
        If *mode_name* is not found in the mode registry.
    """
    registry = get_registry()
    if mode_name not in registry:
        raise KeyError(
            f"Unknown iSHELL mode '{mode_name}'. "
            f"Supported modes: {sorted(registry)}"
        )
    return registry[mode_name]


def get_mode_resource(mode_name, resource_key):
    """
    Return an :class:`importlib.resources` path for a packaged resource
    declared in the mode registry.

    Parameters
    ----------
    mode_name : str
        One of the supported iSHELL modes (e.g. ``"J0"``).
    resource_key : str
        A key in the mode config that maps to a filename, e.g.
        ``"line_list"``, ``"flatinfo"``, or ``"wavecalinfo"``.

    Returns
    -------
    importlib.resources.abc.Traversable
        A path object pointing to the packaged file.  Call ``.is_file()``
        to verify it exists; pass it to ``astropy.io.fits.open()`` directly.

    Raises
    ------
    KeyError
        If *mode_name* or *resource_key* is not found in the registry.
    """
    config = get_mode_config(mode_name)
    if resource_key not in config:
        raise KeyError(
            f"Mode '{mode_name}' has no resource key '{resource_key}'. "
            f"Available keys: {sorted(config)}"
        )
    filename = config[resource_key]
    return files(DATA_PACKAGE) / filename


def get_linearity_cube():
    """
    Load the iSHELL detector linearity correction cube.

    Returns
    -------
    astropy.io.fits.HDUList

    Raises
    ------
    RuntimeError
        If the calibration file is missing or replaced by a Git LFS pointer.
    """

    path = get_detector_resource("linearity")

    if not path.is_file():
        raise RuntimeError(
            "Missing iSHELL linearity calibration file "
            "(ishell_lincorr_CDS.fits). "
            "If you cloned the repository, run `git lfs pull`."
        )

    # Detect Git LFS pointer file
    with path.open("rb") as f:
        start = f.read(64)

    if start.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            "The iSHELL linearity calibration file is a Git LFS pointer.\n"
            "Install Git LFS and run:\n\n"
            "    git lfs pull\n"
        )

    return fits.open(path)


def get_detector_resource(resource_key):
    """
    Return an :class:`importlib.resources` path for a detector-level
    calibration file.

    Parameters
    ----------
    resource_key : str
        One of ``"linearity"``, ``"bad_pixel_mask"``, ``"hot_pixel_mask"``,
        or ``"bias"``.

    Returns
    -------
    importlib.resources.abc.Traversable
        A path object pointing to the packaged file.  Call ``.is_file()``
        to verify it exists; pass it to ``astropy.io.fits.open()`` directly.

    Raises
    ------
    KeyError
        If *resource_key* is not a known detector resource.
    """
    if resource_key not in _DETECTOR_FILES:
        raise KeyError(
            f"Unknown detector resource '{resource_key}'. "
            f"Available keys: {sorted(_DETECTOR_FILES)}"
        )
    return files(DATA_PACKAGE) / _DETECTOR_FILES[resource_key]


def is_placeholder_resource(resource_key):
    """
    Return True if *resource_key* refers to a placeholder detector resource.

    Placeholder files exist only so the resource API resolves without error.
    They contain all-zero data and a ``PLACEHOLDER:`` FITS COMMENT header
    card.  They must be replaced with real calibration data before
    production use.

    Parameters
    ----------
    resource_key : str
        One of ``"linearity"``, ``"bad_pixel_mask"``, ``"hot_pixel_mask"``,
        or ``"bias"``.

    Returns
    -------
    bool
        ``True`` if the resource is a known placeholder; ``False`` otherwise.

    Raises
    ------
    KeyError
        If *resource_key* is not a known detector resource.
    """
    if resource_key not in _DETECTOR_FILES:
        raise KeyError(
            f"Unknown detector resource '{resource_key}'. "
            f"Available keys: {sorted(_DETECTOR_FILES)}"
        )
    return resource_key in _PLACEHOLDER_FILES


def get_bad_pixel_mask():
    """
    Load the iSHELL bad-pixel mask.

    Returns
    -------
    astropy.io.fits.HDUList

    Raises
    ------
    OSError
        If the file cannot be opened (e.g. missing from the package).
    """
    return fits.open(get_detector_resource("bad_pixel_mask"))


def get_hot_pixel_mask():
    """
    Load the iSHELL hot-pixel mask.

    Returns
    -------
    astropy.io.fits.HDUList

    Raises
    ------
    OSError
        If the file cannot be opened (e.g. missing from the package).
    """
    return fits.open(get_detector_resource("hot_pixel_mask"))


def get_bias():
    """
    Load the iSHELL bias frame.

    Returns
    -------
    astropy.io.fits.HDUList

    Raises
    ------
    OSError
        If the file cannot be opened (e.g. missing from the package).
    """
    return fits.open(get_detector_resource("bias"))