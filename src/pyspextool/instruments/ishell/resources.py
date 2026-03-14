from functools import lru_cache
from importlib.resources import files
from astropy.io import fits
import yaml


_ISHELL_PACKAGE = "pyspextool.instruments.ishell"
DATA_PACKAGE = "pyspextool.instruments.ishell.data"


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

    path = files(DATA_PACKAGE) / "ishell_lincorr_CDS.fits"

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