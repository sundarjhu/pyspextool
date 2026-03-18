"""
Utility helpers for FITS file discovery and path handling in iSHELL workflows.

All iSHELL input files may be either plain ``.fits`` or gzip-compressed
``.fits.gz``.  Output files are always written as plain ``.fits``.

Functions
---------
is_fits_file(path)
    Return ``True`` if *path* has a ``.fits`` or ``.fits.gz`` suffix.
strip_fits_suffix(path)
    Remove ``.fits`` or ``.fits.gz`` from *path*, returning a plain string.
ensure_fits_suffix(path)
    Guarantee that the returned path ends with ``.fits`` (outputs only).
find_fits_files(directory)
    Return a sorted list of :class:`pathlib.Path` objects for every
    ``.fits`` and ``.fits.gz`` file in *directory*.
split_fits_path(path)
    Return ``(stem, suffix)`` where *suffix* is ``".fits"`` or
    ``".fits.gz"``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "is_fits_file",
    "strip_fits_suffix",
    "ensure_fits_suffix",
    "find_fits_files",
    "split_fits_path",
]

# Recognised FITS suffixes, ordered so that the compound suffix is tested first.
_FITS_SUFFIXES = (".fits.gz", ".fits")


def is_fits_file(path: str | Path) -> bool:
    """Return ``True`` if *path* ends with ``.fits`` or ``.fits.gz``.

    Parameters
    ----------
    path : str or Path
        File name or full path to test.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_fits_file("frame.fits")
    True
    >>> is_fits_file("frame.fits.gz")
    True
    >>> is_fits_file("frame.txt")
    False
    """
    name = Path(path).name
    return name.endswith(".fits.gz") or name.endswith(".fits")


def strip_fits_suffix(path: str | Path) -> str:
    """Remove ``.fits`` or ``.fits.gz`` from *path*.

    Returns the portion of the path before the FITS suffix as a string.
    If the path does not end with a recognised FITS suffix it is returned
    unchanged.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    str

    Examples
    --------
    >>> strip_fits_suffix("/data/frame.fits")
    '/data/frame'
    >>> strip_fits_suffix("/data/frame.fits.gz")
    '/data/frame'
    >>> strip_fits_suffix("/data/frame.txt")
    '/data/frame.txt'
    """
    p = Path(path)
    name = p.name
    if name.endswith(".fits.gz"):
        stem = name[: -len(".fits.gz")]
    elif name.endswith(".fits"):
        stem = name[: -len(".fits")]
    else:
        return str(path)
    return str(p.parent / stem)


def ensure_fits_suffix(path: str | Path) -> str:
    """Return *path* guaranteed to end with ``.fits``.

    Use this for output paths.  ``.fits.gz`` inputs are converted to
    ``.fits``; paths that already end with ``.fits`` are returned unchanged;
    all other paths have ``.fits`` appended.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    str

    Examples
    --------
    >>> ensure_fits_suffix("/out/result")
    '/out/result.fits'
    >>> ensure_fits_suffix("/out/result.fits")
    '/out/result.fits'
    >>> ensure_fits_suffix("/out/result.fits.gz")
    '/out/result.fits'
    """
    p = Path(path)
    name = p.name
    if name.endswith(".fits.gz"):
        name = name[: -len(".gz")]
        return str(p.parent / name)
    if name.endswith(".fits"):
        return str(p)
    return str(p.parent / (name + ".fits"))


def find_fits_files(directory: str | Path) -> list[Path]:
    """Return a sorted list of ``.fits`` and ``.fits.gz`` files in *directory*.

    Only immediate children of *directory* are considered (non-recursive).

    Parameters
    ----------
    directory : str or Path

    Returns
    -------
    list of Path

    Examples
    --------
    >>> paths = find_fits_files("data/raw")
    >>> [p.name for p in paths]
    ['arc.fits', 'flat.fits.gz', 'flat2.fits']
    """
    d = Path(directory)
    return sorted(p for p in d.iterdir() if p.is_file() and is_fits_file(p))


def split_fits_path(path: str | Path) -> tuple[str, str]:
    """Return ``(stem, suffix)`` for a FITS file path.

    *stem* is the full path with the FITS suffix removed (as a string).
    *suffix* is either ``".fits"`` or ``".fits.gz"``.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    stem : str
    suffix : str

    Raises
    ------
    ValueError
        If *path* does not end with a recognised FITS suffix.

    Examples
    --------
    >>> split_fits_path("/data/frame.fits")
    ('/data/frame', '.fits')
    >>> split_fits_path("/data/frame.fits.gz")
    ('/data/frame', '.fits.gz')
    """
    p = Path(path)
    name = p.name
    if name.endswith(".fits.gz"):
        stem = str(p.parent / name[: -len(".fits.gz")])
        return stem, ".fits.gz"
    if name.endswith(".fits"):
        stem = str(p.parent / name[: -len(".fits")])
        return stem, ".fits"
    raise ValueError(f"Not a recognised FITS file path: {path!r}")
