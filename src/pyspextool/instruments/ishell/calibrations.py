"""
Typed readers and validation for iSHELL packaged calibration resources.

This module provides:

* Dataclasses representing each calibration resource type.
* Reader functions that parse the packaged files into those dataclasses.
* Validation helpers that check file existence, basic structure, and
  dimensional consistency.
* A convenience function :func:`load_mode_calibrations` that loads all
  resources for a given mode in one call.

Runtime-critical resources
--------------------------
The following resources are needed by reduction stages and are marked
as runtime-critical:

* ``*_lines.dat`` – ThAr arc-line lists used for wavelength calibration.
* ``*_flatinfo.fits`` – flat-field order-trace calibration images.
* ``*_wavecalinfo.fits`` – wavelength-calibration metadata and spectral arrays.
* ``ishell_lincorr_CDS.fits`` – detector linearity-correction cube.
* ``ishell_bdpxmk.fits`` / ``ishell_htpxmk.fits`` – bad/hot-pixel masks
  (currently placeholders: all-zero).
* ``ishell_bias.fits`` – bias frame (candidate real data).

Partially-understood formats
-----------------------------
* ``*_flatinfo.fits`` – The image data array and per-order polynomial
  coefficients (OR{n}_B1..B5, OR{n}_T1..T5) are present but their
  precise semantics (edge positions, tilt polynomials, etc.) require
  further IDL-to-Python translation.
* ``*_wavecalinfo.fits`` – Data cube shape is (NORDERS, 4, n_pixels).
  The four planes are spectral arrays; their exact labelling is not yet
  fully documented.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.io import fits

from .resources import (
    get_detector_resource,
    get_mode_resource,
    is_placeholder_resource,
)
from .geometry import (  # noqa: F401 – re-exported for convenience
    OrderGeometry,
    OrderGeometrySet,
    RectificationMap,
    build_order_geometry_set,
)

__all__ = [
    "LineListEntry",
    "LineList",
    "FlatInfo",
    "WaveCalInfo",
    "LinearityCube",
    "PixelMask",
    "BiasFrame",
    "ModeCalibrations",
    "read_line_list",
    "read_flatinfo",
    "read_wavecalinfo",
    "read_linearity_cube",
    "read_pixel_mask",
    "read_bias",
    "load_mode_calibrations",
    # Geometry data-model (iSHELL order geometry support)
    "OrderGeometry",
    "OrderGeometrySet",
    "RectificationMap",
    "build_order_geometry_set",
]

# iSHELL detector dimensions (Teledyne H2RG)
DETECTOR_NROWS = 2048
DETECTOR_NCOLS = 2048
_DETECTOR_SHAPE = (DETECTOR_NROWS, DETECTOR_NCOLS)

# Git LFS magic bytes at the start of a pointer file
_LFS_MAGIC = b"version https://git-lfs"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_lfs(path) -> None:
    """Raise :class:`RuntimeError` if *path* is a Git LFS pointer file."""
    with path.open("rb") as fh:
        head = fh.read(64)
    if head.startswith(_LFS_MAGIC):
        raise RuntimeError(
            f"The file '{path}' is a Git LFS pointer, not the actual data.\n"
            "Install Git LFS and run:\n\n"
            "    git lfs pull\n"
        )


def _check_file_exists(path, label: str) -> None:
    """Raise :class:`FileNotFoundError` if *path* does not exist."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing packaged calibration file for {label}: {path}"
        )


def _open_fits(path, label: str) -> fits.HDUList:
    """Open a FITS file after existence and LFS checks."""
    _check_file_exists(path, label)
    _check_lfs(path)
    return fits.open(str(path))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineListEntry:
    """A single entry in a ThAr arc-line list.

    Parameters
    ----------
    order : int
        Echelle order number.
    wavelength_um : float
        Vacuum wavelength in microns.
    species : str
        Atomic or ionic species (e.g. ``"Th I"``, ``"Ar II"``).
    fit_window_angstrom : float
        Half-width of the fitting window in Ångströms.
    fit_type : str
        Profile shape used for fitting (``"G"`` = Gaussian).
    fit_n_terms : int
        Number of polynomial terms used in the fit.
    """

    order: int
    wavelength_um: float
    species: str
    fit_window_angstrom: float
    fit_type: str
    fit_n_terms: int


@dataclass
class LineList:
    """ThAr emission-line list for one iSHELL mode.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"J0"``).
    entries : list of :class:`LineListEntry`
        All arc lines for this mode, in file order.
    """

    mode: str
    entries: list[LineListEntry] = field(default_factory=list)

    @property
    def n_lines(self) -> int:
        """Total number of lines in the list."""
        return len(self.entries)

    @property
    def orders(self) -> list[int]:
        """Sorted unique order numbers present in the list."""
        return sorted(set(e.order for e in self.entries))

    @property
    def wavelengths_um(self) -> np.ndarray:
        """Vacuum wavelengths (μm) as a 1-D NumPy array."""
        return np.array([e.wavelength_um for e in self.entries])


@dataclass
class FlatInfo:
    """Flat-field order-trace calibration for one iSHELL mode.

    Parameters
    ----------
    mode : str
        iSHELL mode name.
    orders : list of int
        Echelle order numbers present in the flat.
    rotation : int
        Detector rotation code (pySpextool convention).
    plate_scale_arcsec : float
        Spatial plate scale in arcsec/pixel.
    slit_height_pixels : float
        Physical slit height in pixels.
    slit_height_arcsec : float
        Physical slit height in arcsec.
    slit_range_pixels : tuple of (int, int)
        Min/max useful slit extent in pixels.
    resolving_power_pixel : float
        Approximate spectral resolving power per pixel.
    step : int
        Column step size used during order tracing.
    flat_fraction : float
        Fraction of peak flux used to define order edges.
    comm_window : int
        Common-region window size in pixels.
    image : numpy.ndarray, shape (2048, 2048)
        Flat-field detector image.
    xranges : numpy.ndarray, shape (n_orders, 2) or None
        Per-order column ranges ``[x_start, x_end]`` parsed from the
        ``OR{n}_XR`` header keywords.  ``None`` if the header does not
        contain these keywords.
    edge_coeffs : numpy.ndarray, shape (n_orders, 2, n_terms) or None
        Per-order edge polynomial coefficients.  ``edge_coeffs[i, 0, :]``
        is the bottom-edge polynomial for order *i*; ``edge_coeffs[i, 1,
        :]`` is the top-edge polynomial.  Coefficients follow the
        ``numpy.polynomial.polynomial`` convention (constant term first).
        ``None`` if the header does not contain ``OR{n}_B1…`` keywords.
    edge_degree : int or None
        Polynomial degree of the edge fits (``EDGEDEG`` header keyword).
        ``None`` if the keyword is absent.
    """

    mode: str
    orders: list[int]
    rotation: int
    plate_scale_arcsec: float
    slit_height_pixels: float
    slit_height_arcsec: float
    slit_range_pixels: tuple[int, int]
    resolving_power_pixel: float
    step: int
    flat_fraction: float
    comm_window: int
    image: np.ndarray
    xranges: Optional[np.ndarray] = None
    edge_coeffs: Optional[np.ndarray] = None
    edge_degree: Optional[int] = None

    @property
    def n_orders(self) -> int:
        """Number of echelle orders in the flat."""
        return len(self.orders)


@dataclass
class WaveCalInfo:
    """Calibration metadata and data cube read from ``*_wavecalinfo.fits``.

    Parameters
    ----------
    mode : str
        iSHELL mode name.
    n_orders : int
        Number of echelle orders.
    orders : list of int
        Echelle order numbers.
    resolving_power : float
        Nominal spectral resolving power.
    data : numpy.ndarray, shape (n_orders, 4, n_pixels)
        Data cube read from the FITS file.  The plane labelling is **not
        fully documented** in the packaged files.  Based on structural
        inspection: plane 0 contains values inferred to be wavelengths in
        µm (range-checked against J/H/K bands); planes 1–3 are not used
        and their meanings are unconfirmed.  Pixels outside the valid
        column range for each order are set to NaN.
    linelist_name : str
        Filename of the arc-line list referenced by this calibration.
    wcal_type : str
        Calibration type string from the FITS header (e.g. ``"2DXD"``).
    home_order : int
        Reference order from the FITS header (``HOMEORDR``).
    disp_degree : int
        Dispersion polynomial degree from the FITS header (``DISPDEG``).
    order_degree : int
        Cross-order polynomial degree from the FITS header (``ORDRDEG``).
    xranges : numpy.ndarray, shape (n_orders, 2) or None
        Per-order column ranges ``[x_start, x_end]`` parsed from the
        ``OR{n}_XR`` header keywords.  ``None`` if the header does not
        contain these keywords.
    """

    mode: str
    n_orders: int
    orders: list[int]
    resolving_power: float
    data: np.ndarray
    linelist_name: str
    wcal_type: str
    home_order: int
    disp_degree: int
    order_degree: int
    xranges: Optional[np.ndarray] = None

    @property
    def n_pixels(self) -> int:
        """Number of pixels along the dispersion axis."""
        return self.data.shape[2]


@dataclass
class LinearityCube:
    """iSHELL detector linearity-correction cube.

    Parameters
    ----------
    data : numpy.ndarray, shape (n_coeffs, 2048, 2048)
        Pixel-by-pixel polynomial coefficients for the linearity correction.
    dn_lower_limit : float
        Minimum valid signal level in DN.
    dn_upper_limit : float
        Maximum valid signal level in DN (above which saturation flagging
        is applied).
    fit_order : int
        Degree of the linearity polynomial (``n_coeffs - 1``).
    n_pedestals : int
        Number of pedestal samples used when building the correction.
    """

    data: np.ndarray
    dn_lower_limit: float
    dn_upper_limit: float
    fit_order: int
    n_pedestals: int

    @property
    def n_coeffs(self) -> int:
        """Number of polynomial coefficients (fit_order + 1)."""
        return self.data.shape[0]


@dataclass
class PixelMask:
    """iSHELL bad-pixel or hot-pixel mask.

    Parameters
    ----------
    mask_type : str
        Either ``"bad"`` or ``"hot"``.
    data : numpy.ndarray, shape (2048, 2048)
        Mask array.  Zero = good pixel, non-zero = flagged pixel.
    is_placeholder : bool
        ``True`` if this mask is an all-zero placeholder stub and has not
        yet been replaced with real IRTF calibration data.
    """

    mask_type: str
    data: np.ndarray
    is_placeholder: bool


@dataclass
class BiasFrame:
    """iSHELL bias frame.

    Parameters
    ----------
    data : numpy.ndarray, shape (2048, 2048)
        Bias image.
    divisor : int
        Number of raw bias frames co-added before storage (the stored
        frame has already been divided by this value).
    """

    data: np.ndarray
    divisor: int


@dataclass
class ModeCalibrations:
    """All packaged calibration resources for one iSHELL mode.

    Attributes are set by :func:`load_mode_calibrations`.

    Parameters
    ----------
    mode : str
        iSHELL mode name.
    line_list : :class:`LineList`
    flatinfo : :class:`FlatInfo`
    wavecalinfo : :class:`WaveCalInfo`
    """

    mode: str
    line_list: LineList
    flatinfo: FlatInfo
    wavecalinfo: WaveCalInfo


# ---------------------------------------------------------------------------
# FITS header parsing helpers
# ---------------------------------------------------------------------------


def _parse_order_xranges(
    hdr, orders: list[int]
) -> Optional[np.ndarray]:
    """Parse per-order column ranges from ``OR{n}_XR`` header keywords.

    Parameters
    ----------
    hdr : astropy FITS header
    orders : list of int

    Returns
    -------
    ndarray, shape (n_orders, 2) or None
        Column ranges ``[x_start, x_end]`` for each order, or ``None``
        if no ``OR{n}_XR`` keywords are present.
    """
    xranges = []
    for o in orders:
        key = f"OR{o}_XR"
        if key not in hdr:
            return None
        try:
            parts = [int(x) for x in str(hdr[key]).split(",")]
            xranges.append(parts[:2])
        except (ValueError, IndexError):
            return None
    return np.array(xranges, dtype=int)


def _parse_order_edge_coeffs(
    hdr, orders: list[int]
) -> Optional[np.ndarray]:
    """Parse per-order edge polynomial coefficients from the FITS header.

    Reads ``OR{n}_B1`` … ``OR{n}_Bk`` (bottom edge) and ``OR{n}_T1``
    … ``OR{n}_Tk`` (top edge) keywords.  The number of terms *k* is
    determined from the ``EDGEDEG`` keyword (``n_terms = EDGEDEG + 1``).

    Returns
    -------
    ndarray, shape (n_orders, 2, n_terms) or None
        Edge polynomial coefficients in ``numpy.polynomial.polynomial``
        convention (constant term at index 0).  Returns ``None`` if
        ``EDGEDEG`` is absent or any expected keyword is missing.
    """
    if "EDGEDEG" not in hdr:
        return None
    n_terms = int(hdr["EDGEDEG"]) + 1
    result = []
    for o in orders:
        bottom = []
        top = []
        for k in range(1, n_terms + 1):
            bkey = f"OR{o}_B{k}"
            tkey = f"OR{o}_T{k}"
            if bkey not in hdr or tkey not in hdr:
                return None
            bottom.append(float(hdr[bkey]))
            top.append(float(hdr[tkey]))
        result.append([bottom, top])
    return np.array(result, dtype=float)


# ---------------------------------------------------------------------------
# Reader functions
# ---------------------------------------------------------------------------


def read_line_list(mode_name: str) -> LineList:
    """Parse the ThAr arc-line list for *mode_name*.

    Parameters
    ----------
    mode_name : str
        A supported iSHELL mode (e.g. ``"J0"``).

    Returns
    -------
    :class:`LineList`

    Raises
    ------
    KeyError
        If *mode_name* is not in the mode registry.
    FileNotFoundError
        If the packaged ``.dat`` file is missing.
    RuntimeError
        If the packaged file is a Git LFS pointer (run ``git lfs pull``).
    ValueError
        If the file cannot be parsed (unexpected column count or format).
    """
    path = get_mode_resource(mode_name, "line_list")
    label = f"line_list for mode '{mode_name}'"
    _check_file_exists(path, label)
    _check_lfs(path)

    entries: list[LineListEntry] = []

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 6:
                raise ValueError(
                    f"Unexpected number of columns in {path}: "
                    f"expected 6 pipe-separated fields, got {len(parts)} "
                    f"in line: {raw_line!r}"
                )
            try:
                entry = LineListEntry(
                    order=int(parts[0]),
                    wavelength_um=float(parts[1]),
                    species=parts[2],
                    fit_window_angstrom=float(parts[3]),
                    fit_type=parts[4],
                    fit_n_terms=int(parts[5]),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse line in {path}: {raw_line!r}"
                ) from exc
            entries.append(entry)

    if not entries:
        raise ValueError(
            f"Line list for mode '{mode_name}' is empty (no data rows found)."
        )

    return LineList(mode=mode_name, entries=entries)


def read_flatinfo(mode_name: str) -> FlatInfo:
    """Load the flat-field calibration for *mode_name*.

    Parameters
    ----------
    mode_name : str
        A supported iSHELL mode.

    Returns
    -------
    :class:`FlatInfo`

    Raises
    ------
    KeyError
        If *mode_name* is not in the mode registry.
    FileNotFoundError
        If the packaged FITS file is missing.
    RuntimeError
        If the file is a Git LFS pointer.
    ValueError
        If required header keywords are missing or the image shape is
        inconsistent with the detector dimensions.
    """
    path = get_mode_resource(mode_name, "flatinfo")
    label = f"flatinfo for mode '{mode_name}'"

    with _open_fits(path, label) as hdul:
        hdu = hdul[0]
        hdr = hdu.header
        image = hdu.data.copy()

    # Validate detector shape
    if image.shape != _DETECTOR_SHAPE:
        raise ValueError(
            f"flatinfo for mode '{mode_name}' has unexpected image shape "
            f"{image.shape}; expected {_DETECTOR_SHAPE}."
        )

    # Parse required header keywords
    missing = [k for k in ("ORDERS", "ROTATION") if k not in hdr]
    if missing:
        raise ValueError(
            f"flatinfo for mode '{mode_name}' is missing required header "
            f"keywords: {missing}"
        )

    try:
        orders = [int(o) for o in hdr["ORDERS"].split(",")]
    except ValueError as exc:
        raise ValueError(
            f"flatinfo for mode '{mode_name}': cannot parse ORDERS header "
            f"value {hdr['ORDERS']!r} as a comma-separated list of integers."
        ) from exc

    # Parse optional-but-expected keywords.
    # Floats default to NaN (signals "not available" to callers without
    # masking real zero values).  Integer counts default to 0.
    slit_range_raw = hdr.get("SLTH_RNG", "0,0")
    try:
        slit_range = tuple(int(x) for x in slit_range_raw.split(","))
    except ValueError as exc:
        raise ValueError(
            f"flatinfo for mode '{mode_name}': cannot parse SLTH_RNG header "
            f"value {slit_range_raw!r} as a comma-separated pair of integers."
        ) from exc

    return FlatInfo(
        mode=mode_name,
        orders=orders,
        rotation=int(hdr["ROTATION"]),
        plate_scale_arcsec=float(hdr.get("PLTSCALE", float("nan"))),
        slit_height_pixels=float(hdr.get("SLTH_PIX", float("nan"))),
        slit_height_arcsec=float(hdr.get("SLTH_ARC", float("nan"))),
        slit_range_pixels=slit_range,
        resolving_power_pixel=float(hdr.get("RPPIX", float("nan"))),
        step=int(hdr.get("STEP", 0)),
        flat_fraction=float(hdr.get("FLATFRAC", float("nan"))),
        comm_window=int(hdr.get("COMWIN", 0)),
        image=image,
        xranges=_parse_order_xranges(hdr, orders),
        edge_coeffs=_parse_order_edge_coeffs(hdr, orders),
        edge_degree=int(hdr["EDGEDEG"]) if "EDGEDEG" in hdr else None,
    )


def read_wavecalinfo(mode_name: str) -> WaveCalInfo:
    """Load the wavelength-calibration metadata for *mode_name*.

    Parameters
    ----------
    mode_name : str
        A supported iSHELL mode.

    Returns
    -------
    :class:`WaveCalInfo`

    Raises
    ------
    KeyError
        If *mode_name* is not in the mode registry.
    FileNotFoundError
        If the packaged FITS file is missing.
    RuntimeError
        If the file is a Git LFS pointer.
    ValueError
        If required header keywords are missing or the data cube shape is
        inconsistent with the declared number of orders.
    """
    path = get_mode_resource(mode_name, "wavecalinfo")
    label = f"wavecalinfo for mode '{mode_name}'"

    with _open_fits(path, label) as hdul:
        hdu = hdul[0]
        hdr = hdu.header
        data = hdu.data.copy()

    # Required header keywords
    missing = [k for k in ("NORDERS", "ORDERS") if k not in hdr]
    if missing:
        raise ValueError(
            f"wavecalinfo for mode '{mode_name}' is missing required header "
            f"keywords: {missing}"
        )

    n_orders = int(hdr["NORDERS"])
    try:
        orders = [int(o) for o in hdr["ORDERS"].split(",")]
    except ValueError as exc:
        raise ValueError(
            f"wavecalinfo for mode '{mode_name}': cannot parse ORDERS header "
            f"value {hdr['ORDERS']!r} as a comma-separated list of integers."
        ) from exc

    # Validate cube shape: must be (n_orders, 4, n_pixels)
    if data.ndim != 3:
        raise ValueError(
            f"wavecalinfo for mode '{mode_name}' data cube has unexpected "
            f"number of dimensions {data.ndim}; expected 3."
        )
    if data.shape[0] != n_orders:
        raise ValueError(
            f"wavecalinfo for mode '{mode_name}' data cube first axis "
            f"{data.shape[0]} != NORDERS={n_orders}."
        )
    if data.shape[1] != 4:
        raise ValueError(
            f"wavecalinfo for mode '{mode_name}' data cube second axis "
            f"{data.shape[1]} != 4 (expected 4 spectral planes)."
        )

    # Consistency: header order count must match list length
    if len(orders) != n_orders:
        raise ValueError(
            f"wavecalinfo for mode '{mode_name}': ORDERS string lists "
            f"{len(orders)} orders but NORDERS={n_orders}."
        )

    return WaveCalInfo(
        mode=mode_name,
        n_orders=n_orders,
        orders=orders,
        resolving_power=float(hdr.get("RP", float("nan"))),
        data=data,
        linelist_name=hdr.get("LINELIST", ""),
        wcal_type=hdr.get("WCALTYPE", ""),
        home_order=int(hdr.get("HOMEORDR", 0)),
        disp_degree=int(hdr.get("DISPDEG", 0)),
        order_degree=int(hdr.get("ORDRDEG", 0)),
        xranges=_parse_order_xranges(hdr, orders),
    )


def read_linearity_cube() -> LinearityCube:
    """Load the iSHELL detector linearity-correction cube.

    Returns
    -------
    :class:`LinearityCube`

    Raises
    ------
    FileNotFoundError
        If the packaged FITS file is missing.
    RuntimeError
        If the file is a Git LFS pointer.
    ValueError
        If the data cube shape is inconsistent with detector dimensions.
    """
    path = get_detector_resource("linearity")
    label = "linearity cube"

    with _open_fits(path, label) as hdul:
        hdu = hdul[0]
        hdr = hdu.header
        data = hdu.data.copy()

    # Validate: must be (n_coeffs, 2048, 2048)
    if data.ndim != 3:
        raise ValueError(
            f"Linearity cube has unexpected number of dimensions {data.ndim}; "
            "expected 3 (n_coeffs, rows, cols)."
        )
    if data.shape[1:] != _DETECTOR_SHAPE:
        raise ValueError(
            f"Linearity cube spatial dimensions {data.shape[1:]} do not match "
            f"detector shape {_DETECTOR_SHAPE}."
        )

    return LinearityCube(
        data=data,
        dn_lower_limit=float(hdr.get("DNLOLIM", 0.0)),
        dn_upper_limit=float(hdr.get("DNUPLIM", 37000.0)),
        fit_order=int(hdr.get("FITORDER", data.shape[0] - 1)),
        n_pedestals=int(hdr.get("NPEDS", 0)),
    )


def read_pixel_mask(mask_type: str) -> PixelMask:
    """Load the iSHELL bad-pixel or hot-pixel mask.

    Parameters
    ----------
    mask_type : str
        Either ``"bad"`` or ``"hot"``.

    Returns
    -------
    :class:`PixelMask`

    Raises
    ------
    ValueError
        If *mask_type* is not ``"bad"`` or ``"hot"``.
    FileNotFoundError
        If the packaged FITS file is missing.
    RuntimeError
        If the file is a Git LFS pointer.
    ValueError
        If the mask shape is inconsistent with detector dimensions.
    """
    if mask_type not in ("bad", "hot"):
        raise ValueError(
            f"mask_type must be 'bad' or 'hot', got {mask_type!r}"
        )

    resource_key = f"{mask_type}_pixel_mask"
    path = get_detector_resource(resource_key)
    label = f"{mask_type}-pixel mask"

    with _open_fits(path, label) as hdul:
        data = hdul[0].data.copy()

    if data.shape != _DETECTOR_SHAPE:
        raise ValueError(
            f"{mask_type.capitalize()}-pixel mask shape {data.shape} does not "
            f"match detector shape {_DETECTOR_SHAPE}."
        )

    return PixelMask(
        mask_type=mask_type,
        data=data,
        is_placeholder=is_placeholder_resource(resource_key),
    )


def read_bias() -> BiasFrame:
    """Load the iSHELL bias frame.

    Returns
    -------
    :class:`BiasFrame`

    Raises
    ------
    FileNotFoundError
        If the packaged FITS file is missing.
    RuntimeError
        If the file is a Git LFS pointer.
    ValueError
        If the bias frame shape is inconsistent with detector dimensions.
    """
    path = get_detector_resource("bias")
    label = "bias frame"

    with _open_fits(path, label) as hdul:
        hdu = hdul[0]
        data = hdu.data.copy()
        divisor = int(hdu.header.get("DIVISOR", 1))

    if data.shape != _DETECTOR_SHAPE:
        raise ValueError(
            f"Bias frame shape {data.shape} does not match detector "
            f"shape {_DETECTOR_SHAPE}."
        )

    return BiasFrame(data=data, divisor=divisor)


def load_mode_calibrations(mode_name: str) -> ModeCalibrations:
    """Load all packaged calibration resources for *mode_name*.

    This is a convenience wrapper that calls :func:`read_line_list`,
    :func:`read_flatinfo`, and :func:`read_wavecalinfo` in sequence and
    returns a single :class:`ModeCalibrations` object.

    Parameters
    ----------
    mode_name : str
        A supported iSHELL mode (e.g. ``"J0"``).

    Returns
    -------
    :class:`ModeCalibrations`

    Raises
    ------
    KeyError
        If *mode_name* is not in the mode registry.
    FileNotFoundError
        If any required file is missing.
    RuntimeError
        If any file is a Git LFS pointer.
    ValueError
        If any resource fails structural validation.
    """
    return ModeCalibrations(
        mode=mode_name,
        line_list=read_line_list(mode_name),
        flatinfo=read_flatinfo(mode_name),
        wavecalinfo=read_wavecalinfo(mode_name),
    )
