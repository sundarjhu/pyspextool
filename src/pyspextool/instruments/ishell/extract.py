"""
iSHELL spectral extraction for J/H/K modes.

Scope
-----
J, H, and K modes only (J0–J3, H1–H3, K1–K3, Kgas).
L, Lp, and M modes are explicitly out of scope.

Overview
--------
This module bridges the iSHELL preprocessing stage and the generic
pySpextool spectral extraction engine (:func:`~pyspextool.extract.extraction.extract_1dxd`).

The iSHELL-specific work performed here is:

1. **Build extraction arrays** — translate the :class:`OrderGeometrySet`
   geometry into the ``ordermask``, ``wavecal``, and ``spatcal`` 2-D arrays
   expected by :func:`~pyspextool.extract.extraction.extract_1dxd`.
2. **Assemble extraction parameters** — construct ``trace_coefficients``,
   ``aperture_radii``, and ``aperture_signs`` arrays in the format expected
   by the generic engine.
3. **Propagate the quality bitmask** — convert the preprocessing bitmask to
   the ``linmax_bitmask`` argument accepted by
   :func:`~pyspextool.extract.extraction.extract_1dxd` so that any flagged
   pixel (non-linearity, bad flat pixel, etc.) is recorded in the output
   spectral flag array.
4. **Call the generic extractor** and return the result together with
   extraction metadata.

What is generic vs iSHELL-specific
------------------------------------
* **Generic** — all spectral extraction arithmetic (aperture masking,
  background fitting, optimal/sum extraction, variance propagation) is
  performed by :func:`~pyspextool.extract.extraction.extract_1dxd`.
* **iSHELL-specific** — order geometry representation
  (:class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`),
  wavelength and spatial calibration arrays derived from that geometry, and
  the mapping from preprocessing output to extractor inputs.

Provisional tilt model
-----------------------
The current preprocessing rectification uses a placeholder zero-tilt model.
Spectral-line curvature is **not** corrected.  This module accepts that
limitation and documents it transparently.  The quality bitmask propagated
through this module includes a ``'tilt_provisional'`` flag in the returned
metadata dictionary; callers must inspect this flag before interpreting
extracted spectra as science-quality.

See ``docs/ishell_extraction_guide.md`` for a full description of what
is and is not yet scientifically grounded.

Public API
----------
:func:`build_extraction_arrays`
    Construct ordermask, wavecal, and spatcal 2-D arrays from an
    :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`.

:func:`extract_spectra`
    Main entry point: accepts the output of
    :func:`~pyspextool.instruments.ishell.preprocess.preprocess_science_frames`,
    builds extraction arrays, calls the generic extractor, and returns
    extracted spectra plus metadata.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyspextool.extract.extraction import extract_1dxd
from pyspextool.io.check import check_parameter
from pyspextool.pyspextoolerror import pySpextoolError
from pyspextool.instruments.ishell.geometry import OrderGeometrySet
from pyspextool.instruments.ishell.ishell import _DEFAULT_PLATE_SCALE

__all__ = [
    "build_extraction_arrays",
    "extract_spectra",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default extraction aperture half-radius in arcseconds when the caller
#: does not supply ``aperture_radii_arcsec``.
_DEFAULT_APERTURE_RADIUS_ARCSEC = 0.5


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_extraction_arrays(
    geometry: OrderGeometrySet,
    nrows: int,
    ncols: int,
    plate_scale_arcsec: float = _DEFAULT_PLATE_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ordermask, wavecal, and spatcal 2-D arrays from an
    :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`.

    These arrays are the iSHELL-specific inputs required by the generic
    :func:`~pyspextool.extract.extraction.extract_1dxd` extractor.

    Parameters
    ----------
    geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        Order-geometry object with wavelength solutions populated
        (``geometry.has_wavelength_solution()`` must be ``True``).
    nrows : int
        Number of detector rows in the science image.
    ncols : int
        Number of detector columns in the science image.
    plate_scale_arcsec : float, optional
        Spatial plate scale in arcsec/pixel.  Defaults to the iSHELL
        standard of 0.125 arcsec/pixel.

    Returns
    -------
    ordermask : ndarray of int, shape (nrows, ncols)
        Each pixel is set to its echelle order number.  Inter-order
        pixels are set to zero.
    wavecal : ndarray of float, shape (nrows, ncols)
        Each pixel inside an order is set to its wavelength in microns,
        evaluated from the order wavelength polynomial at the corresponding
        detector column.  Inter-order pixels are NaN.
    spatcal : ndarray of float, shape (nrows, ncols)
        Each pixel inside an order is set to its spatial position in
        arcseconds.  The convention is 0 arcsec at the bottom order edge
        and ``slit_height_arcsec`` at the top order edge, following the
        pySpextool ``simulate_wavecal_1dxd`` convention so that the default
        aperture center (half the slit height) aligns with the geometrical
        slit center.  Inter-order pixels are NaN.

    Raises
    ------
    ValueError
        If *geometry* does not have a wavelength solution
        (``geometry.has_wavelength_solution()`` is ``False``).
    ValueError
        If *geometry* is empty (no orders).

    Notes
    -----
    The wavelength at each pixel is assumed to be constant along a column
    within one order (i.e. the dispersion direction is parallel to columns).
    This is the standard 1DXD assumption and is consistent with the
    ``simulate_wavecal_1dxd`` helper used for SpeX/uSpeX.
    """
    check_parameter('build_extraction_arrays', 'geometry', geometry,
                    'OrderGeometrySet')
    check_parameter('build_extraction_arrays', 'nrows', nrows, 'int')
    check_parameter('build_extraction_arrays', 'ncols', ncols, 'int')
    check_parameter('build_extraction_arrays', 'plate_scale_arcsec',
                    plate_scale_arcsec, ['int', 'float'])

    if not geometry.geometries:
        raise ValueError(
            "build_extraction_arrays: OrderGeometrySet is empty "
            "(no orders).  Populate it from flat calibration data first.")

    if not geometry.has_wavelength_solution():
        raise ValueError(
            "build_extraction_arrays: the supplied geometry does not have "
            "a wavelength solution "
            "(geometry.has_wavelength_solution() is False).  "
            "Call build_geometry_from_wavecalinfo() first.")

    ordermask = np.zeros((nrows, ncols), dtype=int)
    wavecal = np.full((nrows, ncols), np.nan, dtype=float)
    spatcal = np.full((nrows, ncols), np.nan, dtype=float)

    for geom in geometry.geometries:
        x_start = int(np.clip(geom.x_start, 0, ncols - 1))
        x_end = int(np.clip(geom.x_end, 0, ncols - 1))
        cols = np.arange(x_start, x_end + 1, dtype=float)

        bot_edges = geom.eval_bottom_edge(cols)
        top_edges = geom.eval_top_edge(cols)

        # Wavelength at each column (constant along column within an order)
        wavs = np.polynomial.polynomial.polyval(cols, geom.wave_coeffs)

        for j_idx in range(len(cols)):
            col = int(cols[j_idx])
            row_bot = int(np.clip(np.floor(bot_edges[j_idx]), 0, nrows - 1))
            row_top = int(np.clip(np.ceil(top_edges[j_idx]), 0, nrows))

            if row_top <= row_bot:
                continue

            rows = np.arange(row_bot, row_top)

            # ordermask: mark pixels with their order number
            ordermask[row_bot:row_top, col] = geom.order

            # wavecal: wavelength is constant along a column in the order
            wavecal[row_bot:row_top, col] = wavs[j_idx]

            # spatcal: linear mapping from 0 (bottom edge) to
            # slit_height_arcsec (top edge), following the
            # simulate_wavecal_1dxd convention.
            slit_height_pix = top_edges[j_idx] - bot_edges[j_idx]
            if slit_height_pix > 0:
                slit_height_arcsec = slit_height_pix * plate_scale_arcsec
                slope = slit_height_arcsec / slit_height_pix
                intercept = -slope * bot_edges[j_idx]
                spatcal[row_bot:row_top, col] = (
                    slope * rows.astype(float) + intercept
                )

    return ordermask, wavecal, spatcal


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------


def extract_spectra(
    preprocess_result: dict,
    geometry: OrderGeometrySet,
    aperture_radii_arcsec: float | npt.ArrayLike = _DEFAULT_APERTURE_RADIUS_ARCSEC,
    aperture_positions_arcsec: npt.ArrayLike | None = None,
    aperture_signs: npt.ArrayLike | None = None,
    bg_annulus: list | npt.ArrayLike | None = None,
    bg_regions: str | None = None,
    bg_fitdegree: int = 1,
    plate_scale_arcsec: float = _DEFAULT_PLATE_SCALE,
    verbose: bool = False,
) -> tuple[list, dict]:
    """Extract spectra from preprocessed iSHELL science frames.

    This function is the primary integration point between the iSHELL
    preprocessing stage and the generic pySpextool extraction engine.  It:

    1. Validates that *preprocess_result* contains the required keys.
    2. Calls :func:`build_extraction_arrays` to derive ``ordermask``,
       ``wavecal``, and ``spatcal`` from *geometry*.
    3. Assembles ``trace_coefficients``, ``aperture_radii``, and
       ``aperture_signs`` in the format expected by
       :func:`~pyspextool.extract.extraction.extract_1dxd`.
    4. Converts the preprocessing bitmask to a ``linmax_bitmask`` so that
       flagged pixels are propagated to the output spectral flag array.
    5. Calls the generic :func:`~pyspextool.extract.extraction.extract_1dxd`
       and returns the extracted spectra together with metadata.

    .. warning::
        **Provisional tilt model.**  If ``preprocess_result['tilt_provisional']``
        is ``True``, the rectification step used a placeholder zero-tilt model
        and spectral-line curvature has **not** been corrected.  The extracted
        spectra are structurally valid but are not science-quality until a
        proper 2-D tilt solution is available.  See
        ``docs/ishell_extraction_guide.md`` for details.

    Parameters
    ----------
    preprocess_result : dict
        Output dictionary from
        :func:`~pyspextool.instruments.ishell.preprocess.preprocess_science_frames`.
        Must contain keys ``'image'``, ``'variance'``, ``'bitmask'``.

    geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        Order-geometry object with wavelength solutions populated.  Must
        satisfy ``geometry.has_wavelength_solution() == True``.

    aperture_radii_arcsec : float or array_like, optional
        Extraction aperture half-radius (or radii) in arcseconds.

        * **Scalar** — the same radius is used for all apertures and all
          orders.
        * **1-D array of length naps** — one radius per aperture, the same
          for all orders.
        * **2-D array of shape (norders, naps)** — one radius per
          (order, aperture) pair.

        Defaults to ``0.5`` arcsec.

    aperture_positions_arcsec : array_like of float or None, optional
        Aperture center positions in arcseconds measured from the bottom
        edge of the slit (the same coordinate system as ``spatcal``).

        * ``None`` (default): a **single centered aperture** is placed at
          half the slit height at each order's midpoint column.
        * 1-D array: one position per aperture; applied to every order.

        For A-B nodded observations pass two positions (e.g. the nod
        throw offsets from the slit bottom) and set
        ``aperture_signs=[1, -1]``.

    aperture_signs : array_like of int or None, optional
        Sign of each aperture: ``+1`` for a positive source, ``-1`` for a
        negative source (e.g. the B-beam in A-B nodded mode).

        * ``None`` (default): a single ``[1]`` (positive aperture).
        * Must have the same length as *aperture_positions_arcsec*.

    bg_annulus : array_like or None, optional
        Background annulus passed directly to
        :func:`~pyspextool.extract.extraction.extract_1dxd`.

    bg_regions : str or None, optional
        Background region string passed directly to
        :func:`~pyspextool.extract.extraction.extract_1dxd`.

    bg_fitdegree : int, optional
        Polynomial degree for background fitting.  Default 1.

    plate_scale_arcsec : float, optional
        Spatial plate scale in arcsec/pixel used when building the
        ``spatcal`` array.  Defaults to 0.125 arcsec/pixel (the iSHELL
        standard).

    verbose : bool, optional
        Print progress messages.  Default ``False``.

    Returns
    -------
    spectra : list of ndarray, length norders * naps
        Each element is a ``(4, nwave)`` array where:

        * Row 0 — wavelength (microns).
        * Row 1 — intensity (DN s⁻¹).
        * Row 2 — uncertainty (DN s⁻¹).
        * Row 3 — spectral quality flag (0 = clean; non-zero if any
          bitmask-flagged pixel fell in the aperture at that wavelength).

        The ordering is aperture-minor, order-major: element
        ``i * naps + k`` corresponds to order index ``i`` and aperture
        index ``k``.

    metadata : dict
        Extraction metadata with the following keys:

        ``'orders'`` : list of int
            Echelle order numbers that were extracted.
        ``'n_apertures'`` : int
            Number of apertures extracted per order.
        ``'aperture_positions_arcsec'`` : ndarray
            Aperture positions actually used (arcsec from slit bottom).
        ``'aperture_radii_arcsec'`` : ndarray, shape (norders, naps)
            Aperture radii actually used.
        ``'aperture_signs'`` : ndarray
            Aperture signs actually used.
        ``'subtraction_mode'`` : str
            Propagated from *preprocess_result*.
        ``'rectified'`` : bool
            Propagated from *preprocess_result*.
        ``'tilt_provisional'`` : bool
            Propagated from *preprocess_result*.  ``True`` when the
            preprocessing rectification used the provisional zero-tilt
            model; extracted spectra should not be treated as
            science-quality in this case.
        ``'plate_scale_arcsec'`` : float
            Plate scale used to build ``spatcal``.

    Raises
    ------
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If *preprocess_result* is missing required keys.
    ValueError
        If *geometry* does not have a wavelength solution or is empty.
    ValueError
        If the image shape in *preprocess_result* is inconsistent with
        *geometry*.
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If *aperture_positions_arcsec* and *aperture_signs* have
        inconsistent lengths.

    Examples
    --------
    Minimal extraction from a pair of A-B frames:

    >>> from pyspextool.instruments.ishell.preprocess import (
    ...     preprocess_science_frames)
    >>> from pyspextool.instruments.ishell.extract import extract_spectra
    >>> from pyspextool.instruments.ishell.calibrations import (
    ...     read_flatinfo, read_wavecalinfo)
    >>> from pyspextool.instruments.ishell.wavecal import (
    ...     build_geometry_from_wavecalinfo)
    >>> fi = read_flatinfo('K1')
    >>> wci = read_wavecalinfo('K1')
    >>> geom = build_geometry_from_wavecalinfo(wci, fi)
    >>> pre = preprocess_science_frames(
    ...     ['/data/ishell_A.fits', '/data/ishell_B.fits'],
    ...     flat_info=fi, geometry=geom, subtraction_mode='A-B')
    >>> spectra, meta = extract_spectra(pre, geom)
    >>> spectra[0].shape   # (4, nwave) for the first (order, aperture)
    (4, 2048)
    >>> meta['tilt_provisional']
    True
    """
    # ------------------------------------------------------------------
    # Validate preprocess_result
    # ------------------------------------------------------------------
    required_keys = {'image', 'variance', 'bitmask'}
    if not isinstance(preprocess_result, dict):
        raise pySpextoolError(
            "extract_spectra: preprocess_result must be a dict "
            "(output of preprocess_science_frames).")
    missing = required_keys - set(preprocess_result.keys())
    if missing:
        raise pySpextoolError(
            f"extract_spectra: preprocess_result is missing required keys: "
            f"{sorted(missing)}.  Ensure it is the output of "
            f"preprocess_science_frames().")

    check_parameter('extract_spectra', 'geometry', geometry,
                    'OrderGeometrySet')
    check_parameter('extract_spectra', 'bg_fitdegree', bg_fitdegree, 'int')
    check_parameter('extract_spectra', 'plate_scale_arcsec',
                    plate_scale_arcsec, ['int', 'float'])
    check_parameter('extract_spectra', 'verbose', verbose, 'bool')

    image = np.asarray(preprocess_result['image'], dtype=float)
    variance = np.asarray(preprocess_result['variance'], dtype=float)
    bitmask = np.asarray(preprocess_result['bitmask'])

    nrows, ncols = image.shape

    if variance.shape != (nrows, ncols):
        raise ValueError(
            f"extract_spectra: variance shape {variance.shape} does not "
            f"match image shape {image.shape}.")
    if bitmask.shape != (nrows, ncols):
        raise ValueError(
            f"extract_spectra: bitmask shape {bitmask.shape} does not "
            f"match image shape {image.shape}.")

    # ------------------------------------------------------------------
    # Warn if tilt model is provisional
    # ------------------------------------------------------------------
    tilt_provisional = bool(preprocess_result.get('tilt_provisional', False))
    if tilt_provisional:
        warnings.warn(
            "extract_spectra: preprocess_result['tilt_provisional'] is True.  "
            "The rectification was performed with a provisional zero-tilt "
            "model; spectral-line curvature has not been corrected.  "
            "Extracted spectra are structurally valid but not "
            "science-quality pending a proper 2-D tilt solution.  "
            "See docs/ishell_extraction_guide.md for details.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Build extraction arrays from geometry
    # ------------------------------------------------------------------
    if verbose:
        print(" Building ordermask, wavecal, spatcal from geometry.")

    ordermask, wavecal, spatcal = build_extraction_arrays(
        geometry=geometry,
        nrows=nrows,
        ncols=ncols,
        plate_scale_arcsec=plate_scale_arcsec,
    )

    extract_orders = np.array(geometry.orders, dtype=int)
    norders = len(extract_orders)

    # ------------------------------------------------------------------
    # Determine aperture parameters
    # ------------------------------------------------------------------
    # Resolve aperture_signs
    if aperture_signs is None:
        aperture_signs = np.array([1], dtype=int)
    else:
        aperture_signs = np.asarray(aperture_signs, dtype=int)

    naps = len(aperture_signs)

    # Resolve aperture_positions_arcsec
    # Default: single centered aperture at half the slit height for each order
    if aperture_positions_arcsec is None:
        if naps != 1:
            raise pySpextoolError(
                "extract_spectra: aperture_positions_arcsec must be supplied "
                "when naps > 1 (aperture_signs has more than one element).  "
                "Pass the aperture positions in arcsec from the slit bottom.")
        # Compute slit height from geometry edge polynomials at the midpoint
        # column of the first order (they should be similar across orders)
        first_geom = geometry.geometries[0]
        col_mid = 0.5 * (first_geom.x_start + first_geom.x_end)
        row_bot = float(first_geom.eval_bottom_edge(col_mid))
        row_top = float(first_geom.eval_top_edge(col_mid))
        slit_height_arcsec = (row_top - row_bot) * plate_scale_arcsec
        aperture_positions_arcsec = np.array([slit_height_arcsec / 2.0])
    else:
        aperture_positions_arcsec = np.asarray(aperture_positions_arcsec,
                                               dtype=float)
        if len(aperture_positions_arcsec) != naps:
            raise pySpextoolError(
                f"extract_spectra: aperture_positions_arcsec has "
                f"{len(aperture_positions_arcsec)} elements but aperture_signs "
                f"has {naps}.  They must have the same length.")

    # Resolve aperture_radii_arcsec → shape (norders, naps)
    radii_arr = np.asarray(aperture_radii_arcsec, dtype=float)
    if radii_arr.ndim == 0:
        # Scalar: same radius for all orders and apertures
        aperture_radii = np.full((norders, naps), float(radii_arr))
    elif radii_arr.ndim == 1:
        if len(radii_arr) == naps:
            # One radius per aperture, same for all orders
            aperture_radii = np.tile(radii_arr, (norders, 1))
        elif len(radii_arr) == norders:
            # One radius per order, same for all apertures
            aperture_radii = np.tile(radii_arr[:, np.newaxis], (1, naps))
        else:
            raise pySpextoolError(
                f"extract_spectra: aperture_radii_arcsec 1-D array has "
                f"length {len(radii_arr)} but norders={norders} and "
                f"naps={naps}.  Supply a scalar, length-naps, or "
                f"(norders, naps) array.")
    elif radii_arr.ndim == 2:
        if radii_arr.shape != (norders, naps):
            raise pySpextoolError(
                f"extract_spectra: aperture_radii_arcsec shape "
                f"{radii_arr.shape} does not match expected "
                f"(norders={norders}, naps={naps}).")
        aperture_radii = radii_arr
    else:
        raise pySpextoolError(
            "extract_spectra: aperture_radii_arcsec must be scalar, 1-D, "
            "or 2-D.")

    # Build trace_coefficients: shape (norders * naps, n_terms)
    # Each row is a polynomial of wavelength giving the aperture center in
    # the spatcal coordinate system (arcsec from slit bottom).
    # We use a constant (degree-0) polynomial for each aperture.
    trace_coefficients = np.zeros((norders * naps, 2), dtype=float)
    for i in range(norders):
        for k in range(naps):
            trace_coefficients[i * naps + k, 0] = aperture_positions_arcsec[k]
            # Linear term = 0: aperture center does not move with wavelength

    # ------------------------------------------------------------------
    # Build linmax_bitmask for flag propagation
    # Any non-zero bitmask value (linearity, bad flat, etc.) is flagged.
    # Replace NaN image pixels with 0 to prevent NaN propagation in sum
    # extraction; the bitmask already records them.
    # ------------------------------------------------------------------
    linmax_bitmask = (bitmask != 0).astype(int)
    nan_pixels = ~np.isfinite(image)
    image_clean = image.copy()
    image_clean[nan_pixels] = 0.0
    linmax_bitmask[nan_pixels] = 1

    var_clean = variance.copy()
    var_nan = ~np.isfinite(var_clean)
    var_clean[var_nan] = 0.0

    # ------------------------------------------------------------------
    # Call the generic extractor
    # ------------------------------------------------------------------
    if verbose:
        print(
            f" Extracting {norders} order(s), {naps} aperture(s) "
            f"(A-B mode: {preprocess_result.get('subtraction_mode', 'unknown')}).")

    spectra = extract_1dxd(
        image=image_clean,
        variance=var_clean,
        ordermask=ordermask,
        wavecal=wavecal,
        spatcal=spatcal,
        extract_orders=extract_orders,
        trace_coefficients=trace_coefficients,
        aperture_radii=aperture_radii,
        aperture_signs=aperture_signs,
        linmax_bitmask=linmax_bitmask,
        bg_annulus=bg_annulus,
        bg_regions=bg_regions,
        bg_fitdegree=bg_fitdegree,
        progressbar=verbose,
    )

    # ------------------------------------------------------------------
    # Build metadata
    # ------------------------------------------------------------------
    metadata = {
        'orders': geometry.orders,
        'n_apertures': naps,
        'aperture_positions_arcsec': aperture_positions_arcsec,
        'aperture_radii_arcsec': aperture_radii,
        'aperture_signs': aperture_signs,
        'subtraction_mode': preprocess_result.get('subtraction_mode', None),
        'rectified': preprocess_result.get('rectified', False),
        'tilt_provisional': tilt_provisional,
        'plate_scale_arcsec': plate_scale_arcsec,
    }

    return spectra, metadata
