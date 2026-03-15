"""
iSHELL science-frame preprocessing pipeline.

Scope
-----
J, H, and K modes only (J0–J3, H1–H3, K1–K3, Kgas).
L, Lp, and M modes are explicitly out of scope.

Preprocessing stage sequence
-----------------------------
1. **Input validation** — check subtraction mode, file count, flat validity,
   and geometry readiness.
2. **Raw FITS ingestion** — load each file with
   :func:`~pyspextool.instruments.ishell.ishell.load_data`, which reads the
   3-extension iSHELL MEF structure, normalises to DN s⁻¹, and flags
   non-linear pixels.
3. **Pair subtraction** (A-B mode) or single-frame passthrough (A mode) —
   background sky emission is removed by subtracting the B-beam from the
   A-beam.  Variance and bitmask are propagated accordingly.
4. **Flat-field division** — the science image and variance are divided by
   the flat-field image stored in the supplied :class:`FlatInfo` object.
   Safe division is applied: pixels where the flat is zero or NaN are set
   to NaN and flagged in the bitmask.
5. **Order rectification** — if a fully populated
   :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` is
   provided (``geometry.has_wavelength_solution() == True``),
   :func:`~pyspextool.instruments.ishell.ishell._rectify_orders` is called
   to resample each echelle order onto a rectilinear grid.  Pixels outside
   all order footprints are set to NaN in the output image.  If *geometry*
   is ``None``, rectification is skipped.

   .. warning::
       **Provisional tilt model.** The current implementation uses a
       placeholder zero-tilt model: ``tilt_coeffs = [0.0]`` for every order.
       No spectral-line tilt has been measured; the resampling is
       structurally valid (NaN masking and edge geometry are applied) but
       does **not** remove spectral-line curvature.  Full 2-D tilt/curvature
       fitting requires real 2-D ThAr arc-lamp frames as input and is
       deferred to a future PR.  See
       ``docs/ishell_preprocessing_guide.md`` §4 for details.

Subtraction modes
-----------------
``'A-B'``
    Subtract the B-beam from the A-beam.  ``files`` must contain exactly
    two entries (A-beam first, B-beam second).  This is the primary mode
    for iSHELL J/H/K nodded science observations.

``'A'``
    Single A-beam frame; no background subtraction.  ``files`` must contain
    exactly one entry.

``'A-Sky'`` / ``'A-Dark'`` are not yet supported because the current
iSHELL read path reads only raw MEF science frames and provides no
separate sky or dark image.  These modes can be added when appropriate
calibration frames become available in the iSHELL data model.

Public API
----------
:func:`preprocess_science_frames`
    Main entry point for iSHELL science-frame preprocessing.

What is scientifically grounded
---------------------------------
* Raw MEF ingestion (3-extension iSHELL format) — confirmed against the
  iSHELL Spextool Manual §2.3.
* Non-linearity flagging using pedestal+signal sum vs. ``LINCORMAX=30000 DN``
  (Manual Table 4).
* A-B pair subtraction for sky removal (standard nod strategy).
* Flat-field division (illumination correction; photon-response non-uniformity
  removal).
* Variance propagation through all stages.

What remains provisional
------------------------
* **H2RG polynomial linearity correction** — the correction algorithm itself
  is scaffolded but not yet applied (Phase 2 task).
* **Reference-pixel bias subtraction** — scaffolded but not yet applied
  (Phase 2 task).
* **Spectral-line tilt model** — the zero-tilt placeholder allows structurally
  valid rectification but is not scientifically accurate.

References
----------
* iSHELL Spextool Manual v10jan2020, Cushing et al. (IRTF internal document)
* Vacca, Cushing, & Rayner (2003) PASP, 115, 389 — *not* used here (SpeX
  ALADDIN specific); iSHELL uses the H2RG polynomial correction.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyspextool.io.check import check_parameter
from pyspextool.pyspextoolerror import pySpextoolError
from pyspextool.instruments.ishell.ishell import (
    load_data,
    _rectify_orders,
    LINCORMAX_DN,
    SUPPORTED_MODES,
)
from pyspextool.instruments.ishell.calibrations import FlatInfo
from pyspextool.instruments.ishell.geometry import OrderGeometrySet

__all__ = [
    "preprocess_science_frames",
    "SUPPORTED_SUBTRACTION_MODES",
]

#: Subtraction modes currently implemented for iSHELL J/H/K preprocessing.
SUPPORTED_SUBTRACTION_MODES = ('A-B', 'A')

#: Default non-linearity threshold (DN) and bitmask bit used when no external
#: ``linearity_info`` dict is provided.  Matches the iSHELL Spextool Manual
#: Table 4 value (LINCRMAX = 30000) and bit 0.
_DEFAULT_LINEARITY_INFO = {'max': LINCORMAX_DN, 'bit': 0}

#: Bitmask bit set when the flat pixel is zero/NaN (unsafe division).
_FLAT_ZERO_BIT = 4


def preprocess_science_frames(
    files: Sequence[str],
    flat_info: FlatInfo,
    geometry: OrderGeometrySet | None = None,
    subtraction_mode: str = 'A-B',
    linearity_info: dict | None = None,
    linearity_correction: bool = True,
    keywords: list | None = None,
    verbose: bool = False,
) -> dict:
    """
    Preprocess one or two iSHELL raw science frames for J/H/K modes.

    This function implements the iSHELL science-frame preprocessing pipeline
    described in the module docstring:

    1. Load and validate raw MEF FITS files.
    2. Apply A-B pair subtraction (or single-frame passthrough).
    3. Divide by the flat-field image.
    4. (Optionally) rectify orders using the geometry/resampling path.

    .. warning::
        **Provisional tilt model.** When *geometry* is supplied, the
        rectification step uses a placeholder zero-tilt model.  No
        spectral-line tilt is removed.  See the module docstring and
        ``docs/ishell_preprocessing_guide.md`` §4 for the full status of
        what is and is not yet scientifically grounded.

    Parameters
    ----------
    files : sequence of str
        Full paths to raw iSHELL MEF FITS files.

        * ``'A'`` mode: exactly **one** file.
        * ``'A-B'`` mode: exactly **two** files (A-beam first, B-beam
          second).

    flat_info : :class:`~pyspextool.instruments.ishell.calibrations.FlatInfo`
        Loaded flat-field calibration for the observing mode.  Must have a
        non-None ``image`` attribute with at least some non-zero pixels.
        The ``image`` is used directly as the flat divisor; it should
        contain the normalised flat, not the raw counts.

    geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` or None, optional
        Order-geometry object with wavelength solutions populated.  When
        supplied, ``geometry.has_wavelength_solution()`` must be ``True``
        and ``geometry.mode`` should match ``flat_info.mode``.

        * If ``None`` (default): rectification is **skipped**; the function
          returns the flat-fielded image in detector coordinates.
        * If supplied but wavelength solution is absent: raises
          :class:`ValueError`.

        NOTE: The current tilt model is a provisional zero placeholder.
        Rectification is structurally valid (NaN masking, edge geometry)
        but does not remove spectral-line curvature.

    subtraction_mode : {'A-B', 'A'}, default ``'A-B'``
        Sky/background subtraction mode.  See module docstring for details.

    linearity_info : dict or None, optional
        Non-linearity flagging configuration with keys:

        ``'max'`` : int
            Pixel DN threshold; pixels above this value are flagged.
            Defaults to ``LINCORMAX_DN`` (30000) from the iSHELL module.
        ``'bit'`` : int
            Bitmask bit to set for flagged pixels.  Defaults to 0.

        If ``None``, the default ``{'max': 30000, 'bit': 0}`` is used.

    linearity_correction : bool, default ``True``
        Passed through to :func:`~pyspextool.instruments.ishell.ishell.load_data`.
        Currently a no-op (H2RG polynomial correction is a Phase 2 task);
        accepted for interface compatibility.

    keywords : list of str or None, optional
        Additional FITS header keywords to retain.

    verbose : bool, default ``False``
        Print progress messages.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'image'`` : ndarray, shape ``(nrows, ncols)``
            Science image after all preprocessing stages, in DN s⁻¹.
            Pixels outside order footprints are NaN when rectified;
            the full detector frame is returned when *geometry* is None.

        ``'variance'`` : ndarray, shape ``(nrows, ncols)``
            Variance image propagated through all stages, in (DN s⁻¹)².

        ``'bitmask'`` : ndarray of uint8, shape ``(nrows, ncols)``
            Combined quality bitmask.  The following bits may be set:

            * Bit ``linearity_info['bit']`` — non-linear pixel (from any
              input frame).
            * Bit 4 — flat pixel is zero or NaN (unsafe division; science
              value set to NaN).

        ``'hdrinfo'`` : dict
            Normalised header keyword dictionary from the A-beam frame
            (see :func:`~pyspextool.instruments.ishell.ishell.get_header`).

        ``'subtraction_mode'`` : str
            The subtraction mode that was applied.

        ``'flat_applied'`` : bool
            ``True`` (flat division is always applied by this function).

        ``'rectified'`` : bool
            ``True`` if order rectification was applied, ``False`` if
            *geometry* was ``None`` and rectification was skipped.

        ``'tilt_provisional'`` : bool
            ``True`` when ``'rectified'`` is ``True``: the tilt model is
            a provisional zero placeholder and has **not** been measured
            from arc-lamp data.  ``False`` when rectification was skipped.

    Raises
    ------
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If ``subtraction_mode`` is not in
        :data:`SUPPORTED_SUBTRACTION_MODES`.
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If the number of ``files`` does not match the subtraction mode.
    :class:`ValueError`
        If ``flat_info.image`` is ``None`` or all-zero / all-NaN.
    :class:`ValueError`
        If *geometry* is supplied and ``geometry.has_wavelength_solution()``
        is ``False``.
    :class:`ValueError`
        If *geometry* is supplied and its mode does not match
        ``flat_info.mode``.

    Examples
    --------
    Minimal example with A-B pair subtraction and rectification:

    >>> from pyspextool.instruments.ishell.calibrations import (
    ...     read_flatinfo, read_wavecalinfo, read_line_list)
    >>> from pyspextool.instruments.ishell.wavecal import (
    ...     build_geometry_from_wavecalinfo)
    >>> from pyspextool.instruments.ishell.preprocess import (
    ...     preprocess_science_frames)
    >>> fi = read_flatinfo('K1')
    >>> wci = read_wavecalinfo('K1')
    >>> geom = build_geometry_from_wavecalinfo(wci, fi)
    >>> result = preprocess_science_frames(
    ...     files=['/data/ishell_0100.a.fits', '/data/ishell_0101.b.fits'],
    ...     flat_info=fi,
    ...     geometry=geom,
    ...     subtraction_mode='A-B',
    ... )
    >>> result['image'].shape
    (2048, 2048)
    >>> result['rectified']
    True
    >>> result['tilt_provisional']
    True
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    check_parameter('preprocess_science_frames', 'files', list(files), 'list')
    check_parameter('preprocess_science_frames', 'flat_info', flat_info,
                    'FlatInfo')
    check_parameter('preprocess_science_frames', 'geometry', geometry,
                    ['NoneType', 'OrderGeometrySet'])
    check_parameter('preprocess_science_frames', 'subtraction_mode',
                    subtraction_mode, 'str')
    check_parameter('preprocess_science_frames', 'linearity_info',
                    linearity_info, ['NoneType', 'dict'])
    check_parameter('preprocess_science_frames', 'linearity_correction',
                    linearity_correction, 'bool')
    check_parameter('preprocess_science_frames', 'keywords', keywords,
                    ['NoneType', 'list'])
    check_parameter('preprocess_science_frames', 'verbose', verbose, 'bool')

    files = list(files)

    if subtraction_mode not in SUPPORTED_SUBTRACTION_MODES:
        raise pySpextoolError(
            f"preprocess_science_frames: subtraction_mode "
            f"{subtraction_mode!r} is not supported.  "
            f"Supported modes: {SUPPORTED_SUBTRACTION_MODES}.")

    if subtraction_mode == 'A-B' and len(files) != 2:
        raise pySpextoolError(
            f"preprocess_science_frames: 'A-B' mode requires exactly 2 "
            f"files (A-beam and B-beam); got {len(files)}.")

    if subtraction_mode == 'A' and len(files) != 1:
        raise pySpextoolError(
            f"preprocess_science_frames: 'A' mode requires exactly 1 file; "
            f"got {len(files)}.")

    # Validate flat
    flat_image = flat_info.image
    if flat_image is None:
        raise ValueError(
            "preprocess_science_frames: flat_info.image is None.  "
            "The FlatInfo object must contain a valid flat-field image.")
    flat_arr = np.asarray(flat_image, dtype=float)
    if np.all(flat_arr == 0) or not np.any(np.isfinite(flat_arr)):
        raise ValueError(
            "preprocess_science_frames: flat_info.image is all-zero or all-NaN; "
            "cannot safely divide by this flat.")

    # Validate geometry (if provided)
    if geometry is not None:
        if not geometry.has_wavelength_solution():
            raise ValueError(
                "preprocess_science_frames: the supplied geometry does not "
                "have a wavelength solution "
                "(geometry.has_wavelength_solution() is False).  "
                "Call build_geometry_from_wavecalinfo() or "
                "build_geometry_from_arc_lines() first.")
        if geometry.mode != flat_info.mode:
            raise ValueError(
                f"preprocess_science_frames: geometry.mode={geometry.mode!r} "
                f"does not match flat_info.mode={flat_info.mode!r}.  "
                f"Load matching flat and geometry calibrations.")

    if linearity_info is None:
        linearity_info = _DEFAULT_LINEARITY_INFO

    # ------------------------------------------------------------------
    # Stage 1: Load raw frames
    # ------------------------------------------------------------------
    if verbose:
        print(f" Loading A-beam: {files[0]}")

    img_a, var_a, hdrinfo_a, mask_a = load_data(
        files[0], linearity_info, keywords,
        linearity_correction=linearity_correction)

    if subtraction_mode == 'A-B':
        if verbose:
            print(f" Loading B-beam: {files[1]}")
        img_b, var_b, hdrinfo_b, mask_b = load_data(
            files[1], linearity_info, keywords,
            linearity_correction=linearity_correction)

    # ------------------------------------------------------------------
    # Stage 2: A-B pair subtraction (or single-frame passthrough)
    # ------------------------------------------------------------------
    if subtraction_mode == 'A-B':
        if verbose:
            print(" Applying A-B pair subtraction.")
        img = img_a - img_b
        var = var_a + var_b
        # Combine bitmasks: any bit set in either frame propagates
        bitmask = np.bitwise_or(mask_a, mask_b).astype(np.uint8)
    else:
        img = img_a.copy()
        var = var_a.copy()
        bitmask = mask_a.copy()

    hdrinfo = hdrinfo_a  # A-beam header carries the source signal

    # ------------------------------------------------------------------
    # Stage 3: Flat-field division
    # ------------------------------------------------------------------
    if verbose:
        print(" Applying flat-field correction.")

    # Identify unsafe flat pixels (zero or NaN)
    flat_bad = (flat_arr == 0) | ~np.isfinite(flat_arr)
    if flat_bad.any():
        # Flag these pixels in the bitmask
        bitmask[flat_bad] = np.bitwise_or(
            bitmask[flat_bad], np.uint8(2 ** _FLAT_ZERO_BIT)).astype(np.uint8)

    # Safe division: replace bad flat pixels with 1 temporarily,
    # then set the result to NaN
    safe_flat = flat_arr.copy()
    safe_flat[flat_bad] = 1.0

    img = img / safe_flat
    var = var / safe_flat ** 2
    img[flat_bad] = np.nan
    var[flat_bad] = np.nan

    # ------------------------------------------------------------------
    # Stage 4: Order rectification
    # ------------------------------------------------------------------
    do_rectify = geometry is not None
    tilt_provisional = False

    if do_rectify:
        if verbose:
            print(
                " Rectifying orders (NOTE: provisional zero-tilt model).")
            warnings.warn(
                "preprocess_science_frames: order rectification uses a "
                "provisional zero-tilt model.  Spectral-line tilt is not "
                "corrected.  See docs/ishell_preprocessing_guide.md §4.",
                RuntimeWarning,
                stacklevel=2,
            )

        plate_scale = flat_info.plate_scale_arcsec
        if not np.isfinite(plate_scale) or plate_scale <= 0:
            from pyspextool.instruments.ishell.ishell import (
                _DEFAULT_PLATE_SCALE)
            plate_scale = _DEFAULT_PLATE_SCALE

        img = _rectify_orders(img, geometry, plate_scale_arcsec=plate_scale)
        # Propagate variance through the same rectification (identity
        # remapping with the zero-tilt placeholder preserves the variance
        # structure; the interpolation introduces sub-pixel smoothing
        # which is acceptable given the provisional tilt model).
        var = _rectify_orders(var, geometry, plate_scale_arcsec=plate_scale)
        tilt_provisional = True

    # ------------------------------------------------------------------
    # Return structured result
    # ------------------------------------------------------------------
    return {
        'image': img,
        'variance': var,
        'bitmask': bitmask,
        'hdrinfo': hdrinfo,
        'subtraction_mode': subtraction_mode,
        'flat_applied': True,
        'rectified': do_rectify,
        'tilt_provisional': tilt_provisional,
    }
