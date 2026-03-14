"""
iSHELL instrument module for pySpextool.

NASA IRTF iSHELL cross-dispersed echelle spectrograph.
Teledyne H2RG 2048 x 2048 near-IR detector.

Scope: J, H, and K modes only.  L, Lp, and M modes are not implemented.

Interface contract
------------------
This module exposes three callables that are discovered at runtime by
``pyspextool.extract.load_image`` via ``importlib.import_module``:

    read_fits(files, linearity_info, keywords, pair_subtract,
              rotate, linearity_correction, extra, verbose)
        -> (data, var, hdrinfo, bitmask)

    get_header(header, keywords)
        -> dict

    load_data(file, linearity_info, keywords, coefficients)
        -> (data, var, hdrinfo, bitmask)

Key differences from SpeX / uSpeX
-----------------------------------
1.  Detector: Teledyne H2RG (2048 x 2048) vs. ALADDIN InSb (1024 x 1024).
    Linearity correction uses a pixel-by-pixel polynomial evaluated on the
    measured DN; the SpeX ``slowcnts``-based Vacca et al. (2004) algorithm
    does not apply.
2.  Bias: H2RG provides 4-side reference pixels (bottom 4 rows, top 4 rows,
    left 4 columns, right 4 columns) that are used for bias subtraction.
    uSpeX corrects 32-amplifier bias drifts column-wise; that method does
    not apply here.
3.  Header keywords: iSHELL uses different FITS keyword names (e.g.
    ``UTSTART`` instead of ``TIME_OBS``, ``CO_ADDS`` instead of ``DIVISOR``).
4.  Order geometry: iSHELL echelle orders are tilted with respect to detector
    columns.  Raw images must be *rectified* before the generic
    ``extract_1dxd()`` routine can be applied.  See ``rectify_orders()``.

Status
------
Phase 0 (scaffolding): public functions raise ``NotImplementedError`` where
instrument-specific logic has not yet been implemented.  All signatures and
docstrings are final; only the function bodies require completion.
"""

import logging
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.time import Time

from pyspextool.io.check import check_parameter
from pyspextool.io.fitsheader import get_headerinfo
from pyspextool.utils.arrays import idl_rotate
from pyspextool.utils.loop_progress import loop_progress
from pyspextool.pyspextoolerror import pySpextoolError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detector constants (TBD: confirm from IRTF documentation / real data)
# ---------------------------------------------------------------------------

#: H2RG detector dimensions (science pixels, excluding reference pixel border)
NROWS = 2048
NCOLS = 2048

#: H2RG reference pixel border width (pixels)
REF_PIX_WIDTH = 4

#: Approximate detector gain (e-/DN).  Confirm with IRTF instrument team.
GAIN_ELECTRONS_PER_DN = 1.8  # TBD

#: Approximate read noise for Fowler-16 sampling (e-).  Confirm with IRTF.
READNOISE_ELECTRONS = 8.0  # TBD

#: Supported observing modes in this module (J/H/K only; L/Lp/M excluded).
SUPPORTED_MODES = ('J', 'H', 'K')

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def read_fits(
        files: list,
        linearity_info: dict,
        keywords: list = None,
        pair_subtract: bool = False,
        rotate: int = 0,
        linearity_correction: bool = True,
        extra: dict = None,
        verbose: bool = False):
    """
    Read one or more iSHELL FITS image files.

    Parameters
    ----------
    files : list of str
        Full paths to iSHELL FITS files.

    linearity_info : dict
        ``"max"`` : int
            DN threshold above which a pixel is flagged as non-linear.
        ``"bit"`` : int
            Bitmask bit to set for non-linear pixels.

    keywords : list of str, optional
        Additional FITS keywords to retain beyond the standard set.

    pair_subtract : bool, default False
        If True, subtract consecutive image pairs (A - B nod pattern).

    rotate : int in {0 … 7}, default 0
        IDL-rotate sense applied after loading:

        =====  =========  ===========================
        Value  Transpose  Rotation (counterclockwise)
        =====  =========  ===========================
        0      No         0°
        1      No         90°
        2      No         180°
        3      No         270°
        4      Yes        0°
        5      Yes        90°
        6      Yes        180°
        7      Yes        270°
        =====  =========  ===========================

    linearity_correction : bool, default True
        If True, apply the H2RG polynomial linearity correction.

    extra : dict, optional
        Reserved for future instrument-specific options.

    verbose : bool, default False
        Print progress messages.

    Returns
    -------
    data : ndarray, shape (nimages, nrows, ncols)  or  (nrows, ncols)
        Flat-fielded science images in DN s⁻¹.
    var : ndarray, same shape as *data*
        Variance images in (DN s⁻¹)².
    hdrinfo : list of dict
        Normalised header keyword dictionaries (one per image).
    bitmask : ndarray, same shape as *data*
        Integer bitmask (non-linear pixels flagged).
    """

    check_parameter('read_fits', 'files', files, 'list')
    check_parameter('read_fits', 'linearity_info', linearity_info, 'dict')
    check_parameter('read_fits', 'keywords', keywords, ['NoneType', 'list'])
    check_parameter('read_fits', 'pair_subtract', pair_subtract, 'bool')
    check_parameter('read_fits', 'rotate', rotate, 'int')
    check_parameter('read_fits', 'linearity_correction', linearity_correction,
                    'bool')
    check_parameter('read_fits', 'verbose', verbose, 'bool')

    nfiles = len(files)
    if pair_subtract and nfiles % 2 != 0:
        raise pySpextoolError(
            'read_fits: pair_subtract requires an even number of files; '
            f'{nfiles} files were provided.')

    data_list = []
    var_list = []
    hdrinfo_list = []
    bitmask_list = []

    nimages = nfiles // 2 if pair_subtract else nfiles
    indices = range(0, nfiles, 2) if pair_subtract else range(nfiles)

    for i, idx in enumerate(indices):
        if verbose:
            loop_progress(i, 0, nimages)

        img_a, var_a, hdr_a, mask_a = load_data(
            files[idx], linearity_info, keywords,
            coefficients=None,
            linearity_correction=linearity_correction)

        if pair_subtract:
            img_b, var_b, hdr_b, mask_b = load_data(
                files[idx + 1], linearity_info, keywords,
                coefficients=None,
                linearity_correction=linearity_correction)
            img_a = img_a - img_b
            var_a = var_a + var_b
            mask_a = (mask_a | mask_b).astype(np.int8)

        if rotate != 0:
            img_a = idl_rotate(img_a, rotate)
            var_a = idl_rotate(var_a, rotate)
            mask_a = idl_rotate(mask_a, rotate)

        data_list.append(img_a)
        var_list.append(var_a)
        hdrinfo_list.append(hdr_a)
        bitmask_list.append(mask_a)

    data = np.squeeze(np.stack(data_list, axis=0))
    var = np.squeeze(np.stack(var_list, axis=0))
    bitmask = np.squeeze(np.stack(bitmask_list, axis=0))

    return data, var, hdrinfo_list, bitmask


def get_header(header: fits.Header, keywords: list) -> dict:
    """
    Extract and normalise iSHELL FITS header keywords.

    Maps iSHELL-specific keyword names to the standard pySpextool names used
    throughout the pipeline.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The primary HDU header from an iSHELL FITS file.

    keywords : list of str
        Additional keywords (beyond the standard set) to include in the
        output dictionary.  These are passed through without renaming.

    Returns
    -------
    dict
        Keys are pySpextool-standard names; values are header values.

    Notes
    -----
    Standard keywords normalised by this function:

    ============  ===================  ======================================
    Output key    iSHELL keyword       Notes
    ============  ===================  ======================================
    INSTRUME      INSTRUME             No rename needed
    MODE          PINSTMOD / MODE      Observing mode (J/H/K)
    FILENAME      IRAFNAME             Original IRAF file name
    OBSDATE       DATE-OBS             UT date (YYYY-MM-DD)
    OBSTIME       UTSTART              UT start time (HH:MM:SS.S)
    OBSMJD        MJD_OBS              MJD at start; computed if absent
    EXPTIME       ITIME                Integration time per frame (s)
    RA            RA                   Right ascension (deg)
    DEC           DEC                  Declination (deg)
    POSANGLE      POSANGLE             Slit position angle (deg E of N)
    HOURANG       TCS_HA               Hour angle (h)
    AIRMASS       TCS_AM               Airmass
    SLIT          SLIT                 Slit name
    GRATING       ECHNAME              Echelle grating name
    CO_ADDS       CO_ADDS              Number of co-adds
    ============  ===================  ======================================

    .. warning::
        Keyword names and availability must be confirmed against real iSHELL
        FITS headers.  The mapping above is based on IRTF public documentation
        and may be incomplete.
    """

    check_parameter('get_header', 'header', header, 'Header')
    check_parameter('get_header', 'keywords', keywords, ['NoneType', 'list'])

    raise NotImplementedError(
        'get_header() for iSHELL is not yet implemented.  '
        'Phase 1 task: map iSHELL FITS keyword names to pySpextool standard '
        'names and confirm against real iSHELL data.  '
        'See docs/ishell_design_memo.md §5.3 and Blocker B9.')


def load_data(
        file: str,
        linearity_info: dict,
        keywords: list,
        coefficients: npt.ArrayLike = None,
        linearity_correction: bool = True) -> tuple:
    """
    Load a single iSHELL FITS file, apply corrections, and return arrays.

    Parameters
    ----------
    file : str
        Full path to a single iSHELL FITS file.

    linearity_info : dict
        ``"max"`` : int   DN threshold for non-linearity flag.
        ``"bit"`` : int   Bitmask bit to set.

    keywords : list of str
        FITS keywords to retain in the header dictionary.

    coefficients : ndarray, optional
        Pixel-by-pixel polynomial linearity-correction coefficients,
        shape ``(ncoefficients, nrows, ncols)``.  If *None* and
        *linearity_correction* is True, the coefficients are loaded from
        the instrument package (if available).

    linearity_correction : bool, default True
        Apply polynomial linearity correction to the raw data.

    Returns
    -------
    data : ndarray, shape (nrows, ncols)
        Science image in DN s⁻¹ (divided by ITIME × CO_ADDS).
    var : ndarray, shape (nrows, ncols)
        Variance image in (DN s⁻¹)².
    hdrinfo : dict
        Normalised header keyword dictionary from ``get_header()``.
    bitmask : ndarray of int8, shape (nrows, ncols)
        Bitmask with non-linear pixels flagged.
    """

    check_parameter('load_data', 'file', file, 'str')
    check_parameter('load_data', 'linearity_info', linearity_info, 'dict')
    check_parameter('load_data', 'keywords', keywords, ['NoneType', 'list'])
    check_parameter('load_data', 'linearity_correction', linearity_correction,
                    'bool')

    raise NotImplementedError(
        'load_data() for iSHELL is not yet implemented.  '
        'Phase 1 task: implement H2RG reference-pixel bias subtraction, '
        'polynomial linearity correction, and variance estimation.  '
        'See docs/ishell_design_memo.md §5.3.')


# ---------------------------------------------------------------------------
# Private helpers  (Phase 1 implementation targets)
# ---------------------------------------------------------------------------


def _correct_ishell_linearity(
        image: npt.ArrayLike,
        coefficients: npt.ArrayLike,
        lincormax: int) -> tuple:
    """
    Apply a pixel-by-pixel polynomial linearity correction to an iSHELL image.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Raw DN image (after reference-pixel bias subtraction, before
        linearity correction).

    coefficients : ndarray, shape (ncoefficients, nrows, ncols)
        Polynomial coefficients in ascending order (c0 + c1*x + c2*x² + …).
        Derived from detector calibration data provided by IRTF.

    lincormax : int
        DN threshold; pixels above this value are flagged in the returned
        bitmask and not corrected.

    Returns
    -------
    corrected : ndarray, shape (nrows, ncols)
        Linearity-corrected image in DN.
    bitmask : ndarray of int8, shape (nrows, ncols)
        Pixels exceeding *lincormax* flagged with bit 0 set.

    Notes
    -----
    The Vacca et al. (2004) algorithm used for SpeX is **not applicable** to
    the H2RG detector.  The H2RG correction is a direct polynomial
    evaluation:  corrected = Σ_k coefficients[k] × image^k.

    .. todo::
        Phase 1: implement and validate against IRTF-provided calibration.
    """

    raise NotImplementedError(
        '_correct_ishell_linearity() not yet implemented.  '
        'Phase 1 task.  See docs/ishell_design_memo.md Blocker B1.')


def _subtract_reference_pixels(image: npt.ArrayLike) -> npt.ArrayLike:
    """
    Subtract H2RG reference-pixel bias from an iSHELL image.

    The H2RG detector provides 4-pixel-wide reference regions on all four
    edges of the array.  These pixels are not illuminated and track bias and
    1/f noise.  Their median (per-row or per-column, TBD) is subtracted from
    the science region.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Raw H2RG image including reference pixel border.

    Returns
    -------
    ndarray, shape (nrows, ncols)
        Bias-subtracted image.  Reference pixel rows/columns are zeroed.

    Notes
    -----
    .. todo::
        Phase 1: confirm reference pixel layout from IRTF data and implement.
        See docs/ishell_design_memo.md Blocker B6.
    """

    raise NotImplementedError(
        '_subtract_reference_pixels() not yet implemented.  '
        'Phase 1 task.  See docs/ishell_design_memo.md §5.3.')


def _rectify_orders(
        image: npt.ArrayLike,
        wavecalinfo: dict) -> npt.ArrayLike:
    """
    Rectify tilted iSHELL echelle orders onto a rectilinear grid.

    iSHELL echelle orders are tilted with respect to the detector columns.
    This function resamples each order onto a uniform wavelength-vs-spatial
    grid so that the generic ``extract_1dxd()`` routine can be applied.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Flat-fielded, linearity-corrected iSHELL science image.

    wavecalinfo : dict
        Wavelength-calibration information loaded from
        ``<Mode>_wavecalinfo.fits``.  Must contain tilt/curvature
        polynomial coefficients describing the order geometry.

    Returns
    -------
    rectified : ndarray, shape (nrows, ncols)
        Rectified image with orders aligned to detector rows.

    Notes
    -----
    This is the primary new piece of instrument-specific logic required for
    iSHELL and has no equivalent in the SpeX pipeline.

    .. todo::
        Phase 2: implement once ``<Mode>_wavecalinfo.fits`` format is
        confirmed.  See docs/ishell_design_memo.md §5.3 and Blocker B2.
    """

    raise NotImplementedError(
        '_rectify_orders() not yet implemented.  '
        'Phase 2 task.  See docs/ishell_design_memo.md §5.3 and Blocker B2.')
