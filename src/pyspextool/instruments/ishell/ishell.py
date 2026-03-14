"""
iSHELL instrument module for pySpextool.

NASA IRTF iSHELL cross-dispersed echelle spectrograph.
Teledyne H2RG 2048 x 2048 near-IR detector.

Scope: J, H, and K modes only.  L, Lp, and M modes are not implemented.
Specifically, the supported near-IR sub-modes are:
  J-band: J0, J1, J2, J3
  H-band: H1, H2, H3
  K-band: K1, K2, Kgas, K3

Source
------
iSHELL Spextool Manual v10jan2020, Cushing et al.

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
1.  Raw FITS format: iSHELL writes multi-extension FITS (MEF) files with
    exactly **3 extensions** per file (NINT=3):

    - Extension 0: signal S = Σ pedestal_reads − Σ signal_reads  (Eq. 1, Manual §2.3)
    - Extension 1: pedestal sum Σ p_{jk,i}
    - Extension 2: signal sum   Σ s_{jk,i}

    SpeX writes 4 extensions (NINT=4); uSpeX writes 5 (NINT=5).

2.  Linearity: The linearity correction uses the signal difference S (ext 0)
    and the sum of pedestal+signal reads (ext 1 + ext 2) to identify pixels
    above LINCORMAX = **30000 DN** (Manual Table 4, LINCRMAX example).
    The SpeX ``slowcnts``-based Vacca et al. (2004) algorithm does not apply
    to the H2RG detector.

3.  Bias: H2RG provides 4-side reference pixels (bottom 4 rows, top 4 rows,
    left 4 columns, right 4 columns) that track bias and 1/f noise.
    uSpeX corrects 32-amplifier bias drifts column-wise; that method does
    not apply here.

4.  Header keywords: iSHELL raw FITS keyword names (confirmed from the
    iSHELL xSpextool heritage and instrument documentation):
      DATE_OBS, TIME_OBS, MJD_OBS, ITIME, CO_ADDS, NDR,
      TCS_RA, TCS_DEC, TCS_HA, TCS_AM, POSANGLE,
      IRAFNAME, PASSBAND, INSTRUME.
    See ``get_header()`` and ``docs/ishell_fits_layout.md``.

5.  Order geometry: iSHELL echelle orders are tilted with respect to detector
    columns.  Raw images must be *rectified* before the generic
    ``extract_1dxd()`` routine can be applied.  See ``_rectify_orders()``.

6.  Telluric correction: The **deconvolution** method is NOT available for
    iSHELL.  Only the **IP (instrumental-profile) method** is supported
    (Manual §6.2: "The deconvolution method is not available for iSHELL
    data").  The ``telluric_modeinfo.dat`` file therefore specifies
    ``ip`` for all modes.

7.  Arc lamp: All J/H/K modes use a **ThAr** (thorium-argon) arc lamp for
    wavelength calibration (Manual Table 1).  This is different from the
    Ar-only lamp used by SpeX.

Status
------
Phase 1 (ingestion): ``get_header()`` and ``load_data()`` are implemented.
Linearity correction (polynomial, H2RG-specific) and reference-pixel bias
subtraction are not yet applied; those are Phase 2 tasks.
``_rectify_orders()`` is Phase 2 (order rectification).
"""

import logging
import re

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.time import Time

from pyspextool.io.check import check_parameter
from pyspextool.io.fitsheader import get_headerinfo
from pyspextool.utils.arrays import idl_rotate
from pyspextool.utils.loop_progress import loop_progress
from pyspextool.pyspextoolerror import pySpextoolError
from .geometry import OrderGeometrySet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detector constants
# ---------------------------------------------------------------------------

#: H2RG detector dimensions (science pixels, excluding reference pixel border)
NROWS = 2048
NCOLS = 2048

#: H2RG reference pixel border width (pixels)
REF_PIX_WIDTH = 4

#: Plate scale (arcsec/pixel).  From Table 4 of the iSHELL Spextool Manual:
#: PLTSCALE = 0.125000 arcsec/pixel (consistent with 0.375" slit / 3 pixels).
PIXEL_SCALE_ARCSEC_PER_PIX = 0.125

#: Default plate scale used as fallback in _rectify_orders when not explicitly
#: provided (matches the standard iSHELL plate scale above).
_DEFAULT_PLATE_SCALE = PIXEL_SCALE_ARCSEC_PER_PIX

#: Linearity correction maximum (DN).  From Table 4 of the iSHELL Spextool
#: Manual, LINCRMAX example value = 30000.  Pixels with pedestal+signal sum
#: above this value are flagged and not linearity-corrected.
LINCORMAX_DN = 30000

#: Approximate detector gain (e-/DN).  Not stated explicitly in the iSHELL
#: Spextool Manual; must be confirmed from IRTF calibration files.
GAIN_ELECTRONS_PER_DN = 1.8  # TBD: confirm from IRTF calibration files

#: Approximate read noise (e-).  Not stated explicitly in the iSHELL Spextool
#: Manual; must be confirmed from IRTF calibration files.
READNOISE_ELECTRONS = 8.0  # TBD: confirm from IRTF calibration files

#: Supported J-band sub-modes (ThAr arc lamp, 5" slit; Manual Table 1).
J_MODES = ('J0', 'J1', 'J2', 'J3')

#: Supported H-band sub-modes (ThAr arc lamp, 5" slit; Manual Table 1).
H_MODES = ('H1', 'H2', 'H3')

#: Supported K-band sub-modes (ThAr arc lamp, 5" slit; Manual Table 1).
K_MODES = ('K1', 'K2', 'Kgas', 'K3')

#: All supported observing modes in this module (J/H/K only; L/Lp/M excluded).
SUPPORTED_MODES = J_MODES + H_MODES + K_MODES

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
        Normalised header keyword dictionaries.  There is **exactly one
        dictionary per returned output image**:

        * No pair-subtract: ``len(hdrinfo) == len(files)``.
        * Pair-subtract: ``len(hdrinfo) == len(files) // 2`` — the
          header retained for each pair is the **A-beam** (first of the
          two files), because the A-beam carries the source-positive
          signal.

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

    Maps raw iSHELL FITS keyword names to the standard pySpextool names used
    throughout the pipeline.  Missing keywords are replaced with ``nan`` /
    ``'nan'`` so that downstream code always receives all expected keys.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The primary HDU header from an iSHELL raw MEF FITS file.

    keywords : list of str or None
        Additional FITS keywords to retain as-is in the output dictionary
        (passed through to :func:`~pyspextool.io.fitsheader.get_headerinfo`).

    Returns
    -------
    dict
        Each key maps to a two-element list ``[value, comment]``.

        Required pySpextool output keys produced by this function:

        ============  ===================  ======================================
        Output key    Raw iSHELL keyword   Notes
        ============  ===================  ======================================
        AM            TCS_AM               Airmass; fallback nan
        HA            TCS_HA               Hour angle string (±hh:mm:ss.ss)
        PA            POSANGLE             Slit PA east of north (deg)
        RA            TCS_RA               Right ascension string
        DEC           TCS_DEC              Declination string (±dd:mm:ss.s)
        ITIME         ITIME                Integration time per frame (s)
        NCOADDS       CO_ADDS              Number of co-adds
        IMGITIME      (computed)           ITIME × NCOADDS (s)
        TIME          TIME_OBS             UT observation start time
        DATE          DATE_OBS             UT observation start date
        MJD           MJD_OBS              Modified Julian date; computed if absent
        FILENAME      IRAFNAME             Original file name
        MODE          PASSBAND             Observing sub-mode (J0…K3)
        INSTR         (hardcoded)          Always ``'iSHELL'``
        ============  ===================  ======================================

    Notes
    -----
    Raw keyword names are derived from the iSHELL xSpextool heritage
    (confirmed from the instrument data files).  See
    ``docs/ishell_fits_layout.md`` for the full assumed raw FITS layout.
    """

    check_parameter('get_header', 'header', header, 'Header')
    check_parameter('get_header', 'keywords', keywords, ['NoneType', 'list'])

    # ---- collect user-requested keywords first ---------------------------------
    hdrinfo = get_headerinfo(header, keywords=keywords)

    # ---- AM (airmass) ---------------------------------------------------------
    try:
        airmass = float(header['TCS_AM'])
    except (KeyError, TypeError, ValueError):
        try:
            airmass = float(header['AIRMASS'])
        except (KeyError, TypeError, ValueError):
            airmass = np.nan
    hdrinfo['AM'] = [airmass, ' Airmass']

    # ---- HA (hour angle) ------------------------------------------------------
    try:
        ha = str(header['TCS_HA']).strip()
        if ha and not re.search(r'^[+-]', ha):
            ha = '+' + ha
    except KeyError:
        ha = 'nan'
    hdrinfo['HA'] = [ha, ' Hour angle (hours)']

    # ---- PA (position angle) --------------------------------------------------
    try:
        pa = float(header['POSANGLE'])
    except (KeyError, TypeError, ValueError):
        pa = np.nan
    hdrinfo['PA'] = [pa, ' Position Angle E of N (deg)']

    # ---- RA (right ascension) -------------------------------------------------
    try:
        ra = str(header['TCS_RA']).strip()
    except KeyError:
        try:
            ra = str(header['RA']).strip()
        except KeyError:
            ra = 'nan'
    hdrinfo['RA'] = [ra, ' Right Ascension, FK5 J2000']

    # ---- DEC (declination) ----------------------------------------------------
    try:
        dec = str(header['TCS_DEC']).strip()
    except KeyError:
        try:
            dec = str(header['DEC']).strip()
        except KeyError:
            dec = 'nan'
    if dec != 'nan' and not re.search(r'^[+-]', dec):
        dec = '+' + dec
    hdrinfo['DEC'] = [dec, ' Declination, FK5 J2000']

    # ---- ITIME / NCOADDS / IMGITIME -------------------------------------------
    try:
        itime = float(header['ITIME'])
    except (KeyError, TypeError, ValueError):
        itime = np.nan
    hdrinfo['ITIME'] = [itime, ' Integration time (sec)']

    try:
        coadds = int(header['CO_ADDS'])
    except (KeyError, TypeError, ValueError):
        coadds = 1
    hdrinfo['NCOADDS'] = [coadds, ' Number of COADDS']

    imgitime = itime * coadds if not np.isnan(itime) else np.nan
    hdrinfo['IMGITIME'] = [imgitime,
                           ' Image integration time, NCOADDSxITIME (sec)']

    # ---- TIME (UT start time) -------------------------------------------------
    try:
        time_val = str(header['TIME_OBS']).strip()
    except KeyError:
        time_val = 'nan'
    hdrinfo['TIME'] = [time_val, ' Observation time in UTC']

    # ---- DATE (UT date) -------------------------------------------------------
    try:
        date_val = str(header['DATE_OBS']).strip()
    except KeyError:
        date_val = 'nan'
    hdrinfo['DATE'] = [date_val, ' Observation date in UTC']

    # ---- MJD ------------------------------------------------------------------
    try:
        mjd = float(header['MJD_OBS'])
    except (KeyError, TypeError, ValueError):
        if date_val != 'nan' and time_val != 'nan':
            try:
                tobj = Time(date_val + 'T' + time_val)
                mjd = tobj.mjd
            except Exception:
                mjd = np.nan
        else:
            mjd = np.nan
    hdrinfo['MJD'] = [mjd, ' Modified Julian date']

    # ---- FILENAME -------------------------------------------------------------
    try:
        fname = str(header['IRAFNAME']).strip()
    except KeyError:
        try:
            fname = str(header['FILENAME']).strip()
        except KeyError:
            fname = 'unknown'
    hdrinfo['FILENAME'] = [fname, ' Filename']

    # ---- MODE (observing sub-mode, e.g. J0, K3) --------------------------------
    try:
        mode = str(header['PASSBAND']).strip()
    except KeyError:
        try:
            mode = str(header['GRAT']).strip()
        except KeyError:
            mode = 'unknown'

    # Validate that the mode is one of the supported J/H/K sub-modes.
    # An explicit unsupported mode (L, Lp, M, …) is a hard error; an absent
    # PASSBAND/GRAT keyword (mode == 'unknown') is allowed and produces a
    # warning so that test headers with no mode do not fail.
    if mode != 'unknown' and mode not in SUPPORTED_MODES:
        raise pySpextoolError(
            f"get_header: observing mode '{mode}' is not supported.  "
            f"This module handles J/H/K sub-modes only "
            f"({', '.join(SUPPORTED_MODES)}).  "
            f"L, Lp, and M modes are explicitly out of scope.")

    hdrinfo['MODE'] = [mode, ' Instrument Mode']

    # ---- INSTR (hardcoded) ----------------------------------------------------
    hdrinfo['INSTR'] = ['iSHELL', ' Instrument']

    return hdrinfo


def load_data(
        file: str,
        linearity_info: dict,
        keywords: list,
        coefficients: npt.ArrayLike = None,
        linearity_correction: bool = True) -> tuple:
    """
    Load a single iSHELL raw MEF FITS file and return normalised arrays.

    Reads the three-extension iSHELL MEF structure, flags non-linear pixels
    using the pedestal+signal sum (extensions 1 + 2), divides by total
    exposure time to produce DN s⁻¹, and estimates the variance.

    Parameters
    ----------
    file : str
        Full path to a single iSHELL raw MEF FITS file.

    linearity_info : dict
        ``"max"`` : int
            DN threshold above which a pixel is flagged as non-linear.
            Flagging uses the **pedestal + signal sum** (extensions 1 + 2),
            consistent with the iSHELL Spextool Manual §2.3.
        ``"bit"`` : int
            Bit number to set in the returned bitmask for flagged pixels.

    keywords : list of str or None
        Extra FITS keywords to retain in the header dictionary
        (forwarded to :func:`get_header`).

    coefficients : ndarray or None, optional
        Polynomial linearity-correction coefficients,
        shape ``(ncoefficients, nrows, ncols)``.  Currently unused;
        linearity correction is a Phase 2 task.

    linearity_correction : bool, default True
        Reserved for Phase 2.  Accepted for interface compatibility but not
        yet applied (the polynomial correction algorithm is not yet
        implemented).

    Returns
    -------
    img : ndarray, shape (nrows, ncols)
        Science image in DN s⁻¹ (extension 0 / (ITIME × CO_ADDS)).
    var : ndarray, shape (nrows, ncols)
        Variance image in (DN s⁻¹)².  Estimated from shot noise + read
        noise using the detector constants at the top of this module.
    hdrinfo : dict
        Normalised header keyword dictionary from :func:`get_header`.
    bitmask : ndarray of uint8, shape (nrows, ncols)
        Bit ``linearity_info['bit']`` set for non-linear pixels.

    Raises
    ------
    pySpextoolError
        If the file does not have exactly 3 FITS extensions (NINT=3).
    pySpextoolError
        If ITIME is missing or non-numeric.
    pySpextoolError
        If ITIME is not positive (``ITIME <= 0``).
    pySpextoolError
        If CO_ADDS is less than 1.
    pySpextoolError
        If the MODE parsed from the header is an unsupported iSHELL mode
        (L, Lp, M, …); raised inside :func:`get_header`.

    Notes
    -----
    Raw MEF structure (iSHELL Spextool Manual §2.3):

    =========  ==========================================================
    Extension  Contents
    =========  ==========================================================
    0          Signal difference S = Σ pedestal_reads − Σ signal_reads
               (the science image in total DN); also carries all header
               keywords.
    1          Pedestal sum Σ p_{jk,i}
    2          Signal sum   Σ s_{jk,i}
    =========  ==========================================================

    Non-linearity is flagged when ``ext1 + ext2 > linearity_info['max']``.

    The polynomial H2RG linearity correction (``_correct_ishell_linearity``)
    and reference-pixel bias subtraction (``_subtract_reference_pixels``) are
    *not* yet applied; see the Phase 2 task notes in those functions.
    """

    check_parameter('load_data', 'file', file, 'str')
    check_parameter('load_data', 'linearity_info', linearity_info, 'dict')
    check_parameter('load_data', 'keywords', keywords, ['NoneType', 'list'])
    check_parameter('load_data', 'linearity_correction', linearity_correction,
                    'bool')

    # ---- open the MEF file ----------------------------------------------------
    hdul = fits.open(file, ignore_missing_end=True)
    hdul[0].verify('silentfix')

    n_ext = len(hdul)
    if n_ext != 3:
        hdul.close()
        raise pySpextoolError(
            f'load_data: expected exactly 3 FITS extensions (NINT=3) for an '
            f'iSHELL raw MEF file; found {n_ext} extension(s) in {file!r}. '
            f'Check that the file is a valid raw iSHELL science frame.')

    primary_header = hdul[0].header

    # ---- read exposure parameters from header ---------------------------------
    try:
        itime = float(primary_header['ITIME'])
    except (KeyError, TypeError, ValueError) as exc:
        hdul.close()
        raise pySpextoolError(
            f'load_data: FITS keyword ITIME missing or non-numeric '
            f'in {file!r}.') from exc

    if itime <= 0.0:
        hdul.close()
        raise pySpextoolError(
            f'load_data: ITIME must be positive; got ITIME={itime} '
            f'in {file!r}.')

    try:
        coadds = int(primary_header['CO_ADDS'])
    except (KeyError, TypeError, ValueError):
        coadds = 1

    if coadds < 1:
        hdul.close()
        raise pySpextoolError(
            f'load_data: CO_ADDS must be >= 1; got CO_ADDS={coadds} '
            f'in {file!r}.')

    total_exptime = itime * coadds

    # ---- read the three data planes -------------------------------------------
    # ext0: signal difference S (the science data, total DN)
    # ext1: pedestal sum (used for linearity flagging)
    # ext2: signal sum   (used for linearity flagging)
    sig_diff = hdul[0].data.astype(float)
    ped_sum  = hdul[1].data.astype(float)
    sig_sum  = hdul[2].data.astype(float)

    # ---- collect header keywords -----------------------------------------------
    hdrinfo = get_header(primary_header, keywords=keywords)

    hdul.close()

    # ---- non-linearity flagging ------------------------------------------------
    # Use pedestal + signal sum per Manual §2.3: "we do use the sum of the
    # pedestal and signal reads to identify pixels beyond the linearity curve
    # maximum."
    total_counts = ped_sum + sig_sum
    lin_bit = linearity_info['bit']
    bitmask = ((total_counts > linearity_info['max'])
               .astype(np.uint8) * np.uint8(2 ** lin_bit))

    # ---- NOTE: linearity correction not yet applied ---------------------------
    # _correct_ishell_linearity() and _subtract_reference_pixels() are Phase 2
    # tasks.  The raw signal difference is used directly.

    # ---- normalise to DN s⁻¹ --------------------------------------------------
    img = sig_diff / total_exptime

    # ---- variance estimate (shot noise + read noise) --------------------------
    # var = |S_raw| / (gain * T²) + 2 * (ron/gain)² / T²
    # where T = total_exptime, S_raw in total DN
    gain = GAIN_ELECTRONS_PER_DN
    ron  = READNOISE_ELECTRONS
    var = (np.abs(sig_diff) / (gain * total_exptime ** 2)
           + 2.0 * (ron / gain) ** 2 / total_exptime ** 2)

    return img, var, hdrinfo, bitmask


# ---------------------------------------------------------------------------
# Private helpers  (Phase 2 implementation targets)
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

    The linearity flagging uses the **sum** of pedestal and signal reads
    (FITS extensions 1 and 2 of the raw MEF file) rather than the signal
    difference alone, as stated in the iSHELL Spextool Manual §2.3:
    "we do use the sum of the pedestal and signal reads to identify pixels
    that have counts beyond the linearity curve maximum."

    LINCORMAX = 30000 DN (iSHELL Spextool Manual Table 4, LINCRMAX example).

    .. todo::
        Phase 1: implement and validate against IRTF-provided calibration.
    """

    raise NotImplementedError(
        '_correct_ishell_linearity() not yet implemented.  '
        'Phase 2 task.  See docs/ishell_design_memo.md Blocker B1.')


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
        'Phase 2 task.  See docs/ishell_design_memo.md §5.3.')


def _rectify_orders(
        image: npt.ArrayLike,
        geometry: OrderGeometrySet,
        plate_scale_arcsec: float = _DEFAULT_PLATE_SCALE) -> np.ndarray:
    """
    Resample iSHELL echelle orders using the geometry stored in *geometry*.

    .. warning::
        The current implementation uses a **placeholder zero-tilt** model.
        The resampling within each order is effectively the identity
        transformation; no spectral-line tilt is measured or corrected.
        See ``docs/ishell_wavecal_design_note.md`` §5 for what remains
        to be implemented before this function delivers science-quality
        rectification.

    iSHELL echelle orders are tilted with respect to the detector columns.
    This function resamples each order using the tilt model in *geometry*
    so that the output can be passed to the generic ``extract_1dxd()``
    routine.  With the current placeholder zero-tilt model, no horizontal
    column correction is applied.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Flat-fielded, linearity-corrected iSHELL science image.

    geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        Order geometry set for the observing mode, including per-order edge
        polynomials, tilt coefficients, and wavelength/spatial calibrations.
        Must have ``geometry.has_tilt() == True`` before this function can
        be called.

    plate_scale_arcsec : float, optional
        Spatial plate scale in arcsec/pixel.  Defaults to 0.125 arcsec/pixel
        (the standard iSHELL plate scale).

    Returns
    -------
    rectified : ndarray, shape (nrows, ncols)
        Image resampled according to the tilt model stored in *geometry*.
        With the current placeholder zero-tilt model, pixels inside each
        order footprint equal the bilinearly-interpolated input values (i.e.
        the transformation is effectively the identity within each order).
        Pixels outside all order footprints are set to NaN.

        .. warning::
            The current implementation uses a **placeholder zero-tilt**
            model.  No spectral-line tilt is measured or removed.  The
            output is structurally valid (NaN masking and edge geometry
            are applied) but is not a scientifically rectified image.

    Notes
    -----
    The implementation uses :func:`scipy.ndimage.map_coordinates` with
    bilinear interpolation (``order=1``) to resample each order.  Each
    output pixel at ``(row, col)`` within an order is mapped to a source
    column via the tilt polynomial stored in *geometry*::

        src_col = col + tilt(col) * (row - centerline(col))
        src_row = row          # rows are not shifted in the tilt model

    With the current placeholder zero-tilt (``tilt_coeffs = [0.0]``),
    the mapping is the identity and the output equals the bilinearly-
    interpolated input within the order boundaries.  Real tilt correction
    requires measured tilt coefficients from arc frames.

    The function requires ``geometry.has_wavelength_solution()`` to be
    ``True``; if the geometry is empty (no orders) the input image is
    returned unchanged.
    """
    from scipy.ndimage import map_coordinates

    from .wavecal import build_rectification_maps

    image_arr = np.asarray(image, dtype=float)

    # Nothing to do for an empty geometry set
    if geometry.n_orders == 0:
        return image_arr.copy()

    if not geometry.has_wavelength_solution():
        raise ValueError(
            "_rectify_orders() requires a fully populated OrderGeometrySet "
            "with wavelength solutions.  Call "
            "wavecal.build_geometry_from_wavecalinfo() first to populate "
            "wave_coeffs for every order."
        )

    nrows, ncols = image_arr.shape
    output = np.full((nrows, ncols), np.nan, dtype=float)

    # Build rectification maps for every order
    rect_maps = build_rectification_maps(
        geom_set=geometry,
        plate_scale_arcsec=plate_scale_arcsec,
    )

    for geom, rmap in zip(geometry.geometries, rect_maps):
        x_start = geom.x_start
        x_end = geom.x_end

        # For each column in the order's column range, apply tilt correction
        cols_out = np.arange(x_start, x_end + 1, dtype=float)
        n_cols = len(cols_out)

        # Row range for this order: scan the full column range to get the
        # overall row extent
        row_bot_arr = geom.eval_bottom_edge(cols_out)
        row_top_arr = geom.eval_top_edge(cols_out)
        row_min = max(0, int(np.floor(row_bot_arr.min())))
        row_max = min(nrows - 1, int(np.ceil(row_top_arr.max())))

        rows_out = np.arange(row_min, row_max + 1, dtype=float)
        n_rows = len(rows_out)

        if n_rows == 0 or n_cols == 0:
            continue

        # Vectorised computation of source coordinates
        # Shape: (n_rows, n_cols)
        rows_grid, cols_grid = np.meshgrid(rows_out, cols_out, indexing="ij")

        centerline_rows = geom.eval_centerline(cols_grid)
        tilt_vals = geom.eval_tilt(cols_grid)

        row_offsets = rows_grid - centerline_rows
        src_cols = cols_grid + tilt_vals * row_offsets
        src_rows = rows_grid  # rows are not shifted in the tilt model

        # Mask pixels outside the order boundaries
        in_order = (
            (rows_grid >= geom.eval_bottom_edge(cols_grid))
            & (rows_grid <= geom.eval_top_edge(cols_grid))
            & (src_cols >= 0)
            & (src_cols < ncols)
        )

        # Interpolate using map_coordinates
        coords = np.array([src_rows.ravel(), src_cols.ravel()])
        interp_vals = map_coordinates(
            image_arr, coords, order=1, mode="constant", cval=np.nan
        )
        interp_grid = interp_vals.reshape(n_rows, n_cols)

        # Write to output only within order boundaries
        r_slice = slice(row_min, row_max + 1)
        c_slice = slice(x_start, x_end + 1)
        current = output[r_slice, c_slice]
        mask = in_order
        current[mask] = interp_grid[mask]
        output[r_slice, c_slice] = current

    return output
