"""
iSHELL 2DXD detector diagnostics for pySpextool.

NASA IRTF iSHELL cross-dispersed echelle spectrograph.
Teledyne H2RG 2048 x 2048 near-IR detector.

This module provides lightweight diagnostic functions for inspecting raw
iSHELL calibration frames (QTH flat fields and ThAr arc frames).  The
routines are intentionally simple — they are meant to be used interactively
to understand detector geometry and order structure *before* implementing
order tracing and full 2DXD spectral extraction.

Functions
---------
plot_flat_orders
    Display a raw flat-field frame with optional horizontal-cut overlays
    and a median column profile.

plot_detector_cross_section
    Extract and plot a column-averaged vertical profile, useful for
    locating echelle orders by eye.

plot_arc_frame
    Display a raw ThAr arc frame so that line density and order spacing
    can be assessed visually.

Notes
-----
All three functions read only **extension 0** (the PRIMARY extension)
of iSHELL raw FITS files.  iSHELL raw files contain three extensions:

  - Extension 0 (PRIMARY)  : signal frame S = Σ pedestal − Σ signal reads
  - Extension 1 (SUM_PED)  : sum of pedestal reads
  - Extension 2 (SUM_SAM)  : sum of signal reads

For visual diagnostics the raw PRIMARY signal frame is sufficient.

See Also
--------
docs/ishell_2dxd_notes.md : Background notes on iSHELL order geometry and
    how these diagnostics prepare for order-tracing development.
"""

import logging

import matplotlib.pyplot as pl
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator

from pyspextool.io.check import check_parameter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Number of detector rows (spatial axis) on the iSHELL H2RG array.
NROWS: int = 2048

#: Number of detector columns (dispersion axis) on the iSHELL H2RG array.
NCOLS: int = 2048

#: Default matplotlib figure size for 2-D image diagnostics (inches).
_DEFAULT_IMAGE_FIGSIZE: tuple = (9, 9)

#: Default matplotlib figure size for 1-D profile diagnostics (inches).
_DEFAULT_PROFILE_FIGSIZE: tuple = (9, 5)

#: Default font size used in all diagnostic plots (points).
_DEFAULT_FONT_SIZE: int = 12


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_flat_orders(
    flat_file: str,
    cuts: list | None = None,
    show_column_profile: bool = True,
    figure_size: tuple = _DEFAULT_IMAGE_FIGSIZE,
    font_size: int = _DEFAULT_FONT_SIZE,
    output_fullpath: str | None = None,
    showblock: bool = False,
    plot_number: int | None = None,
) -> None:
    """
    Display a raw iSHELL QTH flat-field frame.

    Loads the PRIMARY extension of *flat_file*, scales the image with
    the ZScale algorithm (robust against outlier pixels), and optionally
    overlays horizontal cuts and/or a median column profile panel beneath
    the image.  The combination quickly reveals the echelle order pattern
    on the detector.

    Parameters
    ----------
    flat_file : str
        Full path to a raw iSHELL flat-field FITS file.

    cuts : list of int, optional
        Row indices at which to draw horizontal dashed lines on the image.
        Useful for marking the approximate centres of echelle orders found
        by eye.  Pass ``None`` (default) to skip.

    show_column_profile : bool, default True
        If ``True``, add a second panel below the image showing the median
        intensity along every row (i.e. the cross-dispersion profile).
        This collapses the 2-D flat along the column axis and reveals the
        brightness envelope of each echelle order.

    figure_size : tuple of (float, float), default (9, 9)
        Figure dimensions in inches.

    font_size : int, default 12
        Base font size used throughout the plot.

    output_fullpath : str, optional
        If provided, save the figure to this path instead of (or in
        addition to) displaying it on screen.

    showblock : bool, default False
        Passed to ``matplotlib.pyplot.show(block=…)``.  Set to ``True``
        to block the calling process until the window is closed.

    plot_number : int, optional
        Matplotlib figure number.  Reuses an existing figure window when
        provided.

    Returns
    -------
    None

    Examples
    --------
    >>> from pyspextool.ishell.diagnostics import plot_flat_orders
    >>> plot_flat_orders('data/testdata/ishell_h1_calibrations/raw/'
    ...                  'icm.2026A060.260214.flat.00050.a.fits',
    ...                  show_column_profile=True,
    ...                  showblock=True)
    """

    #
    # Validate parameters
    #

    check_parameter('plot_flat_orders', 'flat_file', flat_file, 'str')
    check_parameter('plot_flat_orders', 'cuts', cuts, ['NoneType', 'list'])
    check_parameter('plot_flat_orders', 'show_column_profile',
                    show_column_profile, 'bool')
    check_parameter('plot_flat_orders', 'figure_size', figure_size, 'tuple')
    check_parameter('plot_flat_orders', 'font_size', font_size,
                    ['int', 'float'])
    check_parameter('plot_flat_orders', 'output_fullpath', output_fullpath,
                    ['NoneType', 'str'])
    check_parameter('plot_flat_orders', 'showblock', showblock, 'bool')
    check_parameter('plot_flat_orders', 'plot_number', plot_number,
                    ['NoneType', 'int'])

    #
    # Load the FITS frame (primary extension only)
    #

    image, header = _load_primary_extension(flat_file)
    title = _make_title(header, flat_file, label='Flat field')
    logger.debug('Loaded flat: %s  shape=%s', flat_file, image.shape)

    #
    # Build the figure
    #

    font = {'weight': 'normal', 'size': font_size}
    rc('font', **font)

    fig = pl.figure(num=plot_number, figsize=figure_size)
    pl.clf()

    if show_column_profile:
        ax_img, ax_prof = fig.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [3, 1]},
        )
    else:
        ax_img = fig.add_subplot(111)

    # --- 2-D image panel ---

    vmin, vmax = _zscale_limits(image)
    ax_img.imshow(
        image,
        vmin=vmin,
        vmax=vmax,
        cmap='gray',
        origin='lower',
        aspect='auto',
    )
    ax_img.set_xlabel('Columns (pixels)')
    ax_img.set_ylabel('Rows (pixels)')
    ax_img.set_title(title, fontsize=font_size)
    _style_axes(ax_img)

    # Overlay horizontal cut lines if requested
    if cuts is not None:
        for row in cuts:
            ax_img.axhline(row, color='cyan', linestyle='--', linewidth=0.8,
                           alpha=0.7)

    # --- Median column profile panel ---

    if show_column_profile:
        row_indices = np.arange(image.shape[0])

        # Median across all columns → cross-dispersion brightness profile.
        # This collapses dispersion (column) axis so that echelle orders
        # appear as peaks along the row (spatial) axis.
        median_profile = np.nanmedian(image, axis=1)

        ax_prof.plot(row_indices, median_profile, color='steelblue',
                     linewidth=0.8)
        ax_prof.set_xlabel('Rows (pixels)')
        ax_prof.set_ylabel('Median DN')
        ax_prof.set_title('Median column profile (cross-dispersion)',
                          fontsize=font_size - 1)
        ax_prof.set_xlim(0, image.shape[0] - 1)
        _style_axes(ax_prof)

        if cuts is not None:
            for row in cuts:
                ax_prof.axvline(row, color='cyan', linestyle='--',
                                linewidth=0.8, alpha=0.7)

    pl.tight_layout()

    #
    # Output
    #

    if output_fullpath is not None:
        pl.savefig(output_fullpath)
        logger.info('Flat diagnostic saved to %s', output_fullpath)

    pl.show(block=showblock)
    if not showblock:
        pl.pause(1)


def plot_detector_cross_section(
    flat_file: str,
    column: int = 1024,
    width: int = 20,
    mark_peaks: bool = True,
    figure_size: tuple = _DEFAULT_PROFILE_FIGSIZE,
    font_size: int = _DEFAULT_FONT_SIZE,
    output_fullpath: str | None = None,
    showblock: bool = False,
    plot_number: int | None = None,
) -> None:
    """
    Plot a column-averaged vertical profile to identify echelle orders.

    This diagnostic is used to locate echelle orders in iSHELL flat-field
    frames.  Rather than extracting a single detector column — which is
    sensitive to individual bad pixels, cosmic rays, and line-structure
    artefacts — the function averages the signal over a band of columns
    (``column - width`` to ``column + width``) and plots the resulting
    median profile against row number.

    Echelle orders appear as broad peaks in the profile, separated by
    lower-signal inter-order gaps.  Candidate order centres can be
    highlighted automatically by passing ``mark_peaks=True``.

    Parameters
    ----------
    flat_file : str
        Full path to a raw iSHELL flat-field FITS file.

    column : int, default 1024
        Zero-based index of the *central* column of the averaging band.
        The default (1024) is the centre of the 2048-pixel detector.

    width : int, default 20
        Half-width of the column band used for averaging.  The profile is
        computed as the median over columns
        ``max(0, column - width) : min(ncols, column + width + 1)``.
        Wider bands suppress noise more aggressively; narrower bands
        localise the measurement.

    mark_peaks : bool, default True
        If ``True``, detect local maxima in the averaged profile using
        ``scipy.signal.find_peaks`` and mark candidate order centres with
        vertical dashed lines.

    figure_size : tuple of (float, float), default (9, 5)
        Figure dimensions in inches.

    font_size : int, default 12
        Base font size used throughout the plot.

    output_fullpath : str, optional
        If provided, save the figure to this path.

    showblock : bool, default False
        Passed to ``matplotlib.pyplot.show(block=…)``.

    plot_number : int, optional
        Matplotlib figure number.

    Returns
    -------
    None

    Examples
    --------
    >>> from pyspextool.ishell.diagnostics import plot_detector_cross_section
    >>> plot_detector_cross_section(
    ...     'data/testdata/ishell_h1_calibrations/raw/'
    ...     'icm.2026A060.260214.flat.00050.a.fits',
    ...     column=1024,
    ...     width=20,
    ...     showblock=True,
    ... )
    """

    #
    # Validate parameters
    #

    check_parameter('plot_detector_cross_section', 'flat_file', flat_file,
                    'str')
    check_parameter('plot_detector_cross_section', 'column', column, 'int')
    check_parameter('plot_detector_cross_section', 'width', width, 'int')
    check_parameter('plot_detector_cross_section', 'mark_peaks', mark_peaks,
                    'bool')
    check_parameter('plot_detector_cross_section', 'figure_size', figure_size,
                    'tuple')
    check_parameter('plot_detector_cross_section', 'font_size', font_size,
                    ['int', 'float'])
    check_parameter('plot_detector_cross_section', 'output_fullpath',
                    output_fullpath, ['NoneType', 'str'])
    check_parameter('plot_detector_cross_section', 'showblock', showblock,
                    'bool')
    check_parameter('plot_detector_cross_section', 'plot_number', plot_number,
                    ['NoneType', 'int'])

    #
    # Load the FITS frame
    #

    image, header = _load_primary_extension(flat_file)

    ncols = image.shape[1]
    if not (0 <= column < ncols):
        raise ValueError(
            f'column={column} is out of range for image with {ncols} columns.'
        )

    # Clamp the column band to valid array bounds
    col_lo = max(0, column - width)
    col_hi = min(ncols, column + width + 1)
    actual_width = col_hi - col_lo

    # Column-averaged profile: median across the selected column band.
    # Using the median rather than the mean suppresses hot pixels and
    # cosmic rays that would dominate a simple column slice.
    profile = np.nanmedian(image[:, col_lo:col_hi], axis=1)
    row_indices = np.arange(image.shape[0])

    title = _make_title(
        header, flat_file,
        label=f'Column-averaged detector cross section '
              f'(cols {col_lo}–{col_hi - 1}, n={actual_width})',
    )
    logger.debug(
        'Cross-section: file=%s  col_lo=%d  col_hi=%d',
        flat_file, col_lo, col_hi,
    )

    #
    # Build the figure
    #

    font = {'weight': 'normal', 'size': font_size}
    rc('font', **font)

    fig = pl.figure(num=plot_number, figsize=figure_size)
    pl.clf()
    ax = fig.add_subplot(111)

    ax.plot(row_indices, profile, color='steelblue', linewidth=0.8)
    ax.set_xlabel('Rows (pixels)')
    ax.set_ylabel('Median signal (DN)')
    ax.set_title(title, fontsize=font_size)
    ax.set_xlim(0, image.shape[0] - 1)
    _style_axes(ax)

    # Optionally mark candidate order centres with vertical dashed lines.
    # Using both a prominence threshold and a minimum inter-peak distance
    # filters noise spikes and avoids flagging multiple peaks within a
    # single broad order.  The 10 % prominence and distance=50 defaults
    # are intentionally conservative starting points; users should inspect
    # the profile and adjust if needed.
    if mark_peaks:
        try:
            from scipy.signal import find_peaks

            peak_prominence = 0.10 * (np.nanmax(profile) - np.nanmin(profile))
            peaks, _ = find_peaks(
                profile,
                prominence=peak_prominence,
                distance=50,      # minimum separation between peaks (pixels)
            )

            for pk in peaks:
                ax.axvline(pk, color='tomato', linestyle='--',
                           linewidth=0.8, alpha=0.75)

            if len(peaks) > 0:
                logger.info(
                    'Candidate order centres at rows: %s',
                    ', '.join(str(p) for p in peaks),
                )
        except ImportError:
            logger.warning(
                'scipy not available; skipping peak detection '
                '(install scipy to enable mark_peaks).'
            )

    pl.tight_layout()

    #
    # Output
    #

    if output_fullpath is not None:
        pl.savefig(output_fullpath)
        logger.info('Cross-section diagnostic saved to %s', output_fullpath)

    pl.show(block=showblock)
    if not showblock:
        pl.pause(1)


def plot_arc_frame(
    arc_file: str,
    figure_size: tuple = _DEFAULT_IMAGE_FIGSIZE,
    font_size: int = _DEFAULT_FONT_SIZE,
    output_fullpath: str | None = None,
    showblock: bool = False,
    plot_number: int | None = None,
) -> None:
    """
    Display a raw iSHELL ThAr arc lamp frame.

    Loads the PRIMARY extension of *arc_file* and renders it with ZScale
    intensity mapping.  Because arc frames are dominated by narrow emission
    lines the ZScale algorithm is applied with a relatively high contrast
    parameter to prevent the faint continuum from being lost.

    The plot is useful for:

    * Assessing ThAr line density across each echelle order.
    * Checking whether orders are well-separated (inter-order gap visible).
    * Visually confirming that the dispersion axis is along columns.

    Parameters
    ----------
    arc_file : str
        Full path to a raw iSHELL ThAr arc FITS file.

    figure_size : tuple of (float, float), default (9, 9)
        Figure dimensions in inches.

    font_size : int, default 12
        Base font size used throughout the plot.

    output_fullpath : str, optional
        If provided, save the figure to this path.

    showblock : bool, default False
        Passed to ``matplotlib.pyplot.show(block=…)``.

    plot_number : int, optional
        Matplotlib figure number.

    Returns
    -------
    None

    Examples
    --------
    >>> from pyspextool.ishell.diagnostics import plot_arc_frame
    >>> plot_arc_frame('data/testdata/ishell_h1_calibrations/raw/'
    ...                'icm.2026A060.260214.arc.00055.a.fits',
    ...                showblock=True)
    """

    #
    # Validate parameters
    #

    check_parameter('plot_arc_frame', 'arc_file', arc_file, 'str')
    check_parameter('plot_arc_frame', 'figure_size', figure_size, 'tuple')
    check_parameter('plot_arc_frame', 'font_size', font_size, ['int', 'float'])
    check_parameter('plot_arc_frame', 'output_fullpath', output_fullpath,
                    ['NoneType', 'str'])
    check_parameter('plot_arc_frame', 'showblock', showblock, 'bool')
    check_parameter('plot_arc_frame', 'plot_number', plot_number,
                    ['NoneType', 'int'])

    #
    # Load the FITS frame
    #

    image, header = _load_primary_extension(arc_file)
    title = _make_title(header, arc_file, label='ThAr arc frame')
    logger.debug('Loaded arc: %s  shape=%s', arc_file, image.shape)

    #
    # Build the figure
    #

    font = {'weight': 'normal', 'size': font_size}
    rc('font', **font)

    fig = pl.figure(num=plot_number, figsize=figure_size)
    pl.clf()
    ax = fig.add_subplot(111)

    # Arc frames have high dynamic range; use ZScale with higher contrast
    # to reveal faint inter-order regions while preserving bright lines.
    vmin, vmax = _zscale_limits(image, contrast=0.15)

    ax.imshow(
        image,
        vmin=vmin,
        vmax=vmax,
        cmap='gray',
        origin='lower',
        aspect='auto',
    )
    ax.set_xlabel('Columns (pixels)')
    ax.set_ylabel('Rows (pixels)')
    ax.set_title(title, fontsize=font_size)
    _style_axes(ax)

    pl.tight_layout()

    #
    # Output
    #

    if output_fullpath is not None:
        pl.savefig(output_fullpath)
        logger.info('Arc diagnostic saved to %s', output_fullpath)

    pl.show(block=showblock)
    if not showblock:
        pl.pause(1)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_primary_extension(fits_file: str) -> tuple[npt.NDArray, fits.Header]:
    """
    Load the PRIMARY extension of an iSHELL raw FITS file.

    iSHELL writes three extensions per file.  Extension 0 (PRIMARY) holds
    the signal frame S = Σ pedestal reads − Σ signal reads and is the
    appropriate frame for visual diagnostics.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.

    Returns
    -------
    image : ndarray
        A (2048, 2048) float32 array containing the primary signal frame.
    header : astropy.io.fits.Header
        The FITS header of the PRIMARY extension.

    Raises
    ------
    FileNotFoundError
        If *fits_file* does not exist.
    OSError
        If the file cannot be opened as a valid FITS file.
    """

    with fits.open(fits_file, memmap=False) as hdul:
        header = hdul[0].header
        image = hdul[0].data.astype(np.float32)

    return image, header


def _zscale_limits(
    image: npt.NDArray,
    contrast: float = 0.25,
) -> tuple[float, float]:
    """
    Compute ZScale display limits for *image*.

    Uses :class:`astropy.visualization.ZScaleInterval` which fits a
    linear model to the sorted pixel distribution and clips at a
    fraction (*contrast*) of the fitted slope.  This is robust against
    hot pixels and cosmic rays.

    Parameters
    ----------
    image : ndarray
        The 2-D image array.
    contrast : float, default 0.25
        ZScale contrast parameter.  Smaller values give a wider
        (lower-contrast) display range.

    Returns
    -------
    vmin, vmax : (float, float)
        Lower and upper display limits.
    """

    interval = ZScaleInterval(contrast=contrast)
    vmin, vmax = interval.get_limits(image)
    return float(vmin), float(vmax)


def _make_title(
    header: fits.Header,
    filepath: str,
    label: str = '',
) -> str:
    """
    Build a concise plot title from the FITS header.

    Attempts to read the IRAFNAME, PASSBAND, and OBJECT keywords from
    *header*.  Falls back to the base filename when keywords are absent.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header of the PRIMARY extension.
    filepath : str
        Path to the source file (used as fallback label).
    label : str, default ''
        Short descriptor prepended to the title string (e.g. 'Flat field').

    Returns
    -------
    str
        A title string suitable for a matplotlib Axes.set_title call.
    """

    import os

    irafname = header.get('IRAFNAME', os.path.basename(filepath))
    passband = header.get('PASSBAND', '')
    obj = header.get('OBJECT', '')

    parts = [p for p in (label, irafname, passband, obj) if p]
    return '  |  '.join(parts)


def _style_axes(ax: pl.Axes) -> None:
    """
    Apply the standard pySpextool tick style to a matplotlib Axes.

    Sets minor ticks, inward-pointing ticks on all four sides, and
    matching tick widths consistent with the rest of the pySpextool
    plotting suite.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to style.
    """

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
        right=True, left=True, top=True, bottom=True,
        which='both', direction='in', width=1.5,
    )
    ax.tick_params(which='minor', length=3)
    ax.tick_params(which='major', length=5)
