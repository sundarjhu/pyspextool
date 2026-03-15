#!/usr/bin/env python
"""
ishell_diagnostics.py — iSHELL 2DXD calibration inspection script.

This script automatically loads example flat-field and arc frames from
the standard test-data location and runs the three diagnostic functions
defined in ``pyspextool.ishell.diagnostics``.  It is intended as a quick
sanity check that the test data are readable and that the diagnostic
plots render correctly.

Usage
-----
Run from the top-level repository directory::

    python scripts/ishell_diagnostics.py

All three plots will be displayed sequentially (non-blocking by default).
To write PNG files instead of displaying windows, set SAVE_PLOTS = True
below.

Data location
-------------
data/testdata/ishell_h1_calibrations/raw/

Files used:
  - icm.2026A060.260214.flat.00050.a.fits  (QTH flat field, H1 mode)
  - icm.2026A060.260214.arc.00055.a.fits   (ThAr arc frame, H1 mode)
"""

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the package is importable when running from the repo root without
# a formal installation step.
# ---------------------------------------------------------------------------

_repo_src = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src')
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from pyspextool.ishell.diagnostics import (   # noqa: E402
    plot_arc_frame,
    plot_detector_cross_section,
    plot_flat_orders,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s  %(name)s  %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _find_data_dir() -> str:
    """
    Return the absolute path to the raw test-data directory.

    Looks for the directory relative to this script's location so that
    the script works regardless of the current working directory.
    """

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    data_dir = os.path.join(
        repo_root,
        'data', 'testdata', 'ishell_h1_calibrations', 'raw',
    )
    return data_dir


def _pick_file(data_dir: str, pattern: str) -> str:
    """
    Return the first file in *data_dir* whose name contains *pattern*.

    Parameters
    ----------
    data_dir : str
        Directory to search.
    pattern : str
        Sub-string that the filename must contain.

    Returns
    -------
    str
        Full path to the matching file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """

    candidates = sorted(
        f for f in os.listdir(data_dir) if pattern in f
    )
    if not candidates:
        raise FileNotFoundError(
            f'No file matching "{pattern}" found in {data_dir}'
        )
    return os.path.join(data_dir, candidates[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(save_plots: bool = False, output_dir: str | None = None) -> None:
    """
    Run all three iSHELL diagnostic plots on the standard test dataset.

    Parameters
    ----------
    save_plots : bool, default False
        If ``True``, save PNG files to *output_dir* instead of displaying
        interactive windows.
    output_dir : str, optional
        Directory in which to write PNG files when *save_plots* is ``True``.
        Defaults to the current working directory.
    """

    data_dir = _find_data_dir()

    if not os.path.isdir(data_dir):
        logger.error(
            'Test data directory not found: %s\n'
            'Make sure Git LFS objects are fetched (git lfs pull).',
            data_dir,
        )
        sys.exit(1)

    flat_file = _pick_file(data_dir, 'flat')
    arc_file = _pick_file(data_dir, 'arc')

    logger.info('Using flat: %s', flat_file)
    logger.info('Using arc:  %s', arc_file)

    if save_plots:
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Flat-field overview with cross-dispersion profile
    # -----------------------------------------------------------------------

    logger.info('Plotting flat-field overview …')
    plot_flat_orders(
        flat_file,
        show_column_profile=True,
        showblock=False,
        plot_number=1,
        output_fullpath=(
            os.path.join(output_dir, 'flat_orders.png')
            if save_plots else None
        ),
    )

    # -----------------------------------------------------------------------
    # 2. Detector cross-section at the central column
    # -----------------------------------------------------------------------

    logger.info('Plotting detector cross-section …')
    plot_detector_cross_section(
        flat_file,
        column=1024,         # central column of the 2048-pixel detector
        width=20,            # half-width of the averaging band (width=20 → up to 41 columns)
        showblock=False,
        plot_number=2,
        output_fullpath=(
            os.path.join(output_dir, 'cross_section.png')
            if save_plots else None
        ),
    )

    # -----------------------------------------------------------------------
    # 3. ThAr arc frame overview
    # -----------------------------------------------------------------------

    logger.info('Plotting ThAr arc frame …')
    plot_arc_frame(
        arc_file,
        showblock=False,
        plot_number=3,
        output_fullpath=(
            os.path.join(output_dir, 'arc_frame.png')
            if save_plots else None
        ),
    )

    logger.info('All diagnostic plots complete.')


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Run iSHELL 2DXD calibration diagnostics on the standard '
            'test dataset.'
        ),
    )
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='Save PNG files instead of displaying interactive windows.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        metavar='DIR',
        help=(
            'Directory in which to write PNG files (default: current '
            'working directory).  Only used when --save is set.'
        ),
    )
    args = parser.parse_args()

    main(save_plots=args.save, output_dir=args.output_dir)
