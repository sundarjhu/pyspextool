"""
Write provisional wavecal/spatcal calibration products to FITS files.

This module implements the **ninth stage** of the iSHELL 2DXD reduction
scaffold: persisting the provisional wavelength-calibration (wavecal) and
spatial-calibration (spatcal) products produced by Stage 8 as structured
FITS calibration files.

What this module does
---------------------
* Accepts a :class:`~pyspextool.instruments.ishell.calibration_products.CalibrationProductSet`
  (the Stage-8 output).

* Writes a wavecal FITS file with one image extension per echelle order,
  containing the wavelength axis and the spatial-fraction axis.

* Writes a spatcal FITS file with one image extension per echelle order,
  containing the spatial-fraction axis and the detector-row mapping.

* Provides a convenience wrapper that writes both files at once.

What this module does NOT do (by design)
-----------------------------------------
* **No final Spextool FITS format compatibility** – the output files are
  clearly labeled as provisional scaffold products and use a simplified
  structure.  Conversion to the final Spextool calibration format is
  deferred to a later stage.

* **No WCS keywords** – the FITS headers contain only minimal metadata;
  WCS keywords are not written.

* **No flux calibration** – these files contain only coordinate axes;
  no flux or sensitivity data is included.

* **No tilt or curvature corrections** – inherited from the scaffold
  simplifications documented in
  ``docs/ishell_scaffold_constraints.md``.

* **No interactive wavelength refinement** – the wavelength axis is taken
  directly from the provisional Stage-8 output.

Relationship to prior stages
-----------------------------
Stage 8 (:mod:`~pyspextool.instruments.ishell.calibration_products`)
produces a :class:`~pyspextool.instruments.ishell.calibration_products.CalibrationProductSet`
whose :class:`~pyspextool.instruments.ishell.calibration_products.WaveCalProduct`
and :class:`~pyspextool.instruments.ishell.calibration_products.SpatCalProduct`
objects hold the provisional calibration axes.  This stage simply
serialises those axes to FITS so that downstream code can consume them
without re-running the full calibration chain.

FITS structure
--------------
Wavecal file (``*_wavecal.fits``):
  - Primary HDU : empty (no data); header records mode, number of orders,
    and a ``PRODTYPE = 'WAVECAL_PROVISIONAL'`` keyword.
  - Extension HDUs (one per order, named ``ORDER_<order_number>``):
    - Image data : 2-D float32 array of shape ``(2, n_spectral)`` where
      row 0 is the wavelength axis (µm) and row 1 is the spatial-fraction
      axis repeated to fill the row (or truncated to ``n_spectral`` if
      ``n_spatial != n_spectral``).

      .. note::
         Storing both axes in a 2-D image keeps the per-order data
         self-contained without requiring a binary-table extension.

    - Header keywords:
      ``ORDER``, ``PRODTYPE``, ``NSPECTRA``, ``NSPATIAL``, ``WAVMIN``,
      ``WAVMAX``, ``MODE``.

Spatcal file (``*_spatcal.fits``):
  - Primary HDU : empty; header records mode, number of orders, and
    ``PRODTYPE = 'SPATCAL_PROVISIONAL'``.
  - Extension HDUs (one per order, named ``ORDER_<order_number>``):
    - Image data : 2-D float32 array of shape ``(2, n_spatial)`` where
      row 0 is the spatial-fraction axis and row 1 is the detector-row
      mapping.
    - Header keywords:
      ``ORDER``, ``PRODTYPE``, ``NSPATIAL``, ``SFMIN``, ``SFMAX``,
      ``MODE``.

Public API
----------
- :func:`write_wavecal_fits` – write wavecal products to FITS.
- :func:`write_spatcal_fits` – write spatcal products to FITS.
- :func:`write_calibration_fits` – convenience wrapper that writes both.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np

try:
    import astropy.io.fits as fits
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "astropy is required for FITS I/O.  Install it with: pip install astropy"
    ) from exc

from .calibration_products import CalibrationProductSet

__all__ = [
    "write_wavecal_fits",
    "write_spatcal_fits",
    "write_calibration_fits",
]

# Keyword used to identify provisional scaffold products in FITS headers.
_WAVECAL_PRODTYPE = "WAVECAL_PROVISIONAL"
_SPATCAL_PRODTYPE = "SPATCAL_PROVISIONAL"

# ISO-8601 timestamp included in primary headers.
_DATETIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime(_DATETIME_FMT)


# ---------------------------------------------------------------------------
# write_wavecal_fits
# ---------------------------------------------------------------------------


def write_wavecal_fits(
    calibration_products: CalibrationProductSet,
    output_path: str,
) -> None:
    """Write provisional wavelength calibration products to a FITS file.

    Creates one image extension per echelle order.  Each extension stores a
    ``(2, n_spectral)`` float32 array:

    - Row 0 : wavelength axis in µm.
    - Row 1 : spatial-fraction axis (``n_spatial`` values, padded or
      truncated to ``n_spectral`` to form a rectangular 2-D array).

    Parameters
    ----------
    calibration_products : :class:`~pyspextool.instruments.ishell.calibration_products.CalibrationProductSet`
        Stage-8 output to serialise.  Must contain at least one order.
    output_path : str
        Full path for the output FITS file.  Parent directories must
        already exist; the file is written atomically by *astropy*.

    Raises
    ------
    ValueError
        If *calibration_products* contains no orders.
    OSError
        If the parent directory of *output_path* does not exist or is not
        writable.
    """
    if calibration_products.n_orders == 0:
        raise ValueError(
            "calibration_products is empty (n_orders == 0); "
            "cannot write wavecal FITS."
        )

    parent = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(parent):
        raise OSError(
            f"Parent directory does not exist: {parent!r}"
        )

    hdul = fits.HDUList()

    # ------------------------------------------------------------------
    # Primary HDU
    # ------------------------------------------------------------------
    primary_header = fits.Header()
    primary_header["SIMPLE"] = True
    primary_header["PRODTYPE"] = (
        _WAVECAL_PRODTYPE,
        "Provisional scaffold wavecal product",
    )
    primary_header["MODE"] = (
        calibration_products.mode,
        "iSHELL observing mode",
    )
    primary_header["NORDERS"] = (
        calibration_products.n_orders,
        "Number of echelle orders",
    )
    primary_header["DATE"] = (_utc_now(), "UTC date/time of file creation")
    primary_header["COMMENT"] = (
        "Provisional wavecal scaffold product.  "
        "Not compatible with final Spextool FITS format."
    )
    hdul.append(fits.PrimaryHDU(header=primary_header))

    # ------------------------------------------------------------------
    # One image extension per order
    # ------------------------------------------------------------------
    for wcp in calibration_products.wavecal_products:
        n_spectral = wcp.n_spectral
        n_spatial = wcp.n_spatial

        # Pad or truncate spatial_frac to match spectral axis length so
        # that we can store both in a single 2-D image array.
        if n_spatial <= n_spectral:
            spatial_row = np.full(n_spectral, np.nan, dtype=np.float32)
            spatial_row[:n_spatial] = wcp.spatial_frac.astype(np.float32)
        else:
            spatial_row = wcp.spatial_frac[:n_spectral].astype(np.float32)

        data = np.vstack(
            [
                wcp.wavelength_um.astype(np.float32),
                spatial_row,
            ]
        )  # shape (2, n_spectral)

        ext_header = fits.Header()
        ext_header["ORDER"] = (wcp.order, "Echelle order number")
        ext_header["PRODTYPE"] = (
            _WAVECAL_PRODTYPE,
            "Provisional scaffold wavecal product",
        )
        ext_header["NSPECTRA"] = (n_spectral, "Length of wavelength axis")
        ext_header["NSPATIAL"] = (n_spatial, "Length of spatial axis")
        ext_header["WAVMIN"] = (
            float(np.nanmin(wcp.wavelength_um)),
            "Min wavelength (um)",
        )
        ext_header["WAVMAX"] = (
            float(np.nanmax(wcp.wavelength_um)),
            "Max wavelength (um)",
        )
        ext_header["MODE"] = (
            calibration_products.mode,
            "iSHELL observing mode",
        )
        ext_header["COMMENT"] = (
            "Row 0: wavelength axis (um).  Row 1: spatial_frac axis "
            "(NaN-padded if n_spatial < n_spectral)."
        )

        hdul.append(
            fits.ImageHDU(
                data=data,
                header=ext_header,
                name=f"ORDER_{wcp.order}",
            )
        )

    hdul.writeto(output_path, overwrite=True)


# ---------------------------------------------------------------------------
# write_spatcal_fits
# ---------------------------------------------------------------------------


def write_spatcal_fits(
    calibration_products: CalibrationProductSet,
    output_path: str,
) -> None:
    """Write provisional spatial calibration products to a FITS file.

    Creates one image extension per echelle order.  Each extension stores a
    ``(2, n_spatial)`` float32 array:

    - Row 0 : spatial-fraction axis in ``[0, 1]``.
    - Row 1 : detector-row mapping (provisional proxy at this scaffold
      stage).

    Parameters
    ----------
    calibration_products : :class:`~pyspextool.instruments.ishell.calibration_products.CalibrationProductSet`
        Stage-8 output to serialise.  Must contain at least one order.
    output_path : str
        Full path for the output FITS file.  Parent directories must
        already exist.

    Raises
    ------
    ValueError
        If *calibration_products* contains no orders.
    OSError
        If the parent directory of *output_path* does not exist or is not
        writable.
    """
    if calibration_products.n_orders == 0:
        raise ValueError(
            "calibration_products is empty (n_orders == 0); "
            "cannot write spatcal FITS."
        )

    parent = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(parent):
        raise OSError(
            f"Parent directory does not exist: {parent!r}"
        )

    hdul = fits.HDUList()

    # ------------------------------------------------------------------
    # Primary HDU
    # ------------------------------------------------------------------
    primary_header = fits.Header()
    primary_header["SIMPLE"] = True
    primary_header["PRODTYPE"] = (
        _SPATCAL_PRODTYPE,
        "Provisional scaffold spatcal product",
    )
    primary_header["MODE"] = (
        calibration_products.mode,
        "iSHELL observing mode",
    )
    primary_header["NORDERS"] = (
        calibration_products.n_orders,
        "Number of echelle orders",
    )
    primary_header["DATE"] = (_utc_now(), "UTC date/time of file creation")
    primary_header["COMMENT"] = (
        "Provisional spatcal scaffold product.  "
        "Not compatible with final Spextool FITS format."
    )
    hdul.append(fits.PrimaryHDU(header=primary_header))

    # ------------------------------------------------------------------
    # One image extension per order
    # ------------------------------------------------------------------
    for scp in calibration_products.spatcal_products:
        data = np.vstack(
            [
                scp.spatial_frac.astype(np.float32),
                scp.detector_rows.astype(np.float32),
            ]
        )  # shape (2, n_spatial)

        ext_header = fits.Header()
        ext_header["ORDER"] = (scp.order, "Echelle order number")
        ext_header["PRODTYPE"] = (
            _SPATCAL_PRODTYPE,
            "Provisional scaffold spatcal product",
        )
        ext_header["NSPATIAL"] = (scp.n_spatial, "Length of spatial axis")
        ext_header["SFMIN"] = (
            float(np.nanmin(scp.spatial_frac)),
            "Min spatial fraction",
        )
        ext_header["SFMAX"] = (
            float(np.nanmax(scp.spatial_frac)),
            "Max spatial fraction",
        )
        ext_header["MODE"] = (
            calibration_products.mode,
            "iSHELL observing mode",
        )
        ext_header["COMMENT"] = (
            "Row 0: spatial_frac axis [0,1].  "
            "Row 1: detector_rows (provisional)."
        )

        hdul.append(
            fits.ImageHDU(
                data=data,
                header=ext_header,
                name=f"ORDER_{scp.order}",
            )
        )

    hdul.writeto(output_path, overwrite=True)


# ---------------------------------------------------------------------------
# write_calibration_fits  (convenience wrapper)
# ---------------------------------------------------------------------------


def write_calibration_fits(
    calibration_products: CalibrationProductSet,
    output_dir: str,
) -> tuple[str, str]:
    """Write both wavecal and spatcal provisional FITS calibration files.

    Calls :func:`write_wavecal_fits` and :func:`write_spatcal_fits` and
    writes the output files into *output_dir*.  The file names are derived
    from the mode stored in *calibration_products*:

    - ``<mode>_wavecal.fits``
    - ``<mode>_spatcal.fits``

    Parameters
    ----------
    calibration_products : :class:`~pyspextool.instruments.ishell.calibration_products.CalibrationProductSet`
        Stage-8 output to serialise.  Must contain at least one order.
    output_dir : str
        Directory in which the output files are written.  Must already
        exist.

    Returns
    -------
    tuple of (str, str)
        ``(wavecal_path, spatcal_path)`` – the full paths of the two
        files that were written.

    Raises
    ------
    ValueError
        If *calibration_products* contains no orders.
    OSError
        If *output_dir* does not exist or is not writable.
    """
    if calibration_products.n_orders == 0:
        raise ValueError(
            "calibration_products is empty (n_orders == 0); "
            "cannot write calibration FITS."
        )

    if not os.path.isdir(output_dir):
        raise OSError(f"Output directory does not exist: {output_dir!r}")

    mode = calibration_products.mode
    wavecal_path = os.path.join(output_dir, f"{mode}_wavecal.fits")
    spatcal_path = os.path.join(output_dir, f"{mode}_spatcal.fits")

    write_wavecal_fits(calibration_products, wavecal_path)
    write_spatcal_fits(calibration_products, spatcal_path)

    return wavecal_path, spatcal_path
