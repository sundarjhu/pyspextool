"""
Aperture-aware spectral-extraction scaffold for iSHELL 2DXD reduction.

This module implements the **eleventh stage** of the iSHELL 2DXD reduction
scaffold: aperture-masked extraction with optional per-column background
subtraction.

.. note::
    This is a **scaffold implementation**.  It is intentionally simple and
    does **not** represent final science-quality iSHELL or Spextool
    extraction.  See ``docs/ishell_aperture_extraction.md`` for a full list
    of limitations.

Pipeline stage summary
----------------------
.. list-table::
   :header-rows: 1

   * - Stage
     - Module
     - Purpose
   * - 1
     - ``tracing.py``
     - Flat-order tracing
   * - 2
     - ``arc_tracing.py``
     - Arc-line tracing
   * - 3
     - ``wavecal_2d.py``
     - Per-order provisional wavelength mapping
   * - 4
     - ``wavecal_2d_surface.py``
     - Provisional global wavelength surface
   * - 5
     - ``wavecal_2d_refine.py``
     - Coefficient-surface refinement
   * - 6
     - ``rectification_indices.py``
     - Rectification-index generation
   * - 7
     - ``rectified_orders.py``
     - Rectified order images
   * - 8
     - ``calibration_products.py``
     - Provisional calibration product containers
   * - 9
     - ``calibration_fits.py``
     - FITS calibration writer
   * - 10
     - ``extracted_spectra.py``
     - Whole-slit provisional extraction
   * - **11**
     - **``aperture_extraction.py``**
     - **Aperture-aware extraction (this module)**

What this module does
---------------------
* Accepts a
  :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
  (the Stage-7 output) and an :class:`ApertureDefinition`.

* For each order, selects spatial pixels inside the aperture defined by
  *center_frac* ± *radius_frac* in fractional slit coordinates ``[0, 1]``.

* Optionally estimates a per-column background as the median of spatial
  pixels in a background annulus (``background_inner`` < |distance| ≤
  ``background_outer``), and subtracts it from the aperture pixels before
  collapsing.

* Collapses the aperture pixels along the spatial axis using either a sum
  or a mean.

* Returns an :class:`ExtractedApertureSpectrumSet` collecting one
  :class:`ExtractedApertureSpectrum` per echelle order.

What this module does NOT do (by design)
-----------------------------------------
* **No optimal extraction** – profile-weighted extraction belongs to a
  later stage.

* **No profile fitting** – the spatial profile is not modelled here.

* **No science-quality uncertainty propagation** – no variance arrays
  are computed; error propagation is deferred.

* **No aperture finding / centroiding** – the aperture center and radius
  must be supplied explicitly.

* **No sky modelling beyond simple median subtraction** – background
  is estimated as the median of background-annulus pixels per column.

* **No telluric correction** – that belongs to a later stage.

* **No order merging** – each order is extracted independently.

Extraction methods
------------------
``"sum"``
    Aperture pixels are summed via :func:`numpy.nansum`.  Spectral columns
    where all aperture pixels are NaN are set to NaN.

``"mean"``
    Aperture pixels are averaged via :func:`numpy.nanmean`.  Spectral
    columns where all aperture pixels are NaN are set to NaN.

NaN handling
------------
:func:`numpy.nanmedian`, :func:`numpy.nansum`, and :func:`numpy.nanmean`
are used throughout so that NaN values (e.g. out-of-bounds rectification
pixels) are excluded automatically.  Spectral columns with no valid
aperture pixels return NaN in the output flux.

Public API
----------
- :func:`extract_with_aperture` – main entry point.
- :class:`ApertureDefinition` – aperture and background region definition.
- :class:`ExtractedApertureSpectrum` – per-order 1-D spectrum container.
- :class:`ExtractedApertureSpectrumSet` – full result container.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from .rectified_orders import RectifiedOrderSet

__all__ = [
    "ApertureDefinition",
    "ExtractedApertureSpectrum",
    "ExtractedApertureSpectrumSet",
    "extract_with_aperture",
]

# Supported extraction methods.
_VALID_METHODS = {"sum", "mean"}


# ---------------------------------------------------------------------------
# Aperture definition
# ---------------------------------------------------------------------------


@dataclass
class ApertureDefinition:
    """Definition of an object aperture (and optional background region).

    All spatial coordinates are expressed in **fractional slit coordinates**
    in ``[0, 1]``, consistent with the ``spatial_frac`` axis stored in each
    :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrder`.

    Parameters
    ----------
    center_frac : float
        Fractional slit position of the aperture center.  Must be in
        ``[0, 1]``.
    radius_frac : float
        Half-width of the object aperture in fractional slit units.  Must
        be > 0.  Spatial pixels with
        ``|spatial_frac - center_frac| <= radius_frac`` are included in
        the aperture.
    background_inner : float or None, optional
        Inner edge of the background annulus in fractional slit distance
        from *center_frac*.  If ``None`` (default), background subtraction
        is disabled.  Must satisfy
        ``background_inner > radius_frac`` when provided.
    background_outer : float or None, optional
        Outer edge of the background annulus in fractional slit distance
        from *center_frac*.  Must satisfy
        ``background_outer > background_inner`` when provided.  Must be
        ``None`` if *background_inner* is ``None``.

    Notes
    -----
    The aperture region is symmetric about *center_frac*.  The background
    annulus is also symmetric; pixels on both sides of the object (above
    and below in slit direction) contribute to the background estimate
    provided they satisfy the distance criterion.

    This is a scaffold simplification.  Real Spextool supports independent
    background apertures on either side of the object and uses more
    sophisticated sky-subtraction algorithms.
    """

    center_frac: float
    radius_frac: float
    background_inner: Optional[float] = None
    background_outer: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate aperture parameters."""
        if not (0.0 <= self.center_frac <= 1.0):
            raise ValueError(
                f"center_frac must be in [0, 1]; got {self.center_frac!r}."
            )
        if self.radius_frac <= 0.0:
            raise ValueError(
                f"radius_frac must be > 0; got {self.radius_frac!r}."
            )
        # Background region consistency checks.
        if (self.background_inner is None) != (self.background_outer is None):
            raise ValueError(
                "background_inner and background_outer must both be provided "
                "or both be None."
            )
        if self.background_inner is not None:
            if self.background_inner <= self.radius_frac:
                raise ValueError(
                    f"background_inner ({self.background_inner!r}) must be "
                    f"> radius_frac ({self.radius_frac!r})."
                )
            if self.background_outer <= self.background_inner:
                raise ValueError(
                    f"background_outer ({self.background_outer!r}) must be "
                    f"> background_inner ({self.background_inner!r})."
                )

    @property
    def has_background(self) -> bool:
        """True if a background region is defined."""
        return self.background_inner is not None


# ---------------------------------------------------------------------------
# Per-order 1-D spectrum
# ---------------------------------------------------------------------------


@dataclass
class ExtractedApertureSpectrum:
    """Provisional aperture-extracted 1-D spectrum for one echelle order.

    Parameters
    ----------
    order : int
        Echelle order number.
    wavelength_um : ndarray, shape (n_spectral,)
        Wavelength axis in µm.
    flux : ndarray, shape (n_spectral,)
        Aperture-extracted 1-D flux.  Background-subtracted if
        ``subtract_background=True`` was passed to
        :func:`extract_with_aperture`.  Spectral columns with no valid
        aperture pixels are NaN.
    background_flux : ndarray or None, shape (n_spectral,)
        Per-column background estimate (median of background-annulus
        pixels), or ``None`` if background subtraction was not performed.
        This is the value that was subtracted from the aperture pixels
        before collapsing (prior to applying the *method*).
    aperture : :class:`ApertureDefinition`
        The aperture definition used for this extraction.
    method : str
        Extraction method used; one of ``"sum"`` or ``"mean"``.
    n_pixels_used : int
        Number of spatial pixels inside the aperture that contributed to
        the extraction (i.e. aperture pixels that were not entirely NaN
        across all spectral columns).

    Notes
    -----
    ``background_flux`` contains the per-column median *before* it is
    subtracted from (and propagated through) the aperture extraction.
    If ``subtract_background=False``, ``background_flux`` is ``None``.

    No variance or uncertainty is computed; that belongs to a later stage.
    """

    order: int
    wavelength_um: npt.NDArray  # shape (n_spectral,)
    flux: npt.NDArray  # shape (n_spectral,)
    background_flux: Optional[npt.NDArray]  # shape (n_spectral,) or None
    aperture: ApertureDefinition
    method: str
    n_pixels_used: int

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels in the extracted spectrum."""
        return len(self.wavelength_um)


# ---------------------------------------------------------------------------
# Full-set container
# ---------------------------------------------------------------------------


@dataclass
class ExtractedApertureSpectrumSet:
    """Collection of aperture-extracted 1-D spectra for all orders.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    spectra : list of :class:`ExtractedApertureSpectrum`
        Per-order extracted spectra, in the same order as the input
        :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`.
    """

    mode: str
    spectra: list[ExtractedApertureSpectrum] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    @property
    def orders(self) -> list[int]:
        """List of order numbers in the set (in storage order)."""
        return [sp.order for sp in self.spectra]

    @property
    def n_orders(self) -> int:
        """Number of orders in the set."""
        return len(self.spectra)

    def get_order(self, order_number: int) -> ExtractedApertureSpectrum:
        """Return the :class:`ExtractedApertureSpectrum` for a given order.

        Parameters
        ----------
        order_number : int
            Echelle order number to look up.

        Returns
        -------
        :class:`ExtractedApertureSpectrum`

        Raises
        ------
        KeyError
            If *order_number* is not present in the set.
        """
        for sp in self.spectra:
            if sp.order == order_number:
                return sp
        raise KeyError(
            f"Order {order_number} not found in ExtractedApertureSpectrumSet "
            f"for mode {self.mode!r}.  Available orders: {self.orders}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_aperture_pixels(flux_ap: npt.NDArray) -> int:
    """Return the number of spatial rows in the aperture that are not fully NaN.

    Parameters
    ----------
    flux_ap : ndarray, shape (n_ap_spatial, n_spectral)
        Flux sub-image restricted to aperture rows.

    Returns
    -------
    int
        Number of rows that have at least one finite value.
    """
    any_finite = np.any(np.isfinite(flux_ap), axis=1)
    return int(np.sum(any_finite))


def _extract_1d(flux_ap: npt.NDArray, method: str) -> npt.NDArray:
    """Collapse a 2-D aperture flux image along the spatial axis.

    Parameters
    ----------
    flux_ap : ndarray, shape (n_ap_spatial, n_spectral)
        Flux sub-image restricted to aperture rows (background already
        subtracted if requested).
    method : str
        ``"sum"`` or ``"mean"``.

    Returns
    -------
    ndarray, shape (n_spectral,)
        Collapsed 1-D spectrum.  Spectral columns where all aperture pixels
        are NaN are set to NaN.
    """
    all_nan_cols = np.all(~np.isfinite(flux_ap), axis=0)

    if method == "sum":
        collapsed = np.nansum(flux_ap, axis=0)
        collapsed[all_nan_cols] = np.nan
    else:
        # method == "mean"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            collapsed = np.nanmean(flux_ap, axis=0)
        # nanmean already returns NaN for all-NaN columns; be explicit.
        collapsed[all_nan_cols] = np.nan

    return collapsed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_with_aperture(
    rectified_orders: RectifiedOrderSet,
    aperture: ApertureDefinition,
    *,
    method: str = "sum",
    subtract_background: bool = True,
) -> ExtractedApertureSpectrumSet:
    """Extract 1-D spectra from rectified orders using a spatial aperture.

    For each order in *rectified_orders*, selects spatial pixels inside
    the aperture defined by *aperture*, optionally subtracts a
    per-column background estimated from a background annulus, then
    collapses the aperture pixels along the spatial axis.

    Parameters
    ----------
    rectified_orders : :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
        Stage-7 output.  Must contain at least one order.
    aperture : :class:`ApertureDefinition`
        Aperture and (optionally) background region definition.
    method : str, optional
        Collapse method.  One of ``"sum"`` (default) or ``"mean"``.
    subtract_background : bool, optional
        If ``True`` (default) **and** *aperture* has a background region
        defined (``aperture.has_background``), estimate a per-column
        background as the median of background-annulus pixels and subtract
        it from each aperture pixel before collapsing.  If ``False``, or
        if *aperture* has no background region, no subtraction is
        performed.

    Returns
    -------
    :class:`ExtractedApertureSpectrumSet`
        One :class:`ExtractedApertureSpectrum` per order in
        *rectified_orders*.

    Raises
    ------
    ValueError
        If *rectified_orders* is empty (``n_orders == 0``).
    ValueError
        If *method* is not one of ``"sum"`` or ``"mean"``.
    ValueError
        If any order's ``flux`` field is not a 2-D array.

    Notes
    -----
    **Algorithm**

    For each order:

    1. Compute ``dist = |spatial_frac - center_frac|`` for every spatial
       pixel.
    2. Select aperture pixels: ``dist <= radius_frac``.
    3. If background subtraction is enabled and the aperture defines a
       background region, select background pixels:
       ``background_inner < dist <= background_outer``.
       Compute a per-column background estimate as
       :func:`numpy.nanmedian` of those pixels (shape ``(n_spectral,)``).
       Subtract the background from every aperture pixel.
    4. Collapse the (background-subtracted) aperture pixel rows along the
       spatial axis using *method*.
    5. Set spectral columns with no valid aperture pixels to NaN.

    **Scaffold simplifications**

    - Background is estimated as a simple median of annulus pixels; no
      polynomial fitting or interpolation across the annulus is performed.
    - No optimal extraction, no PSF fitting, no per-pixel variance.
    - The aperture must be supplied explicitly; no centroiding or
      automatic aperture finding is performed.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if rectified_orders.n_orders == 0:
        raise ValueError(
            "rectified_orders is empty (n_orders == 0); "
            "cannot extract spectra."
        )
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_METHODS)!r}; "
            f"got {method!r}."
        )

    # Whether background subtraction will actually be applied.
    do_background = subtract_background and aperture.has_background

    # ------------------------------------------------------------------
    # Extract one spectrum per order
    # ------------------------------------------------------------------
    spectra: list[ExtractedApertureSpectrum] = []

    for ro in rectified_orders.rectified_orders:
        flux_2d = np.asarray(ro.flux, dtype=float)

        if flux_2d.ndim != 2:
            raise ValueError(
                f"Order {ro.order}: flux must be a 2-D array; "
                f"got shape {flux_2d.shape}."
            )

        spatial_frac = np.asarray(ro.spatial_frac, dtype=float)
        # shape (n_spatial,)
        dist = np.abs(spatial_frac - aperture.center_frac)

        # -- Aperture mask: shape (n_spatial,) boolean --
        ap_mask = dist <= aperture.radius_frac

        # Sub-image for aperture pixels: shape (n_ap, n_spectral)
        flux_ap = flux_2d[ap_mask, :]

        if flux_ap.shape[0] == 0:
            # No spatial pixels inside the aperture for this order.
            n_spectral = flux_2d.shape[1]
            flux_1d = np.full(n_spectral, np.nan)
            bg_1d: Optional[npt.NDArray] = (
                np.full(n_spectral, np.nan) if do_background else None
            )
            spectra.append(
                ExtractedApertureSpectrum(
                    order=ro.order,
                    wavelength_um=ro.wavelength_um.copy(),
                    flux=flux_1d,
                    background_flux=bg_1d,
                    aperture=aperture,
                    method=method,
                    n_pixels_used=0,
                )
            )
            continue

        # -- Background estimation --
        bg_1d = None
        if do_background:
            bg_mask = (dist > aperture.background_inner) & (
                dist <= aperture.background_outer
            )
            flux_bg = flux_2d[bg_mask, :]  # shape (n_bg, n_spectral)

            if flux_bg.shape[0] == 0:
                # No background pixels available; produce NaN background.
                bg_1d = np.full(flux_2d.shape[1], np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bg_1d = np.nanmedian(flux_bg, axis=0)  # (n_spectral,)

            # Subtract per-column background from each aperture row.
            flux_ap = flux_ap - bg_1d[np.newaxis, :]  # broadcast

        # -- Collapse --
        n_pixels_used = _count_aperture_pixels(flux_ap)
        flux_1d = _extract_1d(flux_ap, method)

        spectra.append(
            ExtractedApertureSpectrum(
                order=ro.order,
                wavelength_um=ro.wavelength_um.copy(),
                flux=flux_1d,
                background_flux=bg_1d,
                aperture=aperture,
                method=method,
                n_pixels_used=n_pixels_used,
            )
        )

    return ExtractedApertureSpectrumSet(
        mode=rectified_orders.mode,
        spectra=spectra,
    )
