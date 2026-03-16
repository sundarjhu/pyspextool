"""
Provisional spectral-extraction scaffold for iSHELL 2DXD reduction.

This module implements the **ninth stage** of the iSHELL 2DXD reduction
scaffold: collapsing each rectified order image along the spatial axis to
produce a provisional 1-D spectrum.

What this module does
---------------------
* Accepts a :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
  (the Stage-7 output).

* For each order, collapses the rectified flux image (shape
  ``(n_spatial, n_spectral)``) along the spatial axis (axis 0) using
  either a sum or a mean, handling NaN values via :func:`numpy.nansum` /
  :func:`numpy.nanmean`.

* Returns an :class:`ExtractedSpectrumSet` collecting one
  :class:`ExtractedOrderSpectrum` per echelle order.

What this module does NOT do (by design)
-----------------------------------------
* **No optimal extraction** – profile-weighted extraction belongs to a
  later stage.

* **No profile fitting** – the spatial profile is not modelled here.

* **No science-quality uncertainty propagation** – the ``variance``
  field is a placeholder (``None``) in this scaffold; Poisson/read-noise
  error propagation is deferred.

* **No background subtraction** – background removal is not yet
  performed; the entire spatial extent of the order is collapsed.

* **No telluric correction** – that belongs to a later stage.

* **No stitching or merging across orders** – each order is extracted
  independently.

Extraction methods
------------------
``"sum"``
    The spatial pixels are summed via :func:`numpy.nansum`.  NaN-valued
    spatial pixels at each spectral column are excluded from the sum.
    A spectral pixel where *all* spatial rows are NaN is set to NaN.

``"mean"``
    The spatial pixels are averaged via :func:`numpy.nanmean`.  NaN-valued
    spatial pixels are excluded from the mean.  A spectral pixel where
    *all* spatial rows are NaN is set to NaN (NumPy raises a
    ``RuntimeWarning`` in that case; the result is NaN).

NaN handling
------------
Before collapsing, :func:`_count_valid_spatial_rows` counts the number of
non-NaN spatial pixels at each spectral column.  Spectral columns where
*all* spatial rows are NaN are recorded in ``n_spatial_used`` as zero and
the collapsed value is NaN.  This is a scaffold simplification; a later
stage should apply a spatial mask based on a sky/background aperture.

Public API
----------
- :func:`extract_rectified_orders` – main entry point.
- :class:`ExtractedOrderSpectrum` – per-order 1-D spectrum container.
- :class:`ExtractedSpectrumSet` – full result container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from .rectified_orders import RectifiedOrderSet

__all__ = [
    "ExtractedOrderSpectrum",
    "ExtractedSpectrumSet",
    "extract_rectified_orders",
]

# Supported extraction methods.
_VALID_METHODS = {"sum", "mean"}


# ---------------------------------------------------------------------------
# Per-order 1-D spectrum
# ---------------------------------------------------------------------------


@dataclass
class ExtractedOrderSpectrum:
    """Provisional 1-D extracted spectrum for one echelle order.

    Stores the collapsed flux and wavelength axis produced by summing or
    averaging the rectified flux image along the spatial axis.

    Parameters
    ----------
    order : int
        Echelle order number (matches the order stored in the parent
        :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrder`).
    wavelength_um : ndarray, shape (n_spectral,)
        Wavelength axis in µm, propagated unchanged from the rectified
        order's ``wavelength_um`` field.
    flux : ndarray, shape (n_spectral,)
        Collapsed 1-D flux array.  Each element is the sum or mean of the
        valid (non-NaN) spatial pixels at that spectral column.  Spectral
        columns where all spatial pixels are NaN are set to NaN.
    method : str
        Extraction method used; one of ``"sum"`` or ``"mean"``.
    n_spatial_used : int
        Number of spatial rows in the rectified image that contributed to
        the extraction (i.e. rows that were not entirely NaN for *any*
        spectral column).  This is a rough diagnostic; see notes below.
    variance : ndarray or None, shape (n_spectral,)
        Placeholder for a future variance (uncertainty-squared) array.
        Always ``None`` in this scaffold stage; proper error propagation
        is deferred to a later stage.

    Notes
    -----
    ``n_spatial_used`` is computed as the number of spatial rows that are
    not entirely NaN across all spectral columns.  It is a single integer
    per order, not a per-spectral-pixel count.  A later stage should track
    the per-column number of valid rows.

    The ``variance`` field is intentionally left as ``None``.  It is
    included here so that downstream code can check for its presence
    without requiring a schema change when uncertainty propagation is
    added.
    """

    order: int
    wavelength_um: npt.NDArray  # shape (n_spectral,)
    flux: npt.NDArray  # shape (n_spectral,)
    method: str
    n_spatial_used: int
    variance: Optional[npt.NDArray] = None  # placeholder

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels in the extracted spectrum."""
        return len(self.wavelength_um)


# ---------------------------------------------------------------------------
# Full-set container
# ---------------------------------------------------------------------------


@dataclass
class ExtractedSpectrumSet:
    """Collection of provisional 1-D extracted spectra for all orders.

    Stores one :class:`ExtractedOrderSpectrum` per echelle order, together
    with the iSHELL mode identifier.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    spectra : list of :class:`ExtractedOrderSpectrum`
        Per-order extracted spectra, in the same order as the input
        :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`.
    """

    mode: str
    spectra: list[ExtractedOrderSpectrum] = field(default_factory=list)

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

    def get_order(self, order_number: int) -> ExtractedOrderSpectrum:
        """Return the :class:`ExtractedOrderSpectrum` for a given order number.

        Parameters
        ----------
        order_number : int
            Echelle order number to look up.

        Returns
        -------
        :class:`ExtractedOrderSpectrum`

        Raises
        ------
        KeyError
            If *order_number* is not present in the set.
        """
        for sp in self.spectra:
            if sp.order == order_number:
                return sp
        raise KeyError(
            f"Order {order_number} not found in ExtractedSpectrumSet for mode "
            f"{self.mode!r}.  Available orders: {self.orders}"
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _count_valid_spatial_rows(flux: npt.NDArray) -> int:
    """Return the number of spatial rows that are not entirely NaN.

    A spatial row is considered valid if at least one spectral pixel in
    that row is finite (non-NaN).

    Parameters
    ----------
    flux : ndarray, shape (n_spatial, n_spectral)
        Rectified flux image.

    Returns
    -------
    int
        Number of rows in *flux* that contain at least one non-NaN value.
    """
    # any_finite[j] is True if row j has at least one finite value.
    any_finite = np.any(np.isfinite(flux), axis=1)
    return int(np.sum(any_finite))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_rectified_orders(
    rectified_orders: RectifiedOrderSet,
    *,
    method: str = "sum",
) -> ExtractedSpectrumSet:
    """Collapse each rectified order image along the spatial axis to produce
    provisional 1-D extracted spectra.

    For each order in *rectified_orders*, the rectified flux array (shape
    ``(n_spatial, n_spectral)``) is collapsed along axis 0 (the spatial
    axis) using *method*.  NaN-valued spatial pixels are excluded from the
    collapse via :func:`numpy.nansum` / :func:`numpy.nanmean`.

    Parameters
    ----------
    rectified_orders : :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
        Stage-7 output.  Must contain at least one order.  Each order's
        ``flux`` field must be a 2-D array with shape
        ``(n_spatial, n_spectral)``.
    method : str, optional
        Collapse method.  Must be one of:

        ``"sum"`` (default)
            Spatial pixels are summed with :func:`numpy.nansum`.
        ``"mean"``
            Spatial pixels are averaged with :func:`numpy.nanmean`.

    Returns
    -------
    :class:`ExtractedSpectrumSet`
        One :class:`ExtractedOrderSpectrum` per order in *rectified_orders*.
        Each spectrum has:

        - ``wavelength_um`` copied from the rectified order.
        - ``flux`` of shape ``(n_spectral,)``.
        - ``method`` set to *method*.
        - ``n_spatial_used`` counting valid (non-entirely-NaN) spatial rows.
        - ``variance`` set to ``None`` (placeholder).

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
    **What is being collapsed**

    The entire spatial extent of the rectified order is collapsed.  No
    sky aperture, background aperture, or object aperture is applied.
    All non-NaN spatial rows contribute equally (for ``"mean"``) or
    additively (for ``"sum"``).

    **Assumptions**

    - The input rectified images have already been background-subtracted
      or are flat-field frames.  No background is subtracted here.
    - NaN values in the flux image arise from out-of-bounds source
      coordinates in the rectification step (Stage 7); they are excluded
      from the collapse automatically.

    **Limitations**

    - No optimal extraction.
    - No profile-weighted extraction.
    - No per-pixel variance propagation.
    - No aperture masking.
    - No sky subtraction.

    These belong to later stages of the reduction pipeline.
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
            f"method must be one of {sorted(_VALID_METHODS)!r}; got {method!r}."
        )

    # ------------------------------------------------------------------
    # Extract one spectrum per order
    # ------------------------------------------------------------------
    spectra: list[ExtractedOrderSpectrum] = []

    for ro in rectified_orders.rectified_orders:
        flux_2d = np.asarray(ro.flux, dtype=float)

        if flux_2d.ndim != 2:
            raise ValueError(
                f"Order {ro.order}: flux must be a 2-D array; "
                f"got shape {flux_2d.shape}."
            )

        n_spatial_used = _count_valid_spatial_rows(flux_2d)

        if method == "sum":
            # nansum returns 0 where all inputs are NaN; we want NaN there.
            collapsed = np.nansum(flux_2d, axis=0)
            # Overwrite columns that are all-NaN with NaN.
            all_nan_cols = np.all(~np.isfinite(flux_2d), axis=0)
            collapsed[all_nan_cols] = np.nan
        else:
            # method == "mean"
            # nanmean already returns NaN where all inputs are NaN.
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                collapsed = np.nanmean(flux_2d, axis=0)

        spectra.append(
            ExtractedOrderSpectrum(
                order=ro.order,
                wavelength_um=ro.wavelength_um.copy(),
                flux=collapsed,
                method=method,
                n_spatial_used=n_spatial_used,
                variance=None,
            )
        )

    return ExtractedSpectrumSet(
        mode=rectified_orders.mode,
        spectra=spectra,
    )
