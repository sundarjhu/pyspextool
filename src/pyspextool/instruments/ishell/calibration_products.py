"""
Provisional wavecal/spatcal calibration-product generation for iSHELL 2DXD.

This module implements the **eighth stage** of the iSHELL 2DXD reduction
scaffold: constructing provisional wavelength-calibration (wavecal) and
spatial-calibration (spatcal) products from the rectified order images
produced by Stage 7.

What this module does
---------------------
* Accepts a :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
  (the Stage-7 output).

* For each order, extracts the wavelength axis (µm) and the fractional
  spatial axis as provisional calibration data.

* Packages each axis into typed dataclass containers:
  :class:`WaveCalProduct` and :class:`SpatCalProduct`.

* Returns a :class:`CalibrationProductSet` that holds both sets and
  provides a common order-lookup interface.

What this module does NOT do (by design)
-----------------------------------------
* **No final FITS calibration file writing** – writing FITS-format
  calibration files is deferred to a later stage.

* **No physical spatial calibration in arcseconds** – the spatial axis
  remains in fractional slit-position units ``[0, 1]``; conversion to
  arcseconds requires a later stage.

* **No sigma-clipping or wavelength refinement** – the wavelength axis
  is taken directly from the rectified grid; further refinement belongs
  to a later stage.

* **No tilt or curvature correction** – inherited from the scaffold
  simplifications documented in
  ``docs/ishell_scaffold_constraints.md``.

* **No extraction** – this module produces calibration axes only; no
  spectral extraction is performed.

Relationship to prior stages
-----------------------------
Stage 7 (:mod:`~pyspextool.instruments.ishell.rectified_orders`) produces
a ``RectifiedOrderSet`` whose per-order ``wavelength_um`` and
``spatial_frac`` arrays define the output grid of the rectification.  This
stage simply lifts those arrays into named calibration-product containers
so that downstream code can consume them in a structured, mode-independent
format.

Public API
----------
- :func:`build_calibration_products` – main entry point.
- :class:`WaveCalProduct` – per-order provisional wavelength calibration.
- :class:`SpatCalProduct` – per-order provisional spatial calibration.
- :class:`CalibrationProductSet` – full result container.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .rectified_orders import RectifiedOrderSet

__all__ = [
    "WaveCalProduct",
    "SpatCalProduct",
    "CalibrationProductSet",
    "build_calibration_products",
]


# ---------------------------------------------------------------------------
# Per-order wavecal product
# ---------------------------------------------------------------------------


@dataclass
class WaveCalProduct:
    """Provisional wavelength calibration for one echelle order.

    Holds the wavelength axis extracted directly from the rectified order
    grid produced by Stage 7.

    Parameters
    ----------
    order : int
        Echelle order number (matches the order stored in the
        :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrder`).
    wavelength_um : ndarray, shape (n_spectral,)
        Wavelength axis of the rectified grid, in µm.  Uniformly spaced
        from the minimum to the maximum wavelength predicted for this order
        by the Stage-5 coefficient surface.
    spatial_frac : ndarray, shape (n_spatial,)
        Fractional spatial axis of the rectified grid in ``[0.0, 1.0]``.
        Included here for convenience so that callers can associate each
        spectral pixel with the spatial coverage of the order.
    rectified_flux : ndarray, shape (n_spatial, n_spectral)
        Reference to the rectified flux image from Stage 7.  This is the
        flux array on the ``(spatial_frac × wavelength_um)`` grid.

    Notes
    -----
    The wavelength axis is *provisional*: it is derived from a polynomial
    wavelength surface that has not been sigma-clipped or refined against
    a science-quality line list.  It is suitable for scaffold development
    but not for science-quality reductions.
    """

    order: int
    wavelength_um: npt.NDArray  # shape (n_spectral,)
    spatial_frac: npt.NDArray  # shape (n_spatial,)
    rectified_flux: npt.NDArray  # shape (n_spatial, n_spectral)

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels in the wavelength axis."""
        return len(self.wavelength_um)

    @property
    def n_spatial(self) -> int:
        """Number of spatial pixels in the spatial axis."""
        return len(self.spatial_frac)


# ---------------------------------------------------------------------------
# Per-order spatcal product
# ---------------------------------------------------------------------------


@dataclass
class SpatCalProduct:
    """Provisional spatial calibration for one echelle order.

    Holds the fractional spatial axis and a mapping from spatial coordinate
    to detector row extracted from the rectified order grid.

    Parameters
    ----------
    order : int
        Echelle order number.
    spatial_frac : ndarray, shape (n_spatial,)
        Fractional slit-position axis in ``[0.0, 1.0]``.  ``0.0`` is the
        bottom order edge; ``1.0`` is the top order edge.
    detector_rows : ndarray, shape (n_spatial,)
        Median detector row (fractional) corresponding to each spatial
        fraction in ``spatial_frac``.  Derived from the ``src_rows``
        mapping at the central spectral pixel of the rectified order so
        that a single representative row is associated with each spatial
        coordinate.

    Notes
    -----
    Physical spatial calibration in arcseconds is not yet available at
    this scaffold stage.  The ``spatial_frac`` axis is a scaffold
    simplification; see ``docs/ishell_scaffold_constraints.md``.
    """

    order: int
    spatial_frac: npt.NDArray  # shape (n_spatial,)
    detector_rows: npt.NDArray  # shape (n_spatial,)

    @property
    def n_spatial(self) -> int:
        """Number of spatial pixels."""
        return len(self.spatial_frac)


# ---------------------------------------------------------------------------
# Full-set container
# ---------------------------------------------------------------------------


@dataclass
class CalibrationProductSet:
    """Collection of provisional wavecal and spatcal products for all orders.

    Stores one :class:`WaveCalProduct` and one :class:`SpatCalProduct` per
    echelle order, together with the iSHELL mode identifier.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    wavecal_products : list of :class:`WaveCalProduct`
        Per-order provisional wavelength calibrations, in the same order
        as the input :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`.
    spatcal_products : list of :class:`SpatCalProduct`
        Per-order provisional spatial calibrations, in the same order as
        ``wavecal_products``.
    """

    mode: str
    wavecal_products: list[WaveCalProduct] = field(default_factory=list)
    spatcal_products: list[SpatCalProduct] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    @property
    def orders(self) -> list[int]:
        """List of order numbers in the set (in storage order)."""
        return [wcp.order for wcp in self.wavecal_products]

    @property
    def n_orders(self) -> int:
        """Number of orders in the set."""
        return len(self.wavecal_products)

    def get_order(self, order_number: int) -> tuple[WaveCalProduct, SpatCalProduct]:
        """Return the wavecal and spatcal products for a given order number.

        Parameters
        ----------
        order_number : int
            Echelle order number to look up.

        Returns
        -------
        tuple of (:class:`WaveCalProduct`, :class:`SpatCalProduct`)
            The wavecal and spatcal products for the requested order.

        Raises
        ------
        KeyError
            If *order_number* is not present in the set.
        """
        for wcp, scp in zip(self.wavecal_products, self.spatcal_products):
            if wcp.order == order_number:
                return wcp, scp
        raise KeyError(
            f"Order {order_number} not found in CalibrationProductSet for mode "
            f"{self.mode!r}.  Available orders: {self.orders}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_calibration_products(
    rectified_orders: RectifiedOrderSet,
) -> CalibrationProductSet:
    """Build provisional wavecal and spatcal calibration products from
    Stage-7 rectified orders.

    For each order in *rectified_orders*:

    1. The ``wavelength_um`` axis is used as the provisional wavecal axis.
    2. The ``spatial_frac`` axis is used as the provisional spatcal axis.
    3. The median detector row at the central spectral pixel is computed
       from the ``spatial_frac`` axis (linearly mapped to the row range
       stored in ``spatial_frac`` itself) as a representative
       row-position mapping.
    4. :class:`WaveCalProduct` and :class:`SpatCalProduct` objects are
       constructed from the extracted axes.

    Parameters
    ----------
    rectified_orders : :class:`~pyspextool.instruments.ishell.rectified_orders.RectifiedOrderSet`
        Stage-7 output.  Must contain at least one order.

    Returns
    -------
    :class:`CalibrationProductSet`
        One :class:`WaveCalProduct` and one :class:`SpatCalProduct` per
        order in *rectified_orders*.

    Raises
    ------
    ValueError
        If *rectified_orders* is empty (``n_orders == 0``).
    ValueError
        If any order has a wavelength axis and spatial axis that are
        inconsistent with the flux array shape.
    """
    if rectified_orders.n_orders == 0:
        raise ValueError(
            "rectified_orders is empty (n_orders == 0); "
            "cannot build calibration products."
        )

    wavecal_products: list[WaveCalProduct] = []
    spatcal_products: list[SpatCalProduct] = []

    for ro in rectified_orders.rectified_orders:
        # ------------------------------------------------------------------
        # Validate axis/flux consistency
        # ------------------------------------------------------------------
        expected_shape = (len(ro.spatial_frac), len(ro.wavelength_um))
        if ro.flux.shape != expected_shape:
            raise ValueError(
                f"Order {ro.order}: flux shape {ro.flux.shape} is inconsistent "
                f"with spatial_frac length {len(ro.spatial_frac)} and "
                f"wavelength_um length {len(ro.wavelength_um)}.  "
                f"Expected shape {expected_shape}."
            )

        # ------------------------------------------------------------------
        # Wavecal product
        # ------------------------------------------------------------------
        wavecal_products.append(
            WaveCalProduct(
                order=ro.order,
                wavelength_um=ro.wavelength_um.copy(),
                spatial_frac=ro.spatial_frac.copy(),
                rectified_flux=ro.flux,
            )
        )

        # ------------------------------------------------------------------
        # Spatcal product
        # ------------------------------------------------------------------
        # At this scaffold stage the physical row coordinates are not
        # available in the RectifiedOrder (they live in the
        # RectificationIndexOrder from Stage 6).  We provisionally set
        # detector_rows equal to spatial_frac as a placeholder; a later
        # stage will replace this with the physical src_row values.
        detector_rows = ro.spatial_frac.copy()

        spatcal_products.append(
            SpatCalProduct(
                order=ro.order,
                spatial_frac=ro.spatial_frac.copy(),
                detector_rows=detector_rows,
            )
        )

    return CalibrationProductSet(
        mode=rectified_orders.mode,
        wavecal_products=wavecal_products,
        spatcal_products=spatcal_products,
    )
