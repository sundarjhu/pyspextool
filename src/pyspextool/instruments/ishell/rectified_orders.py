"""
Provisional rectified-order image generation for iSHELL 2DXD reduction.

This module implements the **seventh stage** of the iSHELL 2DXD reduction
scaffold: generating provisional rectified order images by interpolating the
detector image onto the rectified (wavelength × spatial) grid defined by
the Stage-6 rectification indices.

What this module does
---------------------
* Accepts a raw detector image (2-D NumPy array) and a
  :class:`~pyspextool.instruments.ishell.rectification_indices.RectificationIndexSet`
  (the Stage-6 output).

* For each order, uses the ``src_rows`` and ``src_cols`` index arrays to
  look up the corresponding detector coordinates in the rectified grid,
  and fills the rectified flux image by **bilinear interpolation** of the
  detector image at those sub-pixel positions.

* Returns a :class:`RectifiedOrderSet` collecting :class:`RectifiedOrder`
  objects, one per echelle order.

What this module does NOT do (by design)
-----------------------------------------
* **No tilt or curvature correction** – those steps belong to a later stage.

* **No physical spatial calibration** – spatial axis remains in fractional
  slit-position units ``[0, 1]``; conversion to arcseconds is not yet
  available at this scaffold stage.

* **No final wavecal/spatcal FITS calibration files** – these are produced
  in a later stage.

* **No flux-conserving resampling** – simple bilinear interpolation is used;
  flux conservation (drizzle-style resampling) is deferred to a later stage.

* **No optimal extraction** – that belongs to later stages.

* **No science-quality rectification** – the result is a provisional
  scaffold for development and pipeline validation.

Interpolation method
--------------------
Bilinear interpolation is performed using
:class:`scipy.interpolate.RegularGridInterpolator` with
``method="linear"``.  The detector image is sampled at the fractional
``(row, col)`` coordinates provided by the rectification indices.

Out-of-bounds coordinates are clamped to the detector boundary by using
``bounds_error=False`` and ``fill_value=np.nan``, so any rectified pixel
whose source coordinates fall outside the detector image is set to NaN.

Coordinate conventions
-----------------------
**Detector image**
    A 2-D array with shape ``(n_rows, n_cols)``.  ``image[row, col]`` is
    the pixel at detector row *row* and column *col*.

**Rectification indices**
    ``src_rows[j, i]`` is the fractional detector row for spatial output
    pixel *j* at spectral output pixel *i*.  ``src_cols[i]`` is the
    fractional detector column for spectral output pixel *i*.

**Rectified output**
    ``flux[j, i]`` is the interpolated flux at spatial pixel *j*, spectral
    pixel *i*.  The axes are:

    - axis 0 (rows): spatial, indexed by ``output_spatial_frac``.
    - axis 1 (columns): spectral, indexed by ``output_wavelengths_um``.

Public API
----------
- :func:`build_rectified_orders` – main entry point.
- :class:`RectifiedOrder` – per-order result container.
- :class:`RectifiedOrderSet` – full result container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from .rectification_indices import RectificationIndexSet

__all__ = [
    "RectifiedOrder",
    "RectifiedOrderSet",
    "build_rectified_orders",
]


# ---------------------------------------------------------------------------
# Per-order rectified image
# ---------------------------------------------------------------------------


@dataclass
class RectifiedOrder:
    """Provisional rectified image for one echelle order.

    Stores the rectified flux image and the coordinate axes produced by
    interpolating a detector frame onto the rectified
    (wavelength × spatial) grid.

    Parameters
    ----------
    order : int
        Echelle order number (or index, matching the upstream
        :class:`~pyspextool.instruments.ishell.rectification_indices.RectificationIndexOrder`).
    order_index : int
        Zero-based position of this order in the parent
        :class:`RectifiedOrderSet`.
    wavelength_um : ndarray, shape (n_spectral,)
        Wavelength axis of the rectified grid, in µm.  Uniformly spaced
        from the minimum to the maximum wavelength predicted for this order.
    spatial_frac : ndarray, shape (n_spatial,)
        Spatial axis of the rectified grid as a fractional slit position
        in ``[0.0, 1.0]``.  ``0.0`` is the bottom order edge; ``1.0`` is
        the top order edge.
    flux : ndarray, shape (n_spatial, n_spectral)
        Rectified flux image.  ``flux[j, i]`` is the interpolated
        detector flux at spatial pixel *j* and spectral pixel *i*.
        Pixels whose source coordinates fall outside the detector image
        are set to ``NaN``.
    source_image_shape : tuple of int
        Shape ``(n_rows, n_cols)`` of the detector image that was
        interpolated to produce this rectified order.

    Notes
    -----
    No tilt, curvature, or spatial calibration is applied.  The spatial
    axis is fractional slit position, not arcseconds.

    This is a provisional scaffold; see the module docstring and
    ``docs/ishell_rectified_orders.md`` for limitations.
    """

    order: int
    order_index: int
    wavelength_um: npt.NDArray  # shape (n_spectral,)
    spatial_frac: npt.NDArray  # shape (n_spatial,)
    flux: npt.NDArray  # shape (n_spatial, n_spectral)
    source_image_shape: tuple[int, int]

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels in the rectified output."""
        return len(self.wavelength_um)

    @property
    def n_spatial(self) -> int:
        """Number of spatial pixels in the rectified output."""
        return len(self.spatial_frac)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the rectified flux array: ``(n_spatial, n_spectral)``."""
        return (self.n_spatial, self.n_spectral)


# ---------------------------------------------------------------------------
# Full-set container
# ---------------------------------------------------------------------------


@dataclass
class RectifiedOrderSet:
    """Collection of provisional rectified order images.

    Stores one :class:`RectifiedOrder` per echelle order, plus metadata
    describing the source detector image and the iSHELL mode.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    rectified_orders : list of :class:`RectifiedOrder`
        Per-order rectified images.  One entry per echelle order, in the
        same order as the input
        :class:`~pyspextool.instruments.ishell.rectification_indices.RectificationIndexSet`.
    source_image_shape : tuple of int
        Shape ``(n_rows, n_cols)`` of the detector image used to generate
        this set.
    """

    mode: str
    rectified_orders: list[RectifiedOrder] = field(default_factory=list)
    source_image_shape: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    @property
    def orders(self) -> list[int]:
        """List of order numbers in the set (in storage order)."""
        return [ro.order for ro in self.rectified_orders]

    @property
    def n_orders(self) -> int:
        """Number of orders in the set."""
        return len(self.rectified_orders)

    def get_order(self, order: int) -> RectifiedOrder:
        """Return the :class:`RectifiedOrder` for a given order number.

        Parameters
        ----------
        order : int
            Order number (or index) to look up.

        Returns
        -------
        :class:`RectifiedOrder`

        Raises
        ------
        KeyError
            If *order* is not present in the set.
        """
        for ro in self.rectified_orders:
            if ro.order == order:
                return ro
        raise KeyError(
            f"Order {order} not found in RectifiedOrderSet for mode "
            f"{self.mode!r}.  Available orders: {self.orders}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_rectified_orders(
    image: npt.NDArray,
    rectification_indices: RectificationIndexSet,
) -> RectifiedOrderSet:
    """Generate provisional rectified order images by interpolating the
    detector image at the coordinates defined by the rectification indices.

    For each echelle order in *rectification_indices*, the detector image is
    interpolated at the sub-pixel ``(src_row, src_col)`` positions to fill a
    rectified flux array on the (wavelength × spatial) grid.

    Parameters
    ----------
    image : ndarray, shape (n_rows, n_cols)
        Raw detector image to be rectified.  Must be a 2-D array.
    rectification_indices : :class:`~pyspextool.instruments.ishell.rectification_indices.RectificationIndexSet`
        Rectification indices from Stage 6.  Must contain at least one
        order.

    Returns
    -------
    :class:`RectifiedOrderSet`
        One :class:`RectifiedOrder` per order in *rectification_indices*.

    Raises
    ------
    ValueError
        If *image* is not a 2-D array, if *image* dimensions do not match
        the coordinate ranges in any rectification order, or if
        *rectification_indices* is empty.

    Notes
    -----
    **Interpolation**

    Bilinear interpolation is performed using
    :class:`scipy.interpolate.RegularGridInterpolator` with
    ``method="linear"``.  Source coordinates that fall outside the
    detector image bounds are filled with ``NaN`` (``bounds_error=False``,
    ``fill_value=np.nan``).

    **Algorithm**

    For each order *k*:

    1. Retrieve ``src_rows`` (shape ``(n_spatial, n_spectral)``) and
       ``src_cols`` (shape ``(n_spectral,)``) from the
       :class:`~pyspextool.instruments.ishell.rectification_indices.RectificationIndexOrder`.

    2. Build a ``(n_spatial × n_spectral, 2)`` query array from
       ``src_rows`` and the broadcast of ``src_cols``.

    3. Evaluate the ``RegularGridInterpolator`` at the query points.

    4. Reshape the result to ``(n_spatial, n_spectral)`` to form the
       rectified flux image.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(
            f"image must be a 2-D array; got shape {image.shape}"
        )
    if rectification_indices.n_orders == 0:
        raise ValueError(
            "rectification_indices is empty (n_orders == 0); "
            "cannot build rectified orders."
        )

    n_rows, n_cols = image.shape

    # Build a RegularGridInterpolator over the full detector image.
    # Axis 0 is rows, axis 1 is columns.
    row_axis = np.arange(n_rows, dtype=float)
    col_axis = np.arange(n_cols, dtype=float)
    interp = RegularGridInterpolator(
        (row_axis, col_axis),
        image,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # ------------------------------------------------------------------
    # Build one RectifiedOrder per index order
    # ------------------------------------------------------------------
    rectified_orders: list[RectifiedOrder] = []

    for idx_order in rectification_indices.index_orders:
        n_spatial = idx_order.n_spatial
        n_spectral = idx_order.n_spectral

        # src_rows has shape (n_spatial, n_spectral).
        # src_cols has shape (n_spectral,); broadcast to (n_spatial, n_spectral).
        src_rows = idx_order.src_rows  # (n_spatial, n_spectral)
        src_cols = np.broadcast_to(
            idx_order.src_cols[np.newaxis, :], (n_spatial, n_spectral)
        )

        # Stack into (n_spatial * n_spectral, 2) query array.
        query_pts = np.column_stack(
            [src_rows.ravel(), src_cols.ravel()]
        )

        # Evaluate interpolator and reshape.
        flux_flat = interp(query_pts)
        flux = flux_flat.reshape(n_spatial, n_spectral)

        rectified_orders.append(
            RectifiedOrder(
                order=idx_order.order,
                order_index=len(rectified_orders),
                wavelength_um=idx_order.output_wavelengths_um.copy(),
                spatial_frac=idx_order.output_spatial_frac.copy(),
                flux=flux,
                source_image_shape=(n_rows, n_cols),
            )
        )

    return RectifiedOrderSet(
        mode=rectification_indices.mode,
        rectified_orders=rectified_orders,
        source_image_shape=(n_rows, n_cols),
    )
