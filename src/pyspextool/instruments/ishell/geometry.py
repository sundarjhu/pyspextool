"""
Geometry data-model for iSHELL echelle order representation.

This module defines the minimum set of data structures needed to represent
iSHELL order geometry before full wavelength calibration.  The key objects
are:

* :class:`OrderGeometry` – geometric description of a single echelle order,
  including edge polynomials, optional tilt/curvature, and optional wavelength
  and spatial calibration polynomials.
* :class:`RectificationMap` – per-order mapping from raw detector coordinates
  to a rectilinear (wavelength × spatial) grid, ready for resampling.
* :class:`OrderGeometrySet` – collection of :class:`OrderGeometry` objects for
  all echelle orders in one iSHELL mode.

Design notes
------------
All edge and tilt polynomials follow the ``numpy.polynomial.polynomial``
convention: ``coeffs[k]`` is the coefficient of ``x**k`` (i.e. the constant
term is ``coeffs[0]``).  Evaluation uses
``np.polynomial.polynomial.polyval(x, coeffs)``.

This convention matches the existing pySpextool convention used in
``extract/flat.py`` (``read_flatcal_file``) and ``normalize_flat``.

Geometry objects live entirely within the ``instruments/ishell/`` package and
do **not** touch any generic ``extract/`` module, preserving SpeX/uSpeX
compatibility.

See ``docs/ishell_geometry_design_note.md`` for a full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "OrderGeometry",
    "RectificationMap",
    "OrderGeometrySet",
    "build_order_geometry_set",
]


# ---------------------------------------------------------------------------
# OrderGeometry
# ---------------------------------------------------------------------------


@dataclass
class OrderGeometry:
    """Geometric description of one iSHELL echelle order on the detector.

    Parameters
    ----------
    order : int
        Echelle order number.
    x_start : int
        First valid detector column (inclusive).
    x_end : int
        Last valid detector column (inclusive).
    bottom_edge_coeffs : ndarray, shape (n_terms,)
        Polynomial coefficients for the bottom order edge:
        ``row = polyval(col, coeffs)``.
    top_edge_coeffs : ndarray, shape (n_terms,)
        Polynomial coefficients for the top order edge:
        ``row = polyval(col, coeffs)``.
    tilt_coeffs : ndarray, shape (n_terms,) or None
        Polynomial coefficients for the spectral-line tilt slope as a
        function of detector column:
        ``tilt_slope = polyval(col, coeffs)``.
        ``None`` if tilt information is not yet available (i.e. wavecal not
        yet computed).  For SpeX-style instruments, lines are column-
        perpendicular so this field is not needed.
    curvature_coeffs : ndarray, shape (n_terms,) or None
        Polynomial coefficients for the spectral-line curvature as a
        function of column.  ``None`` if not yet computed.
    wave_coeffs : ndarray, shape (n_terms,) or None
        Polynomial coefficients for the wavelength along the centerline as a
        function of detector column:
        ``wavelength_um = polyval(col, coeffs)``.
        ``None`` until a wavelength solution is available.
    spatcal_coeffs : ndarray, shape (n_terms,) or None
        Polynomial coefficients for the spatial calibration:
        ``arcsec = polyval(row_offset, coeffs)``
        where ``row_offset = row - eval_centerline(col)``.
        Use :meth:`eval_centerline` to obtain the centerline row at a given
        column before computing the row offset.
        ``None`` until a spatial calibration is available.

    Notes
    -----
    The ``centerline_coeffs`` property computes the per-order centerline as
    the arithmetic mean of the bottom and top edge polynomials.  This is
    exact when both edges are polynomial functions of the same degree.

    The optional fields (``tilt_coeffs``, ``curvature_coeffs``,
    ``wave_coeffs``, ``spatcal_coeffs``) start as ``None`` after flat
    tracing and are filled in progressively as the wavecal step runs.
    """

    order: int
    x_start: int
    x_end: int
    bottom_edge_coeffs: np.ndarray
    top_edge_coeffs: np.ndarray
    tilt_coeffs: Optional[np.ndarray] = None
    curvature_coeffs: Optional[np.ndarray] = None
    wave_coeffs: Optional[np.ndarray] = None
    spatcal_coeffs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def x_range(self) -> tuple[int, int]:
        """Column range as ``(x_start, x_end)``."""
        return (self.x_start, self.x_end)

    @property
    def centerline_coeffs(self) -> np.ndarray:
        """Polynomial coefficients for the order centerline.

        Computed as the arithmetic mean of the bottom and top edge
        polynomials coefficient-by-coefficient.  The result has the same
        number of terms as the edge polynomials; evaluation gives the
        midpoint row at each column.

        Returns
        -------
        ndarray, shape (n_terms,)
        """
        return (self.bottom_edge_coeffs + self.top_edge_coeffs) / 2.0

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def eval_bottom_edge(self, cols: npt.ArrayLike) -> np.ndarray:
        """Evaluate the bottom edge polynomial at *cols*.

        Parameters
        ----------
        cols : array_like
            Detector column coordinates.

        Returns
        -------
        ndarray
            Row positions of the bottom edge at each column.
        """
        return np.polynomial.polynomial.polyval(
            np.asarray(cols, dtype=float), self.bottom_edge_coeffs
        )

    def eval_top_edge(self, cols: npt.ArrayLike) -> np.ndarray:
        """Evaluate the top edge polynomial at *cols*.

        Parameters
        ----------
        cols : array_like
            Detector column coordinates.

        Returns
        -------
        ndarray
            Row positions of the top edge at each column.
        """
        return np.polynomial.polynomial.polyval(
            np.asarray(cols, dtype=float), self.top_edge_coeffs
        )

    def eval_centerline(self, cols: npt.ArrayLike) -> np.ndarray:
        """Evaluate the order centerline at *cols*.

        Parameters
        ----------
        cols : array_like
            Detector column coordinates.

        Returns
        -------
        ndarray
            Row positions of the order center at each column.
        """
        return np.polynomial.polynomial.polyval(
            np.asarray(cols, dtype=float), self.centerline_coeffs
        )

    def eval_tilt(self, cols: npt.ArrayLike) -> np.ndarray:
        """Evaluate the tilt polynomial at *cols*.

        Parameters
        ----------
        cols : array_like
            Detector column coordinates.

        Returns
        -------
        ndarray
            Tilt slope (row offset per column offset) at each column.

        Raises
        ------
        RuntimeError
            If ``tilt_coeffs`` is ``None`` (tilt not yet computed).
        """
        if self.tilt_coeffs is None:
            raise RuntimeError(
                f"Tilt coefficients for order {self.order} are not yet "
                "available.  They are populated during the wavecal step."
            )
        return np.polynomial.polynomial.polyval(
            np.asarray(cols, dtype=float), self.tilt_coeffs
        )

    # ------------------------------------------------------------------
    # Availability predicates
    # ------------------------------------------------------------------

    def has_tilt(self) -> bool:
        """``True`` if tilt polynomial coefficients are available."""
        return self.tilt_coeffs is not None

    def has_curvature(self) -> bool:
        """``True`` if curvature polynomial coefficients are available."""
        return self.curvature_coeffs is not None

    def has_wavelength_solution(self) -> bool:
        """``True`` if wavelength polynomial coefficients are available."""
        return self.wave_coeffs is not None

    def has_spatcal(self) -> bool:
        """``True`` if spatial-calibration polynomial coefficients are available."""
        return self.spatcal_coeffs is not None

    def n_columns(self) -> int:
        """Number of detector columns spanned by this order."""
        return self.x_end - self.x_start + 1


# ---------------------------------------------------------------------------
# RectificationMap
# ---------------------------------------------------------------------------


@dataclass
class RectificationMap:
    """Per-order mapping from raw detector to rectified wavelength × spatial coordinates.

    Encodes how to resample a tilted iSHELL echelle order from the detector
    frame onto a rectilinear (wavelength × spatial) grid, ready for bilinear
    (or higher-order) interpolation.

    This object represents the *output* of the rectification computation; it
    is an input to the actual resampling step in ``_rectify_orders()``.

    Parameters
    ----------
    order : int
        Echelle order number.
    output_wavelengths_um : ndarray, shape (n_spectral,)
        Wavelength axis of the rectified output grid, in microns.
    output_spatial_arcsec : ndarray, shape (n_spatial,)
        Spatial axis of the rectified output grid, in arcseconds.
    src_rows : ndarray, shape (n_spatial, n_spectral)
        Fractional source detector row for each output-grid pixel.
        Used as the row argument to the interpolation function.
    src_cols : ndarray, shape (n_spatial, n_spectral)
        Fractional source detector column for each output-grid pixel.
        Used as the column argument to the interpolation function.

    Notes
    -----
    ``src_rows`` and ``src_cols`` contain *fractional* (sub-pixel) coordinates
    in the raw detector frame.  Callers should pass them to
    ``scipy.ndimage.map_coordinates`` or an equivalent function with the
    appropriate interpolation order.

    The arrays are computed by the wavecal step once tilt and wavelength
    polynomials are available on the parent :class:`OrderGeometry`.  They
    are **not** available from flat tracing alone.
    """

    order: int
    output_wavelengths_um: np.ndarray
    output_spatial_arcsec: np.ndarray
    src_rows: np.ndarray
    src_cols: np.ndarray

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the rectified output array: ``(n_spatial, n_spectral)``."""
        return (len(self.output_spatial_arcsec), len(self.output_wavelengths_um))

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels in the output grid."""
        return len(self.output_wavelengths_um)

    @property
    def n_spatial(self) -> int:
        """Number of spatial pixels in the output grid."""
        return len(self.output_spatial_arcsec)


# ---------------------------------------------------------------------------
# OrderGeometrySet
# ---------------------------------------------------------------------------


@dataclass
class OrderGeometrySet:
    """Collection of :class:`OrderGeometry` objects for all echelle orders in one mode.

    This is the primary geometry container passed through the iSHELL
    reduction pipeline.  It is constructed from flat-field calibration data
    immediately after order tracing and augmented with tilt/wavelength
    information after the wavecal step.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"K1"``).
    geometries : list of :class:`OrderGeometry`
        Per-order geometry objects, one per echelle order.  The list may
        be provided in any order; it is stored in the supplied order.

    Notes
    -----
    This object intentionally does **not** duplicate information already
    present in :class:`~pyspextool.instruments.ishell.calibrations.FlatInfo`
    or :class:`~pyspextool.instruments.ishell.calibrations.WaveCalInfo`.
    It is an intermediate runtime object that bridges flat tracing and
    wavelength calibration.
    """

    mode: str
    geometries: list[OrderGeometry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    @property
    def orders(self) -> list[int]:
        """Sorted list of echelle order numbers."""
        return sorted(g.order for g in self.geometries)

    @property
    def n_orders(self) -> int:
        """Number of orders in the set."""
        return len(self.geometries)

    def get_order(self, order: int) -> OrderGeometry:
        """Return the :class:`OrderGeometry` for a given order number.

        Parameters
        ----------
        order : int
            Echelle order number to look up.

        Returns
        -------
        :class:`OrderGeometry`

        Raises
        ------
        KeyError
            If *order* is not present in the set.
        """
        for g in self.geometries:
            if g.order == order:
                return g
        raise KeyError(
            f"Order {order} not found in OrderGeometrySet for mode {self.mode!r}.  "
            f"Available orders: {self.orders}"
        )

    # ------------------------------------------------------------------
    # Availability predicates
    # ------------------------------------------------------------------

    def _all_have(self, predicate) -> bool:
        """Return ``True`` if the set is non-empty and *predicate* holds for every order."""
        return bool(self.geometries) and all(predicate(g) for g in self.geometries)

    def has_tilt(self) -> bool:
        """``True`` if **all** orders have tilt polynomial coefficients."""
        return self._all_have(lambda g: g.has_tilt())

    def has_wavelength_solution(self) -> bool:
        """``True`` if **all** orders have wavelength polynomial coefficients."""
        return self._all_have(lambda g: g.has_wavelength_solution())

    def has_spatcal(self) -> bool:
        """``True`` if **all** orders have spatial-calibration coefficients."""
        return self._all_have(lambda g: g.has_spatcal())


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_order_geometry_set(
    mode: str,
    orders: npt.ArrayLike,
    x_ranges: npt.ArrayLike,
    edge_coeffs: npt.ArrayLike,
) -> OrderGeometrySet:
    """Construct an :class:`OrderGeometrySet` from flat-field calibration data.

    This factory converts the arrays returned by
    :func:`~pyspextool.extract.flat.read_flatcal_file` into an
    :class:`OrderGeometrySet`.  It is the intended bridge between the
    flat-tracing step and the rectification / wavecal steps.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"K1"``).
    orders : array_like of int, shape (n_orders,)
        Echelle order numbers, in the same order as the rows of
        *x_ranges* and *edge_coeffs*.
    x_ranges : array_like of int, shape (n_orders, 2)
        Column ranges ``[x_start, x_end]`` for each order, as returned by
        :func:`~pyspextool.extract.flat.read_flatcal_file` under key
        ``'xranges'``.
    edge_coeffs : array_like of float, shape (n_orders, 2, n_terms)
        Polynomial coefficients for the bottom (index 0) and top (index 1)
        edges of each order.  ``edge_coeffs[i, 0, :]`` is the bottom-edge
        polynomial for order *i* and ``edge_coeffs[i, 1, :]`` is the top-edge
        polynomial.  Follows the ``numpy.polynomial.polynomial`` convention
        where ``coeffs[k]`` is the coefficient of ``x**k``.

    Returns
    -------
    :class:`OrderGeometrySet`

    Raises
    ------
    ValueError
        If *orders*, *x_ranges*, and *edge_coeffs* have inconsistent leading
        dimensions.

    Examples
    --------
    Typical usage after calling ``read_flatcal_file``:

    >>> flatinfo = read_flatcal_file("K1_flatinfo.fits")
    >>> geom_set = build_order_geometry_set(
    ...     mode="K1",
    ...     orders=flatinfo["orders"],
    ...     x_ranges=flatinfo["xranges"],
    ...     edge_coeffs=flatinfo["edgecoeffs"],
    ... )
    """
    orders_arr = np.asarray(orders, dtype=int)
    x_ranges_arr = np.asarray(x_ranges, dtype=int)
    edge_coeffs_arr = np.asarray(edge_coeffs, dtype=float)

    n_orders = len(orders_arr)

    if x_ranges_arr.shape != (n_orders, 2):
        raise ValueError(
            f"x_ranges must have shape (n_orders, 2) = ({n_orders}, 2); "
            f"got {x_ranges_arr.shape}"
        )
    if edge_coeffs_arr.ndim != 3 or edge_coeffs_arr.shape[:2] != (n_orders, 2):
        raise ValueError(
            f"edge_coeffs must have shape (n_orders, 2, n_terms) = "
            f"({n_orders}, 2, ...); got {edge_coeffs_arr.shape}"
        )

    geometries = []
    for i in range(n_orders):
        geom = OrderGeometry(
            order=int(orders_arr[i]),
            x_start=int(x_ranges_arr[i, 0]),
            x_end=int(x_ranges_arr[i, 1]),
            bottom_edge_coeffs=edge_coeffs_arr[i, 0, :].copy(),
            top_edge_coeffs=edge_coeffs_arr[i, 1, :].copy(),
        )
        geometries.append(geom)

    return OrderGeometrySet(mode=mode, geometries=geometries)
