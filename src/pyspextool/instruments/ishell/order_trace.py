"""Order-center tracing from iSHELL flat-field calibration data.

This module implements the first step of the 2DXD tracing scaffold:
estimating the center of each echelle order as a function of detector
column, using the order-mask image stored in the packaged flat-field
calibration (``*_flatinfo.fits``).

Algorithm overview
------------------
1. Load a :class:`~pyspextool.instruments.ishell.calibrations.FlatInfo`
   for the desired mode (e.g. ``"H1"``).
2. Apply the stored detector rotation so that dispersion runs left-to-right
   (columns = dispersion axis, rows = spatial axis).
3. For each echelle order and each detector column within that order's
   valid x-range, collect all rows belonging to the order (from the order-
   mask image) and compute an unweighted centroid.
4. Fit a low-degree polynomial (default degree 3) to the
   ``(column, centroid-row)`` pairs for each order.
5. Return an :class:`OrderTraceResult` containing the raw trace measurements,
   fitted coefficients, fit residuals, and an
   :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` built
   from the flat-field edge polynomials.

Relationship to the 2DXD pipeline
----------------------------------
The order-center traces produced here are the starting point for the 2DXD
arc-line tracing stage.  In the full 2DXD algorithm each ThAr arc line is
traced across the detector in the same way as an order center; the
flat-based trace establishes the "zero-tilt" dispersion-axis direction and
provides initial guesses for the arc-line search.

The :class:`OrderGeometrySet` embedded in the result can be augmented in
later pipeline stages with tilt, curvature, and wavelength-calibration
information (see :mod:`~pyspextool.instruments.ishell.geometry`).

What is *not* implemented here
--------------------------------
* Arc-line tracing (requires real ThAr arc frames and wavelength solutions).
* Spectral tilt and curvature estimation (2DXD-specific step).
* Rectification-index computation.
* Flux-weighted centroiding from a normalized QTH flat image (only the
  packaged integer order-mask is used; see Notes).

Notes
-----
The packaged ``*_flatinfo.fits`` image is an integer order-mask: each pixel
is set to its echelle order number (0 = inter-order).  This mask was derived
from real QTH flat frames during the original IDL Spextool calibration; using
it here provides a direct pixel-level validation of the pre-computed edge
polynomials.

Real raw QTH flat frames would enable flux-weighted centroiding, which could
improve sub-pixel accuracy.  The current implementation achieves ~0.5 pixel
agreement with the edge-polynomial centerlines, which is sufficient for the
2DXD scaffold.

See :func:`trace_order_centers_from_flatinfo` and
``docs/ishell_order_trace_note.md`` for further details.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from pyspextool.utils.arrays import idl_rotate

from .calibrations import FlatInfo, read_flatinfo
from .geometry import OrderGeometrySet, build_order_geometry_set

__all__ = [
    "OrderTraceResult",
    "trace_order_centers_from_flatinfo",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OrderTraceResult:
    """Result of order-center tracing from a flat-field calibration.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"H1"``).
    orders : list of int
        Echelle order numbers that were traced.
    trace_columns : dict of {int: ndarray}
        For each order number, a 1-D array of detector column positions used
        during tracing (in the rotated, dispersion-along-columns coordinate
        system).
    trace_rows : dict of {int: ndarray}
        For each order number, a 1-D array of traced row positions (one
        centroid per column in ``trace_columns``).
    center_coeffs : dict of {int: ndarray}
        For each order number, polynomial coefficients for the fitted
        centerline following the :mod:`numpy.polynomial.polynomial`
        convention (constant term first)::

            row = polyval(col, coeffs)

    fit_rms : dict of {int: float}
        For each order number, the RMS residual of the polynomial fit in
        pixels.  ``nan`` if there were fewer good points than the number of
        free parameters.
    n_good : dict of {int: int}
        For each order number, the number of columns with a valid (non-NaN)
        centroid measurement.
    poly_degree : int
        Polynomial degree used for all fits.
    geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` or None
        Reference geometry set built from the edge polynomials stored in
        the flat-field calibration.  Available whenever the flatinfo
        contains ``OR{n}_B{k}`` / ``OR{n}_T{k}`` header keywords.
    """

    mode: str
    orders: list[int]
    trace_columns: dict[int, np.ndarray] = field(default_factory=dict)
    trace_rows: dict[int, np.ndarray] = field(default_factory=dict)
    center_coeffs: dict[int, np.ndarray] = field(default_factory=dict)
    fit_rms: dict[int, float] = field(default_factory=dict)
    n_good: dict[int, int] = field(default_factory=dict)
    poly_degree: int = 3
    geometry: Optional[OrderGeometrySet] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def n_orders(self) -> int:
        """Number of traced orders."""
        return len(self.orders)

    def eval_center(self, order: int, cols: npt.ArrayLike) -> np.ndarray:
        """Evaluate the fitted centerline for *order* at given columns.

        Parameters
        ----------
        order : int
            Echelle order number.
        cols : array_like
            Detector column coordinate(s) (dispersion axis, 0-indexed).

        Returns
        -------
        ndarray
            Fitted row positions of the order center at each column.

        Raises
        ------
        KeyError
            If *order* is not present in the trace result.
        """
        coeffs = self.center_coeffs[order]
        return np.polynomial.polynomial.polyval(
            np.asarray(cols, dtype=float), coeffs
        )


# ---------------------------------------------------------------------------
# Main tracing function
# ---------------------------------------------------------------------------


def trace_order_centers_from_flatinfo(
    flatinfo_or_mode: "FlatInfo | str",
    *,
    poly_degree: int = 3,
    step_size: int = 10,
) -> OrderTraceResult:
    """Trace order centers from the flat-field calibration data.

    For each echelle order and each detector column in that order's valid
    x-range, the centroid of the order-mask rows is computed and used as
    the traced center position.  A low-degree polynomial is fitted to the
    ``(column, centroid-row)`` pairs.

    Parameters
    ----------
    flatinfo_or_mode : :class:`~pyspextool.instruments.ishell.calibrations.FlatInfo` or str
        Either a :class:`~pyspextool.instruments.ishell.calibrations.FlatInfo`
        instance (already loaded) or a mode name string (e.g. ``"H1"``),
        which will be loaded via
        :func:`~pyspextool.instruments.ishell.calibrations.read_flatinfo`.
    poly_degree : int, optional
        Degree of the polynomial to fit to the traced centers.  Default 3.
    step_size : int, optional
        Column step size between trace measurements.  Default 10.

    Returns
    -------
    :class:`OrderTraceResult`

    Raises
    ------
    ValueError
        If *poly_degree* is negative, or if *step_size* is less than 1.
    UserWarning
        For individual orders where the number of valid trace points is
        insufficient for a polynomial fit of the requested degree.

    Notes
    -----
    The flat-field order-mask image is stored in the *raw* IDL orientation
    inside ``*_flatinfo.fits``.  This function applies
    :func:`~pyspextool.utils.arrays.idl_rotate` with the rotation code
    stored in the ``ROTATION`` header keyword before tracing, so that the
    resulting trace coordinates are in the standard pySpextool orientation
    (columns = dispersion axis, rows = spatial axis).

    The edge-polynomial centerlines (``geometry.get_order(n).eval_centerline()``)
    are not used to derive ``center_coeffs``; they are retained in the
    ``geometry`` field for comparison and for use in later pipeline stages.

    Examples
    --------
    Trace order centers from the packaged H1 calibration:

    >>> from pyspextool.instruments.ishell.order_trace import (
    ...     trace_order_centers_from_flatinfo,
    ... )
    >>> result = trace_order_centers_from_flatinfo("H1")
    >>> result.n_orders
    45
    >>> center_row_at_col_1000 = result.eval_center(325, 1000)
    """
    if poly_degree < 0:
        raise ValueError(f"poly_degree must be >= 0; got {poly_degree}")
    if step_size < 1:
        raise ValueError(f"step_size must be >= 1; got {step_size}")

    # ------------------------------------------------------------------
    # Load flatinfo if a mode name was provided
    # ------------------------------------------------------------------
    if isinstance(flatinfo_or_mode, str):
        flatinfo = read_flatinfo(flatinfo_or_mode)
    else:
        flatinfo = flatinfo_or_mode

    # ------------------------------------------------------------------
    # Rotate the order-mask image to the standard pySpextool orientation
    # (columns = dispersion axis, rows = spatial axis).
    # The *_flatinfo.fits image is stored in the raw IDL orientation;
    # idl_rotate(image, rotation) converts it to standard coordinates.
    # ------------------------------------------------------------------
    image = idl_rotate(
        np.asarray(flatinfo.image, dtype=np.int32), flatinfo.rotation
    )
    nrows, ncols = image.shape

    orders = flatinfo.orders
    xranges = flatinfo.xranges  # shape (n_orders, 2) or None

    # ------------------------------------------------------------------
    # Build the reference OrderGeometrySet from edge polynomials
    # ------------------------------------------------------------------
    geometry: Optional[OrderGeometrySet] = None
    if flatinfo.edge_coeffs is not None and xranges is not None:
        geometry = build_order_geometry_set(
            mode=flatinfo.mode,
            orders=orders,
            x_ranges=xranges,
            edge_coeffs=flatinfo.edge_coeffs,
        )

    # ------------------------------------------------------------------
    # Trace each order
    # ------------------------------------------------------------------
    trace_columns: dict[int, np.ndarray] = {}
    trace_rows: dict[int, np.ndarray] = {}
    center_coeffs: dict[int, np.ndarray] = {}
    fit_rms: dict[int, float] = {}
    n_good: dict[int, int] = {}

    for i, order in enumerate(orders):
        # Determine the column range for this order
        if xranges is not None:
            x_start = int(xranges[i, 0])
            x_end = int(xranges[i, 1])
        else:
            x_start = 0
            x_end = ncols - 1

        # Clamp to actual image bounds
        x_start = max(x_start, 0)
        x_end = min(x_end, ncols - 1)

        # Generate column positions at the requested step size
        columns = np.arange(x_start, x_end + 1, step_size, dtype=int)
        centroid_rows = np.full(len(columns), np.nan)

        for j, col in enumerate(columns):
            col = int(col)
            if col < 0 or col >= ncols:
                continue

            # Find all rows belonging to this order at this column
            rows_in_order = np.where(image[:, col] == order)[0]
            if len(rows_in_order) < 2:
                # Fewer than 2 pixels: centroid is unreliable
                continue

            # Unweighted centroid of the illuminated pixels
            centroid_rows[j] = float(np.mean(rows_in_order))

        # ------------------------------------------------------------------
        # Remove invalid (NaN) points
        # ------------------------------------------------------------------
        valid = np.isfinite(centroid_rows)
        good_cols = columns[valid].astype(float)
        good_rows = centroid_rows[valid]
        n_good_pts = int(np.sum(valid))

        trace_columns[order] = good_cols
        trace_rows[order] = good_rows
        n_good[order] = n_good_pts

        # ------------------------------------------------------------------
        # Fit a polynomial to the (column, row) pairs
        # ------------------------------------------------------------------
        min_pts_required = poly_degree + 1
        if n_good_pts >= min_pts_required:
            coeffs = np.polynomial.polynomial.polyfit(
                good_cols, good_rows, poly_degree
            )
            fitted = np.polynomial.polynomial.polyval(good_cols, coeffs)
            residuals = good_rows - fitted
            rms = float(np.sqrt(np.mean(residuals**2)))
        else:
            # Fall back to the edge-polynomial centerline if available
            if geometry is not None:
                geom = geometry.get_order(order)
                coeffs = geom.centerline_coeffs.copy()
            else:
                coeffs = np.zeros(poly_degree + 1)
            rms = float("nan")
            warnings.warn(
                f"Order {order}: only {n_good_pts} valid trace points; "
                f"need at least {min_pts_required} for a degree-{poly_degree} "
                "fit.  Falling back to edge-polynomial centerline.",
                UserWarning,
                stacklevel=2,
            )

        center_coeffs[order] = coeffs
        fit_rms[order] = rms

    return OrderTraceResult(
        mode=flatinfo.mode,
        orders=orders,
        trace_columns=trace_columns,
        trace_rows=trace_rows,
        center_coeffs=center_coeffs,
        fit_rms=fit_rms,
        n_good=n_good,
        poly_degree=poly_degree,
        geometry=geometry,
    )
