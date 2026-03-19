"""
Provisional rectification-index scaffold for iSHELL 2DXD reduction.

This module implements the **sixth stage** of the iSHELL 2DXD reduction
scaffold: constructing per-order provisional rectification indices that map
detector pixels to a rectified (wavelength × spatial) coordinate system.

What this module does
---------------------
* Accepts an :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
  (order edge polynomials and column ranges from flat tracing) and a
  :class:`~pyspextool.instruments.ishell.wavecal_2d_refine.RefinedCoefficientSurface`
  (the Stage-5 output).

* For each order, defines a provisional rectified grid with two axes:

  - a **spectral axis** uniformly spaced in wavelength (µm), spanning the
    order's wavelength range as predicted by the refined surface; and

  - a **spatial axis** uniformly spaced as a fractional slit position in
    ``[0, 1]``, where ``0`` is the bottom order edge and ``1`` is the top.

* For each output grid point ``(wavelength_i, spatial_frac_j)``, computes
  approximate detector coordinates ``(src_row, src_col)``:

  - ``src_col`` is obtained by sampling the refined surface at
    ``n_col_samples`` columns across the order's range and inverting the
    resulting wavelength → column curve via linear interpolation.

  - ``src_row`` is computed from the flat-tracing edge polynomials::

        src_row = bottom_edge(src_col) + spatial_frac * (top_edge(src_col)
                                                         - bottom_edge(src_col))

* Returns a :class:`RectificationIndexSet` collecting
  :class:`RectificationIndexOrder` objects, one per echelle order.

What this module does NOT do (by design)
-----------------------------------------
* **No final wavecal/spatcal image generation** – the index arrays produced
  here are *inputs* to a future resampling step; no science images are
  created.

* **No actual image resampling or interpolation** – detector pixels are not
  remapped onto the rectified grid in this stage.

* **No science-quality rectification** – the result is a provisional
  scaffold for development and pipeline validation.

* **No high-order curvature modelling** – spectral-line tilt and curvature
  corrections go beyond what the current flat-tracing + wavelength-surface
  scaffold supports and are not implemented here.

* **No iterative sigma-clipping** – the wavelength surface used for column
  inversion is taken as-is from Stage 5.

Coordinate conventions
-----------------------
The rectified grid uses two axes:

**Spectral axis** (wavelength-like)
    Uniformly spaced wavelengths in µm, spanning the range
    ``[wav_min, wav_max]`` for the order.  The range is estimated by
    evaluating the refined coefficient surface at both ends of the order's
    detector column range.

**Spatial axis** (slit-position-like)
    Fractional slit position in ``[0.0, 1.0]``: ``0.0`` is the bottom order
    edge as defined by the flat-tracing edge polynomial; ``1.0`` is the top
    edge.  Conversion to physical arcseconds requires a spatial calibration
    (``spatcal_coeffs`` on :class:`~pyspextool.instruments.ishell.geometry.OrderGeometry`)
    that is not yet available at this scaffold stage.

**Detector mapping**
    ``src_cols[i]`` – detector column (fractional) for spectral output pixel
    *i*, obtained by linear interpolation of the surface-sampled
    wavelength → column curve.

    ``src_rows[j, i]`` – detector row (fractional) for spatial output pixel
    *j* at spectral pixel *i*, obtained from the flat-tracing edge
    polynomials::

        row = bottom_edge(col) + frac * (top_edge(col) - bottom_edge(col))

    Both arrays contain fractional (sub-pixel) detector coordinates and are
    intended for later use with ``scipy.ndimage.map_coordinates`` or an
    equivalent interpolation routine.

Order matching convention
--------------------------
Orders are matched between *geometry* and *surface* by **echelle order
number**.  For each geometry order, the function looks up the corresponding
surface fit by order number (treating both as integers).  Geometry orders
with no surface fit are silently skipped; this is expected in real data
where some orders lack enough arc-line matches for the Stage-5 surface fit.
A :exc:`ValueError` is raised only if *no* orders are found in common.

If the geometry was produced by
:meth:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace.to_order_geometry_set`
it uses placeholder 0-indexed order numbers, not real echelle order numbers.
In that case, pass the
:class:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalWavelengthMap`
as the optional *wav_map* argument; the map's solution list is used to
resolve the placeholder index → echelle order number correspondence.

Assumptions and limitations
----------------------------
* The per-order wavelength function is assumed to be approximately monotone
  in detector column.  A :exc:`RuntimeWarning` is emitted if the sampled
  surface deviates from strict monotonicity, which may indicate that the
  column inversion is inaccurate.

* The column inversion accuracy scales with ``n_col_samples``.  The default
  value of 1024 is conservative for iSHELL detector widths (~2048 columns);
  increase it if sub-pixel accuracy is needed.

* The spatial mapping assumes straight, un-tilted slit edges.  Tilt or
  curvature corrections that require ``tilt_coeffs`` or ``curvature_coeffs``
  from the arc-line tracing step are not applied at this scaffold stage.

* This is a provisional scaffold.  See the module docstring and the
  developer notes in ``docs/ishell_rectification_indices.md`` for the full
  list of limitations and what remains for a full 2DXD solution.

Public API
----------
- :func:`build_rectification_indices` – main entry point.
- :class:`RectificationIndexOrder` – per-order result container.
- :class:`RectificationIndexSet` – full result container.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import numpy.typing as npt

from .geometry import OrderGeometrySet

if TYPE_CHECKING:
    from .wavecal_2d import ProvisionalWavelengthMap
    from .wavecal_2d_refine import RefinedCoefficientSurface

__all__ = [
    "RectificationIndexOrder",
    "RectificationIndexSet",
    "build_rectification_indices",
]


# ---------------------------------------------------------------------------
# Per-order rectification index
# ---------------------------------------------------------------------------


@dataclass
class RectificationIndexOrder:
    """Provisional rectification index for one echelle order.

    Stores the mapping between a provisional rectified
    (wavelength × spatial) grid and the corresponding approximate
    detector (row, column) coordinates.

    Parameters
    ----------
    order : int
        Echelle order number (or index, depending on what the upstream
        geometry uses; see :class:`~pyspextool.instruments.ishell.geometry.OrderGeometry`).
    order_index : int
        Zero-based position of this order in the parent
        :class:`RectificationIndexSet`.
    output_wavelengths_um : ndarray, shape (n_spectral,)
        Wavelength axis of the rectified grid, in µm.  Uniformly spaced
        from the minimum to the maximum wavelength predicted for this order
        by the refined coefficient surface.
    output_spatial_frac : ndarray, shape (n_spatial,)
        Spatial axis of the rectified grid as a fractional slit position
        in ``[0.0, 1.0]``.  ``0.0`` corresponds to the bottom order edge;
        ``1.0`` corresponds to the top order edge.
    src_cols : ndarray, shape (n_spectral,)
        Approximate detector column (fractional) for each output wavelength
        pixel.  Obtained by inverting the order's wavelength surface via
        linear interpolation over a dense column sample.
    src_rows : ndarray, shape (n_spatial, n_spectral)
        Approximate detector row (fractional) for each
        ``(spatial, spectral)`` output grid point.  ``src_rows[j, i]``
        is the row corresponding to spatial fraction
        ``output_spatial_frac[j]`` at column ``src_cols[i]``.

    Notes
    -----
    Both ``src_cols`` and ``src_rows`` contain *fractional* (sub-pixel)
    detector coordinates.  They are intended to be passed to
    ``scipy.ndimage.map_coordinates`` or an equivalent function with the
    appropriate interpolation order.

    No tilt or curvature correction is applied.  The spatial mapping uses
    the flat-tracing edge polynomials only.

    This is a provisional scaffold; see the module docstring and
    ``docs/ishell_rectification_indices.md`` for limitations.
    """

    order: int
    order_index: int
    output_wavelengths_um: npt.NDArray  # shape (n_spectral,)
    output_spatial_frac: npt.NDArray  # shape (n_spatial,)
    src_cols: npt.NDArray  # shape (n_spectral,)
    src_rows: npt.NDArray  # shape (n_spatial, n_spectral)

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def n_spectral(self) -> int:
        """Number of spectral pixels in the output grid."""
        return len(self.output_wavelengths_um)

    @property
    def n_spatial(self) -> int:
        """Number of spatial pixels in the output grid."""
        return len(self.output_spatial_frac)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the rectified output array: ``(n_spatial, n_spectral)``."""
        return (self.n_spatial, self.n_spectral)


# ---------------------------------------------------------------------------
# Full-set container
# ---------------------------------------------------------------------------


@dataclass
class RectificationIndexSet:
    """Collection of per-order provisional rectification indices.

    Stores one :class:`RectificationIndexOrder` per echelle order, plus
    the mode name from the parent geometry.

    Parameters
    ----------
    mode : str
        iSHELL observing mode (e.g. ``"H1"``).
    index_orders : list of :class:`RectificationIndexOrder`
        Per-order rectification indices.  One entry per echelle order,
        in the same order as the input
        :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`.
    """

    mode: str
    index_orders: list[RectificationIndexOrder] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    @property
    def orders(self) -> list[int]:
        """List of order numbers in the set (in storage order)."""
        return [io.order for io in self.index_orders]

    @property
    def n_orders(self) -> int:
        """Number of orders in the set."""
        return len(self.index_orders)

    def get_order(self, order: int) -> RectificationIndexOrder:
        """Return the :class:`RectificationIndexOrder` for a given order number.

        Parameters
        ----------
        order : int
            Order number (or index) to look up.

        Returns
        -------
        :class:`RectificationIndexOrder`

        Raises
        ------
        KeyError
            If *order* is not present in the set.
        """
        for io in self.index_orders:
            if io.order == order:
                return io
        raise KeyError(
            f"Order {order} not found in RectificationIndexSet for mode "
            f"{self.mode!r}.  Available orders: {self.orders}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_rectification_indices(
    geometry: OrderGeometrySet,
    surface: Optional["RefinedCoefficientSurface"] = None,
    *,
    wavelength_func: Optional[Callable[[npt.ArrayLike, float], npt.NDArray]] = None,
    fitted_order_numbers: Optional[list[int]] = None,
    wav_map: Optional["ProvisionalWavelengthMap"] = None,
    n_spectral: int = 256,
    n_spatial: int = 64,
    n_col_samples: int = 1024,
) -> RectificationIndexSet:
    """Build provisional per-order rectification indices.

    For each echelle order in *geometry*, constructs a mapping from a
    provisional rectified (wavelength × spatial) grid to approximate
    detector (row, column) coordinates.

    Two wavelength-evaluation paths are supported:

    **Scaffold path** (``surface`` is provided):
        Uses the :class:`~pyspextool.instruments.ishell.wavecal_2d_refine.RefinedCoefficientSurface`
        from Stage 5 to predict wavelength at each column.  Orders are
        matched between *geometry* and *surface* by echelle order number.

    **K3 1DXD path** (``wavelength_func`` is provided):
        Uses a callable ``wavelength_func(cols, order_number)`` to predict
        wavelength.  The callable must accept a 1-D column array and a
        scalar order number, and return a 1-D wavelength array.  Pass
        ``fitted_order_numbers`` to specify which orders the function covers;
        only those geometry orders are processed.  This path is intended for
        the IDL-style 1DXD model from
        :mod:`~pyspextool.instruments.ishell.wavecal_k3_idlstyle`.

    Parameters
    ----------
    geometry : :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet`
        Order geometry from flat tracing (edge polynomials and column
        ranges).  Orders are matched to surface fits by echelle order
        number (the ``order`` attribute of each
        :class:`~pyspextool.instruments.ishell.geometry.OrderGeometry`).
        If the geometry was produced by
        :meth:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace.to_order_geometry_set`
        it uses placeholder 0-indexed order numbers; in that case pass
        *wav_map* (scaffold path) or *fitted_order_numbers* (1DXD path) to
        supply the real order-number correspondence.
    surface : :class:`~pyspextool.instruments.ishell.wavecal_2d_refine.RefinedCoefficientSurface`, optional
        Refined coefficient surface from Stage 5 (**scaffold path**).
        Mutually exclusive with *wavelength_func*.
    wavelength_func : callable, optional
        **K3 1DXD path.**  A callable with signature::

            wavelength_func(cols: array_like, order_number: float) -> ndarray

        where *cols* is a 1-D array of detector columns and *order_number*
        is a scalar echelle order number.  The return value must be a 1-D
        array of wavelengths in µm with the same length as *cols*.
        Mutually exclusive with *surface*.
    fitted_order_numbers : list of int, optional
        **K3 1DXD path only.**  Echelle order numbers covered by
        *wavelength_func*.  Orders in *geometry* whose echelle order number
        (or whose placeholder index resolved via *wav_map*) is not in this
        list are silently skipped.  If ``None`` when *wavelength_func* is
        given, all geometry orders are attempted.
    wav_map : :class:`~pyspextool.instruments.ishell.wavecal_2d.ProvisionalWavelengthMap`, optional
        Provisional wavelength map from Stage 3.  When provided, its
        solution list is used to resolve placeholder geometry order
        indices to real echelle order numbers.  Required when *geometry*
        contains 0-indexed placeholder order numbers (as produced by
        :meth:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace.to_order_geometry_set`).
        Valid for both paths.
    n_spectral : int, default 256
        Number of spectral (wavelength) pixels in the output rectified
        grid.  Must be >= 2.
    n_spatial : int, default 64
        Number of spatial pixels in the output rectified grid.  Must be
        >= 2.
    n_col_samples : int, default 1024
        Number of detector columns sampled when inverting the wavelength
        function to obtain ``src_cols``.  Higher values give a more
        accurate inversion.  Must be >= *n_spectral*.

    Returns
    -------
    :class:`RectificationIndexSet`
        One :class:`RectificationIndexOrder` per matched echelle order.

    Raises
    ------
    ValueError
        If both *surface* and *wavelength_func* are provided, or neither
        is provided.
    ValueError
        If *geometry* is empty, no orders are found in common, or
        *n_spectral* < 2, *n_spatial* < 2, or *n_col_samples* < *n_spectral*.

    Warns
    -----
    RuntimeWarning
        If the sampled wavelength grid for any order is not
        monotone in detector column.  The column inversion may be
        inaccurate in that case.

    Notes
    -----
    **Algorithm**

    For each order *i*:

    1. Sample the wavelength function at ``n_col_samples`` detector columns
       spanning ``[x_start, x_end]`` to obtain a dense
       ``(col, wavelength)`` curve.

    2. Sort by wavelength and build a uniform output wavelength axis::

           output_wavs = np.linspace(wav_min, wav_max, n_spectral)

    3. Invert wavelength → column via linear interpolation::

           src_cols = np.interp(output_wavs, sorted_wavs, sorted_cols)

    4. Build the spatial axis::

           spatial_frac = np.linspace(0.0, 1.0, n_spatial)

    5. Compute ``src_rows`` from the flat-tracing edge polynomials::

           src_rows[j, i] = (
               bottom_edge(src_cols[i])
               + spatial_frac[j] * (top_edge(src_cols[i])
                                    - bottom_edge(src_cols[i]))
           )

    **Order matching (scaffold path)**

    Orders are matched by echelle order number.  If *wav_map* is provided,
    it is used to build a mapping from geometry placeholder index to real
    echelle order number via ``wav_map.solutions``.  Otherwise, the
    ``order`` attribute of each geometry is used directly as the order
    number.

    Geometry orders with no corresponding surface fit are skipped silently;
    this is expected for real data where some orders lack enough arc-line
    matches.  A :exc:`ValueError` is raised only if *no* orders are found
    in common.

    **Order matching (K3 1DXD path)**

    Same placeholder-index resolution as the scaffold path (via *wav_map*
    if provided).  If *fitted_order_numbers* is given, only orders whose
    echelle order number appears in that list are processed; others are
    skipped silently.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if surface is not None and wavelength_func is not None:
        raise ValueError(
            "Provide either 'surface' (scaffold path) or 'wavelength_func' "
            "(K3 1DXD path), not both."
        )
    if surface is None and wavelength_func is None:
        raise ValueError(
            "Either 'surface' (scaffold path) or 'wavelength_func' "
            "(K3 1DXD path) must be provided."
        )

    if n_spectral < 2:
        raise ValueError(f"n_spectral must be >= 2; got {n_spectral}")
    if n_spatial < 2:
        raise ValueError(f"n_spatial must be >= 2; got {n_spatial}")
    if n_col_samples < n_spectral:
        raise ValueError(
            f"n_col_samples ({n_col_samples}) must be >= n_spectral ({n_spectral})"
        )

    n_geom = geometry.n_orders

    if n_geom == 0:
        raise ValueError("geometry contains no orders")

    # ------------------------------------------------------------------
    # Scaffold path: validate surface
    # ------------------------------------------------------------------
    surf_order_lookup: dict[int, int] = {}
    if surface is not None:
        n_surf = len(surface.per_order_order_numbers)
        if n_surf == 0:
            raise ValueError(
                "surface has no fitted orders (per_order_order_numbers is empty)"
            )
        surf_order_lookup = {
            int(round(float(on))): i
            for i, on in enumerate(surface.per_order_order_numbers)
        }

    # ------------------------------------------------------------------
    # 1DXD path: build the set of covered order numbers
    # ------------------------------------------------------------------
    func_order_set: Optional[set[int]] = None
    if wavelength_func is not None and fitted_order_numbers is not None:
        func_order_set = set(int(o) for o in fitted_order_numbers)

    # ------------------------------------------------------------------
    # Build geometry-order → echelle-order-number mapping
    # ------------------------------------------------------------------
    # When wav_map is provided, use its solutions to map geometry
    # placeholder indices to real echelle order numbers.  Otherwise,
    # treat geometry.order directly as the echelle order number.
    if wav_map is not None:
        geom_order_to_echelle: dict[int, int] = {
            sol.order_index: int(round(float(sol.order_number)))
            for sol in wav_map.order_solutions
        }
    else:
        geom_order_to_echelle = {
            int(geom.order): int(geom.order) for geom in geometry.geometries
        }

    # ------------------------------------------------------------------
    # Build per-order indices
    # ------------------------------------------------------------------
    index_orders: list[RectificationIndexOrder] = []

    for geom in geometry.geometries:
        echelle_order = geom_order_to_echelle.get(int(geom.order))
        if echelle_order is None:
            # No wav_map entry for this geometry order; skip.
            continue

        # ---- Select wavelength evaluation method ----
        if wavelength_func is not None:
            # K3 1DXD path: check order is covered
            if func_order_set is not None and echelle_order not in func_order_set:
                continue
            order_number = float(echelle_order)

            def _eval_wav(col_grid, _order_number=order_number):  # noqa: E731
                return wavelength_func(col_grid, _order_number)

        else:
            # Scaffold path: look up in surface
            surf_i = surf_order_lookup.get(echelle_order)
            if surf_i is None:
                # No surface fit for this echelle order; skip silently.
                continue
            order_number = float(surface.per_order_order_numbers[surf_i])  # type: ignore[union-attr]

            def _eval_wav(col_grid, _order_number=order_number):  # noqa: E731
                return surface.eval_array(  # type: ignore[union-attr]
                    np.full(len(col_grid), _order_number), col_grid
                )

        order_index = len(index_orders)  # position in the output set

        # Step 1: sample the wavelength function across the column range.
        col_grid = np.linspace(float(geom.x_start), float(geom.x_end), n_col_samples)
        wav_grid = _eval_wav(col_grid)

        # Check for non-monotonicity and warn.
        diffs = np.diff(wav_grid)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            warnings.warn(
                f"Order {geom.order} (order_number={order_number}): the sampled "
                "wavelength surface is not strictly monotone in detector "
                "column.  The column-inversion for rectification may be "
                "inaccurate.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Step 2: sort by wavelength to allow monotone interpolation.
        sort_idx = np.argsort(wav_grid)
        sorted_wavs = wav_grid[sort_idx]
        sorted_cols = col_grid[sort_idx]

        # Step 3: build output wavelength axis and invert to get src_cols.
        wav_min = sorted_wavs[0]
        wav_max = sorted_wavs[-1]
        output_wavs = np.linspace(wav_min, wav_max, n_spectral)
        src_cols = np.interp(output_wavs, sorted_wavs, sorted_cols)

        # Step 4: build spatial (slit-fraction) axis.
        output_spatial_frac = np.linspace(0.0, 1.0, n_spatial)

        # Step 5: map (spatial_frac, col) → detector row via edge polynomials.
        # bottom_rows / top_rows both have shape (n_spectral,).
        bottom_rows = geom.eval_bottom_edge(src_cols)
        top_rows = geom.eval_top_edge(src_cols)
        # Broadcasting:  frac[:, None] * width[None, :]
        # src_rows shape: (n_spatial, n_spectral)
        src_rows = bottom_rows[np.newaxis, :] + output_spatial_frac[
            :, np.newaxis
        ] * (top_rows - bottom_rows)[np.newaxis, :]

        index_orders.append(
            RectificationIndexOrder(
                order=geom.order,
                order_index=order_index,
                output_wavelengths_um=output_wavs,
                output_spatial_frac=output_spatial_frac,
                src_cols=src_cols,
                src_rows=src_rows,
            )
        )

    if len(index_orders) == 0:
        if wavelength_func is not None:
            raise ValueError(
                "No orders matched between geometry and wavelength_func.  "
                f"Geometry orders: {geometry.orders}.  "
                f"fitted_order_numbers: {fitted_order_numbers}.  "
                "If the geometry uses placeholder 0-indexed order numbers, pass "
                "the ProvisionalWavelengthMap as wav_map to supply the real "
                "order-number correspondence."
            )
        raise ValueError(
            "No orders matched between geometry and surface.  "
            f"Geometry orders: {geometry.orders}.  "
            f"Surface order numbers: {list(surface.per_order_order_numbers)}.  "  # type: ignore[union-attr]
            "If the geometry uses placeholder 0-indexed order numbers, pass "
            "the ProvisionalWavelengthMap as wav_map to supply the real "
            "order-number correspondence."
        )

    return RectificationIndexSet(mode=geometry.mode, index_orders=index_orders)
