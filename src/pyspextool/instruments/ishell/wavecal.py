"""
iSHELL-specific geometry population and rectification support.

.. warning::
    **This module is a structural bootstrap, not a full wavecal pipeline.**

    It reads the pre-packaged ``*_wavecalinfo.fits`` files and fits
    polynomials to the values stored in plane 0 of the data cube.  Those
    stored values are *inferred* to represent wavelengths in microns (the
    numbers fall in the expected J/H/K band ranges and the FITS header
    contains 2DXD polynomial coefficients), but the exact provenance and
    physical meaning of each data-cube plane are **not formally documented**
    in the packaged files.  See the "Confirmed vs inferred" table in
    ``docs/ishell_wavecal_design_note.md``.

    In particular:

    * Tilt coefficients are set to a **placeholder zero** – no real
      spectral-line tilt measurement is performed.  With zero tilt the
      resampling step is an identity within each order footprint.
    * Spatial calibration uses a simple linear plate-scale model with no
      curvature correction.
    * The implementation does **not** replicate the interactive 1DXD/2DXD
      arc-line fitting procedure from legacy IDL Spextool.

Public API
----------
- :func:`build_geometry_from_wavecalinfo` – populate an
  :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` from
  stored ``WaveCalInfo`` and ``FlatInfo`` calibrations.
- :func:`build_rectification_maps` – create a
  :class:`~pyspextool.instruments.ishell.geometry.RectificationMap` for
  every order in an ``OrderGeometrySet``.

Implementation approach (polynomial fit to stored reference arrays)
--------------------------------------------------------------------
For each echelle order this module:

1. Extracts the stored array from plane 0 of ``WaveCalInfo.data``
   (structurally validated; semantics inferred as wavelengths in µm).
2. Maps array indices to detector columns using the ``xranges`` parsed
   from the FITS header (``OR{n}_XR`` keywords).
3. Fits a polynomial (degree = ``wavecalinfo.disp_degree``) to
   ``(columns, plane-0 values)`` within the valid column range.
4. Stores the fitted polynomial coefficients as
   :attr:`~pyspextool.instruments.ishell.geometry.OrderGeometry.wave_coeffs`.

This is **not** a re-implementation of the 1DXD/2DXD wavecal procedure;
it is a compact representation of the values that were already stored in
the packaged reference files.

Tilt model (placeholder: constant zero)
----------------------------------------
``tilt_coeffs`` is set to ``[0.0]`` for every order.  This is a
**temporary placeholder** – no arc-line centroid fitting is performed.
With zero tilt the rectification step applies no horizontal correction and
the output within each order footprint equals the bilinearly-interpolated
input.  Science-quality rectification requires real tilt measurement from
ThAr arc frames (see *What remains incomplete* below).

Spatial calibration (provisional linear model)
-----------------------------------------------
The spatial calibration maps row offset from the order centerline to
arcseconds::

    arcsec = spatcal_coeffs[0] + spatcal_coeffs[1] * row_offset

A simple linear model ``spatcal_coeffs = [0.0, plate_scale_arcsec]`` is
used, derived from ``FlatInfo.plate_scale_arcsec``.  No curvature
correction is applied.

What remains incomplete relative to legacy IDL Spextool
---------------------------------------------------------
1.  **Real tilt measurement** – the IDL pipeline fits arc-line centroids
    at multiple rows to measure spectral-line tilt as a function of column.
    This requires ThAr arc frames, not just stored calibration data.
2.  **Slit curvature / higher-order distortion** – curvature within each
    order is not modelled by the current zero-tilt placeholder.
3.  **Full 2DXD wavecal iteration** – interactive arc-line identification
    and iterative outlier rejection are not implemented here; the stored
    reference arrays are used directly.
4.  **Wavelength uncertainty propagation** – fit residuals and covariance
    are not stored in the geometry objects.
5.  **Plane 0 semantic validation** – the assumption that plane 0 contains
    wavelengths in µm is inferred from value ranges, not formally
    documented.  Planes 1–3 of the data cube are not used and their
    meanings are not confirmed.
6.  **L/Lp/M modes** – out of scope per the problem statement.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .geometry import OrderGeometry, OrderGeometrySet, RectificationMap

if TYPE_CHECKING:
    from .calibrations import FlatInfo, WaveCalInfo

__all__ = [
    "build_geometry_from_wavecalinfo",
    "build_rectification_maps",
]

# iSHELL default plate scale (arcsec/pixel) – used as fallback when
# FlatInfo.plate_scale_arcsec is NaN or missing.
_DEFAULT_PLATE_SCALE = 0.125


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_geometry_from_wavecalinfo(
    wavecalinfo: "WaveCalInfo",
    flatinfo: "FlatInfo",
    dispersion_degree: int | None = None,
) -> OrderGeometrySet:
    """Populate an :class:`~.geometry.OrderGeometrySet` from stored calibrations.

    Reads plane 0 of the ``WaveCalInfo`` data cube (inferred to contain
    wavelength-like values in µm) and fits a polynomial to those values for
    each order.  Edge polynomials come from the ``FlatInfo``.  The tilt is
    set to a **placeholder zero** and the spatial calibration uses a simple
    linear plate-scale model – neither is scientifically validated.

    Parameters
    ----------
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Stored calibration metadata for the mode.  Must have ``xranges``
        populated (parsed from the ``OR{n}_XR`` FITS header keywords).
    flatinfo : :class:`~.calibrations.FlatInfo`
        Flat-field calibration for the same mode.  Must have
        ``edge_coeffs`` and ``xranges`` populated.
    dispersion_degree : int or None, optional
        Polynomial degree for the per-order fit to plane 0 values.
        Defaults to ``wavecalinfo.disp_degree`` when ``None``.

    Returns
    -------
    :class:`~.geometry.OrderGeometrySet`
        Geometry set with ``wave_coeffs``, ``tilt_coeffs`` (placeholder
        zero), and ``spatcal_coeffs`` (linear plate-scale model) populated
        for every order.

    Raises
    ------
    ValueError
        If ``flatinfo`` and ``wavecalinfo`` refer to different modes, or
        if ``flatinfo.edge_coeffs`` / ``wavecalinfo.xranges`` are not
        available, or if order lists do not match.
    """
    _validate_inputs(wavecalinfo, flatinfo)

    mode = wavecalinfo.mode
    degree = int(dispersion_degree) if dispersion_degree is not None else int(
        wavecalinfo.disp_degree
    )

    plate_scale = _get_plate_scale(flatinfo)
    geometries = []

    for i, order_num in enumerate(wavecalinfo.orders):
        # ------------------------------------------------------------------
        # Edge geometry from FlatInfo
        # ------------------------------------------------------------------
        edge_c = flatinfo.edge_coeffs[i]  # shape (2, n_terms)
        x_start, x_end = flatinfo.xranges[i]

        # ------------------------------------------------------------------
        # Wavelength solution from stored WaveCalInfo data cube
        # ------------------------------------------------------------------
        wave_coeffs = _derive_wave_coeffs(
            wavecalinfo=wavecalinfo,
            order_idx=i,
            x_start=int(x_start),
            x_end=int(x_end),
            degree=degree,
        )

        # ------------------------------------------------------------------
        # Tilt (provisional: constant zero)
        # ------------------------------------------------------------------
        tilt_coeffs = np.array([0.0])

        # ------------------------------------------------------------------
        # Spatial calibration (linear: arcsec = plate_scale * row_offset)
        # ------------------------------------------------------------------
        spatcal_coeffs = np.array([0.0, plate_scale])

        geom = OrderGeometry(
            order=int(order_num),
            x_start=int(x_start),
            x_end=int(x_end),
            bottom_edge_coeffs=edge_c[0].copy(),
            top_edge_coeffs=edge_c[1].copy(),
            wave_coeffs=wave_coeffs,
            tilt_coeffs=tilt_coeffs,
            spatcal_coeffs=spatcal_coeffs,
        )
        geometries.append(geom)

    return OrderGeometrySet(mode=mode, geometries=geometries)


def build_rectification_maps(
    geom_set: OrderGeometrySet,
    plate_scale_arcsec: float,
    slit_height_arcsec: float | None = None,
    n_spatial: int | None = None,
) -> list[RectificationMap]:
    """Build a :class:`~.geometry.RectificationMap` for every order.

    Computes fractional source detector coordinates ``(src_rows,
    src_cols)`` for each point on a regular (wavelength × spatial) output
    grid, ready for resampling with
    ``scipy.ndimage.map_coordinates``.

    Parameters
    ----------
    geom_set : :class:`~.geometry.OrderGeometrySet`
        Geometry set with ``wave_coeffs`` and ``tilt_coeffs`` populated.
    plate_scale_arcsec : float
        Spatial plate scale in arcsec/pixel.
    slit_height_arcsec : float or None, optional
        Full slit height in arcsec.  If ``None``, the spatial grid is
        determined from the order edge polynomials evaluated at the
        midpoint column.
    n_spatial : int or None, optional
        Number of spatial output pixels.  If ``None``, inferred from
        ``slit_height_arcsec / plate_scale_arcsec``.

    Returns
    -------
    list of :class:`~.geometry.RectificationMap`
        One entry per order in ``geom_set``, in the same order as
        ``geom_set.geometries``.

    Raises
    ------
    ValueError
        If ``geom_set`` has no wavelength solution.
    """
    if not geom_set.has_wavelength_solution():
        raise ValueError(
            "OrderGeometrySet does not have a wavelength solution.  "
            "Call build_geometry_from_wavecalinfo() first."
        )

    maps = []
    for geom in geom_set.geometries:
        rmap = _build_order_rectification_map(
            geom=geom,
            plate_scale_arcsec=plate_scale_arcsec,
            slit_height_arcsec=slit_height_arcsec,
            n_spatial=n_spatial,
        )
        maps.append(rmap)
    return maps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(wavecalinfo: "WaveCalInfo", flatinfo: "FlatInfo") -> None:
    """Raise ValueError if the two calibrations are inconsistent."""
    if wavecalinfo.mode != flatinfo.mode:
        raise ValueError(
            f"wavecalinfo.mode={wavecalinfo.mode!r} does not match "
            f"flatinfo.mode={flatinfo.mode!r}."
        )
    if wavecalinfo.orders != flatinfo.orders:
        raise ValueError(
            f"Order lists differ between wavecalinfo and flatinfo for "
            f"mode '{wavecalinfo.mode}'."
        )
    if flatinfo.edge_coeffs is None:
        raise ValueError(
            f"flatinfo for mode '{flatinfo.mode}' does not have edge "
            "coefficients (edge_coeffs is None).  Re-read the flatinfo "
            "with a version of read_flatinfo() that parses OR{n}_B/T headers."
        )
    if flatinfo.xranges is None:
        raise ValueError(
            f"flatinfo for mode '{flatinfo.mode}' does not have per-order "
            "column ranges (xranges is None).  Re-read the flatinfo with a "
            "version that parses OR{n}_XR headers."
        )
    if wavecalinfo.xranges is None:
        raise ValueError(
            f"wavecalinfo for mode '{wavecalinfo.mode}' does not have per-order "
            "column ranges (xranges is None).  Re-read with a version that "
            "parses OR{n}_XR headers."
        )


def _get_plate_scale(flatinfo: "FlatInfo") -> float:
    """Return a finite plate scale from FlatInfo, or the default."""
    ps = flatinfo.plate_scale_arcsec
    if np.isfinite(ps) and ps > 0:
        return float(ps)
    return _DEFAULT_PLATE_SCALE


def _derive_wave_coeffs(
    wavecalinfo: "WaveCalInfo",
    order_idx: int,
    x_start: int,
    x_end: int,
    degree: int,
) -> np.ndarray:
    """Fit a polynomial to plane 0 of the stored data cube for one order.

    .. note::
        The physical meaning of plane 0 is **inferred** (values fall in the
        expected J/H/K wavelength ranges in µm) but is not formally
        documented.  The column mapping ``array_index -> x_start + array_index``
        is a structural assumption based on how ``OR{n}_XR`` header ranges
        relate to NaN boundaries in the cube; it is not directly confirmed
        by instrument documentation.

    Parameters
    ----------
    wavecalinfo : WaveCalInfo
    order_idx : int
        Index into ``wavecalinfo.data`` first axis.
    x_start, x_end : int
        Detector column range (inclusive) for this order, from
        ``wavecalinfo.xranges[order_idx]``.
    degree : int
        Polynomial degree.

    Returns
    -------
    ndarray, shape (degree + 1,)
        Polynomial coefficients in ``numpy.polynomial.polynomial``
        convention.
    """
    wav_array = wavecalinfo.data[order_idx, 0, :]
    valid = ~np.isnan(wav_array)

    if not valid.any():
        raise ValueError(
            f"wavecalinfo order index {order_idx} has no valid (non-NaN) values "
            "in data cube plane 0."
        )

    n_valid = int(valid.sum())
    # Array index j → detector column x_start + j
    cols = np.arange(n_valid, dtype=float) + x_start
    wavs = wav_array[valid]

    # Ensure degree doesn't exceed available points
    fit_degree = min(degree, n_valid - 1)
    if fit_degree < degree:
        warnings.warn(
            f"Order index {order_idx}: requested degree {degree} but only "
            f"{n_valid} valid pixels; reducing to degree {fit_degree}.",
            RuntimeWarning,
            stacklevel=3,
        )

    coeffs = np.polynomial.polynomial.polyfit(cols, wavs, fit_degree)
    return coeffs


def _build_order_rectification_map(
    geom: OrderGeometry,
    plate_scale_arcsec: float,
    slit_height_arcsec: float | None,
    n_spatial: int | None,
) -> RectificationMap:
    """Build a RectificationMap for one order.

    Output grid
    -----------
    - **Spectral axis**: one point per detector column in
      ``[x_start, x_end]``, wavelengths evaluated from ``wave_coeffs``.
    - **Spatial axis**: uniformly spaced points spanning the order height
      at the midpoint column, in arcseconds relative to the centerline
      (positive = top of slit).

    Source coordinates
    ------------------
    For each output (spatial_i, spectral_j) point:

    * ``col_nom = spectral_j`` (the column index in the output corresponds
      directly to a detector column).
    * ``row_ctr = eval_centerline(col_nom)``
    * ``row_offset_pix = spatial_arcsec[i] / plate_scale_arcsec``
    * ``tilt = eval_tilt(col_nom)``  (= 0 in current implementation)
    * ``src_col = col_nom + tilt * row_offset_pix``
    * ``src_row = row_ctr + row_offset_pix``
    """
    cols_out = np.arange(geom.x_start, geom.x_end + 1, dtype=float)
    wavs_out = np.polynomial.polynomial.polyval(cols_out, geom.wave_coeffs)

    # Spatial grid: span the order height at the midpoint column
    col_mid = 0.5 * (geom.x_start + geom.x_end)
    row_bot = float(geom.eval_bottom_edge(col_mid))
    row_top = float(geom.eval_top_edge(col_mid))
    order_height_pix = row_top - row_bot

    if slit_height_arcsec is not None:
        half_slit = 0.5 * slit_height_arcsec
    else:
        half_slit = 0.5 * order_height_pix * plate_scale_arcsec

    if n_spatial is None:
        n_spatial = max(2, int(round(2.0 * half_slit / plate_scale_arcsec)))

    arcsec_out = np.linspace(-half_slit, half_slit, n_spatial)

    n_spec = len(cols_out)
    n_spat = len(arcsec_out)

    # Build source coordinate grids (n_spatial, n_spectral)
    src_rows = np.empty((n_spat, n_spec), dtype=float)
    src_cols = np.empty((n_spat, n_spec), dtype=float)

    for j, col in enumerate(cols_out):
        row_ctr = float(geom.eval_centerline(col))
        tilt = float(geom.eval_tilt(col))  # zero-tilt approximation (see module docstring)
        row_offsets = arcsec_out / plate_scale_arcsec
        src_rows[:, j] = row_ctr + row_offsets
        src_cols[:, j] = col + tilt * row_offsets

    return RectificationMap(
        order=geom.order,
        output_wavelengths_um=wavs_out,
        output_spatial_arcsec=arcsec_out,
        src_rows=src_rows,
        src_cols=src_cols,
    )
