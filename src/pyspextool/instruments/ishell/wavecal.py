"""
iSHELL-specific geometry population and rectification support.

This module provides two distinct wavecal paths:

1. **Structural bootstrap** (:func:`build_geometry_from_wavecalinfo`):
   Fits polynomials to the pre-stored plane-0 reference wavelength arrays
   in the packaged ``*_wavecalinfo.fits`` files.  This is a compact
   polynomial re-encoding of values that were already stored by the
   IDL Spextool pipeline; it is not a re-measurement.

2. **Measured centerline wavelength solution** (:func:`build_geometry_from_arc_lines`):
   Uses the packaged ThAr line lists together with the data stored in
   plane 1 of ``WaveCalInfo.data`` (which appears to be the arc-lamp
   spectrum based on header units, but is not yet verified against the IDL
   source) to measure arc-line centroids via Gaussian profile fitting and
   derive a per-order centerline wavelength solution from those centroid
   positions.

   This derives wavelength coefficients from actual arc-line position
   measurements rather than from a polynomial re-fit of the pre-stored
   plane-0 reference array.  It does **not** implement full 2-D line
   tracing, tilt measurement, or the 2DXD global model.  Tilt coefficients
   remain the placeholder zero because tilt measurement requires spatial
   variation of line positions across the slit (a 2-D arc image), which is
   not stored in the packaged ``*_wavecalinfo.fits`` files.  This
   limitation is documented below and in
   ``docs/ishell_wavecal_design_note.md``.

Data cube plane semantics (partially confirmed)
-------------------------------------------------
Inspection of the packaged ``*_wavecalinfo.fits`` files together with the
FITS header keywords ``XUNITS`` (``"um"``) and ``YUNITS`` (``"DN / s"``)
provides partial evidence about the plane contents:

* **Plane 0** – wavelength grid in µm along the order centerline (confirmed
  by ``XUNITS="um"`` and value ranges consistent with J/H/K bands).
* **Plane 1** – appears to contain the ThAr arc-lamp spectrum in DN/s based
  on ``YUNITS="DN / s"`` and value ranges consistent with a real arc
  spectrum, but this interpretation has not been verified against the IDL
  source code.
* **Planes 2–3** – meanings are not confirmed and are not used by this code.
  Plane 2 may be an uncertainty array; plane 3 appears to carry integer
  flags (observed values 0, 1, 2) but the exact encoding is unverified.

Public API
----------
- :func:`fit_arc_line_centroids` – identify ThAr arc lines in the data
  stored in plane 1 of the wavecalinfo cube (interpreted as arc spectrum)
  and measure their centroids via Gaussian fitting.
- :func:`build_geometry_from_arc_lines` – populate an
  :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` with
  a centerline wavelength solution derived from measured arc-line centroid
  positions.
- :func:`build_geometry_from_wavecalinfo` – structural bootstrap: populate
  an :class:`~pyspextool.instruments.ishell.geometry.OrderGeometrySet` from
  the stored plane-0 reference arrays.
- :func:`build_rectification_maps` – create a
  :class:`~pyspextool.instruments.ishell.geometry.RectificationMap` for
  every order in an ``OrderGeometrySet``.

Measured centerline wavelength solution vs bootstrap
------------------------------------------------------
:func:`build_geometry_from_arc_lines` (measured centerline):

1. For each order in the line list, locates each expected arc line using
   the plane-0 wavelength grid to predict the pixel position.
2. Extracts a window of the plane-1 data (interpreted as arc spectrum)
   centred on the predicted position.
3. Fits a Gaussian profile to measure the precise pixel centroid.
4. Filters measurements by minimum SNR and centroid-within-window checks.
5. Fits a polynomial to the accepted ``(centroid_pixel, wavelength_um)``
   pairs for each order.

:func:`build_geometry_from_wavecalinfo` (bootstrap):

1. Extracts the stored plane-0 array (wavelengths in µm) for each order.
2. Fits a polynomial to ``(column, plane-0 value)`` pairs.

The measured approach derives coefficients from actual line-centroid
measurements rather than from a polynomial fit to pre-stored values.

Tilt model (provisional placeholder zero – 1-D data limitation)
----------------------------------------------------------------
``tilt_coeffs`` is set to ``[0.0]`` for every order in **both** paths.
This is a **provisional placeholder**, not a measured quantity.

Measuring spectral-line tilt requires knowing how a line's pixel centroid
shifts along the *spatial* (cross-dispersion) direction.  That information
is encoded in a 2-D arc image, which is not stored in the packaged
``*_wavecalinfo.fits`` files (only the 1-D centerline arc spectrum is
stored).  Tilt measurement will require real 2-D ThAr arc-lamp frames as
input; this step is deferred to a later PR.

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
1.  **Tilt measurement** – the IDL pipeline fits arc-line centroids at
    multiple spatial rows to measure spectral-line tilt as a function of
    column.  This requires 2-D ThAr arc frames that are not stored in the
    packaged calibration files.
2.  **Slit curvature / higher-order distortion** – curvature within each
    order is not modelled by the current zero-tilt placeholder.
3.  **Full 2DXD wavecal iteration** – interactive arc-line identification
    and iterative outlier rejection against a 2DXD global model are not
    implemented here.
4.  **Wavelength uncertainty propagation** – fit residuals and covariance
    are not stored in the geometry objects.
5.  **L/Lp/M modes** – out of scope per the problem statement.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from .geometry import OrderGeometry, OrderGeometrySet, RectificationMap

if TYPE_CHECKING:
    from .calibrations import FlatInfo, LineList, WaveCalInfo

__all__ = [
    "fit_arc_line_centroids",
    "build_geometry_from_arc_lines",
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

    **Structural bootstrap path.** Reads plane 0 of the ``WaveCalInfo``
    data cube (confirmed by FITS header ``XUNITS="um"`` to contain
    wavelengths in µm) and fits a polynomial to those values for each order.
    Edge polynomials come from the ``FlatInfo``.  The tilt is set to a
    **placeholder zero** (requires 2-D arc images; see module docstring) and
    the spatial calibration uses a simple linear plate-scale model.

    For a wavelength solution derived from measured arc-line centroids, use
    :func:`build_geometry_from_arc_lines` instead.

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


def fit_arc_line_centroids(
    wavecalinfo: "WaveCalInfo",
    line_list: "LineList",
    snr_min: float = 3.0,
    half_window_pix: int = 15,
) -> dict[int, list[tuple[float, float, float]]]:
    """Identify ThAr arc lines and measure their centroids from stored data.

    For each echelle order this function:

    1. Retrieves the wavelength grid (plane 0) and the plane-1 data
       (interpreted as the arc-lamp spectrum; see module docstring for
       caveats about plane semantics) from ``wavecalinfo.data``.
    2. For every line in ``line_list`` that belongs to the order, computes
       the expected pixel position from the wavelength grid.
    3. Extracts a window of the plane-1 data centred on that position.
    4. Fits a Gaussian profile to measure the precise pixel centroid.
    5. Keeps only measurements with SNR ≥ ``snr_min`` and centroids that
       fall strictly inside the extraction window.

    Parameters
    ----------
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Must have ``xranges`` populated and ``data`` with valid plane 0
        (wavelengths in µm) and plane 1 (interpreted as arc-lamp spectrum;
        not formally verified against the IDL source).
    line_list : :class:`~.calibrations.LineList`
        ThAr arc-line list for the same mode as ``wavecalinfo``.
    snr_min : float, optional
        Minimum signal-to-noise ratio for a centroid measurement to be
        accepted.  Default is 3.0.
    half_window_pix : int, optional
        Half-width (in pixels) of the extraction window used for Gaussian
        fitting.  Default is 15.

    Returns
    -------
    dict mapping ``order_number -> list of (centroid_pixel, wavelength_um, snr)``
        For each order that has at least one accepted measurement, the
        list contains one 3-tuple per accepted arc line: the measured
        centroid pixel position (float), the known vacuum wavelength in µm
        (float), and the SNR of the arc peak (float).
        Orders with no accepted measurements have empty lists.

    Raises
    ------
    ValueError
        If ``wavecalinfo.xranges`` is ``None``.

    Notes
    -----
    Only the order centerline is used (1-D data from the stored spectrum).
    Tilt (spatial variation of line position across the slit) cannot be
    measured from this 1-D data; see the module docstring for details.
    """
    if wavecalinfo.xranges is None:
        raise ValueError(
            "wavecalinfo.xranges is None; cannot determine column mapping.  "
            "Re-read with a version that parses OR{n}_XR headers."
        )

    results: dict[int, list[tuple[float, float, float]]] = {}

    for i, order_num in enumerate(wavecalinfo.orders):
        wav_arr = wavecalinfo.data[i, 0, :]
        arc_arr = wavecalinfo.data[i, 1, :]
        valid = ~np.isnan(wav_arr)

        if not valid.any():
            results[order_num] = []
            continue

        wav_valid = wav_arr[valid]
        arc_valid = arc_arr[valid]
        x_start = int(wavecalinfo.xranges[i, 0])
        cols = np.arange(int(valid.sum()), dtype=float) + x_start

        # Robust noise estimate (median absolute deviation)
        median_val = float(np.median(arc_valid))
        noise = float(np.median(np.abs(arc_valid - median_val))) * 1.4826
        if noise < 1e-10:
            noise = max(float(np.std(arc_valid)), 1e-10)

        order_lines = [e for e in line_list.entries if e.order == order_num]
        order_results: list[tuple[float, float, float]] = []

        for entry in order_lines:
            # Predict pixel from wavelength grid
            diff = np.abs(wav_valid - entry.wavelength_um)
            best_idx = int(np.argmin(diff))

            # Reject if predicted position is outside the valid wavelength range
            # (tolerance = 10 nm, i.e. 0.010 µm)
            if diff[best_idx] > 0.010:
                continue

            # Extract window around predicted position
            i0 = max(0, best_idx - half_window_pix)
            i1 = min(len(cols), best_idx + half_window_pix + 1)
            x_win = cols[i0:i1]
            y_win = arc_valid[i0:i1]

            if len(x_win) < 5:
                continue

            # Peak SNR check
            peak_val = float(y_win.max())
            snr = peak_val / noise if noise > 0 else 0.0
            if snr < snr_min:
                continue

            # Gaussian centroid fit
            centroid = _fit_gaussian_centroid(x_win, y_win)
            if centroid is None:
                continue

            # Accept only if centroid is strictly within the window
            if not (x_win[0] < centroid < x_win[-1]):
                continue

            order_results.append((float(centroid), float(entry.wavelength_um), float(snr)))

        # Remove blended lines: if two measured centroids are within
        # blend_threshold pixels of each other, keep the one with higher SNR
        # (or reject both if their pixels are indistinguishable).  This
        # prevents degenerate polynomial fits caused by two wavelengths
        # mapping to the same detector position.
        order_results = _deblend_centroids(order_results, blend_threshold=2.0)

        results[order_num] = order_results

    return results


def build_geometry_from_arc_lines(
    wavecalinfo: "WaveCalInfo",
    flatinfo: "FlatInfo",
    line_list: "LineList",
    dispersion_degree: int | None = None,
    snr_min: float = 3.0,
    min_lines_per_order: int = 4,
) -> OrderGeometrySet:
    """Populate an :class:`~.geometry.OrderGeometrySet` with a measured centerline wavelength solution.

    **Measured centerline wavelength solution.**  This function
    identifies ThAr arc lines in the plane-1 data of the stored
    ``WaveCalInfo`` (interpreted as an arc-lamp spectrum; see module
    docstring for caveats about plane semantics), measures their pixel
    centroids via Gaussian profile fitting, and derives a per-order
    centerline polynomial wavelength solution from those centroid
    measurements.

    This produces wavelength coefficients from arc-line position
    measurements rather than from refitting the pre-stored plane-0
    reference array (:func:`build_geometry_from_wavecalinfo`).  It
    operates only along the order centerline and does **not** measure
    spectral tilt, implement full 2-D line tracing, or replicate the
    full 1DXD/2DXD pipeline from legacy IDL Spextool.

    Tilt coefficients are still set to a **provisional placeholder zero**
    because tilt measurement requires spatial variation of line positions
    across the slit (i.e. a 2-D arc image), which is not available from
    the stored 1-D centerline data.  See the module docstring for a full
    discussion.

    Parameters
    ----------
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Stored calibration for the mode.  Must have ``xranges`` populated
        and ``data`` with valid plane 0 (wavelengths) and plane 1
        (interpreted as arc-lamp spectrum; see module docstring).
    flatinfo : :class:`~.calibrations.FlatInfo`
        Flat-field calibration for the same mode.  Must have
        ``edge_coeffs`` and ``xranges`` populated.
    line_list : :class:`~.calibrations.LineList`
        ThAr arc-line list for the same mode.
    dispersion_degree : int or None, optional
        Polynomial degree for the per-order centerline wavelength fit.
        Defaults to ``wavecalinfo.disp_degree`` when ``None``.
    snr_min : float, optional
        Minimum arc-line SNR for a centroid to be accepted.  Default 3.0.
    min_lines_per_order : int, optional
        Minimum number of accepted centroid measurements required to
        attempt a polynomial fit for an order.  If fewer lines are
        measured, that order falls back to the plane-0 bootstrap fit and
        a ``RuntimeWarning`` is emitted.  Default is 4.

    Returns
    -------
    :class:`~.geometry.OrderGeometrySet`
        Geometry set with ``wave_coeffs`` populated from measured arc-line
        centroids (or from the plane-0 bootstrap fallback for orders with
        too few accepted lines), ``tilt_coeffs`` as placeholder zero, and
        ``spatcal_coeffs`` as the linear plate-scale model.

    Raises
    ------
    ValueError
        If ``flatinfo`` and ``wavecalinfo`` refer to different modes, if
        ``flatinfo.edge_coeffs`` / ``wavecalinfo.xranges`` are not
        available, or if order lists do not match.
    """
    _validate_inputs(wavecalinfo, flatinfo)

    mode = wavecalinfo.mode
    degree = int(dispersion_degree) if dispersion_degree is not None else int(
        wavecalinfo.disp_degree
    )

    plate_scale = _get_plate_scale(flatinfo)

    # Measure arc-line centroids from stored arc spectrum
    centroids = fit_arc_line_centroids(
        wavecalinfo=wavecalinfo,
        line_list=line_list,
        snr_min=snr_min,
    )

    geometries = []

    for i, order_num in enumerate(wavecalinfo.orders):
        edge_c = flatinfo.edge_coeffs[i]  # shape (2, n_terms)
        x_start, x_end = flatinfo.xranges[i]

        order_centroids = centroids.get(order_num, [])

        if len(order_centroids) >= min_lines_per_order:
            # Fit polynomial to measured centroid positions
            meas_cols = np.array([c[0] for c in order_centroids], dtype=float)
            meas_wavs = np.array([c[1] for c in order_centroids], dtype=float)
            fit_degree = min(degree, len(order_centroids) - 1)
            if fit_degree < degree:
                warnings.warn(
                    f"Order {order_num}: requested degree {degree} but only "
                    f"{len(order_centroids)} arc lines accepted; "
                    f"reducing to degree {fit_degree}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            wave_coeffs = np.polynomial.polynomial.polyfit(
                meas_cols, meas_wavs, fit_degree
            )
        else:
            # Fallback: bootstrap from stored plane-0 reference array
            warnings.warn(
                f"Order {order_num}: only {len(order_centroids)} arc-line "
                f"centroid(s) accepted (minimum {min_lines_per_order}); "
                "falling back to plane-0 bootstrap for this order.",
                RuntimeWarning,
                stacklevel=2,
            )
            wave_coeffs = _derive_wave_coeffs(
                wavecalinfo=wavecalinfo,
                order_idx=i,
                x_start=int(x_start),
                x_end=int(x_end),
                degree=degree,
            )

        # Tilt: provisional placeholder zero
        # Cannot be measured from 1-D centerline spectra; requires 2-D arc images.
        tilt_coeffs = np.array([0.0])

        # Spatial calibration (linear)
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

    Plane 0 is confirmed (by FITS header ``XUNITS="um"``) to contain
    wavelengths in µm along the order centerline.  The column mapping
    ``array_index → x_start + array_index`` is based on how ``OR{n}_XR``
    header ranges map to the valid (non-NaN) portion of the data cube.

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


def _deblend_centroids(
    centroids: list[tuple[float, float, float]],
    blend_threshold: float = 2.0,
) -> list[tuple[float, float, float]]:
    """Remove blended line measurements.

    If two accepted centroid measurements fall within *blend_threshold*
    pixels of each other (e.g. an unresolved blend), both are rejected.
    This prevents degenerate polynomial fitting caused by two wavelength
    values mapping to the same detector position.

    Parameters
    ----------
    centroids : list of (centroid_pixel, wavelength_um, snr)
    blend_threshold : float
        Pixel separation below which two measurements are considered blended.

    Returns
    -------
    list of (centroid_pixel, wavelength_um, snr)
        Filtered list with blended pairs removed.
    """
    if len(centroids) < 2:
        return list(centroids)

    # Sort by centroid pixel
    sorted_c = sorted(centroids, key=lambda t: t[0])
    keep = [True] * len(sorted_c)

    for j in range(len(sorted_c) - 1):
        if not keep[j]:
            continue
        if abs(sorted_c[j + 1][0] - sorted_c[j][0]) < blend_threshold:
            # Reject both (neither centroid is reliable)
            keep[j] = False
            keep[j + 1] = False

    return [c for c, ok in zip(sorted_c, keep) if ok]


def _fit_gaussian_centroid(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> float | None:
    """Fit a Gaussian profile to (x, y) data and return the centroid.

    Uses ``scipy.optimize.curve_fit`` with a 4-parameter Gaussian model
    ``f(x) = amplitude * exp(-0.5 * ((x - mu) / sigma)**2) + background``.
    Falls back to a flux-weighted centroid if the Gaussian fit fails.

    Parameters
    ----------
    x : array_like, shape (n,)
        Pixel positions.
    y : array_like, shape (n,)
        Arc-spectrum values.

    Returns
    -------
    float or None
        Fitted centroid position in the same units as *x*, or ``None`` if
        even the weighted-centroid fallback is not meaningful (e.g. all
        flux ≤ 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    i_peak = int(np.argmax(y))
    x0 = x[i_peak]
    amplitude = float(y[i_peak])
    background = float(np.percentile(y, 20))

    # Estimate sigma from half-maximum points.
    # above_half spans the full FWHM; sigma = FWHM / 2.355.
    above_half = y > 0.5 * (amplitude - background) + background
    if above_half.sum() >= 2:
        sigma_init = max((x[above_half].max() - x[above_half].min()) / 2.355, 0.5)
    else:
        sigma_init = 1.0

    def _gaussian(xv, amp, mu, sig, bg):
        return amp * np.exp(-0.5 * ((xv - mu) / sig) ** 2) + bg

    try:
        bounds = (
            [0.0, float(x[0]), 0.3, -np.inf],
            [np.inf, float(x[-1]), float(x[-1] - x[0]), np.inf],
        )
        popt, _ = curve_fit(
            _gaussian, x, y,
            p0=[amplitude - background, x0, sigma_init, background],
            bounds=bounds,
            maxfev=2000,
        )
        return float(popt[1])
    except Exception:
        pass

    # Fallback: flux-weighted centroid using positive values only
    y_pos = np.maximum(y - background, 0.0)
    total = float(y_pos.sum())
    if total <= 0.0:
        return None
    return float(np.sum(x * y_pos) / total)


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
        tilt = float(geom.eval_tilt(col))  # provisional zero-tilt (see module docstring)
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
