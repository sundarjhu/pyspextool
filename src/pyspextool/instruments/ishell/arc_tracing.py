"""
2-D arc-line tracing scaffold for iSHELL.

This module defines the data structures and tracing functions that form the
primary arc-line input to the provisional 2DXD wavelength-mapping scaffold
(see :mod:`~pyspextool.instruments.ishell.wavecal_2dxd`).

Design intent
-------------
The *intended* production path is:

1. Real 2-D iSHELL H1 arc-lamp FITS frames (2048 × 2048 detector images).
2. Order-edge geometry from ``FlatInfo`` (packaged flat calibration).
3. :func:`trace_arc_lines_from_2d_image` – traces arc lines in each order
   by fitting Gaussian profiles along spatial rows of the 2-D image.
4. Returns :class:`ArcTraceResult` containing :class:`TracedArcLine` objects
   with detector-space positions.
5. :func:`~.wavecal_2dxd.fit_provisional_2dxd` matches those positions to
   reference wavelengths and fits a global 2-D polynomial.

**Current status (Phase 0 scaffold):** Real 2-D raw arc FITS files are not
yet available in the test data directory
(``data/testdata/ishell_h1_calibrations/raw/``).  The function
:func:`trace_arc_lines_from_2d_image` is therefore **declared but not
fully implemented** (it raises :class:`NotImplementedError` until raw frames
are available).

In lieu of real 2-D arc data, :func:`build_arc_trace_result_from_wavecalinfo`
uses the 1-D centerline arc spectrum stored in **plane 1** of the packaged
``*_wavecalinfo.fits`` data cube as a proxy for the per-order arc signal.
This is **explicitly a provisional substitute**, not the intended long-term
path.  The FITS header keyword ``YUNITS = "DN / s"`` is consistent with an
arc-lamp flux spectrum, but the interpretation of plane 1 has not been
formally verified against the IDL source code.

Data structures
---------------
- :class:`TracedArcLine` – one traced arc line in detector space, with the
  measured centroid column used as the **representative detector coordinate**
  for provisional wavelength matching.
- :class:`ArcTraceResult` – ordered collection of :class:`TracedArcLine`
  objects for one iSHELL mode.

Public API
----------
- :class:`TracedArcLine`
- :class:`ArcTraceResult`
- :func:`trace_arc_lines_from_1d_spectrum` – trace arc lines from a 1-D arc
  spectrum array (used by the packaged-data proxy path).
- :func:`build_arc_trace_result_from_wavecalinfo` – convenience wrapper that
  builds :class:`ArcTraceResult` from the packaged ``WaveCalInfo`` plane-1
  data; the primary usable function until real 2-D frames are available.
- :func:`trace_arc_lines_from_2d_image` – **stub** for future 2-D arc image
  tracing (raises :class:`NotImplementedError`).

See ``docs/ishell_2dxd_algorithm_note.md`` for the full design context and
what remains to be implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from .calibrations import FlatInfo, LineList, LineListEntry, WaveCalInfo

__all__ = [
    "TracedArcLine",
    "ArcTraceResult",
    "trace_arc_lines_from_1d_spectrum",
    "build_arc_trace_result_from_wavecalinfo",
    "trace_arc_lines_from_2d_image",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TracedArcLine:
    """One traced arc line in detector space.

    Represents a single ThAr (or Ar/Th) arc-lamp emission line whose
    position has been measured on the detector via Gaussian profile fitting.

    Parameters
    ----------
    order : int
        Echelle order number.
    wavelength_um : float
        Known vacuum wavelength in µm from the reference line list.
    species : str
        Atomic/ionic species label from the line list (e.g. ``"Th I"``).
    seed_col : float
        **Predicted** detector column used to seed the centroid fit.
        Derived from the stored plane-0 wavelength grid by linear
        interpolation: ``seed_col = interp(wavelength_um, wave_grid, cols)``.
    centroid_col : float
        **Measured** sub-pixel centroid column from the Gaussian fit.  This
        is the **representative detector coordinate** used for provisional
        wavelength matching in the 2DXD polynomial fit.  The physical meaning
        is: "at what column does this line's peak fall on the detector,
        measured along the order centerline?"
    centerline_row : float
        Order centerline row at *seed_col*.  Evaluated from the flat-field
        edge polynomials: ``centerline_row = (bottom_edge + top_edge) / 2``
        at ``col = seed_col``.  Provided for downstream 2-D tilt modelling
        (not used in the current provisional 1-D fit).
    snr : float
        Ratio of the arc-line peak flux to the local noise estimate (median
        absolute deviation × 1.4826).
    fit_residual_pix : float
        Absolute difference between *seed_col* and *centroid_col* in pixels.
        Large values indicate a mismatch between the stored wavelength grid
        and the actual arc-line position.

    Notes
    -----
    The distinction between *seed_col* and *centroid_col* is important:

    * *seed_col* comes from the stored reference wavelength grid (plane 0 of
      the packaged ``WaveCalInfo`` cube) and is used only to locate the
      search window.
    * *centroid_col* is the actual Gaussian-fit measurement and is the
      quantity used for the provisional 2DXD fit.

    In the intended 2-D tracing path, *seed_col* would instead come from a
    rough wavelength model applied to a raw arc-lamp frame, and the Gaussian
    fit would be performed across multiple spatial rows to measure both
    the centroid column and the line tilt.  The current implementation only
    uses the order-centerline profile (1-D), so tilt remains uncharacterised.
    """

    order: int
    wavelength_um: float
    species: str
    seed_col: float
    centroid_col: float
    centerline_row: float
    snr: float
    fit_residual_pix: float


@dataclass
class ArcTraceResult:
    """Collection of traced arc lines for one iSHELL mode.

    This is the primary output of the arc-tracing step and the primary input
    to :func:`~.wavecal_2dxd.fit_provisional_2dxd`.  It aggregates all
    :class:`TracedArcLine` objects for one mode into a single container.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"H1"``).
    lines : list of :class:`TracedArcLine`
        All accepted traced lines, in the order they were found (by order
        number, then by column within each order).
    source : str
        Description of the data source used to produce these traces.  Should
        be one of:

        * ``"wavecalinfo_plane1"`` – 1-D centerline arc spectrum from plane 1
          of the packaged ``*_wavecalinfo.fits`` file (current default).
        * ``"raw_2d_arc_image"`` – real 2-D arc-lamp frame (intended future
          path; requires raw FITS data in
          ``data/testdata/ishell_h1_calibrations/raw/``).
    per_order_counts : dict mapping ``order_number → int``
        Number of accepted traced lines per order.  Set by the tracing
        functions; includes orders with zero accepted lines.
    """

    mode: str
    lines: list[TracedArcLine] = field(default_factory=list)
    source: str = "wavecalinfo_plane1"
    per_order_counts: dict = field(default_factory=dict)

    @property
    def n_lines(self) -> int:
        """Total number of accepted traced lines."""
        return len(self.lines)

    @property
    def orders(self) -> list[int]:
        """Sorted list of unique order numbers that have at least one line."""
        return sorted(set(ln.order for ln in self.lines))

    @property
    def n_orders_with_lines(self) -> int:
        """Number of orders with at least one accepted traced line."""
        return len(self.orders)

    def get_order_lines(self, order: int) -> list[TracedArcLine]:
        """Return all traced lines for a given order number.

        Parameters
        ----------
        order : int
            Echelle order number.

        Returns
        -------
        list of :class:`TracedArcLine`
            May be empty if no lines were traced for this order.
        """
        return [ln for ln in self.lines if ln.order == order]

    def to_flat_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return flat arrays of (centroid_col, order, wavelength_um, snr).

        Convenience method for passing data directly into the 2DXD polynomial
        fitting routine.

        Returns
        -------
        centroid_cols : ndarray, shape (n_lines,)
        orders : ndarray, shape (n_lines,)
        wavelengths_um : ndarray, shape (n_lines,)
        snr_values : ndarray, shape (n_lines,)
        """
        if not self.lines:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty, empty
        centroid_cols = np.array([ln.centroid_col for ln in self.lines], dtype=float)
        orders = np.array([ln.order for ln in self.lines], dtype=float)
        wavelengths = np.array([ln.wavelength_um for ln in self.lines], dtype=float)
        snrs = np.array([ln.snr for ln in self.lines], dtype=float)
        return centroid_cols, orders, wavelengths, snrs


# ---------------------------------------------------------------------------
# Public tracing functions
# ---------------------------------------------------------------------------


def trace_arc_lines_from_1d_spectrum(
    arc_spectrum_1d: npt.ArrayLike,
    wavelength_grid_1d: npt.ArrayLike,
    x_start: int,
    line_list_entries: "list[LineListEntry]",
    centerline_row: float,
    order: int,
    snr_min: float = 3.0,
    half_window_pix: int = 15,
    blend_threshold_pix: float = 2.0,
) -> list[TracedArcLine]:
    """Trace arc lines in one echelle order from a 1-D arc spectrum.

    This function performs Gaussian centroid fitting of arc-lamp emission
    lines in a 1-D spectral profile along the order centerline.  It is the
    workhorse of :func:`build_arc_trace_result_from_wavecalinfo`.

    **Data source note:** This function operates on a 1-D array extracted
    from the order centerline.  In the proxy path (current default), that
    array is plane 1 of the packaged ``*_wavecalinfo.fits`` data cube,
    interpreted as the ThAr arc-lamp spectrum.  In the intended future path
    it would be a row-collapsed or centerline-extracted profile from a real
    2-D arc-lamp frame.  The tracing algorithm itself is the same in both
    cases.

    **Representative coordinate:** The *centroid_col* field of each returned
    :class:`TracedArcLine` is the sub-pixel Gaussian centroid along the
    detector column axis.  This is the representative detector coordinate
    used for provisional wavelength matching.  The *seed_col* is the
    prediction from the stored wavelength grid and is only used to locate the
    fitting window.

    Parameters
    ----------
    arc_spectrum_1d : array_like, shape (n_valid,)
        Arc-lamp flux values along the order centerline.
    wavelength_grid_1d : array_like, shape (n_valid,)
        Wavelength values (µm) corresponding to each pixel of
        *arc_spectrum_1d*.  Used to locate the predicted position of each
        reference line.
    x_start : int
        Detector column index of the first element of *arc_spectrum_1d*.
        All column values are computed as ``x_start + array_index``.
    line_list_entries : list of :class:`~.calibrations.LineListEntry`
        Reference arc lines for this order only.
    centerline_row : float
        Order centerline row at the midpoint column (from ``FlatInfo`` edge
        polynomials).  Stored in each :class:`TracedArcLine` for downstream
        use; not used in the fitting itself.
    order : int
        Echelle order number (stored in results for bookkeeping).
    snr_min : float, optional
        Minimum arc-line peak SNR for a measurement to be accepted.
        Default 3.0.
    half_window_pix : int, optional
        Half-width (pixels) of the Gaussian fitting window.  Default 15.
    blend_threshold_pix : float, optional
        Centroid separation below which two lines are considered blended and
        both are rejected.  Default 2.0 pixels.

    Returns
    -------
    list of :class:`TracedArcLine`
        Accepted traced lines for this order, sorted by *centroid_col*.
        May be empty if no lines pass the SNR and centroid-quality cuts.
    """
    arc = np.asarray(arc_spectrum_1d, dtype=float)
    wav = np.asarray(wavelength_grid_1d, dtype=float)

    if len(arc) == 0 or len(wav) == 0:
        return []

    n = len(arc)
    cols = np.arange(n, dtype=float) + x_start

    # Robust noise estimate
    median_val = float(np.median(arc))
    noise = float(np.median(np.abs(arc - median_val))) * 1.4826
    if noise < 1e-10:
        noise = max(float(np.std(arc)), 1e-10)

    raw_results: list[tuple[float, float, float, float, str]] = []
    # (seed_col, centroid_col, wavelength_um, snr, species)

    for entry in line_list_entries:
        # Predict pixel from the stored wavelength grid
        diff = np.abs(wav - entry.wavelength_um)
        best_idx = int(np.argmin(diff))

        # Reject if the nearest grid point is > 10 nm away
        if diff[best_idx] > 0.010:
            continue

        seed_col = float(cols[best_idx])

        # Extract fitting window
        i0 = max(0, best_idx - half_window_pix)
        i1 = min(n, best_idx + half_window_pix + 1)
        x_win = cols[i0:i1]
        y_win = arc[i0:i1]

        if len(x_win) < 5:
            continue

        # SNR check on peak value
        peak_val = float(y_win.max())
        snr = peak_val / noise
        if snr < snr_min:
            continue

        # Gaussian centroid fit
        centroid = _fit_gaussian_centroid(x_win, y_win)
        if centroid is None:
            continue

        # Accept only if centroid is strictly within the window
        if not (x_win[0] < centroid < x_win[-1]):
            continue

        raw_results.append(
            (seed_col, centroid, entry.wavelength_um, snr, entry.species)
        )

    # Deblend: reject both members of any pair separated by < blend_threshold
    raw_results = _deblend(raw_results, blend_threshold_pix)

    # Build TracedArcLine objects
    traced: list[TracedArcLine] = []
    for seed_col, centroid_col, wavelength_um, snr, species in raw_results:
        traced.append(
            TracedArcLine(
                order=order,
                wavelength_um=wavelength_um,
                species=species,
                seed_col=seed_col,
                centroid_col=centroid_col,
                centerline_row=centerline_row,
                snr=snr,
                fit_residual_pix=abs(centroid_col - seed_col),
            )
        )

    return sorted(traced, key=lambda t: t.centroid_col)


def build_arc_trace_result_from_wavecalinfo(
    wavecalinfo: "WaveCalInfo",
    line_list: "LineList",
    flatinfo: Optional["FlatInfo"] = None,
    snr_min: float = 3.0,
    half_window_pix: int = 15,
) -> ArcTraceResult:
    """Build an :class:`ArcTraceResult` from the packaged WaveCalInfo data.

    **Data source:** This function uses **plane 1** of the packaged
    ``*_wavecalinfo.fits`` data cube as a 1-D proxy for the arc-lamp
    spectrum.  The FITS header keyword ``YUNITS = "DN / s"`` is consistent
    with an arc-lamp flux spectrum; the exact content of plane 1 has not been
    formally verified against the IDL source code.

    **This is a provisional substitute** for the intended path that starts
    from real 2-D raw arc-lamp FITS frames.  When raw frames become available
    in ``data/testdata/ishell_h1_calibrations/raw/``, use
    :func:`trace_arc_lines_from_2d_image` instead.

    The **representative detector coordinate** for each traced line is the
    Gaussian-fit centroid column (*centroid_col*) measured along the order
    centerline.  The seed column is predicted from the stored plane-0
    wavelength grid.

    Parameters
    ----------
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Must have ``xranges`` populated and ``data`` with valid plane 0
        (wavelengths in µm) and plane 1 (1-D arc spectrum proxy).
    line_list : :class:`~.calibrations.LineList`
        ThAr reference line list for the same mode.
    flatinfo : :class:`~.calibrations.FlatInfo` or None, optional
        If provided, used to evaluate the order centerline row at the
        mid-order column.  When ``None``, *centerline_row* is set to 0.0
        for all lines (a harmless placeholder until tilt modelling is
        implemented).
    snr_min : float, optional
        Minimum arc-line SNR for acceptance.  Default 3.0.
    half_window_pix : int, optional
        Half-width of the Gaussian fitting window.  Default 15.

    Returns
    -------
    :class:`ArcTraceResult`
        Contains all accepted :class:`TracedArcLine` objects for all orders,
        plus per-order counts.

    Raises
    ------
    ValueError
        If ``wavecalinfo.xranges`` is ``None``.
    """
    if wavecalinfo.xranges is None:
        raise ValueError(
            "wavecalinfo.xranges is None; cannot determine per-order column "
            "mapping.  Re-read with a version that parses OR{n}_XR headers."
        )

    all_lines: list[TracedArcLine] = []
    per_order_counts: dict[int, int] = {}

    for i, order_num in enumerate(wavecalinfo.orders):
        wav_arr = wavecalinfo.data[i, 0, :]
        arc_arr = wavecalinfo.data[i, 1, :]
        valid = ~np.isnan(wav_arr)

        if not valid.any():
            per_order_counts[order_num] = 0
            continue

        wav_valid = wav_arr[valid]
        arc_valid = arc_arr[valid]
        x_start = int(wavecalinfo.xranges[i, 0])

        # Evaluate centerline row at the mid-order column
        if flatinfo is not None:
            x_end = int(wavecalinfo.xranges[i, 1])
            col_mid = 0.5 * (x_start + x_end)
            # Find this order's index in flatinfo
            try:
                fi_idx = flatinfo.orders.index(order_num)
                edge_c = flatinfo.edge_coeffs[fi_idx]  # shape (2, n_terms)
                row_bot = float(
                    np.polynomial.polynomial.polyval(col_mid, edge_c[0])
                )
                row_top = float(
                    np.polynomial.polynomial.polyval(col_mid, edge_c[1])
                )
                centerline_row = 0.5 * (row_bot + row_top)
            except (ValueError, IndexError, TypeError):
                centerline_row = 0.0
        else:
            centerline_row = 0.0

        order_entries = [e for e in line_list.entries if e.order == order_num]

        traced = trace_arc_lines_from_1d_spectrum(
            arc_spectrum_1d=arc_valid,
            wavelength_grid_1d=wav_valid,
            x_start=x_start,
            line_list_entries=order_entries,
            centerline_row=centerline_row,
            order=order_num,
            snr_min=snr_min,
            half_window_pix=half_window_pix,
        )

        per_order_counts[order_num] = len(traced)
        all_lines.extend(traced)

    return ArcTraceResult(
        mode=wavecalinfo.mode,
        lines=all_lines,
        source="wavecalinfo_plane1",
        per_order_counts=per_order_counts,
    )


def trace_arc_lines_from_2d_image(
    arc_image: npt.ArrayLike,
    flatinfo: "FlatInfo",
    line_list: "LineList",
    wavecalinfo_meta: "WaveCalInfo",
    snr_min: float = 3.0,
    half_window_pix: int = 15,
) -> ArcTraceResult:
    """Trace arc lines from a real 2-D arc-lamp detector image.

    .. note::
       **Not yet implemented (stub).**  This function is declared to
       establish the intended interface for the production 2-D arc-tracing
       path.  It will raise :class:`NotImplementedError` until real 2-D
       raw arc-lamp FITS frames are available in
       ``data/testdata/ishell_h1_calibrations/raw/`` and a proper 2-D
       tracing algorithm is implemented.

    In the intended implementation, this function would:

    1. For each echelle order (using edge polynomials from *flatinfo*),
       extract a straight-slit sub-image of the 2-D arc frame.
    2. Collapse or profile-fit along the spatial axis to locate line
       positions at multiple rows.
    3. Fit Gaussian profiles at each row to measure the centroid column as
       a function of row (enabling tilt measurement).
    4. Return :class:`TracedArcLine` objects with both centroid and tilt
       information.

    The *centroid_col* of each returned :class:`TracedArcLine` would be the
    centroid measured at the order centerline row.  The *centerline_row*
    field would be set from the flat-field edge polynomials.

    Parameters
    ----------
    arc_image : array_like, shape (2048, 2048)
        Raw 2-D arc-lamp detector image, already dark-subtracted and
        linearity-corrected.
    flatinfo : :class:`~.calibrations.FlatInfo`
        Order geometry from the flat-field calibration.
    line_list : :class:`~.calibrations.LineList`
        ThAr reference line list for the mode.
    wavecalinfo_meta : :class:`~.calibrations.WaveCalInfo`
        Provides the order list, ``xranges``, and a rough wavelength grid
        (plane 0) for seeding line positions.  Only metadata is used here;
        the arc-spectrum data (plane 1) is **not** used by this function –
        the arc signal comes from *arc_image* instead.
    snr_min : float, optional
        Minimum arc-line SNR.  Default 3.0.
    half_window_pix : int, optional
        Half-width of the spatial fitting window.  Default 15.

    Returns
    -------
    :class:`ArcTraceResult`
        Source will be ``"raw_2d_arc_image"``.

    Raises
    ------
    NotImplementedError
        Always – until real 2-D tracing is implemented.
    """
    raise NotImplementedError(
        "trace_arc_lines_from_2d_image() is not yet implemented.  "
        "Real 2-D raw arc-lamp FITS frames are required; place them in "
        "data/testdata/ishell_h1_calibrations/raw/ and implement the "
        "2-D tracing algorithm in this function.\n\n"
        "For the current provisional path, use "
        "build_arc_trace_result_from_wavecalinfo() instead."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_gaussian_centroid(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> float | None:
    """Fit a Gaussian profile and return the centroid.

    Uses ``scipy.optimize.curve_fit`` with a 4-parameter model::

        f(x) = amplitude * exp(-0.5 * ((x - mu) / sigma)**2) + background

    Falls back to flux-weighted centroid if the Gaussian fit fails.

    Parameters
    ----------
    x, y : array_like
        Pixel positions and arc-flux values.

    Returns
    -------
    float or None
        Centroid position in the same units as *x*, or ``None`` if even
        the flux-weighted fallback is not meaningful (all flux ≤ 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    i_peak = int(np.argmax(y))
    x0 = x[i_peak]
    amplitude = float(y[i_peak])
    background = float(np.percentile(y, 20))

    above_half = y > 0.5 * (amplitude - background) + background
    if above_half.sum() >= 2:
        sigma_init = max(
            (x[above_half].max() - x[above_half].min()) / 2.355, 0.5
        )
    else:
        sigma_init = 1.0

    def _gauss(xv, amp, mu, sig, bg):
        return amp * np.exp(-0.5 * ((xv - mu) / sig) ** 2) + bg

    try:
        bounds = (
            [0.0, float(x[0]), 0.3, -np.inf],
            [np.inf, float(x[-1]), float(x[-1] - x[0]), np.inf],
        )
        popt, _ = curve_fit(
            _gauss,
            x,
            y,
            p0=[amplitude - background, x0, sigma_init, background],
            bounds=bounds,
            maxfev=2000,
        )
        return float(popt[1])
    except Exception:
        pass

    # Flux-weighted centroid fallback
    y_pos = np.maximum(y - background, 0.0)
    total = float(y_pos.sum())
    if total <= 0.0:
        return None
    return float(np.sum(x * y_pos) / total)


def _deblend(
    results: list[tuple[float, float, float, float, str]],
    blend_threshold: float,
) -> list[tuple[float, float, float, float, str]]:
    """Remove blended pairs from raw centroid results.

    If two results have centroids within *blend_threshold* pixels of each
    other, both are rejected (neither centroid is reliable for wavelength
    fitting).

    Parameters
    ----------
    results : list of (seed_col, centroid_col, wavelength_um, snr, species)
    blend_threshold : float
        Pixel separation below which two lines are considered blended.

    Returns
    -------
    list with blended pairs removed
    """
    if len(results) < 2:
        return list(results)

    sorted_r = sorted(results, key=lambda t: t[1])  # sort by centroid_col
    keep = [True] * len(sorted_r)

    for j in range(len(sorted_r) - 1):
        if not keep[j]:
            continue
        if abs(sorted_r[j + 1][1] - sorted_r[j][1]) < blend_threshold:
            keep[j] = False
            keep[j + 1] = False

    return [r for r, ok in zip(sorted_r, keep) if ok]
