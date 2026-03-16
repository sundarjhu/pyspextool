"""
Provisional 2DXD wavelength-mapping scaffold for iSHELL.

This module fits the first bridge between traced arc-line positions in
detector space and the global 2-D echelle polynomial model ("2DXD") used by
the legacy IDL Spextool pipeline.

Corrected data flow (this module)
----------------------------------
The primary input to this module is an :class:`~.arc_tracing.ArcTraceResult`
produced by :mod:`~pyspextool.instruments.ishell.arc_tracing`.  The intended
path is::

    arc-lamp detector data
        ↓ arc_tracing.build_arc_trace_result_from_wavecalinfo()  [proxy]
        ↓    or arc_tracing.trace_arc_lines_from_2d_image()      [future]
    ArcTraceResult  (list of TracedArcLine objects with detector positions)
        ↓ collect_centroid_data_from_arc_traces()
    ArcLineCentroidData  (flat (x, m, λ) arrays)
        ↓ fit_provisional_2dxd()
    ProvisionalWaveCal2DXD  (fitted 2-D polynomial + per-order 1-D coeffs)

**Data source note (Phase 0 proxy):** Until real 2-D raw arc-lamp FITS
frames are available in ``data/testdata/ishell_h1_calibrations/raw/``, the
:func:`~.arc_tracing.build_arc_trace_result_from_wavecalinfo` function uses
**plane 1** of the packaged ``*_wavecalinfo.fits`` data cube as a 1-D
centerline arc-spectrum proxy.  This is explicitly a provisional substitute;
the FITS header keyword ``YUNITS = "DN / s"`` is consistent with arc-lamp
flux, but the exact content of plane 1 has not been formally verified.  The
tracing algorithm (Gaussian centroid fitting) is the same as the production
path; only the data source differs.

The stored IDL 2DXD polynomial (``P2W_C*`` FITS keywords) is kept as
**reference / comparison metadata** and is demoted from the primary fit
engine.  See :class:`TwoDXDCoefficients` and
:func:`read_stored_2dxd_coeffs`.

What this module implements
---------------------------
1.  **Reading stored IDL 2DXD coefficients** (reference role).
    :func:`read_stored_2dxd_coeffs` reads ``P2W_C*`` from the FITS header
    into a :class:`TwoDXDCoefficients` dataclass.  The stored polynomial is
    attached to the result for downstream comparison; the evaluation
    convention is unverified.

2.  **Assembling flat centroid arrays from arc traces.**
    :func:`collect_centroid_data_from_arc_traces` converts an
    :class:`~.arc_tracing.ArcTraceResult` into an
    :class:`ArcLineCentroidData` array of ``(centroid_col, order, λ_ref)``
    triples.  The *centroid_col* is the Gaussian-fit column from the
    :class:`~.arc_tracing.TracedArcLine` – the representative detector
    coordinate for provisional λ matching.

3.  **Fitting a provisional 2DXD polynomial.**
    :func:`fit_provisional_2dxd` accepts an :class:`~.arc_tracing.ArcTraceResult`
    and fits::

        λ(x, m) = Σ_{i=0}^{Nd} Σ_{j=0}^{No} A_{ij}
                  * ((2*x / (n_px − 1)) − 1)^i
                  * (home_order / m)^j

    where *x* is the detector column (*centroid_col* of each traced line),
    *m* is the echelle order number, and *home_order* / *Nd* / *No* come
    from the stored FITS header metadata.

4.  **Auxiliary path (legacy / comparison).**
    :func:`collect_centroid_data` is retained as an auxiliary function that
    calls :func:`~.wavecal.fit_arc_line_centroids` directly on the packaged
    WaveCalInfo plane-1 data without going through the arc_tracing layer.
    **This is not the primary path** and is kept only for backward
    compatibility and comparison.

What this module does NOT implement
------------------------------------
* Spectral-line tilt measurement (requires 2-D spatial variation across slit).
* Slit curvature or higher-order distortion.
* Final rectification-index generation.
* Iterative arc-line identification or sigma-clipping.
* Full science-quality 2DXD calibration.

See ``docs/ishell_2dxd_algorithm_note.md`` for a full developer description,
a comparison with the legacy IDL pipeline, and what remains unimplemented.

Public API
----------
- :class:`TwoDXDCoefficients` – stored IDL 2DXD coefficient container
  (reference role).
- :class:`ArcLineCentroidData` – flat (x, m, λ) arrays from traced lines.
- :class:`ProvisionalWaveCal2DXD` – primary result of a provisional 2DXD fit.
- :func:`read_stored_2dxd_coeffs` – reads P2W_C* keywords.
- :func:`collect_centroid_data_from_arc_traces` – PRIMARY: converts
  ArcTraceResult to ArcLineCentroidData.
- :func:`fit_provisional_2dxd` – main entry point; accepts ArcTraceResult.
- :func:`collect_centroid_data` – AUXILIARY: legacy path via
  fit_arc_line_centroids().
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .arc_tracing import ArcTraceResult
    from .calibrations import FlatInfo, LineList, WaveCalInfo

__all__ = [
    "TwoDXDCoefficients",
    "ArcLineCentroidData",
    "ProvisionalWaveCal2DXD",
    "read_stored_2dxd_coeffs",
    "collect_centroid_data_from_arc_traces",
    "fit_provisional_2dxd",
    "collect_centroid_data",
]

# iSHELL H2RG detector column count – used for x normalization.
_DETECTOR_NCOLS = 2048


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TwoDXDCoefficients:
    """Pre-stored IDL 2DXD polynomial coefficients from ``*_wavecalinfo.fits``.

    The IDL Spextool pipeline stores a fully-solved 2DXD polynomial in the
    FITS header as ``P2W_C00`` … ``P2W_C{n}`` (where *n* =
    ``(disp_degree+1) * (order_degree+1) − 1``).  This dataclass stores
    those raw values together with the degree and reference-order metadata
    needed to reconstruct the polynomial.

    **Role in this module:** These are **reference / comparison metadata**,
    not the primary source of the provisional fit.  The fit is derived from
    traced arc-line positions via :func:`fit_provisional_2dxd`.

    .. warning::
       The **exact evaluation convention** of the stored polynomial
       (coordinate normalization applied to detector columns and order numbers)
       has **not** been verified against the IDL source code.
       :meth:`eval_provisional` implements one plausible normalization and is
       offered for comparison only.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"H1"``).
    coeffs_flat : ndarray, shape ``((disp_degree+1) * (order_degree+1),)``
        Raw coefficient values read from ``P2W_C*`` FITS keywords, in the
        order ``C00, C01, …, C{n}``.  The indexing convention is::

            k = j * (disp_degree + 1) + i

        where *i* is the dispersion (column) power index and *j* is the
        cross-dispersion (order) power index.  This convention has been
        read directly from the FITS header and has not been verified.
    disp_degree : int
        Polynomial degree along the dispersion axis (``DISPDEG`` keyword).
    order_degree : int
        Polynomial degree along the order axis (``ORDRDEG`` keyword).
    home_order : int
        Reference echelle order used for order normalization (``HOMEORDR``).
    n_pixels : int
        Number of detector columns used for column normalization.  Defaults
        to :data:`_DETECTOR_NCOLS` (2048).
    """

    mode: str
    coeffs_flat: np.ndarray
    disp_degree: int
    order_degree: int
    home_order: int
    n_pixels: int = _DETECTOR_NCOLS

    @property
    def n_disp_terms(self) -> int:
        """Number of polynomial terms along the dispersion axis."""
        return self.disp_degree + 1

    @property
    def n_order_terms(self) -> int:
        """Number of polynomial terms along the order axis."""
        return self.order_degree + 1

    @property
    def n_coeffs(self) -> int:
        """Total number of polynomial coefficients."""
        return self.n_disp_terms * self.n_order_terms

    @property
    def coeffs_matrix(self) -> np.ndarray:
        """Coefficient matrix, shape ``(n_disp_terms, n_order_terms)``.

        Element ``[i, j]`` multiplies ``x_norm^i * order_norm^j`` where
        the normalization convention is described in the class docstring.

        The flat coefficient array uses the indexing convention
        ``k = j * (disp_degree + 1) + i`` (j varies slowly, i varies fast),
        so the correct reshape is ``(n_order_terms, n_disp_terms)`` followed
        by a transpose to produce shape ``(n_disp_terms, n_order_terms)``.
        """
        return self.coeffs_flat.reshape(
            self.n_order_terms, self.n_disp_terms
        ).T

    def eval_provisional(
        self,
        x: npt.ArrayLike,
        m: npt.ArrayLike,
    ) -> np.ndarray:
        """Evaluate the stored polynomial with a provisional normalization.

        .. warning::
           **Provisional / comparison only.**  Whether the stored ``P2W_C*``
           coefficients use the normalization applied here has not been
           verified against IDL source code.

        The polynomial evaluated is::

            λ(x, m) = Σ_{i,j} C[i,j]
                      * ((2*x / (n_px − 1)) − 1)^i
                      * (home_order / m)^j

        Parameters
        ----------
        x : array_like
            Detector column positions.
        m : array_like
            Echelle order numbers.

        Returns
        -------
        ndarray
            Provisional wavelength estimates in µm.
        """
        x = np.asarray(x, dtype=float)
        m = np.asarray(m, dtype=float)
        x_norm = (2.0 * x / (self.n_pixels - 1.0)) - 1.0
        inv_order = self.home_order / m
        C = self.coeffs_matrix  # shape (n_disp, n_order)
        result = np.zeros_like(x_norm)
        for j in range(self.n_order_terms):
            for i in range(self.n_disp_terms):
                result = result + C[i, j] * (x_norm ** i) * (inv_order ** j)
        return result


@dataclass
class ArcLineCentroidData:
    """Flat collection of arc-line centroid measurements across all orders.

    Aggregates ``(centroid_col, order, λ_reference)`` triples from
    :class:`~.arc_tracing.TracedArcLine` objects (primary path) or from
    :func:`collect_centroid_data` (auxiliary / legacy path).

    The *centroid_col* values are the **representative detector coordinates**
    for provisional λ matching: the Gaussian-fit column positions measured
    along the order centerline by the arc-tracing step.

    Parameters
    ----------
    mode : str
        iSHELL mode name.
    columns : ndarray, shape ``(n_points,)``
        Measured centroid column positions (from :attr:`~.arc_tracing.TracedArcLine.centroid_col`).
    orders : ndarray, shape ``(n_points,)``
        Echelle order number for each measurement.
    wavelengths_um : ndarray, shape ``(n_points,)``
        Known vacuum wavelengths in µm from the reference line list.
    snr_values : ndarray, shape ``(n_points,)``
        Peak SNR of each arc-line centroid.
    per_order_counts : dict mapping ``order_number → count``
        Number of accepted measurements per order.
    source : str
        Description of the data source (e.g. ``"arc_tracing"`` or
        ``"wavecal_fit_arc_line_centroids"``).
    """

    mode: str
    columns: np.ndarray
    orders: np.ndarray
    wavelengths_um: np.ndarray
    snr_values: np.ndarray
    per_order_counts: dict = field(default_factory=dict)
    source: str = "arc_tracing"

    @property
    def n_points(self) -> int:
        """Total number of accepted centroid measurements."""
        return int(len(self.columns))

    @property
    def n_orders_with_data(self) -> int:
        """Number of echelle orders that have at least one measurement."""
        return int(sum(1 for v in self.per_order_counts.values() if v > 0))


@dataclass
class ProvisionalWaveCal2DXD:
    """Result of the provisional 2DXD wavelength fit.

    Primary output of :func:`fit_provisional_2dxd`.  Designed to be reusable
    for later coefficient-surface fitting and order-dependent wavelength
    modelling.  **Not** a final science-quality calibration.

    The input to the fit is an :class:`~.arc_tracing.ArcTraceResult`
    (detector-space traced arc-line positions), converted to flat arrays by
    :func:`collect_centroid_data_from_arc_traces`.

    Provisionally unimplemented items
    ----------------------------------
    * Tilt coefficients are placeholder zeros.
    * Slit curvature is not modelled.
    * Per-pixel wavelength uncertainty is not propagated.
    * Iterative outlier rejection is not applied.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"H1"``).
    stored_2dxd : :class:`TwoDXDCoefficients`
        Pre-stored IDL 2DXD coefficients.  **Reference / comparison role**
        only; not used for the fitted polynomial.
    centroid_data : :class:`ArcLineCentroidData`
        The centroid input data, built from the :class:`~.arc_tracing.ArcTraceResult`.
    fitted_coeffs : ndarray, shape ``(n_disp_terms, n_order_terms)``
        Provisional 2-D polynomial coefficients.  Element ``[i, j]``
        multiplies::

            ((2*x / (n_px − 1)) − 1)^i * (home_order / m)^j

        where *x* is the *centroid_col* of each traced arc line.
    fitted_disp_degree : int
        Dispersion polynomial degree used for the fit.
    fitted_order_degree : int
        Cross-order polynomial degree used for the fit.
    home_order : int
        Reference order used for order normalization.
    n_pixels : int
        Detector column count used for column normalization (2048).
    rms_residual_um : float
        RMS fit residual in µm over the centroid input data.
    n_orders_fitted : int
        Number of orders that contributed traced centroids to the global fit.
    n_orders_bootstrap : int
        Number of orders whose per-order 1-D coefficient fell back to the
        plane-0 bootstrap (too few accepted centroids).
    per_order_wave_coeffs : dict mapping ``order_number → ndarray``
        Per-order 1-D polynomial wavelength coefficients for
        :class:`~.geometry.OrderGeometrySet` compatibility.
    """

    mode: str
    stored_2dxd: TwoDXDCoefficients
    centroid_data: ArcLineCentroidData
    fitted_coeffs: np.ndarray
    fitted_disp_degree: int
    fitted_order_degree: int
    home_order: int
    n_pixels: int
    rms_residual_um: float
    n_orders_fitted: int
    n_orders_bootstrap: int
    per_order_wave_coeffs: dict = field(default_factory=dict)

    def eval_wavelength(
        self,
        x: npt.ArrayLike,
        m: npt.ArrayLike,
    ) -> np.ndarray:
        """Evaluate the provisional wavelength polynomial.

        Computes::

            λ(x, m) = Σ_{i=0}^{Nd} Σ_{j=0}^{No} A_{i,j}
                      * ((2*x / (n_px − 1)) − 1)^i
                      * (home_order / m)^j

        where the coefficients were determined by OLS fitting to the
        traced arc-line centroid positions in :attr:`centroid_data`.

        Parameters
        ----------
        x : array_like
            Detector column positions.
        m : array_like
            Echelle order numbers.

        Returns
        -------
        ndarray
            Provisional wavelength estimates in µm.  Suitable for
            order-level wavelength mapping; not science-quality.
        """
        x = np.asarray(x, dtype=float)
        m = np.asarray(m, dtype=float)
        return _eval_2dxd_poly(
            x=x,
            m=m,
            coeffs=self.fitted_coeffs,
            home_order=self.home_order,
            n_pixels=self.n_pixels,
        )

    @property
    def n_disp_terms(self) -> int:
        """Number of polynomial terms along the dispersion axis."""
        return self.fitted_disp_degree + 1

    @property
    def n_order_terms(self) -> int:
        """Number of polynomial terms along the order axis."""
        return self.fitted_order_degree + 1


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def read_stored_2dxd_coeffs(mode_name: str) -> TwoDXDCoefficients:
    """Read the pre-stored IDL 2DXD polynomial coefficients from a FITS file.

    Reads the ``P2W_C*`` keywords from the primary HDU header of the
    ``*_wavecalinfo.fits`` file for *mode_name* and returns a
    :class:`TwoDXDCoefficients` container.

    **Role:** Reference / comparison metadata.  These are the polynomial
    coefficients computed by the IDL Spextool pipeline; they are attached
    to the :class:`ProvisionalWaveCal2DXD` result for comparison but are
    not used to compute the provisional fit.

    Parameters
    ----------
    mode_name : str
        An iSHELL mode name (e.g. ``"H1"``).

    Returns
    -------
    :class:`TwoDXDCoefficients`

    Raises
    ------
    KeyError
        If *mode_name* is not in the mode registry.
    FileNotFoundError
        If the packaged FITS file is missing.
    RuntimeError
        If the file is a Git LFS pointer.
    ValueError
        If required ``P2W_C*`` or degree keywords are missing.
    """
    from astropy.io import fits as _fits

    from .resources import get_mode_resource

    path = get_mode_resource(mode_name, "wavecalinfo")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing packaged wavecalinfo for mode '{mode_name}': {path}"
        )
    with path.open("rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"The file '{path}' is a Git LFS pointer, not the actual data.\n"
            "Install Git LFS and run:\n\n    git lfs pull\n"
        )

    with _fits.open(str(path)) as hdul:
        hdr = hdul[0].header

    for req_key in ("DISPDEG", "ORDRDEG", "HOMEORDR"):
        if req_key not in hdr:
            raise ValueError(
                f"wavecalinfo for mode '{mode_name}' is missing required "
                f"2DXD header keyword: {req_key!r}"
            )

    disp_deg = int(hdr["DISPDEG"])
    order_deg = int(hdr["ORDRDEG"])
    home_order = int(hdr["HOMEORDR"])
    n_terms = (disp_deg + 1) * (order_deg + 1)

    coeffs_list: list[float] = []
    for k in range(n_terms):
        key = f"P2W_C{k:02d}"
        if key not in hdr:
            raise ValueError(
                f"wavecalinfo for mode '{mode_name}' is missing P2W "
                f"coefficient keyword {key!r}.  Expected {n_terms} terms "
                f"for DISPDEG={disp_deg}, ORDRDEG={order_deg}."
            )
        coeffs_list.append(float(hdr[key]))

    return TwoDXDCoefficients(
        mode=mode_name,
        coeffs_flat=np.array(coeffs_list, dtype=float),
        disp_degree=disp_deg,
        order_degree=order_deg,
        home_order=home_order,
        n_pixels=_DETECTOR_NCOLS,
    )


def collect_centroid_data_from_arc_traces(
    arc_trace_result: "ArcTraceResult",
) -> ArcLineCentroidData:
    """Convert an :class:`~.arc_tracing.ArcTraceResult` to flat centroid arrays.

    This is the **primary** function for assembling input data for the
    2DXD polynomial fit.  It flattens the per-order :class:`~.arc_tracing.TracedArcLine`
    objects into parallel 1-D arrays suitable for
    :func:`fit_provisional_2dxd`.

    The **representative detector coordinate** used for each line is
    :attr:`~.arc_tracing.TracedArcLine.centroid_col` – the Gaussian-fit
    centroid column measured along the order centerline.  This is the
    quantity matched to the reference wavelength in the provisional 2DXD fit.

    Parameters
    ----------
    arc_trace_result : :class:`~.arc_tracing.ArcTraceResult`
        Output of :func:`~.arc_tracing.build_arc_trace_result_from_wavecalinfo`
        or (future) :func:`~.arc_tracing.trace_arc_lines_from_2d_image`.

    Returns
    -------
    :class:`ArcLineCentroidData`
        Flat arrays ready for :func:`fit_provisional_2dxd`.
    """
    cols, orders, wavs, snrs = arc_trace_result.to_flat_arrays()
    return ArcLineCentroidData(
        mode=arc_trace_result.mode,
        columns=cols,
        orders=orders,
        wavelengths_um=wavs,
        snr_values=snrs,
        per_order_counts=dict(arc_trace_result.per_order_counts),
        source=f"arc_tracing:{arc_trace_result.source}",
    )


def fit_provisional_2dxd(
    arc_trace_result: "ArcTraceResult",
    wavecalinfo: "WaveCalInfo",
    flatinfo: "FlatInfo",
    line_list: "LineList",
    disp_degree: int | None = None,
    order_degree: int | None = None,
    min_lines_per_order: int = 4,
) -> ProvisionalWaveCal2DXD:
    """Fit a provisional 2DXD wavelength polynomial from traced arc-line positions.

    **Primary entry point** for the 2DXD scaffold.  The workflow is:

    1.  Read the stored IDL 2DXD coefficients from the FITS header
        (:func:`read_stored_2dxd_coeffs`).
    2.  Convert the :class:`~.arc_tracing.ArcTraceResult` to flat centroid
        arrays (:func:`collect_centroid_data_from_arc_traces`).  The
        representative detector coordinate for each line is
        :attr:`~.arc_tracing.TracedArcLine.centroid_col`.
    3.  Fit a provisional 2-D polynomial to the
        ``(centroid_col, order, λ_reference)`` data::

            λ(x, m) = Σ_{i,j} A_{i,j}
                      * ((2*x / (n_px − 1)) − 1)^i
                      * (home_order / m)^j

    4.  Derive per-order 1-D wavelength polynomials via
        :func:`~.wavecal.build_geometry_from_arc_lines` for
        :class:`~.geometry.OrderGeometrySet` compatibility.

    Parameters
    ----------
    arc_trace_result : :class:`~.arc_tracing.ArcTraceResult`
        Traced arc-line positions.  **This is the primary data source.**
        Produced by :func:`~.arc_tracing.build_arc_trace_result_from_wavecalinfo`
        (proxy path) or :func:`~.arc_tracing.trace_arc_lines_from_2d_image`
        (future 2-D path).
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Provides polynomial-degree metadata (``DISPDEG``, ``ORDRDEG``,
        ``HOMEORDR``) and stored IDL coefficients (reference role).
    flatinfo : :class:`~.calibrations.FlatInfo`
        Flat-field calibration for per-order 1-D polynomial fallback.
    line_list : :class:`~.calibrations.LineList`
        ThAr reference line list for per-order polynomial fallback.
    disp_degree : int or None, optional
        Dispersion polynomial degree.  Defaults to ``wavecalinfo.disp_degree``.
    order_degree : int or None, optional
        Cross-order polynomial degree.  Defaults to ``wavecalinfo.order_degree``.
    min_lines_per_order : int, optional
        Minimum traced centroids per order for that order to use the measured
        centroid path in the per-order fallback.  Default 4.

    Returns
    -------
    :class:`ProvisionalWaveCal2DXD`

    Raises
    ------
    ValueError
        If *arc_trace_result* and *wavecalinfo* refer to different modes,
        if *wavecalinfo* and *flatinfo* refer to different modes, or if
        the collected centroid data has fewer than
        ``(disp_degree+1) * (order_degree+1)`` points.

    Notes
    -----
    **Arc trace data source.**  The quality of the provisional fit depends
    on the quality of the :class:`~.arc_tracing.ArcTraceResult`.  When using
    :func:`~.arc_tracing.build_arc_trace_result_from_wavecalinfo`, the
    centroid positions come from the 1-D centerline arc-spectrum proxy stored
    in plane 1 of the packaged ``*_wavecalinfo.fits`` file.  This is not
    equivalent to tracing from a real 2-D arc image.

    **Polynomial normalization.**  The column normalization
    ``(2x/(n_px−1))−1`` maps columns to ``[−1, 1]``; ``home_order/m`` is
    motivated by the echelle equation ``m·λ ≈ const``.  Fits to the stored
    plane-0 wavelengths reproduce those values with RMS ≲ 1 nm (see
    ``docs/ishell_2dxd_algorithm_note.md``).
    """
    if arc_trace_result.mode != wavecalinfo.mode:
        raise ValueError(
            f"arc_trace_result.mode={arc_trace_result.mode!r} does not match "
            f"wavecalinfo.mode={wavecalinfo.mode!r}."
        )
    if wavecalinfo.mode != flatinfo.mode:
        raise ValueError(
            f"wavecalinfo.mode={wavecalinfo.mode!r} does not match "
            f"flatinfo.mode={flatinfo.mode!r}."
        )

    nd = int(disp_degree) if disp_degree is not None else int(wavecalinfo.disp_degree)
    no = int(order_degree) if order_degree is not None else int(wavecalinfo.order_degree)

    # ------------------------------------------------------------------
    # Step 1: Read stored IDL 2DXD coefficients (reference role)
    # ------------------------------------------------------------------
    stored = read_stored_2dxd_coeffs(wavecalinfo.mode)

    # ------------------------------------------------------------------
    # Step 2: Build flat centroid arrays from traced arc-line positions
    # ------------------------------------------------------------------
    centroid_data = collect_centroid_data_from_arc_traces(arc_trace_result)

    # ------------------------------------------------------------------
    # Step 3: Fit provisional global 2DXD polynomial
    # ------------------------------------------------------------------
    home_order = stored.home_order
    n_pixels = stored.n_pixels
    n_terms = (nd + 1) * (no + 1)

    if centroid_data.n_points < n_terms:
        raise ValueError(
            f"Insufficient centroid data for mode '{wavecalinfo.mode}': "
            f"{centroid_data.n_points} accepted centroids (from arc traces) "
            f"but the 2DXD fit requires at least {n_terms} "
            f"(DISPDEG={nd}, ORDRDEG={no}).  "
            "Consider reducing the polynomial degrees or lowering snr_min "
            "in build_arc_trace_result_from_wavecalinfo()."
        )

    fitted_coeffs, rms_residual_um = _fit_2dxd_ols(
        x=centroid_data.columns,
        m=centroid_data.orders,
        lam=centroid_data.wavelengths_um,
        nd=nd,
        no=no,
        home_order=home_order,
        n_pixels=n_pixels,
    )

    # ------------------------------------------------------------------
    # Step 4: Derive per-order 1-D wave_coeffs for OrderGeometrySet compat
    # ------------------------------------------------------------------
    from .wavecal import build_geometry_from_arc_lines as _build_arc

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        geom_set = _build_arc(
            wavecalinfo=wavecalinfo,
            flatinfo=flatinfo,
            line_list=line_list,
            snr_min=3.0,
            min_lines_per_order=min_lines_per_order,
        )

    n_bootstrap = sum(
        1 for w in w_list
        if issubclass(w.category, RuntimeWarning)
        and "falling back to plane-0 bootstrap" in str(w.message)
    )
    n_orders_fitted = geom_set.n_orders - n_bootstrap

    per_order_wave_coeffs: dict[int, np.ndarray] = {
        g.order: g.wave_coeffs.copy()
        for g in geom_set.geometries
        if g.wave_coeffs is not None
    }

    return ProvisionalWaveCal2DXD(
        mode=wavecalinfo.mode,
        stored_2dxd=stored,
        centroid_data=centroid_data,
        fitted_coeffs=fitted_coeffs,
        fitted_disp_degree=nd,
        fitted_order_degree=no,
        home_order=home_order,
        n_pixels=n_pixels,
        rms_residual_um=float(rms_residual_um),
        n_orders_fitted=n_orders_fitted,
        n_orders_bootstrap=n_bootstrap,
        per_order_wave_coeffs=per_order_wave_coeffs,
    )


def collect_centroid_data(
    wavecalinfo: "WaveCalInfo",
    line_list: "LineList",
    snr_min: float = 3.0,
    half_window_pix: int = 15,
) -> ArcLineCentroidData:
    """Collect arc-line centroid measurements via the legacy path.

    .. note::
       **Auxiliary / legacy path.**  This function calls
       :func:`~.wavecal.fit_arc_line_centroids` directly on the packaged
       WaveCalInfo plane-1 data **without going through the arc_tracing
       layer**.  It is kept for backward compatibility and comparison.  The
       primary path for the 2DXD scaffold is
       :func:`collect_centroid_data_from_arc_traces`.

    Parameters
    ----------
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Stored wavelength calibration.  Must have ``xranges`` populated.
    line_list : :class:`~.calibrations.LineList`
        ThAr arc-line list for the same mode.
    snr_min : float, optional
        Minimum SNR.  Default 3.0.
    half_window_pix : int, optional
        Half-width of fitting window.  Default 15.

    Returns
    -------
    :class:`ArcLineCentroidData`
        Source field is ``"wavecal_fit_arc_line_centroids"``.
    """
    from .wavecal import fit_arc_line_centroids

    centroid_dict = fit_arc_line_centroids(
        wavecalinfo=wavecalinfo,
        line_list=line_list,
        snr_min=snr_min,
        half_window_pix=half_window_pix,
    )

    all_cols: list[float] = []
    all_orders: list[int] = []
    all_wavs: list[float] = []
    all_snrs: list[float] = []
    per_order_counts: dict[int, int] = {}

    for order_num in wavecalinfo.orders:
        measurements = centroid_dict.get(order_num, [])
        per_order_counts[order_num] = len(measurements)
        for centroid_pix, wavelength_um, snr in measurements:
            all_cols.append(centroid_pix)
            all_orders.append(order_num)
            all_wavs.append(wavelength_um)
            all_snrs.append(snr)

    return ArcLineCentroidData(
        mode=wavecalinfo.mode,
        columns=np.array(all_cols, dtype=float),
        orders=np.array(all_orders, dtype=float),
        wavelengths_um=np.array(all_wavs, dtype=float),
        snr_values=np.array(all_snrs, dtype=float),
        per_order_counts=per_order_counts,
        source="wavecal_fit_arc_line_centroids",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_2dxd_poly(
    x: np.ndarray,
    m: np.ndarray,
    coeffs: np.ndarray,
    home_order: int,
    n_pixels: int,
) -> np.ndarray:
    """Evaluate the 2DXD polynomial using normalized coordinates.

    The polynomial form is::

        λ(x, m) = Σ_{i,j} C[i,j]
                  * ((2*x / (n_pixels − 1)) − 1)^i
                  * (home_order / m)^j

    Parameters
    ----------
    x : ndarray
        Detector column positions.
    m : ndarray
        Echelle order numbers.
    coeffs : ndarray, shape ``(n_disp_terms, n_order_terms)``
        Polynomial coefficient matrix.
    home_order : int
        Reference order for order normalization.
    n_pixels : int
        Detector column count for column normalization.

    Returns
    -------
    ndarray
        Wavelength estimates in µm.
    """
    x_norm = (2.0 * x / (n_pixels - 1.0)) - 1.0
    inv_order = home_order / m
    n_disp, n_ord = coeffs.shape
    result = np.zeros_like(x_norm, dtype=float)
    for j in range(n_ord):
        for i in range(n_disp):
            result = result + coeffs[i, j] * (x_norm ** i) * (inv_order ** j)
    return result


def _fit_2dxd_ols(
    x: np.ndarray,
    m: np.ndarray,
    lam: np.ndarray,
    nd: int,
    no: int,
    home_order: int,
    n_pixels: int,
) -> tuple[np.ndarray, float]:
    """Fit the 2DXD polynomial to centroid data using ordinary least squares.

    Parameters
    ----------
    x, m, lam : ndarray, shape ``(n_points,)``
        Centroid column positions, order numbers, and wavelengths.
    nd, no : int
        Polynomial degrees along the dispersion and order axes.
    home_order : int
        Reference order for normalization.
    n_pixels : int
        Detector column count for normalization.

    Returns
    -------
    coeffs : ndarray, shape ``(nd+1, no+1)``
        Fitted coefficient matrix.
    rms_um : float
        RMS fit residual in µm.
    """
    x_norm = (2.0 * x / (n_pixels - 1.0)) - 1.0
    inv_order = home_order / m

    # Build design matrix: j varies slowly (outer), i varies fast (inner).
    design_cols = []
    for j in range(no + 1):
        for i in range(nd + 1):
            design_cols.append((x_norm ** i) * (inv_order ** j))
    A = np.column_stack(design_cols)  # shape (n_points, (nd+1)*(no+1))

    fit_coeffs_flat, _, _, _ = np.linalg.lstsq(A, lam, rcond=None)

    pred = A @ fit_coeffs_flat
    rms_um = float(np.std(pred - lam))

    # Reshape to (nd+1, no+1): design column k = j*(nd+1)+i → coeffs[i,j]
    coeffs = fit_coeffs_flat.reshape(no + 1, nd + 1).T  # → (nd+1, no+1)

    return coeffs, rms_um

