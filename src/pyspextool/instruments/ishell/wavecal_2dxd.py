"""
Provisional 2DXD wavelength-mapping scaffold for iSHELL.

This module provides the first bridge between per-order arc-line centroid
measurements (produced by
:func:`~pyspextool.instruments.ishell.wavecal.fit_arc_line_centroids`) and
the global 2-D echelle polynomial model ("2DXD") used by the IDL Spextool
pipeline.

What this module implements
---------------------------
1.  **Reading stored IDL 2DXD coefficients.**  The packaged
    ``*_wavecalinfo.fits`` files carry a pre-computed 2DXD polynomial in
    their FITS headers (keywords ``P2W_C00`` … ``P2W_Cnn``) as well as the
    polynomial degrees (``DISPDEG``, ``ORDRDEG``) and the reference order
    (``HOMEORDR``).  :func:`read_stored_2dxd_coeffs` reads those keywords
    and returns a :class:`TwoDXDCoefficients` dataclass.

    .. note::
       The **exact evaluation convention** of the stored coefficients (the
       precise coordinate normalization used when the IDL pipeline wrote
       them) has *not* been verified against the IDL source code.  The raw
       coefficient array is stored and documented, but :meth:`eval_stored`
       is provisional.  See :class:`TwoDXDCoefficients` for details.

2.  **Collecting centroid data across all orders.**
    :func:`collect_centroid_data` calls
    :func:`~pyspextool.instruments.ishell.wavecal.fit_arc_line_centroids`
    and flattens the per-order results into a single
    :class:`ArcLineCentroidData` array of ``(column, order, wavelength)``
    triples.

3.  **Fitting a provisional 2DXD polynomial.**  :func:`fit_provisional_2dxd`
    uses the collected centroid data to fit a 2-D polynomial of the form::

        λ(x, m) = Σ_{i=0}^{Nd} Σ_{j=0}^{No} A_{ij}
                  * ((2*x / (n_px − 1)) − 1)^i
                  * (home_order / m)^j

    where *x* is the detector column, *m* is the echelle order number,
    *n_px* = 2048 (detector width), *home_order* is the reference order from
    the FITS header, *Nd* = ``DISPDEG``, and *No* = ``ORDRDEG``.  The
    normalization ``(2x/(n_px−1) − 1)`` maps column indices to ``[−1, 1]``;
    the factor ``home_order/m`` is dimensionless and physically motivated by
    the echelle equation (``m · λ ≈ const``).

    Numerical experiments on the stored H1 plane-0 data show that this form
    reproduces the plane-0 wavelengths with RMS residual ≲ 1 nm when fit
    using ordinary least squares (see
    ``docs/ishell_2dxd_algorithm_note.md``).

What this module does NOT implement
------------------------------------
* **Tilt measurement** – spectral-line tilt requires spatial variation of
  arc-line positions across the slit, which is not stored in the packaged
  ``*_wavecalinfo.fits`` files.
* **Slit curvature** – no curvature correction is applied.
* **Final rectification indices** – the provisional fit does not generate
  the per-pixel wavelength/spatial index arrays needed for 2-D spectral
  rectification.
* **Iterative outlier rejection** – unlike the IDL pipeline, no interactive
  or iterative sigma-clipping of arc lines is performed.
* **Full 2DXD science-quality calibration** – this is an engineering scaffold
  only.

See ``docs/ishell_2dxd_algorithm_note.md`` for a full developer description,
a comparison with the legacy IDL pipeline, and a list of unresolved issues.

Public API
----------
- :class:`TwoDXDCoefficients` – stored IDL 2DXD coefficient container.
- :class:`ArcLineCentroidData` – flat collection of centroid measurements
  across all orders.
- :class:`ProvisionalWaveCal2DXD` – the primary result of a provisional
  2DXD fit.
- :func:`read_stored_2dxd_coeffs` – reads P2W_C* keywords from a
  ``*_wavecalinfo.fits`` FITS file.
- :func:`collect_centroid_data` – flattens per-order centroid results.
- :func:`fit_provisional_2dxd` – main entry point; returns a
  :class:`ProvisionalWaveCal2DXD`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .wavecal import fit_arc_line_centroids

if TYPE_CHECKING:
    from .calibrations import FlatInfo, LineList, WaveCalInfo

__all__ = [
    "TwoDXDCoefficients",
    "ArcLineCentroidData",
    "ProvisionalWaveCal2DXD",
    "read_stored_2dxd_coeffs",
    "collect_centroid_data",
    "fit_provisional_2dxd",
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

    .. warning::
       The **exact evaluation convention** of the stored polynomial
       (specifically, the coordinate normalization applied to detector
       columns and order numbers before evaluating the polynomial) has
       **not** been verified against the IDL source code.  The
       :meth:`eval_provisional` method implements one plausible normalization
       (Chebyshev-style column normalization, home-order-scaled inverse
       order) and is offered for reference only.  Residuals against the
       stored plane-0 wavelengths should be checked before trusting any
       evaluation.

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
        read directly from the FITS header and should be cross-checked
        against IDL source before relying on it.
    disp_degree : int
        Polynomial degree along the dispersion axis (``DISPDEG`` keyword).
    order_degree : int
        Polynomial degree along the order axis (``ORDRDEG`` keyword).
    home_order : int
        Reference echelle order used for order normalization (``HOMEORDR``
        keyword).
    n_pixels : int
        Number of detector columns used for column normalization.  Defaults
        to :data:`_DETECTOR_NCOLS` (2048) when not otherwise specified.
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
           This evaluation is **provisional**.  The normalization applied
           here (Chebyshev column normalization + home-order-scaled inverse
           order) is one plausible interpretation that gives sub-nm
           residuals when fitting the stored plane-0 data *de novo*.
           Whether the stored ``P2W_C*`` coefficients were written under
           the same convention has **not** been verified against IDL source
           code.  Treat the results as a sanity check only.

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
    """Arc-line centroid measurements collected across all echelle orders.

    This is the flattened representation of the per-order centroid
    dictionaries produced by
    :func:`~pyspextool.instruments.ishell.wavecal.fit_arc_line_centroids`.
    It aggregates all ``(column, order, wavelength)`` triples into 1-D
    arrays that can be used directly as input to a 2-D polynomial fit.

    Parameters
    ----------
    mode : str
        iSHELL mode name.
    columns : ndarray, shape ``(n_points,)``
        Measured centroid column positions (fractional detector pixels).
    orders : ndarray, shape ``(n_points,)``
        Echelle order number for each centroid measurement.
    wavelengths_um : ndarray, shape ``(n_points,)``
        Known vacuum wavelengths in µm from the reference line list.
    snr_values : ndarray, shape ``(n_points,)``
        Peak SNR of each arc-line centroid.
    per_order_counts : dict mapping ``order_number → count``
        Number of accepted centroid measurements per order.
    """

    mode: str
    columns: np.ndarray
    orders: np.ndarray
    wavelengths_um: np.ndarray
    snr_values: np.ndarray
    per_order_counts: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        """Total number of accepted centroid measurements."""
        return int(len(self.columns))

    @property
    def n_orders_with_data(self) -> int:
        """Number of echelle orders that have at least one centroid."""
        return int(sum(1 for v in self.per_order_counts.values() if v > 0))


@dataclass
class ProvisionalWaveCal2DXD:
    """Result of the provisional 2DXD wavelength fit.

    This dataclass is the primary output of :func:`fit_provisional_2dxd`.
    It is designed to be usable as input to later coefficient-surface fitting
    and order-dependent wavelength modelling, but it is **not** a final
    science-quality calibration.

    Provisionally unimplemented items
    ----------------------------------
    * Tilt coefficients are not stored here (zero-tilt placeholder in
      :class:`~.geometry.OrderGeometrySet`).
    * Slit curvature is not modelled.
    * Per-pixel wavelength uncertainty is not propagated.
    * Iterative outlier rejection is not applied.

    Parameters
    ----------
    mode : str
        iSHELL mode name (e.g. ``"H1"``).
    stored_2dxd : :class:`TwoDXDCoefficients`
        Pre-stored IDL 2DXD coefficients read from the FITS header.  Stored
        for reference and downstream comparison; their evaluation convention
        is unverified (see :class:`TwoDXDCoefficients`).
    centroid_data : :class:`ArcLineCentroidData`
        The arc-line centroid input data used for the provisional fit.
    fitted_coeffs : ndarray, shape ``(n_disp_terms, n_order_terms)``
        Provisional 2-D polynomial coefficients fitted to the centroid data.
        Element ``[i, j]`` multiplies::

            ((2*x / (n_px − 1)) − 1)^i * (home_order / m)^j

        The polynomial convention matches :meth:`eval_wavelength`.
    fitted_disp_degree : int
        Dispersion polynomial degree used for the provisional fit.
    fitted_order_degree : int
        Cross-order polynomial degree used for the provisional fit.
    home_order : int
        Reference order used for order normalization.
    n_pixels : int
        Detector column count used for column normalization (typically 2048).
    rms_residual_um : float
        RMS fit residual in µm computed from the centroid data.  This
        reflects only the quality of the polynomial representation of the
        centroid data and does **not** include systematic errors or the
        accuracy of the line-list wavelengths.
    n_orders_fitted : int
        Number of echelle orders that contributed centroid data to the
        global fit (i.e. had ``≥ min_lines_per_order`` accepted centroids).
    n_orders_bootstrap : int
        Number of echelle orders for which the per-order wavelength
        polynomial was derived from the plane-0 bootstrap fallback rather
        than from measured centroids.
    per_order_wave_coeffs : dict mapping ``order_number → ndarray``
        Per-order 1-D polynomial wavelength coefficients (from
        :func:`~.wavecal.build_geometry_from_arc_lines` or the bootstrap
        fallback).  These are provided for compatibility with
        :class:`~.geometry.OrderGeometrySet` and downstream stages.
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

        where the coefficients :attr:`fitted_coeffs` ``[i, j]`` were
        determined by ordinary least-squares fitting to the arc-line
        centroid data in :attr:`centroid_data`.

        Parameters
        ----------
        x : array_like
            Detector column positions.
        m : array_like
            Echelle order numbers.

        Returns
        -------
        ndarray
            Provisional wavelength estimates in µm.  These are suitable
            for order-level wavelength mapping but should not be treated as
            a final science-quality wavelength solution.
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

    The coefficient keywords are named ``P2W_C00``, ``P2W_C01``, …,
    ``P2W_C{n}`` where *n = (DISPDEG+1)*(ORDRDEG+1) − 1*.  The index
    ordering is::

        k = j * (DISPDEG + 1) + i

    where *i* is the column-polynomial power and *j* is the
    order-polynomial power.  **This has not been verified against the IDL
    source code.**

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
        If the FITS header is missing required keywords or the number of
        ``P2W_C*`` keywords does not match ``(DISPDEG+1)*(ORDRDEG+1)``.
    """
    from astropy.io import fits as _fits

    from .resources import get_mode_resource

    path = get_mode_resource(mode_name, "wavecalinfo")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing packaged wavecalinfo for mode '{mode_name}': {path}"
        )
    # Check for Git LFS pointer
    with path.open("rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"The file '{path}' is a Git LFS pointer, not the actual data.\n"
            "Install Git LFS and run:\n\n    git lfs pull\n"
        )

    with _fits.open(str(path)) as hdul:
        hdr = hdul[0].header

    # Read degree and reference-order metadata
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

    # Read coefficients in order P2W_C00, P2W_C01, ...
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


def collect_centroid_data(
    wavecalinfo: "WaveCalInfo",
    line_list: "LineList",
    snr_min: float = 3.0,
    half_window_pix: int = 15,
) -> ArcLineCentroidData:
    """Collect arc-line centroid measurements across all echelle orders.

    Calls
    :func:`~pyspextool.instruments.ishell.wavecal.fit_arc_line_centroids`
    and concatenates the per-order results into flat arrays suitable for a
    global 2-D polynomial fit.

    Parameters
    ----------
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Stored wavelength calibration for the mode.  Must have ``xranges``
        populated and ``data`` with valid plane 0 and plane 1 arrays.
    line_list : :class:`~.calibrations.LineList`
        ThAr arc-line list for the same mode.
    snr_min : float, optional
        Minimum SNR for a centroid measurement to be accepted.
        Default is 3.0.
    half_window_pix : int, optional
        Half-width (pixels) of the Gaussian fitting window.
        Default is 15.

    Returns
    -------
    :class:`ArcLineCentroidData`
        Flat collection of all accepted ``(column, order, wavelength)``
        triples across all orders.

    Notes
    -----
    Orders in *line_list* that are not in *wavecalinfo* are silently
    skipped.  Orders with no accepted centroids are recorded in
    ``per_order_counts`` with count 0.
    """
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
    )


def fit_provisional_2dxd(
    wavecalinfo: "WaveCalInfo",
    flatinfo: "FlatInfo",
    line_list: "LineList",
    disp_degree: int | None = None,
    order_degree: int | None = None,
    snr_min: float = 3.0,
    min_lines_per_order: int = 4,
) -> ProvisionalWaveCal2DXD:
    """Fit a provisional 2DXD wavelength polynomial from arc-line centroids.

    This is the main entry point for the 2DXD scaffold.  It:

    1.  Reads the pre-stored IDL 2DXD polynomial coefficients from the
        FITS header (:func:`read_stored_2dxd_coeffs`).
    2.  Collects arc-line centroid measurements across all orders
        (:func:`collect_centroid_data`).
    3.  Fits a provisional 2-D polynomial::

            λ(x, m) = Σ_{i,j} A_{i,j}
                      * ((2*x / (n_px − 1)) − 1)^i
                      * (home_order / m)^j

        to the collected ``(column, order, wavelength)`` centroid data
        using ordinary least squares.
    4.  Also derives per-order 1-D wavelength polynomials using
        :func:`~.wavecal.build_geometry_from_arc_lines` for compatibility
        with :class:`~.geometry.OrderGeometrySet`.

    Parameters
    ----------
    wavecalinfo : :class:`~.calibrations.WaveCalInfo`
        Stored wavelength calibration for the mode.
    flatinfo : :class:`~.calibrations.FlatInfo`
        Flat-field calibration for the same mode.
    line_list : :class:`~.calibrations.LineList`
        ThAr arc-line list for the mode.
    disp_degree : int or None, optional
        Degree of the 2-D polynomial along the dispersion axis.  Defaults
        to ``wavecalinfo.disp_degree``.
    order_degree : int or None, optional
        Degree of the 2-D polynomial along the order axis.  Defaults to
        ``wavecalinfo.order_degree``.
    snr_min : float, optional
        Minimum SNR threshold for arc-line centroids.  Default 3.0.
    min_lines_per_order : int, optional
        Minimum accepted centroids per order required for that order to
        contribute to the global 2-D fit.  Orders below this threshold are
        counted as ``n_orders_bootstrap`` in the result.  Default 4.

    Returns
    -------
    :class:`ProvisionalWaveCal2DXD`
        Contains the stored IDL coefficients, the centroid data, the fitted
        polynomial, per-order 1-D wavelength polynomials, and quality
        metrics.

    Raises
    ------
    ValueError
        If *wavecalinfo* and *flatinfo* refer to different modes, or if
        the centroid data contains fewer than
        ``(disp_degree + 1) * (order_degree + 1)`` accepted points
        (underdetermined fit).

    Notes
    -----
    **Polynomial form.**  The normalization used here—Chebyshev-style column
    normalization to ``[−1, 1]`` and home-order-scaled inverse order—is
    physically motivated by the echelle equation and is numerically verified
    to reproduce the stored plane-0 wavelengths with RMS residual ≲ 1 nm
    when fitted to the plane-0 data (see
    ``docs/ishell_2dxd_algorithm_note.md``).  The stored IDL ``P2W_C*``
    coefficients appear to use a *different* evaluation convention that has
    not yet been reverse-engineered.

    **Provisional status.**  The fitted polynomial does not model tilt,
    curvature, or spatial variation along the slit.  It represents a 1-D
    (centerline) wavelength solution across all orders, suitable as a
    starting point for coefficient-surface fitting.
    """
    if wavecalinfo.mode != flatinfo.mode:
        raise ValueError(
            f"wavecalinfo.mode={wavecalinfo.mode!r} does not match "
            f"flatinfo.mode={flatinfo.mode!r}."
        )

    nd = int(disp_degree) if disp_degree is not None else int(wavecalinfo.disp_degree)
    no = int(order_degree) if order_degree is not None else int(wavecalinfo.order_degree)

    # ------------------------------------------------------------------
    # Step 1: Read stored IDL 2DXD coefficients
    # ------------------------------------------------------------------
    stored = read_stored_2dxd_coeffs(wavecalinfo.mode)

    # ------------------------------------------------------------------
    # Step 2: Collect centroid data across all orders
    # ------------------------------------------------------------------
    centroid_data = collect_centroid_data(
        wavecalinfo=wavecalinfo,
        line_list=line_list,
        snr_min=snr_min,
    )

    # ------------------------------------------------------------------
    # Step 3: Fit provisional global 2DXD polynomial
    # ------------------------------------------------------------------
    home_order = stored.home_order
    n_pixels = stored.n_pixels
    n_terms = (nd + 1) * (no + 1)

    if centroid_data.n_points < n_terms:
        raise ValueError(
            f"Insufficient centroid data for mode '{wavecalinfo.mode}': "
            f"{centroid_data.n_points} accepted centroids but the 2DXD "
            f"fit requires at least {n_terms} (DISPDEG={nd}, ORDRDEG={no}).  "
            "Consider reducing the polynomial degrees or lowering snr_min."
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
    # Step 4: Derive per-order 1-D wave_coeffs using the existing wavecal path
    # ------------------------------------------------------------------
    from .wavecal import build_geometry_from_arc_lines as _build_arc

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        geom_set = _build_arc(
            wavecalinfo=wavecalinfo,
            flatinfo=flatinfo,
            line_list=line_list,
            snr_min=snr_min,
            min_lines_per_order=min_lines_per_order,
        )

    # Count bootstrap vs fitted orders from the warnings emitted
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
    """Evaluate the 2DXD polynomial using the normalized coordinates.

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

    # Build design matrix: columns correspond to (i, j) pairs with
    # j varying slowly (outer loop) and i varying fast (inner loop).
    cols = []
    for j in range(no + 1):
        for i in range(nd + 1):
            cols.append((x_norm ** i) * (inv_order ** j))
    A = np.column_stack(cols)  # shape (n_points, (nd+1)*(no+1))

    fit_coeffs_flat, residuals, rank, sv = np.linalg.lstsq(A, lam, rcond=None)

    pred = A @ fit_coeffs_flat
    rms_um = float(np.std(pred - lam))

    # Reshape to (nd+1, no+1): A[i,j] at index j*(nd+1)+i
    coeffs = fit_coeffs_flat.reshape(no + 1, nd + 1).T  # → (nd+1, no+1)

    return coeffs, rms_um
