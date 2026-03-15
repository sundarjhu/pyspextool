"""
iSHELL telluric correction for J/H/K modes using the Vega-model method.

This module implements the xtellcor-style Vega-model telluric correction for
iSHELL (NASA IRTF) near-IR spectra in J/H/K modes (J0–J3, H1–H3, K1–K3,
Kgas).  It operates on merged 1-D spectra produced by the iSHELL merge stage
(:func:`~pyspextool.instruments.ishell.merge.merge_extracted_orders`).

Algorithm
---------
The correction follows the Spextool xtellcor method:

1. **Build IP kernel.**
   Load the pre-computed instrument-profile (IP) coefficients for the
   requested slit width from ``IP_coefficients.dat`` and construct the
   convolution kernel.  The deconvolution method is **not** available for
   iSHELL; only the IP method is supported (iSHELL Spextool Manual §6.2).

2. **Modify the Vega model.**
   Shift the high-resolution Kurucz Vega model to the radial velocity of
   the A0 V standard, convolve with the IP kernel to match the observed
   resolving power, redden to match the standard's B−V colour, and scale to
   its observed V-band magnitude.

3. **Construct the correction spectrum.**

   .. math::

       c(\\lambda) = V_{\\rm modified}(\\lambda)\\,/\\,F_{\\rm std}(\\lambda)

4. **Apply the correction.**

   .. math::

       F_{\\rm corr}(\\lambda) = F_{\\rm sci}(\\lambda) \\times c(\\lambda)

   Uncertainties are propagated in quadrature.  Quality flags are combined
   via bitwise OR.

5. **Residual wavelength shift (optional).**
   A sub-pixel cross-correlation shift of the correction spectrum relative to
   the science spectrum is found over a user-specified wavelength range that
   minimises the RMS of the corrected spectrum.  This compensates for small
   wavelength-solution differences between the science and standard
   observations.

Known limitations
-----------------
* **No hydrogen-line EW fitting.**  The iSHELL pipeline does not perform
  hydrogen-line equivalent-width matching because hydrogen lines span
  multiple echelle orders.  Residual H-line absorption artefacts may remain
  in the corrected spectrum.
* **Provisional tilt model.**  If ``tilt_provisional=True`` the upstream
  rectification used a placeholder zero-tilt model.  Merged spectra are
  structurally valid but not science-quality pending a proper 2-D tilt
  solution.
* **Placeholder IP coefficients.**  The 4.000 arcsec slit entry in
  ``IP_coefficients.dat`` is a scaled estimate; measurements from real
  iSHELL data are needed before production use.

Public API
----------
:func:`correct_telluric`
    Apply the Vega-model telluric correction to a merged iSHELL 1-D spectrum.
"""

from __future__ import annotations

import logging
import warnings
from importlib.resources import files

import numpy as np
import numpy.typing as npt

from pyspextool.io.check import check_parameter
from pyspextool.pyspextoolerror import pySpextoolError
from pyspextool.telluric.core import (
    find_shift,
    make_instrument_profile,
    make_telluric_spectrum,
)
from pyspextool.utils.interpolate import linear_interp1d
from pyspextool.utils.math import combine_flag_stack

__all__ = [
    "correct_telluric",
    "_SUPPORTED_MODES",
    "_ISHELL_PLATE_SCALE_ARCSEC_PER_PIX",
]

logger = logging.getLogger(__name__)

_ISHELL_PACKAGE = "pyspextool.instruments.ishell"

#: iSHELL plate scale (arcsec per pixel), from the iSHELL Spextool Manual
#: Table 4 (PLTSCALE example).
_ISHELL_PLATE_SCALE_ARCSEC_PER_PIX: float = 0.125

#: Supported iSHELL observing modes for the J/H/K bands.
#: L, Lp, and M modes are out of scope for this module.
_SUPPORTED_MODES: frozenset[str] = frozenset(
    ["J0", "J1", "J2", "J3", "H1", "H2", "H3", "K1", "K2", "K3", "Kgas"]
)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def correct_telluric(
    science_spectrum: npt.ArrayLike,
    standard_spectrum: npt.ArrayLike,
    vega_wavelength: npt.ArrayLike,
    vega_fluxdensity: npt.ArrayLike,
    vega_continuum: npt.ArrayLike,
    vega_fitted_continuum: npt.ArrayLike,
    slit_width_arcsec: float,
    standard_bmag: float,
    standard_vmag: float,
    standard_rv: float = 0.0,
    apply_shift: bool = False,
    shift_wavelength_range: list | None = None,
    tilt_provisional: bool = False,
    ew_scale: float = 1.0,
    intensity_unit: str = "erg s-1 cm-2 A-1",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply Vega-model telluric correction to a merged iSHELL spectrum.

    Implements the xtellcor/Spextool Vega-model telluric-correction algorithm
    for iSHELL J/H/K modes, operating on merged 1-D spectra produced by
    :func:`~pyspextool.instruments.ishell.merge.merge_extracted_orders`.

    .. note::
        The deconvolution kernel method is **not** available for iSHELL.  Only
        the IP (instrument-profile) method is used here, as stated in the
        iSHELL Spextool Manual §6.2.

    .. warning::
        Hydrogen-line equivalent-width fitting is **not** performed.  iSHELL
        hydrogen lines span multiple echelle orders, making per-line EW
        optimisation impractical on merged spectra.  Residual H-line artefacts
        may remain in the corrected spectrum.

    Parameters
    ----------
    science_spectrum : array-like, shape (4, nwave)
        Merged science 1-D spectrum with rows:

        * Row 0 — wavelength (µm), monotonically increasing.
        * Row 1 — intensity (flux density).
        * Row 2 — uncertainty.
        * Row 3 — spectral quality flag (integer bitmask, 0 = clean).

    standard_spectrum : array-like, shape (4, nwave_std)
        Merged A0 V standard-star 1-D spectrum in the same format as
        *science_spectrum*.  The wavelength grids of science and standard
        need not be identical.

    vega_wavelength : array-like, shape (nmodel,)
        Wavelengths (µm) of the high-resolution Kurucz Vega model.

    vega_fluxdensity : array-like, shape (nmodel,)
        Flux density of the Vega model in ``erg s⁻¹ cm⁻² Å⁻¹``.

    vega_continuum : array-like, shape (nmodel,)
        Continuum flux density of the Vega model.

    vega_fitted_continuum : array-like, shape (nmodel,)
        Fitted continuum flux density of the Vega model.

    slit_width_arcsec : float
        iSHELL slit width in arcseconds.  Must match one of the values in
        ``IP_coefficients.dat`` (0.375, 0.750, 1.500, or 4.000).

    standard_bmag : float
        B-band magnitude of the A0 V standard star.

    standard_vmag : float
        V-band magnitude of the A0 V standard star.

    standard_rv : float, default 0.0
        Heliocentric radial velocity of the A0 V standard star in km s⁻¹.
        Used to shift the Vega model to the standard's rest frame.  A
        barycentric correction should be applied to this value before calling
        if it is available from the FITS metadata.

    apply_shift : bool, default False
        If ``True``, perform a sub-pixel cross-correlation shift of the
        correction spectrum relative to the science spectrum to minimise the
        RMS over *shift_wavelength_range*.  Requires *shift_wavelength_range*
        to be provided.

    shift_wavelength_range : list of two floats or None, default None
        ``[lambda_min, lambda_max]`` wavelength range (µm) used for the
        residual-shift minimisation.  Ignored if *apply_shift* is ``False``.

    tilt_provisional : bool, default False
        If ``True``, the upstream rectification used a placeholder zero-tilt
        model; a :class:`RuntimeWarning` is emitted.  See
        ``docs/ishell_telluric_guide.md`` for details.

    ew_scale : float, default 1.0
        Global equivalent-width scale factor applied uniformly to all Vega
        model absorption lines.  Hydrogen-line EW fitting is not performed
        (see note above); callers may pass a value measured externally.

    intensity_unit : str, default ``"erg s-1 cm-2 A-1"``
        Flux density units for the output corrected spectrum.  Must be a
        unit recognised by
        :func:`~pyspextool.utils.units.convert_fluxdensity`.

    Returns
    -------
    corrected_spectrum : ndarray, shape (4, nwave)
        Telluric-corrected science spectrum on the same wavelength grid as
        *science_spectrum*:

        * Row 0 — wavelength (µm), unchanged from *science_spectrum*.
        * Row 1 — corrected flux density.
        * Row 2 — propagated uncertainty.
        * Row 3 — quality flag (bitwise OR of science and correction flags).

    correction_spectrum : ndarray, shape (4, nwave_std)
        Telluric correction spectrum on the standard's wavelength grid:

        * Row 0 — wavelength (µm).
        * Row 1 — correction values (= modified Vega / A0 V standard).
        * Row 2 — uncertainty on the correction.
        * Row 3 — quality flag from the standard spectrum.

    metadata : dict
        Summary of correction parameters, with keys:

        * ``"kernel_method"`` — always ``"ip"`` for iSHELL.
        * ``"slit_width_arcsec"`` — slit width used (float).
        * ``"ip_coefficients"`` — (3,) array ``[c0, c1, c2]``.
        * ``"shift_applied"`` — pixel shift applied (float; 0.0 if none).
        * ``"ew_scale"`` — the *ew_scale* value used.
        * ``"tilt_provisional"`` — value of *tilt_provisional*.
        * ``"intensity_unit"`` — flux density unit string.

    Raises
    ------
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If *science_spectrum* or *standard_spectrum* is not a 2-D array with
        exactly 4 rows.
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If *slit_width_arcsec* is not present in ``IP_coefficients.dat``.
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If the Vega model has insufficient wavelength coverage for the
        standard spectrum.

    Examples
    --------
    Correct a synthetic spectrum (placeholder Vega model):

    >>> import numpy as np
    >>> from pyspextool.instruments.ishell.telluric import correct_telluric
    >>> nwave = 512
    >>> wave = np.linspace(1.1, 1.35, nwave)
    >>> sci = np.array([wave, np.ones(nwave), 0.01 * np.ones(nwave),
    ...                 np.zeros(nwave)])
    >>> std = sci.copy()
    >>> vwave = np.linspace(0.9, 1.6, 5000)
    >>> vfd = np.ones(5000)
    >>> corrected, correction, meta = correct_telluric(
    ...     sci, std, vwave, vfd, vfd, vfd,
    ...     slit_width_arcsec=0.375,
    ...     standard_bmag=9.8, standard_vmag=9.5)
    >>> corrected.shape
    (4, 512)
    """
    # ------------------------------------------------------------------
    # Convert inputs to numpy arrays
    # ------------------------------------------------------------------
    science_spectrum = np.asarray(science_spectrum, dtype=float)
    standard_spectrum = np.asarray(standard_spectrum, dtype=float)
    vega_wavelength = np.asarray(vega_wavelength, dtype=float)
    vega_fluxdensity = np.asarray(vega_fluxdensity, dtype=float)
    vega_continuum = np.asarray(vega_continuum, dtype=float)
    vega_fitted_continuum = np.asarray(vega_fitted_continuum, dtype=float)

    # ------------------------------------------------------------------
    # Validate shapes
    # ------------------------------------------------------------------
    if science_spectrum.ndim != 2 or science_spectrum.shape[0] != 4:
        raise pySpextoolError(
            "correct_telluric: science_spectrum must be a (4, nwave) array. "
            f"Got shape {science_spectrum.shape}."
        )

    if standard_spectrum.ndim != 2 or standard_spectrum.shape[0] != 4:
        raise pySpextoolError(
            "correct_telluric: standard_spectrum must be a (4, nwave) array. "
            f"Got shape {standard_spectrum.shape}."
        )

    check_parameter("correct_telluric", "slit_width_arcsec",
                    slit_width_arcsec, ["float", "int"])

    check_parameter("correct_telluric", "standard_bmag",
                    standard_bmag, ["float", "int"])

    check_parameter("correct_telluric", "standard_vmag",
                    standard_vmag, ["float", "int"])

    check_parameter("correct_telluric", "standard_rv",
                    standard_rv, ["float", "int", "float64"])

    check_parameter("correct_telluric", "apply_shift", apply_shift, "bool")

    check_parameter("correct_telluric", "shift_wavelength_range",
                    shift_wavelength_range, ["list", "NoneType"])

    check_parameter("correct_telluric", "tilt_provisional",
                    tilt_provisional, "bool")

    check_parameter("correct_telluric", "ew_scale",
                    ew_scale, ["float", "int"])

    check_parameter("correct_telluric", "intensity_unit",
                    intensity_unit, "str")

    # ------------------------------------------------------------------
    # Emit warning if the upstream tilt model is provisional
    # ------------------------------------------------------------------
    if tilt_provisional:
        warnings.warn(
            "correct_telluric: tilt_provisional=True — the upstream "
            "rectification used a placeholder zero-tilt model.  Merged "
            "spectra are structurally valid but not science-quality pending "
            "a proper 2-D tilt solution.  See docs/ishell_telluric_guide.md.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Load iSHELL IP coefficients for the requested slit width
    # ------------------------------------------------------------------
    ip_path = files(_ISHELL_PACKAGE) / "IP_coefficients.dat"
    with ip_path.open("r") as fh:
        slitw_arc, c0, c1, c2 = np.loadtxt(fh, comments="#", unpack=True)

    z = np.where(np.isclose(slitw_arc, float(slit_width_arcsec)))[0]
    if len(z) == 0:
        available = ", ".join(str(s) for s in slitw_arc)
        raise pySpextoolError(
            f"correct_telluric: slit_width_arcsec={slit_width_arcsec} not "
            f"found in IP_coefficients.dat.  Available widths: {available}."
        )

    ip_coefficients = np.array([float(c0[z[0]]), float(c1[z[0]]), float(c2[z[0]])])

    # ------------------------------------------------------------------
    # Build the IP convolution kernel
    # ------------------------------------------------------------------
    std_wave = standard_spectrum[0]
    min_wave = float(np.nanmin(std_wave))
    max_wave = float(np.nanmax(std_wave))

    # Median dispersion of the (non-uniform) standard spectrum
    std_diffs = np.diff(std_wave)
    positive_diffs = std_diffs[std_diffs > 0.0]
    if len(positive_diffs) == 0:
        raise pySpextoolError(
            "correct_telluric: standard_spectrum has no valid (positive) "
            "wavelength spacing — cannot compute dispersion."
        )
    standard_dispersion = float(np.median(positive_diffs))

    # Median dispersion of the Vega model over the standard's wavelength range
    z_vega = np.where(
        (vega_wavelength >= min_wave) & (vega_wavelength <= max_wave)
    )[0]
    if len(z_vega) < 2:
        raise pySpextoolError(
            "correct_telluric: Vega model has insufficient wavelength coverage "
            f"over the standard's range [{min_wave:.4f}, {max_wave:.4f}] µm.  "
            "Ensure the Vega model spans the full wavelength range of the "
            "standard spectrum."
        )
    vega_dispersion = float(np.median(np.diff(vega_wavelength[z_vega])))

    # iSHELL slit width in detector pixels
    slit_width_pix = float(slit_width_arcsec) / _ISHELL_PLATE_SCALE_ARCSEC_PER_PIX

    # FWHM of the standard spectrum in wavelength units
    standard_fwhm = slit_width_pix * standard_dispersion

    # Number of kernel pixels: span 10 FWHMs, enforce odd length
    nkernel = max(1, int(np.round(10.0 * standard_fwhm / vega_dispersion)))
    if nkernel % 2 == 0:
        nkernel += 1

    # x in standard-star pixel units (matches the convention in get_kernels.py)
    x = (
        np.arange(-(nkernel // 2), nkernel // 2 + 1)
        * vega_dispersion
        / standard_dispersion
    )

    # Build the IP profile; fall back to a delta function if c1=c2=0
    if ip_coefficients[1] == 0.0 and ip_coefficients[2] == 0.0:
        # Placeholder coefficients — no convolution applied
        kernel = np.zeros(nkernel)
        kernel[nkernel // 2] = 1.0
        logger.warning(
            "correct_telluric: IP coefficients for slit_width_arcsec=%s are "
            "all zero (placeholder). Using a delta-function kernel (no "
            "convolution). Replace IP_coefficients.dat values with real "
            "measurements before production use.",
            slit_width_arcsec,
        )
    else:
        kernel = make_instrument_profile(x, ip_coefficients)

    # ------------------------------------------------------------------
    # Construct the telluric correction spectrum
    # (correction = modified Vega / A0 V standard)
    # ------------------------------------------------------------------

    # EW scale control points: constant scale over the full wavelength range.
    # Hydrogen-line EW fitting is NOT performed for iSHELL.
    control_points = np.array([min_wave, max_wave])
    control_values = np.array([float(ew_scale), float(ew_scale)])

    telluric_correction, telluric_unc, _modified_vega, _modified_vega_cont = (
        make_telluric_spectrum(
            standard_spectrum[0],  # wavelength
            standard_spectrum[1],  # flux density
            standard_spectrum[2],  # uncertainty
            float(standard_rv),
            float(standard_vmag),
            float(standard_bmag),
            vega_wavelength,
            vega_fluxdensity,
            vega_continuum,
            vega_fitted_continuum,
            kernel,
            control_points,
            control_values,
            0,  # order number (not meaningful for a merged spectrum)
            intensity_unit,
        )
    )

    # ------------------------------------------------------------------
    # Build the correction spectrum array (on the standard's wavelength grid)
    # ------------------------------------------------------------------
    correction_spectrum = np.zeros_like(standard_spectrum)
    correction_spectrum[0] = standard_spectrum[0]
    correction_spectrum[1] = telluric_correction
    correction_spectrum[2] = telluric_unc
    correction_spectrum[3] = standard_spectrum[3]

    # ------------------------------------------------------------------
    # Interpolate correction onto the science wavelength grid
    # ------------------------------------------------------------------
    sci_wave = science_spectrum[0]

    tc_f, tc_u = linear_interp1d(
        standard_spectrum[0],
        telluric_correction,
        sci_wave,
        input_u=telluric_unc,
    )

    # ------------------------------------------------------------------
    # Residual wavelength shift (optional)
    # ------------------------------------------------------------------
    shift_applied = 0.0
    if apply_shift and shift_wavelength_range is not None:
        shift_applied = find_shift(
            sci_wave,
            science_spectrum[1],
            tc_f,
            list(shift_wavelength_range),
        )
        if shift_applied != 0.0:
            x_pix = np.arange(len(sci_wave), dtype=float)
            tc_f, tc_u = linear_interp1d(
                x_pix + shift_applied,
                tc_f,
                x_pix,
                input_u=tc_u,
            )

    # ------------------------------------------------------------------
    # Apply the correction to the science spectrum
    # science_corrected = science × correction
    # ------------------------------------------------------------------
    corrected_flux = science_spectrum[1] * tc_f

    # Propagate uncertainties in quadrature:
    #   σ_corrected² = (correction × σ_science)² + (science × σ_correction)²
    corrected_unc = np.sqrt(
        (tc_f * science_spectrum[2]) ** 2
        + (science_spectrum[1] * tc_u) ** 2
    )

    # ------------------------------------------------------------------
    # Combine quality flags (bitwise OR)
    # ------------------------------------------------------------------
    sci_mask = np.asarray(science_spectrum[3], dtype=np.uint8)

    # Mark science pixels outside the standard's wavelength coverage
    outside_coverage = (sci_wave < min_wave) | (sci_wave > max_wave)
    coverage_mask = np.where(outside_coverage, np.uint8(1), np.uint8(0))

    combined_flags = combine_flag_stack(
        np.stack([sci_mask, coverage_mask])
    )

    # ------------------------------------------------------------------
    # Assemble the corrected spectrum array
    # ------------------------------------------------------------------
    corrected_spectrum = np.zeros_like(science_spectrum)
    corrected_spectrum[0] = sci_wave
    corrected_spectrum[1] = corrected_flux
    corrected_spectrum[2] = corrected_unc
    corrected_spectrum[3] = combined_flags

    # ------------------------------------------------------------------
    # Build metadata dict
    # ------------------------------------------------------------------
    metadata: dict = {
        "kernel_method": "ip",
        "slit_width_arcsec": float(slit_width_arcsec),
        "ip_coefficients": ip_coefficients,
        "shift_applied": float(shift_applied),
        "ew_scale": float(ew_scale),
        "tilt_provisional": bool(tilt_provisional),
        "intensity_unit": intensity_unit,
    }

    logger.info(
        "correct_telluric: correction applied (slit=%.3f\", shift=%.3f px, "
        "ew_scale=%.3f, tilt_provisional=%s).",
        slit_width_arcsec,
        shift_applied,
        ew_scale,
        tilt_provisional,
    )

    return corrected_spectrum, correction_spectrum, metadata
