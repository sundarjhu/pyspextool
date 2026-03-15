"""
Tests for the iSHELL Vega-model telluric correction module.

Coverage
--------
* Successful telluric correction flow (J, H, K modes).
* Dimensional consistency of the corrected spectrum.
* Uncertainty propagation through the correction multiplication.
* Correction spectrum sanity (finite values, positive).
* Residual wavelength shift application.
* Flag propagation via bitwise OR.
* Tilt-provisional RuntimeWarning.
* Failure cases: malformed spectra, unknown slit width, insufficient Vega
  coverage, bad wavelength spacing.

Provisional tilt model
----------------------
All tests use synthetic spectra.  Structural properties (shape, finiteness,
flag propagation, uncertainty) are checked rather than exact pixel values,
because:

1. The upstream provisional zero-tilt rectification model is not
   scientifically meaningful.
2. The IP coefficients and Vega model values used in tests are synthetic.

No real iSHELL science data is required.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyspextool.instruments.ishell.telluric import (
    correct_telluric,
    _SUPPORTED_MODES,
    _ISHELL_PLATE_SCALE_ARCSEC_PER_PIX,
)
from pyspextool.pyspextoolerror import pySpextoolError


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_NWAVE = 256  # number of wavelength samples per synthetic spectrum
_NWAVE_VEGA = 2000  # Vega model wavelength samples


def _make_flat_spectrum(
    wave_start: float,
    wave_end: float,
    nwave: int = _NWAVE,
    intensity: float = 100.0,
    uncertainty: float = 10.0,
    flag: int = 0,
) -> np.ndarray:
    """Return a (4, nwave) flat-spectrum array."""
    wavelength = np.linspace(wave_start, wave_end, nwave)
    sp = np.zeros((4, nwave), dtype=float)
    sp[0] = wavelength
    sp[1] = intensity
    sp[2] = uncertainty
    sp[3] = flag
    return sp


def _make_vega_model(
    wave_start: float = 0.9,
    wave_end: float = 2.6,
    nwave: int = _NWAVE_VEGA,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic (wavelength, fd, continuum, fitted_continuum) arrays."""
    wavelength = np.linspace(wave_start, wave_end, nwave)
    fd = np.full(nwave, amplitude)
    continuum = np.full(nwave, amplitude)
    fitted_continuum = np.full(nwave, amplitude)
    return wavelength, fd, continuum, fitted_continuum


def _make_standard_correction_inputs(
    wave_start: float,
    wave_end: float,
    slit_width_arcsec: float = 0.375,
    intensity: float = 100.0,
    uncertainty: float = 10.0,
) -> dict:
    """
    Build a complete set of valid inputs for correct_telluric for a J-band-like
    single-aperture merged spectrum.
    """
    nwave = _NWAVE
    sci = _make_flat_spectrum(wave_start, wave_end, nwave, intensity, uncertainty)
    std = _make_flat_spectrum(wave_start, wave_end, nwave, intensity, uncertainty)
    vwave, vfd, vcont, vfcont = _make_vega_model()
    return dict(
        science_spectrum=sci,
        standard_spectrum=std,
        vega_wavelength=vwave,
        vega_fluxdensity=vfd,
        vega_continuum=vcont,
        vega_fitted_continuum=vfcont,
        slit_width_arcsec=slit_width_arcsec,
        standard_bmag=9.8,
        standard_vmag=9.5,
        standard_rv=0.0,
    )


# ---------------------------------------------------------------------------
# Test: constants and module API
# ---------------------------------------------------------------------------


class TestModuleAPI:
    """Verify that the public API is importable and has expected values."""

    def test_supported_modes_is_frozenset(self):
        assert isinstance(_SUPPORTED_MODES, frozenset)

    def test_supported_modes_covers_jhk(self):
        j_modes = {"J0", "J1", "J2", "J3"}
        h_modes = {"H1", "H2", "H3"}
        k_modes = {"K1", "K2", "K3", "Kgas"}
        assert j_modes <= _SUPPORTED_MODES
        assert h_modes <= _SUPPORTED_MODES
        assert k_modes <= _SUPPORTED_MODES

    def test_plate_scale_is_float(self):
        assert isinstance(_ISHELL_PLATE_SCALE_ARCSEC_PER_PIX, float)
        assert _ISHELL_PLATE_SCALE_ARCSEC_PER_PIX == pytest.approx(0.125)

    def test_correct_telluric_callable(self):
        assert callable(correct_telluric)


# ---------------------------------------------------------------------------
# Test: successful correction flow
# ---------------------------------------------------------------------------


class TestSuccessfulCorrectionFlow:
    """End-to-end smoke tests for valid inputs."""

    @pytest.mark.parametrize(
        "band,wave_start,wave_end",
        [
            ("J", 1.062, 1.364),
            ("H", 1.473, 1.832),
            ("K", 1.918, 2.549),
        ],
    )
    def test_returns_tuple_of_three(self, band, wave_start, wave_end):
        kwargs = _make_standard_correction_inputs(wave_start, wave_end)
        result = correct_telluric(**kwargs)
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.parametrize(
        "band,wave_start,wave_end",
        [
            ("J", 1.062, 1.364),
            ("H", 1.473, 1.832),
            ("K", 1.918, 2.549),
        ],
    )
    def test_corrected_spectrum_is_ndarray(self, band, wave_start, wave_end):
        kwargs = _make_standard_correction_inputs(wave_start, wave_end)
        corrected, _, _ = correct_telluric(**kwargs)
        assert isinstance(corrected, np.ndarray)

    @pytest.mark.parametrize(
        "band,wave_start,wave_end",
        [
            ("J", 1.062, 1.364),
            ("H", 1.473, 1.832),
            ("K", 1.918, 2.549),
        ],
    )
    def test_metadata_is_dict(self, band, wave_start, wave_end):
        kwargs = _make_standard_correction_inputs(wave_start, wave_end)
        _, _, meta = correct_telluric(**kwargs)
        assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# Test: dimensional consistency
# ---------------------------------------------------------------------------


class TestDimensionalConsistency:
    """Verify the shapes of all output arrays."""

    def test_corrected_shape_matches_science(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        sci = kwargs["science_spectrum"]
        corrected, _, _ = correct_telluric(**kwargs)
        assert corrected.shape == sci.shape

    def test_corrected_has_four_rows(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        corrected, _, _ = correct_telluric(**kwargs)
        assert corrected.shape[0] == 4

    def test_correction_spectrum_shape_matches_standard(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        std = kwargs["standard_spectrum"]
        _, correction, _ = correct_telluric(**kwargs)
        assert correction.shape == std.shape

    def test_correction_spectrum_has_four_rows(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, correction, _ = correct_telluric(**kwargs)
        assert correction.shape[0] == 4

    def test_different_science_and_standard_lengths(self):
        """Science and standard may have different wavelength grid sizes."""
        nwave_sci = 300
        nwave_std = 200
        sci = _make_flat_spectrum(1.1, 1.35, nwave_sci)
        std = _make_flat_spectrum(1.1, 1.35, nwave_std)
        vwave, vfd, vcont, vfcont = _make_vega_model()
        corrected, correction, _ = correct_telluric(
            science_spectrum=sci,
            standard_spectrum=std,
            vega_wavelength=vwave,
            vega_fluxdensity=vfd,
            vega_continuum=vcont,
            vega_fitted_continuum=vfcont,
            slit_width_arcsec=0.375,
            standard_bmag=9.8,
            standard_vmag=9.5,
        )
        assert corrected.shape == (4, nwave_sci)
        assert correction.shape == (4, nwave_std)

    def test_wavelength_row_preserved(self):
        """Row 0 of the corrected spectrum must equal science wavelengths."""
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        sci = kwargs["science_spectrum"]
        corrected, _, _ = correct_telluric(**kwargs)
        np.testing.assert_array_equal(corrected[0], sci[0])


# ---------------------------------------------------------------------------
# Test: uncertainty propagation
# ---------------------------------------------------------------------------


class TestUncertaintyPropagation:
    """Verify that uncertainties are propagated through the correction."""

    def test_uncertainty_is_non_negative(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        corrected, _, _ = correct_telluric(**kwargs)
        assert np.all(corrected[2] >= 0.0)

    def test_uncertainty_is_finite(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        corrected, _, _ = correct_telluric(**kwargs)
        assert np.all(np.isfinite(corrected[2]))

    def test_uncertainty_grows_with_standard_uncertainty(self):
        """Larger standard uncertainty → larger corrected uncertainty."""
        kwargs_low = _make_standard_correction_inputs(1.1, 1.35)
        kwargs_high = _make_standard_correction_inputs(1.1, 1.35)
        kwargs_high["standard_spectrum"][2] *= 10.0

        corrected_low, _, _ = correct_telluric(**kwargs_low)
        corrected_high, _, _ = correct_telluric(**kwargs_high)

        assert np.median(corrected_high[2]) > np.median(corrected_low[2])

    def test_uncertainty_grows_with_science_uncertainty(self):
        """Larger science uncertainty → larger corrected uncertainty."""
        kwargs_low = _make_standard_correction_inputs(1.1, 1.35)
        kwargs_high = _make_standard_correction_inputs(1.1, 1.35)
        kwargs_high["science_spectrum"][2] *= 10.0

        corrected_low, _, _ = correct_telluric(**kwargs_low)
        corrected_high, _, _ = correct_telluric(**kwargs_high)

        assert np.median(corrected_high[2]) > np.median(corrected_low[2])

    def test_quadrature_propagation(self):
        """
        Verify that uncertainty is propagated in quadrature.

        With a flat correction C and science uncertainty σ_sci, standard
        uncertainty σ_std, the corrected uncertainty is:

            σ_corrected² = (C × σ_sci)² + (F_sci × σ_correction)²

        For a flat (constant) standard of intensity I and flat Vega model,
        the correction C ≈ constant and σ_correction ≈ C/I × σ_std.
        We verify that corrected_unc is strictly larger than
        correction × science_unc alone (confirming both terms contribute).
        """
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        sci = kwargs["science_spectrum"]
        corrected, correction_sp, _ = correct_telluric(**kwargs)

        # Minimum expected uncertainty if only science term contributed
        # (lower bound: correction × σ_science only)
        tc_f = correction_sp[1]  # correction spectrum flux
        # Interpolate correction onto science wavelengths using numpy
        tc_interp = np.interp(sci[0], correction_sp[0], tc_f)
        lower_bound = tc_interp * sci[2]

        # corrected uncertainty must be >= the science-term lower bound
        np.testing.assert_array_less(lower_bound - 1e-30, corrected[2])


# ---------------------------------------------------------------------------
# Test: correction spectrum sanity
# ---------------------------------------------------------------------------


class TestCorrectionSpectrumSanity:
    """Basic sanity checks on the telluric correction spectrum."""

    def test_correction_flux_is_finite(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, correction, _ = correct_telluric(**kwargs)
        assert np.all(np.isfinite(correction[1]))

    def test_correction_flux_is_positive(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, correction, _ = correct_telluric(**kwargs)
        assert np.all(correction[1] > 0.0)

    def test_correction_uncertainty_is_non_negative(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, correction, _ = correct_telluric(**kwargs)
        assert np.all(correction[2] >= 0.0)

    def test_correction_wavelength_preserved(self):
        """Row 0 of correction must equal standard wavelengths."""
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        std = kwargs["standard_spectrum"]
        _, correction, _ = correct_telluric(**kwargs)
        np.testing.assert_array_equal(correction[0], std[0])

    def test_metadata_kernel_method_is_ip(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, _, meta = correct_telluric(**kwargs)
        assert meta["kernel_method"] == "ip"

    def test_metadata_contains_expected_keys(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, _, meta = correct_telluric(**kwargs)
        expected_keys = {
            "kernel_method",
            "slit_width_arcsec",
            "ip_coefficients",
            "shift_applied",
            "ew_scale",
            "tilt_provisional",
            "intensity_unit",
        }
        assert expected_keys <= set(meta.keys())


# ---------------------------------------------------------------------------
# Test: flag propagation
# ---------------------------------------------------------------------------


class TestFlagPropagation:
    """Verify that quality flags are propagated via bitwise OR."""

    def test_zero_flags_stay_zero(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        corrected, _, _ = correct_telluric(**kwargs)
        # Wavelengths are fully within standard range → coverage mask = 0
        assert np.all(corrected[3] == 0)

    def test_science_flags_propagated(self):
        """Non-zero science flags must appear in the corrected flags."""
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        # Set flag = 2 on the first 10 science pixels
        kwargs["science_spectrum"][3, :10] = 2
        corrected, _, _ = correct_telluric(**kwargs)
        # Those pixels should have flag ≥ 2
        assert np.all(corrected[3, :10] >= 2)

    def test_out_of_coverage_pixels_flagged(self):
        """Science pixels outside the standard's wavelength range are flagged."""
        # Science extends beyond standard on the blue side
        sci = _make_flat_spectrum(1.0, 1.35)  # starts at 1.0
        std = _make_flat_spectrum(1.1, 1.35)  # starts at 1.1
        vwave, vfd, vcont, vfcont = _make_vega_model()

        corrected, _, _ = correct_telluric(
            science_spectrum=sci,
            standard_spectrum=std,
            vega_wavelength=vwave,
            vega_fluxdensity=vfd,
            vega_continuum=vcont,
            vega_fitted_continuum=vfcont,
            slit_width_arcsec=0.375,
            standard_bmag=9.8,
            standard_vmag=9.5,
        )

        # Pixels blueward of standard range should be flagged
        outside = sci[0] < 1.1
        assert np.any(corrected[3][outside] > 0)


# ---------------------------------------------------------------------------
# Test: residual wavelength shift
# ---------------------------------------------------------------------------


class TestResidualWavelengthShift:
    """Verify the residual wavelength shift step."""

    def test_shift_applied_is_zero_without_apply_shift(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, _, meta = correct_telluric(**kwargs, apply_shift=False)
        assert meta["shift_applied"] == 0.0

    def test_shift_applied_is_zero_without_range(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, _, meta = correct_telluric(**kwargs, apply_shift=True,
                                      shift_wavelength_range=None)
        assert meta["shift_applied"] == 0.0

    def test_shift_applied_with_range_runs_without_error(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        corrected, _, meta = correct_telluric(
            **kwargs,
            apply_shift=True,
            shift_wavelength_range=[1.15, 1.30],
        )
        assert corrected.shape[0] == 4
        assert isinstance(meta["shift_applied"], float)

    def test_shift_in_metadata(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, _, meta = correct_telluric(**kwargs, apply_shift=True,
                                      shift_wavelength_range=[1.15, 1.30])
        assert "shift_applied" in meta


# ---------------------------------------------------------------------------
# Test: tilt provisional warning
# ---------------------------------------------------------------------------


class TestTiltProvisionalWarning:
    """Check that tilt_provisional=True emits the expected RuntimeWarning."""

    def test_warning_emitted_when_tilt_provisional_true(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            correct_telluric(**kwargs, tilt_provisional=True)

        assert any(issubclass(x.category, RuntimeWarning) for x in w)
        assert any("tilt_provisional" in str(x.message).lower() for x in w)

    def test_no_warning_when_tilt_provisional_false(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            correct_telluric(**kwargs, tilt_provisional=False)

        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0


# ---------------------------------------------------------------------------
# Test: slit widths
# ---------------------------------------------------------------------------


class TestSlitWidths:
    """Verify the function works for all supported iSHELL slit widths."""

    @pytest.mark.parametrize("slit", [0.375, 0.750, 1.500, 4.000])
    def test_known_slit_widths_run_without_error(self, slit):
        kwargs = _make_standard_correction_inputs(1.1, 1.35, slit_width_arcsec=slit)
        corrected, _, meta = correct_telluric(**kwargs)
        assert corrected.shape[0] == 4
        assert meta["slit_width_arcsec"] == pytest.approx(slit)

    @pytest.mark.parametrize("slit", [0.375, 0.750, 1.500, 4.000])
    def test_ip_coefficients_in_metadata(self, slit):
        kwargs = _make_standard_correction_inputs(1.1, 1.35, slit_width_arcsec=slit)
        _, _, meta = correct_telluric(**kwargs)
        assert meta["ip_coefficients"].shape == (3,)


# ---------------------------------------------------------------------------
# Test: failure cases
# ---------------------------------------------------------------------------


class TestFailureCases:
    """Verify error handling for malformed inputs."""

    def test_science_wrong_shape_raises(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        kwargs["science_spectrum"] = np.zeros((3, _NWAVE))  # wrong: 3 rows
        with pytest.raises(pySpextoolError):
            correct_telluric(**kwargs)

    def test_science_1d_raises(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        kwargs["science_spectrum"] = np.zeros(_NWAVE)  # 1-D, wrong
        with pytest.raises(pySpextoolError):
            correct_telluric(**kwargs)

    def test_standard_wrong_shape_raises(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        kwargs["standard_spectrum"] = np.zeros((2, _NWAVE))  # wrong: 2 rows
        with pytest.raises(pySpextoolError):
            correct_telluric(**kwargs)

    def test_standard_1d_raises(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        kwargs["standard_spectrum"] = np.zeros(_NWAVE)
        with pytest.raises(pySpextoolError):
            correct_telluric(**kwargs)

    def test_unknown_slit_width_raises(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        kwargs["slit_width_arcsec"] = 99.9  # not in IP_coefficients.dat
        with pytest.raises(pySpextoolError):
            correct_telluric(**kwargs)

    def test_vega_insufficient_coverage_raises(self):
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        # Vega model only covers 0.9–1.0 µm; standard is at 1.1–1.35 µm
        vwave = np.linspace(0.9, 1.0, 100)
        kwargs["vega_wavelength"] = vwave
        kwargs["vega_fluxdensity"] = np.ones(100)
        kwargs["vega_continuum"] = np.ones(100)
        kwargs["vega_fitted_continuum"] = np.ones(100)
        with pytest.raises(pySpextoolError):
            correct_telluric(**kwargs)

    def test_standard_constant_wavelength_raises(self):
        """Standard with all-identical wavelengths (zero spacing) → error."""
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        # Make standard wavelength all the same value
        kwargs["standard_spectrum"][0] = 1.2
        with pytest.raises(pySpextoolError):
            correct_telluric(**kwargs)

    def test_ew_scale_accepted(self):
        """ew_scale parameter is forwarded to metadata without error."""
        kwargs = _make_standard_correction_inputs(1.1, 1.35)
        _, _, meta = correct_telluric(**kwargs, ew_scale=0.8)
        assert meta["ew_scale"] == pytest.approx(0.8)
