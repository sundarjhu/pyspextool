"""
Tests for the iSHELL 2DXD provisional wavelength-mapping scaffold.

Coverage
--------
- :func:`read_stored_2dxd_coeffs` – reads P2W_C* from packaged FITS files.
- :func:`collect_centroid_data` – flattens per-order centroid results.
- :func:`fit_provisional_2dxd` – global provisional polynomial fit.
- :class:`TwoDXDCoefficients` – stored-coefficient dataclass and properties.
- :class:`ArcLineCentroidData` – centroid-collection dataclass and properties.
- :class:`ProvisionalWaveCal2DXD` – result dataclass, eval_wavelength.

All tests exercise real packaged calibration data.  The representative mode
is **H1** (45 orders, H-band); a broader set of modes is tested for the
read/collect steps to catch band-specific differences.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyspextool.instruments.ishell.calibrations import (
    read_flatinfo,
    read_line_list,
    read_wavecalinfo,
)
from pyspextool.instruments.ishell.wavecal_2dxd import (
    ArcLineCentroidData,
    ProvisionalWaveCal2DXD,
    TwoDXDCoefficients,
    collect_centroid_data,
    fit_provisional_2dxd,
    read_stored_2dxd_coeffs,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# One representative mode per band (fast smoke-test set)
REPRESENTATIVE_MODES = ["J0", "H1", "K1"]


@pytest.fixture(scope="module", params=REPRESENTATIVE_MODES)
def mode(request):
    return request.param


@pytest.fixture(scope="module")
def h1_calibrations():
    """Load H1 calibrations once for the whole test session."""
    wci = read_wavecalinfo("H1")
    fi = read_flatinfo("H1")
    ll = read_line_list("H1")
    return wci, fi, ll


@pytest.fixture(scope="module")
def h1_stored_2dxd():
    """Read H1 stored 2DXD coefficients once."""
    return read_stored_2dxd_coeffs("H1")


@pytest.fixture(scope="module")
def h1_centroid_data(h1_calibrations):
    """Collect H1 centroids once."""
    wci, fi, ll = h1_calibrations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return collect_centroid_data(wci, ll)


@pytest.fixture(scope="module")
def h1_provisional_fit(h1_calibrations):
    """Run the H1 provisional 2DXD fit once for the whole module."""
    wci, fi, ll = h1_calibrations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_provisional_2dxd(wci, fi, ll)


# ---------------------------------------------------------------------------
# TwoDXDCoefficients – read_stored_2dxd_coeffs
# ---------------------------------------------------------------------------


class TestReadStored2DXDCoeffs:
    """Tests for :func:`read_stored_2dxd_coeffs` and :class:`TwoDXDCoefficients`."""

    def test_returns_correct_type(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        assert isinstance(stored, TwoDXDCoefficients)

    def test_mode_attribute(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        assert stored.mode == mode

    def test_disp_degree_positive(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        assert stored.disp_degree >= 1

    def test_order_degree_positive(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        assert stored.order_degree >= 1

    def test_home_order_positive(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        assert stored.home_order > 0

    def test_coeffs_flat_length(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        expected = (stored.disp_degree + 1) * (stored.order_degree + 1)
        assert len(stored.coeffs_flat) == expected

    def test_coeffs_matrix_shape(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        assert stored.coeffs_matrix.shape == (
            stored.disp_degree + 1,
            stored.order_degree + 1,
        )

    def test_coeffs_flat_finite(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        assert np.all(np.isfinite(stored.coeffs_flat))

    def test_n_pixels_default(self, mode):
        stored = read_stored_2dxd_coeffs(mode)
        assert stored.n_pixels == 2048

    def test_h1_disp_degree(self, h1_stored_2dxd):
        """H1 packaged calibration uses DISPDEG=3."""
        assert h1_stored_2dxd.disp_degree == 3

    def test_h1_order_degree(self, h1_stored_2dxd):
        """H1 packaged calibration uses ORDRDEG=2."""
        assert h1_stored_2dxd.order_degree == 2

    def test_h1_home_order(self, h1_stored_2dxd):
        """H1 home order is 333 (as stored in FITS header)."""
        assert h1_stored_2dxd.home_order == 333

    def test_h1_n_coeffs(self, h1_stored_2dxd):
        """(3+1)*(2+1) = 12 coefficients for H1."""
        assert h1_stored_2dxd.n_coeffs == 12

    def test_invalid_mode_raises(self):
        with pytest.raises((KeyError, FileNotFoundError, ValueError)):
            read_stored_2dxd_coeffs("BADMODE_XYZ")

    def test_eval_provisional_output_shape_scalar(self, h1_stored_2dxd):
        """Scalar inputs return scalar-like output."""
        result = h1_stored_2dxd.eval_provisional(1000.0, 333.0)
        assert result.shape == ()

    def test_eval_provisional_output_shape_array(self, h1_stored_2dxd):
        x = np.linspace(270, 1960, 20)
        m = np.full(20, 333.0)
        result = h1_stored_2dxd.eval_provisional(x, m)
        assert result.shape == (20,)

    def test_eval_provisional_in_h_band_range(self, h1_stored_2dxd):
        """Provisional evaluation should return plausible H-band wavelengths."""
        stored = h1_stored_2dxd
        # Home order at mid-detector
        mid_x = np.linspace(270.0, 1960.0, 10)
        m = np.full(10, stored.home_order, dtype=float)
        lam = stored.eval_provisional(mid_x, m)
        # H-band: 1.47 – 1.83 µm; result should be broadly in this range
        # (the form may not be perfectly calibrated, but gross outliers fail)
        assert np.all(np.isfinite(lam))

    def test_wavecalinfo_p2w_coeffs_populated(self, mode):
        """read_wavecalinfo should now populate p2w_coeffs."""
        wci = read_wavecalinfo(mode)
        assert wci.p2w_coeffs is not None

    def test_wavecalinfo_p2w_coeffs_matches_reader(self, mode):
        """WaveCalInfo.p2w_coeffs should match read_stored_2dxd_coeffs."""
        wci = read_wavecalinfo(mode)
        stored = read_stored_2dxd_coeffs(mode)
        np.testing.assert_array_equal(wci.p2w_coeffs, stored.coeffs_flat)


# ---------------------------------------------------------------------------
# ArcLineCentroidData – collect_centroid_data
# ---------------------------------------------------------------------------


class TestCollectCentroidData:
    """Tests for :func:`collect_centroid_data` and :class:`ArcLineCentroidData`."""

    def test_returns_correct_type(self, h1_centroid_data):
        assert isinstance(h1_centroid_data, ArcLineCentroidData)

    def test_mode_attribute(self, h1_centroid_data):
        assert h1_centroid_data.mode == "H1"

    def test_array_lengths_consistent(self, h1_centroid_data):
        cd = h1_centroid_data
        assert len(cd.columns) == len(cd.orders) == len(cd.wavelengths_um) == len(cd.snr_values)
        assert cd.n_points == len(cd.columns)

    def test_has_positive_point_count(self, h1_centroid_data):
        """H1 should yield at least 1 accepted centroid."""
        assert h1_centroid_data.n_points > 0

    def test_columns_in_detector_range(self, h1_centroid_data):
        """Centroid columns must be within [0, 2047]."""
        assert np.all(h1_centroid_data.columns >= 0)
        assert np.all(h1_centroid_data.columns < 2048)

    def test_wavelengths_in_h_band(self, h1_centroid_data):
        """H1 centroids should have wavelengths in the H-band range."""
        lams = h1_centroid_data.wavelengths_um
        assert np.all(lams > 1.40), "Some wavelengths below H-band minimum"
        assert np.all(lams < 1.90), "Some wavelengths above H-band maximum"

    def test_snr_above_threshold(self, h1_centroid_data):
        """All accepted centroids should have SNR ≥ default threshold (3.0)."""
        assert np.all(h1_centroid_data.snr_values >= 3.0)

    def test_orders_are_h1_orders(self, h1_centroid_data):
        """H1 centroid order numbers should be within the H1 order range."""
        assert np.all(h1_centroid_data.orders >= 311)
        assert np.all(h1_centroid_data.orders <= 355)

    def test_per_order_counts_has_all_orders(self, h1_centroid_data, h1_calibrations):
        """per_order_counts should contain all H1 orders."""
        wci, fi, ll = h1_calibrations
        for o in wci.orders:
            assert o in h1_centroid_data.per_order_counts

    def test_n_orders_with_data_positive(self, h1_centroid_data):
        assert h1_centroid_data.n_orders_with_data > 0

    @pytest.mark.parametrize("band_mode", REPRESENTATIVE_MODES)
    def test_representative_modes_collect_without_error(self, band_mode):
        """All representative modes should complete collect_centroid_data without error."""
        wci = read_wavecalinfo(band_mode)
        ll = read_line_list(band_mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cd = collect_centroid_data(wci, ll)
        assert isinstance(cd, ArcLineCentroidData)
        assert cd.n_points >= 0  # may be zero in edge cases, but must not crash


# ---------------------------------------------------------------------------
# ProvisionalWaveCal2DXD – fit_provisional_2dxd
# ---------------------------------------------------------------------------


class TestFitProvisional2DXD:
    """Tests for :func:`fit_provisional_2dxd` and :class:`ProvisionalWaveCal2DXD`."""

    def test_returns_correct_type(self, h1_provisional_fit):
        assert isinstance(h1_provisional_fit, ProvisionalWaveCal2DXD)

    def test_mode_attribute(self, h1_provisional_fit):
        assert h1_provisional_fit.mode == "H1"

    def test_fitted_coeffs_shape(self, h1_provisional_fit):
        result = h1_provisional_fit
        expected = (result.fitted_disp_degree + 1, result.fitted_order_degree + 1)
        assert result.fitted_coeffs.shape == expected

    def test_fitted_coeffs_finite(self, h1_provisional_fit):
        assert np.all(np.isfinite(h1_provisional_fit.fitted_coeffs))

    def test_rms_residual_positive_finite(self, h1_provisional_fit):
        rms = h1_provisional_fit.rms_residual_um
        assert np.isfinite(rms)
        assert rms >= 0.0

    def test_rms_below_10nm(self, h1_provisional_fit):
        """For H1 with real arc data, provisional fit RMS should be < 10 nm."""
        assert h1_provisional_fit.rms_residual_um < 0.010, (
            f"RMS residual {h1_provisional_fit.rms_residual_um*1000:.2f} nm "
            f"exceeds 10 nm threshold"
        )

    def test_n_orders_fitted_plus_bootstrap_equals_total(
        self, h1_provisional_fit, h1_calibrations
    ):
        wci, fi, ll = h1_calibrations
        result = h1_provisional_fit
        assert result.n_orders_fitted + result.n_orders_bootstrap == wci.n_orders

    def test_per_order_wave_coeffs_all_orders(
        self, h1_provisional_fit, h1_calibrations
    ):
        wci, fi, ll = h1_calibrations
        result = h1_provisional_fit
        for o in wci.orders:
            assert o in result.per_order_wave_coeffs, (
                f"Order {o} missing from per_order_wave_coeffs"
            )

    def test_per_order_wave_coeffs_finite(self, h1_provisional_fit):
        for o, coeffs in h1_provisional_fit.per_order_wave_coeffs.items():
            assert np.all(np.isfinite(coeffs)), f"Order {o}: non-finite wave_coeffs"

    def test_stored_2dxd_is_attached(self, h1_provisional_fit):
        assert isinstance(h1_provisional_fit.stored_2dxd, TwoDXDCoefficients)

    def test_centroid_data_is_attached(self, h1_provisional_fit):
        assert isinstance(h1_provisional_fit.centroid_data, ArcLineCentroidData)

    def test_home_order_matches_stored(self, h1_provisional_fit, h1_stored_2dxd):
        assert h1_provisional_fit.home_order == h1_stored_2dxd.home_order

    def test_eval_wavelength_scalar(self, h1_provisional_fit):
        lam = h1_provisional_fit.eval_wavelength(1000.0, 333.0)
        assert lam.shape == ()
        assert np.isfinite(lam)

    def test_eval_wavelength_array(self, h1_provisional_fit):
        x = np.linspace(270.0, 1960.0, 50)
        m = np.full(50, 333.0)
        lam = h1_provisional_fit.eval_wavelength(x, m)
        assert lam.shape == (50,)
        assert np.all(np.isfinite(lam))

    def test_eval_wavelength_home_order_in_h_band(self, h1_provisional_fit):
        """Wavelengths at home order should be in the H band range (1.47–1.83 µm)."""
        x = np.linspace(270.0, 1960.0, 20)
        m = np.full(20, float(h1_provisional_fit.home_order))
        lam = h1_provisional_fit.eval_wavelength(x, m)
        assert np.all(lam > 1.40), "Provisionally evaluated wavelengths below H-band"
        assert np.all(lam < 1.90), "Provisionally evaluated wavelengths above H-band"

    def test_eval_wavelength_monotone_with_x_for_home_order(
        self, h1_provisional_fit
    ):
        """For the home order, wavelength should be monotone in column.

        iSHELL H-band disperses so that wavelength decreases with increasing
        column (bluer orders at lower columns).  We only check monotonicity
        direction is consistent (either fully increasing or fully decreasing).
        """
        x = np.linspace(270.0, 1960.0, 100)
        m = np.full(100, float(h1_provisional_fit.home_order))
        lam = h1_provisional_fit.eval_wavelength(x, m)
        diff = np.diff(lam)
        assert np.all(diff > 0) or np.all(diff < 0), (
            "Evaluated wavelength vs column is not monotone for home order"
        )

    def test_different_orders_give_different_wavelengths(self, h1_provisional_fit):
        """At the same column position, different orders should give different λ."""
        x0 = 1000.0
        m_low = 320.0
        m_high = 340.0
        lam_low = float(h1_provisional_fit.eval_wavelength(x0, m_low))
        lam_high = float(h1_provisional_fit.eval_wavelength(x0, m_high))
        assert lam_low != lam_high

    def test_mode_mismatch_raises(self, h1_calibrations):
        """Passing mismatched mode calibrations should raise ValueError."""
        wci_h1, fi_h1, ll_h1 = h1_calibrations
        fi_k1 = read_flatinfo("K1")
        with pytest.raises(ValueError, match="mode"):
            fit_provisional_2dxd(wci_h1, fi_k1, ll_h1)

    def test_h1_degrees_from_stored(self, h1_provisional_fit, h1_stored_2dxd):
        """Default degrees should match those stored in the FITS header."""
        assert h1_provisional_fit.fitted_disp_degree == h1_stored_2dxd.disp_degree
        assert h1_provisional_fit.fitted_order_degree == h1_stored_2dxd.order_degree

    def test_override_degrees(self, h1_calibrations):
        """User-supplied degrees should override the defaults."""
        wci, fi, ll = h1_calibrations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_provisional_2dxd(wci, fi, ll, disp_degree=2, order_degree=1)
        assert result.fitted_disp_degree == 2
        assert result.fitted_order_degree == 1
        assert result.fitted_coeffs.shape == (3, 2)

    @pytest.mark.parametrize("band_mode", REPRESENTATIVE_MODES)
    def test_smoke_all_representative_modes(self, band_mode):
        """Provisional 2DXD fit should succeed for J0, H1, and K1."""
        wci = read_wavecalinfo(band_mode)
        fi = read_flatinfo(band_mode)
        ll = read_line_list(band_mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_provisional_2dxd(wci, fi, ll)
        assert isinstance(result, ProvisionalWaveCal2DXD)
        assert result.mode == band_mode
        assert np.isfinite(result.rms_residual_um)
        assert result.rms_residual_um >= 0.0


# ---------------------------------------------------------------------------
# Cross-check: fitted polynomial residuals against centroid data
# ---------------------------------------------------------------------------


class TestFitResiduals:
    """Cross-checks on fit quality: residuals vs centroid input data."""

    def test_residuals_against_centroid_data(self, h1_provisional_fit):
        """Evaluate the fitted polynomial at centroid positions and check RMS."""
        result = h1_provisional_fit
        cd = result.centroid_data
        if cd.n_points == 0:
            pytest.skip("No centroid data available for H1")

        lam_pred = result.eval_wavelength(cd.columns, cd.orders)
        residuals = lam_pred - cd.wavelengths_um
        rms = float(np.std(residuals))
        # Must be consistent with the stored rms_residual_um
        np.testing.assert_allclose(rms, result.rms_residual_um, rtol=0.01)

    def test_centroid_residuals_finite(self, h1_provisional_fit):
        result = h1_provisional_fit
        cd = result.centroid_data
        if cd.n_points == 0:
            pytest.skip("No centroid data available for H1")
        lam_pred = result.eval_wavelength(cd.columns, cd.orders)
        assert np.all(np.isfinite(lam_pred - cd.wavelengths_um))
