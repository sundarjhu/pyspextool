"""
Tests for the corrected iSHELL 2DXD provisional wavelength-mapping scaffold.

Architecture under test
-----------------------
The corrected data flow is::

    build_arc_trace_result_from_wavecalinfo()   ← arc_tracing PRIMARY path
        ↓
    ArcTraceResult  (TracedArcLine objects with detector positions)
        ↓
    collect_centroid_data_from_arc_traces()
        ↓
    ArcLineCentroidData  (flat x, m, λ arrays)
        ↓
    fit_provisional_2dxd()
        ↓
    ProvisionalWaveCal2DXD

Coverage
--------
- :mod:`arc_tracing`: TracedArcLine, ArcTraceResult,
  trace_arc_lines_from_1d_spectrum, build_arc_trace_result_from_wavecalinfo,
  trace_arc_lines_from_2d_image stub.
- :func:`read_stored_2dxd_coeffs` – P2W_C* from packaged FITS (reference role).
- :func:`collect_centroid_data_from_arc_traces` – PRIMARY path.
- :func:`fit_provisional_2dxd` – new signature (takes ArcTraceResult).
- :func:`collect_centroid_data` – AUXILIARY legacy path.
- :class:`TwoDXDCoefficients`, :class:`ArcLineCentroidData`,
  :class:`ProvisionalWaveCal2DXD` – dataclasses and properties.

Data sources
------------
- Packaged H1 calibration files (via read_wavecalinfo / read_flatinfo /
  read_line_list) are used by all tests.  The arc-tracing tests use the
  wavecalinfo plane-1 data as a 1-D centerline arc-spectrum proxy (source
  = ``"wavecalinfo_plane1"``).
- No real 2-D raw arc FITS frames are available yet; tests for the real-data
  path are represented by the stub test below.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyspextool.instruments.ishell.arc_tracing import (
    ArcTraceResult,
    TracedArcLine,
    build_arc_trace_result_from_wavecalinfo,
    trace_arc_lines_from_1d_spectrum,
    trace_arc_lines_from_2d_image,
)
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
    collect_centroid_data_from_arc_traces,
    fit_provisional_2dxd,
    read_stored_2dxd_coeffs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def h1_arc_traces(h1_calibrations):
    """Build H1 ArcTraceResult once (primary path via wavecalinfo plane-1 proxy)."""
    wci, fi, ll = h1_calibrations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_arc_trace_result_from_wavecalinfo(wci, ll, fi)


@pytest.fixture(scope="module")
def h1_centroid_data(h1_arc_traces):
    """Collect H1 centroid data from arc traces (primary path)."""
    return collect_centroid_data_from_arc_traces(h1_arc_traces)


@pytest.fixture(scope="module")
def h1_provisional_fit(h1_arc_traces, h1_calibrations):
    """Run the H1 provisional 2DXD fit once (primary arc_tracing path)."""
    wci, fi, ll = h1_calibrations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_provisional_2dxd(h1_arc_traces, wci, fi, ll)


# ---------------------------------------------------------------------------
# arc_tracing: TracedArcLine and ArcTraceResult
# ---------------------------------------------------------------------------


class TestTracedArcLine:
    """Tests for :class:`~.arc_tracing.TracedArcLine` dataclass."""

    def test_is_frozen(self, h1_arc_traces):
        """TracedArcLine instances should be immutable."""
        if not h1_arc_traces.lines:
            pytest.skip("No traced lines available")
        ln = h1_arc_traces.lines[0]
        with pytest.raises((AttributeError, TypeError)):
            ln.order = 999  # type: ignore[misc]

    def test_fields_finite(self, h1_arc_traces):
        """All numeric fields of every traced line should be finite."""
        for ln in h1_arc_traces.lines:
            assert np.isfinite(ln.wavelength_um)
            assert np.isfinite(ln.seed_col)
            assert np.isfinite(ln.centroid_col)
            assert np.isfinite(ln.centerline_row)
            assert np.isfinite(ln.snr)
            assert np.isfinite(ln.fit_residual_pix)

    def test_centroid_col_vs_seed_col_small_residual(self, h1_arc_traces):
        """The fit_residual_pix should be < half_window (15 px) for all lines."""
        for ln in h1_arc_traces.lines:
            assert abs(ln.centroid_col - ln.seed_col) == pytest.approx(
                ln.fit_residual_pix, rel=1e-9
            )
            assert ln.fit_residual_pix < 15.0, (
                f"Order {ln.order}: fit_residual_pix={ln.fit_residual_pix:.2f} "
                "exceeds half-window (15 px)"
            )

    def test_snr_above_minimum(self, h1_arc_traces):
        """All accepted lines should have SNR ≥ default minimum (3.0)."""
        for ln in h1_arc_traces.lines:
            assert ln.snr >= 3.0

    def test_species_is_string(self, h1_arc_traces):
        """Species label should be a string for all lines (may be empty)."""
        for ln in h1_arc_traces.lines:
            assert isinstance(ln.species, str)


class TestArcTraceResult:
    """Tests for :class:`~.arc_tracing.ArcTraceResult` dataclass."""

    def test_returns_correct_type(self, h1_arc_traces):
        assert isinstance(h1_arc_traces, ArcTraceResult)

    def test_mode_attribute(self, h1_arc_traces):
        assert h1_arc_traces.mode == "H1"

    def test_source_is_wavecalinfo_plane1(self, h1_arc_traces):
        """Proxy path should set source to 'wavecalinfo_plane1'."""
        assert h1_arc_traces.source == "wavecalinfo_plane1"

    def test_n_lines_positive(self, h1_arc_traces):
        assert h1_arc_traces.n_lines > 0

    def test_n_orders_with_lines_positive(self, h1_arc_traces):
        assert h1_arc_traces.n_orders_with_lines > 0

    def test_per_order_counts_all_h1_orders(self, h1_arc_traces, h1_calibrations):
        """per_order_counts should include all H1 orders."""
        wci, fi, ll = h1_calibrations
        for o in wci.orders:
            assert o in h1_arc_traces.per_order_counts

    def test_get_order_lines(self, h1_arc_traces):
        """get_order_lines() should return a subset of the lines for that order."""
        for o in h1_arc_traces.orders:
            order_lines = h1_arc_traces.get_order_lines(o)
            assert all(ln.order == o for ln in order_lines)
            assert len(order_lines) == h1_arc_traces.per_order_counts[o]

    def test_to_flat_arrays_lengths(self, h1_arc_traces):
        """to_flat_arrays() should return equal-length arrays."""
        cols, orders, wavs, snrs = h1_arc_traces.to_flat_arrays()
        n = h1_arc_traces.n_lines
        assert len(cols) == n
        assert len(orders) == n
        assert len(wavs) == n
        assert len(snrs) == n

    def test_to_flat_arrays_centroid_col_used(self, h1_arc_traces):
        """to_flat_arrays() should return centroid_col (not seed_col) as x."""
        cols, _, _, _ = h1_arc_traces.to_flat_arrays()
        expected = np.array([ln.centroid_col for ln in h1_arc_traces.lines])
        np.testing.assert_array_equal(cols, expected)

    def test_columns_in_detector_range(self, h1_arc_traces):
        """Centroid columns should be within [0, 2047]."""
        cols, _, _, _ = h1_arc_traces.to_flat_arrays()
        assert np.all(cols >= 0) and np.all(cols < 2048)

    def test_wavelengths_in_h_band(self, h1_arc_traces):
        """H1 traced line wavelengths should be in the H-band range."""
        _, _, wavs, _ = h1_arc_traces.to_flat_arrays()
        assert np.all(wavs > 1.40) and np.all(wavs < 1.90)

    @pytest.mark.parametrize("band_mode", REPRESENTATIVE_MODES)
    def test_representative_modes_collect_without_error(self, band_mode):
        """build_arc_trace_result_from_wavecalinfo should work for J0, H1, K1."""
        wci = read_wavecalinfo(band_mode)
        ll = read_line_list(band_mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = build_arc_trace_result_from_wavecalinfo(wci, ll)
        assert isinstance(result, ArcTraceResult)
        assert result.mode == band_mode
        assert result.source == "wavecalinfo_plane1"


class TestTraceArcLinesFrom2dImageStub:
    """Tests for the 2-D arc tracing stub."""

    def test_raises_not_implemented(self, h1_calibrations):
        """trace_arc_lines_from_2d_image() must raise NotImplementedError."""
        wci, fi, ll = h1_calibrations
        with pytest.raises(NotImplementedError):
            trace_arc_lines_from_2d_image(None, fi, ll, wci)

    def test_error_message_mentions_raw_path(self, h1_calibrations):
        """The error message should mention the testdata path."""
        wci, fi, ll = h1_calibrations
        with pytest.raises(NotImplementedError, match="testdata"):
            trace_arc_lines_from_2d_image(None, fi, ll, wci)


class TestTraceArcLinesFrom1dSpectrum:
    """Tests for :func:`~.arc_tracing.trace_arc_lines_from_1d_spectrum`."""

    def test_returns_list(self, h1_calibrations):
        """Should return a list even with minimal data."""
        wci, fi, ll = h1_calibrations
        # Use first H1 order
        order_num = wci.orders[0]
        idx = 0
        wav = wci.data[idx, 0, :]
        arc = wci.data[idx, 1, :]
        valid = ~np.isnan(wav)
        entries = [e for e in ll.entries if e.order == order_num]
        x_start = int(wci.xranges[idx, 0])
        result = trace_arc_lines_from_1d_spectrum(
            arc_spectrum_1d=arc[valid],
            wavelength_grid_1d=wav[valid],
            x_start=x_start,
            line_list_entries=entries,
            centerline_row=0.0,
            order=order_num,
        )
        assert isinstance(result, list)
        assert all(isinstance(t, TracedArcLine) for t in result)

    def test_empty_input_returns_empty(self, h1_calibrations):
        """Empty arrays should return empty list without error."""
        result = trace_arc_lines_from_1d_spectrum(
            arc_spectrum_1d=np.array([]),
            wavelength_grid_1d=np.array([]),
            x_start=0,
            line_list_entries=[],
            centerline_row=0.0,
            order=333,
        )
        assert result == []

    def test_snr_filter_respected(self, h1_calibrations):
        """With very high snr_min, should return fewer (or zero) lines."""
        wci, fi, ll = h1_calibrations
        order_num = wci.orders[0]
        idx = 0
        wav = wci.data[idx, 0, :]
        arc = wci.data[idx, 1, :]
        valid = ~np.isnan(wav)
        entries = [e for e in ll.entries if e.order == order_num]
        x_start = int(wci.xranges[idx, 0])
        result_default = trace_arc_lines_from_1d_spectrum(
            arc[valid], wav[valid], x_start, entries, 0.0, order_num, snr_min=3.0
        )
        result_strict = trace_arc_lines_from_1d_spectrum(
            arc[valid], wav[valid], x_start, entries, 0.0, order_num, snr_min=1e9
        )
        assert len(result_strict) <= len(result_default)
        assert len(result_strict) == 0


# ---------------------------------------------------------------------------
# TwoDXDCoefficients – read_stored_2dxd_coeffs (reference role)
# ---------------------------------------------------------------------------


class TestReadStored2DXDCoeffs:
    """Tests for :func:`read_stored_2dxd_coeffs` and :class:`TwoDXDCoefficients`.

    These store the IDL P2W_C* polynomial coefficients in a reference/
    comparison role.  The coefficients are read from packaged FITS headers
    and attached to :class:`ProvisionalWaveCal2DXD` for comparison; they are
    not used as the primary source of the provisional fit.
    """

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
        assert h1_stored_2dxd.disp_degree == 3

    def test_h1_order_degree(self, h1_stored_2dxd):
        assert h1_stored_2dxd.order_degree == 2

    def test_h1_home_order(self, h1_stored_2dxd):
        assert h1_stored_2dxd.home_order == 333

    def test_h1_n_coeffs(self, h1_stored_2dxd):
        assert h1_stored_2dxd.n_coeffs == 12

    def test_invalid_mode_raises(self):
        with pytest.raises((KeyError, FileNotFoundError, ValueError)):
            read_stored_2dxd_coeffs("BADMODE_XYZ")

    def test_eval_provisional_output_shape_scalar(self, h1_stored_2dxd):
        result = h1_stored_2dxd.eval_provisional(1000.0, 333.0)
        assert result.shape == ()

    def test_eval_provisional_output_shape_array(self, h1_stored_2dxd):
        x = np.linspace(270, 1960, 20)
        m = np.full(20, 333.0)
        result = h1_stored_2dxd.eval_provisional(x, m)
        assert result.shape == (20,)

    def test_wavecalinfo_p2w_coeffs_populated(self, mode):
        """read_wavecalinfo should populate p2w_coeffs."""
        wci = read_wavecalinfo(mode)
        assert wci.p2w_coeffs is not None

    def test_wavecalinfo_p2w_coeffs_matches_reader(self, mode):
        """WaveCalInfo.p2w_coeffs should match read_stored_2dxd_coeffs."""
        wci = read_wavecalinfo(mode)
        stored = read_stored_2dxd_coeffs(mode)
        np.testing.assert_array_equal(wci.p2w_coeffs, stored.coeffs_flat)


# ---------------------------------------------------------------------------
# collect_centroid_data_from_arc_traces (PRIMARY path)
# ---------------------------------------------------------------------------


class TestCollectCentroidDataFromArcTraces:
    """Tests for :func:`collect_centroid_data_from_arc_traces` (PRIMARY path).

    This function converts an ArcTraceResult to flat centroid arrays.
    The representative detector coordinate is centroid_col (Gaussian-fit
    column), not seed_col.
    """

    def test_returns_correct_type(self, h1_centroid_data):
        assert isinstance(h1_centroid_data, ArcLineCentroidData)

    def test_mode_attribute(self, h1_centroid_data):
        assert h1_centroid_data.mode == "H1"

    def test_source_indicates_arc_tracing(self, h1_centroid_data):
        """Source should reference the arc_tracing path, not legacy path."""
        assert "arc_tracing" in h1_centroid_data.source

    def test_array_lengths_consistent(self, h1_centroid_data):
        cd = h1_centroid_data
        n = cd.n_points
        assert len(cd.columns) == n
        assert len(cd.orders) == n
        assert len(cd.wavelengths_um) == n
        assert len(cd.snr_values) == n

    def test_n_points_matches_arc_traces(self, h1_centroid_data, h1_arc_traces):
        """n_points must equal the number of lines in the ArcTraceResult."""
        assert h1_centroid_data.n_points == h1_arc_traces.n_lines

    def test_columns_are_centroid_cols(self, h1_centroid_data, h1_arc_traces):
        """Columns should come from centroid_col, not seed_col."""
        expected = np.array([ln.centroid_col for ln in h1_arc_traces.lines])
        np.testing.assert_array_equal(h1_centroid_data.columns, expected)

    def test_per_order_counts_match_arc_traces(self, h1_centroid_data, h1_arc_traces):
        """per_order_counts should match the ArcTraceResult counts."""
        for o, count in h1_arc_traces.per_order_counts.items():
            assert h1_centroid_data.per_order_counts[o] == count

    def test_has_positive_point_count(self, h1_centroid_data):
        assert h1_centroid_data.n_points > 0

    def test_columns_in_detector_range(self, h1_centroid_data):
        assert np.all(h1_centroid_data.columns >= 0)
        assert np.all(h1_centroid_data.columns < 2048)

    def test_wavelengths_in_h_band(self, h1_centroid_data):
        lams = h1_centroid_data.wavelengths_um
        assert np.all(lams > 1.40) and np.all(lams < 1.90)

    def test_snr_above_threshold(self, h1_centroid_data):
        assert np.all(h1_centroid_data.snr_values >= 3.0)


# ---------------------------------------------------------------------------
# collect_centroid_data – AUXILIARY / legacy path
# ---------------------------------------------------------------------------


class TestCollectCentroidDataLegacy:
    """Tests for :func:`collect_centroid_data` (AUXILIARY legacy path).

    This path calls fit_arc_line_centroids() directly, bypassing arc_tracing.
    It is kept for backward compatibility and comparison.  Its source should
    indicate it comes from the legacy path.
    """

    @pytest.fixture(scope="class")
    def h1_legacy_centroid_data(self, h1_calibrations):
        wci, fi, ll = h1_calibrations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return collect_centroid_data(wci, ll)

    def test_returns_correct_type(self, h1_legacy_centroid_data):
        assert isinstance(h1_legacy_centroid_data, ArcLineCentroidData)

    def test_source_indicates_legacy(self, h1_legacy_centroid_data):
        """Source should indicate it came from fit_arc_line_centroids, not arc_tracing."""
        assert "wavecal_fit_arc_line_centroids" in h1_legacy_centroid_data.source

    def test_n_points_matches_primary_path(
        self, h1_legacy_centroid_data, h1_centroid_data
    ):
        """Legacy and primary paths should yield the same centroid counts.

        Both use the same underlying Gaussian-fit algorithm on the same
        plane-1 data; results should be identical.
        """
        assert h1_legacy_centroid_data.n_points == h1_centroid_data.n_points

    def test_columns_equal_primary_path(
        self, h1_legacy_centroid_data, h1_centroid_data
    ):
        """Legacy and primary paths should give the same centroid columns."""
        np.testing.assert_allclose(
            np.sort(h1_legacy_centroid_data.columns),
            np.sort(h1_centroid_data.columns),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# fit_provisional_2dxd – new signature (takes ArcTraceResult)
# ---------------------------------------------------------------------------


class TestFitProvisional2DXD:
    """Tests for :func:`fit_provisional_2dxd` with the corrected arc_tracing path."""

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
        assert np.isfinite(rms) and rms >= 0.0

    def test_rms_below_10nm(self, h1_provisional_fit):
        """For H1 with the proxy arc spectrum, fit RMS should be < 10 nm."""
        assert h1_provisional_fit.rms_residual_um < 0.010

    def test_centroid_data_source_indicates_arc_tracing(self, h1_provisional_fit):
        """centroid_data.source should reflect the arc_tracing primary path."""
        assert "arc_tracing" in h1_provisional_fit.centroid_data.source

    def test_stored_2dxd_is_reference_role(self, h1_provisional_fit, h1_stored_2dxd):
        """stored_2dxd should be attached for reference/comparison."""
        assert isinstance(h1_provisional_fit.stored_2dxd, TwoDXDCoefficients)
        assert h1_provisional_fit.stored_2dxd.mode == "H1"
        np.testing.assert_array_equal(
            h1_provisional_fit.stored_2dxd.coeffs_flat,
            h1_stored_2dxd.coeffs_flat,
        )

    def test_n_orders_fitted_plus_bootstrap_equals_total(
        self, h1_provisional_fit, h1_calibrations
    ):
        wci, fi, ll = h1_calibrations
        result = h1_provisional_fit
        assert result.n_orders_fitted + result.n_orders_bootstrap == wci.n_orders

    def test_per_order_wave_coeffs_all_orders(self, h1_provisional_fit, h1_calibrations):
        wci, fi, ll = h1_calibrations
        for o in wci.orders:
            assert o in h1_provisional_fit.per_order_wave_coeffs

    def test_per_order_wave_coeffs_finite(self, h1_provisional_fit):
        for o, coeffs in h1_provisional_fit.per_order_wave_coeffs.items():
            assert np.all(np.isfinite(coeffs))

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
        assert lam.shape == (50,) and np.all(np.isfinite(lam))

    def test_eval_wavelength_home_order_in_h_band(self, h1_provisional_fit):
        x = np.linspace(270.0, 1960.0, 20)
        m = np.full(20, float(h1_provisional_fit.home_order))
        lam = h1_provisional_fit.eval_wavelength(x, m)
        assert np.all(lam > 1.40) and np.all(lam < 1.90)

    def test_eval_wavelength_monotone(self, h1_provisional_fit):
        x = np.linspace(270.0, 1960.0, 100)
        m = np.full(100, float(h1_provisional_fit.home_order))
        lam = h1_provisional_fit.eval_wavelength(x, m)
        diff = np.diff(lam)
        assert np.all(diff > 0) or np.all(diff < 0)

    def test_different_orders_give_different_wavelengths(self, h1_provisional_fit):
        lam_low = float(h1_provisional_fit.eval_wavelength(1000.0, 320.0))
        lam_high = float(h1_provisional_fit.eval_wavelength(1000.0, 340.0))
        assert lam_low != lam_high

    def test_arc_trace_mode_mismatch_raises(self, h1_calibrations):
        """Mismatched arc_trace_result and wavecalinfo modes should raise ValueError."""
        wci_h1, fi_h1, ll_h1 = h1_calibrations
        wci_k1 = read_wavecalinfo("K1")
        ll_k1 = read_line_list("K1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arc_traces_k1 = build_arc_trace_result_from_wavecalinfo(wci_k1, ll_k1)
        with pytest.raises(ValueError, match="mode"):
            fit_provisional_2dxd(arc_traces_k1, wci_h1, fi_h1, ll_h1)

    def test_wavecalinfo_flatinfo_mismatch_raises(self, h1_arc_traces, h1_calibrations):
        """Mismatched wavecalinfo and flatinfo modes should raise ValueError."""
        wci_h1, fi_h1, ll_h1 = h1_calibrations
        fi_k1 = read_flatinfo("K1")
        with pytest.raises(ValueError, match="mode"):
            fit_provisional_2dxd(h1_arc_traces, wci_h1, fi_k1, ll_h1)

    def test_h1_degrees_from_stored(self, h1_provisional_fit, h1_stored_2dxd):
        assert h1_provisional_fit.fitted_disp_degree == h1_stored_2dxd.disp_degree
        assert h1_provisional_fit.fitted_order_degree == h1_stored_2dxd.order_degree

    def test_override_degrees(self, h1_arc_traces, h1_calibrations):
        wci, fi, ll = h1_calibrations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_provisional_2dxd(
                h1_arc_traces, wci, fi, ll, disp_degree=2, order_degree=1
            )
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
            arc_traces = build_arc_trace_result_from_wavecalinfo(wci, ll, fi)
            result = fit_provisional_2dxd(arc_traces, wci, fi, ll)
        assert isinstance(result, ProvisionalWaveCal2DXD)
        assert result.mode == band_mode
        assert np.isfinite(result.rms_residual_um) and result.rms_residual_um >= 0.0
        # centroid_data must reflect arc_tracing primary path
        assert "arc_tracing" in result.centroid_data.source


# ---------------------------------------------------------------------------
# Cross-check: fit residuals vs centroid data
# ---------------------------------------------------------------------------


class TestFitResiduals:
    """Cross-checks: residuals of fitted polynomial against centroid input data."""

    def test_residuals_consistent_with_stored_rms(self, h1_provisional_fit):
        result = h1_provisional_fit
        cd = result.centroid_data
        if cd.n_points == 0:
            pytest.skip("No centroid data for H1")
        lam_pred = result.eval_wavelength(cd.columns, cd.orders)
        residuals = lam_pred - cd.wavelengths_um
        rms = float(np.std(residuals))
        np.testing.assert_allclose(rms, result.rms_residual_um, rtol=0.01)

    def test_centroid_residuals_finite(self, h1_provisional_fit):
        result = h1_provisional_fit
        cd = result.centroid_data
        if cd.n_points == 0:
            pytest.skip("No centroid data for H1")
        lam_pred = result.eval_wavelength(cd.columns, cd.orders)
        assert np.all(np.isfinite(lam_pred - cd.wavelengths_um))

