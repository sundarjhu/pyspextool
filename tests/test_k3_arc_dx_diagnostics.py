"""
Tests for k3_arc_dx_diagnostics.py — Stage 3b pixel-error diagnostics.

Coverage:
  1. WEAK_ORDERS constant contains the expected hardcoded orders.
  2. OrderDxStats dataclass has the correct fields and __str__ format.
  3. _smooth_spectrum applies Gaussian smoothing correctly.
  4. _detect_peaks finds peaks above median + 3*MAD threshold.
  5. _invert_model_for_order produces column values inside [col_start, col_end].
  6. _compute_dx returns nearest-peak distance for each predicted position.
  7. run_k3_arc_dx_diagnostics returns OrderDxStats for each known order.
  8. run_k3_arc_dx_diagnostics skips orders absent from arc_spectra gracefully.
  9. Plots are saved to qa_dx/order_<N>.png when save_plots=True.
 10. run_k3_arc_dx_diagnostics does not raise when out_dir=None.
"""

from __future__ import annotations

import os
import math
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

from pyspextool.instruments.ishell.k3_arc_dx_diagnostics import (
    WEAK_ORDERS,
    OrderDxStats,
    run_k3_arc_dx_diagnostics,
    _smooth_spectrum,
    _detect_peaks,
    _invert_model_for_order,
    _compute_dx,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_model(order_ref: float = 204.0, wdeg: int = 2, odeg: int = 1):
    """Return an IdlStyle1DXDModel-like mock with a simple linear wavelength→col mapping.

    λ(col, order) = C0 + C1 * col  (ignoring order dependence for simplicity)
    where C0 = 2.0 µm at col=0, C1 = 5e-4 µm/pixel.
    """
    from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel

    # Build a minimal coefficient matrix: only C[1,0] is non-zero (linear in col)
    coeffs = np.zeros((wdeg + 1, odeg + 1), dtype=float)
    coeffs[0, 0] = 2.0          # constant term (µm)
    coeffs[1, 0] = 5e-4         # linear-in-col coefficient

    model = IdlStyle1DXDModel(
        mode="K3",
        wdeg=wdeg,
        odeg=odeg,
        order_ref=order_ref,
        coeffs=coeffs,
        fitted_order_numbers=[int(order_ref)],
        fit_rms_um=0.001,
        n_lines=10,
        n_lines_total=12,
        n_lines_rejected=2,
        accepted_mask=np.ones(12, dtype=bool),
        median_residual_um=0.0,
        n_orders_fit=1,
    )
    return model


def _make_arc_spectrum(order_number: int, col_start: int = 100, n_cols: int = 400):
    """Return an OrderArcSpectrum with a synthetic spectrum containing known peaks."""
    from pyspextool.instruments.ishell.wavecal_k3_idlstyle import OrderArcSpectrum

    rng = np.random.default_rng(seed=order_number)
    flux = rng.normal(10.0, 1.0, size=n_cols).astype(float)
    # Insert bright peaks at offsets 50, 150, 250 within the spectrum
    for offset in [50, 150, 250]:
        if offset < n_cols:
            flux[offset] += 200.0

    return OrderArcSpectrum(
        order_index=0,
        order_number=order_number,
        col_start=col_start,
        col_end=col_start + n_cols - 1,
        flux=flux,
    )


def _make_arc_spectra_set(order_numbers):
    """Return an OrderArcSpectraSet populated with synthetic spectra."""
    from pyspextool.instruments.ishell.wavecal_k3_idlstyle import OrderArcSpectraSet

    spectra = [_make_arc_spectrum(o) for o in order_numbers]
    return OrderArcSpectraSet(mode="K3", spectra=spectra, aperture_half_width=3)


def _make_line_list(order_numbers, n_lines_per_order: int = 5):
    """Return a LineList with synthetic entries for each order."""
    from pyspextool.instruments.ishell.calibrations import LineList, LineListEntry

    entries = []
    for order in order_numbers:
        # Generate n_lines_per_order wavelengths uniformly spread over [2.05, 2.35] µm
        wavs = np.linspace(2.05, 2.35, n_lines_per_order)
        for wav in wavs:
            entries.append(
                LineListEntry(
                    order=order,
                    wavelength_um=float(wav),
                    species="Th I",
                    fit_window_angstrom=2.0,
                    fit_type="G",
                    fit_n_terms=3,
                )
            )
    return LineList(mode="K3", entries=entries)


# ---------------------------------------------------------------------------
# 1. WEAK_ORDERS constant
# ---------------------------------------------------------------------------


class TestWeakOrdersConstant:
    def test_contains_required_orders(self):
        expected = [204, 205, 212, 213, 214, 217, 221, 228]
        assert WEAK_ORDERS == expected, f"WEAK_ORDERS mismatch: {WEAK_ORDERS}"

    def test_is_list_of_ints(self):
        assert all(isinstance(o, int) for o in WEAK_ORDERS)

    def test_is_not_empty(self):
        assert len(WEAK_ORDERS) > 0


# ---------------------------------------------------------------------------
# 2. OrderDxStats dataclass
# ---------------------------------------------------------------------------


class TestOrderDxStats:
    def test_basic_construction(self):
        dx = np.array([1.0, 2.5, 0.3])
        stats = OrderDxStats(
            order_number=204,
            n_predicted=3,
            n_peaks=10,
            dx_values=dx,
            median_dx=1.0,
            min_dx=0.3,
            max_dx=2.5,
        )
        assert stats.order_number == 204
        assert stats.n_predicted == 3
        assert stats.n_peaks == 10
        np.testing.assert_array_equal(stats.dx_values, dx)
        assert stats.median_dx == pytest.approx(1.0)
        assert stats.min_dx == pytest.approx(0.3)
        assert stats.max_dx == pytest.approx(2.5)

    def test_str_includes_order_and_stats(self):
        stats = OrderDxStats(
            order_number=205,
            n_predicted=4,
            n_peaks=8,
            dx_values=np.array([0.5, 1.0, 1.5, 2.0]),
            median_dx=1.25,
            min_dx=0.5,
            max_dx=2.0,
        )
        s = str(stats)
        assert "205" in s
        assert "1.25" in s
        assert "0.50" in s
        assert "2.00" in s

    def test_default_nan_values(self):
        stats = OrderDxStats(order_number=212, n_predicted=0, n_peaks=0)
        assert math.isnan(stats.median_dx)
        assert math.isnan(stats.min_dx)
        assert math.isnan(stats.max_dx)
        assert len(stats.dx_values) == 0


# ---------------------------------------------------------------------------
# 3. _smooth_spectrum
# ---------------------------------------------------------------------------


class TestSmoothSpectrum:
    def test_output_shape_matches_input(self):
        flux = np.ones(200, dtype=float)
        smoothed = _smooth_spectrum(flux, sigma=1.5)
        assert smoothed.shape == flux.shape

    def test_nan_preserved(self):
        flux = np.ones(100, dtype=float)
        flux[30:35] = np.nan
        smoothed = _smooth_spectrum(flux, sigma=1.5)
        assert np.all(np.isnan(smoothed[30:35]))

    def test_reduces_sharp_spike(self):
        """A single-pixel spike must be attenuated by Gaussian smoothing."""
        flux = np.zeros(100, dtype=float)
        flux[50] = 1000.0
        smoothed = _smooth_spectrum(flux, sigma=1.5)
        # Peak should be lower after smoothing
        assert smoothed[50] < 1000.0
        # Total energy should be approximately conserved
        assert pytest.approx(smoothed.sum(), rel=0.01) == flux.sum()

    def test_constant_spectrum_unchanged(self):
        flux = np.full(200, 5.0, dtype=float)
        smoothed = _smooth_spectrum(flux, sigma=2.0)
        np.testing.assert_allclose(smoothed, flux, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. _detect_peaks
# ---------------------------------------------------------------------------


class TestDetectPeaks:
    def _make_columns(self, n=500, start=100):
        return np.arange(start, start + n, dtype=float)

    def test_finds_obvious_peaks(self):
        rng = np.random.default_rng(0)
        flux = rng.normal(10.0, 1.0, size=500)
        cols = self._make_columns(500)
        # Insert obvious peaks
        for idx in [100, 250, 400]:
            flux[idx] += 500.0
        peaks = _detect_peaks(flux, cols)
        # All inserted peaks should be detected
        for idx in [100, 250, 400]:
            expected_col = cols[idx]
            assert expected_col in peaks, f"Expected peak at col {expected_col}"

    def test_no_peaks_flat_spectrum(self):
        flux = np.full(300, 10.0, dtype=float)
        cols = self._make_columns(300)
        peaks = _detect_peaks(flux, cols)
        # A perfectly flat spectrum has no local maxima above threshold
        assert len(peaks) == 0

    def test_all_nan_returns_empty(self):
        flux = np.full(100, np.nan)
        cols = self._make_columns(100)
        peaks = _detect_peaks(flux, cols)
        assert len(peaks) == 0

    def test_returns_column_values_not_indices(self):
        """Returned values must be actual column coordinates, not array indices."""
        rng = np.random.default_rng(1)
        flux = rng.normal(5.0, 0.5, size=200)
        col_start = 500
        cols = np.arange(col_start, col_start + 200, dtype=float)
        flux[100] += 300.0
        peaks = _detect_peaks(flux, cols)
        if len(peaks) > 0:
            assert peaks.min() >= col_start


# ---------------------------------------------------------------------------
# 5. _invert_model_for_order
# ---------------------------------------------------------------------------


class TestInvertModelForOrder:
    def test_predictions_in_range(self):
        model = _make_model()
        order = 204
        col_start, col_end = 100, 499
        # Wavelengths from model.eval at a few columns
        ref_wavs = np.array([
            model.eval(200.0, order),
            model.eval(300.0, order),
            model.eval(400.0, order),
        ])
        x_pred = _invert_model_for_order(model, order, col_start, col_end, ref_wavs)
        assert x_pred.shape == ref_wavs.shape
        # Should be approximately 200, 300, 400
        np.testing.assert_allclose(x_pred, [200.0, 300.0, 400.0], atol=0.5)

    def test_output_length_matches_input(self):
        model = _make_model()
        ref_wavs = np.linspace(2.1, 2.3, 10)
        x_pred = _invert_model_for_order(model, 204, 0, 2047, ref_wavs)
        assert len(x_pred) == len(ref_wavs)


# ---------------------------------------------------------------------------
# 6. _compute_dx
# ---------------------------------------------------------------------------


class TestComputeDx:
    def test_exact_match(self):
        x_pred = np.array([100.0, 200.0, 300.0])
        peak_cols = np.array([100.0, 200.0, 300.0])
        dx = _compute_dx(x_pred, peak_cols)
        np.testing.assert_allclose(dx, [0.0, 0.0, 0.0])

    def test_nearest_peak_selected(self):
        x_pred = np.array([150.0])
        peak_cols = np.array([100.0, 148.0, 200.0])
        dx = _compute_dx(x_pred, peak_cols)
        assert dx[0] == pytest.approx(2.0)

    def test_empty_peaks_returns_inf(self):
        x_pred = np.array([100.0, 200.0])
        dx = _compute_dx(x_pred, np.empty(0))
        assert np.all(np.isinf(dx))
        assert len(dx) == 2

    def test_empty_predictions_returns_empty(self):
        dx = _compute_dx(np.empty(0), np.array([100.0, 200.0]))
        assert len(dx) == 0


# ---------------------------------------------------------------------------
# 7. run_k3_arc_dx_diagnostics — returns stats for known orders
# ---------------------------------------------------------------------------


class TestRunK3ArcDxDiagnostics:
    def _build_inputs(self, order_numbers):
        model = _make_model(order_ref=float(min(order_numbers)))
        arc_spectra = _make_arc_spectra_set(order_numbers)
        line_list = _make_line_list(order_numbers)
        return model, arc_spectra, line_list

    def test_returns_stats_for_all_present_orders(self):
        orders = [204, 205]
        model, arc_spectra, line_list = self._build_inputs(orders)
        with tempfile.TemporaryDirectory() as tmp:
            results = run_k3_arc_dx_diagnostics(
                model=model,
                arc_spectra=arc_spectra,
                line_list=line_list,
                out_dir=tmp,
                save_plots=True,
                weak_orders=orders,
            )
        assert len(results) == 2
        order_nums = [r.order_number for r in results]
        assert 204 in order_nums
        assert 205 in order_nums

    def test_stats_fields_are_populated(self):
        orders = [212]
        model, arc_spectra, line_list = self._build_inputs(orders)
        with tempfile.TemporaryDirectory() as tmp:
            results = run_k3_arc_dx_diagnostics(
                model=model,
                arc_spectra=arc_spectra,
                line_list=line_list,
                out_dir=tmp,
                save_plots=True,
                weak_orders=orders,
            )
        assert len(results) == 1
        s = results[0]
        assert s.n_predicted >= 0
        assert s.n_peaks >= 0
        assert len(s.dx_values) == s.n_predicted

    # ------------------------------------------------------------------
    # 8. Graceful skip for absent orders
    # ------------------------------------------------------------------

    def test_skips_orders_absent_from_arc_spectra(self):
        """Orders not in arc_spectra should be silently skipped."""
        model = _make_model()
        arc_spectra = _make_arc_spectra_set([204])
        line_list = _make_line_list([204, 999])
        results = run_k3_arc_dx_diagnostics(
            model=model,
            arc_spectra=arc_spectra,
            line_list=line_list,
            out_dir=None,
            save_plots=False,
            weak_orders=[204, 999],
        )
        order_nums = [r.order_number for r in results]
        assert 204 in order_nums
        assert 999 not in order_nums

    def test_all_orders_absent_returns_empty_list(self):
        model = _make_model()
        arc_spectra = _make_arc_spectra_set([204])
        line_list = _make_line_list([204])
        results = run_k3_arc_dx_diagnostics(
            model=model,
            arc_spectra=arc_spectra,
            line_list=line_list,
            out_dir=None,
            save_plots=False,
            weak_orders=[998, 999],
        )
        assert results == []

    # ------------------------------------------------------------------
    # 9. Plots saved to qa_dx/order_<N>.png
    # ------------------------------------------------------------------

    def test_plots_saved_to_out_dir(self):
        orders = [204, 205]
        model, arc_spectra, line_list = self._build_inputs(orders)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "qa_dx")
            run_k3_arc_dx_diagnostics(
                model=model,
                arc_spectra=arc_spectra,
                line_list=line_list,
                out_dir=out_dir,
                save_plots=True,
                weak_orders=orders,
            )
            for o in orders:
                expected = os.path.join(out_dir, f"order_{o}.png")
                assert os.path.isfile(expected), f"Missing plot: {expected}"

    # ------------------------------------------------------------------
    # 10. No error when out_dir=None
    # ------------------------------------------------------------------

    def test_no_error_when_out_dir_none(self):
        orders = [204]
        model, arc_spectra, line_list = self._build_inputs(orders)
        results = run_k3_arc_dx_diagnostics(
            model=model,
            arc_spectra=arc_spectra,
            line_list=line_list,
            out_dir=None,
            save_plots=False,
            weak_orders=orders,
        )
        assert len(results) == 1

    def test_custom_weak_orders_override(self):
        """Passing weak_orders= overrides WEAK_ORDERS."""
        orders = [204]
        model, arc_spectra, line_list = self._build_inputs(orders)
        results = run_k3_arc_dx_diagnostics(
            model=model,
            arc_spectra=arc_spectra,
            line_list=line_list,
            out_dir=None,
            save_plots=False,
            weak_orders=[204],
        )
        assert all(r.order_number == 204 for r in results)
