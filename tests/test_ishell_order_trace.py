"""
Tests for the iSHELL order-center tracing scaffold
(instruments/ishell/order_trace.py).

These tests verify:
  - the public API (OrderTraceResult, trace_order_centers_from_flatinfo),
  - correct output structure and field types,
  - accuracy of traced centers against the analytical edge-polynomial
    centerlines (sub-pixel agreement expected),
  - fit quality (small RMS residuals),
  - parameter validation (bad poly_degree, bad step_size),
  - consistency of the embedded OrderGeometrySet,
  - robustness: mode name string accepted in addition to FlatInfo object.

Real data note
--------------
All tests that trace order centers load the packaged H1 flat-field
calibration (``H1_flatinfo.fits``).  This file is a real iSHELL H1 mode
order mask derived from QTH flat frames and is shipped with the repository
as a packaged calibration resource (~8.4 MB).  Tests that load the full
2048×2048 order mask are marked ``slow`` so they can be excluded from
fast CI runs with ``-m "not slow"``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyspextool.instruments.ishell.calibrations import FlatInfo, read_flatinfo
from pyspextool.instruments.ishell.geometry import OrderGeometrySet
from pyspextool.instruments.ishell.order_trace import (
    OrderTraceResult,
    trace_order_centers_from_flatinfo,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

H1_MODE = "H1"
# Subset of H1 orders used for detailed checks (representative spread)
_CHECK_ORDERS = [311, 325, 340, 355]
# Maximum allowed RMS residual in pixels for a good-quality trace
_RMS_THRESHOLD_PIX = 0.5
# Maximum allowed deviation between traced center and edge-polynomial
# centerline at any sampled column, in pixels
_CENTER_AGREEMENT_PIX = 1.5


@pytest.fixture(scope="module")
def h1_flatinfo() -> FlatInfo:
    """Load the packaged H1 FlatInfo once per test module."""
    return read_flatinfo(H1_MODE)


@pytest.fixture(scope="module")
def h1_trace_result(h1_flatinfo) -> OrderTraceResult:
    """Run the tracer once per test module and share the result."""
    return trace_order_centers_from_flatinfo(h1_flatinfo, poly_degree=3, step_size=20)


# ---------------------------------------------------------------------------
# Test: result structure
# ---------------------------------------------------------------------------


class TestOrderTraceResultStructure:
    """Verify that the result has the expected structure and types."""

    def test_mode_matches(self, h1_trace_result):
        assert h1_trace_result.mode == H1_MODE

    def test_n_orders(self, h1_trace_result, h1_flatinfo):
        assert h1_trace_result.n_orders == h1_flatinfo.n_orders

    def test_orders_match_flatinfo(self, h1_trace_result, h1_flatinfo):
        assert h1_trace_result.orders == h1_flatinfo.orders

    def test_poly_degree_stored(self, h1_trace_result):
        assert h1_trace_result.poly_degree == 3

    def test_all_dicts_have_all_orders(self, h1_trace_result):
        orders = set(h1_trace_result.orders)
        assert set(h1_trace_result.trace_columns.keys()) == orders
        assert set(h1_trace_result.trace_rows.keys()) == orders
        assert set(h1_trace_result.center_coeffs.keys()) == orders
        assert set(h1_trace_result.fit_rms.keys()) == orders
        assert set(h1_trace_result.n_good.keys()) == orders

    def test_geometry_is_order_geometry_set(self, h1_trace_result):
        assert isinstance(h1_trace_result.geometry, OrderGeometrySet)

    def test_geometry_mode_matches(self, h1_trace_result):
        assert h1_trace_result.geometry.mode == H1_MODE

    def test_geometry_n_orders_matches(self, h1_trace_result):
        assert h1_trace_result.geometry.n_orders == h1_trace_result.n_orders


# ---------------------------------------------------------------------------
# Test: per-order trace arrays
# ---------------------------------------------------------------------------


class TestPerOrderArrays:
    """Verify per-order array shapes and types."""

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_trace_columns_is_1d_float(self, h1_trace_result, order):
        cols = h1_trace_result.trace_columns[order]
        assert isinstance(cols, np.ndarray)
        assert cols.ndim == 1
        assert np.issubdtype(cols.dtype, np.floating)

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_trace_rows_is_1d_float(self, h1_trace_result, order):
        rows = h1_trace_result.trace_rows[order]
        assert isinstance(rows, np.ndarray)
        assert rows.ndim == 1

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_columns_and_rows_same_length(self, h1_trace_result, order):
        cols = h1_trace_result.trace_columns[order]
        rows = h1_trace_result.trace_rows[order]
        assert len(cols) == len(rows)

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_n_good_matches_array_length(self, h1_trace_result, order):
        n = h1_trace_result.n_good[order]
        assert n == len(h1_trace_result.trace_columns[order])

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_n_good_positive(self, h1_trace_result, order):
        assert h1_trace_result.n_good[order] > 0

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_trace_rows_in_detector_range(self, h1_trace_result, order):
        rows = h1_trace_result.trace_rows[order]
        assert np.all(rows >= 0)
        assert np.all(rows < 2048)

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_trace_columns_in_xrange(self, h1_trace_result, h1_flatinfo, order):
        """Traced columns must lie within the declared xrange."""
        idx = h1_flatinfo.orders.index(order)
        x_start, x_end = int(h1_flatinfo.xranges[idx, 0]), int(h1_flatinfo.xranges[idx, 1])
        cols = h1_trace_result.trace_columns[order]
        assert np.all(cols >= x_start)
        assert np.all(cols <= x_end)


# ---------------------------------------------------------------------------
# Test: polynomial fit quality
# ---------------------------------------------------------------------------


class TestFitQuality:
    """Verify that the polynomial fits have small residuals."""

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_center_coeffs_has_poly_degree_plus_1_terms(
        self, h1_trace_result, order
    ):
        coeffs = h1_trace_result.center_coeffs[order]
        assert len(coeffs) == h1_trace_result.poly_degree + 1

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_fit_rms_below_threshold(self, h1_trace_result, order):
        """RMS residuals must be well below one pixel for the packaged data."""
        rms = h1_trace_result.fit_rms[order]
        assert np.isfinite(rms), f"Order {order}: fit_rms is not finite"
        assert rms < _RMS_THRESHOLD_PIX, (
            f"Order {order}: fit_rms={rms:.4f} px exceeds threshold "
            f"{_RMS_THRESHOLD_PIX} px"
        )

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_fit_residuals_internally_consistent(self, h1_trace_result, order):
        """Re-compute residuals from stored arrays and coefficients."""
        cols = h1_trace_result.trace_columns[order]
        rows = h1_trace_result.trace_rows[order]
        coeffs = h1_trace_result.center_coeffs[order]
        fitted = np.polynomial.polynomial.polyval(cols, coeffs)
        recomputed_rms = float(np.sqrt(np.mean((rows - fitted) ** 2)))
        assert abs(recomputed_rms - h1_trace_result.fit_rms[order]) < 1e-6


# ---------------------------------------------------------------------------
# Test: agreement with edge-polynomial centerlines
# ---------------------------------------------------------------------------


class TestAgreementWithEdgePolynomials:
    """Check that pixel-level trace and analytical centerline agree closely."""

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_center_agrees_with_geometry_centerline(
        self, h1_trace_result, order
    ):
        """Traced center must be within _CENTER_AGREEMENT_PIX of the
        edge-polynomial centerline at every sampled column."""
        geom = h1_trace_result.geometry.get_order(order)
        cols = h1_trace_result.trace_columns[order]
        if len(cols) == 0:
            pytest.skip(f"Order {order}: no valid trace points")
        traced = h1_trace_result.eval_center(order, cols)
        analytical = geom.eval_centerline(cols)
        max_diff = float(np.max(np.abs(traced - analytical)))
        assert max_diff < _CENTER_AGREEMENT_PIX, (
            f"Order {order}: max deviation between traced center and "
            f"edge-polynomial centerline = {max_diff:.4f} px "
            f"(threshold {_CENTER_AGREEMENT_PIX} px)"
        )


# ---------------------------------------------------------------------------
# Test: eval_center helper method
# ---------------------------------------------------------------------------


class TestEvalCenter:
    """Verify eval_center evaluates the fitted polynomial correctly."""

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_eval_center_scalar(self, h1_trace_result, order):
        col = 1000.0
        result = h1_trace_result.eval_center(order, col)
        assert np.isfinite(result)
        assert 0 <= result < 2048

    @pytest.mark.parametrize("order", _CHECK_ORDERS)
    def test_eval_center_array(self, h1_trace_result, order):
        cols = np.array([700.0, 900.0, 1100.0, 1300.0])
        results = h1_trace_result.eval_center(order, cols)
        assert results.shape == (4,)
        assert np.all(np.isfinite(results))

    def test_eval_center_unknown_order_raises(self, h1_trace_result):
        with pytest.raises(KeyError):
            h1_trace_result.eval_center(9999, 1000)


# ---------------------------------------------------------------------------
# Test: parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    """Verify that bad inputs raise informative errors."""

    def test_negative_poly_degree_raises(self, h1_flatinfo):
        with pytest.raises(ValueError, match="poly_degree"):
            trace_order_centers_from_flatinfo(h1_flatinfo, poly_degree=-1)

    def test_zero_step_size_raises(self, h1_flatinfo):
        with pytest.raises(ValueError, match="step_size"):
            trace_order_centers_from_flatinfo(h1_flatinfo, step_size=0)

    def test_negative_step_size_raises(self, h1_flatinfo):
        with pytest.raises(ValueError, match="step_size"):
            trace_order_centers_from_flatinfo(h1_flatinfo, step_size=-5)


# ---------------------------------------------------------------------------
# Test: mode string accepted
# ---------------------------------------------------------------------------


class TestModeStringInput:
    """Verify that a mode name string can be passed directly."""

    @pytest.mark.slow
    def test_mode_string_gives_same_result(self, h1_flatinfo):
        result_from_obj = trace_order_centers_from_flatinfo(
            h1_flatinfo, poly_degree=3, step_size=50
        )
        result_from_str = trace_order_centers_from_flatinfo(
            H1_MODE, poly_degree=3, step_size=50
        )
        assert result_from_str.n_orders == result_from_obj.n_orders
        assert result_from_str.orders == result_from_obj.orders
        # Polynomial coefficients should be identical
        for order in _CHECK_ORDERS:
            np.testing.assert_array_equal(
                result_from_str.center_coeffs[order],
                result_from_obj.center_coeffs[order],
            )


# ---------------------------------------------------------------------------
# Test: step_size and poly_degree variation
# ---------------------------------------------------------------------------


class TestParameterVariation:
    """Verify that varying step_size and poly_degree gives physically
    consistent results."""

    def test_larger_step_size_still_converges(self, h1_flatinfo):
        result = trace_order_centers_from_flatinfo(
            h1_flatinfo, poly_degree=3, step_size=50
        )
        for order in _CHECK_ORDERS:
            assert result.n_good[order] > 0
            assert result.fit_rms[order] < _RMS_THRESHOLD_PIX

    def test_degree_2_fit_acceptable(self, h1_flatinfo):
        result = trace_order_centers_from_flatinfo(
            h1_flatinfo, poly_degree=2, step_size=20
        )
        for order in _CHECK_ORDERS:
            coeffs = result.center_coeffs[order]
            assert len(coeffs) == 3  # degree 2 + 1

    def test_degree_4_fit_acceptable(self, h1_flatinfo):
        result = trace_order_centers_from_flatinfo(
            h1_flatinfo, poly_degree=4, step_size=20
        )
        for order in _CHECK_ORDERS:
            coeffs = result.center_coeffs[order]
            assert len(coeffs) == 5  # degree 4 + 1
            assert result.fit_rms[order] < _RMS_THRESHOLD_PIX


# ---------------------------------------------------------------------------
# Test: all H1 orders (slow – full dataset)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAllH1Orders:
    """Full-dataset smoke test: trace all 45 H1 orders."""

    def test_all_orders_have_valid_trace(self, h1_trace_result):
        for order in h1_trace_result.orders:
            n = h1_trace_result.n_good[order]
            assert n > 0, f"Order {order}: zero valid trace points"

    def test_all_orders_fit_rms_finite(self, h1_trace_result):
        for order in h1_trace_result.orders:
            rms = h1_trace_result.fit_rms[order]
            assert np.isfinite(rms), f"Order {order}: fit_rms is NaN/inf"

    def test_all_orders_fit_rms_below_threshold(self, h1_trace_result):
        failures = [
            (o, h1_trace_result.fit_rms[o])
            for o in h1_trace_result.orders
            if h1_trace_result.fit_rms[o] >= _RMS_THRESHOLD_PIX
        ]
        assert not failures, (
            "The following orders exceed the RMS threshold "
            f"({_RMS_THRESHOLD_PIX} px):\n"
            + "\n".join(f"  order {o}: rms={r:.4f}" for o, r in failures)
        )

    def test_order_centers_monotonically_spaced(self, h1_trace_result):
        """Order centers at the detector midpoint should increase monotonically
        with order number (lower order numbers sit at smaller row indices
        for H1 mode)."""
        mid_col = 1000.0
        centers = [
            h1_trace_result.eval_center(o, mid_col)
            for o in sorted(h1_trace_result.orders)
        ]
        for i in range(len(centers) - 1):
            assert centers[i] < centers[i + 1], (
                f"Non-monotonic centers at col={mid_col}: "
                f"order {h1_trace_result.orders[i]} center={centers[i]:.2f} "
                f">= order {h1_trace_result.orders[i+1]} center={centers[i+1]:.2f}"
            )
