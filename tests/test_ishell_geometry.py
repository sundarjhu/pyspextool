"""
Tests for the iSHELL order geometry data-model (geometry.py).

These tests verify:
  - OrderGeometry objects can be created with mandatory fields only,
  - OrderGeometry polynomial evaluation helpers work correctly,
  - OrderGeometry optional fields (tilt, curvature, wave, spatcal) default to None,
  - RectificationMap can be created and exposes correct shape properties,
  - OrderGeometrySet holds multiple orders, supports look-up, and exposes
    correct aggregate predicates,
  - build_order_geometry_set() correctly converts flat-field arrays into
    an OrderGeometrySet,
  - The geometry types are importable from calibrations.py (re-export),
  - None of this breaks existing SpeX / uSpeX test infrastructure.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyspextool.instruments.ishell.geometry import (
    OrderGeometry,
    OrderGeometrySet,
    RectificationMap,
    build_order_geometry_set,
)

# Also verify the re-export from calibrations module works
from pyspextool.instruments.ishell.calibrations import (
    OrderGeometry as _CalOG,
    OrderGeometrySet as _CalOGS,
    RectificationMap as _CalRM,
    build_order_geometry_set as _cal_build,
)


# ---------------------------------------------------------------------------
# Constants reused across tests
# ---------------------------------------------------------------------------

# Simple linear edge polynomials: row = a + b*col
# (numpy.polynomial convention: coeffs[k] = coefficient of x**k)
BOTTOM_COEFFS = np.array([500.0, 0.0])   # flat at row 500
TOP_COEFFS    = np.array([540.0, 0.0])   # flat at row 540
CENTERLINE    = np.array([520.0, 0.0])   # midpoint

# Slightly slanted edges
BOTTOM_SLANT  = np.array([490.0, 0.02])  # row = 490 + 0.02*col
TOP_SLANT     = np.array([530.0, 0.02])  # row = 530 + 0.02*col

# Two representative iSHELL K-band order numbers
ORDER_A = 98
ORDER_B = 99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(order=ORDER_A, bottom=BOTTOM_COEFFS, top=TOP_COEFFS,
                x_start=200, x_end=1800) -> OrderGeometry:
    """Return a minimal OrderGeometry with no optional fields."""
    return OrderGeometry(
        order=order,
        x_start=x_start,
        x_end=x_end,
        bottom_edge_coeffs=bottom.copy(),
        top_edge_coeffs=top.copy(),
    )


# ===========================================================================
# 1. OrderGeometry – construction and basic properties
# ===========================================================================


class TestOrderGeometryConstruction:
    """OrderGeometry must be constructable with required fields only."""

    def test_creates_with_required_fields(self):
        g = _make_order()
        assert g.order == ORDER_A
        assert g.x_start == 200
        assert g.x_end == 1800

    def test_optional_fields_default_to_none(self):
        g = _make_order()
        assert g.tilt_coeffs is None
        assert g.curvature_coeffs is None
        assert g.wave_coeffs is None
        assert g.spatcal_coeffs is None

    def test_x_range_property(self):
        g = _make_order()
        assert g.x_range == (200, 1800)

    def test_n_columns(self):
        g = _make_order()
        assert g.n_columns() == 1601  # 1800 - 200 + 1

    def test_has_tilt_false_by_default(self):
        g = _make_order()
        assert g.has_tilt() is False

    def test_has_curvature_false_by_default(self):
        g = _make_order()
        assert g.has_curvature() is False

    def test_has_wavelength_solution_false_by_default(self):
        g = _make_order()
        assert g.has_wavelength_solution() is False

    def test_has_spatcal_false_by_default(self):
        g = _make_order()
        assert g.has_spatcal() is False

    def test_has_tilt_true_when_set(self):
        g = _make_order()
        g.tilt_coeffs = np.array([0.1, 0.0])
        assert g.has_tilt() is True

    def test_has_wavelength_solution_true_when_set(self):
        g = _make_order()
        g.wave_coeffs = np.array([1.2, 1e-5])
        assert g.has_wavelength_solution() is True


# ===========================================================================
# 2. OrderGeometry – polynomial evaluation
# ===========================================================================


class TestOrderGeometryEvaluation:
    """Edge, centerline, and tilt polynomials evaluate correctly."""

    def test_eval_bottom_edge_flat(self):
        g = _make_order(bottom=BOTTOM_COEFFS)  # constant 500
        result = g.eval_bottom_edge(np.array([0.0, 500.0, 1000.0]))
        np.testing.assert_allclose(result, [500.0, 500.0, 500.0])

    def test_eval_top_edge_flat(self):
        g = _make_order(top=TOP_COEFFS)        # constant 540
        result = g.eval_top_edge(np.array([0.0, 500.0, 1000.0]))
        np.testing.assert_allclose(result, [540.0, 540.0, 540.0])

    def test_eval_centerline_flat(self):
        g = _make_order(bottom=BOTTOM_COEFFS, top=TOP_COEFFS)
        result = g.eval_centerline(np.array([0.0, 1000.0]))
        np.testing.assert_allclose(result, [520.0, 520.0])

    def test_centerline_coeffs_property(self):
        g = _make_order(bottom=BOTTOM_COEFFS, top=TOP_COEFFS)
        expected = (BOTTOM_COEFFS + TOP_COEFFS) / 2.0
        np.testing.assert_allclose(g.centerline_coeffs, expected)

    def test_eval_bottom_edge_slanted(self):
        g = _make_order(bottom=BOTTOM_SLANT)   # row = 490 + 0.02*col
        cols = np.array([0.0, 500.0, 1000.0])
        expected = 490.0 + 0.02 * cols
        np.testing.assert_allclose(g.eval_bottom_edge(cols), expected)

    def test_eval_centerline_slanted(self):
        g = _make_order(bottom=BOTTOM_SLANT, top=TOP_SLANT)
        center_coeffs = (BOTTOM_SLANT + TOP_SLANT) / 2.0   # [510, 0.02]
        cols = np.array([0.0, 200.0, 1000.0])
        expected = np.polynomial.polynomial.polyval(cols, center_coeffs)
        np.testing.assert_allclose(g.eval_centerline(cols), expected)

    def test_eval_tilt_raises_without_coeffs(self):
        g = _make_order()
        with pytest.raises(RuntimeError, match="not yet available"):
            g.eval_tilt(np.array([100.0]))

    def test_eval_tilt_returns_values_when_set(self):
        g = _make_order()
        g.tilt_coeffs = np.array([0.05, 1e-5])   # tilt = 0.05 + 1e-5*col
        cols = np.array([0.0, 1000.0])
        expected = np.polynomial.polynomial.polyval(cols, g.tilt_coeffs)
        np.testing.assert_allclose(g.eval_tilt(cols), expected)

    def test_scalar_col_works(self):
        """Scalar column argument must work (not just arrays)."""
        g = _make_order()
        result = g.eval_centerline(500.0)
        assert float(result) == pytest.approx(520.0)


# ===========================================================================
# 3. RectificationMap – construction and shape
# ===========================================================================


class TestRectificationMap:
    """RectificationMap stores output grids and source coordinates."""

    @pytest.fixture()
    def rmap(self):
        n_spec, n_spat = 200, 64
        return RectificationMap(
            order=ORDER_A,
            output_wavelengths_um=np.linspace(2.0, 2.4, n_spec),
            output_spatial_arcsec=np.linspace(-3.0, 3.0, n_spat),
            src_rows=np.zeros((n_spat, n_spec)),
            src_cols=np.zeros((n_spat, n_spec)),
        )

    def test_output_shape(self, rmap):
        assert rmap.output_shape == (64, 200)

    def test_n_spectral(self, rmap):
        assert rmap.n_spectral == 200

    def test_n_spatial(self, rmap):
        assert rmap.n_spatial == 64

    def test_order_attribute(self, rmap):
        assert rmap.order == ORDER_A

    def test_src_arrays_shape(self, rmap):
        assert rmap.src_rows.shape == (64, 200)
        assert rmap.src_cols.shape == (64, 200)

    def test_wavelength_range(self, rmap):
        assert rmap.output_wavelengths_um[0] == pytest.approx(2.0)
        assert rmap.output_wavelengths_um[-1] == pytest.approx(2.4)


# ===========================================================================
# 4. OrderGeometrySet – collection interface
# ===========================================================================


class TestOrderGeometrySet:
    """OrderGeometrySet manages multiple OrderGeometry objects."""

    @pytest.fixture()
    def two_order_set(self):
        ga = _make_order(order=ORDER_A, x_start=100, x_end=900)
        gb = _make_order(order=ORDER_B, x_start=950, x_end=1850)
        return OrderGeometrySet(mode="K1", geometries=[ga, gb])

    def test_n_orders(self, two_order_set):
        assert two_order_set.n_orders == 2

    def test_orders_sorted(self, two_order_set):
        orders = two_order_set.orders
        assert orders == sorted(orders)
        assert ORDER_A in orders
        assert ORDER_B in orders

    def test_get_order_returns_correct_object(self, two_order_set):
        g = two_order_set.get_order(ORDER_A)
        assert g.order == ORDER_A

    def test_get_order_raises_for_missing(self, two_order_set):
        with pytest.raises(KeyError):
            two_order_set.get_order(9999)

    def test_has_tilt_false_when_none_set(self, two_order_set):
        assert two_order_set.has_tilt() is False

    def test_has_tilt_false_when_only_partial(self, two_order_set):
        # Only one order has tilt set → all-or-nothing, should still be False
        two_order_set.get_order(ORDER_A).tilt_coeffs = np.array([0.0])
        assert two_order_set.has_tilt() is False

    def test_has_tilt_requires_all_orders_to_have_tilt(self, two_order_set):
        two_order_set.get_order(ORDER_A).tilt_coeffs = np.array([0.05])
        two_order_set.get_order(ORDER_B).tilt_coeffs = np.array([0.04])
        assert two_order_set.has_tilt() is True

    def test_has_wavelength_solution_false_initially(self, two_order_set):
        assert two_order_set.has_wavelength_solution() is False

    def test_empty_set_has_tilt_false(self):
        gs = OrderGeometrySet(mode="K1")
        assert gs.has_tilt() is False

    def test_mode_attribute(self, two_order_set):
        assert two_order_set.mode == "K1"

    def test_multiple_modes_independent(self):
        ga = _make_order(order=ORDER_A)
        gb = _make_order(order=ORDER_B)
        set_k1 = OrderGeometrySet(mode="K1", geometries=[ga])
        set_k2 = OrderGeometrySet(mode="K2", geometries=[gb])
        assert set_k1.mode == "K1"
        assert set_k2.mode == "K2"
        assert set_k1.n_orders == 1
        assert set_k2.n_orders == 1


# ===========================================================================
# 5. build_order_geometry_set – factory function
# ===========================================================================


class TestBuildOrderGeometrySet:
    """build_order_geometry_set converts flat-field arrays to OrderGeometrySet."""

    @pytest.fixture()
    def flat_arrays(self):
        """Synthetic flat-field arrays for 3 orders with linear edge polynomials."""
        n_orders = 3
        orders = np.array([97, 98, 99], dtype=int)
        x_ranges = np.array(
            [[100, 700], [720, 1320], [1340, 1940]], dtype=int
        )
        # Shape: (n_orders, 2, n_terms) – 2-term linear polynomial [const, slope]
        edge_coeffs = np.array([
            [[480.0, 0.01], [520.0, 0.01]],   # order 97
            [[440.0, 0.02], [480.0, 0.02]],   # order 98
            [[400.0, 0.03], [440.0, 0.03]],   # order 99
        ])
        return orders, x_ranges, edge_coeffs

    def test_returns_order_geometry_set(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        assert isinstance(result, OrderGeometrySet)

    def test_correct_mode(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        assert result.mode == "K1"

    def test_n_orders(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        assert result.n_orders == 3

    def test_order_numbers(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        assert result.orders == [97, 98, 99]

    def test_x_ranges_stored(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        assert result.get_order(97).x_start == 100
        assert result.get_order(97).x_end == 700
        assert result.get_order(99).x_start == 1340
        assert result.get_order(99).x_end == 1940

    def test_edge_coeffs_stored(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        g97 = result.get_order(97)
        np.testing.assert_allclose(g97.bottom_edge_coeffs, [480.0, 0.01])
        np.testing.assert_allclose(g97.top_edge_coeffs,    [520.0, 0.01])

    def test_optional_fields_are_none(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        for g in result.geometries:
            assert g.tilt_coeffs is None
            assert g.wave_coeffs is None

    def test_mismatched_shapes_raise_value_error(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        # x_ranges with wrong number of orders
        with pytest.raises(ValueError, match="x_ranges must have shape"):
            build_order_geometry_set("K1", orders, x_ranges[:2, :], edge_coeffs)

    def test_edge_coeffs_wrong_shape_raises(self, flat_arrays):
        orders, x_ranges, edge_coeffs = flat_arrays
        with pytest.raises(ValueError, match="edge_coeffs must have shape"):
            build_order_geometry_set("K1", orders, x_ranges, edge_coeffs[:2, :, :])

    def test_evaluation_after_build(self, flat_arrays):
        """Polynomials built from flat data must evaluate correctly."""
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        g97 = result.get_order(97)
        # bottom = 480 + 0.01*col; top = 520 + 0.01*col; center = 500 + 0.01*col
        col = np.array([0.0, 500.0])
        np.testing.assert_allclose(g97.eval_bottom_edge(col), [480.0, 485.0])
        np.testing.assert_allclose(g97.eval_top_edge(col),    [520.0, 525.0])
        np.testing.assert_allclose(g97.eval_centerline(col),  [500.0, 505.0])

    def test_copies_of_edge_arrays(self, flat_arrays):
        """build_order_geometry_set must store copies, not views."""
        orders, x_ranges, edge_coeffs = flat_arrays
        result = build_order_geometry_set("K1", orders, x_ranges, edge_coeffs)
        # Mutate the source array; geometry must be unaffected
        edge_coeffs[0, 0, 0] = -999.0
        g97 = result.get_order(97)
        assert g97.bottom_edge_coeffs[0] == pytest.approx(480.0)


# ===========================================================================
# 6. Re-export from calibrations.py
# ===========================================================================


class TestCalibrationsReexport:
    """Geometry types must be importable from calibrations.py."""

    def test_order_geometry_same_class(self):
        assert _CalOG is OrderGeometry

    def test_order_geometry_set_same_class(self):
        assert _CalOGS is OrderGeometrySet

    def test_rectification_map_same_class(self):
        assert _CalRM is RectificationMap

    def test_build_function_same_object(self):
        assert _cal_build is build_order_geometry_set


# ===========================================================================
# 7. Non-regression – existing iSHELL calibration imports still work
# ===========================================================================


class TestNonRegression:
    """Existing calibration types must still be importable and functional."""

    def test_flatinfo_import(self):
        from pyspextool.instruments.ishell.calibrations import FlatInfo
        assert FlatInfo is not None

    def test_wavecalinfo_import(self):
        from pyspextool.instruments.ishell.calibrations import WaveCalInfo
        assert WaveCalInfo is not None

    def test_linelist_import(self):
        from pyspextool.instruments.ishell.calibrations import LineList
        assert LineList is not None

    def test_read_line_list_j0(self):
        """read_line_list must still work for J0 (regression check)."""
        from pyspextool.instruments.ishell.calibrations import read_line_list
        ll = read_line_list("J0")
        assert ll.n_lines > 0

    def test_read_flatinfo_k1(self):
        """read_flatinfo must still work for K1 (regression check)."""
        from pyspextool.instruments.ishell.calibrations import read_flatinfo
        fi = read_flatinfo("K1")
        assert fi.n_orders > 0

    def test_ishell_module_imports(self):
        """ishell.py must still be importable (regression check)."""
        import pyspextool.instruments.ishell.ishell as ishell_mod
        assert hasattr(ishell_mod, "read_fits")
        assert hasattr(ishell_mod, "get_header")
        assert hasattr(ishell_mod, "load_data")

    def test_rectify_orders_empty_geometry_returns_input(self):
        """_rectify_orders() with an empty geometry set returns the input unchanged."""
        from pyspextool.instruments.ishell.ishell import _rectify_orders
        from pyspextool.instruments.ishell.geometry import OrderGeometrySet
        dummy_img = np.zeros((10, 10))
        dummy_geom = OrderGeometrySet(mode="K1")
        result = _rectify_orders(dummy_img, dummy_geom)
        np.testing.assert_array_equal(result, dummy_img)

    def test_rectify_orders_no_wave_solution_raises_value_error(self):
        """_rectify_orders() must raise ValueError if geometry has no wavelength solution."""
        from pyspextool.instruments.ishell.ishell import _rectify_orders
        from pyspextool.instruments.ishell.geometry import OrderGeometry, OrderGeometrySet
        dummy_img = np.zeros((100, 100))
        # Build a geometry without wavelength solution
        g = OrderGeometry(
            order=233,
            x_start=10,
            x_end=50,
            bottom_edge_coeffs=np.array([20.0, 0.0]),
            top_edge_coeffs=np.array([40.0, 0.0]),
        )
        geom = OrderGeometrySet(mode="K1", geometries=[g])
        with pytest.raises(ValueError, match="wavelength solution"):
            _rectify_orders(dummy_img, geom)
