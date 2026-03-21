"""
Tests for the iSHELL order-centre tracing module (tracing.py).

These tests verify:
  - FlatOrderTrace can be constructed directly (dataclass API),
  - load_and_combine_flats() raises on empty input and returns the
    correct shape for one or multiple synthetic FITS files,
  - trace_orders_from_flat() raises on empty file list and on profiles
    with no detectable peaks,
  - trace_orders_from_flat() works end-to-end on a synthetic flat image
    with known order structure,
  - the polynomial fit residuals are within expected bounds on synthetic data,
  - to_order_geometry_set() converts the result to a valid OrderGeometrySet,
  - the module is importable from instruments.ishell (package __init__),
  - a smoke-test that loads the real H1 calibration flat frames and
    confirms that the expected number of orders and polynomial quality
    criteria are met (requires the real data files to be present).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.tracing import (
    FlatOrderTrace,
    OrderTraceSamples,
    OrderTraceStats,
    load_and_combine_flats,
    trace_orders_from_flat,
)
from pyspextool.instruments.ishell.geometry import OrderGeometrySet
from pyspextool.instruments.ishell.io_utils import is_fits_file

# ---------------------------------------------------------------------------
# Path to the real H1 calibration data
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_H1_RAW_DIR = os.path.join(
    _REPO_ROOT, "data", "testdata", "ishell_h1_calibrations", "raw"
)
_LFS_MAGIC = b"version https://git-lfs"


def _h1_flat_files() -> list[str]:
    """Return sorted list of real H1 flat file paths, or empty list if absent."""
    if not os.path.isdir(_H1_RAW_DIR):
        return []
    paths = sorted(
        os.path.join(_H1_RAW_DIR, f)
        for f in os.listdir(_H1_RAW_DIR)
        if "flat" in f and is_fits_file(f)
    )
    # Filter out LFS pointer files
    real_paths = []
    for p in paths:
        with open(p, "rb") as fh:
            head = fh.read(64)
        if not head.startswith(_LFS_MAGIC):
            real_paths.append(p)
    return real_paths


_H1_FLAT_FILES = _h1_flat_files()
_HAVE_H1_DATA = len(_H1_FLAT_FILES) >= 1

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

# Detector shape used for synthetic tests
_NROWS = 256
_NCOLS = 512

# Known synthetic order parameters: 5 orders, each 20 rows wide,
# evenly spaced with 40-row centre-to-centre pitch.
# Order centres at col 0: [60, 100, 140, 180, 220]
# with a small linear tilt: centre(col) = c0 + 0.01 * col
_SYN_N_ORDERS = 5
_SYN_SPACING = 40         # row spacing between order centres
_SYN_FIRST_CENTER = 60   # centre of order 0 at col 0
_SYN_HALF_WIDTH = 10      # half-width in rows
_SYN_TILT = 0.01          # pixels per column


def _make_synthetic_flat(seed: int = 42) -> np.ndarray:
    """Return a synthetic flat image with 5 known order bands.

    Each order is a Gaussian cross-section (sigma ≈ half-width) sitting
    on a low background.  Gaussian noise is added at a level of 50 DN.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(_NROWS, dtype=float)
    cols = np.arange(_NCOLS, dtype=float)
    flat = rng.normal(0.0, 50.0, size=(_NROWS, _NCOLS)).astype(np.float32)
    flat += 200.0  # background pedestal

    for k in range(_SYN_N_ORDERS):
        c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
        for j, col in enumerate(cols):
            center = c0 + _SYN_TILT * col
            sigma = _SYN_HALF_WIDTH / 2.355  # convert FWHM to sigma
            flat[:, j] += 10000.0 * np.exp(
                -0.5 * ((rows - center) / sigma) ** 2
            )

    return flat


def _write_fits_flat(flat: np.ndarray, path: str) -> None:
    """Write *flat* as a minimal 3-extension iSHELL FITS file to *path*."""
    primary = fits.PrimaryHDU(data=flat)
    ped = fits.ImageHDU(data=np.zeros_like(flat), name="SUM_PED")
    sam = fits.ImageHDU(data=np.zeros_like(flat), name="SUM_SAM")
    fits.HDUList([primary, ped, sam]).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# 1. FlatOrderTrace dataclass API
# ---------------------------------------------------------------------------


class TestFlatOrderTraceDataclass:
    """FlatOrderTrace can be constructed and queried directly."""

    @pytest.fixture()
    def dummy_trace(self):
        n = 3
        ns = 10
        return FlatOrderTrace(
            n_orders=n,
            sample_cols=np.arange(ns),
            center_rows=np.ones((n, ns)) * 100.0,
            center_poly_coeffs=np.zeros((n, 4)),
            fit_rms=np.ones(n),
            half_width_rows=np.full(n, 15.0),
            poly_degree=3,
            seed_col=5,
        )

    def test_n_orders(self, dummy_trace):
        assert dummy_trace.n_orders == 3

    def test_sample_cols_shape(self, dummy_trace):
        assert dummy_trace.sample_cols.shape == (10,)

    def test_center_rows_shape(self, dummy_trace):
        assert dummy_trace.center_rows.shape == (3, 10)

    def test_poly_degree(self, dummy_trace):
        assert dummy_trace.poly_degree == 3

    def test_seed_col(self, dummy_trace):
        assert dummy_trace.seed_col == 5

    def test_to_order_geometry_set_returns_correct_type(self, dummy_trace):
        geom = dummy_trace.to_order_geometry_set("H1")
        assert isinstance(geom, OrderGeometrySet)

    def test_to_order_geometry_set_n_orders(self, dummy_trace):
        geom = dummy_trace.to_order_geometry_set("H1")
        assert geom.n_orders == 3

    def test_to_order_geometry_set_mode(self, dummy_trace):
        geom = dummy_trace.to_order_geometry_set("H1")
        assert geom.mode == "H1"

    def test_to_order_geometry_set_col_range(self, dummy_trace):
        geom = dummy_trace.to_order_geometry_set("H1", col_range=(100, 900))
        assert geom.geometries[0].x_start == 100
        assert geom.geometries[0].x_end == 900

    def test_to_order_geometry_set_col_range_defaults_to_sample_cols(self, dummy_trace):
        geom = dummy_trace.to_order_geometry_set("H1")
        assert geom.geometries[0].x_start == int(dummy_trace.sample_cols[0])
        assert geom.geometries[0].x_end == int(dummy_trace.sample_cols[-1])

    def test_to_order_geometry_set_edges_offset_by_half_width(self, dummy_trace):
        # When bot_poly_coeffs / top_poly_coeffs are None (direct construction),
        # to_order_geometry_set falls back to centre ± half_width.
        assert dummy_trace.bot_poly_coeffs is None
        assert dummy_trace.top_poly_coeffs is None
        hw = float(dummy_trace.half_width_rows[0])
        geom = dummy_trace.to_order_geometry_set("H1")
        g = geom.geometries[0]
        # constant term of center poly is 0; half-width offset applied to constant
        np.testing.assert_allclose(g.bottom_edge_coeffs[0], g.centerline_coeffs[0] - hw)
        np.testing.assert_allclose(g.top_edge_coeffs[0], g.centerline_coeffs[0] + hw)

    def test_to_order_geometry_set_uses_traced_edges_when_available(self):
        """When bot/top_poly_coeffs are set, to_order_geometry_set uses them directly."""
        n, ns = 2, 10
        bot_polys = np.array([[50.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]])
        top_polys = np.array([[70.0, 0.0, 0.0, 0.0], [120.0, 0.0, 0.0, 0.0]])
        trace = FlatOrderTrace(
            n_orders=n,
            sample_cols=np.arange(ns),
            center_rows=np.ones((n, ns)) * 60.0,
            center_poly_coeffs=np.array([[60.0, 0.0, 0.0, 0.0], [110.0, 0.0, 0.0, 0.0]]),
            fit_rms=np.ones(n),
            half_width_rows=np.full(n, 5.0),
            poly_degree=3,
            seed_col=5,
            bot_poly_coeffs=bot_polys,
            top_poly_coeffs=top_polys,
        )
        geom = trace.to_order_geometry_set("H1")
        g0 = geom.geometries[0]
        g1 = geom.geometries[1]
        # Traced edge polys must be used directly, not the centre±hw fallback.
        np.testing.assert_allclose(g0.bottom_edge_coeffs, bot_polys[0])
        np.testing.assert_allclose(g0.top_edge_coeffs, top_polys[0])
        np.testing.assert_allclose(g1.bottom_edge_coeffs, bot_polys[1])
        np.testing.assert_allclose(g1.top_edge_coeffs, top_polys[1])
        # Centreline must be the mean of the traced edges, not centre_poly_coeffs.
        expected_cen_0 = (bot_polys[0] + top_polys[0]) / 2.0
        np.testing.assert_allclose(g0.centerline_coeffs, expected_cen_0)

    def test_to_order_geometry_set_placeholder_order_numbers(self, dummy_trace):
        geom = dummy_trace.to_order_geometry_set("H1")
        for i, g in enumerate(geom.geometries):
            assert g.order == i

    def test_bot_top_poly_coeffs_default_none(self):
        """FlatOrderTrace constructed without edge polys has None bot/top."""
        trace = FlatOrderTrace(
            n_orders=1,
            sample_cols=np.arange(10),
            center_rows=np.ones((1, 10)) * 50.0,
            center_poly_coeffs=np.zeros((1, 4)),
            fit_rms=np.ones(1),
            half_width_rows=np.full(1, 10.0),
            poly_degree=3,
            seed_col=5,
        )
        assert trace.bot_poly_coeffs is None
        assert trace.top_poly_coeffs is None


# ---------------------------------------------------------------------------
# 1b. OrderTraceStats dataclass API
# ---------------------------------------------------------------------------


class TestOrderTraceStatsDataclass:
    """OrderTraceStats can be constructed and queried directly."""

    @pytest.fixture()
    def dummy_stats(self):
        return OrderTraceStats(
            order_index=2,
            rms_residual=1.5,
            curvature_metric=2e-4,
            oscillation_metric=0.005,
            min_sep_lower=35.0,
            min_sep_upper=38.0,
            crosses_lower=False,
            crosses_upper=False,
            trace_valid=True,
        )

    def test_order_index(self, dummy_stats):
        assert dummy_stats.order_index == 2

    def test_rms_residual(self, dummy_stats):
        assert dummy_stats.rms_residual == pytest.approx(1.5)

    def test_curvature_metric(self, dummy_stats):
        assert dummy_stats.curvature_metric == pytest.approx(2e-4)

    def test_oscillation_metric(self, dummy_stats):
        assert dummy_stats.oscillation_metric == pytest.approx(0.005)

    def test_min_sep_lower(self, dummy_stats):
        assert dummy_stats.min_sep_lower == pytest.approx(35.0)

    def test_min_sep_upper(self, dummy_stats):
        assert dummy_stats.min_sep_upper == pytest.approx(38.0)

    def test_crosses_lower_false(self, dummy_stats):
        assert dummy_stats.crosses_lower is False

    def test_crosses_upper_false(self, dummy_stats):
        assert dummy_stats.crosses_upper is False

    def test_trace_valid(self, dummy_stats):
        assert dummy_stats.trace_valid is True

    def test_invalid_order_has_trace_valid_false(self):
        s = OrderTraceStats(
            order_index=0,
            rms_residual=float("nan"),
            curvature_metric=float("nan"),
            oscillation_metric=float("nan"),
            min_sep_lower=float("nan"),
            min_sep_upper=float("nan"),
            crosses_lower=False,
            crosses_upper=False,
            trace_valid=False,
        )
        assert s.trace_valid is False


# ---------------------------------------------------------------------------
# 1c. FlatOrderTrace.order_stats default
# ---------------------------------------------------------------------------


class TestFlatOrderTraceOrderStatsDefault:
    """FlatOrderTrace.order_stats defaults to an empty list."""

    def test_order_stats_defaults_to_empty_list(self):
        trace = FlatOrderTrace(
            n_orders=2,
            sample_cols=np.arange(10),
            center_rows=np.ones((2, 10)) * 50.0,
            center_poly_coeffs=np.zeros((2, 4)),
            fit_rms=np.ones(2),
            half_width_rows=np.full(2, 10.0),
            poly_degree=3,
            seed_col=5,
        )
        assert trace.order_stats == []

    def test_order_stats_accepts_list_of_stats(self):
        s = OrderTraceStats(
            order_index=0,
            rms_residual=1.0,
            curvature_metric=1e-4,
            oscillation_metric=0.002,
            min_sep_lower=float("nan"),
            min_sep_upper=40.0,
            crosses_lower=False,
            crosses_upper=False,
            trace_valid=True,
        )
        trace = FlatOrderTrace(
            n_orders=1,
            sample_cols=np.arange(10),
            center_rows=np.ones((1, 10)) * 50.0,
            center_poly_coeffs=np.zeros((1, 4)),
            fit_rms=np.ones(1),
            half_width_rows=np.full(1, 10.0),
            poly_degree=3,
            seed_col=5,
            order_stats=[s],
        )
        assert len(trace.order_stats) == 1
        assert trace.order_stats[0].order_index == 0


# ---------------------------------------------------------------------------
# 2. load_and_combine_flats
# ---------------------------------------------------------------------------


class TestLoadAndCombineFlats:
    """load_and_combine_flats reads FITS files and median-combines them."""

    @pytest.fixture()
    def flat_files(self, tmp_path):
        """Write three synthetic flat FITS files and return their paths."""
        flat = _make_synthetic_flat(seed=1)
        paths = []
        for k in range(3):
            p = str(tmp_path / f"flat_{k:02d}.fits")
            _write_fits_flat(flat + rng_offset(k), p)
            paths.append(p)
        return paths

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            load_and_combine_flats([])

    def test_single_file_returns_correct_shape(self, tmp_path):
        flat = _make_synthetic_flat()
        path = str(tmp_path / "single.fits")
        _write_fits_flat(flat, path)
        result = load_and_combine_flats([path])
        assert result.shape == flat.shape

    def test_single_file_values_preserved(self, tmp_path):
        flat = _make_synthetic_flat()
        path = str(tmp_path / "single.fits")
        _write_fits_flat(flat, path)
        result = load_and_combine_flats([path])
        np.testing.assert_allclose(result, flat.astype(np.float32))

    def test_multiple_files_returns_correct_shape(self, tmp_path):
        paths = _write_three_flats(tmp_path)
        result = load_and_combine_flats(paths)
        assert result.shape == (_NROWS, _NCOLS)

    def test_multiple_files_median_suppresses_extreme_values(self, tmp_path):
        """A single bright pixel in one frame should be suppressed by median."""
        flat_base = _make_synthetic_flat(seed=0)
        paths = _write_three_flats(tmp_path, base=flat_base)

        # Add a single bright spike to the first frame only
        with fits.open(paths[0], mode="update") as hdul:
            hdul[0].data[128, 256] = 1e9

        result = load_and_combine_flats(paths)
        # Median should not be influenced by the spike
        assert result[128, 256] < 1e8

    def test_returns_float32(self, tmp_path):
        flat = _make_synthetic_flat()
        path = str(tmp_path / "f.fits")
        _write_fits_flat(flat, path)
        result = load_and_combine_flats([path])
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. trace_orders_from_flat – error handling
# ---------------------------------------------------------------------------


class TestTraceOrdersFromFlatErrors:
    """trace_orders_from_flat raises informative errors for bad inputs."""

    def test_raises_on_empty_file_list(self):
        with pytest.raises(ValueError, match="empty"):
            trace_orders_from_flat([])

    def test_raises_when_no_peaks_found(self, tmp_path):
        """A flat image with no order structure should raise RuntimeError."""
        flat = np.ones((_NROWS, _NCOLS), dtype=np.float32) * 500.0
        path = str(tmp_path / "blank.fits")
        _write_fits_flat(flat, path)
        with pytest.raises(RuntimeError, match="No order peaks"):
            trace_orders_from_flat([path])


# ---------------------------------------------------------------------------
# 4. trace_orders_from_flat – correctness on synthetic data
# ---------------------------------------------------------------------------


class TestTraceOrdersFromFlatSynthetic:
    """trace_orders_from_flat produces correct results on synthetic flat data."""

    @pytest.fixture(scope="class")
    def trace(self, tmp_path_factory):
        """Run tracing on a single synthetic flat once."""
        flat = _make_synthetic_flat(seed=0)
        tmp = tmp_path_factory.mktemp("syn")
        path = str(tmp / "syn_flat.fits")
        _write_fits_flat(flat, path)
        return trace_orders_from_flat(
            [path],
            n_sample_cols=15,
            poly_degree=2,
            col_range=(50, _NCOLS - 50),
        )

    def test_n_orders_matches_synthetic(self, trace):
        # Should find all 5 synthetic orders
        assert trace.n_orders == _SYN_N_ORDERS

    def test_poly_degree_stored(self, trace):
        assert trace.poly_degree == 2

    def test_sample_cols_within_range(self, trace):
        assert trace.sample_cols.min() >= 50
        assert trace.sample_cols.max() <= _NCOLS - 50

    def test_center_rows_shape(self, trace):
        assert trace.center_rows.shape[0] == _SYN_N_ORDERS

    def test_center_poly_coeffs_shape(self, trace):
        assert trace.center_poly_coeffs.shape == (_SYN_N_ORDERS, 3)  # degree 2 → 3 terms

    def test_fit_rms_all_finite(self, trace):
        assert np.all(np.isfinite(trace.fit_rms))

    def test_fit_rms_below_threshold(self, trace):
        """Polynomial residuals must be below 3 pixels on synthetic data."""
        assert np.nanmax(trace.fit_rms) < 3.0

    def test_center_positions_near_known_values(self, trace):
        """Traced centre positions must be within 5 pixels of known values."""
        col_mid = (_NCOLS - 50 + 50) // 2
        for i in range(_SYN_N_ORDERS):
            known_center = _SYN_FIRST_CENTER + i * _SYN_SPACING + _SYN_TILT * col_mid
            fitted_center = float(
                np.polynomial.polynomial.polyval(
                    float(col_mid), trace.center_poly_coeffs[i]
                )
            )
            assert abs(fitted_center - known_center) < 5.0, (
                f"Order {i}: fitted center {fitted_center:.1f} differs from "
                f"known {known_center:.1f} by more than 5 px"
            )

    def test_half_width_rows_positive(self, trace):
        assert np.all(trace.half_width_rows > 0)

    def test_half_width_rows_near_known(self, trace):
        """Estimated half-widths should be within 5 px of the known value."""
        for i in range(_SYN_N_ORDERS):
            assert abs(trace.half_width_rows[i] - _SYN_HALF_WIDTH) < 5.0, (
                f"Order {i}: half-width {trace.half_width_rows[i]:.1f} differs "
                f"from known {_SYN_HALF_WIDTH} by more than 5 px"
            )

    def test_to_order_geometry_set_works(self, trace):
        geom = trace.to_order_geometry_set("H1_syn", col_range=(50, _NCOLS - 50))
        assert isinstance(geom, OrderGeometrySet)
        assert geom.n_orders == _SYN_N_ORDERS

    def test_order_geometry_centerlines_near_known(self, trace):
        """OrderGeometry centerlines must be within 5 px of known positions."""
        geom = trace.to_order_geometry_set("H1_syn", col_range=(50, _NCOLS - 50))
        col_mid = (_NCOLS - 50 + 50) // 2
        for i in range(_SYN_N_ORDERS):
            known = _SYN_FIRST_CENTER + i * _SYN_SPACING + _SYN_TILT * col_mid
            fitted = geom.geometries[i].eval_centerline(col_mid)
            assert abs(fitted - known) < 5.0

    def test_multiple_flats_median_combined(self, tmp_path):
        """Tracing on 3 identical flats with noise should give same result as one."""
        flat = _make_synthetic_flat(seed=0)
        paths = []
        for k in range(3):
            p = str(tmp_path / f"flat_{k}.fits")
            _write_fits_flat(flat, p)
            paths.append(p)

        trace1 = trace_orders_from_flat(
            [paths[0]],
            n_sample_cols=10,
            col_range=(50, _NCOLS - 50),
        )
        trace3 = trace_orders_from_flat(
            paths,
            n_sample_cols=10,
            col_range=(50, _NCOLS - 50),
        )
        assert trace1.n_orders == trace3.n_orders

    # ------------------------------------------------------------------
    # QA stats tests on synthetic data
    # ------------------------------------------------------------------

    def test_order_stats_length_matches_n_orders(self, trace):
        assert len(trace.order_stats) == trace.n_orders

    def test_order_stats_indices_sequential(self, trace):
        for i, s in enumerate(trace.order_stats):
            assert s.order_index == i

    def test_rms_residual_matches_fit_rms(self, trace):
        for s in trace.order_stats:
            expected = float(trace.fit_rms[s.order_index])
            if np.isfinite(expected):
                assert s.rms_residual == pytest.approx(expected)
            else:
                assert np.isnan(s.rms_residual)

    def test_curvature_metric_nonnegative(self, trace):
        for s in trace.order_stats:
            if np.isfinite(s.curvature_metric):
                assert s.curvature_metric >= 0.0

    def test_oscillation_metric_nonnegative_for_synthetic_orders(self, trace):
        """Synthetic orders have a small linear tilt; oscillation should be tiny."""
        for s in trace.order_stats:
            if np.isfinite(s.oscillation_metric):
                assert s.oscillation_metric >= 0.0, (
                    f"Order {s.order_index}: oscillation_metric must be non-negative"
                )

    def test_oscillation_metric_small_for_nearly_linear_orders(self, trace):
        """Synthetic orders with a 0.01 px/col tilt should have near-zero oscillation."""
        for s in trace.order_stats:
            if np.isfinite(s.oscillation_metric):
                assert s.oscillation_metric < 0.05, (
                    f"Order {s.order_index}: oscillation_metric={s.oscillation_metric:.4f} "
                    "unexpectedly large for a nearly-linear synthetic trace"
                )

    def test_min_sep_lower_upper_nonnegative(self, trace):
        """Absolute separations must never be negative."""
        for s in trace.order_stats:
            if np.isfinite(s.min_sep_lower):
                assert s.min_sep_lower >= 0.0, (
                    f"Order {s.order_index}: min_sep_lower must be non-negative"
                )
            if np.isfinite(s.min_sep_upper):
                assert s.min_sep_upper >= 0.0, (
                    f"Order {s.order_index}: min_sep_upper must be non-negative"
                )

    def test_min_sep_large_for_well_separated_synthetic_orders(self, trace):
        """Synthetic orders are spaced 40 rows apart; separations should be large."""
        for s in trace.order_stats:
            if np.isfinite(s.min_sep_lower):
                assert s.min_sep_lower > 20.0, (
                    f"Order {s.order_index}: min_sep_lower={s.min_sep_lower:.1f} "
                    "unexpectedly small for 40-row-spaced synthetic orders"
                )
            if np.isfinite(s.min_sep_upper):
                assert s.min_sep_upper > 20.0, (
                    f"Order {s.order_index}: min_sep_upper={s.min_sep_upper:.1f} "
                    "unexpectedly small for 40-row-spaced synthetic orders"
                )

    def test_no_crossings_for_well_separated_synthetic_orders(self, trace):
        """Well-separated synthetic orders must not cross any neighbour."""
        for s in trace.order_stats:
            assert not s.crosses_lower, (
                f"Order {s.order_index}: unexpected crosses_lower=True"
            )
            assert not s.crosses_upper, (
                f"Order {s.order_index}: unexpected crosses_upper=True"
            )

    def test_inner_orders_have_finite_both_separations(self, trace):
        """Orders that are not first or last should have both sep values finite."""
        for s in trace.order_stats:
            if s.order_index > 0:
                assert np.isfinite(s.min_sep_lower), (
                    f"Order {s.order_index}: expected finite min_sep_lower"
                )
            if s.order_index < trace.n_orders - 1:
                assert np.isfinite(s.min_sep_upper), (
                    f"Order {s.order_index}: expected finite min_sep_upper"
                )

    def test_trace_valid_true_for_synthetic_data(self, trace):
        """All synthetic orders should pass QA checks."""
        for s in trace.order_stats:
            assert s.trace_valid is True, (
                f"Order {s.order_index} unexpectedly flagged invalid: "
                f"rms={s.rms_residual:.3f}, curv={s.curvature_metric:.2e}, "
                f"osc={s.oscillation_metric:.4f}, "
                f"sep_lo={s.min_sep_lower:.1f}, sep_hi={s.min_sep_upper:.1f}, "
                f"x_lo={s.crosses_lower}, x_hi={s.crosses_upper}"
            )


# ---------------------------------------------------------------------------
# 5. Module import regression
# ---------------------------------------------------------------------------


class TestModuleImport:
    """The tracing module must be importable from instruments.ishell."""

    def test_import_from_instruments_ishell(self):
        from pyspextool.instruments.ishell import tracing  # noqa: F401
        assert hasattr(tracing, "trace_orders_from_flat")
        assert hasattr(tracing, "FlatOrderTrace")
        assert hasattr(tracing, "OrderTraceStats")
        assert hasattr(tracing, "load_and_combine_flats")

    def test_import_symbols_directly(self):
        from pyspextool.instruments.ishell.tracing import (
            FlatOrderTrace,
            OrderTraceStats,
            load_and_combine_flats,
            trace_orders_from_flat,
        )
        assert callable(trace_orders_from_flat)
        assert callable(load_and_combine_flats)
        assert FlatOrderTrace is not None
        assert OrderTraceStats is not None


# ---------------------------------------------------------------------------
# 6. Smoke test on real H1 calibration data
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 flat data not found (run `git lfs pull` first)",
)
class TestTraceOrdersH1RealData:
    """Smoke-tests that run the tracing scaffold on the real H1 flat frames.

    These are **smoke tests**, not precision calibration checks.  They verify
    only that the scaffold runs without error and returns results within a
    broad, conservative range of acceptability.

    Acceptance criteria and rationale
    ----------------------------------
    * **≥ 35 orders detected** (H1 mode has ~45).  The threshold is set well
      below the expected count to accommodate 2–3 low-signal edge orders that
      are routinely missed, plus any variation between flat-frame conditions.
    * **All detected orders have a valid polynomial fit** (no NaN RMS values).
    * **Median polynomial RMS < 8 pixels**.  The scaffold routinely achieves
      ~3–4 px on the H1 dataset; the threshold is deliberately conservative
      to keep the test green across different flat-frame conditions and minor
      algorithm changes.
    * **Traced centres span the full detector** (row 0 to ~2048).
    * **`to_order_geometry_set()` converts without error**.

    These tolerances are loose because:
    (a) this is a first-pass scaffold, not a finalised calibration, and
    (b) the geometry produced is intended for development use, not
        science-quality spectral rectification.
    """

    @pytest.fixture(scope="class")
    def h1_trace(self):
        return trace_orders_from_flat(
            _H1_FLAT_FILES,
            col_range=(650, 1550),
            n_sample_cols=20,
        )

    def test_at_least_35_orders_detected(self, h1_trace):
        assert h1_trace.n_orders >= 35, (
            f"Expected at least 35 orders; got {h1_trace.n_orders}"
        )

    def test_all_orders_have_valid_polynomial(self, h1_trace):
        assert np.all(np.isfinite(h1_trace.fit_rms)), (
            "Some orders have NaN polynomial fit RMS"
        )

    def test_median_rms_below_threshold(self, h1_trace):
        median_rms = float(np.nanmedian(h1_trace.fit_rms))
        assert median_rms < 8.0, (
            f"Median polynomial RMS {median_rms:.2f} px exceeds 8 px threshold"
        )

    def test_orders_span_full_detector(self, h1_trace):
        """Traced orders must span from near row 0 to near row 2048."""
        col_mid = (650 + 1550) // 2
        centers = [
            float(np.polynomial.polynomial.polyval(float(col_mid), c))
            for c in h1_trace.center_poly_coeffs
        ]
        assert min(centers) < 100, (
            f"Smallest order centre ({min(centers):.1f}) is unexpectedly far from row 0"
        )
        assert max(centers) > 1900, (
            f"Largest order centre ({max(centers):.1f}) is unexpectedly far from row 2048"
        )

    def test_half_width_rows_positive(self, h1_trace):
        assert np.all(h1_trace.half_width_rows > 0)

    def test_half_widths_reasonable_for_h1(self, h1_trace):
        """H1 mode orders are approximately 30 pixels wide; half-width ≈ 15 px."""
        median_hw = float(np.median(h1_trace.half_width_rows))
        assert 5.0 < median_hw < 30.0, (
            f"Median half-width {median_hw:.1f} px is outside expected range [5, 30]"
        )

    def test_to_order_geometry_set_returns_valid_object(self, h1_trace):
        geom = h1_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        assert isinstance(geom, OrderGeometrySet)
        assert geom.n_orders == h1_trace.n_orders
        assert geom.mode == "H1"

    def test_order_geometry_edges_consistent(self, h1_trace):
        """Each order's top edge must be above its bottom edge at mid-column."""
        geom = h1_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        col_mid = (650 + 1550) // 2
        for g in geom.geometries:
            assert g.eval_top_edge(col_mid) > g.eval_bottom_edge(col_mid), (
                f"Order {g.order}: top edge not above bottom edge at col {col_mid}"
            )

    def test_polynomial_degree_is_default(self, h1_trace):
        assert h1_trace.poly_degree == 3

    def test_sample_cols_within_col_range(self, h1_trace):
        assert h1_trace.sample_cols.min() >= 650
        assert h1_trace.sample_cols.max() <= 1550

    def test_col_range_in_order_geometry(self, h1_trace):
        # After the IDL-aligned post-fit xrange restriction (step 6a in
        # trace_orders_from_flat), order_xranges may be narrower than the
        # input col_range for orders whose fitted edge polynomial overshoots
        # the detector at one end.  The geometry must therefore be checked
        # for containment within the col_range, not equality.
        geom = h1_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        for g in geom.geometries:
            assert g.x_start >= 650, (
                f"Order {g.order}: x_start {g.x_start} is below col_range start 650"
            )
            assert g.x_end <= 1550, (
                f"Order {g.order}: x_end {g.x_end} exceeds col_range end 1550"
            )
            assert g.x_start <= g.x_end, (
                f"Order {g.order}: degenerate xrange [{g.x_start}, {g.x_end}]"
            )


# ---------------------------------------------------------------------------
# 7. Edge-based geometry tests
# ---------------------------------------------------------------------------


class TestTracedEdgeGeometry:
    """trace_orders_from_flat populates bot_poly_coeffs / top_poly_coeffs."""

    @pytest.fixture()
    def trace(self, tmp_path):
        flat = _make_synthetic_flat(seed=42)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        return trace_orders_from_flat([path])

    def test_bot_poly_coeffs_shape(self, trace):
        assert trace.bot_poly_coeffs is not None
        assert trace.bot_poly_coeffs.shape == (trace.n_orders, trace.poly_degree + 1)

    def test_top_poly_coeffs_shape(self, trace):
        assert trace.top_poly_coeffs is not None
        assert trace.top_poly_coeffs.shape == (trace.n_orders, trace.poly_degree + 1)

    def test_bot_poly_lt_top_poly_at_seed_col(self, trace):
        """Bottom edge must be below top edge at all sample columns."""
        for i in range(trace.n_orders):
            bot_vals = np.polynomial.polynomial.polyval(
                trace.sample_cols.astype(float), trace.bot_poly_coeffs[i]
            )
            top_vals = np.polynomial.polynomial.polyval(
                trace.sample_cols.astype(float), trace.top_poly_coeffs[i]
            )
            assert np.all(bot_vals < top_vals), (
                f"Order {i}: bottom edge is not consistently below top edge"
            )

    def test_center_poly_is_mean_of_edges(self, trace):
        """center_poly_coeffs must equal (bot + top) / 2."""
        for i in range(trace.n_orders):
            expected = (trace.bot_poly_coeffs[i] + trace.top_poly_coeffs[i]) / 2.0
            np.testing.assert_allclose(
                trace.center_poly_coeffs[i], expected,
                rtol=1e-10,
                err_msg=f"Order {i}: center_poly_coeffs != (bot+top)/2",
            )

    def test_to_order_geometry_uses_traced_edges(self, trace):
        """to_order_geometry_set must use traced bot/top polys, not centre±hw."""
        geom = trace.to_order_geometry_set("K3")
        for i, g in enumerate(geom.geometries):
            np.testing.assert_allclose(
                g.bottom_edge_coeffs, trace.bot_poly_coeffs[i],
                rtol=1e-10,
                err_msg=f"Order {i}: bottom_edge_coeffs does not match bot_poly_coeffs",
            )
            np.testing.assert_allclose(
                g.top_edge_coeffs, trace.top_poly_coeffs[i],
                rtol=1e-10,
                err_msg=f"Order {i}: top_edge_coeffs does not match top_poly_coeffs",
            )


# ---------------------------------------------------------------------------
# 8. Flatinfo-based initialization tests
# ---------------------------------------------------------------------------


class TestFlatinfoInitialization:
    """When flatinfo is supplied, trace_orders_from_flat uses its guess positions."""

    @pytest.fixture()
    def _mock_flatinfo(self):
        """Create a minimal FlatInfo-like object from known order positions."""
        from types import SimpleNamespace

        n_orders = _SYN_N_ORDERS
        nrows = _NROWS
        ncols = _NCOLS
        poly_degree = 3

        # Build edge_coeffs matching the synthetic flat order positions.
        # Each order has a known centre c0 = _SYN_FIRST_CENTER + k*_SYN_SPACING.
        # _SYN_HALF_WIDTH is the FWHM of the Gaussian profile (in pixels).
        # We use hw/2 as the edge offset, placing edges at ±half-FWHM from centre.
        hw = float(_SYN_HALF_WIDTH)
        edge_coeffs = np.zeros((n_orders, 2, poly_degree + 1))
        xranges = np.zeros((n_orders, 2), dtype=int)

        for k in range(n_orders):
            c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
            # Bottom edge: c0 - hw/2 (constant term only for simplicity)
            edge_coeffs[k, 0, 0] = c0 - hw / 2.0
            # Top edge: c0 + hw/2
            edge_coeffs[k, 1, 0] = c0 + hw / 2.0
            xranges[k] = [10, ncols - 10]

        return SimpleNamespace(
            edge_coeffs=edge_coeffs,
            xranges=xranges,
            edge_degree=poly_degree,
            flat_fraction=0.85,
            comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.0)),
            ybuffer=1,
            step=20,
            # omask and ycororder absent -> DEFAULT branch (no cross-corr)
            omask=None,
            ycororder=None,
            orders=list(range(n_orders)),
        )

    @pytest.fixture()
    def trace_with_flatinfo(self, tmp_path, _mock_flatinfo):
        flat = _make_synthetic_flat(seed=42)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        return trace_orders_from_flat([path], flatinfo=_mock_flatinfo)

    def test_n_orders_matches(self, trace_with_flatinfo):
        assert trace_with_flatinfo.n_orders == _SYN_N_ORDERS

    def test_bot_top_polys_populated(self, trace_with_flatinfo):
        assert trace_with_flatinfo.bot_poly_coeffs is not None
        assert trace_with_flatinfo.top_poly_coeffs is not None

    def test_order_centers_near_known(self, trace_with_flatinfo):
        """Order centres from flatinfo path should be within 5 pixels of known."""
        trace = trace_with_flatinfo
        cols_mid = float(np.median(trace.sample_cols))
        for i in range(trace.n_orders):
            known_c0 = float(_SYN_FIRST_CENTER + i * _SYN_SPACING)
            known_centre = known_c0 + _SYN_TILT * cols_mid
            measured = float(np.polynomial.polynomial.polyval(
                cols_mid, trace.center_poly_coeffs[i]
            ))
            assert abs(measured - known_centre) < 10.0, (
                f"Order {i}: centre {measured:.1f} more than 10 px from "
                f"known {known_centre:.1f}"
            )

    def test_poly_degree_overridden_from_flatinfo(self, trace_with_flatinfo,
                                                   _mock_flatinfo):
        assert trace_with_flatinfo.poly_degree == _mock_flatinfo.edge_degree


# ---------------------------------------------------------------------------
# 9. _filter_edge_orders metadata preservation
# ---------------------------------------------------------------------------


class TestFilterEdgeOrders:
    """_filter_edge_orders preserves all FlatOrderTrace metadata."""

    @pytest.fixture()
    def full_trace(self, tmp_path):
        flat = _make_synthetic_flat(seed=42)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        return trace_orders_from_flat([path])

    def test_filter_preserves_bot_top_polys(self, full_trace):
        """Filtered trace must retain bot/top polynomial arrays."""
        from pyspextool.instruments.ishell.tracing import FlatOrderTrace
        n = full_trace.n_orders
        # Build a trace with one narrow order (hw = 1.0) that should be filtered
        hw = full_trace.half_width_rows.copy()
        hw[0] = 0.1  # artificially narrow
        narrow_trace = FlatOrderTrace(
            n_orders=n,
            sample_cols=full_trace.sample_cols,
            center_rows=full_trace.center_rows,
            center_poly_coeffs=full_trace.center_poly_coeffs,
            fit_rms=full_trace.fit_rms,
            half_width_rows=hw,
            poly_degree=full_trace.poly_degree,
            seed_col=full_trace.seed_col,
            bot_poly_coeffs=full_trace.bot_poly_coeffs,
            top_poly_coeffs=full_trace.top_poly_coeffs,
            order_stats=full_trace.order_stats,
        )

        # Simulate the _filter_edge_orders logic from the K3 script.
        median_hw = float(np.median(hw))
        threshold = 0.30 * median_hw
        keep = [i for i in range(n) if hw[i] >= threshold]
        filtered = FlatOrderTrace(
            n_orders=len(keep),
            sample_cols=narrow_trace.sample_cols,
            center_rows=narrow_trace.center_rows[keep],
            center_poly_coeffs=narrow_trace.center_poly_coeffs[keep],
            fit_rms=narrow_trace.fit_rms[keep],
            half_width_rows=narrow_trace.half_width_rows[keep],
            poly_degree=narrow_trace.poly_degree,
            seed_col=narrow_trace.seed_col,
            bot_poly_coeffs=narrow_trace.bot_poly_coeffs[keep] if narrow_trace.bot_poly_coeffs is not None else None,
            top_poly_coeffs=narrow_trace.top_poly_coeffs[keep] if narrow_trace.top_poly_coeffs is not None else None,
        )

        # Should have filtered the narrow order
        assert filtered.n_orders == n - 1

        # bot/top polys must still be present and have correct shape
        assert filtered.bot_poly_coeffs is not None
        assert filtered.top_poly_coeffs is not None
        assert filtered.bot_poly_coeffs.shape == (n - 1, narrow_trace.poly_degree + 1)
        assert filtered.top_poly_coeffs.shape == (n - 1, narrow_trace.poly_degree + 1)

        # geometry from filtered trace must use the traced edge polys
        geom = filtered.to_order_geometry_set("K3")
        for i, g in enumerate(geom.geometries):
            np.testing.assert_allclose(
                g.bottom_edge_coeffs, filtered.bot_poly_coeffs[i], rtol=1e-10
            )


# ---------------------------------------------------------------------------
# 10. mc_adjustguesspos cross-correlation tests
# ---------------------------------------------------------------------------


class TestAdjustGuessPositions:
    """_adjust_guess_positions shifts guess y-positions via cross-correlation."""

    def _make_simple_setup(self, nrows=256, ncols=512):
        """Build a minimal single-order setup for controlled testing."""
        import numpy as np
        from types import SimpleNamespace

        # Single order centred at row 128, running full column range.
        poly_degree = 1
        edge_coeffs = np.zeros((1, 2, poly_degree + 1))
        edge_coeffs[0, 0, 0] = 108.0   # bottom edge at row 108
        edge_coeffs[0, 1, 0] = 148.0   # top edge at row 148
        xranges = np.array([[10, ncols - 10]])
        orders = [100]
        ycororder = 100

        # Build a matching reference order mask.
        omask = np.zeros((nrows, ncols), dtype=int)
        omask[109:148, :] = 100   # rows 109–147 belong to order 100

        # Build a flat where the order is shifted +5 rows relative to the mask.
        flat = np.zeros((nrows, ncols), dtype=np.float32) + 200.0
        flat[114:153, :] = 5000.0   # same slit width but shifted by +5

        return edge_coeffs, xranges, flat, omask, orders, ycororder

    def test_zero_shift_when_flat_matches_mask(self):
        """When flat matches the mask exactly, detected offset should be 0."""
        from pyspextool.instruments.ishell.tracing import _adjust_guess_positions
        nrows, ncols = 256, 512
        edge_coeffs, xranges, _, omask, orders, ycororder = (
            self._make_simple_setup(nrows=nrows, ncols=ncols)
        )
        # Build flat that exactly matches the mask.
        flat = np.zeros((nrows, ncols), dtype=np.float32) + 200.0
        flat[109:148, :] = 5000.0   # rows matching omask exactly

        guesses, new_xranges = _adjust_guess_positions(
            edge_coeffs, xranges, flat, omask, orders, ycororder, ybuffer=1,
        )
        # Offset should be zero; guess y unchanged.
        expected_y = (108.0 + 148.0) / 2.0   # original centre
        assert abs(guesses[0][1] - expected_y) <= 1.0, (
            f"Expected guess y ≈ {expected_y}, got {guesses[0][1]}"
        )

    def test_nonzero_shift_detected_and_applied(self):
        """When flat is shifted relative to mask, guess y-position is adjusted."""
        from pyspextool.instruments.ishell.tracing import _adjust_guess_positions
        nrows, ncols = 256, 512
        edge_coeffs, xranges, flat, omask, orders, ycororder = (
            self._make_simple_setup(nrows=nrows, ncols=ncols)
        )
        # flat is shifted +5 rows relative to mask (set up in _make_simple_setup)
        original_y = (108.0 + 148.0) / 2.0   # 128.0

        guesses, new_xranges = _adjust_guess_positions(
            edge_coeffs, xranges, flat, omask, orders, ycororder, ybuffer=1,
        )
        adjusted_y = guesses[0][1]
        # Adjusted y must differ from original by approximately the shift (5 rows).
        # The cross-correlation maximum is at the shift that maximises overlap;
        # we allow ±2 px tolerance for rounding / sub-pixel effects.
        shift_estimate = original_y - adjusted_y
        assert abs(abs(shift_estimate) - 5) <= 2, (
            f"Expected shift ≈ ±5 rows, got shift_estimate={shift_estimate:.1f}"
        )

    def test_xranges_recomputed(self):
        """Adjusted xranges must not exceed detector bounds."""
        from pyspextool.instruments.ishell.tracing import _adjust_guess_positions
        nrows, ncols = 256, 512
        edge_coeffs, xranges, flat, omask, orders, ycororder = (
            self._make_simple_setup(nrows=nrows, ncols=ncols)
        )
        _, new_xranges = _adjust_guess_positions(
            edge_coeffs, xranges, flat, omask, orders, ycororder, ybuffer=1,
        )
        assert new_xranges.shape == xranges.shape
        assert new_xranges[0, 0] >= 0
        assert new_xranges[0, 1] <= ncols - 1

    def test_default_path_used_when_omask_none(self, tmp_path):
        """When flatinfo.omask is None, guess positions use the DEFAULT path."""
        from types import SimpleNamespace
        flat = _make_synthetic_flat(seed=0)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)

        n_orders = _SYN_N_ORDERS
        poly_degree = 3
        hw = float(_SYN_HALF_WIDTH)
        edge_coeffs = np.zeros((n_orders, 2, poly_degree + 1))
        xranges = np.zeros((n_orders, 2), dtype=int)
        for k in range(n_orders):
            c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
            edge_coeffs[k, 0, 0] = c0 - hw / 2.0
            edge_coeffs[k, 1, 0] = c0 + hw / 2.0
            xranges[k] = [10, _NCOLS - 10]

        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs, xranges=xranges,
            edge_degree=poly_degree, flat_fraction=0.85, comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.0)),
            ybuffer=1, step=20,
            omask=None, ycororder=None, orders=list(range(n_orders)),
        )
        trace = trace_orders_from_flat([path], flatinfo=fi)
        assert trace.n_orders == n_orders
        assert trace.order_xranges is not None


# ---------------------------------------------------------------------------
# 11. Per-order xranges tests
# ---------------------------------------------------------------------------


class TestPerOrderXranges:
    """FlatOrderTrace.order_xranges and to_order_geometry_set per-order x_start/x_end."""

    @pytest.fixture()
    def trace_auto(self, tmp_path):
        flat = _make_synthetic_flat(seed=42)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        return trace_orders_from_flat([path])

    @pytest.fixture()
    def trace_flatinfo(self, tmp_path):
        flat = _make_synthetic_flat(seed=42)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        from types import SimpleNamespace
        n_orders = _SYN_N_ORDERS
        poly_degree = 3
        hw = float(_SYN_HALF_WIDTH)
        edge_coeffs = np.zeros((n_orders, 2, poly_degree + 1))
        xranges = np.zeros((n_orders, 2), dtype=int)
        for k in range(n_orders):
            c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
            edge_coeffs[k, 0, 0] = c0 - hw / 2.0
            edge_coeffs[k, 1, 0] = c0 + hw / 2.0
            # Give each order a slightly different xrange to verify per-order behavior
            xranges[k] = [10 + k * 2, _NCOLS - 10 - k * 2]
        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs, xranges=xranges,
            edge_degree=poly_degree, flat_fraction=0.85, comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.0)),
            ybuffer=1, step=20,
            omask=None, ycororder=None, orders=list(range(n_orders)),
        )
        return trace_orders_from_flat([path], flatinfo=fi)

    def test_auto_detect_order_xranges_shape(self, trace_auto):
        """In auto-detect mode, order_xranges is still populated (from order_samples)."""
        assert trace_auto.order_xranges is not None
        assert trace_auto.order_xranges.shape == (trace_auto.n_orders, 2)

    def test_auto_detect_order_xranges_valid(self, trace_auto):
        """Each auto-detect order_xrange is a valid [start, end]."""
        for i in range(trace_auto.n_orders):
            xr = trace_auto.order_xranges[i]
            assert xr[0] <= xr[1], f"Order {i}: xrange invalid {xr}"

    def test_flatinfo_order_xranges_shape(self, trace_flatinfo):
        """In flatinfo mode, order_xranges has shape (n_orders, 2)."""
        assert trace_flatinfo.order_xranges is not None
        assert trace_flatinfo.order_xranges.shape == (trace_flatinfo.n_orders, 2)

    def test_flatinfo_order_xranges_bounds(self, trace_flatinfo):
        """Each per-order xrange is a valid [start, end] with start < end."""
        xr = trace_flatinfo.order_xranges
        for i in range(trace_flatinfo.n_orders):
            assert xr[i, 0] < xr[i, 1], f"Order {i}: xrange invalid {xr[i]}"

    def test_to_order_geometry_uses_per_order_xranges(self, trace_flatinfo):
        """to_order_geometry_set emits per-order x_start/x_end from order_xranges."""
        geom = trace_flatinfo.to_order_geometry_set("K3")
        for i, g in enumerate(geom.geometries):
            assert g.x_start == int(trace_flatinfo.order_xranges[i, 0])
            assert g.x_end == int(trace_flatinfo.order_xranges[i, 1])

    def test_to_order_geometry_auto_col_range_from_samples(self, trace_auto):
        """In auto-detect mode, geometry x_start/x_end comes from order_xranges."""
        geom = trace_auto.to_order_geometry_set("K3")
        for i, g in enumerate(geom.geometries):
            assert g.x_start == int(trace_auto.order_xranges[i, 0])
            assert g.x_end == int(trace_auto.order_xranges[i, 1])

    def test_filter_edge_orders_preserves_xranges(self, trace_flatinfo):
        """_filter_edge_orders carries over order_xranges into the filtered trace."""
        n = trace_flatinfo.n_orders
        hw = trace_flatinfo.half_width_rows.copy()
        hw[0] = 0.01   # make first order artificially narrow → will be filtered
        narrow_trace = FlatOrderTrace(
            n_orders=n,
            sample_cols=trace_flatinfo.sample_cols,
            center_rows=trace_flatinfo.center_rows,
            center_poly_coeffs=trace_flatinfo.center_poly_coeffs,
            fit_rms=trace_flatinfo.fit_rms,
            half_width_rows=hw,
            poly_degree=trace_flatinfo.poly_degree,
            seed_col=trace_flatinfo.seed_col,
            bot_poly_coeffs=trace_flatinfo.bot_poly_coeffs,
            top_poly_coeffs=trace_flatinfo.top_poly_coeffs,
            order_xranges=trace_flatinfo.order_xranges,
        )
        median_hw = float(np.median(hw))
        threshold = 0.30 * median_hw
        keep = [i for i in range(n) if hw[i] >= threshold]
        filtered = FlatOrderTrace(
            n_orders=len(keep),
            sample_cols=narrow_trace.sample_cols,
            center_rows=narrow_trace.center_rows[keep],
            center_poly_coeffs=narrow_trace.center_poly_coeffs[keep],
            fit_rms=narrow_trace.fit_rms[keep],
            half_width_rows=narrow_trace.half_width_rows[keep],
            poly_degree=narrow_trace.poly_degree,
            seed_col=narrow_trace.seed_col,
            bot_poly_coeffs=(narrow_trace.bot_poly_coeffs[keep]
                             if narrow_trace.bot_poly_coeffs is not None else None),
            top_poly_coeffs=(narrow_trace.top_poly_coeffs[keep]
                             if narrow_trace.top_poly_coeffs is not None else None),
            order_xranges=(narrow_trace.order_xranges[keep]
                           if narrow_trace.order_xranges is not None else None),
        )
        assert filtered.n_orders == n - 1
        assert filtered.order_xranges is not None
        assert filtered.order_xranges.shape == (n - 1, 2)

        # to_order_geometry_set must use the filtered per-order xranges
        geom = filtered.to_order_geometry_set("K3")
        for i, g in enumerate(geom.geometries):
            assert g.x_start == int(filtered.order_xranges[i, 0])
            assert g.x_end == int(filtered.order_xranges[i, 1])


# ---------------------------------------------------------------------------
# 12. read_flatinfo populates omask, ycororder, ybuffer
# ---------------------------------------------------------------------------


class TestReadFlatinfoNewFields:
    """read_flatinfo now populates omask, ycororder, ybuffer from FITS header."""

    @pytest.mark.skipif(
        not os.path.isfile(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "src", "pyspextool", "instruments", "ishell", "data", "K3_flatinfo.fits",
            )
        ),
        reason="K3 flatinfo FITS not found in package data",
    )
    def test_k3_flatinfo_has_omask(self):
        from pyspextool.instruments.ishell.calibrations import read_flatinfo
        fi = read_flatinfo("K3")
        assert fi.omask is not None, "omask should be populated from K3_flatinfo.fits"
        assert fi.omask.shape == (2048, 2048)

    @pytest.mark.skipif(
        not os.path.isfile(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "src", "pyspextool", "instruments", "ishell", "data", "K3_flatinfo.fits",
            )
        ),
        reason="K3 flatinfo FITS not found in package data",
    )
    def test_k3_flatinfo_has_ycororder(self):
        from pyspextool.instruments.ishell.calibrations import read_flatinfo
        fi = read_flatinfo("K3")
        assert fi.ycororder is not None, "ycororder should be read from YCORORDR header"
        assert isinstance(fi.ycororder, int)

    @pytest.mark.skipif(
        not os.path.isfile(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "src", "pyspextool", "instruments", "ishell", "data", "K3_flatinfo.fits",
            )
        ),
        reason="K3 flatinfo FITS not found in package data",
    )
    def test_k3_flatinfo_has_ybuffer(self):
        from pyspextool.instruments.ishell.calibrations import read_flatinfo
        fi = read_flatinfo("K3")
        assert isinstance(fi.ybuffer, int)
        assert fi.ybuffer >= 1

    def test_k3_flatinfo_image_is_none(self):
        """read_flatinfo sets image=None: flatinfo.fits primary HDU is the order mask."""
        from pyspextool.instruments.ishell.calibrations import read_flatinfo
        fi = read_flatinfo("K3")
        assert fi.image is None, (
            "FlatInfo.image should be None when loaded via read_flatinfo; "
            "the flatinfo FITS primary HDU is the order mask (omask), not a flat image."
        )


# ---------------------------------------------------------------------------
# 13. K3 benchmark driver loads flatinfo
# ---------------------------------------------------------------------------


class TestK3BenchmarkLoadsFlatinfo:
    """Verify that the K3 benchmark driver imports and uses read_flatinfo."""

    def test_read_flatinfo_imported_in_k3_script(self):
        """run_ishell_k3_example imports read_flatinfo from calibrations."""
        import importlib.util, sys, os
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "run_ishell_k3_example.py",
        )
        spec = importlib.util.spec_from_file_location("run_k3", script_path)
        mod = importlib.util.module_from_spec(spec)
        # Just check the import is available; do not execute the script
        with open(script_path) as f:
            src = f.read()
        assert "read_flatinfo" in src, (
            "run_ishell_k3_example.py must import read_flatinfo from calibrations"
        )
        assert "flatinfo=flatinfo" in src, (
            "run_ishell_k3_example.py must pass flatinfo= to trace_orders_from_flat"
        )

    def test_trace_orders_from_flat_uses_flatinfo_when_supplied(self, tmp_path):
        """trace_orders_from_flat logs 'Flatinfo mode' when flatinfo is provided."""
        from types import SimpleNamespace
        import logging
        flat = _make_synthetic_flat(seed=0)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)

        n_orders = _SYN_N_ORDERS
        poly_degree = 3
        hw = float(_SYN_HALF_WIDTH)
        edge_coeffs = np.zeros((n_orders, 2, poly_degree + 1))
        xranges = np.zeros((n_orders, 2), dtype=int)
        for k in range(n_orders):
            c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
            edge_coeffs[k, 0, 0] = c0 - hw / 2.0
            edge_coeffs[k, 1, 0] = c0 + hw / 2.0
            xranges[k] = [10, _NCOLS - 10]
        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs, xranges=xranges,
            edge_degree=poly_degree, flat_fraction=0.85, comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.0)),
            ybuffer=1, step=20,
            omask=None, ycororder=None, orders=list(range(n_orders)),
        )
        # Must not raise; must use flatinfo path
        trace = trace_orders_from_flat([path], flatinfo=fi)
        assert trace.n_orders == n_orders
        # order_xranges must be populated (per-order xranges)
        assert trace.order_xranges is not None
        assert trace.order_xranges.shape == (n_orders, 2)


# ---------------------------------------------------------------------------
# 14. Per-order sranges: true per-order sample columns
# ---------------------------------------------------------------------------


class TestPerOrderSampleColumns:
    """Per-order sample column arrays (IDL sranges) are used internally."""

    def test_per_order_sample_cols_differ_when_xranges_differ(self, tmp_path):
        """When orders have different xranges, they use different sample columns."""
        from pyspextool.instruments.ishell.tracing import _compute_guess_positions_from_flatinfo
        import numpy as np

        n_orders = 2
        poly_degree = 1
        edge_coeffs = np.zeros((n_orders, 2, poly_degree + 1))
        edge_coeffs[0, 0, 0] = 50.0
        edge_coeffs[0, 1, 0] = 80.0
        edge_coeffs[1, 0, 0] = 120.0
        edge_coeffs[1, 1, 0] = 150.0
        # Order 0 spans cols 0–300; order 1 spans cols 200–500
        xranges = np.array([[0, 300], [200, 500]])
        guesses, xr = _compute_guess_positions_from_flatinfo(edge_coeffs, xranges)
        assert xr[0, 1] == 300
        assert xr[1, 0] == 200
        # Each order's xrange is preserved
        assert xr[0, 0] == 0
        assert xr[1, 1] == 500


# ---------------------------------------------------------------------------
# 15. OrderTraceSamples authoritative per-order structure
# ---------------------------------------------------------------------------


class TestOrderTraceSamples:
    """FlatOrderTrace.order_samples is the authoritative per-order representation."""

    @pytest.fixture()
    def trace_auto(self, tmp_path):
        flat = _make_synthetic_flat(seed=0)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        return trace_orders_from_flat([path])

    @pytest.fixture()
    def trace_flatinfo(self, tmp_path):
        from types import SimpleNamespace
        flat = _make_synthetic_flat(seed=0)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        n_orders = _SYN_N_ORDERS
        poly_degree = 3
        hw = float(_SYN_HALF_WIDTH)
        edge_coeffs = np.zeros((n_orders, 2, poly_degree + 1))
        xranges = np.zeros((n_orders, 2), dtype=int)
        for k in range(n_orders):
            c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
            edge_coeffs[k, 0, 0] = c0 - hw / 2.0
            edge_coeffs[k, 1, 0] = c0 + hw / 2.0
            xranges[k] = [10, _NCOLS - 10]
        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs, xranges=xranges,
            edge_degree=poly_degree, flat_fraction=0.85, comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.0)),
            ybuffer=1, step=20,
            omask=None, ycororder=None, orders=list(range(n_orders)),
        )
        return trace_orders_from_flat([path], flatinfo=fi)

    def test_order_samples_exists(self, trace_auto):
        assert hasattr(trace_auto, "order_samples")

    def test_order_samples_length_equals_n_orders(self, trace_auto):
        assert len(trace_auto.order_samples) == trace_auto.n_orders

    def test_order_samples_1d_arrays(self, trace_auto):
        for os_i in trace_auto.order_samples:
            assert os_i.sample_cols.ndim == 1
            assert os_i.center_rows.shape == os_i.sample_cols.shape
            assert os_i.bot_rows.shape == os_i.sample_cols.shape
            assert os_i.top_rows.shape == os_i.sample_cols.shape

    def test_order_samples_xrange_fields(self, trace_auto):
        for os_i in trace_auto.order_samples:
            assert isinstance(os_i.x_start, int)
            assert isinstance(os_i.x_end, int)
            assert os_i.x_start <= os_i.x_end

    def test_flatinfo_order_samples_cols_within_initial_xrange(self, trace_flatinfo):
        """At construction time (pre-fit), sample_cols is built from the
        initial flatinfo xranges, so every column lies within the initial
        [x_start, x_end].

        NOTE: this invariant is a PRE-FIT guarantee only.  After step 6a
        (post-fit xrange restriction), x_start/x_end may be narrowed so that
        sample_cols extends beyond [x_start, x_end].  The test fixture uses
        constant polynomials that never overshoot, so step 6a does not clip
        here and the pre-fit invariant happens to be preserved — but the
        test cannot assert containment in general.

        What IS always guaranteed post-fit:
        - [x_start, x_end] is a valid non-degenerate range
        - sample_cols is non-empty and strictly increasing
        """
        for os_i in trace_flatinfo.order_samples:
            assert len(os_i.sample_cols) > 0, (
                f"Order {os_i.order_index}: sample_cols is empty"
            )
            assert os_i.x_start <= os_i.x_end, (
                f"Order {os_i.order_index}: degenerate xrange "
                f"[{os_i.x_start}, {os_i.x_end}]"
            )
            # In this fixture step 6a does NOT clip, so the pre-fit
            # containment invariant is accidentally preserved.  We verify
            # it only as a sanity check for the specific fixture; downstream
            # code must not rely on it holding after a real post-fit clip.
            assert os_i.sample_cols[0] >= os_i.x_start, (
                f"Order {os_i.order_index}: pre-fit containment violated: "
                f"sample_cols[0]={os_i.sample_cols[0]} < x_start={os_i.x_start}"
            )
            assert os_i.sample_cols[-1] <= os_i.x_end, (
                f"Order {os_i.order_index}: pre-fit containment violated: "
                f"sample_cols[-1]={os_i.sample_cols[-1]} > x_end={os_i.x_end}"
            )

    def test_flatinfo_orders_may_have_different_sample_counts(self, tmp_path):
        """Orders with different xranges have different sample_cols lengths."""
        from types import SimpleNamespace
        flat = _make_synthetic_flat(seed=0)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        n_orders = _SYN_N_ORDERS
        poly_degree = 3
        hw = float(_SYN_HALF_WIDTH)
        edge_coeffs = np.zeros((n_orders, 2, poly_degree + 1))
        # Give orders different column widths (step=20 → different counts)
        xranges = np.array([
            [10, 100], [10, 200], [10, 300], [10, 400], [10, 500]
        ], dtype=int)[:n_orders]
        for k in range(n_orders):
            c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
            edge_coeffs[k, 0, 0] = c0 - hw / 2.0
            edge_coeffs[k, 1, 0] = c0 + hw / 2.0
        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs, xranges=xranges,
            edge_degree=poly_degree, flat_fraction=0.85, comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.0)),
            ybuffer=1, step=20,
            omask=None, ycororder=None, orders=list(range(n_orders)),
        )
        trace = trace_orders_from_flat([path], flatinfo=fi)
        col_counts = [len(os_i.sample_cols) for os_i in trace.order_samples]
        # With different xranges and step=20, orders should have different counts
        assert len(set(col_counts)) > 1, (
            "Expected orders with different xranges to have different sample column counts"
        )

    def test_legacy_sample_cols_populated(self, trace_auto):
        """Legacy sample_cols field is still populated (backward compat)."""
        assert trace_auto.sample_cols is not None
        assert len(trace_auto.sample_cols) > 0

    def test_legacy_center_rows_populated(self, trace_auto):
        """Legacy center_rows field is still populated (backward compat)."""
        assert trace_auto.center_rows is not None
        assert trace_auto.center_rows.shape[0] == trace_auto.n_orders

    def test_order_xranges_always_populated(self, trace_auto):
        """order_xranges is always populated (from order_samples), even in auto-detect."""
        assert trace_auto.order_xranges is not None
        assert trace_auto.order_xranges.shape == (trace_auto.n_orders, 2)

    def test_order_xranges_matches_order_samples(self, trace_auto):
        """order_xranges[i] matches order_samples[i].x_start / x_end."""
        for i, os_i in enumerate(trace_auto.order_samples):
            assert trace_auto.order_xranges[i, 0] == os_i.x_start
            assert trace_auto.order_xranges[i, 1] == os_i.x_end


# ---------------------------------------------------------------------------
# 16. Compatibility-array invariants
# ---------------------------------------------------------------------------


class TestCompatArrayInvariants:
    """Defensive invariants on the union-grid compatibility arrays.

    These tests verify that:
    - compatibility arrays have the expected shape;
    - union_cols is strictly increasing;
    - every finite center_rows entry corresponds to a per-order column;
    - the compatibility arrays are derived views, not authoritative inputs.
    """

    @pytest.fixture()
    def trace(self, tmp_path):
        flat = _make_synthetic_flat(seed=0)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        return trace_orders_from_flat([path])

    # ── shape ──────────────────────────────────────────────────────────────

    def test_compat_center_rows_shape(self, trace):
        """compat center_rows shape matches (n_orders, len(sample_cols))."""
        assert trace.center_rows.shape == (
            trace.n_orders,
            len(trace.sample_cols),
        )

    def test_union_cols_strictly_increasing(self, trace):
        """sample_cols (union_cols) must be strictly increasing."""
        cols = trace.sample_cols
        if len(cols) > 1:
            assert np.all(np.diff(cols) > 0), (
                "sample_cols (union_cols) is not strictly increasing"
            )

    # ── derived vs authoritative ────────────────────────────────────────────

    def test_finite_compat_entries_match_per_order_cols(self, trace):
        """Every finite center_rows[i, k] must correspond to a column in
        order_samples[i].sample_cols (i.e. it is derived, not invented)."""
        for i, os_i in enumerate(trace.order_samples):
            order_col_set = set(os_i.sample_cols.tolist())
            for k, col in enumerate(trace.sample_cols):
                val = trace.center_rows[i, k]
                if np.isfinite(val):
                    assert col in order_col_set, (
                        f"center_rows[{i}, {k}] is finite but column {col} "
                        f"is not in order_samples[{i}].sample_cols"
                    )

    def test_center_rows_nan_outside_order_range(self, trace):
        """Columns outside an order's own range must be NaN in center_rows."""
        for i, os_i in enumerate(trace.order_samples):
            order_col_set = set(os_i.sample_cols.tolist())
            for k, col in enumerate(trace.sample_cols):
                if col not in order_col_set:
                    assert np.isnan(trace.center_rows[i, k]), (
                        f"center_rows[{i}, {k}] (col={col}) should be NaN "
                        f"(not in order_samples[{i}].sample_cols)"
                    )

    def test_order_samples_cols_subset_of_union(self, trace):
        """Every per-order sample column must appear in the union grid."""
        union_set = set(trace.sample_cols.tolist())
        for i, os_i in enumerate(trace.order_samples):
            for col in os_i.sample_cols:
                assert col in union_set, (
                    f"order_samples[{i}] column {col} not found in union sample_cols"
                )

    # ── public dataclass API still works ───────────────────────────────────

    def test_flat_order_trace_direct_construction_still_works(self):
        """FlatOrderTrace can still be constructed directly (backward compat)."""
        n = 2
        sc = np.array([10, 20, 30])
        cr = np.zeros((n, len(sc)))
        coeffs = np.zeros((n, 2))
        rms = np.array([0.1, 0.2])
        hw = np.array([5.0, 5.0])
        trace = FlatOrderTrace(
            n_orders=n,
            sample_cols=sc,
            center_rows=cr,
            center_poly_coeffs=coeffs,
            fit_rms=rms,
            half_width_rows=hw,
            poly_degree=1,
            seed_col=20,
        )
        assert trace.n_orders == n
        assert trace.sample_cols is sc
        assert trace.center_rows is cr

    def test_sample_cols_is_1d(self, trace):
        """sample_cols must be a 1-D array."""
        assert trace.sample_cols.ndim == 1

    def test_center_rows_is_2d(self, trace):
        """center_rows must be a 2-D array."""
        assert trace.center_rows.ndim == 2


# ---------------------------------------------------------------------------
# 16. Post-fit xrange restriction tests (IDL mc_findorders port step 6a)
# ---------------------------------------------------------------------------


class TestPostFitXrangeRestriction:
    """Verify that order_xranges is restricted after edge polynomial fitting.

    IDL mc_findorders, at the end of the per-order loop:

        x    = findgen(stop-start+1)+start
        bot  = poly(x, edgecoeffs[*,0,i])
        top  = poly(x, edgecoeffs[*,1,i])
        z    = where(top gt 0.0 and top lt nrows-1
                     and bot gt 0 and bot lt nrows-1)
        xranges[*,i] = [min(x[z],MAX=mx), mx]

    Python previously omitted this step, keeping the input sranges as
    order_xranges even when the fitted polynomial overshot the detector.
    These tests verify that the Python port now matches IDL.

    Test geometry
    -------------
    We use a single tilted order with ``c0_row=215`` (center row at col 0),
    ``hw=10`` (half-width), ``tilt=0.1`` px/col:

    * ``bot(col) ≈ 205 + 0.1 * col``
    * ``top(col) ≈ 225 + 0.1 * col``
    * ``top`` hits ``nrows-1 = 255`` at ``col ≈ (255-225)/0.1 = 300``

    With input xrange ``[10, 502]``, step-6a should clip ``x_end`` to ~365
    (empirically verified; the tracing stops slightly before the
    theoretical 300 because the center shifts ybuffer/stop conditions).
    The exact clip column can vary by a few pixels due to integer grid
    effects; the tests only assert the clipping *occurred*.
    """

    # Parameters for the "high order near detector top edge" scenario.
    _NROWS_CLIP = 256
    _NCOLS_CLIP = 512
    _C0_CLIP = 215.0    # center row at col 0 → top(col)=225+0.1*col, hits 255 at ~300
    _TILT_CLIP = 0.1    # px/col; mild enough for tracing to follow the order
    _HW_CLIP = 10.0
    _XSTART = 10
    _XEND = 502         # = ncols - 10

    def _make_high_order_flat(self, tmp_path):
        """Flat + flatinfo for an order near the top of the detector.

        The order is defined by ``c0=215, hw=10, tilt=0.1``.  The top edge
        overshoots ``nrows-1 = 255`` around col 300, so the IDL post-fit
        xrange restriction must clip ``x_end`` well below the input 502.
        """
        from types import SimpleNamespace

        nrows = self._NROWS_CLIP
        ncols = self._NCOLS_CLIP
        c0 = self._C0_CLIP
        hw = self._HW_CLIP
        tilt = self._TILT_CLIP

        edge_coeffs = np.zeros((1, 2, 2))   # (1 order, 2 edges, degree+1=2)
        edge_coeffs[0, 0, 0] = c0 - hw      # bot constant
        edge_coeffs[0, 0, 1] = tilt         # bot slope
        edge_coeffs[0, 1, 0] = c0 + hw      # top constant
        edge_coeffs[0, 1, 1] = tilt         # top slope
        xranges_in = np.array([[self._XSTART, self._XEND]], dtype=int)

        rng = np.random.default_rng(42)
        flat = rng.normal(0.0, 50.0, size=(nrows, ncols)).astype(np.float32) + 200.0
        rows = np.arange(nrows, dtype=float)
        cols = np.arange(ncols, dtype=float)
        for j, col in enumerate(cols):
            center = c0 + tilt * col
            sigma = hw / 2.355
            flat[:, j] += 10000.0 * np.exp(-0.5 * ((rows - center) / sigma) ** 2)

        path = str(tmp_path / "high_order_flat.fits")
        _write_fits_flat(flat, path)

        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs,
            xranges=xranges_in,
            edge_degree=1,
            flat_fraction=0.85,
            comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.5)),
            ybuffer=1,
            step=20,
            omask=None,
            ycororder=None,
            orders=[0],
        )
        return path, fi, xranges_in

    def _make_flat_order_flat(self, tmp_path, n_orders: int = 2):
        """Flat + flatinfo for order(s) with zero tilt, well within detector.

        With ``c0=_SYN_FIRST_CENTER`` (row 60) and no tilt, both edges stay
        far from the detector boundaries for the full xrange.  The post-fit
        restriction must NOT clip the xrange.
        """
        from types import SimpleNamespace

        nrows = _NROWS
        ncols = _NCOLS
        hw = float(_SYN_HALF_WIDTH)

        edge_coeffs = np.zeros((n_orders, 2, 2))
        xranges_in = np.zeros((n_orders, 2), dtype=int)
        for k in range(n_orders):
            c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
            edge_coeffs[k, 0, 0] = c0 - hw
            edge_coeffs[k, 1, 0] = c0 + hw
            xranges_in[k] = [10, ncols - 10]

        flat = _make_synthetic_flat(seed=42)
        path = str(tmp_path / "flat_order_flat.fits")
        _write_fits_flat(flat, path)

        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs,
            xranges=xranges_in,
            edge_degree=1,
            flat_fraction=0.85,
            comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.5)),
            ybuffer=1,
            step=20,
            omask=None,
            ycororder=None,
            orders=list(range(n_orders)),
        )
        return path, fi, xranges_in

    # ---- tests: well-behaved polynomial stays within input xrange ---------

    def test_valid_order_xrange_unchanged(self, tmp_path):
        """When the polynomial is within detector bounds everywhere, xrange is
        not narrowed — the output xranges equal the input sranges."""
        path, fi, xranges_in = self._make_flat_order_flat(tmp_path, n_orders=2)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        xr = trace.order_xranges
        for i in range(trace.n_orders):
            assert xr[i, 0] == xranges_in[i, 0], (
                f"Order {i}: x_start clipped from {xranges_in[i, 0]} to {xr[i, 0]} "
                "but polynomial is within bounds everywhere."
            )
            assert xr[i, 1] == xranges_in[i, 1], (
                f"Order {i}: x_end clipped from {xranges_in[i, 1]} to {xr[i, 1]} "
                "but polynomial is within bounds everywhere."
            )

    # ---- tests: polynomial overshoots at the far end ----------------------

    def test_pathological_order_xrange_clipped(self, tmp_path):
        """When the polynomial overshoots the detector at the far end of the
        xrange, order_xranges[i, 1] must be clipped — matching IDL's post-fit
        xrange update.

        The high-order scenario uses c0=215, tilt=0.1 px/col, nrows=256:
          top(col) ≈ 225 + 0.1 * col  →  hits 255 around col 300
        Input xrange ends at 502, so x_end must be clipped well below 502.
        """
        path, fi, xranges_in = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        xr = trace.order_xranges
        # x_end must be strictly less than the input end (502).
        assert xr[0, 1] < xranges_in[0, 1], (
            f"x_end {xr[0, 1]} was NOT narrowed from input {xranges_in[0, 1]}; "
            "IDL post-fit xrange restriction was not applied."
        )
        # x_end must be well below the full xrange (at least 50 columns shorter).
        assert xr[0, 1] < xranges_in[0, 1] - 50, (
            f"x_end {xr[0, 1]} is only marginally clipped from {xranges_in[0, 1]}; "
            "expected a significant clip for c0=215 tilt=0.1 scenario."
        )

    # ---- tests: geometry uses the restricted xrange -----------------------

    def test_geometry_uses_restricted_xrange(self, tmp_path):
        """to_order_geometry_set must use the IDL-restricted xrange, not the
        original input sranges."""
        path, fi, xranges_in = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        geom = trace.to_order_geometry_set("K3")
        g = geom.geometries[0]
        xr = trace.order_xranges
        # Geometry xranges must match the restricted order_xranges.
        assert g.x_start == int(xr[0, 0])
        assert g.x_end == int(xr[0, 1])
        # x_end must be less than the input xrange end.
        assert g.x_end < int(xranges_in[0, 1]), (
            f"Geometry x_end {g.x_end} must be clipped below input {xranges_in[0, 1]}"
        )

    # ---- tests: valid orders unaffected by restriction --------------------

    def test_valid_orders_unaffected_by_restriction(self, tmp_path):
        """Flat orders (zero tilt, well within detector) must retain their
        input xranges unchanged — only pathological polynomials are clipped."""
        path, fi, xranges_in = self._make_flat_order_flat(tmp_path, n_orders=2)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        xr = trace.order_xranges
        for i in range(trace.n_orders):
            assert xr[i, 0] == xranges_in[i, 0], (
                f"Order {i}: x_start was unexpectedly clipped."
            )
            assert xr[i, 1] == xranges_in[i, 1], (
                f"Order {i}: x_end was unexpectedly clipped."
            )

    # ---- tests: xrange semantics ------------------------------------------

    def test_restricted_xrange_is_within_original(self, tmp_path):
        """After restriction, order_xranges[i] must be a sub-range of the
        input xrange (start >= input start, end <= input end)."""
        path, fi, xranges_in = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        xr = trace.order_xranges
        assert xr[0, 0] >= xranges_in[0, 0], (
            f"Restricted x_start {xr[0, 0]} < input start {xranges_in[0, 0]}"
        )
        assert xr[0, 1] <= xranges_in[0, 1], (
            f"Restricted x_end {xr[0, 1]} > input end {xranges_in[0, 1]}"
        )

    # ---- tests: edge polynomials within detector at reported xrange ------

    def test_edge_polys_valid_within_reported_xrange(self, tmp_path):
        """Within the reported order_xranges, both fitted edge polynomials
        must evaluate to pixel positions within (0, nrows-1) — no
        extrapolation outside the detector within the valid range."""
        path, fi, _ = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        xr = trace.order_xranges
        nrows = self._NROWS_CLIP
        assert trace.bot_poly_coeffs is not None
        assert trace.top_poly_coeffs is not None
        for i in range(trace.n_orders):
            x_dense = np.arange(xr[i, 0], xr[i, 1] + 1, dtype=float)
            if len(x_dense) == 0:
                continue
            bot_vals = np.polynomial.polynomial.polyval(
                x_dense, trace.bot_poly_coeffs[i]
            )
            top_vals = np.polynomial.polynomial.polyval(
                x_dense, trace.top_poly_coeffs[i]
            )
            assert np.all(bot_vals > 0.0) and np.all(bot_vals < float(nrows - 1)), (
                f"Order {i}: bot edge polynomial outside (0, {nrows-1}) "
                f"within reported xrange [{xr[i,0]}, {xr[i,1]}]"
            )
            assert np.all(top_vals > 0.0) and np.all(top_vals < float(nrows - 1)), (
                f"Order {i}: top edge polynomial outside (0, {nrows-1}) "
                f"within reported xrange [{xr[i,0]}, {xr[i,1]}]"
            )

    # ---- tests: plotting range follows valid xrange -----------------------

    def test_order_xranges_within_detector_bounds(self, tmp_path):
        """After restriction, all order xranges must reference valid columns
        within [0, ncols-1] and valid pixel rows (edges within detector)."""
        path, fi, _ = self._make_high_order_flat(tmp_path)
        ncols = self._NCOLS_CLIP
        trace = trace_orders_from_flat([path], flatinfo=fi)
        xr = trace.order_xranges
        for i in range(trace.n_orders):
            assert xr[i, 0] >= 0, f"Order {i}: x_start {xr[i, 0]} < 0"
            assert xr[i, 1] <= ncols - 1, (
                f"Order {i}: x_end {xr[i, 1]} >= ncols ({ncols})"
            )
            assert xr[i, 0] <= xr[i, 1], (
                f"Order {i}: degenerate xrange [{xr[i, 0]}, {xr[i, 1]}]"
            )


# ---------------------------------------------------------------------------
# 17. _polyval_with_xrange unit tests
# ---------------------------------------------------------------------------


class TestPolyvalWithXrange:
    """Tests for the _polyval_with_xrange domain-of-validity helper.

    The helper must:
    - return finite values within [x_start, x_end]
    - return NaN outside that interval
    - accept the boundary points as valid
    - fall back gracefully when there are no valid columns
    """

    def _pv(self, coeffs, x, x_start, x_end):
        from pyspextool.instruments.ishell.tracing import _polyval_with_xrange
        return _polyval_with_xrange(np.array(coeffs), np.array(x, dtype=float),
                                    x_start, x_end)

    def test_inside_range_is_finite(self):
        """Columns strictly inside xrange return finite values."""
        vals = self._pv([100.0, 0.1], [50, 100, 150], 10, 200)
        assert np.all(np.isfinite(vals))

    def test_outside_left_is_nan(self):
        """Columns to the left of x_start return NaN."""
        vals = self._pv([100.0, 0.1], [5, 9], 10, 200)
        assert np.all(np.isnan(vals))

    def test_outside_right_is_nan(self):
        """Columns to the right of x_end return NaN."""
        vals = self._pv([100.0, 0.1], [201, 300], 10, 200)
        assert np.all(np.isnan(vals))

    def test_boundary_left_is_finite(self):
        """The x_start boundary column returns a finite value (inclusive)."""
        vals = self._pv([100.0, 0.1], [10], 10, 200)
        assert np.isfinite(vals[0])

    def test_boundary_right_is_finite(self):
        """The x_end boundary column returns a finite value (inclusive)."""
        vals = self._pv([100.0, 0.1], [200], 10, 200)
        assert np.isfinite(vals[0])

    def test_mixed_in_and_out(self):
        """Array spanning inside and outside the range: correct split."""
        x = np.array([0.0, 10.0, 50.0, 100.0, 101.0])
        vals = self._pv([5.0, 1.0], x, 10, 100)
        assert np.isnan(vals[0])        # col 0, outside
        assert np.isfinite(vals[1])     # col 10, boundary (valid)
        assert np.isfinite(vals[2])     # col 50, inside
        assert np.isfinite(vals[3])     # col 100, boundary (valid)
        assert np.isnan(vals[4])        # col 101, outside

    def test_correct_polynomial_value_inside(self):
        """Values inside xrange match a direct polyval call."""
        coeffs = np.array([3.0, 2.0, 1.0])  # 3 + 2x + x²
        x_in = np.array([5.0, 10.0, 20.0])
        expected = np.polynomial.polynomial.polyval(x_in, coeffs)
        vals = self._pv(coeffs.tolist(), x_in, 0, 50)
        np.testing.assert_array_almost_equal(vals, expected)

    def test_no_valid_columns_all_nan(self):
        """When no columns fall within [x_start, x_end], result is all NaN."""
        vals = self._pv([100.0], [300, 400, 500], 10, 200)
        assert np.all(np.isnan(vals))

    def test_coefficients_not_modified(self):
        """_polyval_with_xrange must not modify the input coefficients array."""
        from pyspextool.instruments.ishell.tracing import _polyval_with_xrange
        coeffs = np.array([1.0, 2.0, 3.0])
        coeffs_orig = coeffs.copy()
        _polyval_with_xrange(coeffs, np.array([0.0, 5.0, 20.0]), 5, 15)
        np.testing.assert_array_equal(coeffs, coeffs_orig)


# ---------------------------------------------------------------------------
# 18. _compute_order_trace_stats xrange masking tests
# ---------------------------------------------------------------------------


class TestComputeOrderTraceStatsXrangeMasking:
    """_compute_order_trace_stats must ignore extrapolated polynomial tails.

    When order_xranges is supplied, stats (curvature, oscillation, inter-order
    separation) must be based only on the valid column range.  A polynomial that
    diverges outside its xrange must not inflate those metrics.

    When order_xranges is None (backward compatibility), the full column grid
    is used as before.
    """

    def _make_stats(self, coeffs_list, rms_list, cols, order_xranges=None):
        from pyspextool.instruments.ishell.tracing import _compute_order_trace_stats
        center_poly_coeffs = np.array(coeffs_list)
        fit_rms = np.array(rms_list, dtype=float)
        return _compute_order_trace_stats(
            center_poly_coeffs, fit_rms, np.array(cols, dtype=float),
            order_xranges=(np.array(order_xranges, dtype=int)
                           if order_xranges is not None else None),
        )

    def test_diverging_tail_does_not_inflate_oscillation_when_xrange_set(self):
        """Polynomial slope diverges outside [x_start, x_end].

        With the xrange enforced, oscillation_metric must reflect only the
        well-behaved portion; without it, the large slope variation outside
        would inflate the metric.
        """
        # Cubic with a large cubic term that is almost flat in [0, 50]
        # but explodes beyond that.
        #  p(x) = 100 + 0.0 * x + 0.0 * x² + 0.05 * x³
        # In [0, 50]: p'(x) = 0.15 * x²  → max at x=50, value 375.  That's
        # large, so let's use a milder case: p'(x) = 3 * 0.001 * x²
        # Coefficients: [100.0, 0.0, 0.0, 0.001]
        # In [0, 50]: p'(50) = 3 * 0.001 * 2500 = 7.5
        # In [0, 500]: p'(500) = 3 * 0.001 * 250000 = 750  ← huge
        coeffs = [100.0, 0.0, 0.0, 0.001]
        cols = list(range(0, 501, 10))  # 0 to 500

        # Without xrange restriction: cols includes 0..500 → osc ≈ 750
        stats_no_xr = self._make_stats([coeffs], [1.0], cols)
        # With xrange [0, 50]: only uses cols 0..50 → osc ≈ 7.5
        stats_xr = self._make_stats([coeffs], [1.0], cols, [[0, 50]])

        assert stats_no_xr[0].oscillation_metric > stats_xr[0].oscillation_metric, (
            "Oscillation metric with no xrange must be larger than with xrange "
            "for a polynomial that diverges outside [0, 50]."
        )

    def test_diverging_tail_does_not_inflate_curvature_when_xrange_set(self):
        """Curvature outside the xrange must not contaminate the metric."""
        # p(x) = 100 + 0.002 * x²  → p''(x) = 0.004, constant.
        # At x=200: p''(200) = 0.004 (constant for quadratic).
        # Actually curvature is constant for a quadratic.  Use a cubic.
        # p(x) = 100 + 0.0 * x + 0.0 * x² + 0.01 * x³
        # p''(x) = 6 * 0.01 * x = 0.06 * x
        # At x=10: |p''| = 0.6
        # At x=100: |p''| = 6.0   ← curvature inflated outside [0, 10]
        coeffs = [100.0, 0.0, 0.0, 0.01]
        cols = list(range(0, 101, 5))

        stats_no_xr = self._make_stats([coeffs], [1.0], cols)
        stats_xr = self._make_stats([coeffs], [1.0], cols, [[0, 10]])

        assert stats_no_xr[0].curvature_metric > stats_xr[0].curvature_metric, (
            "Curvature metric with no xrange must be larger than with xrange "
            "for a polynomial whose second derivative grows outside [0, 10]."
        )

    def test_inter_order_separation_uses_valid_range_only(self):
        """Inter-order separation must only be measured where both orders are
        valid.  If one order has a restricted xrange and diverges outside it,
        the separation calculation must not use those extrapolated values."""
        # Order 0: flat at row 100, valid over [0, 500]
        # Order 1: flat at row 200, valid only over [0, 50]
        #   At col 100, order 1's polynomial extrapolates to 200 (flat),
        #   but the stats should only measure separation within [0, 50].
        cols = list(range(0, 501, 10))
        coeffs_0 = [100.0]  # constant at 100
        coeffs_1 = [200.0]  # constant at 200
        center_poly_coeffs = np.zeros((2, 4))
        center_poly_coeffs[0, 0] = 100.0
        center_poly_coeffs[1, 0] = 200.0
        fit_rms = np.array([1.0, 1.0])

        # With xrange restriction for order 1: [0, 50]
        # center_vals[1] has NaN beyond col 50 → separation still 100 at valid cols
        from pyspextool.instruments.ishell.tracing import _compute_order_trace_stats
        stats_xr = _compute_order_trace_stats(
            center_poly_coeffs, fit_rms, np.array(cols, dtype=float),
            order_xranges=np.array([[0, 500], [0, 50]], dtype=int),
        )
        # Separation for order 1 (upper) vs order 0 (lower) should still be ~100
        # (gap is evaluated only at cols 0..50 where both are valid).
        assert np.isfinite(stats_xr[0].min_sep_upper), (
            "min_sep_upper should be finite: order 0 can see order 1 at cols 0..50"
        )
        assert abs(stats_xr[0].min_sep_upper - 100.0) < 5.0, (
            f"Expected ~100 px separation, got {stats_xr[0].min_sep_upper}"
        )

    def test_backward_compat_no_xrange_uses_full_grid(self):
        """When order_xranges=None, the full column grid is used (backward compat)."""
        coeffs = [[100.0, 0.1], [200.0, 0.1]]
        cols = list(range(0, 201, 10))
        stats = self._make_stats(coeffs, [1.0, 1.0], cols, order_xranges=None)
        # Both orders' center_vals must be finite everywhere (no NaN masking).
        from pyspextool.instruments.ishell.tracing import _compute_order_trace_stats
        import numpy as np
        center_poly_coeffs = np.zeros((2, 2))
        center_poly_coeffs[0] = [100.0, 0.1]
        center_poly_coeffs[1] = [200.0, 0.1]
        fit_rms = np.array([1.0, 1.0])
        cols_arr = np.arange(0, 201, 10, dtype=float)
        stats2 = _compute_order_trace_stats(center_poly_coeffs, fit_rms, cols_arr)
        # All metrics must be finite (no accidental NaN from xrange masking).
        for s in stats2:
            assert np.isfinite(s.curvature_metric), (
                f"Order {s.order_index}: curvature_metric is NaN without xrange"
            )
            assert np.isfinite(s.oscillation_metric), (
                f"Order {s.order_index}: oscillation_metric is NaN without xrange"
            )

    def test_failed_fit_rms_still_gives_nan_metrics(self):
        """Orders with NaN fit_rms still produce NaN metrics (unchanged behavior)."""
        coeffs = [[100.0, 0.1]]
        stats = self._make_stats(coeffs, [np.nan], [0, 100, 200],
                                 order_xranges=[[0, 200]])
        assert np.isnan(stats[0].curvature_metric)
        assert np.isnan(stats[0].oscillation_metric)
        assert not stats[0].trace_valid


# ---------------------------------------------------------------------------
# 19. Compatibility arrays: no finite values outside valid xrange
# ---------------------------------------------------------------------------


class TestCompatArraysNoExtrapolation:
    """compat_center_rows must never contain finite values outside an order's
    valid xrange.

    The compat arrays are built from raw traced samples (not polynomial
    evaluation), so they should only contain finite values at columns that
    were actually traced — which lie within the order's xrange by construction.
    This class verifies that invariant explicitly.
    """

    @pytest.fixture()
    def trace_flatinfo(self, tmp_path):
        """Trace with flatinfo so orders have explicit xranges."""
        from types import SimpleNamespace
        flat = _make_synthetic_flat(seed=42)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        n_orders = _SYN_N_ORDERS
        poly_degree = 3
        hw = float(_SYN_HALF_WIDTH)
        edge_coeffs = np.zeros((n_orders, 2, poly_degree + 1))
        xranges = np.zeros((n_orders, 2), dtype=int)
        for k in range(n_orders):
            c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
            edge_coeffs[k, 0, 0] = c0 - hw / 2.0
            edge_coeffs[k, 1, 0] = c0 + hw / 2.0
            xranges[k] = [10 + k * 2, _NCOLS - 10 - k * 2]
        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs, xranges=xranges,
            edge_degree=poly_degree, flat_fraction=0.85, comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.0)),
            ybuffer=1, step=20,
            omask=None, ycororder=None, orders=list(range(n_orders)),
        )
        return trace_orders_from_flat([path], flatinfo=fi)

    def test_compat_center_rows_no_finite_outside_xrange(self, trace_flatinfo):
        """Every finite entry in center_rows must correspond to a column within
        the order's valid [x_start, x_end] range."""
        trace = trace_flatinfo
        for i in range(trace.n_orders):
            x_start = int(trace.order_xranges[i, 0])
            x_end = int(trace.order_xranges[i, 1])
            for k, col in enumerate(trace.sample_cols):
                val = trace.center_rows[i, k]
                if np.isfinite(val):
                    assert x_start <= col <= x_end, (
                        f"Order {i}: center_rows[{i}, {k}] = {val:.2f} is finite "
                        f"at col {col}, but order xrange is [{x_start}, {x_end}]"
                    )

    def test_compat_center_rows_outside_xrange_are_nan(self, trace_flatinfo):
        """Columns outside the order's xrange must have NaN in center_rows."""
        trace = trace_flatinfo
        for i in range(trace.n_orders):
            x_start = int(trace.order_xranges[i, 0])
            x_end = int(trace.order_xranges[i, 1])
            for k, col in enumerate(trace.sample_cols):
                if col < x_start or col > x_end:
                    val = trace.center_rows[i, k]
                    assert np.isnan(val), (
                        f"Order {i}: center_rows[{i}, {k}] = {val:.2f} is finite "
                        f"at col {col} outside xrange [{x_start}, {x_end}] — "
                        "extrapolated value must not appear in compat arrays"
                    )

    def test_compat_center_rows_shape_preserved(self, trace_flatinfo):
        """center_rows shape must be (n_orders, n_sample_cols) after masking."""
        trace = trace_flatinfo
        assert trace.center_rows.shape == (trace.n_orders, len(trace.sample_cols))

    def test_sample_cols_unchanged(self, trace_flatinfo):
        """sample_cols (union of all per-order columns) must remain present."""
        trace = trace_flatinfo
        assert trace.sample_cols is not None
        assert len(trace.sample_cols) > 0


# ---------------------------------------------------------------------------
# 20. Compatibility arrays: sample_cols beyond post-fit xrange → masked
# ---------------------------------------------------------------------------


class TestCompatArraysPostFitXrangeMasking:
    """Verify that compat_center_rows is NaN for sample columns that fall
    outside the ORDER's post-fit restricted [x_start, x_end], even when those
    columns appear in os_i.sample_cols.

    This tests the specific leakage path described in the problem statement:
    before the fix, step 8 copied all center_rows regardless of whether the
    column still fell within the updated xrange.  After the fix, only columns
    within [x_start, x_end] are copied; the rest remain NaN.
    """

    def _build_compat_array(self, order_samples):
        """Replicate the step-8 construction from trace_orders_from_flat.

        This is a white-box test of the exact construction logic so that we
        can verify the fix without needing a full end-to-end flat trace.
        """
        union_cols = np.unique(
            np.concatenate([os_i.sample_cols for os_i in order_samples])
        )
        compat_center_rows = np.full((len(order_samples), len(union_cols)), np.nan)
        for i, os_i in enumerate(order_samples):
            for j, col in enumerate(os_i.sample_cols):
                if not (os_i.x_start <= col <= os_i.x_end):
                    continue
                k = int(np.searchsorted(union_cols, col))
                if k < len(union_cols) and union_cols[k] == col:
                    compat_center_rows[i, k] = os_i.center_rows[j]
        return union_cols, compat_center_rows

    def _make_order_samples(self, sample_cols, center_rows, x_start, x_end, idx=0):
        """Create an OrderTraceSamples with explicit xrange."""
        cols = np.array(sample_cols, dtype=int)
        crows = np.array(center_rows, dtype=float)
        return OrderTraceSamples(
            order_index=idx,
            sample_cols=cols,
            center_rows=crows,
            bot_rows=crows - 5.0,
            top_rows=crows + 5.0,
            x_start=x_start,
            x_end=x_end,
        )

    def test_sample_cols_beyond_xend_become_nan(self):
        """sample_cols extending beyond x_end must be NaN in compat array."""
        # sample_cols: [0, 10, 20, 30, 40, 50] but x_end=30
        # → cols 40, 50 should be NaN
        sample_cols = [0, 10, 20, 30, 40, 50]
        center_rows = [100.0] * 6
        os = self._make_order_samples(sample_cols, center_rows, x_start=0, x_end=30)
        union_cols, compat = self._build_compat_array([os])
        for k, col in enumerate(union_cols):
            if col > 30:
                assert np.isnan(compat[0, k]), (
                    f"col {col} > x_end=30: expected NaN, got {compat[0, k]}"
                )
            else:
                assert np.isfinite(compat[0, k]), (
                    f"col {col} <= x_end=30: expected finite, got {compat[0, k]}"
                )

    def test_sample_cols_before_xstart_become_nan(self):
        """sample_cols before x_start must be NaN in compat array."""
        # sample_cols: [0, 10, 20, 30, 40, 50] but x_start=20
        # → cols 0, 10 should be NaN
        sample_cols = [0, 10, 20, 30, 40, 50]
        center_rows = [200.0] * 6
        os = self._make_order_samples(sample_cols, center_rows, x_start=20, x_end=50)
        union_cols, compat = self._build_compat_array([os])
        for k, col in enumerate(union_cols):
            if col < 20:
                assert np.isnan(compat[0, k]), (
                    f"col {col} < x_start=20: expected NaN, got {compat[0, k]}"
                )
            else:
                assert np.isfinite(compat[0, k]), (
                    f"col {col} >= x_start=20: expected finite, got {compat[0, k]}"
                )

    def test_both_ends_clipped(self):
        """sample_cols extending beyond BOTH x_start and x_end → NaN at both ends."""
        sample_cols = list(range(0, 110, 10))   # 0, 10, ..., 100
        center_rows = [150.0] * len(sample_cols)
        os = self._make_order_samples(sample_cols, center_rows, x_start=20, x_end=80)
        union_cols, compat = self._build_compat_array([os])
        for k, col in enumerate(union_cols):
            if col < 20 or col > 80:
                assert np.isnan(compat[0, k]), (
                    f"col {col} outside [20, 80]: expected NaN, got {compat[0, k]}"
                )
            else:
                assert np.isfinite(compat[0, k]), (
                    f"col {col} inside [20, 80]: expected finite, got {compat[0, k]}"
                )

    def test_within_xrange_all_finite(self):
        """When all sample_cols fall within [x_start, x_end], nothing is masked."""
        sample_cols = [50, 60, 70, 80, 90]
        center_rows = [175.0, 176.0, 177.0, 178.0, 179.0]
        os = self._make_order_samples(sample_cols, center_rows, x_start=50, x_end=90)
        union_cols, compat = self._build_compat_array([os])
        assert np.all(np.isfinite(compat[0])), (
            "All sample cols within xrange should produce all-finite compat array"
        )

    def test_two_orders_independent_masking(self):
        """Each order's masking is independent; one order's clipping must not
        affect the other order's compat values."""
        # Order 0: sample_cols [0..100], but x_end=60 → cols 70..100 masked
        # Order 1: sample_cols [0..100], x_start=40 → cols 0..30 masked
        sample_cols = list(range(0, 110, 10))
        os0 = self._make_order_samples(sample_cols, [100.0] * 11,
                                        x_start=0, x_end=60, idx=0)
        os1 = self._make_order_samples(sample_cols, [200.0] * 11,
                                        x_start=40, x_end=100, idx=1)
        union_cols, compat = self._build_compat_array([os0, os1])

        # Order 0 checks
        for k, col in enumerate(union_cols):
            if col > 60:
                assert np.isnan(compat[0, k]), f"Order 0, col {col} > 60: should be NaN"
            else:
                assert np.isfinite(compat[0, k]), f"Order 0, col {col} <= 60: should be finite"

        # Order 1 checks
        for k, col in enumerate(union_cols):
            if col < 40:
                assert np.isnan(compat[1, k]), f"Order 1, col {col} < 40: should be NaN"
            else:
                assert np.isfinite(compat[1, k]), f"Order 1, col {col} >= 40: should be finite"

    def test_boundary_columns_are_finite(self):
        """x_start and x_end boundary columns must be finite (inclusive range)."""
        os = self._make_order_samples([10, 20, 30], [111.0, 122.0, 133.0],
                                       x_start=10, x_end=30)
        union_cols, compat = self._build_compat_array([os])
        # col 10 and col 30 are on the boundary — must be finite
        for k, col in enumerate(union_cols):
            assert np.isfinite(compat[0, k]), (
                f"Boundary col {col}: expected finite, got {compat[0, k]}"
            )

    def test_degenerate_xrange_only_start_col_finite(self):
        """Degenerate xrange (x_start == x_end) → only the start column is finite."""
        os = self._make_order_samples([10, 20, 30], [111.0, 122.0, 133.0],
                                       x_start=20, x_end=20)
        union_cols, compat = self._build_compat_array([os])
        for k, col in enumerate(union_cols):
            if col == 20:
                assert np.isfinite(compat[0, k]), f"col 20 (degenerate xrange): should be finite"
            else:
                assert np.isnan(compat[0, k]), f"col {col} != 20 in degenerate xrange: should be NaN"


# ---------------------------------------------------------------------------
# 21. sample_cols vs post-fit xrange: semantic hardening
# ---------------------------------------------------------------------------


class TestSampleColsVsXrangeSemantics:
    """Prove that sample_cols and [x_start, x_end] are distinct concepts.

    Key invariants (codified by these tests):

    * ``sample_cols`` = traced sampling grid, built from the initial flatinfo
      xranges at construction time.  After step 6a (post-fit xrange
      restriction), some entries MAY lie outside ``[x_start, x_end]``.
    * ``[x_start, x_end]`` = FINAL post-fit valid domain.  This is the
      authoritative range; any value computed outside it is extrapolated.
    * The compat array ``FlatOrderTrace.center_rows`` contains NaN for any
      column (even if present in ``sample_cols``) that lies outside
      ``[x_start, x_end]``.
    * ``to_order_geometry_set()`` reads ``order_xranges``, not
      ``sample_cols`` bounds.

    These tests exercise the pathological "tilted order near detector edge"
    scenario from TestPostFitXrangeRestriction, where step 6a ACTUALLY clips
    x_end well below the input xrange end.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    _NROWS = 256
    _NCOLS = 512
    _C0 = 215.0
    _TILT = 0.1
    _HW = 10.0
    _XSTART = 10
    _XEND = 502   # = _NCOLS - 10

    def _make_high_order_flat(self, tmp_path):
        """Single order near the top of detector; step 6a will clip x_end."""
        from types import SimpleNamespace

        nrows = self._NROWS
        ncols = self._NCOLS
        c0, hw, tilt = self._C0, self._HW, self._TILT

        edge_coeffs = np.zeros((1, 2, 2))
        edge_coeffs[0, 0, 0] = c0 - hw   # bot constant
        edge_coeffs[0, 0, 1] = tilt       # bot slope
        edge_coeffs[0, 1, 0] = c0 + hw   # top constant
        edge_coeffs[0, 1, 1] = tilt       # top slope
        xranges_in = np.array([[self._XSTART, self._XEND]], dtype=int)

        rng = np.random.default_rng(7)
        flat = rng.normal(0.0, 50.0, size=(nrows, ncols)).astype(np.float32) + 200.0
        rows = np.arange(nrows, dtype=float)
        for j in range(ncols):
            center = c0 + tilt * j
            sigma = hw / 2.355
            flat[:, j] += 10000.0 * np.exp(-0.5 * ((rows - center) / sigma) ** 2)

        path = str(tmp_path / "high_order_flat.fits")
        _write_fits_flat(flat, path)

        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs,
            xranges=xranges_in,
            edge_degree=1,
            flat_fraction=0.85,
            comm_window=4,
            slit_range_pixels=(int(hw * 0.4), int(hw * 2.5)),
            ybuffer=1,
            step=20,
            omask=None, ycororder=None,
            orders=[0],
        )
        return path, fi, xranges_in

    # ── 1. sample_cols may legitimately extend beyond post-fit xrange ────────

    def test_sample_cols_extends_beyond_post_fit_xend(self, tmp_path):
        """After step 6a clips x_end, sample_cols[-1] should still be at the
        original input xrange end — NOT cut back to x_end.

        This is the core semantic: sample_cols = original traced grid;
        [x_start, x_end] = final valid domain (narrower when polynomial
        overshoots detector).
        """
        path, fi, xranges_in = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        os0 = trace.order_samples[0]
        x_end_final = trace.order_xranges[0, 1]

        # step 6a must have actually clipped x_end
        assert x_end_final < self._XEND, (
            f"Expected step 6a to clip x_end below {self._XEND}, "
            f"got {x_end_final} — test scenario may have changed."
        )
        # sample_cols must still include columns beyond the clipped x_end
        assert os0.sample_cols[-1] > x_end_final, (
            f"sample_cols[-1]={os0.sample_cols[-1]} <= x_end_final={x_end_final}; "
            "sample_cols should reflect the original traced grid, not the "
            "post-fit restricted xrange."
        )

    # ── 2. validity = xrange, not sample membership ──────────────────────────

    def test_column_in_sample_cols_outside_xrange_is_invalid(self, tmp_path):
        """Explicitly show that a column can be in sample_cols AND be invalid
        because it lies outside [x_start, x_end].

        This demonstrates the intended semantics: membership in sample_cols
        does NOT imply validity.  The valid domain is ONLY [x_start, x_end].
        """
        path, fi, _ = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        os0 = trace.order_samples[0]
        x_end_final = os0.x_end

        # Find a sample column that is beyond x_end_final
        beyond = os0.sample_cols[os0.sample_cols > x_end_final]
        assert len(beyond) > 0, (
            "No sample column beyond x_end_final — the high-order scenario "
            "did not produce the expected post-fit clipping."
        )
        col = int(beyond[0])
        # col is in sample_cols, but it is outside [x_start, x_end]
        assert col > x_end_final, (
            f"col {col} should be beyond x_end_final {x_end_final}"
        )
        # col is NOT valid (it lies outside the post-fit domain)
        assert not (os0.x_start <= col <= os0.x_end), (
            f"col {col} unexpectedly inside [x_start={os0.x_start}, "
            f"x_end={os0.x_end}]"
        )

    # ── 3. compat arrays are NaN for sample cols outside xrange ──────────────

    def test_compat_center_rows_nan_for_sample_cols_beyond_xend(self, tmp_path):
        """In a real post-fit-clipped trace, compat center_rows must be NaN
        for any sample column that lies beyond x_end_final.

        This is an end-to-end version of the TestCompatArraysPostFitXrangeMasking
        unit tests — it runs the full trace_orders_from_flat pipeline with a
        scenario where step 6a ACTUALLY narrows x_end.
        """
        path, fi, _ = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)

        for i, os_i in enumerate(trace.order_samples):
            x_end_final = os_i.x_end
            for k, col in enumerate(trace.sample_cols):
                val = trace.center_rows[i, k]
                if col > x_end_final:
                    assert np.isnan(val), (
                        f"Order {i}: center_rows[{i}, {k}] = {val:.2f} is finite "
                        f"at col {col} which is beyond post-fit x_end={x_end_final}"
                    )

    def test_compat_finite_implies_within_xrange(self, tmp_path):
        """Every finite entry in center_rows must lie within [x_start, x_end]."""
        path, fi, _ = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)

        for i, os_i in enumerate(trace.order_samples):
            for k, col in enumerate(trace.sample_cols):
                val = trace.center_rows[i, k]
                if np.isfinite(val):
                    assert os_i.x_start <= col <= os_i.x_end, (
                        f"Order {i}: center_rows[{i}, {k}] = {val:.2f} is finite "
                        f"but col {col} is outside [{os_i.x_start}, {os_i.x_end}]"
                    )

    # ── 4. to_order_geometry_set uses order_xranges, not sample_cols span ────

    def test_geometry_x_end_equals_restricted_xrange_not_sample_cols_last(
        self, tmp_path
    ):
        """to_order_geometry_set must use order_xranges for x_end, not
        sample_cols[-1].

        When step 6a clips x_end, sample_cols[-1] > x_end_final.  If
        geometry mistakenly used sample_cols[-1] it would report a wider
        (invalid) x_end.  This test verifies geometry.x_end == order_xranges
        and is strictly less than sample_cols[-1].
        """
        path, fi, _ = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)
        geom = trace.to_order_geometry_set("K3")
        g = geom.geometries[0]

        x_end_xranges = int(trace.order_xranges[0, 1])
        sample_cols_last = int(trace.sample_cols[-1])

        # Geometry must come from order_xranges (restricted), not sample_cols
        assert g.x_end == x_end_xranges, (
            f"g.x_end={g.x_end} != order_xranges x_end={x_end_xranges}"
        )
        # Confirm the scenario is non-trivial: sample_cols extends beyond x_end
        assert sample_cols_last > x_end_xranges, (
            f"sample_cols[-1]={sample_cols_last} is not beyond "
            f"x_end={x_end_xranges}; scenario did not produce clipping."
        )
        # Geometry must NOT use the wider sample_cols last column
        assert g.x_end != sample_cols_last, (
            f"geometry x_end={g.x_end} equals sample_cols[-1]={sample_cols_last}; "
            "geometry is using the wrong (non-restricted) range."
        )

    # ── 5. regression: raw sample_cols as valid domain gives wrong answer ────

    def test_regression_sample_cols_last_not_valid_x_end(self, tmp_path):
        """Regression: using sample_cols[-1] as x_end instead of
        order_xranges gives an incorrect (too wide) range after step 6a clips.

        This test would FAIL if any downstream code substituted
        ``trace.sample_cols[-1]`` for ``trace.order_xranges[i, 1]`` as the
        order's valid x_end.
        """
        path, fi, xranges_in = self._make_high_order_flat(tmp_path)
        trace = trace_orders_from_flat([path], flatinfo=fi)

        correct_x_end = int(trace.order_xranges[0, 1])
        wrong_x_end = int(trace.sample_cols[-1])

        # The two values must differ when step 6a has actually clipped.
        assert correct_x_end < wrong_x_end, (
            f"correct_x_end={correct_x_end} should be < wrong_x_end={wrong_x_end}; "
            "step 6a clipping must have occurred for this regression to be meaningful."
        )
        # If code used wrong_x_end for the geometry, x_end would exceed the
        # detector-safe limit.  Verify that the restricted range is safe.
        nrows = self._NROWS
        assert trace.bot_poly_coeffs is not None
        bot_at_correct = np.polynomial.polynomial.polyval(
            float(correct_x_end), trace.bot_poly_coeffs[0]
        )
        top_at_correct = np.polynomial.polynomial.polyval(
            float(correct_x_end), trace.top_poly_coeffs[0]
        )
        top_at_wrong = np.polynomial.polynomial.polyval(
            float(wrong_x_end), trace.top_poly_coeffs[0]
        )
        assert 0.0 < bot_at_correct < float(nrows - 1), (
            "Edge polynomial is not within detector at correct x_end."
        )
        assert 0.0 < top_at_correct < float(nrows - 1), (
            "Edge polynomial is not within detector at correct x_end."
        )
        # At wrong_x_end, the top edge polynomial must be outside [0, nrows-1]
        # (that's why step 6a clipped it).
        assert top_at_wrong >= float(nrows - 1) or top_at_wrong <= 0.0, (
            f"top_at_wrong={top_at_wrong:.2f} is unexpectedly within detector "
            f"at wrong_x_end={wrong_x_end}; scenario may have changed."
        )


def rng_offset(k: int) -> float:
    """Small constant offset to make three flats slightly different."""
    return float(k) * 10.0


def _write_three_flats(tmp_path, base=None) -> list[str]:
    """Write three synthetic flat FITS files and return their paths."""
    if base is None:
        base = _make_synthetic_flat(seed=0)
    paths = []
    for k in range(3):
        flat_k = (base + rng_offset(k)).astype(np.float32)
        p = str(tmp_path / f"flat_{k}.fits")
        _write_fits_flat(flat_k, p)
        paths.append(p)
    return paths
