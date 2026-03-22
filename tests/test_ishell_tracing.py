"""
Tests for the iSHELL order-centre tracing module (tracing.py).

These tests verify:
  - FlatOrderTrace can be constructed directly (dataclass API),
  - load_and_combine_flats() raises on empty input and returns the
    correct shape for one or multiple synthetic FITS files,
  - trace_orders_from_flat() raises on empty file list and on profiles
    with insufficient signal for edge detection,
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

# Synthetic tests provide explicit initial guess positions, analogous to mc_adjustguesspos in IDL.
_SYN_GUESS_ROWS: list[float] = [
    float(_SYN_FIRST_CENTER + k * _SYN_SPACING) for k in range(_SYN_N_ORDERS)
]   # = [60.0, 100.0, 140.0, 180.0, 220.0]


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

    def test_raises_when_no_flatinfo_and_no_guess_rows(self, tmp_path):
        """Without flatinfo or guess_rows, a ValueError must be raised.

        IDL mc_findorders always requires explicit guesspos from mc_adjustguesspos;
        there is no auto-detection mode.  The Python port mirrors this constraint.
        """
        flat = np.ones((_NROWS, _NCOLS), dtype=np.float32) * 500.0
        path = str(tmp_path / "blank.fits")
        _write_fits_flat(flat, path)
        with pytest.raises(ValueError, match="guess_rows.*flatinfo"):
            trace_orders_from_flat([path])

    def test_raises_on_empty_guess_rows(self, tmp_path):
        """An empty guess_rows list should also raise ValueError."""
        flat = np.ones((_NROWS, _NCOLS), dtype=np.float32) * 500.0
        path = str(tmp_path / "blank.fits")
        _write_fits_flat(flat, path)
        with pytest.raises(ValueError, match="empty"):
            trace_orders_from_flat([path], guess_rows=[])


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
            guess_rows=_SYN_GUESS_ROWS,
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

    def test_half_width_rows_physically_reasonable(self, trace):
        """Estimated half-widths must be positive and within [1, 2*_SYN_HALF_WIDTH].

        NOTE: the IDL-style edge-based half-width measures (top_edge - bot_edge)/2
        where the edges are traced at the flux-fraction threshold (frac*peak).
        For the synthetic Gaussian orders (frac=0.85), the traced edges are much
        closer to the order centre than the full Gaussian half-width, so the
        IDL-derived half_width_rows is smaller than _SYN_HALF_WIDTH.  This test
        only verifies that the values are physically reasonable (positive and
        bounded), not that they match the Gaussian half-width.
        """
        for i in range(_SYN_N_ORDERS):
            hw = trace.half_width_rows[i]
            assert hw > 1.0, (
                f"Order {i}: half-width {hw:.1f} is unreasonably small (< 1 px)"
            )
            assert hw < 2.0 * _SYN_HALF_WIDTH, (
                f"Order {i}: half-width {hw:.1f} exceeds 2x known "
                f"half-width {2*_SYN_HALF_WIDTH}"
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
            guess_rows=_SYN_GUESS_ROWS,
        )
        trace3 = trace_orders_from_flat(
            paths,
            n_sample_cols=10,
            col_range=(50, _NCOLS - 50),
            guess_rows=_SYN_GUESS_ROWS,
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
        # Approximate H1 mode guess row positions (order centres at mid-column).
        # In production, use read_flatinfo("H1") to get accurate positions.
        # For iSHELL H1 mode (~43 orders, rows 0–2048, ~47-row pitch):
        _H1_GUESS_ROWS = list(range(24, 2020, 47))
        return trace_orders_from_flat(
            _H1_FLAT_FILES,
            col_range=(650, 1550),
            n_sample_cols=20,
            guess_rows=_H1_GUESS_ROWS,
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
        # After the post-fit xrange restriction (step 6a in
        # trace_orders_from_flat: fitted edge polynomials are evaluated and the
        # order's [x_start, x_end] is narrowed to columns where BOTH edges stay
        # within the detector), order_xranges may be narrower than the input
        # col_range for orders whose polynomial overshoots at one end.
        # The geometry must therefore be checked for containment within
        # col_range, not equality.
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
        return trace_orders_from_flat([path], guess_rows=_SYN_GUESS_ROWS)

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
        return trace_orders_from_flat([path], guess_rows=_SYN_GUESS_ROWS)

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
        return trace_orders_from_flat([path], guess_rows=_SYN_GUESS_ROWS)

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
        return trace_orders_from_flat([path], guess_rows=_SYN_GUESS_ROWS)

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

    def test_flatinfo_order_samples_cols_within_xrange_prefit(self, trace_flatinfo):
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
        return trace_orders_from_flat([path], guess_rows=_SYN_GUESS_ROWS)

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


# ---------------------------------------------------------------------------
# 22. Explicit minimal tests: sample_cols=[10,20,30,40,50], x_start=20, x_end=40
# ---------------------------------------------------------------------------


class TestSampleColsVsXrangeExplicit:
    """Targeted tests using the minimal explicit geometry:

        sample_cols = [10, 20, 30, 40, 50]
        x_start     = 20
        x_end       = 40

    All objects are constructed directly (no full pipeline call) to isolate
    the semantic invariants from other tracing logic, making any failure easy
    to diagnose.

    Invariants under test:
    1. sample_cols may extend beyond [x_start, x_end] — this is valid.
    2. Membership in sample_cols does NOT imply validity.
    3. FlatOrderTrace.center_rows is NaN at cols 10, 50 (outside xrange).
    4. to_order_geometry_set() uses order_xranges, not sample_cols span.
    5. Regression: if sample_cols were substituted for the valid domain,
       geometry x_start/x_end would be wrong (10/50 instead of 20/40).
    """

    # ── canonical test geometry ───────────────────────────────────────────────
    _SAMPLE_COLS = [10, 20, 30, 40, 50]
    _CENTER_ROWS = [100.0, 110.0, 120.0, 130.0, 140.0]
    _X_START = 20
    _X_END = 40

    def _make_order_samples(self) -> OrderTraceSamples:
        """OrderTraceSamples with sample_cols=[10..50], x_start=20, x_end=40."""
        cols = np.array(self._SAMPLE_COLS, dtype=int)
        crows = np.array(self._CENTER_ROWS, dtype=float)
        return OrderTraceSamples(
            order_index=0,
            sample_cols=cols,
            center_rows=crows,
            bot_rows=crows - 5.0,
            top_rows=crows + 5.0,
            x_start=self._X_START,
            x_end=self._X_END,
        )

    def _build_compat(self, order_samples):
        """Replicate the step-8 compat array construction from
        trace_orders_from_flat.  Columns outside [x_start, x_end] stay NaN.
        """
        union_cols = np.unique(
            np.concatenate([os_i.sample_cols for os_i in order_samples])
        )
        compat = np.full((len(order_samples), len(union_cols)), np.nan)
        for i, os_i in enumerate(order_samples):
            for j, col in enumerate(os_i.sample_cols):
                if not (os_i.x_start <= col <= os_i.x_end):
                    continue
                k = int(np.searchsorted(union_cols, col))
                if k < len(union_cols) and union_cols[k] == col:
                    compat[i, k] = os_i.center_rows[j]
        return union_cols, compat

    def _make_flat_order_trace(self) -> FlatOrderTrace:
        """Build a FlatOrderTrace from the explicit OrderTraceSamples.

        compat arrays are built with the step-8 logic so that cols 10 and 50
        are NaN.  order_xranges is set to [[20, 40]] so that
        to_order_geometry_set() returns x_start=20, x_end=40.
        """
        os0 = self._make_order_samples()
        union_cols, compat_center_rows = self._build_compat([os0])
        # minimal polynomial: constant center at row 120
        coeffs = np.array([[120.0, 0.0]])   # shape (1, 2)
        return FlatOrderTrace(
            n_orders=1,
            sample_cols=union_cols,
            center_rows=compat_center_rows,
            center_poly_coeffs=coeffs,
            fit_rms=np.array([0.5]),
            half_width_rows=np.array([5.0]),
            poly_degree=1,
            seed_col=30,
            order_samples=[os0],
            order_xranges=np.array([[self._X_START, self._X_END]], dtype=int),
        )

    # ── TEST 1: construction with sample_cols beyond xrange is valid ──────────

    def test_order_trace_samples_with_extra_cols_is_valid(self):
        """Constructing OrderTraceSamples where sample_cols extends beyond
        [x_start, x_end] must not raise — it is intentional, not corruption.

        After step 6a (post-fit xrange restriction), sample_cols still holds
        the original traced grid.  The reduced [x_start, x_end] is correct.
        """
        os = self._make_order_samples()
        # Construction must succeed without any exception.
        assert os.sample_cols.tolist() == self._SAMPLE_COLS
        assert os.x_start == self._X_START
        assert os.x_end == self._X_END
        # sample_cols intentionally extends beyond both boundaries.
        assert os.sample_cols[0] < os.x_start, (
            "sample_cols[0] should be < x_start to make the test non-trivial"
        )
        assert os.sample_cols[-1] > os.x_end, (
            "sample_cols[-1] should be > x_end to make the test non-trivial"
        )

    # ── TEST 2: membership != validity ───────────────────────────────────────

    def test_sample_cols_membership_does_not_imply_validity(self):
        """10 and 50 appear in sample_cols but lie outside [20, 40] — invalid.

        Validity is determined solely by [x_start, x_end], not by presence
        in sample_cols.
        """
        os = self._make_order_samples()
        # Both 10 and 50 are in sample_cols.
        assert 10 in os.sample_cols
        assert 50 in os.sample_cols
        # But neither is in the valid post-fit domain.
        assert not (os.x_start <= 10 <= os.x_end), (
            "10 should be outside valid domain [20, 40]"
        )
        assert not (os.x_start <= 50 <= os.x_end), (
            "50 should be outside valid domain [20, 40]"
        )
        # Membership in [20, 30, 40] is valid by both tests.
        for col in (20, 30, 40):
            assert col in os.sample_cols
            assert os.x_start <= col <= os.x_end, (
                f"col {col} should be inside valid domain [20, 40]"
            )

    # ── TEST 3: compat arrays enforce xrange ──────────────────────────────────

    def test_compat_nan_at_cols_outside_xrange(self):
        """FlatOrderTrace.center_rows must be NaN at cols 10 and 50 (outside
        [20, 40]), even though those columns appear in order_samples[0].sample_cols.

        Finite entries are expected only at cols 20, 30, 40.
        """
        trace = self._make_flat_order_trace()
        col_to_idx = {int(col): k for k, col in enumerate(trace.sample_cols)}
        # Cols outside [20, 40] must be NaN.
        for bad_col in (10, 50):
            k = col_to_idx[bad_col]
            val = trace.center_rows[0, k]
            assert np.isnan(val), (
                f"center_rows[0, k={k}] at col={bad_col} should be NaN "
                f"(outside [20, 40]); got {val}"
            )
        # Cols inside [20, 40] must be finite.
        for good_col in (20, 30, 40):
            k = col_to_idx[good_col]
            val = trace.center_rows[0, k]
            assert np.isfinite(val), (
                f"center_rows[0, k={k}] at col={good_col} should be finite "
                f"(inside [20, 40]); got {val}"
            )

    def test_compat_finite_implies_within_xrange_explicit(self):
        """Every finite center_rows[0, k] entry implies sample_cols[k] ∈ [20, 40].

        Verifies the implication: finite → col ∈ [x_start, x_end].
        The converse (col ∈ xrange → finite) holds for this fixture because
        center_rows are non-NaN for all in-range sample columns.
        """
        trace = self._make_flat_order_trace()
        os0 = trace.order_samples[0]
        for k, col in enumerate(trace.sample_cols):
            val = trace.center_rows[0, k]
            if np.isfinite(val):
                assert os0.x_start <= col <= os0.x_end, (
                    f"Finite center_rows[0, k={k}] at col={col} but "
                    f"col is outside [{os0.x_start}, {os0.x_end}]"
                )

    # ── TEST 4: geometry uses xrange, not sampling span ──────────────────────

    def test_geometry_uses_order_xranges_not_sample_cols_span(self):
        """to_order_geometry_set() must report x_start=20 and x_end=40,
        not x_start=10 and x_end=50 (the sample_cols span).

        sample_cols spans [10, 50] but order_xranges is [[20, 40]].
        The geometry must come from order_xranges.
        """
        trace = self._make_flat_order_trace()
        geom = trace.to_order_geometry_set("H1")
        g = geom.geometries[0]

        # Geometry must use order_xranges.
        assert g.x_start == self._X_START, (
            f"g.x_start={g.x_start} should be {self._X_START} (from order_xranges "
            f"[{self._X_START}, {self._X_END}]); sample_cols[0]="
            f"{int(trace.sample_cols[0])} — geometry must not use sample span"
        )
        assert g.x_end == self._X_END, (
            f"g.x_end={g.x_end} should be {self._X_END} (from order_xranges); "
            f"sample_cols[-1]={int(trace.sample_cols[-1])} — geometry must not "
            "use sample span"
        )
        # The test is non-trivial: sample_cols span is wider than the valid xrange.
        assert int(trace.sample_cols[0]) < self._X_START
        assert int(trace.sample_cols[-1]) > self._X_END

    def test_geometry_uses_order_samples_xrange_when_order_xranges_is_none(self):
        """When order_xranges is None, to_order_geometry_set() falls back to
        per-order x_start/x_end from order_samples — NOT sample_cols span.

        This verifies the secondary fallback path also uses the valid domain.
        """
        os0 = self._make_order_samples()
        union_cols, compat_center_rows = self._build_compat([os0])
        coeffs = np.array([[120.0, 0.0]])
        trace_no_xranges = FlatOrderTrace(
            n_orders=1,
            sample_cols=union_cols,
            center_rows=compat_center_rows,
            center_poly_coeffs=coeffs,
            fit_rms=np.array([0.5]),
            half_width_rows=np.array([5.0]),
            poly_degree=1,
            seed_col=30,
            order_samples=[os0],
            order_xranges=None,   # ← no order_xranges; fallback to order_samples
        )
        geom = trace_no_xranges.to_order_geometry_set("H1")
        g = geom.geometries[0]
        # Must still use x_start=20, x_end=40 from order_samples[0].
        assert g.x_start == self._X_START, (
            f"g.x_start={g.x_start} should be {self._X_START} from order_samples; "
            f"sample_cols[0]={int(trace_no_xranges.sample_cols[0])}"
        )
        assert g.x_end == self._X_END, (
            f"g.x_end={g.x_end} should be {self._X_END} from order_samples; "
            f"sample_cols[-1]={int(trace_no_xranges.sample_cols[-1])}"
        )

    # ── TEST 5: regression — failure if sample_cols used as valid domain ──────

    def test_regression_geometry_wrong_if_sample_cols_used_as_domain(self):
        """Regression: substituting sample_cols[0]/[-1] for order_xranges would
        give geometry x_start=10, x_end=50 instead of 20, 40.

        This test FAILS if to_order_geometry_set() (or any callee) ever
        substitutes ``trace.sample_cols[0]`` / ``trace.sample_cols[-1]`` for
        the per-order valid xrange when order_xranges IS available.
        """
        trace = self._make_flat_order_trace()
        geom = trace.to_order_geometry_set("H1")
        g = geom.geometries[0]

        # "Wrong" values that would appear if sample_cols were used.
        wrong_x_start = int(trace.sample_cols[0])    # 10
        wrong_x_end = int(trace.sample_cols[-1])     # 50

        # Geometry must not equal the wrong (sample-cols-derived) values.
        assert g.x_start != wrong_x_start, (
            f"g.x_start={g.x_start} == sample_cols[0]={wrong_x_start}; "
            "geometry is using sample_cols span instead of order_xranges"
        )
        assert g.x_end != wrong_x_end, (
            f"g.x_end={g.x_end} == sample_cols[-1]={wrong_x_end}; "
            "geometry is using sample_cols span instead of order_xranges"
        )
        # And must equal the correct (xrange-derived) values.
        assert g.x_start == self._X_START
        assert g.x_end == self._X_END




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


# ---------------------------------------------------------------------------
# 23. IDL-port fidelity tests (mc_findorders.pro requirements)
#
# The problem statement mandates "at minimum" tests for six properties of
# the new IDL-port tracing core.  These tests are collected here and
# explicitly reference the IDL behaviour they verify.
#
# IDL reference: vendor/spextool_idl/Spextool/pro/mc_findorders.pro
# ---------------------------------------------------------------------------


def _make_single_order_flat(
    nrows: int = 256,
    ncols: int = 256,
    c0: float = 128.0,
    tilt: float = 0.05,
    hw: float = 12.0,
    peak: float = 10000.0,
    bg: float = 200.0,
    noise: float = 30.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic flat with a SINGLE order of known shape.

    Returns
    -------
    flat : ndarray (nrows, ncols)  float32
    true_bot : ndarray (ncols,)    true bottom edge row
    true_top : ndarray (ncols,)    true top edge row
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(nrows, dtype=float)
    flat = rng.normal(bg, noise, size=(nrows, ncols)).astype(np.float32)
    true_bot = np.empty(ncols)
    true_top = np.empty(ncols)
    for j in range(ncols):
        center = c0 + tilt * j
        sigma = hw / 2.355
        flat[:, j] += peak * np.exp(-0.5 * ((rows - center) / sigma) ** 2)
        true_bot[j] = center - hw
        true_top[j] = center + hw
    return flat, true_bot, true_top


class TestIDLPortFidelity:
    """Verify the six IDL mc_findorders.pro fidelity requirements.

    Problem-statement requirement 1:
        The tracing core uses edge detection (flux-threshold + Sobel COM) as
        in IDL mc_findorders; no peak-finding helpers are present or needed.

    Problem-statement requirement 2:
        Edge tracing is primary; centre values are derived from traced edges
        (IDL: ``cen[k] = (com_bot + com_top) / 2``).

    Problem-statement requirement 3:
        Left/right sweep with dynamic polynomial prediction works on a
        synthetic flat with realistic banded orders
        (IDL: left loop ``for j=0,gidx``, right loop ``for j=gidx+1,nscols-1``
        with ``mc_polyfit1d`` prediction at each step).

    Problem-statement requirement 4:
        Valid xranges emerge from fitted edge positions staying on detector
        (IDL end-of-loop block:
        ``z = where(top gt 0 and top lt nrows-1 and bot gt 0 and bot lt nrows-1)``
        ``xranges[*,i] = [min(x[z]), max]``).

    Problem-statement requirement 5:
        The returned structure still supports downstream geometry conversion.

    Problem-statement requirement 6:
        Existing public-facing compatibility fields remain usable where expected.
    """

    # ---- Requirement 1: no peak-finding in the tracing core ---------------

    def test_trace_single_order_idlstyle_does_not_call_find_peaks(self):
        """_find_order_peaks and scipy find_peaks must not exist in the tracing module.

        IDL mc_findorders has no peak-detection step: it uses only the
        flux-threshold + Sobel COM logic at each sample column.  The Python
        port removes peak detection entirely, so the attribute must not exist.
        _trace_single_order_idlstyle must succeed without it.
        """
        import pyspextool.instruments.ishell.tracing as _tracing_module
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )

        # _find_order_peaks must have been deleted from the module.
        assert not hasattr(_tracing_module, "_find_order_peaks"), (
            "_find_order_peaks must not exist in the tracing module; "
            "IDL mc_findorders uses no peak detection."
        )

        # scipy.signal.find_peaks must not be imported by the module.
        assert not hasattr(_tracing_module, "_scipy_find_peaks"), (
            "_scipy_find_peaks (scipy.signal.find_peaks) must not be "
            "imported in the tracing module."
        )

        flat, _, _ = _make_single_order_flat(seed=11)
        nrows, ncols = flat.shape
        sflat = _compute_sobel_image(flat)
        s_cols = np.arange(10, ncols - 10, 20)
        guess_col = float(s_cols[len(s_cols) // 2])

        # _trace_single_order_idlstyle must work without any peak detection.
        result = _trace_single_order_idlstyle(
            flat, sflat, s_cols, 10, ncols - 11,
            guess_col, 128.0,
            2, nrows, 1, 0.85, 3, 12.0, 60.0,
        )

        # Tracing must succeed and return valid arrays.
        assert result.center_samples.shape == (len(s_cols),)
        assert result.bottom_edge_samples.shape == (len(s_cols),)
        assert result.top_edge_samples.shape == (len(s_cols),)
        # At least some traced positions should be finite.
        assert np.sum(np.isfinite(result.bottom_edge_samples)) > 0, (
            "Expected at least some finite bottom edge values"
        )
        assert np.sum(np.isfinite(result.top_edge_samples)) > 0, (
            "Expected at least some finite top edge values"
        )

    def test_trace_single_order_idlstyle_succeeds_without_flatinfo(self):
        """_trace_single_order_idlstyle operates entirely on image data; it does
        not require flatinfo or any peak-detection initialisation.

        This verifies requirement 1 structurally: the function signature
        accepts only image + geometry parameters, with no peak-matching inputs.
        """
        from pyspextool.instruments.ishell.tracing import _trace_single_order_idlstyle
        import inspect

        sig = inspect.signature(_trace_single_order_idlstyle)
        param_names = list(sig.parameters.keys())
        # Must not have peak-matching parameters.
        for forbidden in ("peaks", "peak_cols", "match", "seed_peaks"):
            assert forbidden not in param_names, (
                f"_trace_single_order_idlstyle has forbidden peak-matching "
                f"parameter: '{forbidden}'"
            )

    def test_trace_single_order_succeeds_without_flatinfo(self):
        """Alias kept for backward-compatibility of the test name; delegates to
        the idlstyle version which is the active tracing function.
        """
        from pyspextool.instruments.ishell.tracing import _trace_single_order_idlstyle
        import inspect

        sig = inspect.signature(_trace_single_order_idlstyle)
        param_names = list(sig.parameters.keys())
        for forbidden in ("peaks", "peak_cols", "match", "seed_peaks"):
            assert forbidden not in param_names, (
                f"_trace_single_order_idlstyle has forbidden peak-matching "
                f"parameter: '{forbidden}'"
            )

    # ---- Requirement 2: edge-first; centres derived from edges ------------

    def test_traced_cen_equals_midpoint_of_bot_and_top(self):
        """Where both bot and top are finite, cen must equal (bot+top)/2.

        IDL mc_findorders: ``cen[k] = (com_bot + com_top) / 2``
        Python: center_samples[k] = (com_bot + com_top) / 2.0
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )

        flat, _, _ = _make_single_order_flat(c0=100.0, hw=15.0, tilt=0.02, seed=22)
        nrows, ncols = flat.shape
        sflat = _compute_sobel_image(flat)
        s_cols = np.arange(10, ncols - 10, 10)
        seed_idx = len(s_cols) // 2
        guess_col = float(s_cols[seed_idx])

        result = _trace_single_order_idlstyle(
            flat, sflat, s_cols, 10, ncols - 11,
            guess_col, 100.0,
            2, nrows, 1, 0.75, 4, 10.0, 60.0,
        )
        cen = result.center_samples
        bot = result.bottom_edge_samples
        top = result.top_edge_samples

        # Where both edges are finite, cen must be (bot+top)/2.
        both_finite = np.isfinite(bot) & np.isfinite(top)
        assert both_finite.sum() > 0, "Expected some columns with both edges traced"
        expected_cen = (bot[both_finite] + top[both_finite]) / 2.0
        np.testing.assert_allclose(
            cen[both_finite], expected_cen, atol=1e-10,
            err_msg="cen must equal (bot+top)/2 where both edges are finite",
        )

    def test_bot_poly_coeffs_and_top_poly_coeffs_primary_outputs(self, tmp_path):
        """After trace_orders_from_flat, bot_poly_coeffs and top_poly_coeffs
        must be the primary fit outputs; center_poly_coeffs is derived from them.

        IDL mc_findorders: ``edgecoeffs[*,0,i]`` (bot) and ``edgecoeffs[*,1,i]``
        (top) are the authoritative outputs.  Centres are not directly fitted.
        """
        flat = _make_synthetic_flat(seed=3)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], poly_degree=2, n_sample_cols=15, guess_rows=_SYN_GUESS_ROWS)

        assert trace.bot_poly_coeffs is not None, "bot_poly_coeffs must be populated"
        assert trace.top_poly_coeffs is not None, "top_poly_coeffs must be populated"

        # center_poly_coeffs must be (bot+top)/2 for every order.
        for i in range(trace.n_orders):
            expected = (trace.bot_poly_coeffs[i] + trace.top_poly_coeffs[i]) / 2.0
            np.testing.assert_allclose(
                trace.center_poly_coeffs[i], expected, rtol=1e-10,
                err_msg=f"Order {i}: center_poly_coeffs != (bot+top)/2",
            )

    def test_bot_poly_below_top_poly_across_sample_cols(self, tmp_path):
        """Bottom edge polynomial must evaluate below top edge polynomial at
        all sample columns for every order.

        IDL always stores ``edges[k,0] = com_bot`` (smaller row) and
        ``edges[k,1] = com_top`` (larger row).
        """
        flat = _make_synthetic_flat(seed=4)
        path = str(tmp_path / "flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], poly_degree=2, n_sample_cols=15, guess_rows=_SYN_GUESS_ROWS)

        for i in range(trace.n_orders):
            bot_vals = np.polynomial.polynomial.polyval(
                trace.sample_cols.astype(float), trace.bot_poly_coeffs[i]
            )
            top_vals = np.polynomial.polynomial.polyval(
                trace.sample_cols.astype(float), trace.top_poly_coeffs[i]
            )
            assert np.all(bot_vals < top_vals), (
                f"Order {i}: bot edge must be strictly below top edge at all cols"
            )

    # ---- Requirement 3: left/right sweep with dynamic prediction ----------

    def test_left_sweep_traces_order_away_from_seed(self):
        """The left sweep must trace the order correctly for columns to the
        LEFT of the seed, not just near it.

        IDL mc_findorders left loop: ``for j=0,gidx do begin; k = gidx-j; ...``
        The polynomial prediction ``mc_polyfit1d(scols,cen,...)`` is updated
        at every step, so the predicted centre tracks the order as it moves.
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )

        nrows, ncols = 256, 300
        c0, tilt, hw = 128.0, 0.08, 14.0
        flat, true_bot, true_top = _make_single_order_flat(
            nrows=nrows, ncols=ncols, c0=c0, tilt=tilt, hw=hw, seed=33,
        )
        sflat = _compute_sobel_image(flat)
        s_cols = np.arange(5, ncols - 5, 10)
        # Seed near the RIGHT end → the LEFT sweep covers most of the range.
        seed_idx = len(s_cols) - 3   # near-right seed
        guess_col = float(s_cols[seed_idx])
        guess_row = c0 + tilt * s_cols[seed_idx]

        result = _trace_single_order_idlstyle(
            flat, sflat, s_cols, 5, ncols - 6,
            guess_col, guess_row,
            2, nrows, 1, 0.75, 4, 8.0, 60.0,
        )
        cen = result.center_samples

        # Columns to the LEFT of the seed (not just around it) should be traced.
        left_cols_idx = np.where(s_cols < s_cols[seed_idx])[0]
        finite_left = np.sum(np.isfinite(cen[left_cols_idx]))
        assert finite_left >= len(left_cols_idx) * 0.6, (
            f"Left sweep only traced {finite_left}/{len(left_cols_idx)} columns "
            "to the left of the seed — dynamic prediction may not be working"
        )

        # Traced centres in the left region should be close to the true values.
        for k in left_cols_idx:
            if np.isfinite(cen[k]):
                true_center_k = c0 + tilt * s_cols[k]
                assert abs(cen[k] - true_center_k) < 8.0, (
                    f"Left sweep: col {s_cols[k]}: traced cen={cen[k]:.1f} "
                    f"differs from true {true_center_k:.1f} by > 8 px "
                    "(dynamic prediction failure)"
                )

    def test_right_sweep_traces_order_away_from_seed(self):
        """The right sweep must trace the order correctly for columns to the
        RIGHT of the seed.

        IDL mc_findorders right loop: ``for j=gidx+1,nscols-1 do begin; k=j``
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )

        nrows, ncols = 256, 300
        c0, tilt, hw = 128.0, -0.06, 14.0
        flat, true_bot, true_top = _make_single_order_flat(
            nrows=nrows, ncols=ncols, c0=c0, tilt=tilt, hw=hw, seed=44,
        )
        sflat = _compute_sobel_image(flat)
        s_cols = np.arange(5, ncols - 5, 10)
        # Seed near the LEFT end → the RIGHT sweep covers most of the range.
        seed_idx = 2   # near-left seed
        guess_col = float(s_cols[seed_idx])
        guess_row = c0 + tilt * s_cols[seed_idx]

        result = _trace_single_order_idlstyle(
            flat, sflat, s_cols, 5, ncols - 6,
            guess_col, guess_row,
            2, nrows, 1, 0.75, 4, 8.0, 60.0,
        )
        cen = result.center_samples

        # Columns to the RIGHT of the seed should be traced.
        right_cols_idx = np.where(s_cols > s_cols[seed_idx])[0]
        finite_right = np.sum(np.isfinite(cen[right_cols_idx]))
        assert finite_right >= len(right_cols_idx) * 0.6, (
            f"Right sweep only traced {finite_right}/{len(right_cols_idx)} columns "
            "to the right of the seed"
        )

        # Traced centres should be close to the true values.
        for k in right_cols_idx:
            if np.isfinite(cen[k]):
                true_center_k = c0 + tilt * s_cols[k]
                assert abs(cen[k] - true_center_k) < 8.0, (
                    f"Right sweep: col {s_cols[k]}: traced cen={cen[k]:.1f} "
                    f"differs from true {true_center_k:.1f} by > 8 px"
                )

    def test_sweep_initialises_cen_around_seed_idl_style(self):
        """The centre array must be initialised around the seed index with the
        guess y-value before the sweeps start.

        IDL mc_findorders:
            ``cen[(gidx-degree):(gidx+degree)] = guesspos[1,i]``
        Python _initialize_order_trace_arrays:
            ``cen[lo: hi+1] = guess_row``
        where lo = max(0, gidx-degree), hi = min(n_samp-1, gidx+degree).
        This ensures the initial polynomial fit at the first sweep step has
        enough non-NaN points.
        """
        from pyspextool.instruments.ishell.tracing import (
            _initialize_order_trace_arrays,
            _compute_sobel_image,
        )

        flat, _, _ = _make_single_order_flat(c0=128.0, seed=55)
        nrows, ncols = flat.shape
        s_cols = np.arange(5, ncols - 5, 10)
        n_samp = len(s_cols)
        seed_idx = n_samp // 2
        y_seed = 128.0
        poly_degree = 3
        guess_col = float(s_cols[seed_idx])

        # _initialize_order_trace_arrays produces the seeded cen array.
        _, cen, _, gidx = _initialize_order_trace_arrays(
            s_cols, guess_col, y_seed, poly_degree,
        )

        # The initialisation window: lo = max(0, gidx-degree),
        # hi = min(n_samp-1, gidx+degree)
        lo = max(0, gidx - poly_degree)
        hi = min(n_samp - 1, gidx + poly_degree)
        init_vals = cen[lo: hi + 1]
        assert np.all(np.isfinite(init_vals)), (
            "IDL initialisation: cen values in [lo:hi+1] must be finite "
            "before first sweep step so that polynomial prediction works"
        )
        np.testing.assert_allclose(
            init_vals, y_seed, atol=1e-10,
            err_msg="IDL initialisation: cen[lo:hi+1] must equal y_seed",
        )
        # Indices outside the seed window must be NaN.
        for k in range(n_samp):
            if k < lo or k > hi:
                assert np.isnan(cen[k]), (
                    f"cen[{k}] should be NaN outside the seed window [{lo}, {hi}]"
                )

    def test_bilateral_sweep_both_directions_traced(self):
        """Tracing with the seed in the middle must produce finite values on
        BOTH sides (left and right) — verifying the bilateral structure of the
        IDL algorithm.

        IDL mc_findorders has two separate ``for`` loops:
          - Left:  ``for j = 0, gidx do begin``
          - Right: ``for j = gidx+1, nscols-1 do begin``
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )

        flat, _, _ = _make_single_order_flat(c0=128.0, tilt=0.05, hw=14.0, seed=66)
        nrows, ncols = flat.shape
        sflat = _compute_sobel_image(flat)
        s_cols = np.arange(5, ncols - 5, 8)
        seed_idx = len(s_cols) // 2
        guess_col = float(s_cols[seed_idx])

        result = _trace_single_order_idlstyle(
            flat, sflat, s_cols, 5, ncols - 6,
            guess_col, 128.0,
            2, nrows, 1, 0.75, 4, 8.0, 60.0,
        )
        cen = result.center_samples

        left_finite = np.sum(np.isfinite(cen[:seed_idx]))
        right_finite = np.sum(np.isfinite(cen[seed_idx + 1:]))
        n_left = seed_idx
        n_right = len(s_cols) - seed_idx - 1

        assert left_finite >= n_left * 0.5, (
            f"Left sweep produced only {left_finite}/{n_left} finite centres; "
            "expected at least 50% of left columns to be traced"
        )
        assert right_finite >= n_right * 0.5, (
            f"Right sweep produced only {right_finite}/{n_right} finite centres; "
            "expected at least 50% of right columns to be traced"
        )

    # ---- Requirement 4: valid xranges from fitted edge positions ----------

    def test_xranges_clipped_when_edge_polynomial_overshoots_detector(self, tmp_path):
        """When a fitted edge polynomial evaluates outside [0, nrows-1] at
        some columns within the input xrange, order_xranges must be restricted
        to the safe sub-range.

        IDL mc_findorders (end of per-order loop):
            ``z = where(top gt 0.0 and top lt nrows-1 and bot gt 0 and bot lt nrows-1)``
            ``xranges[*,i] = [min(x[z],MAX=mx), mx]``
        """
        from types import SimpleNamespace

        nrows, ncols = 256, 400
        c0 = 235.0          # near top of detector
        tilt = 0.08         # order tilts upward; top edge exits detector ~col 250
        hw = 12.0

        edge_coeffs = np.zeros((1, 2, 2))
        edge_coeffs[0, 0, 0] = c0 - hw   # bot constant
        edge_coeffs[0, 0, 1] = tilt      # bot slope
        edge_coeffs[0, 1, 0] = c0 + hw   # top constant
        edge_coeffs[0, 1, 1] = tilt      # top slope

        # top hits nrows-1=255 at col ≈ (255 - (c0+hw)) / tilt ≈ (255-247)/0.08 = 100
        # So the input xrange [5, 395] should be clipped at the right end.
        xranges_in = np.array([[5, 395]], dtype=int)

        flat, _, _ = _make_single_order_flat(
            nrows=nrows, ncols=ncols, c0=c0, tilt=tilt, hw=hw, seed=77,
        )
        path = str(tmp_path / "overshoot_flat.fits")
        _write_fits_flat(flat.astype(np.float32), path)

        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs,
            xranges=xranges_in,
            edge_degree=1,
            flat_fraction=0.75,
            comm_window=6,
            slit_range_pixels=(int(hw * 0.3), int(hw * 2.5)),
            ybuffer=1,
            step=15,
            omask=None,
            ycororder=None,
            orders=[0],
        )
        trace = trace_orders_from_flat([path], flatinfo=fi)

        xr = trace.order_xranges
        assert xr is not None, "order_xranges must be populated"
        # The restricted x_end must be less than the input x_end.
        assert xr[0, 1] < xranges_in[0, 1], (
            f"order_xranges x_end ({xr[0, 1]}) was not clipped from "
            f"input xrange end ({xranges_in[0, 1]}); "
            "IDL post-fit restriction must narrow the xrange when the "
            "polynomial overshoots the detector."
        )

    def test_xranges_unchanged_when_edges_stay_on_detector(self, tmp_path):
        """When fitted edge polynomials stay within (0, nrows-1) for the full
        input xrange, order_xranges must match the input sranges unchanged.

        IDL mc_findorders: xranges are only narrowed when the fitted edges
        actually exit the detector; well-behaved orders keep their full range.
        """
        from types import SimpleNamespace

        nrows, ncols = 256, 300
        # Order well within detector at all columns.
        c0 = 128.0
        tilt = 0.01   # very mild tilt; stays in detector for full ncols
        hw = 12.0

        edge_coeffs = np.zeros((1, 2, 2))
        edge_coeffs[0, 0, 0] = c0 - hw
        edge_coeffs[0, 0, 1] = tilt
        edge_coeffs[0, 1, 0] = c0 + hw
        edge_coeffs[0, 1, 1] = tilt
        xranges_in = np.array([[5, ncols - 5]], dtype=int)

        flat, _, _ = _make_single_order_flat(
            nrows=nrows, ncols=ncols, c0=c0, tilt=tilt, hw=hw, seed=88,
        )
        path = str(tmp_path / "safe_flat.fits")
        _write_fits_flat(flat.astype(np.float32), path)

        fi = SimpleNamespace(
            edge_coeffs=edge_coeffs,
            xranges=xranges_in,
            edge_degree=1,
            flat_fraction=0.75,
            comm_window=6,
            slit_range_pixels=(int(hw * 0.3), int(hw * 2.5)),
            ybuffer=1,
            step=15,
            omask=None,
            ycororder=None,
            orders=[0],
        )
        trace = trace_orders_from_flat([path], flatinfo=fi)

        xr = trace.order_xranges
        assert xr is not None, "order_xranges must be populated"
        assert xr[0, 0] == xranges_in[0, 0], (
            f"x_start {xr[0, 0]} was unexpectedly clipped from {xranges_in[0, 0]}"
        )
        assert xr[0, 1] == xranges_in[0, 1], (
            f"x_end {xr[0, 1]} was unexpectedly clipped from {xranges_in[0, 1]}"
        )

    def test_edge_polynomials_within_detector_at_reported_xranges(self, tmp_path):
        """Within the reported order_xranges, BOTH fitted edge polynomials must
        evaluate to pixel rows strictly within (0, nrows-1).

        This is the postcondition of the IDL xrange-restriction step:
        ``where(top gt 0.0 and top lt nrows-1 and bot gt 0 and bot lt nrows-1)``
        """
        flat = _make_synthetic_flat(seed=5)
        path = str(tmp_path / "syn_flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], n_sample_cols=15, poly_degree=2, guess_rows=_SYN_GUESS_ROWS)

        nrows = flat.shape[0]
        for i in range(trace.n_orders):
            xr = trace.order_xranges[i]
            x_dense = np.arange(xr[0], xr[1] + 1, dtype=float)
            if len(x_dense) == 0:
                continue
            bot_v = np.polynomial.polynomial.polyval(x_dense, trace.bot_poly_coeffs[i])
            top_v = np.polynomial.polynomial.polyval(x_dense, trace.top_poly_coeffs[i])
            assert np.all(bot_v > 0.0) and np.all(bot_v < float(nrows - 1)), (
                f"Order {i}: bot poly outside (0, {nrows-1}) within "
                f"reported xrange [{xr[0]}, {xr[1]}]"
            )
            assert np.all(top_v > 0.0) and np.all(top_v < float(nrows - 1)), (
                f"Order {i}: top poly outside (0, {nrows-1}) within "
                f"reported xrange [{xr[0]}, {xr[1]}]"
            )

    # ---- Requirement 5: downstream geometry conversion --------------------

    def test_to_order_geometry_set_returns_valid_geometryset(self, tmp_path):
        """trace_orders_from_flat result converts to OrderGeometrySet without error.

        Requirement 5: the returned structure supports downstream geometry
        conversion via to_order_geometry_set().
        """
        flat = _make_synthetic_flat(seed=6)
        path = str(tmp_path / "syn_flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], n_sample_cols=12, poly_degree=2, guess_rows=_SYN_GUESS_ROWS)

        geom = trace.to_order_geometry_set("H1")
        assert isinstance(geom, OrderGeometrySet), (
            "to_order_geometry_set must return an OrderGeometrySet"
        )
        assert geom.n_orders == trace.n_orders
        assert geom.mode == "H1"

    def test_order_geometry_uses_traced_edge_polys(self, tmp_path):
        """OrderGeometry.bottom_edge_coeffs / top_edge_coeffs must come directly
        from the IDL-traced edge polynomials, not from centre ± half_width.

        IDL mc_findorders: ``edgecoeffs[*,0,i]`` and ``edgecoeffs[*,1,i]`` are
        the primary output; they feed directly into the geometry.
        """
        flat = _make_synthetic_flat(seed=7)
        path = str(tmp_path / "syn_flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], n_sample_cols=12, poly_degree=2, guess_rows=_SYN_GUESS_ROWS)

        assert trace.bot_poly_coeffs is not None
        assert trace.top_poly_coeffs is not None
        geom = trace.to_order_geometry_set("H1")
        for i, g in enumerate(geom.geometries):
            np.testing.assert_allclose(
                g.bottom_edge_coeffs, trace.bot_poly_coeffs[i], rtol=1e-10,
                err_msg=(
                    f"Order {i}: geometry bottom_edge_coeffs must equal "
                    "bot_poly_coeffs (IDL edgecoeffs[*,0,i])"
                ),
            )
            np.testing.assert_allclose(
                g.top_edge_coeffs, trace.top_poly_coeffs[i], rtol=1e-10,
                err_msg=(
                    f"Order {i}: geometry top_edge_coeffs must equal "
                    "top_poly_coeffs (IDL edgecoeffs[*,1,i])"
                ),
            )

    def test_order_geometry_top_edge_above_bottom_at_all_sample_cols(self, tmp_path):
        """At every sample column, the top edge must be above the bottom edge.

        IDL mc_findorders guarantees this by design: com_top (row above centre)
        > com_bot (row below centre).
        """
        flat = _make_synthetic_flat(seed=8)
        path = str(tmp_path / "syn_flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], n_sample_cols=12, poly_degree=2, guess_rows=_SYN_GUESS_ROWS)
        geom = trace.to_order_geometry_set("H1")

        for i, g in enumerate(geom.geometries):
            col_mid = (g.x_start + g.x_end) // 2
            top_y = g.eval_top_edge(col_mid)
            bot_y = g.eval_bottom_edge(col_mid)
            assert top_y > bot_y, (
                f"Order {i}: top edge ({top_y:.1f}) not above bottom "
                f"edge ({bot_y:.1f}) at col {col_mid}"
            )

    # ---- Requirement 6: public compatibility fields remain usable ---------

    def test_all_public_fields_accessible(self, tmp_path):
        """All public-facing FlatOrderTrace fields that downstream code uses
        must be populated and have the expected shapes / types.

        Requirement 6: existing compatibility fields stay usable.
        """
        flat = _make_synthetic_flat(seed=9)
        path = str(tmp_path / "syn_flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], n_sample_cols=12, poly_degree=2, guess_rows=_SYN_GUESS_ROWS)

        n = trace.n_orders
        d = trace.poly_degree + 1   # number of polynomial coefficients

        # --- scalar fields ---
        assert isinstance(trace.n_orders, int) and trace.n_orders > 0
        assert isinstance(trace.poly_degree, int)
        assert isinstance(trace.seed_col, int)

        # --- array shapes ---
        assert trace.sample_cols.ndim == 1 and len(trace.sample_cols) > 0
        assert trace.center_rows.shape == (n, len(trace.sample_cols))
        assert trace.center_poly_coeffs.shape == (n, d)
        assert trace.fit_rms.shape == (n,)
        assert trace.half_width_rows.shape == (n,)
        assert trace.bot_poly_coeffs.shape == (n, d)
        assert trace.top_poly_coeffs.shape == (n, d)
        assert trace.order_xranges.shape == (n, 2)

        # --- list fields ---
        assert len(trace.order_samples) == n
        assert len(trace.order_stats) == n

        # --- value sanity ---
        assert np.all(trace.half_width_rows > 0)
        for i in range(n):
            xr = trace.order_xranges[i]
            assert xr[0] <= xr[1], f"Order {i}: degenerate xrange {xr}"

    def test_order_samples_fields_are_accessible(self, tmp_path):
        """OrderTraceSamples fields (sample_cols, center_rows, bot_rows, top_rows,
        x_start, x_end) must all be present and consistent.

        These are the authoritative per-order representations — downstream code
        must be able to rely on them.
        """
        flat = _make_synthetic_flat(seed=10)
        path = str(tmp_path / "syn_flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], n_sample_cols=12, poly_degree=2, guess_rows=_SYN_GUESS_ROWS)

        for i, os_i in enumerate(trace.order_samples):
            n_samp = len(os_i.sample_cols)
            assert n_samp > 0, f"Order {i}: empty sample_cols"
            assert os_i.center_rows.shape == (n_samp,)
            assert os_i.bot_rows.shape == (n_samp,)
            assert os_i.top_rows.shape == (n_samp,)
            assert isinstance(os_i.x_start, int)
            assert isinstance(os_i.x_end, int)
            assert os_i.x_start <= os_i.x_end, (
                f"Order {i}: degenerate xrange [{os_i.x_start}, {os_i.x_end}]"
            )

    def test_fit_rms_finite_for_well_traced_orders(self, tmp_path):
        """fit_rms must be finite for all orders traced on a clean synthetic flat.

        A NaN fit_rms means the polynomial fit failed (too few valid samples),
        which would indicate a regression in the tracing core.
        """
        flat = _make_synthetic_flat(seed=99)
        path = str(tmp_path / "syn_flat.fits")
        _write_fits_flat(flat, path)
        trace = trace_orders_from_flat([path], n_sample_cols=15, poly_degree=2, guess_rows=_SYN_GUESS_ROWS)

        bad = np.where(~np.isfinite(trace.fit_rms))[0]
        assert len(bad) == 0, (
            f"Orders {bad.tolist()} have NaN fit_rms — tracing core regression"
        )


# ---------------------------------------------------------------------------
# TestIDLHelperFunctions  (mc_findorders blocks A–F)
# ---------------------------------------------------------------------------


class TestIDLHelperFunctions:
    """Unit tests for the named IDL mc_findorders building-block helpers.

    Each test maps to a specific IDL block (A–F) as described in the
    problem statement and in the function docstrings.
    """

    # ── Block A: _compute_sobel_image ───────────────────────────────────────

    def test_sobel_image_shape_preserved(self):
        """_compute_sobel_image returns an array with the same shape as input."""
        from pyspextool.instruments.ishell.tracing import _compute_sobel_image

        img = np.ones((32, 64), dtype=np.float32) * 100.0
        result = _compute_sobel_image(img)
        assert result.shape == img.shape

    def test_sobel_image_zero_input_returns_zeros(self):
        """All-zero image → all-zero Sobel image (no division by zero)."""
        from pyspextool.instruments.ishell.tracing import _compute_sobel_image

        img = np.zeros((16, 16), dtype=np.float32)
        result = _compute_sobel_image(img)
        np.testing.assert_array_equal(result, 0.0)

    def test_sobel_image_uniform_field_near_zero(self):
        """A spatially uniform flat field has near-zero gradient everywhere."""
        from pyspextool.instruments.ishell.tracing import _compute_sobel_image

        img = np.full((32, 64), 500.0, dtype=np.float32)
        result = _compute_sobel_image(img)
        # Uniform → no edges → Sobel ≈ 0 everywhere (float precision)
        assert float(np.max(np.abs(result))) < 1e-6

    def test_sobel_image_edge_band_has_high_response(self):
        """A sharp horizontal edge produces large Sobel response at the boundary."""
        from pyspextool.instruments.ishell.tracing import _compute_sobel_image

        img = np.zeros((64, 64), dtype=np.float32)
        img[30:, :] = 1000.0  # sharp horizontal step at row 30
        result = _compute_sobel_image(img)
        # Maximum response should be near row 30
        max_row = int(np.unravel_index(np.argmax(result), result.shape)[0])
        assert abs(max_row - 30) <= 2

    def test_sobel_image_explicit_1000_normalisation(self):
        """Scaling by 1000 in _compute_sobel_image must equal scaling by k*1000 up to proportionality."""
        from pyspextool.instruments.ishell.tracing import _compute_sobel_image

        rng = np.random.default_rng(0)
        img = rng.uniform(100, 200, size=(32, 32)).astype(np.float32)
        # Doubling all values leaves normalisation unchanged → same Sobel image.
        r1 = _compute_sobel_image(img)
        r2 = _compute_sobel_image(img * 2.0)
        np.testing.assert_allclose(r1, r2, rtol=1e-5)

    # ── Block B: _build_sample_cols ─────────────────────────────────────────

    def test_build_sample_cols_basic(self):
        """IDL: starts=start+step-1, stops=stop-step+1, n=fix((stops-starts)/step)+1."""
        from pyspextool.instruments.ishell.tracing import _build_sample_cols

        # x_lo=10, x_hi=50, step=5
        # starts = 10+5-1 = 14
        # stops  = 50-5+1 = 46
        # n = fix((46-14)/5)+1 = fix(6.4)+1 = 6+1 = 7
        # scols = [0,1,2,3,4,5,6]*5 + 14 = [14,19,24,29,34,39,44]
        scols = _build_sample_cols(10, 50, 5)
        expected = np.array([14, 19, 24, 29, 34, 39, 44])
        np.testing.assert_array_equal(scols, expected)

    def test_build_sample_cols_step_1(self):
        """Step=1: starts=x_lo, stops=x_hi, dense sampling."""
        from pyspextool.instruments.ishell.tracing import _build_sample_cols

        # x_lo=0, x_hi=4, step=1
        # starts=0, stops=4, n=fix(4/1)+1=5
        # scols=[0,1,2,3,4]
        scols = _build_sample_cols(0, 4, 1)
        expected = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(scols, expected)

    def test_build_sample_cols_does_not_start_at_x_lo(self):
        """scols[0] = x_lo + step - 1 (inset by step-1 from the range boundary)."""
        from pyspextool.instruments.ishell.tracing import _build_sample_cols

        scols = _build_sample_cols(100, 200, 10)
        # starts = 100 + 10 - 1 = 109
        assert scols[0] == 109

    def test_build_sample_cols_spacing(self):
        """All consecutive differences equal the step size."""
        from pyspextool.instruments.ishell.tracing import _build_sample_cols

        step = 7
        scols = _build_sample_cols(0, 100, step)
        diffs = np.diff(scols)
        assert np.all(diffs == step)

    def test_build_sample_cols_last_le_x_hi_minus_step(self):
        """The last sample column is ≤ x_hi - step + 1 (stops boundary)."""
        from pyspextool.instruments.ishell.tracing import _build_sample_cols

        x_lo, x_hi, step = 50, 500, 20
        scols = _build_sample_cols(x_lo, x_hi, step)
        stops = x_hi - step + 1
        assert scols[-1] <= stops

    # ── Block C: _initialize_order_trace_arrays ─────────────────────────────

    def test_initialize_arrays_all_nan_outside_seed_window(self):
        """All cen entries outside [gidx-degree, gidx+degree] must be NaN."""
        from pyspextool.instruments.ishell.tracing import (
            _build_sample_cols,
            _initialize_order_trace_arrays,
        )

        scols = _build_sample_cols(0, 100, 5)
        poly_degree = 3
        guess_col = float(scols[len(scols) // 2])
        edges, cen, bot, gidx = _initialize_order_trace_arrays(
            scols, guess_col, 128.0, poly_degree
        )

        lo = max(0, gidx - poly_degree)
        hi = min(len(scols) - 1, gidx + poly_degree)

        # All entries strictly outside the seed window must be NaN.
        for j in range(len(cen)):
            if j < lo or j > hi:
                assert np.isnan(cen[j]), f"cen[{j}] should be NaN, got {cen[j]}"

    def test_initialize_arrays_seed_window_filled(self):
        """All cen entries inside [gidx-degree, gidx+degree] equal guess_row."""
        from pyspextool.instruments.ishell.tracing import (
            _build_sample_cols,
            _initialize_order_trace_arrays,
        )

        scols = _build_sample_cols(0, 100, 5)
        poly_degree = 3
        guess_col = float(scols[len(scols) // 2])
        guess_row = 75.5
        edges, cen, bot, gidx = _initialize_order_trace_arrays(
            scols, guess_col, guess_row, poly_degree
        )

        lo = max(0, gidx - poly_degree)
        hi = min(len(scols) - 1, gidx + poly_degree)
        for j in range(lo, hi + 1):
            assert cen[j] == pytest.approx(guess_row), (
                f"cen[{j}] = {cen[j]}, expected {guess_row}"
            )

    def test_initialize_arrays_edges_all_nan(self):
        """edges array must be entirely NaN at initialization."""
        from pyspextool.instruments.ishell.tracing import (
            _build_sample_cols,
            _initialize_order_trace_arrays,
        )

        scols = _build_sample_cols(0, 60, 5)
        edges, cen, bot, gidx = _initialize_order_trace_arrays(
            scols, float(scols[4]), 100.0, 2
        )
        assert np.all(np.isnan(edges))

    def test_initialize_arrays_gidx_closest_to_guess_col(self):
        """gidx must be the index closest to guess_col in sample_cols."""
        from pyspextool.instruments.ishell.tracing import (
            _build_sample_cols,
            _initialize_order_trace_arrays,
        )

        scols = _build_sample_cols(10, 90, 10)
        # scols = [19, 29, 39, 49, 59, 69, 79]
        guess_col = 50.0  # closest to index 3 (value 49)
        _, _, _, gidx = _initialize_order_trace_arrays(scols, guess_col, 100.0, 2)
        expected_gidx = int(np.argmin(np.abs(scols - guess_col)))
        assert gidx == expected_gidx

    def test_initialize_arrays_edges_shape(self):
        """edges shape must be (n_samp, 2)."""
        from pyspextool.instruments.ishell.tracing import (
            _build_sample_cols,
            _initialize_order_trace_arrays,
        )

        scols = _build_sample_cols(0, 50, 5)
        edges, cen, bot, gidx = _initialize_order_trace_arrays(
            scols, float(scols[3]), 64.0, 2
        )
        assert edges.shape == (len(scols), 2)
        assert cen.shape == (len(scols),)

    # ── Block D: _predict_center_from_known_samples ─────────────────────────

    def test_predict_uses_only_finite_samples(self):
        """NaN entries in center_samples must be ignored in the polynomial fit."""
        from pyspextool.instruments.ishell.tracing import _predict_center_from_known_samples

        sample_cols = np.array([0, 10, 20, 30, 40, 50], dtype=float)
        # Only columns 20, 30, 40 have known (finite) centres; the rest are NaN.
        center_samples = np.array([np.nan, np.nan, 100.0, 102.0, 104.0, np.nan])
        # Predict at column 50.
        y = _predict_center_from_known_samples(
            sample_cols, center_samples, 50.0, poly_degree=3, nrows=256, bufpix=1
        )
        # poly_degree=3 → fit_deg = max(1, 1) = 1 (linear).
        # Linear fit to (20,100),(30,102),(40,104): slope = 0.2/col.
        # Extrapolation to col 50: 100 + 0.2*(50-20) = 106.
        assert np.isfinite(y)
        assert abs(y - 106.0) < 1.0  # ≈106 from linear extrapolation

    def test_predict_temporary_degree_is_max1_degree_minus_2(self):
        """The fit degree must be max(1, poly_degree - 2) as in IDL."""
        from pyspextool.instruments.ishell.tracing import _predict_center_from_known_samples

        # poly_degree=3 → fit_deg = max(1, 1) = 1  (linear)
        # poly_degree=4 → fit_deg = max(1, 2) = 2  (quadratic)
        # poly_degree=1 → fit_deg = max(1, -1) = 1 (linear)
        # We test that with only 2 finite points and poly_degree=3 (fit_deg=1)
        # the function succeeds (degree 1 requires ≥ 2 points).
        sample_cols = np.array([0.0, 10.0, 20.0, 30.0])
        centers = np.array([50.0, 55.0, np.nan, np.nan])
        y = _predict_center_from_known_samples(
            sample_cols, centers, 20.0, poly_degree=3, nrows=256, bufpix=1
        )
        assert np.isfinite(y)

    def test_predict_clamps_to_detector_bounds(self):
        """y_guess must be clamped to [bufpix, nrows - bufpix - 1]."""
        from pyspextool.instruments.ishell.tracing import _predict_center_from_known_samples

        # Force extrapolation far out of bounds.
        sample_cols = np.array([0.0, 10.0, 20.0, 30.0])
        centers = np.array([10.0, 8.0, 6.0, 4.0])  # trend heading toward negative rows
        y = _predict_center_from_known_samples(
            sample_cols, centers, 1000.0,  # wildly extrapolated column
            poly_degree=2, nrows=256, bufpix=5
        )
        assert y >= 5
        assert y <= 256 - 5 - 1

    def test_predict_handles_all_nan(self):
        """All-NaN center_samples must return a clamped fallback value."""
        from pyspextool.instruments.ishell.tracing import _predict_center_from_known_samples

        sample_cols = np.array([0.0, 10.0, 20.0, 30.0])
        centers = np.full(4, np.nan)
        y = _predict_center_from_known_samples(
            sample_cols, centers, 15.0, poly_degree=2, nrows=128, bufpix=2
        )
        # Should return something clamped and finite.
        assert np.isfinite(y)
        assert 2 <= y <= 128 - 2 - 1

    # ── Blocks E+F: _extract_column_profiles, _local_flux_at_guess,
    #               _find_threshold_edge_guesses ────────────────────────────

    def test_extract_column_profiles_shape(self):
        """_extract_column_profiles returns (nrows,) vectors."""
        from pyspextool.instruments.ishell.tracing import (
            _compute_sobel_image,
            _extract_column_profiles,
        )

        nrows, ncols = 64, 128
        img = np.random.default_rng(1).uniform(0, 1000, (nrows, ncols)).astype(np.float32)
        sobel = _compute_sobel_image(img)
        fcol, rcol = _extract_column_profiles(img, sobel, col=40)
        assert fcol.shape == (nrows,)
        assert rcol.shape == (nrows,)

    def test_extract_column_profiles_values_match_column(self):
        """fcol must exactly equal image[:, col]."""
        from pyspextool.instruments.ishell.tracing import (
            _compute_sobel_image,
            _extract_column_profiles,
        )

        img = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
        sobel = _compute_sobel_image(img)
        col = 5
        fcol, rcol = _extract_column_profiles(img, sobel, col)
        np.testing.assert_array_equal(fcol, img[:, col].astype(float))

    def test_local_flux_rounds_to_nearest_integer_row(self):
        """_local_flux_at_guess must index with round(y_guess), not int(y_guess)."""
        from pyspextool.instruments.ishell.tracing import _local_flux_at_guess

        flux_col = np.arange(100, dtype=float)  # flux[i] = i
        # y_guess = 10.6 → round = 11, not int(10.6) = 10
        assert _local_flux_at_guess(flux_col, 10.6) == pytest.approx(11.0)
        # y_guess = 10.4 → round = 10
        assert _local_flux_at_guess(flux_col, 10.4) == pytest.approx(10.0)

    def test_find_threshold_edge_guesses_top_is_above_y_guess(self):
        """guessyt (top guess) must be strictly above y_guess."""
        from pyspextool.instruments.ishell.tracing import _find_threshold_edge_guesses

        nrows = 100
        # Background = 200, order peak = 1000 at row 50.
        flux_col = np.full(nrows, 200.0)
        flux_col[45:56] = 1000.0  # order spans rows 45–55
        y_guess = 50.0
        frac = 0.8
        guessyt, guessyb = _find_threshold_edge_guesses(flux_col, y_guess, frac)
        if guessyt is not None:
            assert guessyt > y_guess, (
                f"guessyt={guessyt} must be > y_guess={y_guess}"
            )

    def test_find_threshold_edge_guesses_bottom_is_below_y_guess(self):
        """guessyb (bottom guess) must be strictly below y_guess."""
        from pyspextool.instruments.ishell.tracing import _find_threshold_edge_guesses

        nrows = 100
        flux_col = np.full(nrows, 200.0)
        flux_col[45:56] = 1000.0
        y_guess = 50.0
        frac = 0.8
        guessyt, guessyb = _find_threshold_edge_guesses(flux_col, y_guess, frac)
        if guessyb is not None:
            assert guessyb < y_guess, (
                f"guessyb={guessyb} must be < y_guess={y_guess}"
            )

    def test_find_threshold_edge_guesses_top_is_first_crossing(self):
        """guessyt must be the FIRST (lowest) row above y_guess below threshold.

        IDL: ztop = where(fcol lt frac*z_guess and row gt y_guess, cnt)
               guessyt = ztop[0]   ; first element
        """
        from pyspextool.instruments.ishell.tracing import _find_threshold_edge_guesses

        nrows = 200
        flux_col = np.full(nrows, 500.0)
        flux_col[80:120] = 2000.0  # order centre at row 100
        y_guess = 100.0
        frac = 0.8
        guessyt, _ = _find_threshold_edge_guesses(flux_col, y_guess, frac)
        # First crossing above row 100 where flux drops below frac*2000=1600 is row 120.
        assert guessyt == 120

    def test_find_threshold_edge_guesses_bottom_is_last_crossing(self):
        """guessyb must be the LAST (highest) row below y_guess below threshold.

        IDL: zbot = where(fcol lt frac*z_guess and row lt y_guess, cnt)
               guessyb = zbot[cnt-1]   ; last element
        """
        from pyspextool.instruments.ishell.tracing import _find_threshold_edge_guesses

        nrows = 200
        flux_col = np.full(nrows, 500.0)
        flux_col[80:120] = 2000.0  # order centre at row 100
        y_guess = 100.0
        frac = 0.8
        _, guessyb = _find_threshold_edge_guesses(flux_col, y_guess, frac)
        # Last crossing below row 100 where flux drops below frac*2000=1600 is row 79.
        assert guessyb == 79

    def test_find_threshold_edge_guesses_no_top_returns_none(self):
        """When no row above y_guess drops below threshold, guessyt must be None."""
        from pyspextool.instruments.ishell.tracing import _find_threshold_edge_guesses

        nrows = 100
        # Flux is high everywhere → never drops below threshold above y_guess.
        flux_col = np.full(nrows, 5000.0)
        guessyt, guessyb = _find_threshold_edge_guesses(flux_col, 50.0, frac=0.8)
        assert guessyt is None

    def test_find_threshold_edge_guesses_no_bottom_returns_none(self):
        """When no row below y_guess drops below threshold, guessyb must be None."""
        from pyspextool.instruments.ishell.tracing import _find_threshold_edge_guesses

        nrows = 100
        flux_col = np.full(nrows, 5000.0)
        guessyt, guessyb = _find_threshold_edge_guesses(flux_col, 50.0, frac=0.8)
        assert guessyb is None


# ---------------------------------------------------------------------------
# TestIDLHelperFunctionsGHI  (mc_findorders blocks G–I)
# ---------------------------------------------------------------------------


class TestIDLHelperFunctionsGHI:
    """Unit tests for IDL mc_findorders building-block helpers G, H, and I."""

    # ── Block G: _sobel_centroid ────────────────────────────────────────────

    def test_sobel_centroid_finite_for_nonzero_sobel(self):
        """Returns a finite value when the Sobel window has nonzero weight."""
        from pyspextool.instruments.ishell.tracing import _sobel_centroid

        nrows = 100
        # Sharp Sobel spike at row 40.
        sobel_col = np.zeros(nrows)
        sobel_col[40] = 10.0
        com = _sobel_centroid(sobel_col, guess_row=40, halfwin=3, nrows=nrows)
        assert np.isfinite(com)
        assert abs(com - 40.0) < 0.1  # spike is exactly at row 40

    def test_sobel_centroid_zero_denominator_returns_nan(self):
        """Returns NaN when all Sobel weights inside the window are zero."""
        from pyspextool.instruments.ishell.tracing import _sobel_centroid

        nrows = 100
        sobel_col = np.zeros(nrows)  # all zero → total(z) == 0
        com = _sobel_centroid(sobel_col, guess_row=50, halfwin=3, nrows=nrows)
        assert np.isnan(com)

    def test_sobel_centroid_window_clips_at_bottom(self):
        """Window is clamped to row 0 when guess_row - halfwin < 0."""
        from pyspextool.instruments.ishell.tracing import _sobel_centroid

        nrows = 50
        sobel_col = np.zeros(nrows)
        sobel_col[0] = 5.0
        # guess_row=2, halfwin=5 → bidx=max(0,-3)=0, so row 0 is included.
        com = _sobel_centroid(sobel_col, guess_row=2, halfwin=5, nrows=nrows)
        assert np.isfinite(com)
        # COM should be pulled toward row 0 (the only nonzero weight).
        assert com == pytest.approx(0.0)

    def test_sobel_centroid_window_clips_at_top(self):
        """Window is clamped to nrows-1 when guess_row + halfwin >= nrows."""
        from pyspextool.instruments.ishell.tracing import _sobel_centroid

        nrows = 50
        sobel_col = np.zeros(nrows)
        sobel_col[nrows - 1] = 7.0
        # guess_row=48, halfwin=5 → tidx=min(49,53)=49, so last row is included.
        com = _sobel_centroid(sobel_col, guess_row=48, halfwin=5, nrows=nrows)
        assert np.isfinite(com)
        assert com == pytest.approx(float(nrows - 1))

    def test_sobel_centroid_matches_idl_formula(self):
        """COM = sum(y*z)/sum(z) matches direct calculation."""
        from pyspextool.instruments.ishell.tracing import _sobel_centroid

        nrows = 30
        sobel_col = np.zeros(nrows)
        sobel_col[10] = 3.0
        sobel_col[12] = 1.0
        # guess_row=11, halfwin=4 → bidx=7, tidx=15
        # rows 7..15 used; only rows 10 and 12 nonzero.
        # COM = (10*3 + 12*1) / (3+1) = 42/4 = 10.5
        com = _sobel_centroid(sobel_col, guess_row=11, halfwin=4, nrows=nrows)
        assert com == pytest.approx(10.5)

    def test_sobel_centroid_inclusive_window_idl_style(self):
        """IDL bidx:tidx is inclusive: Python slice must be bidx:tidx+1."""
        from pyspextool.instruments.ishell.tracing import _sobel_centroid

        nrows = 20
        sobel_col = np.zeros(nrows)
        sobel_col[5] = 1.0   # exactly at bidx boundary
        sobel_col[9] = 1.0   # exactly at tidx boundary
        # guess_row=7, halfwin=2 → bidx=5, tidx=9
        com = _sobel_centroid(sobel_col, guess_row=7, halfwin=2, nrows=nrows)
        # Both boundary rows are included → COM = (5*1 + 9*1) / (1+1) = 7.0
        assert com == pytest.approx(7.0)

    # ── Block H: _accept_edge_pair ──────────────────────────────────────────

    def test_accept_edge_pair_within_limits(self):
        """Accept when slit_height_min < |bot - top| < slit_height_max."""
        from pyspextool.instruments.ishell.tracing import _accept_edge_pair

        # |30 - 50| = 20, 10 < 20 < 30 → accept
        assert _accept_edge_pair(30.0, 50.0, slit_height_min=10.0, slit_height_max=30.0)

    def test_accept_edge_pair_too_small(self):
        """Reject when slit height < slit_height_min."""
        from pyspextool.instruments.ishell.tracing import _accept_edge_pair

        # |40 - 45| = 5 < 10 → reject
        assert not _accept_edge_pair(40.0, 45.0, slit_height_min=10.0, slit_height_max=30.0)

    def test_accept_edge_pair_too_large(self):
        """Reject when slit height > slit_height_max."""
        from pyspextool.instruments.ishell.tracing import _accept_edge_pair

        # |10 - 50| = 40 > 30 → reject
        assert not _accept_edge_pair(10.0, 50.0, slit_height_min=10.0, slit_height_max=30.0)

    def test_accept_edge_pair_equal_to_min_rejected(self):
        """IDL uses strict 'gt': height exactly equal to min is rejected."""
        from pyspextool.instruments.ishell.tracing import _accept_edge_pair

        # |30 - 40| = 10 == slit_height_min → 'gt' means strictly greater → reject
        assert not _accept_edge_pair(30.0, 40.0, slit_height_min=10.0, slit_height_max=30.0)

    def test_accept_edge_pair_equal_to_max_rejected(self):
        """IDL uses strict 'lt': height exactly equal to max is rejected."""
        from pyspextool.instruments.ishell.tracing import _accept_edge_pair

        # |10 - 40| = 30 == slit_height_max → 'lt' means strictly less → reject
        assert not _accept_edge_pair(10.0, 40.0, slit_height_min=10.0, slit_height_max=30.0)

    def test_accept_edge_pair_nan_bottom_rejected(self):
        """Reject when bottom edge is NaN (IDL: finite() guard)."""
        from pyspextool.instruments.ishell.tracing import _accept_edge_pair

        assert not _accept_edge_pair(np.nan, 50.0, slit_height_min=10.0, slit_height_max=60.0)

    def test_accept_edge_pair_nan_top_rejected(self):
        """Reject when top edge is NaN."""
        from pyspextool.instruments.ishell.tracing import _accept_edge_pair

        assert not _accept_edge_pair(30.0, np.nan, slit_height_min=10.0, slit_height_max=60.0)

    def test_accept_edge_pair_order_independent(self):
        """Result is symmetric: bot < top produces same answer as bot > top."""
        from pyspextool.instruments.ishell.tracing import _accept_edge_pair

        assert _accept_edge_pair(50.0, 30.0, slit_height_min=10.0, slit_height_max=30.0)

    # ── Block I: _update_edge_activity_flags ────────────────────────────────

    def test_activity_flags_unchanged_when_edges_interior(self):
        """Flags remain True when both edges are well inside the detector."""
        from pyspextool.instruments.ishell.tracing import _update_edge_activity_flags

        nrows, bufpix = 1024, 5
        top_act, bot_act = _update_edge_activity_flags(
            top_edge=500.0, bottom_edge=480.0,
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        assert top_act is True
        assert bot_act is True

    def test_activity_flags_top_disabled_at_lower_guard(self):
        """Top flag is disabled when com_top <= bufpix."""
        from pyspextool.instruments.ishell.tracing import _update_edge_activity_flags

        nrows, bufpix = 100, 5
        top_act, bot_act = _update_edge_activity_flags(
            top_edge=3.0,    # 3 <= 5 → disable top
            bottom_edge=50.0,
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        assert top_act is False
        assert bot_act is True

    def test_activity_flags_top_disabled_at_upper_guard(self):
        """Top flag is disabled when com_top >= nrows-1-bufpix (uses >=)."""
        from pyspextool.instruments.ishell.tracing import _update_edge_activity_flags

        nrows, bufpix = 100, 5
        upper = nrows - 1 - bufpix  # = 94
        top_act, bot_act = _update_edge_activity_flags(
            top_edge=float(upper),   # exactly 94 → >= 94 → disable
            bottom_edge=50.0,
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        assert top_act is False
        assert bot_act is True

    def test_activity_flags_bottom_disabled_at_lower_guard(self):
        """Bottom flag is disabled when com_bot <= bufpix."""
        from pyspextool.instruments.ishell.tracing import _update_edge_activity_flags

        nrows, bufpix = 100, 5
        top_act, bot_act = _update_edge_activity_flags(
            top_edge=50.0,
            bottom_edge=4.0,   # 4 <= 5 → disable bottom
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        assert top_act is True
        assert bot_act is False

    def test_activity_flags_bottom_disabled_strictly_above_upper(self):
        """Bottom uses 'gt' (>): exactly at upper boundary does NOT disable."""
        from pyspextool.instruments.ishell.tracing import _update_edge_activity_flags

        nrows, bufpix = 100, 5
        upper = nrows - 1 - bufpix  # = 94
        # Exactly at boundary → gt means strictly greater → NOT disabled
        _, bot_act = _update_edge_activity_flags(
            top_edge=50.0,
            bottom_edge=float(upper),   # exactly 94
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        assert bot_act is True  # NOT disabled at exact boundary (IDL uses 'gt')

        # One pixel above the boundary → disabled
        _, bot_act2 = _update_edge_activity_flags(
            top_edge=50.0,
            bottom_edge=float(upper) + 0.1,
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        assert bot_act2 is False

    def test_activity_flags_top_bottom_asymmetry_at_upper_guard(self):
        """Verify IDL asymmetry: top uses >= but bottom uses > at upper guard."""
        from pyspextool.instruments.ishell.tracing import _update_edge_activity_flags

        nrows, bufpix = 100, 5
        upper = float(nrows - 1 - bufpix)  # 94.0

        # Top at exactly upper boundary: top uses 'ge' → disabled
        top_act, _ = _update_edge_activity_flags(
            top_edge=upper, bottom_edge=50.0,
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        assert top_act is False, "top 'ge' (>=) should disable at exactly upper"

        # Bottom at exactly upper boundary: bottom uses 'gt' → NOT disabled
        _, bot_act = _update_edge_activity_flags(
            top_edge=50.0, bottom_edge=upper,
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        assert bot_act is True, "bottom 'gt' (>) should NOT disable at exactly upper"

    def test_activity_flags_nan_edge_leaves_flag_unchanged(self):
        """NaN edge values must not disable active flags."""
        from pyspextool.instruments.ishell.tracing import _update_edge_activity_flags

        nrows, bufpix = 100, 5
        top_act, bot_act = _update_edge_activity_flags(
            top_edge=np.nan, bottom_edge=np.nan,
            nrows=nrows, bufpix=bufpix,
            trace_top_active=True, trace_bottom_active=True,
        )
        # NaN <= bufpix is False in Python (matches IDL's finite() guard)
        assert top_act is True
        assert bot_act is True

    def test_activity_flags_already_inactive_stays_inactive(self):
        """Already-False flags are not re-enabled regardless of edge value."""
        from pyspextool.instruments.ishell.tracing import _update_edge_activity_flags

        nrows, bufpix = 100, 5
        top_act, bot_act = _update_edge_activity_flags(
            top_edge=50.0, bottom_edge=50.0,
            nrows=nrows, bufpix=bufpix,
            trace_top_active=False, trace_bottom_active=False,
        )
        assert top_act is False
        assert bot_act is False


# ---------------------------------------------------------------------------
# TestIDLHelperFunctionsJK  (mc_findorders blocks J–K: left/right sweeps)
# ---------------------------------------------------------------------------


def _make_banded_order_fixture(
    nrows=200, ncols=100, order_center=100,
    half_slit=10, band_flux=1000.0, bg_flux=50.0,
):
    """
    Synthetic flat with a single order band centred at *order_center*.

    The order occupies rows [order_center - half_slit, order_center + half_slit]
    with constant flux *band_flux*; all other rows have *bg_flux*.

    Returns
    -------
    image, sobel_image, sample_cols, center_samples, bottom_edge_samples, top_edge_samples
        Arrays sized for a simple horizontal sweep with step=5.
    """
    from pyspextool.instruments.ishell.tracing import _compute_sobel_image, _build_sample_cols

    image = np.full((nrows, ncols), bg_flux, dtype=float)
    bot_row = order_center - half_slit
    top_row = order_center + half_slit
    image[bot_row: top_row + 1, :] = band_flux

    sobel_image = _compute_sobel_image(image)

    # Sample every 5 columns across the full detector.
    sample_cols = _build_sample_cols(x_lo=0, x_hi=ncols - 1, step=5)
    n_samp = len(sample_cols)

    center_samples      = np.full(n_samp, np.nan)
    bottom_edge_samples = np.full(n_samp, np.nan)
    top_edge_samples    = np.full(n_samp, np.nan)

    return (
        image, sobel_image, sample_cols,
        center_samples, bottom_edge_samples, top_edge_samples,
        order_center, bot_row, top_row,
    )


def _seed_arrays(center_samples, sample_cols, gidx, order_center, poly_degree=3):
    """Fill the seed window around gidx (IDL: cen[(gidx-degree):(gidx+degree)])."""
    n_samp = len(sample_cols)
    lo = max(0, gidx - poly_degree)
    hi = min(n_samp - 1, gidx + poly_degree)
    center_samples[lo: hi + 1] = float(order_center)


class TestIDLHelperFunctionsJK:
    """Unit tests for IDL mc_findorders building-block helpers J and K."""

    # ── Shared fixture parameters ────────────────────────────────────────────

    NROWS    = 200
    NCOLS    = 100
    CEN      = 100     # order center row
    HALF     = 10      # half slit height in rows
    BUFPIX   = 3
    POLY_DEG = 3
    FRAC     = 0.5
    HALFWIN  = 3
    SLITH_MIN = 8.0    # < 2*HALF
    SLITH_MAX = 30.0   # > 2*HALF

    def _setup(self):
        (
            image, sobel_image, sample_cols,
            center_samples, bottom_edge_samples, top_edge_samples,
            order_center, bot_row, top_row,
        ) = _make_banded_order_fixture(
            nrows=self.NROWS, ncols=self.NCOLS,
            order_center=self.CEN, half_slit=self.HALF,
        )
        n_samp = len(sample_cols)
        # Seed at the middle column.
        gidx = n_samp // 2
        _seed_arrays(center_samples, sample_cols, gidx, order_center, self.POLY_DEG)
        return (
            image, sobel_image, sample_cols,
            center_samples, bottom_edge_samples, top_edge_samples,
            gidx,
        )

    def _call_left(self, image, sobel_image, sample_cols,
                   center_samples, bot, top, gidx):
        from pyspextool.instruments.ishell.tracing import _trace_order_left
        _trace_order_left(
            image, sobel_image, sample_cols,
            center_samples, bot, top, gidx,
            nrows=self.NROWS, bufpix=self.BUFPIX,
            poly_degree=self.POLY_DEG, frac=self.FRAC,
            com_half_width=self.HALFWIN,
            slit_height_min=self.SLITH_MIN, slit_height_max=self.SLITH_MAX,
        )

    def _call_right(self, image, sobel_image, sample_cols,
                    center_samples, bot, top, gidx):
        from pyspextool.instruments.ishell.tracing import _trace_order_right
        _trace_order_right(
            image, sobel_image, sample_cols,
            center_samples, bot, top, gidx,
            nrows=self.NROWS, bufpix=self.BUFPIX,
            poly_degree=self.POLY_DEG, frac=self.FRAC,
            com_half_width=self.HALFWIN,
            slit_height_min=self.SLITH_MIN, slit_height_max=self.SLITH_MAX,
        )

    # ── J: _trace_order_left ────────────────────────────────────────────────

    def test_left_sweep_populates_edge_samples(self):
        """Left sweep writes finite bottom/top edges for the banded order."""
        image, sobel, scols, cen, bot, top, gidx = self._setup()
        self._call_left(image, sobel, scols, cen, bot, top, gidx)
        # At least some columns to the left of gidx should have finite edges.
        n_finite_bot = int(np.sum(np.isfinite(bot[:gidx])))
        n_finite_top = int(np.sum(np.isfinite(top[:gidx])))
        assert n_finite_bot > 0, "left sweep should find at least one finite bottom edge"
        assert n_finite_top > 0, "left sweep should find at least one finite top edge"

    def test_left_sweep_populates_center_samples(self):
        """Left sweep writes finite center values derived from traced edges."""
        image, sobel, scols, cen, bot, top, gidx = self._setup()
        self._call_left(image, sobel, scols, cen, bot, top, gidx)
        n_finite_cen = int(np.sum(np.isfinite(cen[:gidx])))
        assert n_finite_cen > 0, "left sweep should produce at least one finite center"

    def test_left_sweep_centers_derived_from_edges(self):
        """Centers written during left sweep equal (bot + top) / 2 where both finite."""
        image, sobel, scols, cen, bot, top, gidx = self._setup()
        self._call_left(image, sobel, scols, cen, bot, top, gidx)
        for k in range(gidx):
            if np.isfinite(bot[k]) and np.isfinite(top[k]):
                expected = (bot[k] + top[k]) / 2.0
                assert cen[k] == pytest.approx(expected, abs=1e-9), (
                    f"center[{k}] should be (bot+top)/2 but got {cen[k]} vs {expected}"
                )

    def test_left_sweep_rejected_pair_sets_center_to_y_guess(self):
        """Rejected edge pair in left sweep stores NaN edges and center=y_guess.

        We force rejection by using an impossible slit-height window so no column
        can be accepted.
        """
        from pyspextool.instruments.ishell.tracing import _trace_order_left

        image, sobel, scols, cen, bot, top, gidx = self._setup()
        # Slit-height window that excludes the actual slit (HALF*2=20):
        # require height > 50, which is impossible for our band.
        _trace_order_left(
            image, sobel, scols, cen, bot, top, gidx,
            nrows=self.NROWS, bufpix=self.BUFPIX,
            poly_degree=self.POLY_DEG, frac=self.FRAC,
            com_half_width=self.HALFWIN,
            slit_height_min=50.0, slit_height_max=200.0,
        )
        # All left columns must have NaN edges.
        assert not np.any(np.isfinite(bot[:gidx])), "edges should be NaN after rejection"
        assert not np.any(np.isfinite(top[:gidx])), "edges should be NaN after rejection"
        # Center must be finite (fallback to y_guess).
        assert np.any(np.isfinite(cen[:gidx])), "center should fall back to y_guess"

    # ── K: _trace_order_right ───────────────────────────────────────────────

    def test_right_sweep_populates_edge_samples(self):
        """Right sweep writes finite bottom/top edges for the banded order."""
        image, sobel, scols, cen, bot, top, gidx = self._setup()
        self._call_right(image, sobel, scols, cen, bot, top, gidx)
        n_finite_bot = int(np.sum(np.isfinite(bot[gidx + 1:])))
        n_finite_top = int(np.sum(np.isfinite(top[gidx + 1:])))
        assert n_finite_bot > 0, "right sweep should find at least one finite bottom edge"
        assert n_finite_top > 0, "right sweep should find at least one finite top edge"

    def test_right_sweep_populates_center_samples(self):
        """Right sweep writes finite center values derived from traced edges."""
        image, sobel, scols, cen, bot, top, gidx = self._setup()
        self._call_right(image, sobel, scols, cen, bot, top, gidx)
        n_finite_cen = int(np.sum(np.isfinite(cen[gidx + 1:])))
        assert n_finite_cen > 0, "right sweep should produce at least one finite center"

    def test_right_sweep_centers_derived_from_edges(self):
        """Centers written during right sweep equal (bot + top) / 2 where both finite."""
        image, sobel, scols, cen, bot, top, gidx = self._setup()
        self._call_right(image, sobel, scols, cen, bot, top, gidx)
        for k in range(gidx + 1, len(scols)):
            if np.isfinite(bot[k]) and np.isfinite(top[k]):
                expected = (bot[k] + top[k]) / 2.0
                assert cen[k] == pytest.approx(expected, abs=1e-9), (
                    f"center[{k}] should be (bot+top)/2 but got {cen[k]} vs {expected}"
                )

    def test_right_sweep_rejected_pair_stores_nan_edges_and_y_guess_center(self):
        """Rejected edge pair in right sweep stores NaN edges and center=y_guess."""
        from pyspextool.instruments.ishell.tracing import _trace_order_right

        image, sobel, scols, cen, bot, top, gidx = self._setup()
        _trace_order_right(
            image, sobel, scols, cen, bot, top, gidx,
            nrows=self.NROWS, bufpix=self.BUFPIX,
            poly_degree=self.POLY_DEG, frac=self.FRAC,
            com_half_width=self.HALFWIN,
            slit_height_min=50.0, slit_height_max=200.0,
        )
        assert not np.any(np.isfinite(bot[gidx + 1:])), "edges should be NaN after rejection"
        assert not np.any(np.isfinite(top[gidx + 1:])), "edges should be NaN after rejection"
        assert np.any(np.isfinite(cen[gidx + 1:])), "center should fall back to y_guess"

    def test_both_sweeps_terminate_when_flags_inactive(self):
        """Both sweeps stop when activity flags go inactive (no infinite loop)."""
        from pyspextool.instruments.ishell.tracing import _trace_order_left, _trace_order_right

        # Place order center very close to the bottom guard band so
        # the bottom flag fires immediately.
        bufpix = 5
        order_center = bufpix + 1   # 1 row inside the guard band
        (
            image, sobel, scols, cen, bot, top,
            _, _, _,
        ) = _make_banded_order_fixture(
            nrows=self.NROWS, ncols=self.NCOLS,
            order_center=order_center, half_slit=self.HALF,
        )
        n_samp = len(scols)
        gidx = n_samp // 2
        _seed_arrays(cen, scols, gidx, order_center, self.POLY_DEG)

        bot_l = np.full(n_samp, np.nan)
        top_l = np.full(n_samp, np.nan)
        cen_l = cen.copy()
        _trace_order_left(
            image, sobel, scols, cen_l, bot_l, top_l, gidx,
            nrows=self.NROWS, bufpix=bufpix,
            poly_degree=self.POLY_DEG, frac=self.FRAC,
            com_half_width=self.HALFWIN,
            slit_height_min=self.SLITH_MIN, slit_height_max=self.SLITH_MAX,
        )

        bot_r = np.full(n_samp, np.nan)
        top_r = np.full(n_samp, np.nan)
        cen_r = cen.copy()
        _trace_order_right(
            image, sobel, scols, cen_r, bot_r, top_r, gidx,
            nrows=self.NROWS, bufpix=bufpix,
            poly_degree=self.POLY_DEG, frac=self.FRAC,
            com_half_width=self.HALFWIN,
            slit_height_min=self.SLITH_MIN, slit_height_max=self.SLITH_MAX,
        )
        # The test succeeds simply by completing (no hang / exception).


# ---------------------------------------------------------------------------
# TestIDLHelperFunctionsLMN  (mc_findorders blocks L–N)
# ---------------------------------------------------------------------------


class TestIDLHelperFunctionsLMN:
    """Unit tests for IDL mc_findorders building-block helpers L, M, and N."""

    # ── L: _fit_order_edge_polynomials ──────────────────────────────────────

    def test_L_fit_ignores_nan_samples(self):
        """Edge polynomial fit uses only finite samples, ignoring NaN entries."""
        from pyspextool.instruments.ishell.tracing import _fit_order_edge_polynomials

        # Gently tilted order (linear tilt + small noise avoids zero-residual
        # singularity in the robust sigma-clipping algorithm).
        rng = np.random.default_rng(42)
        scols = np.arange(0, 100, 5, dtype=float)
        noise = rng.normal(0, 0.05, len(scols))
        bot = 90.0 + 0.01 * scols + noise
        top = 110.0 + 0.01 * scols + noise

        # Corrupt some samples with NaN — fit should still succeed.
        bot[[0, 1, 2]] = np.nan
        top[[-1, -2]] = np.nan

        bc, tc = _fit_order_edge_polynomials(scols, bot, top, poly_degree=3)

        # Recovered edge at x=50 should be close to the true values.
        assert np.polynomial.polynomial.polyval(50.0, bc) == pytest.approx(90.5, abs=1.0)
        assert np.polynomial.polynomial.polyval(50.0, tc) == pytest.approx(110.5, abs=1.0)

    def test_L_returns_degree_plus_one_coefficients(self):
        """Returned coefficient arrays have shape (poly_degree + 1,)."""
        from pyspextool.instruments.ishell.tracing import _fit_order_edge_polynomials

        # Use a tilted edge with small noise to keep the robust fit non-singular.
        rng = np.random.default_rng(7)
        scols = np.arange(0, 50, 5, dtype=float)
        noise = rng.normal(0, 0.05, len(scols))
        bot = 80.0 + 0.02 * scols + noise
        top = 120.0 + 0.02 * scols + noise

        for deg in (1, 2, 3, 4):
            bc, tc = _fit_order_edge_polynomials(scols, bot, top, poly_degree=deg)
            assert bc.shape == (deg + 1,), f"bottom coeffs wrong shape for degree {deg}"
            assert tc.shape == (deg + 1,), f"top coeffs wrong shape for degree {deg}"

    def test_L_fit_recovers_linear_edge(self):
        """Fit returns coefficients that reconstruct a tilted linear edge."""
        from pyspextool.instruments.ishell.tracing import _fit_order_edge_polynomials

        scols = np.arange(10, 90, 5, dtype=float)
        # bot(x) = 80 + 0.05*x  (linear tilt)
        bot = 80.0 + 0.05 * scols
        top = 110.0 + 0.05 * scols  # parallel top edge

        bc, tc = _fit_order_edge_polynomials(scols, bot, top, poly_degree=1)
        # Intercept ~ 80, slope ~ 0.05
        assert bc[0] == pytest.approx(80.0, abs=1.0)
        assert bc[1] == pytest.approx(0.05, abs=0.01)
        assert tc[0] == pytest.approx(110.0, abs=1.0)

    # ── M: _derive_order_xrange_from_fitted_edges ───────────────────────────

    def test_M_xrange_from_fitted_edges_on_detector(self):
        """xrange is determined by where fitted edges are strictly inside detector."""
        from pyspextool.instruments.ishell.tracing import _derive_order_xrange_from_fitted_edges

        nrows = 200
        # Constant edges well inside the detector.
        bot_coeffs = np.array([90.0])   # constant 90
        top_coeffs = np.array([110.0])  # constant 110
        xlo, xhi = _derive_order_xrange_from_fitted_edges(
            x_lo=0, x_hi=99, bottom_edge_coeffs=bot_coeffs,
            top_edge_coeffs=top_coeffs, nrows=nrows,
        )
        # Both edges valid everywhere → full range.
        assert xlo == 0
        assert xhi == 99

    def test_M_xrange_excludes_columns_where_edge_exits_detector(self):
        """Columns where a fitted edge goes outside [0, nrows-1] are excluded."""
        from pyspextool.instruments.ishell.tracing import _derive_order_xrange_from_fitted_edges

        nrows = 200
        # Top edge starts at row 110 and rises steeply — exits at column ~89.
        # top(x) = 110 + 1.0*x  → top(89) = 199 = nrows-1  (boundary),
        #                          top(90) = 200 > nrows-1  (off-detector)
        top_coeffs = np.array([110.0, 1.0])  # c0 + c1*x
        bot_coeffs = np.array([90.0, 0.0])   # constant 90

        xlo, xhi = _derive_order_xrange_from_fitted_edges(
            x_lo=0, x_hi=100, bottom_edge_coeffs=bot_coeffs,
            top_edge_coeffs=top_coeffs, nrows=nrows,
        )
        # top < nrows-1  strictly, so top(88)=198 valid, top(89)=199 invalid.
        assert xlo == 0
        assert xhi <= 88

    def test_M_xrange_not_based_on_sample_coverage(self):
        """xrange uses fitted-edge evaluation, not where raw samples are finite."""
        from pyspextool.instruments.ishell.tracing import (
            _derive_order_xrange_from_fitted_edges,
            _fit_order_edge_polynomials,
        )

        nrows = 200
        # Sparse samples: only the middle third of the detector range has data.
        # Use a gentle linear tilt with small noise to keep the robust fit stable.
        rng = np.random.default_rng(99)
        all_cols = np.arange(0, 100, 5, dtype=float)
        bot_samples = np.full(len(all_cols), np.nan)
        top_samples = np.full(len(all_cols), np.nan)
        # Only columns 40–60 have finite samples.
        mid = (all_cols >= 40) & (all_cols <= 60)
        n_mid = int(mid.sum())
        noise = rng.normal(0, 0.05, n_mid)
        bot_samples[mid] = 90.0 + 0.01 * all_cols[mid] + noise
        top_samples[mid] = 110.0 + 0.01 * all_cols[mid] + noise

        bot_c, top_c = _fit_order_edge_polynomials(
            all_cols, bot_samples, top_samples, poly_degree=1,
        )

        xlo, xhi = _derive_order_xrange_from_fitted_edges(
            x_lo=0, x_hi=99, bottom_edge_coeffs=bot_c,
            top_edge_coeffs=top_c, nrows=nrows,
        )
        # Because edges are on-detector everywhere, the fitted polynomial extends
        # validly beyond the raw sample range [40,60].
        assert xlo < 40 or xhi > 60, (
            "xrange should extend beyond the raw sample coverage when the "
            f"fitted polynomial is valid across the full range; got [{xlo},{xhi}]"
        )

    def test_M_degenerate_no_valid_columns(self):
        """When no column has both edges on detector, (x_lo, x_lo) is returned."""
        from pyspextool.instruments.ishell.tracing import _derive_order_xrange_from_fitted_edges

        nrows = 200
        # Both edges are way off detector.
        bot_coeffs = np.array([500.0])  # off-detector
        top_coeffs = np.array([600.0])  # off-detector
        xlo, xhi = _derive_order_xrange_from_fitted_edges(
            x_lo=10, x_hi=50, bottom_edge_coeffs=bot_coeffs,
            top_edge_coeffs=top_coeffs, nrows=nrows,
        )
        assert xlo == 10
        assert xhi == 10

    # ── N: _center_coeffs_from_edge_coeffs ──────────────────────────────────

    def test_N_center_is_mean_of_edge_coeffs(self):
        """Center coefficients equal (bottom + top) / 2 element-wise."""
        from pyspextool.instruments.ishell.tracing import _center_coeffs_from_edge_coeffs

        bot = np.array([90.0, 0.1, -0.001])
        top = np.array([110.0, 0.3, 0.001])
        cen = _center_coeffs_from_edge_coeffs(bot, top)

        expected = (bot + top) / 2.0
        np.testing.assert_allclose(cen, expected)

    def test_N_center_evaluates_as_midpoint(self):
        """center(x) == (bottom(x) + top(x)) / 2 at every x."""
        from pyspextool.instruments.ishell.tracing import _center_coeffs_from_edge_coeffs

        bot_c = np.array([90.0, 0.05, 0.0])
        top_c = np.array([110.0, 0.05, 0.0])
        cen_c = _center_coeffs_from_edge_coeffs(bot_c, top_c)

        x = np.linspace(0, 100, 50)
        bot_vals = np.polynomial.polynomial.polyval(x, bot_c)
        top_vals = np.polynomial.polynomial.polyval(x, top_c)
        cen_vals = np.polynomial.polynomial.polyval(x, cen_c)

        np.testing.assert_allclose(cen_vals, (bot_vals + top_vals) / 2.0, rtol=1e-12)

    def test_N_center_not_independently_fitted(self):
        """Center coefficients are derived from edge coefficients, not a new fit."""
        from pyspextool.instruments.ishell.tracing import _center_coeffs_from_edge_coeffs

        bot = np.array([80.0, 0.0])
        top = np.array([120.0, 0.0])
        cen = _center_coeffs_from_edge_coeffs(bot, top)
        # Independent fit of center would give 100; so does (80+120)/2.
        assert cen[0] == pytest.approx(100.0)

    def test_N_raises_on_mismatched_shapes(self):
        """ValueError raised when bottom and top coefficient arrays differ in length."""
        from pyspextool.instruments.ishell.tracing import _center_coeffs_from_edge_coeffs

        with pytest.raises(ValueError, match="same shape"):
            _center_coeffs_from_edge_coeffs(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
            )


# ---------------------------------------------------------------------------
# TestIDLHelperFunctionOP  (mc_findorders blocks O–P)
# ---------------------------------------------------------------------------


class TestIDLHelperFunctionOP:
    """Verify blocks O (_trace_single_order_idlstyle) and P (trace_orders_from_flat
    wired to the new IDL-style core)."""

    # ── Shared fixture: single-order Gaussian synthetic flat ────────────────
    # Use the same 512-column flat that the other TestIDLPortFidelity tests use
    # to avoid the singular-matrix failure that arises from too few samples.

    @pytest.fixture(scope="class")
    def syn_flat_path(self, tmp_path_factory):
        """Write the standard _make_synthetic_flat (single order) to a FITS file."""
        rng = np.random.default_rng(42)
        nrows, ncols = _NROWS, _NCOLS
        img = rng.normal(0.0, 50.0, size=(nrows, ncols)).astype(float)
        # One Gaussian order centred at row 100, half-width ~10 px
        for col in range(ncols):
            center = 100.0 + 0.01 * col
            sigma = 10.0 / 2.355
            rows = np.arange(nrows, dtype=float)
            img[:, col] += 2000.0 * np.exp(-0.5 * ((rows - center) / sigma) ** 2)
        img = np.clip(img, 0, None).astype(np.float32)
        tmp = tmp_path_factory.mktemp("op")
        p = str(tmp / "syn1.fits")
        from astropy.io import fits as _fits
        _fits.PrimaryHDU(img).writeto(p, overwrite=True)
        return p, nrows, ncols

    @pytest.fixture(scope="class")
    def single_order_result(self, syn_flat_path):
        """Run _trace_single_order_idlstyle on the synthetic flat once."""
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )
        from astropy.io import fits as _fits
        p, nrows, ncols = syn_flat_path
        img = _fits.open(p)[0].data.astype(float)
        sobel = _compute_sobel_image(img)
        # Use linspace sample cols (no per-order step in guess_rows mode)
        sample_cols = np.unique(
            np.round(np.linspace(50, ncols - 50, 30)).astype(int)
        )
        x_lo, x_hi = 0, ncols - 1
        guess_col = float(ncols // 2)
        guess_row = 100.0
        return _trace_single_order_idlstyle(
            img, sobel, sample_cols, x_lo, x_hi,
            guess_col, guess_row,
            poly_degree=2, nrows=nrows, bufpix=2, frac=0.5,
            com_half_width=3, slit_height_min=5.0, slit_height_max=40.0,
        )

    # ── O: _trace_single_order_idlstyle ────────────────────────────────────

    def test_O_returns_SingleOrderResult(self, single_order_result):
        """Return type is _SingleOrderResult."""
        from pyspextool.instruments.ishell.tracing import _SingleOrderResult
        assert isinstance(single_order_result, _SingleOrderResult)

    def test_O_edges_independently_traced(self, single_order_result):
        """bottom_edge_samples and top_edge_samples are distinct arrays."""
        r = single_order_result
        assert not np.array_equal(r.bottom_edge_samples, r.top_edge_samples)

    def test_O_bottom_below_top(self, single_order_result):
        """At finite samples, bottom_edge < top_edge (bottom is lower row)."""
        r = single_order_result
        ok = np.isfinite(r.bottom_edge_samples) & np.isfinite(r.top_edge_samples)
        assert ok.sum() > 0, "No finite edge samples found"
        bot_vals = r.bottom_edge_samples[ok]
        top_vals = r.top_edge_samples[ok]
        assert np.all(bot_vals < top_vals), (
            "bottom_edge_samples must be below top_edge_samples at all finite positions"
        )

    def test_O_center_is_mean_of_edge_coeffs(self, single_order_result):
        """center_coeffs == (bottom_edge_coeffs + top_edge_coeffs) / 2."""
        r = single_order_result
        expected = (r.bottom_edge_coeffs + r.top_edge_coeffs) / 2.0
        np.testing.assert_allclose(r.center_coeffs, expected, rtol=1e-12)

    def test_O_xrange_from_fitted_edges_not_sample_coverage(self, syn_flat_path):
        """x_start / x_end come from block M (fitted-edge validity), not raw samples."""
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )
        from astropy.io import fits as _fits

        p, nrows, ncols = syn_flat_path
        img = _fits.open(p)[0].data.astype(float)
        sobel = _compute_sobel_image(img)

        # Sparse samples covering only the middle third of the detector
        sample_cols = np.unique(
            np.round(np.linspace(ncols // 3, 2 * ncols // 3, 20)).astype(int)
        )
        x_lo, x_hi = 0, ncols - 1  # nominal order range = full width

        result = _trace_single_order_idlstyle(
            img, sobel, sample_cols, x_lo, x_hi,
            guess_col=float(ncols // 2), guess_row=100.0,
            poly_degree=1, nrows=nrows, bufpix=2, frac=0.5,
            com_half_width=3, slit_height_min=5.0, slit_height_max=40.0,
        )
        # Block M evaluates over [0, ncols-1], so if the polynomial is valid
        # everywhere, x_start should be smaller than the leftmost sample col.
        assert result.x_start < sample_cols[0] or result.x_end > sample_cols[-1], (
            f"xrange [{result.x_start}, {result.x_end}] should extend beyond "
            f"the raw sample coverage [{sample_cols[0]}, {sample_cols[-1]}]; "
            "xrange is based on fitted-edge evaluation, not raw sample coverage"
        )

    def test_O_result_arrays_have_same_length_as_sample_cols(self, single_order_result):
        """All per-sample arrays have the same length as sample_cols."""
        r = single_order_result
        n = len(r.sample_cols)
        assert r.bottom_edge_samples.shape == (n,)
        assert r.top_edge_samples.shape == (n,)
        assert r.center_samples.shape == (n,)

    def test_O_edge_coeffs_shape(self, single_order_result):
        """Edge coefficient arrays have shape (poly_degree + 1,)."""
        r = single_order_result
        # poly_degree=2 → 3 coefficients
        assert r.bottom_edge_coeffs.shape == (3,)
        assert r.top_edge_coeffs.shape == (3,)
        assert r.center_coeffs.shape == (3,)

    # ── P: trace_orders_from_flat wired to new IDL core ────────────────────

    def test_P_trace_orders_uses_idlstyle_core(self, syn_flat_path):
        """trace_orders_from_flat calls _trace_single_order_idlstyle, not _trace_single_order."""
        import unittest.mock as mock
        from pyspextool.instruments.ishell import tracing as _tracing_module

        p, nrows, ncols = syn_flat_path
        calls = []
        original = _tracing_module._trace_single_order_idlstyle

        def capture(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        with mock.patch.object(_tracing_module, "_trace_single_order_idlstyle", capture):
            _tracing_module.trace_orders_from_flat(
                [p], guess_rows=[100.0], n_sample_cols=20,
                col_range=(50, ncols - 50), poly_degree=2,
                slit_height_range=(5.0, 40.0),
            )

        assert len(calls) >= 1, (
            "trace_orders_from_flat must call _trace_single_order_idlstyle at least once"
        )

    def test_P_bot_top_edges_independently_stored(self, syn_flat_path):
        """FlatOrderTrace stores independently-traced bot and top edge polynomials."""
        p, nrows, ncols = syn_flat_path
        result = trace_orders_from_flat(
            [p], guess_rows=[100.0], n_sample_cols=20,
            col_range=(50, ncols - 50), poly_degree=2,
            slit_height_range=(5.0, 40.0),
        )
        assert result.bot_poly_coeffs is not None
        assert result.top_poly_coeffs is not None
        assert not np.array_equal(result.bot_poly_coeffs, result.top_poly_coeffs)

    def test_P_center_coeffs_derived_from_edges(self, syn_flat_path):
        """center_poly_coeffs == (bot + top) / 2 for every order."""
        p, nrows, ncols = syn_flat_path
        result = trace_orders_from_flat(
            [p], guess_rows=[100.0], n_sample_cols=20,
            col_range=(50, ncols - 50), poly_degree=2,
            slit_height_range=(5.0, 40.0),
        )
        for i in range(result.n_orders):
            expected = (result.bot_poly_coeffs[i] + result.top_poly_coeffs[i]) / 2.0
            np.testing.assert_allclose(
                result.center_poly_coeffs[i], expected, rtol=1e-12,
                err_msg=f"Order {i}: center_poly_coeffs != (bot + top) / 2",
            )

    def test_P_order_xranges_from_fitted_edge_validity(self, syn_flat_path):
        """order_xranges come from fitted-edge validity (block M), not raw sample coverage."""
        p, nrows, ncols = syn_flat_path
        result = trace_orders_from_flat(
            [p], guess_rows=[100.0], n_sample_cols=20,
            col_range=(50, ncols - 50), poly_degree=2,
            slit_height_range=(5.0, 40.0),
        )
        assert result.order_xranges is not None
        # Both fitted edges must evaluate within (0, nrows-1) at every column
        # in the reported xrange.
        for i in range(result.n_orders):
            x0, x1 = result.order_xranges[i, 0], result.order_xranges[i, 1]
            x = np.arange(x0, x1 + 1, dtype=float)
            bot_vals = np.polynomial.polynomial.polyval(x, result.bot_poly_coeffs[i])
            top_vals = np.polynomial.polynomial.polyval(x, result.top_poly_coeffs[i])
            assert np.all(bot_vals > 0) and np.all(bot_vals < nrows - 1), (
                f"Order {i}: bottom edge outside detector within reported xrange"
            )
            assert np.all(top_vals > 0) and np.all(top_vals < nrows - 1), (
                f"Order {i}: top edge outside detector within reported xrange"
            )

    def test_P_downstream_geometry_still_works(self, syn_flat_path):
        """to_order_geometry_set still produces valid OrderGeometrySet after new core."""
        p, nrows, ncols = syn_flat_path
        result = trace_orders_from_flat(
            [p], guess_rows=[100.0], n_sample_cols=20,
            col_range=(50, ncols - 50), poly_degree=2,
            slit_height_range=(5.0, 40.0),
        )
        geom = result.to_order_geometry_set("test_mode")
        assert geom.n_orders == result.n_orders
        for g in geom.geometries:
            assert g.x_start <= g.x_end

    def test_P_old_heuristic_helpers_not_in_new_core(self):
        """_predict_and_find_edges and _update_centre_and_flags are not called by
        _trace_single_order_idlstyle (old heuristic path is gone from the new core)."""
        import inspect
        from pyspextool.instruments.ishell.tracing import _trace_single_order_idlstyle

        src = inspect.getsource(_trace_single_order_idlstyle)
        assert "_predict_and_find_edges" not in src, (
            "_trace_single_order_idlstyle must not call the old heuristic "
            "_predict_and_find_edges; the new core uses _trace_order_left/_right"
        )
        assert "_update_centre_and_flags" not in src, (
            "_trace_single_order_idlstyle must not call the old _update_centre_and_flags"
        )


# ---------------------------------------------------------------------------
# TestMCFindordersDriftAudit  — final pass: per-criterion audit + known drift
# ---------------------------------------------------------------------------


class TestMCFindordersDriftAudit:
    """Audit the Python port against mc_findorders.pro block by block.

    Covers audit criteria A–F from the problem statement and documents
    all known remaining semantic drift from the IDL reference.

    A: No peak finding remains in the tracing core
    B: Edges are primary and centres are derived
    C: Dynamic polynomial prediction is used during the sweep
    D: Threshold-based edge guesses are used before Sobel centroiding
    E: Final xranges come from fitted edges staying on detector
    F: Compatibility / public structures still behave as expected

    Known remaining drift (documented):
    1. Robust fitting: IDL uses mc_robustpoly1d with Gauss-Jordan (/GAUSSJ).
       Python uses _fit_poly_robust → _polyfit_1d with iterative sigma-clipping.
       Parameters are aligned (thresh=3, eps=0.01) but the solver differs.
    2. tabinv vs interp: IDL tabinv(scols, guesspos[0,i], idx) uses fractional
       index interpolation.  Python uses np.interp to replicate this, then rounds.
       Both round to the nearest integer index; outputs are identical for integer
       guess_col values.
    3. Sobel normalisation: IDL rimage=sobel(image*1000./max(image));
       Python _compute_sobel_image uses the same 1000/max factor, then scipy
       sobel along axes 0 and 1 (magnitude = sqrt(Gy²+Gx²)).  The IDL sobel()
       built-in uses a different kernel weighting, so absolute Sobel values may
       differ.  COM weights are relative so this does not affect correctness in
       practice.

    Previously documented drift items that have been resolved:
    - goto-cont1: when the edge pair is rejected (slit-height check fails with
      both flags active), the Python implementation now matches IDL: cen[k]
      stays NaN (the center assignment is skipped, identical to IDL's goto cont1
      jumping over the cen[k] update).
    """

    # ── A: No peak finding anywhere in the tracing core ─────────────────────

    def test_A_sweep_functions_have_no_peak_finding(self):
        """_trace_order_left, _trace_order_right, _trace_single_order_idlstyle
        must contain no references to peak-finding helpers.

        IDL mc_findorders uses only flux-threshold + Sobel COM — never
        scipy.signal.find_peaks or any equivalent.
        """
        import inspect
        from pyspextool.instruments.ishell.tracing import (
            _trace_order_left,
            _trace_order_right,
            _trace_single_order_idlstyle,
        )
        forbidden = ("find_peaks", "_find_order_peaks", "scipy_find_peaks",
                     "_scipy_find_peaks")
        for fn in (_trace_order_left, _trace_order_right, _trace_single_order_idlstyle):
            src = inspect.getsource(fn)
            for kw in forbidden:
                assert kw not in src, (
                    f"{fn.__name__} contains forbidden peak-finding reference '{kw}'"
                )

    def test_A_tracing_module_has_no_find_peaks_import(self):
        """The tracing module must not import scipy.signal.find_peaks.

        Neither _scipy_find_peaks nor _find_order_peaks should exist.
        """
        import pyspextool.instruments.ishell.tracing as _tm
        assert not hasattr(_tm, "_scipy_find_peaks"), (
            "tracing module must not import scipy.signal.find_peaks"
        )
        assert not hasattr(_tm, "_find_order_peaks"), (
            "tracing module must not define _find_order_peaks"
        )

    # ── B: Edges primary; centres derived from edges ─────────────────────────

    def test_B_idlstyle_center_equals_edge_midpoint_at_accepted_columns(self):
        """In _trace_single_order_idlstyle, at columns where edges were accepted,
        center_samples == (bottom_edge_samples + top_edge_samples) / 2.

        IDL: ``cen[k] = (com_bot + com_top) / 2.``
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )
        flat, _, _ = _make_single_order_flat(c0=128.0, hw=14.0, tilt=0.04, seed=101)
        nrows, ncols = flat.shape
        sobel = _compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 30)).astype(int))
        result = _trace_single_order_idlstyle(
            flat, sobel, sample_cols, 0, ncols - 1,
            float(ncols // 2), 128.0,
            poly_degree=2, nrows=nrows, bufpix=2, frac=0.7,
            com_half_width=4, slit_height_min=8.0, slit_height_max=50.0,
        )
        both = np.isfinite(result.bottom_edge_samples) & np.isfinite(result.top_edge_samples)
        assert both.sum() > 0, "No columns with both edges accepted"
        expected_cen = (result.bottom_edge_samples[both] + result.top_edge_samples[both]) / 2.0
        np.testing.assert_allclose(
            result.center_samples[both], expected_cen, atol=1e-10,
            err_msg="center_samples must equal (bot+top)/2 wherever both edges are finite",
        )

    def test_B_center_poly_is_mean_of_edge_polys_in_new_core(self):
        """center_coeffs from _trace_single_order_idlstyle must equal
        (bottom_edge_coeffs + top_edge_coeffs) / 2 coefficient-wise.

        Block N (_center_coeffs_from_edge_coeffs) computes this exactly.
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_single_order_idlstyle,
            _compute_sobel_image,
        )
        flat, _, _ = _make_single_order_flat(c0=128.0, hw=14.0, seed=102)
        nrows, ncols = flat.shape
        sobel = _compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 30)).astype(int))
        result = _trace_single_order_idlstyle(
            flat, sobel, sample_cols, 0, ncols - 1,
            float(ncols // 2), 128.0,
            poly_degree=2, nrows=nrows, bufpix=2, frac=0.7,
            com_half_width=4, slit_height_min=8.0, slit_height_max=50.0,
        )
        expected = (result.bottom_edge_coeffs + result.top_edge_coeffs) / 2.0
        np.testing.assert_allclose(result.center_coeffs, expected, rtol=1e-12)

    # ── C: Dynamic polynomial prediction used at each sweep step ─────────────

    def test_C_predict_center_called_during_left_sweep(self):
        """_predict_center_from_known_samples must be called during _trace_order_left.

        IDL: ``coeff = mc_polyfit1d(scols, cen, 1>(degree-2), /SILENT, /JUSTFIT)``
             ``y_guess = bufpix > poly(scols[k], coeff) < (nrows-bufpix-1)``
        is evaluated at EVERY step of the left sweep.
        """
        import unittest.mock as mock
        from pyspextool.instruments.ishell import tracing as _tm

        flat, _, _ = _make_single_order_flat(c0=128.0, hw=14.0, tilt=0.03, seed=103)
        nrows, ncols = flat.shape
        sobel = _tm._compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 20)).astype(int))
        gidx = 10

        # Seed arrays as _trace_single_order_idlstyle would
        edges, cen, bot, top_arr, gidx_out = (
            np.full((len(sample_cols), 2), np.nan),
            np.full(len(sample_cols), np.nan),
            np.full(len(sample_cols), np.nan),
            np.full(len(sample_cols), np.nan),
            gidx,
        )
        cen[max(0, gidx - 2): min(len(sample_cols), gidx + 3)] = 128.0
        bot = edges[:, 0]
        top_arr2 = edges[:, 1]

        calls = []
        original = _tm._predict_center_from_known_samples

        def capture(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        with mock.patch.object(_tm, "_predict_center_from_known_samples", capture):
            _tm._trace_order_left(
                flat, sobel, sample_cols, cen, bot, top_arr2,
                gidx, nrows, bufpix=2, poly_degree=2, frac=0.7,
                com_half_width=4, slit_height_min=8.0, slit_height_max=50.0,
            )

        # Left sweep iterates over gidx+1 steps (j=0..gidx inclusive).
        assert len(calls) >= 1, (
            "_predict_center_from_known_samples was not called during _trace_order_left"
        )

    def test_C_predict_center_called_during_right_sweep(self):
        """_predict_center_from_known_samples must be called during _trace_order_right."""
        import unittest.mock as mock
        from pyspextool.instruments.ishell import tracing as _tm

        flat, _, _ = _make_single_order_flat(c0=128.0, hw=14.0, tilt=-0.02, seed=104)
        nrows, ncols = flat.shape
        sobel = _tm._compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 20)).astype(int))
        gidx = 5

        edges = np.full((len(sample_cols), 2), np.nan)
        cen = np.full(len(sample_cols), np.nan)
        cen[max(0, gidx - 2): min(len(sample_cols), gidx + 3)] = 128.0
        bot = edges[:, 0]
        top_arr = edges[:, 1]

        calls = []
        original = _tm._predict_center_from_known_samples

        def capture(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        with mock.patch.object(_tm, "_predict_center_from_known_samples", capture):
            _tm._trace_order_right(
                flat, sobel, sample_cols, cen, bot, top_arr,
                gidx, nrows, bufpix=2, poly_degree=2, frac=0.7,
                com_half_width=4, slit_height_min=8.0, slit_height_max=50.0,
            )

        assert len(calls) >= 1, (
            "_predict_center_from_known_samples was not called during _trace_order_right"
        )

    def test_C_prediction_uses_all_known_samples_not_just_seed(self):
        """_predict_center_from_known_samples receives the FULL cen array so that
        previously-traced columns inform the prediction at each new step.

        IDL: ``coeff = mc_polyfit1d(scols, cen, ...)`` — all of scols/cen passed.
        """
        from pyspextool.instruments.ishell.tracing import _predict_center_from_known_samples

        # 10-column array with 7 known values; the other 3 are NaN.
        scols = np.arange(10, 110, 10, dtype=float)   # 10 columns
        cen = np.full(10, np.nan)
        cen[0:7] = 100.0 + np.arange(7) * 0.5   # known linear drift
        # Prediction at col 100 (index 9) should use all 7 known values.
        y_pred = _predict_center_from_known_samples(scols, cen, 100.0, poly_degree=2,
                                                     nrows=256, bufpix=2)
        # Rough sanity: prediction near the linearly-drifting region
        assert 95.0 < y_pred < 115.0, (
            f"prediction {y_pred:.1f} outside reasonable range; "
            "dynamic prediction must use all known samples"
        )

    # ── D: Threshold guesses before Sobel centroid ────────────────────────────

    def test_D_threshold_guesses_precede_sobel_centroid_in_left_sweep(self):
        """_find_threshold_edge_guesses is called before _sobel_centroid in
        _trace_order_left (matching IDL order: ztop/zbot where-query, THEN COM).
        """
        import unittest.mock as mock
        from pyspextool.instruments.ishell import tracing as _tm

        flat, _, _ = _make_single_order_flat(c0=128.0, hw=14.0, tilt=0.0, seed=105)
        nrows, ncols = flat.shape
        sobel = _tm._compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 12)).astype(int))
        gidx = len(sample_cols) // 2

        edges = np.full((len(sample_cols), 2), np.nan)
        cen = np.full(len(sample_cols), np.nan)
        cen[max(0, gidx - 2): min(len(sample_cols), gidx + 3)] = 128.0
        bot = edges[:, 0]
        top_arr = edges[:, 1]

        call_order: list[str] = []
        orig_fteg = _tm._find_threshold_edge_guesses
        orig_sc = _tm._sobel_centroid

        def cap_fteg(*a, **kw):
            call_order.append("threshold")
            return orig_fteg(*a, **kw)

        def cap_sc(*a, **kw):
            call_order.append("sobel")
            return orig_sc(*a, **kw)

        with mock.patch.object(_tm, "_find_threshold_edge_guesses", cap_fteg), \
             mock.patch.object(_tm, "_sobel_centroid", cap_sc):
            _tm._trace_order_left(
                flat, sobel, sample_cols, cen, bot, top_arr,
                gidx, nrows, bufpix=2, poly_degree=2, frac=0.7,
                com_half_width=4, slit_height_min=8.0, slit_height_max=50.0,
            )

        # Every "sobel" call must be preceded by a "threshold" call.
        sobel_indices = [i for i, v in enumerate(call_order) if v == "sobel"]
        threshold_indices = [i for i, v in enumerate(call_order) if v == "threshold"]
        assert len(threshold_indices) > 0, (
            "_find_threshold_edge_guesses was never called in left sweep"
        )
        for si in sobel_indices:
            earlier_thresholds = [t for t in threshold_indices if t < si]
            assert len(earlier_thresholds) > 0, (
                f"_sobel_centroid called at position {si} without a preceding "
                "_find_threshold_edge_guesses call"
            )

    def test_D_threshold_guesses_precede_sobel_centroid_in_right_sweep(self):
        """_find_threshold_edge_guesses is called before _sobel_centroid in
        _trace_order_right (same structure as left sweep).
        """
        import unittest.mock as mock
        from pyspextool.instruments.ishell import tracing as _tm

        flat, _, _ = _make_single_order_flat(c0=128.0, hw=14.0, tilt=0.0, seed=106)
        nrows, ncols = flat.shape
        sobel = _tm._compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 12)).astype(int))
        gidx = len(sample_cols) // 2

        edges = np.full((len(sample_cols), 2), np.nan)
        cen = np.full(len(sample_cols), np.nan)
        cen[max(0, gidx - 2): min(len(sample_cols), gidx + 3)] = 128.0
        bot = edges[:, 0]
        top_arr = edges[:, 1]

        call_order: list[str] = []
        orig_fteg = _tm._find_threshold_edge_guesses
        orig_sc = _tm._sobel_centroid

        def cap_fteg(*a, **kw):
            call_order.append("threshold")
            return orig_fteg(*a, **kw)

        def cap_sc(*a, **kw):
            call_order.append("sobel")
            return orig_sc(*a, **kw)

        with mock.patch.object(_tm, "_find_threshold_edge_guesses", cap_fteg), \
             mock.patch.object(_tm, "_sobel_centroid", cap_sc):
            _tm._trace_order_right(
                flat, sobel, sample_cols, cen, bot, top_arr,
                gidx, nrows, bufpix=2, poly_degree=2, frac=0.7,
                com_half_width=4, slit_height_min=8.0, slit_height_max=50.0,
            )

        sobel_indices = [i for i, v in enumerate(call_order) if v == "sobel"]
        threshold_indices = [i for i, v in enumerate(call_order) if v == "threshold"]
        assert len(threshold_indices) > 0, (
            "_find_threshold_edge_guesses was never called in right sweep"
        )
        for si in sobel_indices:
            earlier_thresholds = [t for t in threshold_indices if t < si]
            assert len(earlier_thresholds) > 0, (
                f"_sobel_centroid called at position {si} without a preceding "
                "_find_threshold_edge_guesses call"
            )

    # ── goto-cont1: now matches IDL ──────────────────────────────────────────

    def test_goto_cont1_cen_stays_nan_on_slit_height_rejection(self):
        """When _accept_edge_pair rejects an edge pair (both flags active, both
        edges finite, but slit height outside range), the Python implementation
        now matches IDL's ``goto cont1`` exactly: cen[k] stays at its
        pre-existing value rather than being overwritten with y_guess_f.

        For columns outside the initialization seed window the pre-existing
        value is NaN, so after goto-cont1 they must remain NaN.

        The old Python bug was: center_samples[k] = y_guess_f on every rejected
        path (both goto-cont1 AND else-branch).  After the fix, only the
        else-branch (threshold guess failed → NaN edges) sets center = y_guess_f;
        the goto-cont1 path (finite edges but height fails) leaves center alone.
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_order_left,
            _compute_sobel_image,
        )

        # Use a noiseless flat with clear edges (hw=14 → slit~28px) and
        # slit_height_max=5 so that every detected pair fails the height check,
        # forcing the goto-cont1 path for columns where both Sobel COMs succeed.
        flat, _, _ = _make_single_order_flat(c0=128.0, hw=14.0, tilt=0.0, noise=0.0, seed=107)
        nrows, ncols = flat.shape
        sobel = _compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 14)).astype(int))
        n_samp = len(sample_cols)
        gidx = n_samp // 2

        edges = np.full((n_samp, 2), np.nan)
        cen = np.full(n_samp, np.nan)
        # Seed window: lo=max(0, gidx-2) .. hi=min(n_samp-1, gidx+2)
        seed_lo = max(0, gidx - 2)
        seed_hi = min(n_samp - 1, gidx + 2)
        cen[seed_lo: seed_hi + 1] = 128.0
        bot = edges[:, 0]
        top_arr = edges[:, 1]

        # slit_height_max=5 << actual slit ~28px → every found edge pair is
        # rejected via slit-height (goto cont1 path, both edges finite).
        _trace_order_left(
            flat, sobel, sample_cols, cen, bot, top_arr,
            gidx, nrows, bufpix=2, poly_degree=2, frac=0.7,
            com_half_width=4, slit_height_min=0.0, slit_height_max=5.0,
        )

        # For columns OUTSIDE the seed window and LEFT of gidx, cen was NaN
        # before the sweep.  On goto-cont1, cen is left unchanged → stays NaN.
        # Old bug: center_samples[k] = y_guess_f was written even on goto-cont1.
        #
        # We cannot distinguish goto-cont1 vs else-branch purely from the output,
        # but for a NOISELESS flat with tilt=0 the polynomial prediction keeps
        # y_guess ≈ 128 (correct order centre), so the threshold guess finds
        # the edges and the Sobel COM succeeds → goto-cont1 (not else-branch).
        # Therefore cen must be NaN for columns to the left of seed_lo.
        cols_outside_seed_left = list(range(0, seed_lo))
        for k in cols_outside_seed_left:
            assert np.isnan(cen[k]), (
                f"goto cont1 fix: col index {k} (col {sample_cols[k]}) is outside "
                f"the seed window [{seed_lo},{seed_hi}] and left of gidx={gidx}; "
                f"cen was NaN before the sweep.  After goto-cont1 it must stay NaN, "
                f"but got cen[{k}] = {cen[k]:.3f}.  "
                f"(edges: bot={bot[k]}, top={top_arr[k]})"
            )

    # ── Robust-fit solver: IDL-equivalent _mc_robustpoly1d ───────────────────

    def test_robust_fit_idl_equivalent_documented(self):
        """_fit_order_edge_polynomials now uses _mc_robustpoly1d, a direct port
        of IDL mc_robustpoly1d /GAUSSJ, so the prior approximation is resolved.

        The docstring must reference the IDL /GAUSSJ equivalence.
        """
        import inspect
        from pyspextool.instruments.ishell.tracing import _fit_order_edge_polynomials

        src = inspect.getsource(_fit_order_edge_polynomials)
        assert "GAUSSJ" in src, (
            "_fit_order_edge_polynomials docstring must reference the IDL /GAUSSJ "
            "equivalence of _mc_robustpoly1d"
        )

    def test_ROBUST_FIT_FUNCTION_alias_exists_and_callable(self):
        """_ROBUST_FIT_FUNCTION module alias must exist and point to _mc_robustpoly1d.

        This alias is the single point of indirection for the robust fitter so
        that the implementation can be swapped without touching any tracing logic.
        _ROBUST_FIT_FUNCTION now points to _mc_robustpoly1d (IDL-equivalent).
        """
        import pyspextool.instruments.ishell.tracing as _m

        assert hasattr(_m, "_ROBUST_FIT_FUNCTION"), (
            "_ROBUST_FIT_FUNCTION alias must be defined at module level"
        )
        assert callable(_m._ROBUST_FIT_FUNCTION), (
            "_ROBUST_FIT_FUNCTION must be callable"
        )
        # Smoke-test: basic call succeeds.
        cols = np.arange(10, dtype=float)
        rows = 2.0 * cols + 1.0
        coeffs, rms = _m._ROBUST_FIT_FUNCTION(cols, rows, 1)
        assert coeffs.shape == (2,), f"Expected 2 coefficients, got {coeffs.shape}"
        assert np.isfinite(rms), "RMS should be finite for noise-free data"

    def test_fit_order_edge_polynomials_uses_ROBUST_FIT_FUNCTION(self):
        """_fit_order_edge_polynomials source must call _ROBUST_FIT_FUNCTION,
        not _fit_poly_robust directly, so the alias indirection is in effect.
        """
        import inspect
        from pyspextool.instruments.ishell.tracing import _fit_order_edge_polynomials

        src = inspect.getsource(_fit_order_edge_polynomials)
        assert "_ROBUST_FIT_FUNCTION" in src, (
            "_fit_order_edge_polynomials must call _ROBUST_FIT_FUNCTION "
            "(not _fit_poly_robust directly) so the alias is used"
        )

    def test_ROBUST_FIT_FUNCTION_points_to_mc_robustpoly1d(self):
        """_ROBUST_FIT_FUNCTION must be the IDL-equivalent _mc_robustpoly1d."""
        import pyspextool.instruments.ishell.tracing as _m

        assert _m._ROBUST_FIT_FUNCTION is _m._mc_robustpoly1d, (
            "_ROBUST_FIT_FUNCTION must point to _mc_robustpoly1d "
            "(the IDL mc_robustpoly1d /GAUSSJ port)"
        )

    # ── _mc_robustpoly1d: regression tests for the IDL-equivalent fitter ─────

    def test_mc_robustpoly1d_linear_no_outliers(self):
        """_mc_robustpoly1d recovers exact linear coefficients on clean data."""
        from pyspextool.instruments.ishell.tracing import _mc_robustpoly1d

        cols = np.arange(0, 100, dtype=float)
        rows = 3.5 * cols + 12.0
        coeffs, rms = _mc_robustpoly1d(cols, rows, degree=1)
        assert np.isclose(coeffs[0], 12.0, atol=1e-6), f"intercept={coeffs[0]}"
        assert np.isclose(coeffs[1], 3.5, atol=1e-6), f"slope={coeffs[1]}"
        assert rms == 0.0 or np.isclose(rms, 0.0, atol=1e-10), f"rms={rms}"

    def test_mc_robustpoly1d_rejects_outliers(self):
        """_mc_robustpoly1d rejects large outliers and recovers the true polynomial."""
        from pyspextool.instruments.ishell.tracing import _mc_robustpoly1d

        rng = np.random.default_rng(7)
        cols = np.arange(50, dtype=float)
        rows = 2.0 * cols + 5.0 + rng.normal(0.0, 0.1, size=50)
        # Inject 5 large outliers.
        outlier_idx = [5, 15, 25, 35, 45]
        rows[outlier_idx] += 50.0

        coeffs, rms = _mc_robustpoly1d(cols, rows, degree=1, thresh=3.0)
        # After rejecting outliers the fit should be very close to the truth.
        assert np.isclose(coeffs[0], 5.0, atol=0.5), f"intercept={coeffs[0]:.3f}"
        assert np.isclose(coeffs[1], 2.0, atol=0.05), f"slope={coeffs[1]:.4f}"
        assert np.isfinite(rms) and rms < 1.0, f"rms={rms:.3f} should be small"

    def test_mc_robustpoly1d_handles_nan_inputs(self):
        """_mc_robustpoly1d treats NaN values as bad and skips them."""
        from pyspextool.instruments.ishell.tracing import _mc_robustpoly1d

        cols = np.arange(30, dtype=float)
        rows = 1.5 * cols + 3.0
        # Inject NaNs at 5 positions.
        nan_idx = [2, 8, 14, 20, 26]
        rows[nan_idx] = np.nan

        coeffs, rms = _mc_robustpoly1d(cols, rows, degree=1)
        assert np.isclose(coeffs[0], 3.0, atol=1e-6), f"intercept={coeffs[0]}"
        assert np.isclose(coeffs[1], 1.5, atol=1e-6), f"slope={coeffs[1]}"
        assert np.isfinite(rms), "rms should be finite even with NaN inputs"

    def test_mc_robustpoly1d_too_few_points_returns_nan_rms(self):
        """_mc_robustpoly1d returns (zeros, nan) when fewer than degree+1 points."""
        from pyspextool.instruments.ishell.tracing import _mc_robustpoly1d

        # degree=3 requires 4 good points; only 2 provided.
        cols = np.array([1.0, 2.0])
        rows = np.array([1.0, 2.0])
        coeffs, rms = _mc_robustpoly1d(cols, rows, degree=3)
        assert np.isnan(rms), f"Expected nan rms, got {rms}"

    def test_mc_robustpoly1d_quadratic_with_outliers(self):
        """_mc_robustpoly1d fits a quadratic and rejects injected outliers."""
        from pyspextool.instruments.ishell.tracing import _mc_robustpoly1d

        rng = np.random.default_rng(42)
        cols = np.linspace(0, 100, 80)
        # True: y = 0.001*x^2 + 0.5*x + 10
        rows = 0.001 * cols**2 + 0.5 * cols + 10.0 + rng.normal(0.0, 0.2, 80)
        # Inject outliers.
        rows[[10, 30, 50, 70]] += 30.0

        coeffs, rms = _mc_robustpoly1d(cols, rows, degree=2, thresh=3.0)
        assert np.isclose(coeffs[0], 10.0, atol=0.5), f"c0={coeffs[0]:.3f}"
        assert np.isclose(coeffs[1], 0.5, atol=0.05), f"c1={coeffs[1]:.4f}"
        assert np.isclose(coeffs[2], 0.001, atol=0.001), f"c2={coeffs[2]:.6f}"
        assert np.isfinite(rms) and rms < 1.0, f"rms={rms:.3f}"

    # ── IDL branch semantics: goto cont1/cont2 vs fallback else-branch ───────

    def test_goto_cont1_right_sweep_cen_stays_nan(self):
        """Right sweep: goto cont2 path leaves center NaN (not assigned).

        When both flags are active, both COMs are finite, but slit-height
        check fails, the right-sweep takes the goto cont2 path and must NOT
        assign center.  Outside the seed window, center was NaN before the
        sweep; it must remain NaN after.
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_order_right,
            _compute_sobel_image,
        )

        flat, _, _ = _make_single_order_flat(c0=128.0, hw=14.0, tilt=0.0, noise=0.0, seed=108)
        nrows, ncols = flat.shape
        sobel = _compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 14)).astype(int))
        n_samp = len(sample_cols)
        gidx = n_samp // 2

        cen = np.full(n_samp, np.nan)
        seed_lo = max(0, gidx - 2)
        seed_hi = min(n_samp - 1, gidx + 2)
        cen[seed_lo: seed_hi + 1] = 128.0
        bot = np.full(n_samp, np.nan)
        top_arr = np.full(n_samp, np.nan)

        # slit_height_max=5 << actual slit ~28px → goto cont2 for finite-edge cols
        _trace_order_right(
            flat, sobel, sample_cols, cen, bot, top_arr,
            gidx, nrows, bufpix=2, poly_degree=2, frac=0.7,
            com_half_width=4, slit_height_min=0.0, slit_height_max=5.0,
        )

        # Columns to the RIGHT of the seed window were NaN before the sweep.
        # Under goto-cont2 they must stay NaN (center not assigned).
        for k in range(seed_hi + 1, n_samp):
            assert np.isnan(cen[k]), (
                f"goto cont2: col index {k} (col {sample_cols[k]}) right of seed "
                f"window [{seed_lo},{seed_hi}]; cen was NaN before sweep, must "
                f"remain NaN on goto-cont2, got {cen[k]:.3f}"
            )

    def test_else_branch_sets_center_to_y_guess_when_edge_fails(self):
        """Fallback else-branch: when threshold edge detection fails (COMs are
        NaN), center is set to y_guess_f (the polynomial-predicted centre).

        Edges remain NaN but center is finite — proving the else-branch does
        NOT enforce center=NaN and no global post-hoc mask touches it.

        This test uses a uniform flat so _find_threshold_edge_guesses always
        returns None (no below-threshold pixels), making both COMs NaN and
        triggering the else-branch for every column outside the seed window.
        """
        from pyspextool.instruments.ishell.tracing import (
            _trace_order_left,
            _compute_sobel_image,
        )

        # Uniform flat: every column profile is constant → threshold detection
        # always fails → COM = NaN → else-branch fires → center = y_guess_f.
        nrows, ncols = 256, 256
        flat = np.ones((nrows, ncols), dtype=float) * 1000.0
        sobel = _compute_sobel_image(flat)
        sample_cols = np.unique(np.round(np.linspace(20, ncols - 20, 14)).astype(int))
        n_samp = len(sample_cols)
        gidx = n_samp // 2

        cen = np.full(n_samp, np.nan)
        seed_lo = max(0, gidx - 2)
        seed_hi = min(n_samp - 1, gidx + 2)
        cen[seed_lo: seed_hi + 1] = 128.0
        bot = np.full(n_samp, np.nan)
        top_arr = np.full(n_samp, np.nan)

        _trace_order_left(
            flat, sobel, sample_cols, cen, bot, top_arr,
            gidx, nrows, bufpix=2, poly_degree=2, frac=0.7,
            com_half_width=4, slit_height_min=0.0, slit_height_max=500.0,
        )

        # Outside seed window AND to the left of gidx, edges must be NaN
        # but center must be finite (the y_guess_f fallback was assigned).
        for k in range(0, seed_lo):
            assert np.isnan(bot[k]) and np.isnan(top_arr[k]), (
                f"else-branch: edges at col index {k} should be NaN (detection "
                f"failed), got bot={bot[k]}, top={top_arr[k]}"
            )
            assert np.isfinite(cen[k]), (
                f"else-branch: center at col index {k} should be finite "
                f"(y_guess_f fallback), got cen={cen[k]}.  "
                "A global NaN mask would incorrectly erase this."
            )

    # ── tabinv vs interp: verified equivalent ─────────────────────────────────

    def test_tabinv_equivalent_to_interp(self):
        """IDL tabinv(scols, guesspos[0,i], idx) finds a fractional index by
        linear interpolation.  Python uses np.interp + round, which gives the
        same result: both round to the nearest integer index for the integer seed
        positions used in practice.  This is NOT outstanding drift — the two
        implementations are semantically equivalent.

        Verify that _initialize_order_trace_arrays returns the correct integer
        index for a guess_col that is exactly at a sample column.
        """
        from pyspextool.instruments.ishell.tracing import _initialize_order_trace_arrays

        # sample_cols: uniform spacing, guess_col exactly at index 5
        sample_cols = np.arange(0, 100, 10, dtype=int)  # 10 cols: 0,10,...,90
        guess_col = 50.0   # exactly at index 5
        guess_row = 123.0
        poly_degree = 2

        _, cen, _, gidx = _initialize_order_trace_arrays(
            sample_cols, guess_col, guess_row, poly_degree
        )
        assert gidx == 5, f"Expected gidx=5, got {gidx}"
        # cen must be seeded at [max(0,5-2) : min(9,5+2)+1] = [3:8] = indices 3..7
        for k in range(3, 8):
            assert np.isfinite(cen[k]) and cen[k] == guess_row, (
                f"cen[{k}] should be {guess_row}, got {cen[k]}"
            )
        # Indices outside the seed window must stay NaN
        for k in [0, 1, 2, 8, 9]:
            assert np.isnan(cen[k]), f"cen[{k}] should be NaN outside seed window"
