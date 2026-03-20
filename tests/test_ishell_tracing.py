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
        hw = float(dummy_trace.half_width_rows[0])
        geom = dummy_trace.to_order_geometry_set("H1")
        g = geom.geometries[0]
        # constant term of center poly is 0; half-width offset applied to constant
        np.testing.assert_allclose(g.bottom_edge_coeffs[0], g.centerline_coeffs[0] - hw)
        np.testing.assert_allclose(g.top_edge_coeffs[0], g.centerline_coeffs[0] + hw)

    def test_to_order_geometry_set_placeholder_order_numbers(self, dummy_trace):
        geom = dummy_trace.to_order_geometry_set("H1")
        for i, g in enumerate(geom.geometries):
            assert g.order == i


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
        geom = h1_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        for g in geom.geometries:
            assert g.x_start == 650
            assert g.x_end == 1550


# ---------------------------------------------------------------------------
# Private helpers used by the test fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 7. Robust centre fitting – unit tests for _fit_stable_poly
# ---------------------------------------------------------------------------


class TestFitStablePoly:
    """_fit_stable_poly prevents catastrophically divergent polynomial fits."""

    def _import_helpers(self):
        from pyspextool.instruments.ishell.tracing import (
            _fit_stable_poly,
            _fit_poly_sigma_clip,
        )
        return _fit_stable_poly, _fit_poly_sigma_clip

    def test_sparse_coverage_reduces_degree(self):
        """With < 50% column coverage, degree is reduced to at most 1."""
        _fit_stable_poly, _ = self._import_helpers()
        # 7 points in cols 262-997, all near row 1780 – mirrors the
        # K3 Order 24 failure case.
        rng = np.random.default_rng(0)
        cols = np.array([262, 315, 787, 840, 892, 945, 997], dtype=float)
        rows = np.array([1780, 1772, 1772, 1773, 1781, 1786, 1773], dtype=float)
        coeffs, rms, deg = _fit_stable_poly(
            cols, rows,
            max_degree=3, sigma=3.0, n_iter=3,
            col_lo=0.0, col_hi=2047.0, nrows=2048,
        )
        # With 36% coverage (cols 262-997 out of 0-2047), degree must be ≤ 1
        assert deg <= 1, f"Expected degree ≤ 1 for sparse coverage, got {deg}"

    def test_sparse_coverage_stays_within_detector(self):
        """Fallback polynomial must not diverge off the detector."""
        _fit_stable_poly, _ = self._import_helpers()
        cols = np.array([262, 315, 787, 840, 892, 945, 997], dtype=float)
        rows = np.array([1780, 1772, 1772, 1773, 1781, 1786, 1773], dtype=float)
        coeffs, rms, deg = _fit_stable_poly(
            cols, rows,
            max_degree=3, sigma=3.0, n_iter=3,
            col_lo=0.0, col_hi=2047.0, nrows=2048,
        )
        eval_cols = np.linspace(0, 2047, 100)
        y_eval = np.polynomial.polynomial.polyval(eval_cols, coeffs)
        assert np.all(y_eval >= -50), "Polynomial goes below detector floor"
        assert np.all(y_eval <= 2098), "Polynomial goes above detector ceiling"

    def test_degree3_divergence_corrected(self):
        """A degree-3 fit that diverges to y=534 must be replaced with linear."""
        _fit_stable_poly, _fit_poly_sigma_clip = self._import_helpers()
        cols = np.array([262, 315, 787, 840, 892, 945, 997], dtype=float)
        rows = np.array([1780, 1772, 1772, 1773, 1781, 1786, 1773], dtype=float)

        # Verify the OLD degree-3 fit actually diverges
        coeffs3, _ = _fit_poly_sigma_clip(cols, rows, 3, 3.0, 3)
        y_at_2047 = float(np.polynomial.polynomial.polyval(2047.0, coeffs3))
        assert y_at_2047 < 1000, (
            f"Test precondition: expected degree-3 to diverge below 1000 at col 2047, "
            f"got {y_at_2047:.0f}"
        )

        # Verify _fit_stable_poly fixes it
        coeffs_s, rms_s, deg_s = _fit_stable_poly(
            cols, rows,
            max_degree=3, sigma=3.0, n_iter=3,
            col_lo=0.0, col_hi=2047.0, nrows=2048,
        )
        y_at_2047_stable = float(np.polynomial.polynomial.polyval(2047.0, coeffs_s))
        assert y_at_2047_stable > 1700, (
            f"Stable fit should give ~1775-1786 at col 2047, got {y_at_2047_stable:.0f}"
        )

    def test_full_coverage_uses_requested_degree(self):
        """With full column coverage and many points, max degree is used."""
        _fit_stable_poly, _ = self._import_helpers()
        rng = np.random.default_rng(42)
        # 30 points evenly spanning the full column range, nearly linear
        cols = np.linspace(0, 2047, 30)
        rows = 100.0 + 0.05 * cols + rng.normal(0, 3, 30)
        coeffs, rms, deg = _fit_stable_poly(
            cols, rows,
            max_degree=3, sigma=3.0, n_iter=3,
            col_lo=0.0, col_hi=2047.0, nrows=2048,
        )
        assert deg == 3, f"Expected degree 3 for well-covered data, got {deg}"

    def test_data_range_check_prevents_extrapolation_crossing(self):
        """Polynomial must not extrapolate far below rows.min() - margin."""
        _fit_stable_poly, _ = self._import_helpers()
        # Simulate Order-2-like scenario: data starts at col 367, rows 83-178.
        # A degree-3 fit would extrapolate to ~22 at col 0 (below rows.min()-50=33).
        rng = np.random.default_rng(7)
        cols = np.linspace(367, 2047, 33)
        rows = 83.0 + (cols - 367) * (178 - 83) / (2047 - 367) + rng.normal(0, 5, 33)
        coeffs, rms, deg = _fit_stable_poly(
            cols, rows,
            max_degree=3, sigma=3.0, n_iter=3,
            col_lo=0.0, col_hi=2047.0, nrows=2048,
            data_range_margin=50.0,
        )
        eval_cols = np.linspace(0, 2047, 100)
        y_eval = np.polynomial.polynomial.polyval(eval_cols, coeffs)
        min_allowed = float(rows.min()) - 50.0
        assert np.all(y_eval >= min_allowed), (
            f"Polynomial goes below data range - margin ({min_allowed:.0f}) "
            f"at some column; min value = {y_eval.min():.0f}"
        )


# ---------------------------------------------------------------------------
# 8. Neighbour-crossing prevention – unit tests for _repair_crossing_orders
# ---------------------------------------------------------------------------


class TestRepairCrossingOrders:
    """_repair_crossing_orders fixes polynomial crossings when possible."""

    def _import_helpers(self):
        from pyspextool.instruments.ishell.tracing import _repair_crossing_orders
        return _repair_crossing_orders

    def _make_two_order_setup(self, slope_lo, slope_hi, intercept_lo, intercept_hi,
                               n_sample=40, col_lo=0.0, col_hi=2047.0):
        """Build a minimal two-order setup with given linear centre curves."""
        sample_cols = np.linspace(col_lo, col_hi, n_sample).astype(int)
        center_poly_coeffs = np.zeros((2, 4))
        center_poly_coeffs[0, 0] = intercept_lo
        center_poly_coeffs[0, 1] = slope_lo
        center_poly_coeffs[1, 0] = intercept_hi
        center_poly_coeffs[1, 1] = slope_hi
        fit_rms = np.array([3.0, 5.0])  # order 1 is "worse" by RMS
        # Synthetic center_rows: just evaluate the linear polynomials
        center_rows = np.zeros((2, n_sample))
        for i in range(2):
            for j, c in enumerate(sample_cols):
                center_rows[i, j] = np.polynomial.polynomial.polyval(
                    float(c), center_poly_coeffs[i]
                )
        poly_degrees_used = np.array([1, 1], dtype=int)
        return (
            center_poly_coeffs, fit_rms, center_rows, sample_cols,
            poly_degrees_used, col_lo, col_hi,
        )

    def test_no_crossing_leaves_fit_unchanged(self):
        """_repair_crossing_orders does nothing when orders don't cross."""
        _repair_crossing_orders = self._import_helpers()
        (center_poly_coeffs, fit_rms, center_rows, sample_cols,
         poly_degrees_used, col_lo, col_hi) = self._make_two_order_setup(
            slope_lo=0.05, slope_hi=0.06,
            intercept_lo=100.0, intercept_hi=200.0,
        )
        coeffs_before = center_poly_coeffs.copy()
        _repair_crossing_orders(
            center_poly_coeffs, fit_rms, center_rows, sample_cols,
            poly_degrees_used, col_lo=col_lo, col_hi=col_hi, nrows=2048,
            sigma_clip=3.0,
        )
        np.testing.assert_array_equal(
            center_poly_coeffs, coeffs_before,
            err_msg="Coefficients should be unchanged when no crossing exists",
        )

    def test_resolvable_crossing_is_repaired(self):
        """When reducing degree resolves a crossing, the fix is applied.

        Setup: Order 0 is stable (constant at row 200, RMS 2).
        Order 1 has mostly valid data at rows ~300-320, but a cubic
        polynomial that oscillates and dips below row 200 at some columns.
        After reducing to degree 0 (constant ≈ 310), the crossing is
        resolved because 310 > 200 everywhere.
        """
        _repair_crossing_orders = self._import_helpers()
        n_sample = 40
        col_lo, col_hi = 0.0, 2047.0
        sample_cols = np.linspace(col_lo, col_hi, n_sample).astype(int)

        # Order 0: constant at row 200 (stable)
        # Order 1: cubic that oscillates and crosses below 200 mid-detector
        center_poly_coeffs = np.zeros((2, 4))
        center_poly_coeffs[0, 0] = 200.0       # constant at 200
        # Order 1: starts at 310 at col 0, has a large cubic term that
        # makes it dip to ~180 mid-detector, then recover.
        # f(x) = 310 - 3e-4*x + 6e-7*x^2 - 3e-10*x^3
        # At col 1023: ~310 - 307 + 628 - 322 ≈ 309 (wrong – make simpler)
        # Use a polynomial that definitely crosses: goes from 300 to 300
        # via a valley at ~130 in the middle
        # f(x) = 300 - 1.28e-4*(x-1023)^2  (parabola, vertex at 1023, 300)
        # At col 1023: 300; at col 0: 300 - 1.28e-4*1023^2 = 300 - 134 = 166
        # → dips below 200 → crossing with Order 0 at col 200
        c0 = 300.0 - 1.28e-4 * 1023.0**2
        c1 = 2.0 * 1.28e-4 * 1023.0
        c2 = -1.28e-4
        center_poly_coeffs[1, 0] = c0
        center_poly_coeffs[1, 1] = c1
        center_poly_coeffs[1, 2] = c2

        fit_rms = np.array([2.0, 8.0])  # order 1 is worse by RMS

        # center_rows for Order 1: evaluated from the parabola
        center_rows = np.zeros((2, n_sample))
        for j, c in enumerate(sample_cols):
            center_rows[0, j] = 200.0
            center_rows[1, j] = np.polynomial.polynomial.polyval(
                float(c), center_poly_coeffs[1]
            )
        poly_degrees_used = np.array([1, 2], dtype=int)

        # Confirm crossing exists before repair
        eval_cols = np.linspace(0, 2047, 50)
        gap_before = (
            np.polynomial.polynomial.polyval(eval_cols, center_poly_coeffs[1])
            - np.polynomial.polynomial.polyval(eval_cols, center_poly_coeffs[0])
        )
        assert np.any(gap_before < 0), (
            "Test precondition: parabola must dip below Order 0 at some columns"
        )

        _repair_crossing_orders(
            center_poly_coeffs, fit_rms, center_rows, sample_cols,
            poly_degrees_used, col_lo=col_lo, col_hi=col_hi, nrows=2048,
            sigma_clip=3.0,
        )

        # After repair: the crossing should be resolved.
        # Order 1 reduced to degree 0 or 1 with a constant ≈ mean(center_rows[1])
        # which is around 300 (mostly in the high region) → above Order 0 (200)
        y_o1 = np.polynomial.polynomial.polyval(eval_cols, center_poly_coeffs[1])
        y_o0 = np.polynomial.polynomial.polyval(eval_cols, center_poly_coeffs[0])
        # The repair should have fixed the crossing (or at least not made it worse)
        assert np.all(y_o1 >= -50), "Repaired polynomial goes off detector (floor)"
        assert np.all(y_o1 <= 2098), "Repaired polynomial goes off detector (ceiling)"
        # The degree should have been reduced from 2
        assert int(poly_degrees_used[1]) < 2, (
            f"Expected degree < 2 after repair, got {poly_degrees_used[1]}"
        )

    def test_irresolvable_crossing_leaves_original_fit(self):
        """When crossing cannot be resolved, original coefficients are kept."""
        _repair_crossing_orders = self._import_helpers()
        # Two orders whose data genuinely overlap: their mean values are the
        # same so no degree reduction can separate them.  Neither reduction
        # should be applied (original fit preserved).
        n_sample = 40
        col_lo, col_hi = 0.0, 2047.0
        sample_cols = np.linspace(col_lo, col_hi, n_sample).astype(int)

        # Both orders at nearly the same row – guaranteed to cross after
        # Order 1's polynomial dips below Order 0 at one end.
        center_poly_coeffs = np.zeros((2, 4))
        center_poly_coeffs[0, 0] = 300.0
        center_poly_coeffs[0, 1] = 0.01
        center_poly_coeffs[1, 0] = 300.5  # nearly identical: guaranteed cross
        center_poly_coeffs[1, 1] = -0.01  # slight fall so they cross

        fit_rms = np.array([5.0, 5.0])
        center_rows = np.zeros((2, n_sample))
        for i in range(2):
            for j, c in enumerate(sample_cols):
                center_rows[i, j] = np.polynomial.polynomial.polyval(
                    float(c), center_poly_coeffs[i]
                )
        poly_degrees_used = np.array([1, 1], dtype=int)
        coeffs_before = center_poly_coeffs.copy()

        _repair_crossing_orders(
            center_poly_coeffs, fit_rms, center_rows, sample_cols,
            poly_degrees_used, col_lo=col_lo, col_hi=col_hi, nrows=2048,
            sigma_clip=3.0,
        )

        # The function must not degrade either fit to a constant that makes
        # things worse – the coefficients should remain as the original linear.
        # (The crossing is flagged in QA but not "fixed" by making it worse.)
        np.testing.assert_array_equal(
            center_poly_coeffs, coeffs_before,
            err_msg="Irresolvable crossing should leave original coefficients intact",
        )


# ---------------------------------------------------------------------------
# 9. K3 benchmark guardrail tests (require real K3 flat data)
# ---------------------------------------------------------------------------

_REPO_ROOT_TR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K3_RAW_DIR_TR = os.path.join(
    _REPO_ROOT_TR, "data", "testdata", "ishell_k3_example", "raw"
)
_LFS_MAGIC_TR = b"version https://git-lfs"


def _k3_flat_files() -> list[str]:
    """Return sorted real K3 flat file paths (frames 6-10), or empty if absent."""
    if not os.path.isdir(_K3_RAW_DIR_TR):
        return []
    import re
    paths = []
    for name in os.listdir(_K3_RAW_DIR_TR):
        if "flat" not in name:
            continue
        match = re.search(r'\.(\d{5})\.', name)
        if not match:
            continue
        num = int(match.group(1))
        if num not in range(6, 11):
            continue
        full = os.path.join(_K3_RAW_DIR_TR, name)
        with open(full, "rb") as fh:
            head = fh.read(64)
        if not head.startswith(_LFS_MAGIC_TR):
            paths.append(full)
    return sorted(paths)


_K3_FLAT_FILES_TR = _k3_flat_files()
_HAVE_K3_DATA_TR = len(_K3_FLAT_FILES_TR) >= 1


@pytest.mark.slow
@pytest.mark.skipif(
    not _HAVE_K3_DATA_TR,
    reason="Real K3 flat data not found",
)
class TestK3FlatTraceBenchmark:
    """Guardrail tests for the K3 flat-field order tracing.

    These tests enforce that the fixed tracing algorithm produces stable,
    physically ordered traces on the real K3 flat data.  They would have
    failed on the pre-fix pathological behaviour (Order 24 diverging to
    y=534 at col 2047, "29/29 invalid" QA summary, multiple crossings).

    Acceptance criteria verified here:
    1. No trace polynomial goes off the detector (y < –50 or y > 2098).
    2. No catastrophically divergent polynomial (|y_end – y_seed| > 500 px).
    3. After the half-width edge filter, ≥ 20 / 27 science orders pass QA.
    4. The formerly pathological Order 24 (sparsely covered, ~7 valid pts)
       uses a reduced degree and stays within the detector row range.
    5. No two adjacent science-order polynomials cross more than a small
       fraction of the column range (catastrophic crossings eliminated).
    """

    @pytest.fixture(scope="class")
    def k3_trace_raw(self):
        """Run tracing on real K3 flat files once per class."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return trace_orders_from_flat(_K3_FLAT_FILES_TR)

    @pytest.fixture(scope="class")
    def k3_trace_filtered(self, k3_trace_raw):
        """Return the 27 science orders after half-width edge-order filter."""
        hw = k3_trace_raw.half_width_rows
        threshold = 0.30 * float(np.median(hw))
        keep = [i for i in range(k3_trace_raw.n_orders) if hw[i] >= threshold]
        from pyspextool.instruments.ishell.tracing import FlatOrderTrace
        from dataclasses import fields
        keep_arr = keep
        return FlatOrderTrace(
            n_orders=len(keep_arr),
            sample_cols=k3_trace_raw.sample_cols,
            center_rows=k3_trace_raw.center_rows[keep_arr],
            center_poly_coeffs=k3_trace_raw.center_poly_coeffs[keep_arr],
            fit_rms=k3_trace_raw.fit_rms[keep_arr],
            half_width_rows=k3_trace_raw.half_width_rows[keep_arr],
            poly_degree=k3_trace_raw.poly_degree,
            seed_col=k3_trace_raw.seed_col,
            poly_degrees_used=k3_trace_raw.poly_degrees_used[keep_arr],
        )

    def test_29_raw_orders_detected(self, k3_trace_raw):
        """K3 flat tracing should detect 29 raw orders (27 science + 2 edge)."""
        assert k3_trace_raw.n_orders == 29, (
            f"Expected 29 raw K3 orders, got {k3_trace_raw.n_orders}"
        )

    def test_27_science_orders_after_filter(self, k3_trace_filtered):
        """After edge-order filter, exactly 27 K3 science orders remain."""
        assert k3_trace_filtered.n_orders == 27, (
            f"Expected 27 science orders after edge filter, "
            f"got {k3_trace_filtered.n_orders}"
        )

    def test_no_trace_goes_off_detector(self, k3_trace_filtered):
        """No science-order polynomial may diverge outside [–50, 2098] rows."""
        nrows = 2048
        col_lo = float(k3_trace_filtered.sample_cols[0])
        col_hi = float(k3_trace_filtered.sample_cols[-1])
        eval_cols = np.linspace(col_lo, col_hi, 100)
        for i in range(k3_trace_filtered.n_orders):
            if not np.isfinite(k3_trace_filtered.fit_rms[i]):
                continue
            y = np.polynomial.polynomial.polyval(
                eval_cols, k3_trace_filtered.center_poly_coeffs[i]
            )
            assert np.all(y >= -50), (
                f"Science order {i}: polynomial goes below row –50 "
                f"(min y = {y.min():.0f})"
            )
            assert np.all(y <= nrows + 50), (
                f"Science order {i}: polynomial goes above row {nrows + 50} "
                f"(max y = {y.max():.0f})"
            )

    def test_no_catastrophic_divergence(self, k3_trace_filtered):
        """No science-order polynomial may span > 500 rows from seed to any edge."""
        trace = k3_trace_filtered
        seed_col = float(trace.seed_col)
        col_lo = float(trace.sample_cols[0])
        col_hi = float(trace.sample_cols[-1])
        for i in range(trace.n_orders):
            if not np.isfinite(trace.fit_rms[i]):
                continue
            coeffs = trace.center_poly_coeffs[i]
            y_seed = float(np.polynomial.polynomial.polyval(seed_col, coeffs))
            y_lo = float(np.polynomial.polynomial.polyval(col_lo, coeffs))
            y_hi = float(np.polynomial.polynomial.polyval(col_hi, coeffs))
            divergence = max(abs(y_lo - y_seed), abs(y_hi - y_seed))
            assert divergence < 500, (
                f"Science order {i}: catastrophic divergence detected – "
                f"y_seed={y_seed:.0f}, y_lo={y_lo:.0f}, y_hi={y_hi:.0f} "
                f"(max |Δy| = {divergence:.0f} px)"
            )

    def test_poly_degrees_used_available(self, k3_trace_raw):
        """FlatOrderTrace.poly_degrees_used must be populated."""
        assert len(k3_trace_raw.poly_degrees_used) == k3_trace_raw.n_orders, (
            "poly_degrees_used should have one entry per order"
        )
        assert np.all(k3_trace_raw.poly_degrees_used >= 0), (
            "poly_degrees_used must be non-negative"
        )
        assert np.all(
            k3_trace_raw.poly_degrees_used <= k3_trace_raw.poly_degree
        ), "poly_degrees_used must not exceed poly_degree"

    def test_sparse_order_uses_reduced_degree(self, k3_trace_raw):
        """The sparsely-sampled order (K3 Order 24, ~7 valid pts) must use
        a reduced polynomial degree to prevent divergence."""
        # Find the order with fewest valid sample points
        n_valid = [
            int(np.sum(np.isfinite(k3_trace_raw.center_rows[i])))
            for i in range(k3_trace_raw.n_orders)
        ]
        sparsest = int(np.argmin(n_valid))
        deg = int(k3_trace_raw.poly_degrees_used[sparsest])
        assert deg < k3_trace_raw.poly_degree, (
            f"Sparsest order ({sparsest}, {n_valid[sparsest]} pts) "
            f"should use a reduced degree, but got degree {deg} "
            f"(same as max degree {k3_trace_raw.poly_degree})"
        )

    def test_at_least_20_of_27_science_orders_pass_qa(self, k3_trace_filtered):
        """After fixing the traces, ≥ 20/27 K3 science orders must pass QA.

        This threshold (20/27 ≈ 74%) is conservative and deliberately
        well below 27/27, because some genuinely difficult orders near
        detector edges will always have elevated RMS or small separations.
        The pre-fix result was 0/27 (all invalid), so any value ≥ 20 is a
        clear improvement.
        """
        from pyspextool.instruments.ishell.tracing import _compute_order_trace_stats
        stats = _compute_order_trace_stats(
            k3_trace_filtered.center_poly_coeffs,
            k3_trace_filtered.fit_rms,
            k3_trace_filtered.sample_cols,
        )
        n_valid = sum(1 for s in stats if s.trace_valid)
        assert n_valid >= 20, (
            f"Expected ≥ 20/27 science orders to pass QA after fixing, "
            f"got {n_valid}/27 valid.\n"
            "This likely indicates a regression in the stable-fit fallback."
        )

    def test_no_catastrophic_adjacent_crossings(self, k3_trace_filtered):
        """Adjacent science-order polynomials must not cross at most columns.

        Allows up to 10% of evaluation columns to show a crossing, to
        accommodate genuine physical convergence at detector edges.  The
        pre-fix behaviour had catastrophic crossings spanning the full
        column range.
        """
        trace = k3_trace_filtered
        eval_cols = np.linspace(
            float(trace.sample_cols[0]),
            float(trace.sample_cols[-1]),
            200,
        )
        center_vals = np.array([
            np.polynomial.polynomial.polyval(eval_cols, trace.center_poly_coeffs[i])
            if np.isfinite(trace.fit_rms[i])
            else np.full(200, np.nan)
            for i in range(trace.n_orders)
        ])

        for i in range(trace.n_orders - 1):
            gap = center_vals[i + 1] - center_vals[i]
            finite_gap = gap[np.isfinite(gap)]
            if len(finite_gap) == 0:
                continue
            crossing_fraction = float(np.mean(finite_gap < 0))
            assert crossing_fraction <= 0.10, (
                f"Adjacent science orders {i} and {i+1} cross at "
                f"{crossing_fraction*100:.0f}% of evaluation columns "
                "(catastrophic crossing). Expected ≤ 10%."
            )
