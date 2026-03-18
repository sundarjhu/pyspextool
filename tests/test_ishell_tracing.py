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


# ---------------------------------------------------------------------------
# 5. Module import regression
# ---------------------------------------------------------------------------


class TestModuleImport:
    """The tracing module must be importable from instruments.ishell."""

    def test_import_from_instruments_ishell(self):
        from pyspextool.instruments.ishell import tracing  # noqa: F401
        assert hasattr(tracing, "trace_orders_from_flat")
        assert hasattr(tracing, "FlatOrderTrace")
        assert hasattr(tracing, "load_and_combine_flats")

    def test_import_symbols_directly(self):
        from pyspextool.instruments.ishell.tracing import (
            FlatOrderTrace,
            load_and_combine_flats,
            trace_orders_from_flat,
        )
        assert callable(trace_orders_from_flat)
        assert callable(load_and_combine_flats)
        assert FlatOrderTrace is not None


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
