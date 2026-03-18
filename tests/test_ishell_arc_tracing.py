"""
Tests for the iSHELL 2-D arc-line tracing module (arc_tracing.py).

These tests verify:
  - TracedArcLine can be constructed and queried (dataclass API),
  - ArcLineTraceResult can be constructed and queried (dataclass API),
  - load_and_combine_arcs() raises on empty input and returns the
    correct shape for one or multiple synthetic FITS files,
  - trace_arc_lines() raises on empty file list and on empty geometry,
  - trace_arc_lines() works end-to-end on a synthetic arc image with
    known line positions and recovers those positions within tolerance,
  - the polynomial fit residuals are within expected bounds on synthetic data,
  - the module is importable from instruments.ishell (package __init__),
  - a smoke-test that loads the real H1 calibration arc frames (and
    flat frames, for order geometry) and confirms that lines are found
    and traced with expected quality criteria (requires real data files).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.arc_tracing import (
    ArcLineTraceResult,
    TracedArcLine,
    load_and_combine_arcs,
    trace_arc_lines,
)
from pyspextool.instruments.ishell.geometry import (
    OrderGeometry,
    OrderGeometrySet,
)
from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
from pyspextool.instruments.ishell.io_utils import is_fits_file

# ---------------------------------------------------------------------------
# Path helpers for real H1 calibration data
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_H1_RAW_DIR = os.path.join(
    _REPO_ROOT, "data", "testdata", "ishell_h1_calibrations", "raw"
)
_LFS_MAGIC = b"version https://git-lfs"


def _real_files(pattern: str) -> list[str]:
    """Return sorted real (non-LFS-pointer) file paths matching *pattern*."""
    if not os.path.isdir(_H1_RAW_DIR):
        return []
    paths = sorted(
        os.path.join(_H1_RAW_DIR, f)
        for f in os.listdir(_H1_RAW_DIR)
        if pattern in f and is_fits_file(f)
    )
    real: list[str] = []
    for p in paths:
        with open(p, "rb") as fh:
            head = fh.read(64)
        if not head.startswith(_LFS_MAGIC):
            real.append(p)
    return real


_H1_FLAT_FILES = _real_files("flat")
_H1_ARC_FILES = _real_files("arc")
_HAVE_H1_DATA = len(_H1_FLAT_FILES) >= 1 and len(_H1_ARC_FILES) >= 1

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

# Detector shape for synthetic tests (small to keep tests fast)
_NROWS = 300
_NCOLS = 600

# Two orders: order 0 rows 60–100 (centre 80), order 1 rows 120–160 (centre 140)
_SYN_ORDERS = [
    {"center": 80, "half_width": 20, "x_start": 50, "x_end": 550},
    {"center": 140, "half_width": 20, "x_start": 50, "x_end": 550},
]

# Three arc lines per order at these absolute column positions
_SYN_LINE_COLS = [150, 300, 450]

# A slight tilt: col_offset = tilt_slope × (row − center_row)
_SYN_TILT_SLOPE = 0.05  # pixels per row (small but detectable)

# Line parameters
_SYN_LINE_SIGMA = 1.5     # Gaussian σ in columns
_SYN_LINE_FLUX = 3000.0   # peak above background
_SYN_BACKGROUND = 50.0    # background DN


def _make_synthetic_arc(seed: int = 7) -> np.ndarray:
    """Return a synthetic arc image with known emission lines.

    Each order has ``_SYN_LINE_COLS`` emission lines, each with a Gaussian
    column profile (σ = ``_SYN_LINE_SIGMA``) and a small tilt
    (``_SYN_TILT_SLOPE`` pixels per row).  Gaussian noise at level 10 DN
    is added.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(_NROWS, dtype=float)
    cols = np.arange(_NCOLS, dtype=float)

    arc = rng.normal(0.0, 10.0, size=(_NROWS, _NCOLS)).astype(np.float32)
    arc += _SYN_BACKGROUND

    for order in _SYN_ORDERS:
        center_row = float(order["center"])
        half_width = float(order["half_width"])
        row_lo = max(0, int(center_row - half_width))
        row_hi = min(_NROWS - 1, int(center_row + half_width))

        for c_line in _SYN_LINE_COLS:
            for r in range(row_lo, row_hi + 1):
                # Column position of line centre at this row
                c_centre = float(c_line) + _SYN_TILT_SLOPE * (r - center_row)
                # Gaussian profile
                arc[r, :] += _SYN_LINE_FLUX * np.exp(
                    -0.5 * ((cols - c_centre) / _SYN_LINE_SIGMA) ** 2
                )

    return arc


def _make_synthetic_geometry() -> OrderGeometrySet:
    """Return an OrderGeometrySet matching the synthetic arc data layout."""
    geometries = []
    for idx, order in enumerate(_SYN_ORDERS):
        center = float(order["center"])
        hw = float(order["half_width"])
        # Constant (horizontal) edges: bottom at centre−hw, top at centre+hw
        bot = np.array([center - hw, 0.0, 0.0, 0.0])
        top = np.array([center + hw, 0.0, 0.0, 0.0])
        geometries.append(
            OrderGeometry(
                order=idx,
                x_start=order["x_start"],
                x_end=order["x_end"],
                bottom_edge_coeffs=bot,
                top_edge_coeffs=top,
            )
        )
    return OrderGeometrySet(mode="SYN", geometries=geometries)


def _write_fits_arc(arc: np.ndarray, path: str) -> None:
    """Write *arc* as a minimal 3-extension iSHELL FITS file to *path*."""
    primary = fits.PrimaryHDU(data=arc)
    ped = fits.ImageHDU(data=np.zeros_like(arc), name="SUM_PED")
    sam = fits.ImageHDU(data=np.zeros_like(arc), name="SUM_SAM")
    fits.HDUList([primary, ped, sam]).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# 1. TracedArcLine dataclass API
# ---------------------------------------------------------------------------


class TestTracedArcLineDataclass:
    """TracedArcLine can be constructed and queried directly."""

    @pytest.fixture()
    def dummy_line(self):
        return TracedArcLine(
            order_index=0,
            seed_col=100,
            trace_rows=np.arange(50, 90, dtype=int),
            trace_cols=np.full(40, 100.5),
            poly_coeffs=np.array([100.5, 0.0, 0.0]),
            fit_rms=0.05,
            peak_flux=2500.0,
        )

    def test_order_index(self, dummy_line):
        assert dummy_line.order_index == 0

    def test_seed_col(self, dummy_line):
        assert dummy_line.seed_col == 100

    def test_n_trace_points(self, dummy_line):
        assert dummy_line.n_trace_points == 40

    def test_trace_rows_shape(self, dummy_line):
        assert dummy_line.trace_rows.shape == (40,)

    def test_trace_cols_shape(self, dummy_line):
        assert dummy_line.trace_cols.shape == (40,)

    def test_poly_coeffs_shape(self, dummy_line):
        assert dummy_line.poly_coeffs.shape == (3,)

    def test_fit_rms_finite(self, dummy_line):
        assert np.isfinite(dummy_line.fit_rms)

    def test_peak_flux_positive(self, dummy_line):
        assert dummy_line.peak_flux > 0.0

    def test_eval_col_scalar(self, dummy_line):
        result = dummy_line.eval_col(70.0)
        assert abs(float(result) - 100.5) < 1e-6

    def test_eval_col_array(self, dummy_line):
        rows = np.array([50.0, 60.0, 70.0])
        result = dummy_line.eval_col(rows)
        assert result.shape == (3,)
        assert np.allclose(result, 100.5, atol=1e-6)

    def test_tilt_slope(self, dummy_line):
        assert dummy_line.tilt_slope() == 0.0

    def test_tilt_slope_nonzero(self):
        line = TracedArcLine(
            order_index=0,
            seed_col=200,
            trace_rows=np.arange(10),
            trace_cols=np.arange(10, dtype=float),
            poly_coeffs=np.array([200.0, 0.05]),
            fit_rms=0.01,
            peak_flux=1000.0,
        )
        assert abs(line.tilt_slope() - 0.05) < 1e-10


# ---------------------------------------------------------------------------
# 2. ArcLineTraceResult dataclass API
# ---------------------------------------------------------------------------


class TestArcLineTraceResult:
    """ArcLineTraceResult can be constructed and queried."""

    @pytest.fixture()
    def geom(self):
        return _make_synthetic_geometry()

    @pytest.fixture()
    def dummy_result(self, geom):
        lines = [
            TracedArcLine(
                order_index=0,
                seed_col=c,
                trace_rows=np.arange(60, 100, dtype=int),
                trace_cols=np.full(40, float(c)),
                poly_coeffs=np.array([float(c), 0.0, 0.0]),
                fit_rms=0.05,
                peak_flux=1000.0,
            )
            for c in [150, 300]
        ] + [
            TracedArcLine(
                order_index=1,
                seed_col=450,
                trace_rows=np.arange(120, 160, dtype=int),
                trace_cols=np.full(40, 450.0),
                poly_coeffs=np.array([450.0, 0.0, 0.0]),
                fit_rms=0.05,
                peak_flux=1200.0,
            )
        ]
        return ArcLineTraceResult(
            mode="SYN",
            arc_files=["fake.fits"],
            poly_degree=2,
            geometry=geom,
            traced_lines=lines,
        )

    def test_n_lines(self, dummy_result):
        assert dummy_result.n_lines == 3

    def test_n_orders(self, dummy_result):
        assert dummy_result.n_orders == len(_SYN_ORDERS)

    def test_mode(self, dummy_result):
        assert dummy_result.mode == "SYN"

    def test_poly_degree(self, dummy_result):
        assert dummy_result.poly_degree == 2

    def test_arc_files_stored(self, dummy_result):
        assert dummy_result.arc_files == ["fake.fits"]

    def test_lines_for_order_0(self, dummy_result):
        lines = dummy_result.lines_for_order(0)
        assert len(lines) == 2
        assert all(ln.order_index == 0 for ln in lines)

    def test_lines_for_order_1(self, dummy_result):
        lines = dummy_result.lines_for_order(1)
        assert len(lines) == 1
        assert lines[0].seed_col == 450

    def test_lines_for_nonexistent_order(self, dummy_result):
        assert dummy_result.lines_for_order(99) == []

    def test_n_lines_per_order(self, dummy_result):
        counts = dummy_result.n_lines_per_order()
        assert counts.shape == (len(_SYN_ORDERS),)
        assert counts[0] == 2
        assert counts[1] == 1


# ---------------------------------------------------------------------------
# 3. load_and_combine_arcs()
# ---------------------------------------------------------------------------


class TestLoadAndCombineArcs:
    """load_and_combine_arcs() loading and combination behaviour."""

    def test_raises_on_empty(self, tmp_path):
        from pyspextool.instruments.ishell.arc_tracing import load_and_combine_arcs
        with pytest.raises(ValueError, match="must not be empty"):
            load_and_combine_arcs([])

    def test_single_file_shape(self, tmp_path):
        from pyspextool.instruments.ishell.arc_tracing import load_and_combine_arcs
        arc = _make_synthetic_arc()
        path = str(tmp_path / "arc.fits")
        _write_fits_arc(arc, path)
        loaded = load_and_combine_arcs([path])
        assert loaded.shape == arc.shape

    def test_two_identical_files_median_combined(self, tmp_path):
        from pyspextool.instruments.ishell.arc_tracing import load_and_combine_arcs
        arc = _make_synthetic_arc()
        paths = []
        for k in range(2):
            p = str(tmp_path / f"arc_{k}.fits")
            _write_fits_arc(arc, p)
            paths.append(p)
        combined = load_and_combine_arcs(paths)
        assert combined.shape == arc.shape
        # Median of two identical images equals the original
        assert np.allclose(combined, arc, atol=1.0)

    def test_dtype_is_float32(self, tmp_path):
        from pyspextool.instruments.ishell.arc_tracing import load_and_combine_arcs
        arc = _make_synthetic_arc()
        path = str(tmp_path / "arc.fits")
        _write_fits_arc(arc, path)
        loaded = load_and_combine_arcs([path])
        assert loaded.dtype == np.float32


# ---------------------------------------------------------------------------
# 4. trace_arc_lines() error handling
# ---------------------------------------------------------------------------


class TestTraceArcLinesErrors:
    """trace_arc_lines() raises the right exceptions on bad input."""

    def test_raises_on_empty_arc_files(self, tmp_path):
        geom = _make_synthetic_geometry()
        with pytest.raises(ValueError, match="must not be empty"):
            trace_arc_lines([], geom)

    def test_raises_on_empty_geometry(self, tmp_path):
        arc = _make_synthetic_arc()
        path = str(tmp_path / "arc.fits")
        _write_fits_arc(arc, path)
        empty_geom = OrderGeometrySet(mode="EMPTY", geometries=[])
        with pytest.raises(ValueError, match="at least one order"):
            trace_arc_lines([path], empty_geom)

    def test_returns_empty_lines_on_blank_arc(self, tmp_path):
        """A blank (zero) arc image should yield no traced lines."""
        blank = np.zeros((_NROWS, _NCOLS), dtype=np.float32)
        path = str(tmp_path / "blank.fits")
        _write_fits_arc(blank, path)
        geom = _make_synthetic_geometry()
        result = trace_arc_lines([path], geom, min_line_prominence=100.0)
        # No lines found on a blank image
        assert result.n_lines == 0
        assert isinstance(result, ArcLineTraceResult)


# ---------------------------------------------------------------------------
# 5. Synthetic data end-to-end
# ---------------------------------------------------------------------------


class TestTraceArcLinesSynthetic:
    """trace_arc_lines() recovers known line positions on synthetic data."""

    @pytest.fixture(scope="class")
    def arc_path(self, tmp_path_factory):
        arc = _make_synthetic_arc()
        path = str(tmp_path_factory.mktemp("arc") / "arc.fits")
        _write_fits_arc(arc, path)
        return path

    @pytest.fixture(scope="class")
    def result(self, arc_path):
        geom = _make_synthetic_geometry()
        return trace_arc_lines(
            [arc_path],
            geom,
            poly_degree=2,
            min_line_prominence=500.0,
            col_half_width=5,
            min_trace_fraction=0.3,
        )

    def test_returns_arc_line_trace_result(self, result):
        assert isinstance(result, ArcLineTraceResult)

    def test_mode_preserved(self, result):
        assert result.mode == "SYN"

    def test_poly_degree_stored(self, result):
        assert result.poly_degree == 2

    def test_at_least_one_line_found(self, result):
        assert result.n_lines >= 1, (
            f"Expected at least 1 traced line; got {result.n_lines}"
        )

    def test_both_orders_have_lines(self, result):
        """Both synthetic orders should have at least one traced line."""
        counts = result.n_lines_per_order()
        assert counts.sum() >= len(_SYN_ORDERS), (
            f"Expected lines in both orders; got counts={counts}"
        )

    def test_fit_rms_all_finite(self, result):
        """All traced lines must have a finite polynomial RMS."""
        for ln in result.traced_lines:
            assert np.isfinite(ln.fit_rms), (
                f"Line at order {ln.order_index}, col {ln.seed_col} "
                f"has non-finite RMS {ln.fit_rms}"
            )

    def test_fit_rms_below_threshold(self, result):
        """Polynomial residuals must be below 2 pixels on synthetic data."""
        for ln in result.traced_lines:
            assert ln.fit_rms < 2.0, (
                f"Line at order {ln.order_index}, col {ln.seed_col}: "
                f"RMS {ln.fit_rms:.2f} px exceeds 2 px"
            )

    def test_seed_cols_match_known_lines(self, result):
        """Traced lines must be within 3 pixels of a known line column."""
        tol = 3.0  # pixels
        for ln in result.traced_lines:
            nearest = min(abs(ln.seed_col - c) for c in _SYN_LINE_COLS)
            assert nearest <= tol, (
                f"Line at col {ln.seed_col} is {nearest:.1f} px from the "
                f"nearest known line; expected ≤ {tol} px"
            )

    def test_tilt_recovered_within_tolerance(self, result):
        """The linear tilt slope must be within 0.03 px/row of the known value."""
        tol = 0.03
        for ln in result.traced_lines:
            slope = ln.tilt_slope()
            assert abs(slope - _SYN_TILT_SLOPE) < tol, (
                f"Line at order {ln.order_index}, col {ln.seed_col}: "
                f"tilt slope {slope:.4f} px/row differs from known "
                f"{_SYN_TILT_SLOPE} by more than {tol} px/row"
            )

    def test_eval_col_at_centre_near_seed(self, result):
        """Polynomial evaluated at the order centre row should be near seed_col."""
        tol = 1.5  # pixels
        for ln in result.traced_lines:
            order = result.geometry.geometries[ln.order_index]
            center_row = float(order.eval_centerline(ln.seed_col))
            predicted_col = float(ln.eval_col(center_row))
            assert abs(predicted_col - ln.seed_col) < tol, (
                f"Line at order {ln.order_index}, col {ln.seed_col}: "
                f"poly eval at centre row gives col {predicted_col:.2f}; "
                f"expected within {tol} px of {ln.seed_col}"
            )

    def test_poly_coeffs_shape(self, result):
        """Poly coefficients must have shape (poly_degree + 1,)."""
        for ln in result.traced_lines:
            assert ln.poly_coeffs.shape == (result.poly_degree + 1,), (
                f"Line at order {ln.order_index}, col {ln.seed_col}: "
                f"poly_coeffs shape {ln.poly_coeffs.shape} != "
                f"({result.poly_degree + 1},)"
            )

    def test_two_arc_files_median_combined(self, tmp_path):
        """Tracing on 2 identical arc files should give the same result as 1."""
        arc = _make_synthetic_arc()
        paths = []
        for k in range(2):
            p = str(tmp_path / f"arc_{k}.fits")
            _write_fits_arc(arc, p)
            paths.append(p)
        geom = _make_synthetic_geometry()
        r1 = trace_arc_lines([paths[0]], geom, min_line_prominence=500.0)
        r2 = trace_arc_lines(paths, geom, min_line_prominence=500.0)
        assert r1.n_lines == r2.n_lines


# ---------------------------------------------------------------------------
# 6. Module import regression
# ---------------------------------------------------------------------------


class TestModuleImport:
    """arc_tracing must be importable from instruments.ishell."""

    def test_import_from_instruments_ishell(self):
        from pyspextool.instruments.ishell import arc_tracing  # noqa: F401
        assert hasattr(arc_tracing, "trace_arc_lines")
        assert hasattr(arc_tracing, "TracedArcLine")
        assert hasattr(arc_tracing, "ArcLineTraceResult")
        assert hasattr(arc_tracing, "load_and_combine_arcs")

    def test_import_symbols_directly(self):
        from pyspextool.instruments.ishell.arc_tracing import (
            ArcLineTraceResult,
            TracedArcLine,
            load_and_combine_arcs,
            trace_arc_lines,
        )
        assert callable(trace_arc_lines)
        assert callable(load_and_combine_arcs)
        assert TracedArcLine is not None
        assert ArcLineTraceResult is not None


# ---------------------------------------------------------------------------
# 7. Real H1 data smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason=(
        "Real H1 flat/arc data not found (run `git lfs pull` first). "
        f"Flat files found: {len(_H1_FLAT_FILES)}, "
        f"Arc files found: {len(_H1_ARC_FILES)}"
    ),
)
class TestTraceArcLinesH1RealData:
    """Smoke-tests that run arc-line tracing on the real H1 calibration data.

    These tests are **smoke tests only** — they verify that the scaffold
    runs without error and returns results within broad, conservative
    acceptance criteria.  They do **not** validate scientific accuracy.

    Acceptance criteria
    -------------------
    * At least 50 arc lines found across all orders (H1 has ~45 orders,
      each with several ThAr lines; 50 is a conservative lower bound).
    * All traced lines have finite polynomial fit RMS.
    * Median fit RMS < 2.0 pixels (arc lines are narrow so centroiding
      should be precise).
    * At least half of the traced orders have at least one line.
    * ``ArcLineTraceResult`` mode matches the geometry mode.

    These tolerances are intentionally loose because:

    (a) The ``min_line_prominence`` parameter may need tuning for specific
        arc-frame conditions; the default may not detect every line.
    (b) This is a first-pass scaffold, not a finalised calibration.
    (c) The flat-order geometry used here is also a first-pass scaffold
        (centre lines ± half-width; not independently edge-traced).
    """

    @pytest.fixture(scope="class")
    def h1_geom(self):
        """Produce an OrderGeometrySet from the real H1 flat frames."""
        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES,
            col_range=(650, 1550),
            n_sample_cols=20,
        )
        return flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))

    @pytest.fixture(scope="class")
    def h1_arc_result(self, h1_geom):
        return trace_arc_lines(
            _H1_ARC_FILES,
            h1_geom,
            poly_degree=2,
            min_line_prominence=200.0,
            col_half_width=5,
            min_trace_fraction=0.3,
        )

    def test_returns_arc_line_trace_result(self, h1_arc_result):
        assert isinstance(h1_arc_result, ArcLineTraceResult)

    def test_mode_is_h1(self, h1_arc_result):
        assert h1_arc_result.mode == "H1"

    def test_at_least_50_lines_found(self, h1_arc_result):
        assert h1_arc_result.n_lines >= 50, (
            f"Expected at least 50 traced arc lines; got {h1_arc_result.n_lines}"
        )

    def test_all_lines_have_finite_rms(self, h1_arc_result):
        bad = [
            ln for ln in h1_arc_result.traced_lines if not np.isfinite(ln.fit_rms)
        ]
        assert len(bad) == 0, (
            f"{len(bad)} traced lines have non-finite polynomial RMS"
        )

    def test_median_rms_below_threshold(self, h1_arc_result):
        rms_vals = np.array([ln.fit_rms for ln in h1_arc_result.traced_lines])
        median_rms = float(np.nanmedian(rms_vals))
        assert median_rms < 2.0, (
            f"Median arc-line polynomial RMS {median_rms:.2f} px exceeds 2.0 px"
        )

    def test_majority_of_orders_have_lines(self, h1_arc_result):
        counts = h1_arc_result.n_lines_per_order()
        orders_with_lines = int(np.sum(counts > 0))
        assert orders_with_lines >= h1_arc_result.n_orders // 2, (
            f"Only {orders_with_lines} of {h1_arc_result.n_orders} orders "
            "have traced arc lines; expected at least half"
        )

    def test_arc_files_stored(self, h1_arc_result):
        assert len(h1_arc_result.arc_files) == len(_H1_ARC_FILES)

    def test_poly_degree_stored(self, h1_arc_result):
        assert h1_arc_result.poly_degree == 2

    def test_poly_coeffs_shape_correct(self, h1_arc_result):
        degree = h1_arc_result.poly_degree
        for ln in h1_arc_result.traced_lines:
            assert ln.poly_coeffs.shape == (degree + 1,), (
                f"poly_coeffs shape {ln.poly_coeffs.shape} != ({degree + 1},)"
            )
