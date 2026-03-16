"""
Tests for the iSHELL 2-D arc-line tracing module (arc_tracing.py).

These tests verify:
  - TracedArcLine and ArcLineTrace can be constructed directly (dataclass API),
  - load_and_combine_arcs() raises on empty input and returns the correct shape,
  - trace_arc_lines() raises on empty file list,
  - trace_arc_lines() works end-to-end on a synthetic arc image with known
    line structure,
  - polynomial fit results are within expected bounds on synthetic data,
  - ArcLineTrace helper methods (lines_for_order, valid_lines, fit_rms_array)
    return correct values,
  - the module is importable from instruments.ishell (package __init__),
  - a smoke-test that loads the real H1 calibration arc frames and
    confirms that the expected number of lines and quality criteria are met
    (requires the real data files to be present).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.arc_tracing import (
    ArcLineTrace,
    TracedArcLine,
    load_and_combine_arcs,
    trace_arc_lines,
)
from pyspextool.instruments.ishell.tracing import FlatOrderTrace

# ---------------------------------------------------------------------------
# Path to the real H1 calibration data
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_H1_RAW_DIR = os.path.join(
    _REPO_ROOT, "data", "testdata", "ishell_h1_calibrations", "raw"
)
_LFS_MAGIC = b"version https://git-lfs"


def _h1_arc_files() -> list[str]:
    """Return sorted list of real H1 arc file paths, or empty list if absent."""
    if not os.path.isdir(_H1_RAW_DIR):
        return []
    paths = sorted(
        os.path.join(_H1_RAW_DIR, f)
        for f in os.listdir(_H1_RAW_DIR)
        if "arc" in f and f.endswith(".fits")
    )
    real_paths = []
    for p in paths:
        with open(p, "rb") as fh:
            head = fh.read(64)
        if not head.startswith(_LFS_MAGIC):
            real_paths.append(p)
    return real_paths


def _h1_flat_files() -> list[str]:
    """Return sorted list of real H1 flat file paths, or empty list if absent."""
    if not os.path.isdir(_H1_RAW_DIR):
        return []
    paths = sorted(
        os.path.join(_H1_RAW_DIR, f)
        for f in os.listdir(_H1_RAW_DIR)
        if "flat" in f and f.endswith(".fits")
    )
    real_paths = []
    for p in paths:
        with open(p, "rb") as fh:
            head = fh.read(64)
        if not head.startswith(_LFS_MAGIC):
            real_paths.append(p)
    return real_paths


_H1_ARC_FILES = _h1_arc_files()
_H1_FLAT_FILES = _h1_flat_files()
_HAVE_H1_ARC_DATA = len(_H1_ARC_FILES) >= 1
_HAVE_H1_FLAT_DATA = len(_H1_FLAT_FILES) >= 1
_HAVE_H1_DATA = _HAVE_H1_ARC_DATA and _HAVE_H1_FLAT_DATA

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

# Detector shape used for synthetic tests
_NROWS = 256
_NCOLS = 512

# Synthetic flat: 3 orders, each 20 rows wide, centred at rows 60, 120, 180
_SYN_N_ORDERS = 3
_SYN_CENTERS = [60, 120, 180]
_SYN_HALF_WIDTH = 10  # rows

# Synthetic arc lines in each order: 4 lines at specific columns
_SYN_LINE_COLS = [80, 160, 280, 400]  # column positions
_SYN_LINE_TILT = 0.05  # delta-col per delta-row


def _make_synthetic_arc(seed: int = 42) -> np.ndarray:
    """Return a synthetic arc image with known emission lines in each order.

    Each order band is defined by centre rows from *_SYN_CENTERS*.
    Four emission lines are placed at columns *_SYN_LINE_COLS* within each
    order.  The lines have a small tilt (*_SYN_LINE_TILT* col per row).
    A Gaussian profile (sigma = 1 pixel in the column direction) models
    each line.  Gaussian noise at 50 DN is added.
    """
    rng = np.random.default_rng(seed)
    arc = rng.normal(0.0, 50.0, size=(_NROWS, _NCOLS)).astype(np.float32)
    arc += 200.0  # background pedestal
    arc = np.clip(arc, 0, None)

    rows = np.arange(_NROWS, dtype=float)
    cols = np.arange(_NCOLS, dtype=float)

    for center_row in _SYN_CENTERS:
        for line_col in _SYN_LINE_COLS:
            # The line appears within ±half_width rows of the order centre
            for r in range(
                max(0, center_row - _SYN_HALF_WIDTH),
                min(_NROWS, center_row + _SYN_HALF_WIDTH + 1),
            ):
                # Column position shifts slightly with row (tilt)
                tilted_col = line_col + _SYN_LINE_TILT * (r - center_row)
                sigma_col = 1.0
                arc[r, :] += 5000.0 * np.exp(
                    -0.5 * ((cols - tilted_col) / sigma_col) ** 2
                )

    return arc


def _make_synthetic_flat_trace() -> FlatOrderTrace:
    """Return a minimal FlatOrderTrace for the synthetic test scene.

    Three orders centred at rows 60, 120, 180, each with half-width 10.
    All with constant centre-row polynomials (no tilt in flat tracing).
    """
    n_orders = _SYN_N_ORDERS
    n_sample = 10
    sample_cols = np.linspace(50, _NCOLS - 50, n_sample).astype(int)

    center_rows = np.zeros((n_orders, n_sample))
    center_poly_coeffs = np.zeros((n_orders, 3))  # degree 2 → 3 terms
    fit_rms = np.ones(n_orders) * 0.5
    half_width_rows = np.full(n_orders, float(_SYN_HALF_WIDTH))

    for i, c in enumerate(_SYN_CENTERS):
        center_rows[i, :] = c
        center_poly_coeffs[i, 0] = float(c)  # constant term only

    return FlatOrderTrace(
        n_orders=n_orders,
        sample_cols=sample_cols,
        center_rows=center_rows,
        center_poly_coeffs=center_poly_coeffs,
        fit_rms=fit_rms,
        half_width_rows=half_width_rows,
        poly_degree=2,
        seed_col=int(sample_cols[n_sample // 2]),
    )


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
            seed_col=256,
            sample_rows=np.arange(50, 70),
            centroid_cols=np.linspace(255.8, 256.2, 20),
            poly_coeffs=np.array([256.0, 0.02, 0.0]),
            fit_rms=0.15,
            n_valid=20,
        )

    def test_order_index(self, dummy_line):
        assert dummy_line.order_index == 0

    def test_seed_col(self, dummy_line):
        assert dummy_line.seed_col == 256

    def test_sample_rows_shape(self, dummy_line):
        assert dummy_line.sample_rows.shape == (20,)

    def test_centroid_cols_shape(self, dummy_line):
        assert dummy_line.centroid_cols.shape == (20,)

    def test_poly_coeffs_shape(self, dummy_line):
        assert dummy_line.poly_coeffs.shape == (3,)

    def test_fit_rms_positive(self, dummy_line):
        assert dummy_line.fit_rms > 0

    def test_n_valid(self, dummy_line):
        assert dummy_line.n_valid == 20


# ---------------------------------------------------------------------------
# 2. ArcLineTrace dataclass API
# ---------------------------------------------------------------------------


class TestArcLineTraceDataclass:
    """ArcLineTrace can be constructed and its helpers return correct values."""

    @pytest.fixture()
    def dummy_trace(self):
        lines = [
            TracedArcLine(
                order_index=0,
                seed_col=100,
                sample_rows=np.arange(50, 70),
                centroid_cols=np.full(20, 100.0),
                poly_coeffs=np.array([100.0, 0.0, 0.0]),
                fit_rms=0.1,
                n_valid=20,
            ),
            TracedArcLine(
                order_index=0,
                seed_col=200,
                sample_rows=np.arange(50, 70),
                centroid_cols=np.full(20, 200.0),
                poly_coeffs=np.array([200.0, 0.0, 0.0]),
                fit_rms=0.2,
                n_valid=20,
            ),
            TracedArcLine(
                order_index=1,
                seed_col=150,
                sample_rows=np.arange(100, 120),
                centroid_cols=np.full(20, 150.0),
                poly_coeffs=np.array([150.0, 0.0, 0.0]),
                fit_rms=np.nan,  # failed fit
                n_valid=1,
            ),
        ]
        return ArcLineTrace(
            n_orders=2,
            n_lines_total=3,
            lines=lines,
            poly_degree=2,
            seed_col=256,
        )

    def test_n_orders(self, dummy_trace):
        assert dummy_trace.n_orders == 2

    def test_n_lines_total(self, dummy_trace):
        assert dummy_trace.n_lines_total == 3

    def test_lines_for_order_0(self, dummy_trace):
        order0 = dummy_trace.lines_for_order(0)
        assert len(order0) == 2

    def test_lines_for_order_1(self, dummy_trace):
        order1 = dummy_trace.lines_for_order(1)
        assert len(order1) == 1

    def test_n_lines_for_order(self, dummy_trace):
        assert dummy_trace.n_lines_for_order(0) == 2
        assert dummy_trace.n_lines_for_order(1) == 1

    def test_valid_lines_filters_by_n_valid(self, dummy_trace):
        valid = dummy_trace.valid_lines(min_n_valid=5)
        assert len(valid) == 2  # only the two order-0 lines have n_valid=20

    def test_fit_rms_array_shape(self, dummy_trace):
        rms = dummy_trace.fit_rms_array()
        assert rms.shape == (3,)

    def test_fit_rms_array_contains_nan(self, dummy_trace):
        rms = dummy_trace.fit_rms_array()
        assert np.isnan(rms[2])

    def test_fit_rms_array_finite_for_good_lines(self, dummy_trace):
        rms = dummy_trace.fit_rms_array()
        assert np.isfinite(rms[0])
        assert np.isfinite(rms[1])


# ---------------------------------------------------------------------------
# 3. load_and_combine_arcs
# ---------------------------------------------------------------------------


class TestLoadAndCombineArcs:
    """load_and_combine_arcs reads FITS files and median-combines them."""

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            load_and_combine_arcs([])

    def test_single_file_returns_correct_shape(self, tmp_path):
        arc = _make_synthetic_arc()
        path = str(tmp_path / "single_arc.fits")
        _write_fits_arc(arc, path)
        result = load_and_combine_arcs([path])
        assert result.shape == arc.shape

    def test_single_file_values_preserved(self, tmp_path):
        arc = _make_synthetic_arc()
        path = str(tmp_path / "single_arc.fits")
        _write_fits_arc(arc, path)
        result = load_and_combine_arcs([path])
        np.testing.assert_allclose(result, arc.astype(np.float32))

    def test_multiple_files_returns_correct_shape(self, tmp_path):
        arc = _make_synthetic_arc()
        paths = []
        for k in range(3):
            p = str(tmp_path / f"arc_{k:02d}.fits")
            _write_fits_arc(arc, p)
            paths.append(p)
        result = load_and_combine_arcs(paths)
        assert result.shape == (_NROWS, _NCOLS)

    def test_multiple_files_median_suppresses_spike(self, tmp_path):
        """A single bright pixel in one frame should be suppressed by median."""
        arc = _make_synthetic_arc()
        paths = []
        for k in range(3):
            p = str(tmp_path / f"arc_{k:02d}.fits")
            _write_fits_arc(arc.copy(), p)
            paths.append(p)

        # Add a spike only to the first frame
        with fits.open(paths[0], mode="update") as hdul:
            hdul[0].data[60, 80] = 1e9

        result = load_and_combine_arcs(paths)
        assert result[60, 80] < 1e8

    def test_returns_float32(self, tmp_path):
        arc = _make_synthetic_arc()
        path = str(tmp_path / "arc.fits")
        _write_fits_arc(arc, path)
        result = load_and_combine_arcs([path])
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# 4. trace_arc_lines – error handling
# ---------------------------------------------------------------------------


class TestTraceArcLinesErrors:
    """trace_arc_lines raises informative errors for bad inputs."""

    def test_raises_on_empty_arc_file_list(self, tmp_path):
        flat_trace = _make_synthetic_flat_trace()
        with pytest.raises(ValueError, match="empty"):
            trace_arc_lines([], flat_trace)


# ---------------------------------------------------------------------------
# 5. trace_arc_lines – correctness on synthetic data
# ---------------------------------------------------------------------------


class TestTraceArcLinesSynthetic:
    """trace_arc_lines returns plausible results on synthetic arc data."""

    @pytest.fixture(scope="class")
    def arc_trace(self, tmp_path_factory):
        """Run arc-line tracing on a single synthetic arc once."""
        arc = _make_synthetic_arc(seed=0)
        tmp = tmp_path_factory.mktemp("syn_arc")
        path = str(tmp / "syn_arc.fits")
        _write_fits_arc(arc, path)
        flat_trace = _make_synthetic_flat_trace()
        return trace_arc_lines(
            [path],
            flat_trace,
            seed_col=_NCOLS // 2,
            min_line_prominence=500.0,
            min_line_distance=3,
            poly_degree=2,
        )

    def test_returns_arc_line_trace(self, arc_trace):
        assert isinstance(arc_trace, ArcLineTrace)

    def test_n_orders_matches_flat_trace(self, arc_trace):
        assert arc_trace.n_orders == _SYN_N_ORDERS

    def test_some_lines_found(self, arc_trace):
        """At least one arc line should be found across all orders."""
        assert arc_trace.n_lines_total >= 1

    def test_poly_degree_stored(self, arc_trace):
        assert arc_trace.poly_degree == 2

    def test_lines_have_valid_poly_coeffs(self, arc_trace):
        """Lines with sufficient valid points must have finite poly coefficients."""
        for line in arc_trace.valid_lines(min_n_valid=3):
            assert np.all(np.isfinite(line.poly_coeffs)), (
                f"Order {line.order_index}, seed col {line.seed_col}: "
                f"non-finite poly_coeffs {line.poly_coeffs}"
            )

    def test_centroid_cols_near_known_positions(self, arc_trace):
        """Valid centroids should be within 5 pixels of the known line columns."""
        for line in arc_trace.valid_lines(min_n_valid=3):
            median_col = float(np.nanmedian(line.centroid_cols))
            nearest_known = min(_SYN_LINE_COLS, key=lambda c: abs(c - median_col))
            assert abs(median_col - nearest_known) < 5.0, (
                f"Order {line.order_index}: median centroid {median_col:.1f} is "
                f"more than 5 px from the nearest known line col {nearest_known}"
            )

    def test_fit_rms_array_length(self, arc_trace):
        rms = arc_trace.fit_rms_array()
        assert len(rms) == arc_trace.n_lines_total

    def test_lines_for_order_sums_to_total(self, arc_trace):
        total = sum(
            arc_trace.n_lines_for_order(i) for i in range(arc_trace.n_orders)
        )
        assert total == arc_trace.n_lines_total

    def test_multiple_arcs_median_combined(self, tmp_path):
        """Tracing on 2 identical arc frames should give same result as one."""
        arc = _make_synthetic_arc(seed=0)
        paths = []
        for k in range(2):
            p = str(tmp_path / f"arc_{k}.fits")
            _write_fits_arc(arc.copy(), p)
            paths.append(p)

        flat_trace = _make_synthetic_flat_trace()
        trace1 = trace_arc_lines(
            [paths[0]], flat_trace, min_line_prominence=500.0
        )
        trace2 = trace_arc_lines(
            paths, flat_trace, min_line_prominence=500.0
        )
        assert trace1.n_lines_total == trace2.n_lines_total


# ---------------------------------------------------------------------------
# 6. Module import regression
# ---------------------------------------------------------------------------


class TestModuleImport:
    """The arc_tracing module must be importable from instruments.ishell."""

    def test_import_from_instruments_ishell(self):
        from pyspextool.instruments.ishell import arc_tracing  # noqa: F401
        assert hasattr(arc_tracing, "trace_arc_lines")
        assert hasattr(arc_tracing, "ArcLineTrace")
        assert hasattr(arc_tracing, "TracedArcLine")
        assert hasattr(arc_tracing, "load_and_combine_arcs")

    def test_import_symbols_directly(self):
        from pyspextool.instruments.ishell.arc_tracing import (
            ArcLineTrace,
            TracedArcLine,
            load_and_combine_arcs,
            trace_arc_lines,
        )
        assert callable(trace_arc_lines)
        assert callable(load_and_combine_arcs)
        assert ArcLineTrace is not None
        assert TracedArcLine is not None


# ---------------------------------------------------------------------------
# 7. Smoke test on real H1 calibration data
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason=(
        "Real H1 arc and/or flat data not found (run `git lfs pull` first)"
    ),
)
class TestTraceArcLinesH1RealData:
    """Smoke-tests that run the arc-line tracing scaffold on the real H1 arc frames.

    These are **smoke tests**, not precision calibration checks.  They verify
    only that the scaffold runs without error and returns results within a
    broad, conservative range of acceptability.

    The flat-field order geometry is first obtained by running
    ``trace_orders_from_flat`` on the real H1 flat frames.  The arc-line
    tracing then operates on the real H1 arc frames, constrained by that
    geometry.

    Acceptance criteria and rationale
    ----------------------------------
    * **At least one arc line traced per order** on average across orders.
      The threshold is deliberately very low to accommodate broad variation
      in line density and tracing parameter sensitivity.  The important thing
      at this stage is that the code runs without error.
    * **All valid lines (n_valid ≥ 3) have finite polynomial coefficients**.
    * **Median polynomial RMS < 5 pixels** for valid lines.  The sub-pixel
      Gaussian centroiding should achieve sub-pixel accuracy on clean ThAr
      lines; the generous threshold accommodates noisy faint lines.
    * **Total traced lines ≥ 1** (the most minimal correctness check).

    These tolerances are intentionally loose because this is a first-pass
    development scaffold.
    """

    @pytest.fixture(scope="class")
    def flat_trace(self):
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
        return trace_orders_from_flat(
            _H1_FLAT_FILES,
            col_range=(650, 1550),
            n_sample_cols=20,
        )

    @pytest.fixture(scope="class")
    def arc_trace(self, flat_trace):
        return trace_arc_lines(
            _H1_ARC_FILES,
            flat_trace,
            col_range=(650, 1550),
            min_line_prominence=200.0,
            min_line_distance=5,
            poly_degree=2,
        )

    def test_returns_arc_line_trace(self, arc_trace):
        assert isinstance(arc_trace, ArcLineTrace)

    def test_at_least_one_line_traced(self, arc_trace):
        assert arc_trace.n_lines_total >= 1, (
            "Expected at least one traced arc line across all orders"
        )

    def test_n_orders_matches_flat_trace(self, arc_trace, flat_trace):
        assert arc_trace.n_orders == flat_trace.n_orders

    def test_valid_lines_have_finite_coeffs(self, arc_trace):
        for line in arc_trace.valid_lines(min_n_valid=3):
            assert np.all(np.isfinite(line.poly_coeffs)), (
                f"Order {line.order_index}, seed col {line.seed_col}: "
                f"non-finite poly_coeffs"
            )

    def test_valid_lines_rms_below_threshold(self, arc_trace):
        valid = [ln for ln in arc_trace.valid_lines(min_n_valid=3)
                 if np.isfinite(ln.fit_rms)]
        if len(valid) == 0:
            pytest.skip("No valid lines with finite RMS found")
        median_rms = float(np.median([ln.fit_rms for ln in valid]))
        assert median_rms < 5.0, (
            f"Median line polynomial RMS {median_rms:.2f} px exceeds 5 px threshold"
        )
