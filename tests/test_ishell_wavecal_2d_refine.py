"""
Tests for the coefficient-surface refinement scaffold
(wavecal_2d_refine.py).

Coverage:
  - RefinedCoefficientSurface: construction and field access.
  - fit_refined_coefficient_surface on synthetic data:
      * returns correct type and fields,
      * recovers known wavelengths within tolerance,
      * order_smooth_coeffs has the right shape,
      * eval() and eval_array() produce consistent results,
      * raises ValueError for empty input, negative degrees, too few orders,
      * emits RuntimeWarning when order_smooth_degree must be reduced.
  - Smoke-test on real H1 calibration data (skipped when LFS files absent).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from pyspextool.instruments.ishell.arc_tracing import (
    ArcLineTraceResult,
    TracedArcLine,
)
from pyspextool.instruments.ishell.calibrations import (
    LineList,
    LineListEntry,
    WaveCalInfo,
)
from pyspextool.instruments.ishell.geometry import (
    OrderGeometry,
    OrderGeometrySet,
)
from pyspextool.instruments.ishell.wavecal_2d import (
    ProvisionalWavelengthMap,
    fit_provisional_wavelength_map,
)
from pyspextool.instruments.ishell.wavecal_2d_refine import (
    RefinedCoefficientSurface,
    fit_refined_coefficient_surface,
)
from pyspextool.instruments.ishell.io_utils import is_fits_file

# ---------------------------------------------------------------------------
# Path helpers for real H1 calibration data  (mirrors wavecal_2d_surface tests)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_H1_RAW_DIR = os.path.join(
    _REPO_ROOT, "data", "testdata", "ishell_h1_calibrations", "raw"
)
_LFS_MAGIC = b"version https://git-lfs"


def _is_real_fits(path: str) -> bool:
    """Return True if path is a real (non-LFS-pointer) FITS file."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
        return not head.startswith(_LFS_MAGIC)
    except OSError:
        return False


def _real_files(pattern: str) -> list[str]:
    """Return sorted real (non-LFS-pointer) file paths matching *pattern*."""
    if not os.path.isdir(_H1_RAW_DIR):
        return []
    return sorted(
        p
        for p in (
            os.path.join(_H1_RAW_DIR, f)
            for f in os.listdir(_H1_RAW_DIR)
            if pattern in f and is_fits_file(f)
        )
        if _is_real_fits(p)
    )


_H1_FLAT_FILES = _real_files("flat")
_H1_ARC_FILES = _real_files("arc")
_HAVE_H1_DATA = len(_H1_FLAT_FILES) >= 1 and len(_H1_ARC_FILES) >= 1

# Tolerance for per-order RMS and wavelength recovery checks on synthetic
# linear data.  The synthetic wavelength grids are perfectly linear in col,
# so a degree-1 polynomial should fit almost exactly; residuals arise only
# from the integer rounding of seed_col in _make_synthetic_traced_lines.
_SYNTHETIC_RMS_TOL_UM = 5e-3  # 5 nm — generous for a linear scaffold test


# ---------------------------------------------------------------------------
# Synthetic-data helpers (mirrors pattern from test_ishell_wavecal_2d_surface)
# ---------------------------------------------------------------------------

_NROWS = 200
_NCOLS = 800
_N_ORDERS = 4

# Four synthetic orders with linear wavelength-column grids.
_ORDER_PARAMS = [
    {
        "order_number": 311,
        "col_start": 100,
        "col_end": 700,
        "wav_start": 1.640,
        "wav_end": 1.690,
        "center_row": 50,
        "half_width": 18,
    },
    {
        "order_number": 315,
        "col_start": 100,
        "col_end": 700,
        "wav_start": 1.615,
        "wav_end": 1.665,
        "center_row": 100,
        "half_width": 18,
    },
    {
        "order_number": 320,
        "col_start": 100,
        "col_end": 700,
        "wav_start": 1.590,
        "wav_end": 1.640,
        "center_row": 150,
        "half_width": 18,
    },
    {
        "order_number": 325,
        "col_start": 100,
        "col_end": 700,
        "wav_start": 1.565,
        "wav_end": 1.615,
        "center_row": 195,
        "half_width": 18,
    },
]

# 4 known arc lines per order.
_KNOWN_LINES_UM = {
    311: [1.645, 1.655, 1.665, 1.675],
    315: [1.620, 1.630, 1.640, 1.650],
    320: [1.595, 1.605, 1.615, 1.625],
    325: [1.570, 1.580, 1.590, 1.600],
}


def _wav_to_col(wav_um: float, p: dict) -> float:
    frac = (wav_um - p["wav_start"]) / (p["wav_end"] - p["wav_start"])
    return p["col_start"] + frac * (p["col_end"] - p["col_start"])


def _col_to_wav(col: float, p: dict) -> float:
    frac = (col - p["col_start"]) / (p["col_end"] - p["col_start"])
    return p["wav_start"] + frac * (p["wav_end"] - p["wav_start"])


def _make_synthetic_geometry() -> OrderGeometrySet:
    geoms = []
    for i, p in enumerate(_ORDER_PARAMS):
        cr = float(p["center_row"])
        hw = float(p["half_width"])
        geoms.append(
            OrderGeometry(
                order=i,
                x_start=p["col_start"],
                x_end=p["col_end"],
                bottom_edge_coeffs=np.array([cr - hw, 0.0]),
                top_edge_coeffs=np.array([cr + hw, 0.0]),
            )
        )
    return OrderGeometrySet(mode="H1_test", geometries=geoms)


def _make_synthetic_traced_lines(
    geometry: OrderGeometrySet,
    seed: int = 42,
) -> list[TracedArcLine]:
    lines = []
    rng = np.random.default_rng(seed)
    for i, p in enumerate(_ORDER_PARAMS):
        for wav in _KNOWN_LINES_UM[p["order_number"]]:
            col = _wav_to_col(wav, p)
            tilt = float(rng.uniform(-0.02, 0.02))
            cr = float(p["center_row"])
            hw = float(p["half_width"])
            rows = np.arange(int(cr - hw), int(cr + hw) + 1, dtype=float)
            cols = col + tilt * (rows - cr) + rng.normal(0, 0.05, len(rows))
            poly_coeffs = np.polynomial.polynomial.polyfit(rows, cols, 1)
            fit_rms = float(
                np.sqrt(
                    np.mean(
                        (
                            cols
                            - np.polynomial.polynomial.polyval(rows, poly_coeffs)
                        )
                        ** 2
                    )
                )
            )
            lines.append(
                TracedArcLine(
                    order_index=i,
                    seed_col=int(round(col)),
                    trace_rows=rows,
                    trace_cols=cols,
                    poly_coeffs=poly_coeffs,
                    fit_rms=fit_rms,
                    peak_flux=3000.0,
                )
            )
    return lines


def _make_synthetic_arc_trace_result() -> ArcLineTraceResult:
    geom = _make_synthetic_geometry()
    lines = _make_synthetic_traced_lines(geom)
    return ArcLineTraceResult(
        mode="H1_test",
        arc_files=["synthetic_arc.fits"],
        poly_degree=1,
        geometry=geom,
        traced_lines=lines,
    )


def _make_synthetic_wavecalinfo() -> WaveCalInfo:
    n_orders = _N_ORDERS
    n_pixels = 700
    data = np.full((n_orders, 4, n_pixels), np.nan, dtype=float)
    xranges = np.zeros((n_orders, 2), dtype=int)
    for i, p in enumerate(_ORDER_PARAMS):
        x0 = p["col_start"]
        x1 = p["col_end"]
        n = x1 - x0 + 1
        cols = np.arange(n, dtype=float) + x0
        wavs = p["wav_start"] + (cols - x0) / (x1 - x0) * (
            p["wav_end"] - p["wav_start"]
        )
        data[i, 0, :n] = wavs
        xranges[i] = [x0, x1]
    return WaveCalInfo(
        mode="H1_test",
        n_orders=n_orders,
        orders=[p["order_number"] for p in _ORDER_PARAMS],
        resolving_power=75000.0,
        data=data,
        linelist_name="H1_lines.dat",
        wcal_type="2DXD",
        home_order=311,
        disp_degree=3,
        order_degree=2,
        xranges=xranges,
    )


def _make_synthetic_line_list() -> LineList:
    entries = []
    for p in _ORDER_PARAMS:
        for wav in _KNOWN_LINES_UM[p["order_number"]]:
            entries.append(
                LineListEntry(
                    order=p["order_number"],
                    wavelength_um=wav,
                    species="Ar I",
                    fit_window_angstrom=2.0,
                    fit_type="G",
                    fit_n_terms=5,
                )
            )
    return LineList(mode="H1_test", entries=entries)


def _make_synthetic_wav_map() -> ProvisionalWavelengthMap:
    trace = _make_synthetic_arc_trace_result()
    wci = _make_synthetic_wavecalinfo()
    ll = _make_synthetic_line_list()
    return fit_provisional_wavelength_map(trace, wci, ll, dispersion_degree=1)


# ===========================================================================
# 1. RefinedCoefficientSurface: construction
# ===========================================================================


class TestRefinedCoefficientSurfaceConstruction:
    """Unit tests for RefinedCoefficientSurface construction and field access."""

    def _make_minimal_result(self) -> RefinedCoefficientSurface:
        """Build a minimal RefinedCoefficientSurface for testing."""
        n_orders = 4
        disp_degree = 1
        order_smooth_degree = 1
        # order_smooth_coeffs: shape (disp_degree+1, order_smooth_degree+1)
        order_smooth_coeffs = np.zeros((disp_degree + 1, order_smooth_degree + 1))
        per_order_rms_um = np.zeros(n_orders)
        per_order_order_numbers = np.array([311.0, 315.0, 320.0, 325.0])
        smooth_rms_um = np.zeros(disp_degree + 1)
        return RefinedCoefficientSurface(
            mode="H1_test",
            disp_degree=disp_degree,
            order_smooth_degree=order_smooth_degree,
            order_ref=311.0,
            order_smooth_coeffs=order_smooth_coeffs,
            per_order_fits=[],
            per_order_rms_um=per_order_rms_um,
            per_order_order_numbers=per_order_order_numbers,
            smooth_rms_um=smooth_rms_um,
            n_orders_fit=n_orders,
            n_points_total=16,
            n_orders_skipped=0,
        )

    def test_construction(self):
        """RefinedCoefficientSurface can be constructed and fields are accessible."""
        res = self._make_minimal_result()
        assert res.mode == "H1_test"
        assert res.disp_degree == 1
        assert res.order_smooth_degree == 1
        assert res.order_ref == pytest.approx(311.0)
        assert res.n_orders_fit == 4
        assert res.n_points_total == 16
        assert res.n_orders_skipped == 0

    def test_order_smooth_coeffs_shape(self):
        """order_smooth_coeffs has shape (disp_degree+1, order_smooth_degree+1)."""
        res = self._make_minimal_result()
        assert res.order_smooth_coeffs.shape == (
            res.disp_degree + 1,
            res.order_smooth_degree + 1,
        )

    def test_smooth_rms_shape(self):
        """smooth_rms_um has length disp_degree+1."""
        res = self._make_minimal_result()
        assert res.smooth_rms_um.shape == (res.disp_degree + 1,)


# ===========================================================================
# 2. fit_refined_coefficient_surface on synthetic data
# ===========================================================================


class TestFitRefinedCoefficientSurfaceSynthetic:
    """fit_refined_coefficient_surface on fully controlled synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.wav_map = _make_synthetic_wav_map()

    def test_returns_refined_coefficient_surface(self):
        """The function returns a RefinedCoefficientSurface."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert isinstance(result, RefinedCoefficientSurface)

    def test_mode_preserved(self):
        """mode attribute matches wav_map.mode."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert result.mode == "H1_test"

    def test_n_orders_fit(self):
        """n_orders_fit equals the number of orders with enough data."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert result.n_orders_fit == _N_ORDERS

    def test_n_orders_skipped_zero(self):
        """n_orders_skipped is 0 when all orders have enough data."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert result.n_orders_skipped == 0

    def test_n_points_total(self):
        """n_points_total equals the total accepted matches."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert result.n_points_total == self.wav_map.n_total_accepted

    def test_order_smooth_coeffs_shape(self):
        """order_smooth_coeffs has shape (disp_degree+1, order_smooth_degree+1)."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=2, order_smooth_degree=1
        )
        assert result.order_smooth_coeffs.shape == (
            result.disp_degree + 1,
            result.order_smooth_degree + 1,
        )

    def test_per_order_rms_shape(self):
        """per_order_rms_um has length n_orders_fit."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert result.per_order_rms_um.shape == (result.n_orders_fit,)

    def test_smooth_rms_shape(self):
        """smooth_rms_um has length disp_degree+1."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert result.smooth_rms_um.shape == (result.disp_degree + 1,)

    def test_per_order_order_numbers_shape(self):
        """per_order_order_numbers has length n_orders_fit."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert result.per_order_order_numbers.shape == (result.n_orders_fit,)

    def test_order_ref_is_min_order(self):
        """order_ref equals the minimum order number in the fitted data."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        expected = min(p["order_number"] for p in _ORDER_PARAMS)
        assert result.order_ref == pytest.approx(float(expected))

    def test_per_order_fits_length(self):
        """per_order_fits list has n_orders_fit entries."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        assert len(result.per_order_fits) == result.n_orders_fit

    def test_per_order_fits_have_correct_order_numbers(self):
        """per_order_fits entries match the expected order numbers."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        expected = sorted(p["order_number"] for p in _ORDER_PARAMS)
        actual = sorted(f.order_number for f in result.per_order_fits)
        assert actual == expected

    def test_eval_consistent_with_eval_array(self):
        """eval() and eval_array() agree on the same inputs."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        order_nums, cols, _ = self.wav_map.collect_for_surface_fit()
        for i in range(min(4, len(order_nums))):
            scalar = result.eval(float(order_nums[i]), float(cols[i]))
            array_val = result.eval_array(
                np.array([order_nums[i]]), np.array([cols[i]])
            )[0]
            assert scalar == pytest.approx(array_val, rel=1e-10)

    def test_eval_array_shape(self):
        """eval_array returns an array with the right length."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        order_nums, cols, _ = self.wav_map.collect_for_surface_fit()
        preds = result.eval_array(order_nums, cols)
        assert preds.shape == (len(order_nums),)

    def test_eval_returns_float(self):
        """eval() returns a Python float."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        wav = result.eval(311.0, 400.0)
        assert isinstance(wav, float)
        assert np.isfinite(wav)

    def test_rms_small_for_linear_data(self):
        """Per-order RMS is small on synthetic linear-wavelength data."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        # Synthetic data is linear; degree-1 per-order fit should nearly
        # perfect (residuals only from integer-rounding of seed_col).
        assert result.per_order_rms_um.max() < _SYNTHETIC_RMS_TOL_UM, (
            f"Expected small per-order RMS; got max "
            f"{result.per_order_rms_um.max():.4e} µm"
        )

    def test_wavelength_in_h_band_range(self):
        """Predicted wavelengths at matched points fall in the H-band range."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        order_nums, cols, _ = self.wav_map.collect_for_surface_fit()
        preds = result.eval_array(order_nums, cols)
        assert preds.min() >= 1.5
        assert preds.max() <= 1.75

    def test_synthetic_data_recovery(self):
        """Eval surface recovers known wavelengths within 5 nm tolerance."""
        result = fit_refined_coefficient_surface(
            self.wav_map, disp_degree=1, order_smooth_degree=1
        )
        # Check each known arc line wavelength
        for p in _ORDER_PARAMS:
            for wav_true in _KNOWN_LINES_UM[p["order_number"]]:
                col = _wav_to_col(wav_true, p)
                wav_pred = result.eval(float(p["order_number"]), col)
                assert abs(wav_pred - wav_true) < _SYNTHETIC_RMS_TOL_UM, (
                    f"Order {p['order_number']}, col {col:.1f}: "
                    f"predicted {wav_pred:.6f} µm, expected {wav_true:.6f} µm"
                )


# ===========================================================================
# 3. Error handling
# ===========================================================================


class TestFitRefinedCoefficientSurfaceErrors:
    """Error and warning paths."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.wav_map = _make_synthetic_wav_map()

    def test_raises_on_negative_disp_degree(self):
        """ValueError if disp_degree < 0."""
        with pytest.raises(ValueError, match="disp_degree must be >= 0"):
            fit_refined_coefficient_surface(
                self.wav_map, disp_degree=-1, order_smooth_degree=1
            )

    def test_raises_on_negative_order_smooth_degree(self):
        """ValueError if order_smooth_degree < 0."""
        with pytest.raises(ValueError, match="order_smooth_degree must be >= 0"):
            fit_refined_coefficient_surface(
                self.wav_map, disp_degree=1, order_smooth_degree=-1
            )

    def test_raises_on_zero_min_lines(self):
        """ValueError if min_lines_per_order < 1."""
        with pytest.raises(ValueError, match="min_lines_per_order must be >= 1"):
            fit_refined_coefficient_surface(
                self.wav_map, min_lines_per_order=0
            )

    def test_raises_on_empty_wav_map(self):
        """ValueError if wav_map has no accepted matches."""
        geom = _make_synthetic_geometry()
        empty_map = ProvisionalWavelengthMap(
            mode="H1_test",
            order_solutions=[],
            geometry=geom,
            n_total_accepted=0,
            match_tol_um=0.005,
            dispersion_degree=1,
        )
        with pytest.raises(ValueError, match="no accepted arc-line matches"):
            fit_refined_coefficient_surface(empty_map)

    def test_warns_when_order_smooth_degree_reduced(self):
        """RuntimeWarning if order_smooth_degree must be reduced.

        Request order_smooth_degree >= n_orders, which forces reduction.
        """
        # 4 synthetic orders: requesting degree 5 should trigger a warning
        # and reduce to degree 3.
        with pytest.warns(RuntimeWarning, match="order_smooth_degree reduced"):
            result = fit_refined_coefficient_surface(
                self.wav_map,
                disp_degree=1,
                order_smooth_degree=5,
            )
        # The result should still be valid
        assert isinstance(result, RefinedCoefficientSurface)
        assert result.order_smooth_degree < 5

    def test_raises_when_too_few_orders(self):
        """ValueError if fewer than 2 orders have enough data after skipping."""
        from pyspextool.instruments.ishell.wavecal_2d import (
            ProvisionalLineMatch,
            ProvisionalOrderSolution,
        )

        # Build a wav_map with only one order having 2+ accepted matches
        p0 = _ORDER_PARAMS[0]
        wav0 = _KNOWN_LINES_UM[p0["order_number"]][0]
        col0 = _wav_to_col(wav0, p0)
        matches = [
            ProvisionalLineMatch(
                order_index=0,
                order_number=p0["order_number"],
                traced_line=None,
                centerline_col=col0 + i * 10.0,
                predicted_wavelength_um=wav0 + i * 0.01,
                reference_wavelength_um=wav0 + i * 0.01,
                match_residual_um=0.0,
                reference_species="Ar I",
            )
            for i in range(3)
        ]
        sol = ProvisionalOrderSolution(
            order_index=0,
            order_number=p0["order_number"],
            n_candidate=3,
            n_matched=3,
            n_accepted=3,
            wave_coeffs=np.array([1.5, 1e-4]),
            fit_degree=1,
            fit_rms_um=0.0,
            accepted_matches=matches,
            centerline_cols=np.array([m.centerline_col for m in matches]),
            reference_wavs=np.array([m.reference_wavelength_um for m in matches]),
        )
        geom = _make_synthetic_geometry()
        single_order_map = ProvisionalWavelengthMap(
            mode="H1_test",
            order_solutions=[sol],
            geometry=geom,
            n_total_accepted=3,
            match_tol_um=0.005,
            dispersion_degree=1,
        )
        with pytest.raises(ValueError, match="Only.*order"):
            fit_refined_coefficient_surface(
                single_order_map,
                disp_degree=1,
                order_smooth_degree=1,
            )

    def test_warns_on_skipped_order(self):
        """RuntimeWarning emitted for each order skipped due to too few matches."""
        # Request a high min_lines_per_order to force all orders to be skipped
        # except we need at least 2 to avoid a ValueError, so set
        # min_lines_per_order to exactly the count in each order (4) + 1 = 5
        # to force all orders skipped → should raise ValueError
        with pytest.raises(ValueError):
            with pytest.warns(RuntimeWarning):
                fit_refined_coefficient_surface(
                    self.wav_map,
                    disp_degree=1,
                    order_smooth_degree=1,
                    min_lines_per_order=5,  # all 4 orders have only 4 lines
                )


# ===========================================================================
# 4. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1RefinedCoefficientSurfaceSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset."""

    @pytest.fixture(scope="class")
    def h1_wav_map(self):
        """Per-order provisional wavelength map on real H1 data."""
        from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
        from pyspextool.instruments.ishell.calibrations import (
            read_line_list,
            read_wavecalinfo,
        )
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat

        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES,
            col_range=(650, 1550),
        )
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)
        wci = read_wavecalinfo("H1")
        ll = read_line_list("H1")
        return fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )

    def test_fit_refined_surface_runs(self, h1_wav_map):
        """fit_refined_coefficient_surface completes without error on H1 data."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        assert isinstance(result, RefinedCoefficientSurface)

    def test_mode_is_h1(self, h1_wav_map):
        """mode attribute is 'H1'."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        assert result.mode == "H1"

    def test_at_least_two_orders_fitted(self, h1_wav_map):
        """n_orders_fit >= 2."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        assert result.n_orders_fit >= 2

    def test_n_points_positive(self, h1_wav_map):
        """n_points_total > 0."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        assert result.n_points_total > 0

    def test_order_smooth_coeffs_shape(self, h1_wav_map):
        """order_smooth_coeffs has the right shape."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        assert result.order_smooth_coeffs.shape == (
            result.disp_degree + 1,
            result.order_smooth_degree + 1,
        )

    def test_eval_returns_h_band_wavelength(self, h1_wav_map):
        """eval() returns a finite wavelength in the H-band range."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        order_nums, cols, _ = h1_wav_map.collect_for_surface_fit()
        wav = result.eval(float(order_nums[0]), float(cols[0]))
        assert isinstance(wav, float)
        assert np.isfinite(wav)
        assert 1.35 <= wav <= 1.95

    def test_eval_array_shape(self, h1_wav_map):
        """eval_array returns an array with the correct length."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        order_nums, cols, _ = h1_wav_map.collect_for_surface_fit()
        preds = result.eval_array(order_nums, cols)
        assert preds.shape == (len(order_nums),)

    def test_per_order_rms_finite(self, h1_wav_map):
        """Per-order RMS values are all finite."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        assert np.all(np.isfinite(result.per_order_rms_um))

    def test_smooth_rms_finite(self, h1_wav_map):
        """Smooth coefficient RMS values are all finite."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        assert np.all(np.isfinite(result.smooth_rms_um))

    def test_eval_array_h_band_range(self, h1_wav_map):
        """eval_array predictions are in a plausible H-band range."""
        result = fit_refined_coefficient_surface(h1_wav_map, disp_degree=3)
        order_nums, cols, _ = h1_wav_map.collect_for_surface_fit()
        preds = result.eval_array(order_nums, cols)
        assert preds.min() >= 1.35
        assert preds.max() <= 1.95
