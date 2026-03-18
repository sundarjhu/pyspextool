"""
Tests for the provisional global wavelength-surface scaffold
(wavecal_2d_surface.py).

Coverage:
  - GlobalSurfaceFitResult: construction and field access.
  - fit_global_wavelength_surface on synthetic data:
      * returns correct type and fields,
      * recovers known wavelengths within tolerance,
      * coeffs_2d property has the right shape,
      * eval() and eval_array() produce consistent results,
      * raises ValueError for empty input, negative degrees, too few points,
      * emits RuntimeWarning for rank-deficient design matrix.
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
from pyspextool.instruments.ishell.wavecal_2d_surface import (
    GlobalSurfaceFitResult,
    fit_global_wavelength_surface,
)
from pyspextool.instruments.ishell.io_utils import is_fits_file

# ---------------------------------------------------------------------------
# Path helpers for real H1 calibration data
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


# ---------------------------------------------------------------------------
# Synthetic-data helpers (shared with test_ishell_wavecal_2d.py pattern)
# ---------------------------------------------------------------------------

_NROWS = 200
_NCOLS = 800
_N_ORDERS = 4

# Four synthetic orders with linear wavelength-column grids.
# Order numbers chosen to span a realistic echelle range.
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

# 4 known arc lines per order to give enough points for a global surface fit
_KNOWN_LINES_UM = {
    311: [1.645, 1.655, 1.665, 1.675],
    315: [1.620, 1.630, 1.640, 1.650],
    320: [1.595, 1.605, 1.615, 1.625],
    325: [1.570, 1.580, 1.590, 1.600],
}


def _wav_to_col(wav_um: float, p: dict) -> float:
    """Convert wavelength to column in a linear wavelength grid."""
    frac = (wav_um - p["wav_start"]) / (p["wav_end"] - p["wav_start"])
    return p["col_start"] + frac * (p["col_end"] - p["col_start"])


def _col_to_wav(col: float, p: dict) -> float:
    """Convert column to wavelength in a linear wavelength grid."""
    frac = (col - p["col_start"]) / (p["col_end"] - p["col_start"])
    return p["wav_start"] + frac * (p["wav_end"] - p["wav_start"])


def _make_synthetic_geometry() -> OrderGeometrySet:
    """Build an OrderGeometrySet for the synthetic scenario."""
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
    """Build TracedArcLine objects at the known synthetic arc-line positions."""
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
    """Return a complete ArcLineTraceResult for four synthetic orders."""
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
    """Build a WaveCalInfo with a linear plane-0 wavelength grid."""
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
    """Build a LineList with the known synthetic arc lines."""
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
    """Run the per-order fit on synthetic data and return the result."""
    trace = _make_synthetic_arc_trace_result()
    wci = _make_synthetic_wavecalinfo()
    ll = _make_synthetic_line_list()
    return fit_provisional_wavelength_map(
        trace, wci, ll, dispersion_degree=1
    )


# ===========================================================================
# 1. GlobalSurfaceFitResult: construction
# ===========================================================================


class TestGlobalSurfaceFitResultConstruction:
    """Unit tests for GlobalSurfaceFitResult construction and properties."""

    def _make_minimal_result(self) -> GlobalSurfaceFitResult:
        """Build a minimal GlobalSurfaceFitResult for testing."""
        n = 8
        coeffs = np.zeros(15)  # (4+1)*(2+1)=15
        residuals = np.zeros(n)
        return GlobalSurfaceFitResult(
            mode="H1_test",
            n_points=n,
            col_degree=4,
            order_degree=2,
            coeffs=coeffs,
            col_center=400.0,
            col_half=300.0,
            order_ref=311.0,
            rms_um=0.001,
            max_abs_residual_um=0.003,
            order_numbers=np.array([311.0] * n),
            cols=np.linspace(200, 600, n),
            wavs_true=np.linspace(1.60, 1.65, n),
            wavs_pred=np.linspace(1.60, 1.65, n),
            residuals_um=residuals,
            n_orders_used=1,
            rank=15,
            n_params=15,
        )

    def test_construction(self):
        """GlobalSurfaceFitResult can be constructed and fields are accessible."""
        res = self._make_minimal_result()
        assert res.mode == "H1_test"
        assert res.n_points == 8
        assert res.col_degree == 4
        assert res.order_degree == 2
        assert res.rms_um == pytest.approx(0.001)
        assert res.n_orders_used == 1
        assert res.rank == 15
        assert res.n_params == 15

    def test_coeffs_2d_shape(self):
        """coeffs_2d property has shape (col_degree+1, order_degree+1)."""
        res = self._make_minimal_result()
        assert res.coeffs_2d.shape == (5, 3)

    def test_coeffs_2d_values(self):
        """coeffs_2d values correspond to the flat coeffs array."""
        n = 8
        coeffs = np.arange(15, dtype=float)
        res = GlobalSurfaceFitResult(
            mode="H1_test",
            n_points=n,
            col_degree=4,
            order_degree=2,
            coeffs=coeffs,
            col_center=400.0,
            col_half=300.0,
            order_ref=311.0,
            rms_um=0.0,
            max_abs_residual_um=0.0,
            order_numbers=np.zeros(n),
            cols=np.zeros(n),
            wavs_true=np.zeros(n),
            wavs_pred=np.zeros(n),
            residuals_um=np.zeros(n),
            n_orders_used=1,
            rank=15,
            n_params=15,
        )
        expected = np.arange(15, dtype=float).reshape(5, 3)
        np.testing.assert_array_equal(res.coeffs_2d, expected)


# ===========================================================================
# 2. fit_global_wavelength_surface on synthetic data
# ===========================================================================


class TestFitGlobalWavelengthSurfaceSynthetic:
    """fit_global_wavelength_surface on fully controlled synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.wav_map = _make_synthetic_wav_map()

    def test_returns_global_surface_fit_result(self):
        """The function returns a GlobalSurfaceFitResult."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        assert isinstance(result, GlobalSurfaceFitResult)

    def test_mode_preserved(self):
        """mode attribute matches wav_map.mode."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        assert result.mode == "H1_test"

    def test_n_points(self):
        """n_points equals the total number of accepted matches."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        assert result.n_points == self.wav_map.n_total_accepted

    def test_n_orders_used(self):
        """n_orders_used equals the number of distinct order numbers."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        assert result.n_orders_used == _N_ORDERS

    def test_coeffs_shape(self):
        """coeffs has length (col_degree+1)*(order_degree+1)."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=2, order_degree=1
        )
        assert result.coeffs.shape == ((2 + 1) * (1 + 1),)

    def test_coeffs_2d_shape(self):
        """coeffs_2d has shape (col_degree+1, order_degree+1)."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=2, order_degree=1
        )
        assert result.coeffs_2d.shape == (3, 2)

    def test_normalization_params_stored(self):
        """col_center, col_half, and order_ref are stored correctly."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        order_nums, cols, _ = self.wav_map.collect_for_surface_fit()
        expected_col_center = 0.5 * (cols.min() + cols.max())
        expected_col_half = 0.5 * (cols.max() - cols.min())
        expected_order_ref = order_nums.min()

        assert result.col_center == pytest.approx(expected_col_center)
        assert result.col_half == pytest.approx(expected_col_half)
        assert result.order_ref == pytest.approx(expected_order_ref)

    def test_residuals_shape(self):
        """residuals_um has length equal to n_points."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        assert result.residuals_um.shape == (result.n_points,)

    def test_wavs_pred_shape(self):
        """wavs_pred has length equal to n_points."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        assert result.wavs_pred.shape == (result.n_points,)

    def test_residuals_consistent_with_wavs(self):
        """residuals_um = wavs_true - wavs_pred."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        np.testing.assert_allclose(
            result.residuals_um,
            result.wavs_true - result.wavs_pred,
            atol=1e-15,
        )

    def test_rms_consistent_with_residuals(self):
        """rms_um == sqrt(mean(residuals_um**2))."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        expected_rms = float(np.sqrt(np.mean(result.residuals_um**2)))
        assert result.rms_um == pytest.approx(expected_rms)

    def test_max_abs_residual_consistent(self):
        """max_abs_residual_um == max(|residuals_um|)."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        expected = float(np.max(np.abs(result.residuals_um)))
        assert result.max_abs_residual_um == pytest.approx(expected)

    def test_rms_small_for_linear_data(self):
        """RMS is small on synthetic linear-wavelength data with a degree-1 fit."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        # Synthetic data is linear in col × order; degree-1 in both should
        # fit almost perfectly (residuals only from integer-rounding of seed_col).
        assert result.rms_um < 5e-3, (
            f"Expected small RMS on linear synthetic data; got {result.rms_um:.4e} µm"
        )

    def test_eval_consistent_with_eval_array(self):
        """eval() and eval_array() agree on the same inputs."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        order_nums, cols, _ = self.wav_map.collect_for_surface_fit()
        # Evaluate first 4 points with both methods
        for i in range(min(4, result.n_points)):
            scalar = result.eval(order_nums[i], cols[i])
            array_val = result.eval_array(
                np.array([order_nums[i]]), np.array([cols[i]])
            )[0]
            assert scalar == pytest.approx(array_val)

    def test_eval_array_reproduces_wavs_pred(self):
        """eval_array on the input data reproduces wavs_pred."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        recomputed = result.eval_array(result.order_numbers, result.cols)
        np.testing.assert_allclose(recomputed, result.wavs_pred, atol=1e-12)

    def test_n_params_correct(self):
        """n_params == (col_degree+1)*(order_degree+1)."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=3, order_degree=2
        )
        assert result.n_params == (3 + 1) * (2 + 1)

    def test_rank_at_most_n_params(self):
        """rank is at most n_params."""
        result = fit_global_wavelength_surface(
            self.wav_map, col_degree=1, order_degree=1
        )
        assert result.rank <= result.n_params


# ===========================================================================
# 3. Error handling
# ===========================================================================


class TestFitGlobalWavelengthSurfaceErrors:
    """Error and warning paths."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.wav_map = _make_synthetic_wav_map()

    def test_raises_on_negative_col_degree(self):
        """ValueError if col_degree < 0."""
        with pytest.raises(ValueError, match="col_degree must be >= 0"):
            fit_global_wavelength_surface(
                self.wav_map, col_degree=-1, order_degree=1
            )

    def test_raises_on_negative_order_degree(self):
        """ValueError if order_degree < 0."""
        with pytest.raises(ValueError, match="order_degree must be >= 0"):
            fit_global_wavelength_surface(
                self.wav_map, col_degree=1, order_degree=-1
            )

    def test_raises_on_too_few_points(self):
        """ValueError if n_points < n_params."""
        # col_degree=10, order_degree=10 -> 121 params, but only 16 points
        with pytest.raises(ValueError, match="Not enough matched arc lines"):
            fit_global_wavelength_surface(
                self.wav_map, col_degree=10, order_degree=10
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
            fit_global_wavelength_surface(empty_map)

    def test_warns_on_rank_deficient_fit(self):
        """RuntimeWarning if design matrix is rank-deficient.

        Force rank-deficiency by creating a wav_map where all accepted
        matches share identical (order_number, col), so the design matrix
        has only one independent row.
        """
        from pyspextool.instruments.ishell.wavecal_2d import (
            ProvisionalLineMatch,
            ProvisionalOrderSolution,
        )

        # Build a wav_map with 10 identical points — the design matrix
        # will be rank-1 but we request a 4-parameter model.
        n = 10
        col = 400.0
        wav = 1.65
        matches = [
            ProvisionalLineMatch(
                order_index=0,
                order_number=311,
                traced_line=None,
                centerline_col=col,
                predicted_wavelength_um=wav,
                reference_wavelength_um=wav,
                match_residual_um=0.0,
                reference_species="Ar I",
            )
            for _ in range(n)
        ]
        sol = ProvisionalOrderSolution(
            order_index=0,
            order_number=311,
            n_candidate=n,
            n_matched=n,
            n_accepted=n,
            wave_coeffs=np.array([1.5, 1e-4]),
            fit_degree=1,
            fit_rms_um=0.0,
            accepted_matches=matches,
            centerline_cols=np.full(n, col),
            reference_wavs=np.full(n, wav),
        )
        geom = _make_synthetic_geometry()
        degenerate_map = ProvisionalWavelengthMap(
            mode="H1_test",
            order_solutions=[sol],
            geometry=geom,
            n_total_accepted=n,
            match_tol_um=0.005,
            dispersion_degree=1,
        )
        with pytest.warns(RuntimeWarning, match="rank-deficient"):
            fit_global_wavelength_surface(
                degenerate_map, col_degree=1, order_degree=1
            )


# ===========================================================================
# 4. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1GlobalSurfaceSmokeTest:
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

    def test_fit_global_surface_runs(self, h1_wav_map):
        """fit_global_wavelength_surface completes without error on real H1 data."""
        result = fit_global_wavelength_surface(h1_wav_map)
        assert isinstance(result, GlobalSurfaceFitResult)

    def test_mode_is_h1(self, h1_wav_map):
        """mode attribute is 'H1'."""
        result = fit_global_wavelength_surface(h1_wav_map)
        assert result.mode == "H1"

    def test_n_points_consistent(self, h1_wav_map):
        """n_points equals h1_wav_map.n_total_accepted."""
        result = fit_global_wavelength_surface(h1_wav_map)
        assert result.n_points == h1_wav_map.n_total_accepted

    def test_at_least_one_order_used(self, h1_wav_map):
        """n_orders_used >= 1."""
        result = fit_global_wavelength_surface(h1_wav_map)
        assert result.n_orders_used >= 1

    def test_wavelength_range_is_h_band(self, h1_wav_map):
        """All predicted wavelengths fall within the H-band range."""
        result = fit_global_wavelength_surface(h1_wav_map)
        # Evaluate the surface at the fit points
        assert result.wavs_pred.min() >= 1.35
        assert result.wavs_pred.max() <= 1.95

    def test_rms_finite_and_positive(self, h1_wav_map):
        """RMS is finite and positive."""
        result = fit_global_wavelength_surface(h1_wav_map)
        assert np.isfinite(result.rms_um)
        assert result.rms_um > 0.0

    def test_eval_returns_float(self, h1_wav_map):
        """eval() returns a scalar float."""
        result = fit_global_wavelength_surface(h1_wav_map)
        # Use the first matched order/column pair
        order_nums, cols, _ = h1_wav_map.collect_for_surface_fit()
        wav = result.eval(float(order_nums[0]), float(cols[0]))
        assert isinstance(wav, float)
        assert np.isfinite(wav)

    def test_eval_array_shape(self, h1_wav_map):
        """eval_array returns an array with the right length."""
        result = fit_global_wavelength_surface(h1_wav_map)
        order_nums, cols, _ = h1_wav_map.collect_for_surface_fit()
        preds = result.eval_array(order_nums, cols)
        assert preds.shape == (len(order_nums),)

    def test_residuals_h_band_scale(self, h1_wav_map):
        """Residuals are not astronomically large (sanity check only)."""
        result = fit_global_wavelength_surface(h1_wav_map)
        # A provisional surface may have multi-nm residuals; we just check
        # they are not > 100 nm (which would indicate a catastrophic failure).
        assert result.max_abs_residual_um < 0.1, (
            f"Max residual {result.max_abs_residual_um * 1e3:.1f} nm is "
            "suspiciously large for an H-band surface fit."
        )

    def test_lower_degree_still_runs(self, h1_wav_map):
        """A lower-degree surface fit also completes without error."""
        result = fit_global_wavelength_surface(
            h1_wav_map, col_degree=2, order_degree=1
        )
        assert isinstance(result, GlobalSurfaceFitResult)
        assert result.n_params == (2 + 1) * (1 + 1)
