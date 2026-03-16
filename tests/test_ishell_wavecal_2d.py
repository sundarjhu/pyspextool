"""
Tests for the provisional wavelength-mapping scaffold (wavecal_2d.py).

Coverage:
  - ProvisionalLineMatch: construction and field access.
  - ProvisionalOrderSolution: construction, wave_coeffs fit, fit_rms.
  - ProvisionalWavelengthMap: construction, helpers, collect_for_surface_fit,
    to_geometry_set.
  - fit_provisional_wavelength_map on synthetic data:
      * returns correct types,
      * recovers known wavelengths within tolerance,
      * handles order-count mismatch with a warning,
      * handles no reference lines gracefully,
      * handles empty traced-line lists gracefully.
  - Smoke-test on real H1 calibration data (skipped when LFS files absent).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.arc_tracing import (
    ArcLineTraceResult,
    TracedArcLine,
)
from pyspextool.instruments.ishell.calibrations import (
    LineList,
    LineListEntry,
    WaveCalInfo,
    read_line_list,
    read_wavecalinfo,
)
from pyspextool.instruments.ishell.geometry import (
    OrderGeometry,
    OrderGeometrySet,
)
from pyspextool.instruments.ishell.wavecal_2d import (
    ProvisionalLineMatch,
    ProvisionalOrderSolution,
    ProvisionalWavelengthMap,
    fit_provisional_wavelength_map,
)

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
            if pattern in f and f.endswith(".fits")
        )
        if _is_real_fits(p)
    )


_H1_FLAT_FILES = _real_files("flat")
_H1_ARC_FILES = _real_files("arc")
_HAVE_H1_DATA = len(_H1_FLAT_FILES) >= 1 and len(_H1_ARC_FILES) >= 1


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

# Synthetic detector shape (small, fast)
_NROWS = 200
_NCOLS = 800

# Two synthetic orders, each with 3 known arc lines
_N_ORDERS = 2

# Synthetic wavelength grids: linear (for easy inversion)
# Order 0: 1.60–1.65 µm over columns 100–700
# Order 1: 1.55–1.60 µm over columns 100–700
_ORDER_PARAMS = [
    {
        "order_number": 311,
        "col_start": 100,
        "col_end": 700,
        "wav_start": 1.60,
        "wav_end": 1.65,
        "center_row": 80,
        "half_width": 20,
    },
    {
        "order_number": 312,
        "col_start": 100,
        "col_end": 700,
        "wav_start": 1.55,
        "wav_end": 1.60,
        "center_row": 140,
        "half_width": 20,
    },
]

# Known arc lines at specific wavelengths
# (placed well within the synthetic wavelength range)
_KNOWN_LINES_UM = {
    311: [1.610, 1.625, 1.640],
    312: [1.555, 1.570, 1.585],
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
                order=i,  # placeholder 0-based index
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
            # Straight-line trace with a tiny tilt
            tilt = float(rng.uniform(-0.02, 0.02))
            cr = float(p["center_row"])
            hw = float(p["half_width"])
            rows = np.arange(int(cr - hw), int(cr + hw) + 1, dtype=float)
            cols = col + tilt * (rows - cr) + rng.normal(0, 0.05, len(rows))
            poly_coeffs = np.polynomial.polynomial.polyfit(rows, cols, 1)
            fit_rms = float(np.sqrt(np.mean((cols - np.polynomial.polynomial.polyval(rows, poly_coeffs)) ** 2)))
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
    """Return a complete ArcLineTraceResult for two synthetic orders."""
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
    n_pixels = 700  # enough to cover columns 100–700 (we map index 0→col 100)
    data = np.full((n_orders, 4, n_pixels), np.nan, dtype=float)
    xranges = np.zeros((n_orders, 2), dtype=int)

    for i, p in enumerate(_ORDER_PARAMS):
        x0 = p["col_start"]
        x1 = p["col_end"]
        n = x1 - x0 + 1
        cols = np.arange(n, dtype=float) + x0
        # Linear wavelength grid
        wavs = p["wav_start"] + (cols - x0) / (x1 - x0) * (p["wav_end"] - p["wav_start"])
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


# ===========================================================================
# 1. Dataclass API
# ===========================================================================


class TestProvisionalLineMatch:
    """Unit tests for ProvisionalLineMatch construction."""

    def test_construction_and_fields(self):
        """ProvisionalLineMatch can be constructed and fields are accessible."""
        geom = _make_synthetic_geometry()
        lines = _make_synthetic_traced_lines(geom)
        m = ProvisionalLineMatch(
            order_index=0,
            order_number=311,
            traced_line=lines[0],
            centerline_col=300.0,
            predicted_wavelength_um=1.625,
            reference_wavelength_um=1.625,
            match_residual_um=0.0,
            reference_species="Ar I",
        )
        assert m.order_index == 0
        assert m.order_number == 311
        assert m.centerline_col == 300.0
        assert m.reference_wavelength_um == pytest.approx(1.625)
        assert m.match_residual_um == pytest.approx(0.0)
        assert m.reference_species == "Ar I"


class TestProvisionalOrderSolution:
    """Unit tests for ProvisionalOrderSolution construction."""

    def test_no_fit_solution(self):
        """ProvisionalOrderSolution with wave_coeffs=None is valid."""
        sol = ProvisionalOrderSolution(
            order_index=0,
            order_number=311,
            n_candidate=5,
            n_matched=1,
            n_accepted=1,
            wave_coeffs=None,
            fit_degree=3,
            fit_rms_um=float("nan"),
        )
        assert sol.wave_coeffs is None
        assert np.isnan(sol.fit_rms_um)
        assert sol.n_accepted == 1

    def test_valid_fit_solution(self):
        """ProvisionalOrderSolution with valid coefficients stores them."""
        coeffs = np.array([1.5, 1e-4, 1e-8, -1e-12])
        sol = ProvisionalOrderSolution(
            order_index=0,
            order_number=311,
            n_candidate=5,
            n_matched=4,
            n_accepted=4,
            wave_coeffs=coeffs,
            fit_degree=3,
            fit_rms_um=1e-5,
        )
        assert sol.wave_coeffs is not None
        assert len(sol.wave_coeffs) == 4
        assert sol.fit_rms_um == pytest.approx(1e-5)


# ===========================================================================
# 2. fit_provisional_wavelength_map on synthetic data
# ===========================================================================


class TestFitProvisionalWavelengthMapSynthetic:
    """fit_provisional_wavelength_map on fully controlled synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.trace = _make_synthetic_arc_trace_result()
        self.wci = _make_synthetic_wavecalinfo()
        self.ll = _make_synthetic_line_list()

    def test_returns_provisional_wavelength_map(self):
        """The function returns a ProvisionalWavelengthMap."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        assert isinstance(result, ProvisionalWavelengthMap)

    def test_mode_preserved(self):
        """mode attribute matches trace_result.mode."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        assert result.mode == "H1_test"

    def test_n_orders(self):
        """n_orders matches the number of traced orders."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        assert result.n_orders == _N_ORDERS

    def test_all_orders_solved(self):
        """All orders have a polynomial fit when enough lines are available."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1,
            min_lines_per_order=2,
        )
        # Each synthetic order has 3 known lines, so all should be solved
        assert result.n_solved_orders == _N_ORDERS

    def test_total_accepted_matches(self):
        """Total accepted matches equals 3 lines × 2 orders = 6."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        expected = sum(
            len(wavs) for wavs in _KNOWN_LINES_UM.values()
        )
        assert result.n_total_accepted == expected

    def test_wave_coeffs_shape(self):
        """wave_coeffs has the right length for the requested degree."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        for sol in result.solved_orders:
            assert sol.wave_coeffs is not None
            # degree 1 → 2 coefficients
            assert len(sol.wave_coeffs) == 2

    def test_polynomial_recovers_known_wavelengths(self):
        """The fitted polynomial evaluates close to known wavelengths."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        for sol in result.solved_orders:
            p = _ORDER_PARAMS[sol.order_index]
            for wav_true in _KNOWN_LINES_UM[p["order_number"]]:
                col = _wav_to_col(wav_true, p)
                wav_pred = float(
                    np.polynomial.polynomial.polyval(col, sol.wave_coeffs)
                )
                # Should agree to within 2× the match tolerance (0.005 µm)
                assert abs(wav_pred - wav_true) < 0.010, (
                    f"Order {sol.order_number}: predicted {wav_pred:.6f} µm "
                    f"vs true {wav_true:.6f} µm at col {col:.1f}"
                )

    def test_fit_rms_is_small(self):
        """Fit RMS is well below the match tolerance on clean synthetic data."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        for sol in result.solved_orders:
            assert sol.fit_rms_um < 1e-3, (
                f"Order {sol.order_number}: fit_rms_um={sol.fit_rms_um:.2e}"
            )

    def test_collect_for_surface_fit(self):
        """collect_for_surface_fit returns correct array shapes and values."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        order_nums, cols, wavs = result.collect_for_surface_fit()
        n_total = sum(len(v) for v in _KNOWN_LINES_UM.values())
        assert len(order_nums) == n_total
        assert len(cols) == n_total
        assert len(wavs) == n_total
        # All order numbers should be in the expected set
        assert set(order_nums.astype(int)).issubset({311, 312})
        # Wavelengths within band range
        assert wavs.min() >= 1.55
        assert wavs.max() <= 1.65

    def test_to_geometry_set(self):
        """to_geometry_set returns an OrderGeometrySet with wave_coeffs populated."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll, dispersion_degree=1
        )
        geom_set = result.to_geometry_set()
        assert isinstance(geom_set, OrderGeometrySet)
        assert geom_set.n_orders == _N_ORDERS
        for geom in geom_set.geometries:
            assert geom.wave_coeffs is not None

    def test_order_count_mismatch_warning(self):
        """A warning is emitted when the order counts differ."""
        # Modify wavecalinfo to have one extra order
        wci_extra = WaveCalInfo(
            mode="H1_test",
            n_orders=3,
            orders=[311, 312, 313],
            resolving_power=75000.0,
            data=np.full((3, 4, 700), np.nan),
            linelist_name="H1_lines.dat",
            wcal_type="2DXD",
            home_order=311,
            disp_degree=3,
            order_degree=2,
            xranges=np.array([[100, 700], [100, 700], [100, 700]]),
        )
        with pytest.warns(RuntimeWarning, match="2 orders but wavecalinfo has 3"):
            result = fit_provisional_wavelength_map(
                self.trace, wci_extra, self.ll, dispersion_degree=1
            )
        assert result.order_count_mismatch is True
        # Should still process the minimum (2) orders
        assert result.n_orders == 2

    def test_no_reference_lines_gives_no_matches(self):
        """Orders with no reference lines in the line list get no matches."""
        # Empty line list
        empty_ll = LineList(mode="H1_test", entries=[])
        with pytest.warns(RuntimeWarning):
            result = fit_provisional_wavelength_map(
                self.trace, self.wci, empty_ll,
                dispersion_degree=1, min_lines_per_order=1,
            )
        assert result.n_total_accepted == 0
        for sol in result.order_solutions:
            assert sol.wave_coeffs is None

    def test_high_tolerance_accepts_all(self):
        """A very large match_tol_um accepts all plausible candidates."""
        result = fit_provisional_wavelength_map(
            self.trace, self.wci, self.ll,
            dispersion_degree=1, match_tol_um=0.5
        )
        # At least as many accepted as with default tolerance
        assert result.n_total_accepted >= 6

    def test_tight_tolerance_rejects_all(self):
        """A tolerance smaller than any residual rejects all matches.

        To guarantee non-zero residuals, we use a WaveCalInfo whose coarse
        wavelength grid is shifted from the reference wavelengths by a known
        amount (0.050 µm) and set match_tol_um=0.010 µm (smaller than offset).
        """
        offset = 0.050  # µm: all coarse predictions are this far from true
        n_pixels = 700
        data = np.full((_N_ORDERS, 4, n_pixels), np.nan, dtype=float)
        xranges = np.zeros((_N_ORDERS, 2), dtype=int)
        for i, p in enumerate(_ORDER_PARAMS):
            x0, x1 = p["col_start"], p["col_end"]
            n = x1 - x0 + 1
            cols = np.arange(n, dtype=float) + x0
            wavs = (
                p["wav_start"]
                + (cols - x0) / (x1 - x0) * (p["wav_end"] - p["wav_start"])
                + offset
            )
            data[i, 0, :n] = wavs
            xranges[i] = [x0, x1]
        wci_shifted = WaveCalInfo(
            mode="H1_test",
            n_orders=_N_ORDERS,
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
        with pytest.warns(RuntimeWarning):
            result = fit_provisional_wavelength_map(
                self.trace, wci_shifted, self.ll,
                dispersion_degree=1, match_tol_um=0.010, min_lines_per_order=1,
            )
        # Coarse predictions are offset by 0.050 µm > match_tol_um=0.010 µm
        assert result.n_total_accepted == 0

    def test_empty_traced_lines_order(self):
        """An order with no traced lines produces a zero-accepted solution."""
        # Only include lines from order 0 (indices 0, 1, 2)
        partial_trace = ArcLineTraceResult(
            mode="H1_test",
            arc_files=["synthetic_arc.fits"],
            poly_degree=1,
            geometry=self.trace.geometry,
            traced_lines=self.trace.lines_for_order(0),  # only order 0
        )
        with pytest.warns(RuntimeWarning):
            result = fit_provisional_wavelength_map(
                partial_trace, self.wci, self.ll,
                dispersion_degree=1, min_lines_per_order=2,
            )
        # Order 0 should be solved; order 1 has no traced lines
        sol_0 = result.order_solutions[0]
        sol_1 = result.order_solutions[1]
        assert sol_0.wave_coeffs is not None
        assert sol_1.wave_coeffs is None
        assert sol_1.n_candidate == 0

    def test_raises_if_xranges_none(self):
        """ValueError if wavecalinfo.xranges is None."""
        wci_no_xr = WaveCalInfo(
            mode="H1_test",
            n_orders=2,
            orders=[311, 312],
            resolving_power=75000.0,
            data=np.zeros((2, 4, 100)),
            linelist_name="H1_lines.dat",
            wcal_type="2DXD",
            home_order=311,
            disp_degree=3,
            order_degree=2,
            xranges=None,
        )
        with pytest.raises(ValueError, match="xranges is None"):
            fit_provisional_wavelength_map(self.trace, wci_no_xr, self.ll)

    def test_raises_if_no_orders(self):
        """ValueError if trace_result contains no orders."""
        empty_trace = ArcLineTraceResult(
            mode="H1_test",
            arc_files=["synthetic_arc.fits"],
            poly_degree=1,
            geometry=OrderGeometrySet(mode="H1_test", geometries=[]),
            traced_lines=[],
        )
        with pytest.raises(ValueError, match="no orders"):
            fit_provisional_wavelength_map(empty_trace, self.wci, self.ll)


# ===========================================================================
# 3. ProvisionalWavelengthMap helper methods
# ===========================================================================


class TestProvisionalWavelengthMapHelpers:
    """Tests for ProvisionalWavelengthMap methods independent of fitting."""

    def _make_map_with_solutions(
        self, n_accepted_per_order: list[int]
    ) -> ProvisionalWavelengthMap:
        geom = _make_synthetic_geometry()
        sols = []
        for i, n_acc in enumerate(n_accepted_per_order):
            order_num = _ORDER_PARAMS[i]["order_number"]
            if n_acc >= 2:
                cols = np.linspace(200, 600, n_acc)
                wavs = 1.6 + cols * 1e-4
                coeffs = np.polynomial.polynomial.polyfit(cols, wavs, 1)
            else:
                coeffs = None
            matches = []
            for j in range(n_acc):
                col = 200.0 + j * 50.0
                wav = 1.6 + col * 1e-4
                matches.append(
                    ProvisionalLineMatch(
                        order_index=i,
                        order_number=order_num,
                        traced_line=None,  # not needed for these tests
                        centerline_col=col,
                        predicted_wavelength_um=wav,
                        reference_wavelength_um=wav,
                        match_residual_um=0.0,
                        reference_species="Ar I",
                    )
                )
            sols.append(
                ProvisionalOrderSolution(
                    order_index=i,
                    order_number=order_num,
                    n_candidate=n_acc + 1,
                    n_matched=n_acc,
                    n_accepted=n_acc,
                    wave_coeffs=coeffs,
                    fit_degree=1,
                    fit_rms_um=1e-5 if coeffs is not None else float("nan"),
                    accepted_matches=matches,
                    centerline_cols=np.array([m.centerline_col for m in matches]),
                    reference_wavs=np.array(
                        [m.reference_wavelength_um for m in matches]
                    ),
                )
            )
        return ProvisionalWavelengthMap(
            mode="H1_test",
            order_solutions=sols,
            geometry=OrderGeometrySet(mode="H1_test", geometries=geom.geometries),
            n_total_accepted=sum(n_accepted_per_order),
            match_tol_um=0.005,
            dispersion_degree=1,
        )

    def test_n_orders(self):
        wav_map = self._make_map_with_solutions([3, 3])
        assert wav_map.n_orders == 2

    def test_solved_orders_all_solved(self):
        wav_map = self._make_map_with_solutions([3, 3])
        assert wav_map.n_solved_orders == 2
        assert len(wav_map.solved_orders) == 2

    def test_solved_orders_partial(self):
        """solved_orders excludes orders with no fit."""
        wav_map = self._make_map_with_solutions([3, 0])
        assert wav_map.n_solved_orders == 1
        assert wav_map.solved_orders[0].order_number == 311

    def test_collect_for_surface_fit_shapes(self):
        wav_map = self._make_map_with_solutions([3, 3])
        order_nums, cols, wavs = wav_map.collect_for_surface_fit()
        assert len(order_nums) == 6
        assert len(cols) == 6
        assert len(wavs) == 6

    def test_collect_for_surface_fit_empty(self):
        wav_map = self._make_map_with_solutions([0, 0])
        order_nums, cols, wavs = wav_map.collect_for_surface_fit()
        assert len(order_nums) == 0
        assert len(cols) == 0
        assert len(wavs) == 0

    def test_to_geometry_set_has_wave_coeffs(self):
        wav_map = self._make_map_with_solutions([3, 3])
        geom_set = wav_map.to_geometry_set()
        for g in geom_set.geometries:
            assert g.wave_coeffs is not None

    def test_to_geometry_set_fallback_for_unsolved_order(self):
        """to_geometry_set uses fallback_geometry for unsolved orders."""
        wav_map = self._make_map_with_solutions([3, 0])
        # Build a fallback geometry set with known coefficients
        fallback_geom = _make_synthetic_geometry()
        fallback_coeffs = np.array([1.5, 1e-4])
        for g in fallback_geom.geometries:
            object.__setattr__(g, "wave_coeffs", fallback_coeffs.copy())

        geom_set = wav_map.to_geometry_set(fallback_geometry=fallback_geom)
        # Order 0: solved, should use provisional coefficients
        assert geom_set.geometries[0].wave_coeffs is not None
        # Order 1: not solved, should use fallback
        assert geom_set.geometries[1].wave_coeffs is not None


# ===========================================================================
# 4. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1SmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset."""

    @pytest.fixture(scope="class")
    def h1_trace(self):
        """Flat-order trace and arc-line trace on real H1 data."""
        from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat

        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES,
            col_range=(650, 1550),
        )
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)
        return arc_result

    @pytest.fixture(scope="class")
    def h1_wavecalinfo(self):
        return read_wavecalinfo("H1")

    @pytest.fixture(scope="class")
    def h1_line_list(self):
        return read_line_list("H1")

    def test_fit_provisional_wavelength_map_runs(
        self, h1_trace, h1_wavecalinfo, h1_line_list
    ):
        """fit_provisional_wavelength_map completes without error on real H1 data."""
        result = fit_provisional_wavelength_map(
            h1_trace, h1_wavecalinfo, h1_line_list, dispersion_degree=3
        )
        assert isinstance(result, ProvisionalWavelengthMap)

    def test_at_least_one_order_solved(
        self, h1_trace, h1_wavecalinfo, h1_line_list
    ):
        """At least one order has a valid polynomial fit."""
        result = fit_provisional_wavelength_map(
            h1_trace, h1_wavecalinfo, h1_line_list, dispersion_degree=3
        )
        assert result.n_solved_orders >= 1, (
            f"Expected at least 1 solved order; got 0 out of {result.n_orders}. "
            f"Total accepted matches: {result.n_total_accepted}."
        )

    def test_at_least_one_accepted_match(
        self, h1_trace, h1_wavecalinfo, h1_line_list
    ):
        """At least one arc line is matched to a reference wavelength."""
        result = fit_provisional_wavelength_map(
            h1_trace, h1_wavecalinfo, h1_line_list, dispersion_degree=3
        )
        assert result.n_total_accepted >= 1

    def test_collect_for_surface_fit_returns_arrays(
        self, h1_trace, h1_wavecalinfo, h1_line_list
    ):
        """collect_for_surface_fit returns three equal-length arrays."""
        result = fit_provisional_wavelength_map(
            h1_trace, h1_wavecalinfo, h1_line_list, dispersion_degree=3
        )
        order_nums, cols, wavs = result.collect_for_surface_fit()
        assert len(order_nums) == len(cols) == len(wavs)
        assert len(order_nums) == result.n_total_accepted

    def test_wavelength_range_is_h_band(
        self, h1_trace, h1_wavecalinfo, h1_line_list
    ):
        """All matched wavelengths fall within the H-band range (~1.49–1.80 µm)."""
        result = fit_provisional_wavelength_map(
            h1_trace, h1_wavecalinfo, h1_line_list, dispersion_degree=3
        )
        _, _, wavs = result.collect_for_surface_fit()
        if len(wavs) > 0:
            assert wavs.min() >= 1.40, f"Min wavelength {wavs.min():.4f} µm is below H-band."
            assert wavs.max() <= 1.90, f"Max wavelength {wavs.max():.4f} µm is above H-band."

    def test_mode_is_h1(self, h1_trace, h1_wavecalinfo, h1_line_list):
        """mode attribute is 'H1'."""
        result = fit_provisional_wavelength_map(
            h1_trace, h1_wavecalinfo, h1_line_list, dispersion_degree=3
        )
        assert result.mode == "H1"

    def test_to_geometry_set_produces_wavecalibrated_geometry(
        self, h1_trace, h1_wavecalinfo, h1_line_list
    ):
        """to_geometry_set produces an OrderGeometrySet with wave_coeffs."""
        result = fit_provisional_wavelength_map(
            h1_trace, h1_wavecalinfo, h1_line_list, dispersion_degree=3
        )
        geom_set = result.to_geometry_set()
        assert isinstance(geom_set, OrderGeometrySet)
        # At minimum, solved orders should have coefficients
        for sol in result.solved_orders:
            g = geom_set.geometries[sol.order_index]
            assert g.wave_coeffs is not None
