"""
Tests for the provisional rectification-index scaffold
(rectification_indices.py).

Coverage:
  - RectificationIndexOrder: construction and field access.
  - RectificationIndexSet: construction, get_order, orders, n_orders.
  - build_rectification_indices on synthetic data:
      * returns correct type and fields,
      * array shapes are consistent,
      * wavelength axis is monotone,
      * spatial axis is uniformly spaced in [0, 1],
      * src_cols are within the order's column range (with tolerance),
      * src_rows are within the order's row range (with tolerance),
      * output_shape property is consistent,
      * mode is propagated from geometry,
      * get_order round-trips correctly.
  - Error handling:
      * empty geometry → ValueError,
      * mismatched order counts → ValueError,
      * n_spectral < 2 → ValueError,
      * n_spatial < 2 → ValueError,
      * n_col_samples < n_spectral → ValueError,
      * non-monotone surface → RuntimeWarning.
  - Smoke test on real H1 calibration data (skipped when LFS files absent).
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
from pyspextool.instruments.ishell.rectification_indices import (
    RectificationIndexOrder,
    RectificationIndexSet,
    build_rectification_indices,
)
from pyspextool.instruments.ishell.wavecal_2d import (
    ProvisionalWavelengthMap,
    fit_provisional_wavelength_map,
)
from pyspextool.instruments.ishell.wavecal_2d_refine import (
    RefinedCoefficientSurface,
    fit_refined_coefficient_surface,
)

# ---------------------------------------------------------------------------
# Path helpers for real H1 calibration data  (mirrors wavecal_2d_refine tests)
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
# Synthetic-data helpers  (mirrors test_ishell_wavecal_2d_refine)
# ---------------------------------------------------------------------------

_NROWS = 200
_NCOLS = 800
_N_ORDERS = 4

# Four synthetic orders with linear wavelength–column grids.
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

_KNOWN_LINES_UM = {
    311: [1.645, 1.655, 1.665, 1.675],
    315: [1.620, 1.630, 1.640, 1.650],
    320: [1.595, 1.605, 1.615, 1.625],
    325: [1.570, 1.580, 1.590, 1.600],
}


def _wav_to_col(wav_um: float, p: dict) -> float:
    frac = (wav_um - p["wav_start"]) / (p["wav_end"] - p["wav_start"])
    return p["col_start"] + frac * (p["col_end"] - p["col_start"])


def _make_synthetic_geometry() -> OrderGeometrySet:
    geoms = []
    for p in _ORDER_PARAMS:
        cr = float(p["center_row"])
        hw = float(p["half_width"])
        geoms.append(
            OrderGeometry(
                order=p["order_number"],  # use actual echelle order number
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
    data = np.full((n_orders, 4, 700), np.nan, dtype=float)
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


def _make_synthetic_surface() -> RefinedCoefficientSurface:
    wav_map = _make_synthetic_wav_map()
    return fit_refined_coefficient_surface(
        wav_map, disp_degree=1, order_smooth_degree=1
    )


# ===========================================================================
# 1. RectificationIndexOrder: construction
# ===========================================================================


class TestRectificationIndexOrderConstruction:
    """Unit tests for RectificationIndexOrder construction and field access."""

    def _make_minimal(self) -> RectificationIndexOrder:
        n_spectral = 32
        n_spatial = 16
        return RectificationIndexOrder(
            order=311,
            order_index=0,
            output_wavelengths_um=np.linspace(1.64, 1.69, n_spectral),
            output_spatial_frac=np.linspace(0.0, 1.0, n_spatial),
            src_cols=np.linspace(100.0, 700.0, n_spectral),
            src_rows=np.zeros((n_spatial, n_spectral)),
        )

    def test_construction(self):
        """RectificationIndexOrder can be constructed and fields are accessible."""
        rio = self._make_minimal()
        assert rio.order == 311
        assert rio.order_index == 0

    def test_n_spectral(self):
        """n_spectral matches the length of output_wavelengths_um."""
        rio = self._make_minimal()
        assert rio.n_spectral == 32

    def test_n_spatial(self):
        """n_spatial matches the length of output_spatial_frac."""
        rio = self._make_minimal()
        assert rio.n_spatial == 16

    def test_output_shape(self):
        """output_shape is (n_spatial, n_spectral)."""
        rio = self._make_minimal()
        assert rio.output_shape == (16, 32)

    def test_src_rows_shape(self):
        """src_rows has shape (n_spatial, n_spectral)."""
        rio = self._make_minimal()
        assert rio.src_rows.shape == (rio.n_spatial, rio.n_spectral)

    def test_src_cols_shape(self):
        """src_cols has shape (n_spectral,)."""
        rio = self._make_minimal()
        assert rio.src_cols.shape == (rio.n_spectral,)


# ===========================================================================
# 2. RectificationIndexSet: construction
# ===========================================================================


class TestRectificationIndexSetConstruction:
    """Unit tests for RectificationIndexSet construction and interface."""

    def _make_minimal(self) -> RectificationIndexSet:
        n_spectral = 16
        n_spatial = 8
        orders = [
            RectificationIndexOrder(
                order=i,
                order_index=i,
                output_wavelengths_um=np.linspace(1.5 + i * 0.05, 1.55 + i * 0.05, n_spectral),
                output_spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                src_cols=np.linspace(100.0, 700.0, n_spectral),
                src_rows=np.zeros((n_spatial, n_spectral)),
            )
            for i in range(3)
        ]
        return RectificationIndexSet(mode="H1_test", index_orders=orders)

    def test_construction(self):
        """RectificationIndexSet can be constructed."""
        ris = self._make_minimal()
        assert ris.mode == "H1_test"

    def test_n_orders(self):
        """n_orders equals the number of index_orders."""
        ris = self._make_minimal()
        assert ris.n_orders == 3

    def test_orders_list(self):
        """orders property returns order numbers in storage order."""
        ris = self._make_minimal()
        assert ris.orders == [0, 1, 2]

    def test_get_order_found(self):
        """get_order returns the correct RectificationIndexOrder."""
        ris = self._make_minimal()
        rio = ris.get_order(1)
        assert rio.order == 1
        assert rio.order_index == 1

    def test_get_order_not_found(self):
        """get_order raises KeyError for an unknown order."""
        ris = self._make_minimal()
        with pytest.raises(KeyError, match="99"):
            ris.get_order(99)

    def test_empty_set(self):
        """Empty RectificationIndexSet is valid and has n_orders == 0."""
        ris = RectificationIndexSet(mode="test")
        assert ris.n_orders == 0
        assert ris.orders == []


# ===========================================================================
# 3. build_rectification_indices on synthetic data
# ===========================================================================

_N_SPECTRAL = 32
_N_SPATIAL = 16


class TestBuildRectificationIndicesSynthetic:
    """build_rectification_indices on fully controlled synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.geometry = _make_synthetic_geometry()
        self.surface = _make_synthetic_surface()
        self.result = build_rectification_indices(
            self.geometry,
            self.surface,
            n_spectral=_N_SPECTRAL,
            n_spatial=_N_SPATIAL,
        )

    def test_returns_rectification_index_set(self):
        """The function returns a RectificationIndexSet."""
        assert isinstance(self.result, RectificationIndexSet)

    def test_mode_propagated(self):
        """mode matches geometry.mode."""
        assert self.result.mode == self.geometry.mode

    def test_n_orders(self):
        """n_orders equals the number of orders in geometry."""
        assert self.result.n_orders == _N_ORDERS

    def test_orders_list_length(self):
        """orders list has the correct length."""
        assert len(self.result.orders) == _N_ORDERS

    def test_output_wavelengths_shape(self):
        """output_wavelengths_um has length n_spectral for every order."""
        for rio in self.result.index_orders:
            assert rio.output_wavelengths_um.shape == (_N_SPECTRAL,), (
                f"Order {rio.order}: expected ({_N_SPECTRAL},), "
                f"got {rio.output_wavelengths_um.shape}"
            )

    def test_output_spatial_frac_shape(self):
        """output_spatial_frac has length n_spatial for every order."""
        for rio in self.result.index_orders:
            assert rio.output_spatial_frac.shape == (_N_SPATIAL,), (
                f"Order {rio.order}: expected ({_N_SPATIAL},), "
                f"got {rio.output_spatial_frac.shape}"
            )

    def test_src_cols_shape(self):
        """src_cols has shape (n_spectral,) for every order."""
        for rio in self.result.index_orders:
            assert rio.src_cols.shape == (_N_SPECTRAL,), (
                f"Order {rio.order}: expected ({_N_SPECTRAL},), "
                f"got {rio.src_cols.shape}"
            )

    def test_src_rows_shape(self):
        """src_rows has shape (n_spatial, n_spectral) for every order."""
        for rio in self.result.index_orders:
            assert rio.src_rows.shape == (_N_SPATIAL, _N_SPECTRAL), (
                f"Order {rio.order}: expected ({_N_SPATIAL}, {_N_SPECTRAL}), "
                f"got {rio.src_rows.shape}"
            )

    def test_output_shape_property(self):
        """output_shape property equals (n_spatial, n_spectral)."""
        for rio in self.result.index_orders:
            assert rio.output_shape == (_N_SPATIAL, _N_SPECTRAL)

    def test_wavelength_axis_monotone(self):
        """output_wavelengths_um is monotone (increasing or decreasing)."""
        for rio in self.result.index_orders:
            diffs = np.diff(rio.output_wavelengths_um)
            is_increasing = np.all(diffs > 0)
            is_decreasing = np.all(diffs < 0)
            assert is_increasing or is_decreasing, (
                f"Order {rio.order}: wavelength axis is not monotone; "
                f"diffs range [{diffs.min():.4e}, {diffs.max():.4e}]"
            )

    def test_wavelength_axis_uniformly_spaced(self):
        """output_wavelengths_um is uniformly spaced (np.linspace)."""
        for rio in self.result.index_orders:
            diffs = np.diff(rio.output_wavelengths_um)
            assert np.allclose(diffs, diffs[0], rtol=1e-10), (
                f"Order {rio.order}: wavelength axis is not uniformly spaced"
            )

    def test_spatial_frac_starts_at_zero(self):
        """output_spatial_frac starts at 0.0 for every order."""
        for rio in self.result.index_orders:
            assert rio.output_spatial_frac[0] == pytest.approx(0.0)

    def test_spatial_frac_ends_at_one(self):
        """output_spatial_frac ends at 1.0 for every order."""
        for rio in self.result.index_orders:
            assert rio.output_spatial_frac[-1] == pytest.approx(1.0)

    def test_spatial_frac_uniformly_spaced(self):
        """output_spatial_frac is uniformly spaced."""
        for rio in self.result.index_orders:
            diffs = np.diff(rio.output_spatial_frac)
            assert np.allclose(diffs, diffs[0], rtol=1e-10)

    def test_src_cols_within_order_range(self):
        """src_cols lie within the order's column range (with 1-pixel tolerance)."""
        tol = 1.0  # 1 pixel tolerance for interpolation boundary effects
        for i, rio in enumerate(self.result.index_orders):
            geom = self.geometry.geometries[i]
            assert np.all(rio.src_cols >= geom.x_start - tol), (
                f"Order {rio.order}: src_cols below x_start "
                f"(min={rio.src_cols.min():.2f}, x_start={geom.x_start})"
            )
            assert np.all(rio.src_cols <= geom.x_end + tol), (
                f"Order {rio.order}: src_cols above x_end "
                f"(max={rio.src_cols.max():.2f}, x_end={geom.x_end})"
            )

    def test_src_rows_within_order_row_range(self):
        """src_rows lie within the order's row extent (with small tolerance)."""
        tol = 1.0  # 1 pixel tolerance for edge-polynomial evaluation
        for i, rio in enumerate(self.result.index_orders):
            geom = self.geometry.geometries[i]
            # The bottom edge is center_row - half_width; top is center + half_width.
            row_min = float(geom.eval_bottom_edge(np.array([geom.x_start])).min()) - tol
            row_max = float(geom.eval_top_edge(np.array([geom.x_start])).max()) + tol
            assert np.all(rio.src_rows >= row_min), (
                f"Order {rio.order}: src_rows below bottom edge "
                f"(min={rio.src_rows.min():.2f}, expected>={row_min:.2f})"
            )
            assert np.all(rio.src_rows <= row_max), (
                f"Order {rio.order}: src_rows above top edge "
                f"(max={rio.src_rows.max():.2f}, expected<={row_max:.2f})"
            )

    def test_src_rows_spatial_ordering(self):
        """src_rows increases with spatial fraction (bottom to top)."""
        for rio in self.result.index_orders:
            # For each spectral pixel, rows should be non-decreasing in spatial
            row_diffs = np.diff(rio.src_rows, axis=0)
            assert np.all(row_diffs >= 0), (
                f"Order {rio.order}: src_rows not non-decreasing along "
                f"spatial axis (min diff={row_diffs.min():.4e})"
            )

    def test_get_order_round_trip(self):
        """get_order round-trips correctly for each order in the result."""
        for rio in self.result.index_orders:
            fetched = self.result.get_order(rio.order)
            assert fetched is rio

    def test_order_index_values(self):
        """order_index is correct (0-based position in result.index_orders)."""
        for i, rio in enumerate(self.result.index_orders):
            assert rio.order_index == i

    def test_all_values_finite(self):
        """src_cols, src_rows, and output arrays contain only finite values."""
        for rio in self.result.index_orders:
            assert np.all(np.isfinite(rio.src_cols)), f"Order {rio.order}: non-finite src_cols"
            assert np.all(np.isfinite(rio.src_rows)), f"Order {rio.order}: non-finite src_rows"
            assert np.all(np.isfinite(rio.output_wavelengths_um)), (
                f"Order {rio.order}: non-finite output_wavelengths_um"
            )

    def test_wavelengths_in_h_band_range(self):
        """output_wavelengths_um are in a plausible H-band range."""
        for rio in self.result.index_orders:
            assert rio.output_wavelengths_um.min() >= 1.5
            assert rio.output_wavelengths_um.max() <= 1.75


# ===========================================================================
# 4. Error handling
# ===========================================================================


class TestBuildRectificationIndicesErrors:
    """Error and warning paths."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.geometry = _make_synthetic_geometry()
        self.surface = _make_synthetic_surface()

    def test_raises_on_empty_geometry(self):
        """ValueError if geometry contains no orders."""
        empty_geom = OrderGeometrySet(mode="test", geometries=[])
        with pytest.raises(ValueError, match="no orders"):
            build_rectification_indices(empty_geom, self.surface)

    def test_raises_on_no_matching_orders(self):
        """ValueError if no geometry orders match any surface order."""
        # Use an order number that doesn't appear in any surface fitted order
        geoms = [
            OrderGeometry(
                order=9999,  # no match in surface (which has orders 311-325)
                x_start=100,
                x_end=700,
                bottom_edge_coeffs=np.array([20.0, 0.0]),
                top_edge_coeffs=np.array([60.0, 0.0]),
            )
        ]
        no_match_geom = OrderGeometrySet(mode="H1_test", geometries=geoms)
        with pytest.raises(ValueError, match="No orders matched"):
            build_rectification_indices(no_match_geom, self.surface)

    def test_raises_on_n_spectral_too_small(self):
        """ValueError if n_spectral < 2."""
        with pytest.raises(ValueError, match="n_spectral must be >= 2"):
            build_rectification_indices(
                self.geometry, self.surface, n_spectral=1
            )

    def test_raises_on_n_spatial_too_small(self):
        """ValueError if n_spatial < 2."""
        with pytest.raises(ValueError, match="n_spatial must be >= 2"):
            build_rectification_indices(
                self.geometry, self.surface, n_spatial=1
            )

    def test_raises_on_n_col_samples_too_small(self):
        """ValueError if n_col_samples < n_spectral."""
        with pytest.raises(ValueError, match="n_col_samples"):
            build_rectification_indices(
                self.geometry, self.surface, n_spectral=64, n_col_samples=32
            )

    def test_warns_on_non_monotone_surface(self):
        """RuntimeWarning is emitted if the surface is non-monotone in column.

        We construct a RefinedCoefficientSurface with order_smooth_coeffs
        that produce a non-monotone wavelength(col) curve and confirm that
        the warning is raised.
        """
        import copy

        # Clone the surface and override order_smooth_coeffs so that the
        # per-order wavelength vs column function oscillates (non-monotone).
        # A degree-1 disp surface with a zero linear coefficient is constant
        # in column, which np.diff gives all zeros → not monotone.
        bad_surface = copy.copy(self.surface)
        # Set the linear (k=1) coefficient to zero for all orders so the
        # predicted wavelength is flat (not monotone increasing or decreasing).
        bad_coeffs = bad_surface.order_smooth_coeffs.copy()
        bad_coeffs[1, :] = 0.0  # zero out the column-linear term
        bad_surface = RefinedCoefficientSurface(
            mode=bad_surface.mode,
            disp_degree=bad_surface.disp_degree,
            order_smooth_degree=bad_surface.order_smooth_degree,
            order_ref=bad_surface.order_ref,
            order_smooth_coeffs=bad_coeffs,
            per_order_fits=bad_surface.per_order_fits,
            per_order_rms_um=bad_surface.per_order_rms_um,
            per_order_order_numbers=bad_surface.per_order_order_numbers,
            smooth_rms_um=bad_surface.smooth_rms_um,
            n_orders_fit=bad_surface.n_orders_fit,
            n_points_total=bad_surface.n_points_total,
            n_orders_skipped=bad_surface.n_orders_skipped,
        )
        with pytest.warns(RuntimeWarning, match="not strictly monotone"):
            build_rectification_indices(self.geometry, bad_surface)


# ===========================================================================
# 5. Parametric shape consistency check
# ===========================================================================


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 4), (64, 32), (128, 128)])
def test_output_shapes_parametric(n_spectral, n_spatial):
    """Array shapes are consistent for various (n_spectral, n_spatial) values."""
    geometry = _make_synthetic_geometry()
    surface = _make_synthetic_surface()
    result = build_rectification_indices(
        geometry, surface, n_spectral=n_spectral, n_spatial=n_spatial
    )
    for rio in result.index_orders:
        assert rio.output_wavelengths_um.shape == (n_spectral,)
        assert rio.output_spatial_frac.shape == (n_spatial,)
        assert rio.src_cols.shape == (n_spectral,)
        assert rio.src_rows.shape == (n_spatial, n_spectral)
        assert rio.output_shape == (n_spatial, n_spectral)


# ===========================================================================
# 6. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1RectificationIndicesSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset."""

    @pytest.fixture(scope="class")
    def h1_result(self):
        """RectificationIndexSet built from the real H1 calibration chain."""
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
        wav_map = fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )
        surface = fit_refined_coefficient_surface(wav_map, disp_degree=3)
        # Pass wav_map to bridge placeholder geometry indices (0, 1, ...) to
        # real echelle order numbers used by the surface.
        return build_rectification_indices(geom, surface, wav_map=wav_map)

    def test_returns_rectification_index_set(self, h1_result):
        """build_rectification_indices returns a RectificationIndexSet."""
        assert isinstance(h1_result, RectificationIndexSet)

    def test_mode_is_h1(self, h1_result):
        """mode is 'H1'."""
        assert h1_result.mode == "H1"

    def test_n_orders_positive(self, h1_result):
        """n_orders > 0."""
        assert h1_result.n_orders > 0

    def test_default_output_shapes(self, h1_result):
        """Default n_spectral=256 and n_spatial=64 produce correct shapes."""
        for rio in h1_result.index_orders:
            assert rio.src_cols.shape == (256,)
            assert rio.src_rows.shape == (64, 256)

    def test_wavelengths_monotone(self, h1_result):
        """output_wavelengths_um is monotone for every order."""
        for rio in h1_result.index_orders:
            diffs = np.diff(rio.output_wavelengths_um)
            assert np.all(diffs > 0) or np.all(diffs < 0)

    def test_wavelengths_in_h_band(self, h1_result):
        """output_wavelengths_um are in a plausible H-band range."""
        for rio in h1_result.index_orders:
            assert rio.output_wavelengths_um.min() >= 1.35
            assert rio.output_wavelengths_um.max() <= 1.95

    def test_spatial_frac_endpoints(self, h1_result):
        """output_spatial_frac starts at 0.0 and ends at 1.0."""
        for rio in h1_result.index_orders:
            assert rio.output_spatial_frac[0] == pytest.approx(0.0)
            assert rio.output_spatial_frac[-1] == pytest.approx(1.0)

    def test_all_values_finite(self, h1_result):
        """All output arrays contain only finite values."""
        for rio in h1_result.index_orders:
            assert np.all(np.isfinite(rio.src_cols))
            assert np.all(np.isfinite(rio.src_rows))
            assert np.all(np.isfinite(rio.output_wavelengths_um))
