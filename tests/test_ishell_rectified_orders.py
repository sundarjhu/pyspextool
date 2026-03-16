"""
Tests for the provisional rectified-order image generation scaffold
(rectified_orders.py).

Coverage:
  - RectifiedOrder: construction and field access.
  - RectifiedOrderSet: construction, get_order, orders, n_orders.
  - build_rectified_orders on synthetic data:
      * returns correct type and fields,
      * array shapes are consistent,
      * wavelength axis is monotone,
      * spatial axis is uniformly spaced in [0, 1],
      * interpolation correctness on a linear gradient image,
      * out-of-bounds source coordinates produce NaN,
      * mode is propagated from rectification_indices.
  - Error handling:
      * non-2-D image → ValueError,
      * empty rectification index set → ValueError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyspextool.instruments.ishell.geometry import OrderGeometry, OrderGeometrySet
from pyspextool.instruments.ishell.rectification_indices import (
    RectificationIndexOrder,
    RectificationIndexSet,
)
from pyspextool.instruments.ishell.rectified_orders import (
    RectifiedOrder,
    RectifiedOrderSet,
    build_rectified_orders,
)

# ---------------------------------------------------------------------------
# Path helpers for real H1 calibration data  (mirrors rectification_indices tests)
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
# Synthetic-data helpers
# ---------------------------------------------------------------------------

_NROWS = 200
_NCOLS = 800
_N_ORDERS = 3

# Three synthetic orders with simple geometry.
_ORDER_PARAMS = [
    {
        "order_number": 311,
        "col_start": 50,
        "col_end": 350,
        "center_row": 40,
        "half_width": 15,
    },
    {
        "order_number": 315,
        "col_start": 50,
        "col_end": 350,
        "center_row": 100,
        "half_width": 15,
    },
    {
        "order_number": 320,
        "col_start": 50,
        "col_end": 350,
        "center_row": 160,
        "half_width": 15,
    },
]


def _make_synthetic_detector_image(n_rows: int = _NROWS, n_cols: int = _NCOLS) -> np.ndarray:
    """Return a synthetic detector image.

    The image value at ``(row, col)`` is ``row + col`` (a linear gradient).
    This makes it straightforward to verify bilinear interpolation
    correctness: interpolating at ``(r, c)`` should return ``r + c``
    exactly (bilinear interpolation is exact for affine functions).
    """
    rows = np.arange(n_rows, dtype=float)[:, np.newaxis]
    cols = np.arange(n_cols, dtype=float)[np.newaxis, :]
    return rows + cols


def _make_synthetic_rectification_index_set(
    n_spectral: int = 32,
    n_spatial: int = 16,
) -> RectificationIndexSet:
    """Build a RectificationIndexSet from the synthetic order parameters."""
    index_orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        cr = float(p["center_row"])
        hw = float(p["half_width"])
        col_start = float(p["col_start"])
        col_end = float(p["col_end"])

        # Wavelength axis: simple linear proxy (not real wavelengths).
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        output_wavs = np.linspace(wav_start, wav_end, n_spectral)

        # Spatial axis: uniform [0, 1].
        output_spatial_frac = np.linspace(0.0, 1.0, n_spatial)

        # src_cols: map uniformly from col_start to col_end.
        src_cols = np.linspace(col_start, col_end, n_spectral)

        # src_rows: map spatial_frac ∈ [0, 1] to rows from (cr-hw) to (cr+hw).
        bottom = cr - hw
        top = cr + hw
        # Shape: (n_spatial, n_spectral) — rows vary with spatial, cols constant.
        src_rows = (
            bottom
            + output_spatial_frac[:, np.newaxis] * (top - bottom)
        ) * np.ones((1, n_spectral))

        index_orders.append(
            RectificationIndexOrder(
                order=p["order_number"],
                order_index=idx,
                output_wavelengths_um=output_wavs,
                output_spatial_frac=output_spatial_frac,
                src_cols=src_cols,
                src_rows=src_rows,
            )
        )
    return RectificationIndexSet(mode="H1_test", index_orders=index_orders)


# ===========================================================================
# 1. RectifiedOrder: construction
# ===========================================================================


class TestRectifiedOrderConstruction:
    """Unit tests for RectifiedOrder construction and field access."""

    def _make_minimal(self) -> RectifiedOrder:
        n_spectral = 32
        n_spatial = 16
        return RectifiedOrder(
            order=311,
            order_index=0,
            wavelength_um=np.linspace(1.64, 1.69, n_spectral),
            spatial_frac=np.linspace(0.0, 1.0, n_spatial),
            flux=np.zeros((n_spatial, n_spectral)),
            source_image_shape=(200, 800),
        )

    def test_construction(self):
        """RectifiedOrder can be constructed and fields are accessible."""
        ro = self._make_minimal()
        assert ro.order == 311
        assert ro.order_index == 0

    def test_n_spectral(self):
        """n_spectral matches the length of wavelength_um."""
        ro = self._make_minimal()
        assert ro.n_spectral == 32

    def test_n_spatial(self):
        """n_spatial matches the length of spatial_frac."""
        ro = self._make_minimal()
        assert ro.n_spatial == 16

    def test_shape(self):
        """shape is (n_spatial, n_spectral)."""
        ro = self._make_minimal()
        assert ro.shape == (16, 32)

    def test_flux_shape(self):
        """flux has shape (n_spatial, n_spectral)."""
        ro = self._make_minimal()
        assert ro.flux.shape == (ro.n_spatial, ro.n_spectral)

    def test_source_image_shape(self):
        """source_image_shape field is stored correctly."""
        ro = self._make_minimal()
        assert ro.source_image_shape == (200, 800)


# ===========================================================================
# 2. RectifiedOrderSet: construction
# ===========================================================================


class TestRectifiedOrderSetConstruction:
    """Unit tests for RectifiedOrderSet construction and interface."""

    def _make_minimal(self) -> RectifiedOrderSet:
        n_spectral = 16
        n_spatial = 8
        orders = [
            RectifiedOrder(
                order=i,
                order_index=i,
                wavelength_um=np.linspace(1.5 + i * 0.05, 1.55 + i * 0.05, n_spectral),
                spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                flux=np.zeros((n_spatial, n_spectral)),
                source_image_shape=(200, 800),
            )
            for i in range(3)
        ]
        return RectifiedOrderSet(
            mode="H1_test",
            rectified_orders=orders,
            source_image_shape=(200, 800),
        )

    def test_construction(self):
        """RectifiedOrderSet can be constructed."""
        ros = self._make_minimal()
        assert ros.mode == "H1_test"

    def test_n_orders(self):
        """n_orders equals the number of rectified_orders."""
        ros = self._make_minimal()
        assert ros.n_orders == 3

    def test_orders_list(self):
        """orders property returns order numbers in storage order."""
        ros = self._make_minimal()
        assert ros.orders == [0, 1, 2]

    def test_source_image_shape(self):
        """source_image_shape is stored correctly."""
        ros = self._make_minimal()
        assert ros.source_image_shape == (200, 800)

    def test_get_order_found(self):
        """get_order returns the correct RectifiedOrder."""
        ros = self._make_minimal()
        ro = ros.get_order(1)
        assert ro.order == 1
        assert ro.order_index == 1

    def test_get_order_not_found(self):
        """get_order raises KeyError for an unknown order."""
        ros = self._make_minimal()
        with pytest.raises(KeyError, match="99"):
            ros.get_order(99)

    def test_empty_set(self):
        """Empty RectifiedOrderSet is valid and has n_orders == 0."""
        ros = RectifiedOrderSet(mode="test")
        assert ros.n_orders == 0
        assert ros.orders == []


# ===========================================================================
# 3. build_rectified_orders on synthetic data
# ===========================================================================

_N_SPECTRAL = 32
_N_SPATIAL = 16


class TestBuildRectifiedOrdersSynthetic:
    """build_rectified_orders on fully controlled synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.image = _make_synthetic_detector_image()
        self.rect_indices = _make_synthetic_rectification_index_set(
            n_spectral=_N_SPECTRAL, n_spatial=_N_SPATIAL
        )
        self.result = build_rectified_orders(self.image, self.rect_indices)

    def test_returns_rectified_order_set(self):
        """build_rectified_orders returns a RectifiedOrderSet."""
        assert isinstance(self.result, RectifiedOrderSet)

    def test_mode_propagated(self):
        """mode matches rectification_indices.mode."""
        assert self.result.mode == self.rect_indices.mode

    def test_n_orders(self):
        """n_orders equals the number of orders in rectification_indices."""
        assert self.result.n_orders == _N_ORDERS

    def test_source_image_shape(self):
        """source_image_shape matches the input image shape."""
        assert self.result.source_image_shape == self.image.shape

    def test_flux_shape(self):
        """flux has shape (n_spatial, n_spectral) for every order."""
        for ro in self.result.rectified_orders:
            assert ro.flux.shape == (_N_SPATIAL, _N_SPECTRAL), (
                f"Order {ro.order}: expected ({_N_SPATIAL}, {_N_SPECTRAL}), "
                f"got {ro.flux.shape}"
            )

    def test_wavelength_axis_shape(self):
        """wavelength_um has length n_spectral for every order."""
        for ro in self.result.rectified_orders:
            assert ro.wavelength_um.shape == (_N_SPECTRAL,), (
                f"Order {ro.order}: expected ({_N_SPECTRAL},), "
                f"got {ro.wavelength_um.shape}"
            )

    def test_spatial_frac_shape(self):
        """spatial_frac has length n_spatial for every order."""
        for ro in self.result.rectified_orders:
            assert ro.spatial_frac.shape == (_N_SPATIAL,), (
                f"Order {ro.order}: expected ({_N_SPATIAL},), "
                f"got {ro.spatial_frac.shape}"
            )

    def test_shape_property(self):
        """shape property equals (n_spatial, n_spectral)."""
        for ro in self.result.rectified_orders:
            assert ro.shape == (_N_SPATIAL, _N_SPECTRAL)

    def test_wavelength_axis_monotone(self):
        """wavelength_um is monotone (increasing or decreasing) for every order."""
        for ro in self.result.rectified_orders:
            diffs = np.diff(ro.wavelength_um)
            is_increasing = np.all(diffs > 0)
            is_decreasing = np.all(diffs < 0)
            assert is_increasing or is_decreasing, (
                f"Order {ro.order}: wavelength axis is not monotone; "
                f"diffs range [{diffs.min():.4e}, {diffs.max():.4e}]"
            )

    def test_spatial_frac_starts_at_zero(self):
        """spatial_frac starts at 0.0 for every order."""
        for ro in self.result.rectified_orders:
            assert ro.spatial_frac[0] == pytest.approx(0.0)

    def test_spatial_frac_ends_at_one(self):
        """spatial_frac ends at 1.0 for every order."""
        for ro in self.result.rectified_orders:
            assert ro.spatial_frac[-1] == pytest.approx(1.0)

    def test_spatial_frac_uniformly_spaced(self):
        """spatial_frac is uniformly spaced for every order."""
        for ro in self.result.rectified_orders:
            diffs = np.diff(ro.spatial_frac)
            assert np.allclose(diffs, diffs[0], rtol=1e-10)

    def test_interpolation_correctness(self):
        """Bilinear interpolation is exact for the linear gradient image.

        Since image[row, col] = row + col, and bilinear interpolation is
        exact for affine functions, the rectified flux at position (j, i)
        should equal src_rows[j, i] + src_cols[i] to within floating-point
        precision.
        """
        for idx, ro in enumerate(self.result.rectified_orders):
            idx_order = self.rect_indices.index_orders[idx]
            src_rows = idx_order.src_rows  # (n_spatial, n_spectral)
            src_cols = idx_order.src_cols  # (n_spectral,)
            # Expected: flux[j, i] = src_rows[j, i] + src_cols[i]
            expected = src_rows + src_cols[np.newaxis, :]
            np.testing.assert_allclose(
                ro.flux,
                expected,
                atol=1e-10,
                err_msg=f"Order {ro.order}: interpolation deviates from expected",
            )

    def test_all_flux_finite(self):
        """All flux values are finite (no NaN for in-bounds coordinates)."""
        for ro in self.result.rectified_orders:
            assert np.all(np.isfinite(ro.flux)), (
                f"Order {ro.order}: flux contains non-finite values"
            )

    def test_order_index_values(self):
        """order_index is correct (0-based position in result.rectified_orders)."""
        for i, ro in enumerate(self.result.rectified_orders):
            assert ro.order_index == i

    def test_per_order_source_image_shape(self):
        """source_image_shape in each RectifiedOrder matches the input."""
        for ro in self.result.rectified_orders:
            assert ro.source_image_shape == self.image.shape

    def test_get_order_round_trip(self):
        """get_order round-trips correctly for each order in the result."""
        for ro in self.result.rectified_orders:
            fetched = self.result.get_order(ro.order)
            assert fetched is ro

    def test_wavelength_axes_match_indices(self):
        """wavelength_um in each RectifiedOrder matches output_wavelengths_um
        from the corresponding RectificationIndexOrder."""
        for i, ro in enumerate(self.result.rectified_orders):
            idx_order = self.rect_indices.index_orders[i]
            np.testing.assert_array_equal(
                ro.wavelength_um,
                idx_order.output_wavelengths_um,
                err_msg=f"Order {ro.order}: wavelength_um does not match index",
            )

    def test_spatial_frac_axes_match_indices(self):
        """spatial_frac in each RectifiedOrder matches output_spatial_frac
        from the corresponding RectificationIndexOrder."""
        for i, ro in enumerate(self.result.rectified_orders):
            idx_order = self.rect_indices.index_orders[i]
            np.testing.assert_array_equal(
                ro.spatial_frac,
                idx_order.output_spatial_frac,
                err_msg=f"Order {ro.order}: spatial_frac does not match index",
            )


class TestBuildRectifiedOrdersOutOfBounds:
    """Verify that out-of-bounds source coordinates produce NaN."""

    def test_out_of_bounds_coords_are_nan(self):
        """Source coordinates outside the detector image are filled with NaN."""
        image = _make_synthetic_detector_image(n_rows=50, n_cols=100)
        # Create a rectification index that points outside the detector.
        n_spectral = 8
        n_spatial = 4
        # src_cols far outside [0, 99]
        src_cols = np.linspace(200.0, 300.0, n_spectral)
        src_rows = np.ones((n_spatial, n_spectral)) * 25.0
        index_orders = [
            RectificationIndexOrder(
                order=1,
                order_index=0,
                output_wavelengths_um=np.linspace(1.6, 1.7, n_spectral),
                output_spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                src_cols=src_cols,
                src_rows=src_rows,
            )
        ]
        rect_indices = RectificationIndexSet(
            mode="test", index_orders=index_orders
        )
        result = build_rectified_orders(image, rect_indices)
        assert result.n_orders == 1
        ro = result.rectified_orders[0]
        assert np.all(np.isnan(ro.flux)), (
            "Expected all NaN for out-of-bounds coordinates"
        )

    def test_partially_out_of_bounds(self):
        """Some NaN and some finite values when some coords are out of bounds."""
        image = _make_synthetic_detector_image(n_rows=50, n_cols=100)
        n_spectral = 4
        n_spatial = 2
        # First two columns in-bounds, last two out-of-bounds.
        src_cols = np.array([10.0, 20.0, 120.0, 200.0])
        src_rows = np.ones((n_spatial, n_spectral)) * 25.0
        index_orders = [
            RectificationIndexOrder(
                order=1,
                order_index=0,
                output_wavelengths_um=np.linspace(1.6, 1.7, n_spectral),
                output_spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                src_cols=src_cols,
                src_rows=src_rows,
            )
        ]
        rect_indices = RectificationIndexSet(
            mode="test", index_orders=index_orders
        )
        result = build_rectified_orders(image, rect_indices)
        ro = result.rectified_orders[0]
        # First two spectral pixels should be finite, last two should be NaN.
        assert np.all(np.isfinite(ro.flux[:, :2])), (
            "Expected finite values for in-bounds columns"
        )
        assert np.all(np.isnan(ro.flux[:, 2:])), (
            "Expected NaN for out-of-bounds columns"
        )


# ===========================================================================
# 4. Error handling
# ===========================================================================


class TestBuildRectifiedOrdersErrors:
    """Error paths for build_rectified_orders."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rect_indices = _make_synthetic_rectification_index_set()

    def test_raises_on_1d_image(self):
        """ValueError if image is 1-D."""
        bad_image = np.ones(200)
        with pytest.raises(ValueError, match="2-D"):
            build_rectified_orders(bad_image, self.rect_indices)

    def test_raises_on_3d_image(self):
        """ValueError if image is 3-D."""
        bad_image = np.ones((10, 20, 3))
        with pytest.raises(ValueError, match="2-D"):
            build_rectified_orders(bad_image, self.rect_indices)

    def test_raises_on_empty_rectification_index_set(self):
        """ValueError if rectification_indices has no orders."""
        empty_set = RectificationIndexSet(mode="test", index_orders=[])
        image = _make_synthetic_detector_image()
        with pytest.raises(ValueError, match="empty"):
            build_rectified_orders(image, empty_set)

    def test_accepts_float_image(self):
        """float32 and float64 images are both accepted."""
        image_f32 = _make_synthetic_detector_image().astype(np.float32)
        result = build_rectified_orders(image_f32, self.rect_indices)
        assert isinstance(result, RectifiedOrderSet)

    def test_accepts_integer_image(self):
        """Integer images are accepted (converted to float internally)."""
        image_int = np.ones((_NROWS, _NCOLS), dtype=np.int32)
        result = build_rectified_orders(image_int, self.rect_indices)
        assert isinstance(result, RectifiedOrderSet)


# ===========================================================================
# 5. Parametric shape consistency check
# ===========================================================================


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 4), (64, 32), (128, 128)])
def test_output_shapes_parametric(n_spectral, n_spatial):
    """Array shapes are consistent for various (n_spectral, n_spatial) values."""
    image = _make_synthetic_detector_image()
    rect_indices = _make_synthetic_rectification_index_set(
        n_spectral=n_spectral, n_spatial=n_spatial
    )
    result = build_rectified_orders(image, rect_indices)
    for ro in result.rectified_orders:
        assert ro.wavelength_um.shape == (n_spectral,)
        assert ro.spatial_frac.shape == (n_spatial,)
        assert ro.flux.shape == (n_spatial, n_spectral)
        assert ro.shape == (n_spatial, n_spectral)


# ===========================================================================
# 6. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1RectifiedOrdersSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset."""

    @pytest.fixture(scope="class")
    def h1_result(self):
        """RectifiedOrderSet built from the real H1 calibration chain."""
        import astropy.io.fits as fits

        from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
        from pyspextool.instruments.ishell.calibrations import (
            read_line_list,
            read_wavecalinfo,
        )
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
        from pyspextool.instruments.ishell.wavecal_2d import (
            fit_provisional_wavelength_map,
        )
        from pyspextool.instruments.ishell.wavecal_2d_refine import (
            fit_refined_coefficient_surface,
        )

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
        rect_indices = build_rectification_indices(
            geom, surface, wav_map=wav_map
        )

        # Use the first flat file as a representative detector image.
        with fits.open(_H1_FLAT_FILES[0]) as hdul:
            detector_image = hdul[0].data.astype(float)

        return build_rectified_orders(detector_image, rect_indices), rect_indices

    def test_returns_rectified_order_set(self, h1_result):
        """build_rectified_orders returns a RectifiedOrderSet."""
        result, _ = h1_result
        assert isinstance(result, RectifiedOrderSet)

    def test_mode_is_h1(self, h1_result):
        """mode is 'H1'."""
        result, _ = h1_result
        assert result.mode == "H1"

    def test_n_orders_positive(self, h1_result):
        """n_orders > 0."""
        result, _ = h1_result
        assert result.n_orders > 0

    def test_n_orders_matches_indices(self, h1_result):
        """n_orders matches the rectification index set."""
        result, rect_indices = h1_result
        assert result.n_orders == rect_indices.n_orders

    def test_output_shapes_consistent(self, h1_result):
        """Output shapes match the default n_spectral=256 and n_spatial=64."""
        result, _ = h1_result
        for ro in result.rectified_orders:
            assert ro.wavelength_um.shape == (256,)
            assert ro.spatial_frac.shape == (64,)
            assert ro.flux.shape == (64, 256)

    def test_wavelength_axis_monotone(self, h1_result):
        """wavelength_um is monotone for every order."""
        result, _ = h1_result
        for ro in result.rectified_orders:
            diffs = np.diff(ro.wavelength_um)
            assert np.all(diffs > 0) or np.all(diffs < 0), (
                f"Order {ro.order}: wavelength axis is not monotone"
            )

    def test_wavelengths_in_h_band(self, h1_result):
        """wavelength_um is in a plausible H-band range for every order."""
        result, _ = h1_result
        for ro in result.rectified_orders:
            assert ro.wavelength_um.min() >= 1.35
            assert ro.wavelength_um.max() <= 1.95

    def test_spatial_frac_endpoints(self, h1_result):
        """spatial_frac starts at 0.0 and ends at 1.0 for every order."""
        result, _ = h1_result
        for ro in result.rectified_orders:
            assert ro.spatial_frac[0] == pytest.approx(0.0)
            assert ro.spatial_frac[-1] == pytest.approx(1.0)

    def test_source_image_shape_2d(self, h1_result):
        """source_image_shape is a 2-tuple of positive integers."""
        result, _ = h1_result
        shape = result.source_image_shape
        assert len(shape) == 2
        assert shape[0] > 0
        assert shape[1] > 0

    def test_flux_has_some_finite_values(self, h1_result):
        """flux contains at least some finite (non-NaN) values."""
        result, _ = h1_result
        for ro in result.rectified_orders:
            assert np.any(np.isfinite(ro.flux)), (
                f"Order {ro.order}: flux is entirely NaN"
            )
