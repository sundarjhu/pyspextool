"""
Tests for the provisional wavecal/spatcal calibration-product scaffold
(calibration_products.py).

Coverage:
  - WaveCalProduct: construction and field access.
  - SpatCalProduct: construction and field access.
  - CalibrationProductSet: construction, get_order, orders, n_orders.
  - build_calibration_products on synthetic data:
      * returns correct type and fields,
      * axis propagation correctness,
      * order lookup behaviour,
      * n_orders matches input,
      * mode is propagated from rectified_orders.
  - Error handling:
      * empty rectified order set → ValueError,
      * inconsistent axis sizes → ValueError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * number of orders matches rectified order set,
      * wavelength axes are monotonic,
      * spatial axes are in [0, 1].
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyspextool.instruments.ishell.calibration_products import (
    CalibrationProductSet,
    SpatCalProduct,
    WaveCalProduct,
    build_calibration_products,
)
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
# Path helpers for real H1 calibration data  (mirrors other smoke tests)
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
_N_SPECTRAL = 32
_N_SPATIAL = 16

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


def _make_synthetic_rectified_order_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
) -> RectifiedOrderSet:
    """Build a small synthetic RectifiedOrderSet for testing."""
    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        wavelength_um = np.linspace(wav_start, wav_end, n_spectral)
        spatial_frac = np.linspace(0.0, 1.0, n_spatial)
        flux = np.ones((n_spatial, n_spectral)) * (idx + 1.0)
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=wavelength_um,
                spatial_frac=spatial_frac,
                flux=flux,
                source_image_shape=(_NROWS, _NCOLS),
            )
        )
    return RectifiedOrderSet(
        mode="H1_test",
        rectified_orders=orders,
        source_image_shape=(_NROWS, _NCOLS),
    )


def _make_inconsistent_rectified_order_set() -> RectifiedOrderSet:
    """Build a RectifiedOrderSet whose flux shape is inconsistent with axes."""
    n_spectral = 32
    n_spatial = 16
    # flux shape is (n_spatial, n_spectral) but wavelength_um has wrong length.
    wavelength_um = np.linspace(1.55, 1.59, n_spectral + 5)  # wrong length
    spatial_frac = np.linspace(0.0, 1.0, n_spatial)
    flux = np.ones((n_spatial, n_spectral))
    orders = [
        RectifiedOrder(
            order=311,
            order_index=0,
            wavelength_um=wavelength_um,
            spatial_frac=spatial_frac,
            flux=flux,
            source_image_shape=(_NROWS, _NCOLS),
        )
    ]
    return RectifiedOrderSet(
        mode="H1_test",
        rectified_orders=orders,
        source_image_shape=(_NROWS, _NCOLS),
    )


# ===========================================================================
# 1. WaveCalProduct: construction
# ===========================================================================


class TestWaveCalProductConstruction:
    """Unit tests for WaveCalProduct construction and field access."""

    def _make_minimal(self) -> WaveCalProduct:
        n_spectral = 32
        n_spatial = 16
        return WaveCalProduct(
            order=311,
            wavelength_um=np.linspace(1.64, 1.69, n_spectral),
            spatial_frac=np.linspace(0.0, 1.0, n_spatial),
            rectified_flux=np.zeros((n_spatial, n_spectral)),
        )

    def test_construction(self):
        """WaveCalProduct can be constructed and fields are accessible."""
        wcp = self._make_minimal()
        assert wcp.order == 311

    def test_wavelength_um_stored(self):
        """wavelength_um is stored and accessible."""
        wcp = self._make_minimal()
        assert wcp.wavelength_um.shape == (32,)
        assert wcp.wavelength_um[0] == pytest.approx(1.64)
        assert wcp.wavelength_um[-1] == pytest.approx(1.69)

    def test_spatial_frac_stored(self):
        """spatial_frac is stored and accessible."""
        wcp = self._make_minimal()
        assert wcp.spatial_frac.shape == (16,)
        assert wcp.spatial_frac[0] == pytest.approx(0.0)
        assert wcp.spatial_frac[-1] == pytest.approx(1.0)

    def test_rectified_flux_stored(self):
        """rectified_flux is stored and accessible."""
        wcp = self._make_minimal()
        assert wcp.rectified_flux.shape == (16, 32)

    def test_n_spectral(self):
        """n_spectral matches the length of wavelength_um."""
        wcp = self._make_minimal()
        assert wcp.n_spectral == 32

    def test_n_spatial(self):
        """n_spatial matches the length of spatial_frac."""
        wcp = self._make_minimal()
        assert wcp.n_spatial == 16


# ===========================================================================
# 2. SpatCalProduct: construction
# ===========================================================================


class TestSpatCalProductConstruction:
    """Unit tests for SpatCalProduct construction and field access."""

    def _make_minimal(self) -> SpatCalProduct:
        n_spatial = 16
        return SpatCalProduct(
            order=311,
            spatial_frac=np.linspace(0.0, 1.0, n_spatial),
            detector_rows=np.linspace(0.0, 1.0, n_spatial),
        )

    def test_construction(self):
        """SpatCalProduct can be constructed and fields are accessible."""
        scp = self._make_minimal()
        assert scp.order == 311

    def test_spatial_frac_stored(self):
        """spatial_frac is stored and accessible."""
        scp = self._make_minimal()
        assert scp.spatial_frac.shape == (16,)
        assert scp.spatial_frac[0] == pytest.approx(0.0)
        assert scp.spatial_frac[-1] == pytest.approx(1.0)

    def test_detector_rows_stored(self):
        """detector_rows is stored and accessible."""
        scp = self._make_minimal()
        assert scp.detector_rows.shape == (16,)

    def test_n_spatial(self):
        """n_spatial matches the length of spatial_frac."""
        scp = self._make_minimal()
        assert scp.n_spatial == 16


# ===========================================================================
# 3. CalibrationProductSet: construction and interface
# ===========================================================================


class TestCalibrationProductSetConstruction:
    """Unit tests for CalibrationProductSet construction and interface."""

    def _make_minimal(self) -> CalibrationProductSet:
        n_spectral = 16
        n_spatial = 8
        wavecal = [
            WaveCalProduct(
                order=i,
                wavelength_um=np.linspace(1.5 + i * 0.05, 1.55 + i * 0.05, n_spectral),
                spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                rectified_flux=np.zeros((n_spatial, n_spectral)),
            )
            for i in range(3)
        ]
        spatcal = [
            SpatCalProduct(
                order=i,
                spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                detector_rows=np.linspace(0.0, 1.0, n_spatial),
            )
            for i in range(3)
        ]
        return CalibrationProductSet(
            mode="H1_test",
            wavecal_products=wavecal,
            spatcal_products=spatcal,
        )

    def test_construction(self):
        """CalibrationProductSet can be constructed."""
        cps = self._make_minimal()
        assert cps.mode == "H1_test"

    def test_n_orders(self):
        """n_orders equals the number of wavecal products."""
        cps = self._make_minimal()
        assert cps.n_orders == 3

    def test_orders_list(self):
        """orders property returns order numbers in storage order."""
        cps = self._make_minimal()
        assert cps.orders == [0, 1, 2]

    def test_get_order_found(self):
        """get_order returns the correct (WaveCalProduct, SpatCalProduct) pair."""
        cps = self._make_minimal()
        wcp, scp = cps.get_order(1)
        assert wcp.order == 1
        assert scp.order == 1

    def test_get_order_not_found(self):
        """get_order raises KeyError for an unknown order."""
        cps = self._make_minimal()
        with pytest.raises(KeyError, match="99"):
            cps.get_order(99)

    def test_empty_set(self):
        """Empty CalibrationProductSet is valid and has n_orders == 0."""
        cps = CalibrationProductSet(mode="test")
        assert cps.n_orders == 0
        assert cps.orders == []

    def test_get_order_returns_matching_products(self):
        """get_order returns products whose order field matches the key."""
        cps = self._make_minimal()
        wcp, scp = cps.get_order(2)
        assert wcp.order == 2
        assert scp.order == 2


# ===========================================================================
# 4. build_calibration_products on synthetic data
# ===========================================================================


class TestBuildCalibrationProductsSynthetic:
    """build_calibration_products on fully controlled synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ros = _make_synthetic_rectified_order_set()
        self.result = build_calibration_products(self.ros)

    def test_returns_calibration_product_set(self):
        """build_calibration_products returns a CalibrationProductSet."""
        assert isinstance(self.result, CalibrationProductSet)

    def test_mode_propagated(self):
        """mode matches rectified_orders.mode."""
        assert self.result.mode == self.ros.mode

    def test_n_orders(self):
        """n_orders equals the number of orders in rectified_orders."""
        assert self.result.n_orders == _N_ORDERS

    def test_wavecal_products_count(self):
        """len(wavecal_products) equals n_orders."""
        assert len(self.result.wavecal_products) == _N_ORDERS

    def test_spatcal_products_count(self):
        """len(spatcal_products) equals n_orders."""
        assert len(self.result.spatcal_products) == _N_ORDERS

    def test_wavelength_axis_propagated(self):
        """wavelength_um in WaveCalProduct equals the source rectified order's axis."""
        for ro, wcp in zip(self.ros.rectified_orders, self.result.wavecal_products):
            np.testing.assert_array_equal(
                wcp.wavelength_um,
                ro.wavelength_um,
                err_msg=f"Order {ro.order}: wavelength_um not propagated correctly",
            )

    def test_spatial_frac_propagated_to_wavecal(self):
        """spatial_frac in WaveCalProduct equals the source rectified order's axis."""
        for ro, wcp in zip(self.ros.rectified_orders, self.result.wavecal_products):
            np.testing.assert_array_equal(
                wcp.spatial_frac,
                ro.spatial_frac,
                err_msg=f"Order {ro.order}: spatial_frac not propagated to WaveCalProduct",
            )

    def test_spatial_frac_propagated_to_spatcal(self):
        """spatial_frac in SpatCalProduct equals the source rectified order's axis."""
        for ro, scp in zip(self.ros.rectified_orders, self.result.spatcal_products):
            np.testing.assert_array_equal(
                scp.spatial_frac,
                ro.spatial_frac,
                err_msg=f"Order {ro.order}: spatial_frac not propagated to SpatCalProduct",
            )

    def test_order_numbers_propagated(self):
        """Order numbers in products match the rectified order numbers."""
        for ro, wcp, scp in zip(
            self.ros.rectified_orders,
            self.result.wavecal_products,
            self.result.spatcal_products,
        ):
            assert wcp.order == ro.order
            assert scp.order == ro.order

    def test_orders_list_matches(self):
        """orders list matches the order numbers in the rectified order set."""
        assert self.result.orders == self.ros.orders

    def test_wavelength_axis_monotone(self):
        """wavelength_um is monotone for every wavecal product."""
        for wcp in self.result.wavecal_products:
            diffs = np.diff(wcp.wavelength_um)
            is_increasing = np.all(diffs > 0)
            is_decreasing = np.all(diffs < 0)
            assert is_increasing or is_decreasing, (
                f"Order {wcp.order}: wavelength axis is not monotone"
            )

    def test_spatial_frac_in_unit_interval(self):
        """spatial_frac values are in [0, 1] for every product."""
        for wcp in self.result.wavecal_products:
            assert np.all(wcp.spatial_frac >= 0.0)
            assert np.all(wcp.spatial_frac <= 1.0)
        for scp in self.result.spatcal_products:
            assert np.all(scp.spatial_frac >= 0.0)
            assert np.all(scp.spatial_frac <= 1.0)

    def test_spatial_frac_endpoints(self):
        """spatial_frac starts at 0.0 and ends at 1.0."""
        for wcp in self.result.wavecal_products:
            assert wcp.spatial_frac[0] == pytest.approx(0.0)
            assert wcp.spatial_frac[-1] == pytest.approx(1.0)

    def test_wavecal_shapes(self):
        """WaveCalProduct arrays have consistent shapes."""
        for wcp in self.result.wavecal_products:
            assert wcp.wavelength_um.shape == (_N_SPECTRAL,)
            assert wcp.spatial_frac.shape == (_N_SPATIAL,)
            assert wcp.rectified_flux.shape == (_N_SPATIAL, _N_SPECTRAL)

    def test_spatcal_shapes(self):
        """SpatCalProduct arrays have consistent shapes."""
        for scp in self.result.spatcal_products:
            assert scp.spatial_frac.shape == (_N_SPATIAL,)
            assert scp.detector_rows.shape == (_N_SPATIAL,)

    def test_get_order_round_trip(self):
        """get_order returns products with the correct order number."""
        for ro in self.ros.rectified_orders:
            wcp, scp = self.result.get_order(ro.order)
            assert wcp.order == ro.order
            assert scp.order == ro.order

    def test_wavelength_um_is_copy(self):
        """wavelength_um in WaveCalProduct is a copy, not an alias."""
        ros = _make_synthetic_rectified_order_set()
        result = build_calibration_products(ros)
        original = ros.rectified_orders[0].wavelength_um.copy()
        ros.rectified_orders[0].wavelength_um[:] = 0.0
        np.testing.assert_array_equal(
            result.wavecal_products[0].wavelength_um,
            original,
            err_msg="wavelength_um was not copied; mutation of source affected product",
        )

    def test_spatial_frac_is_copy(self):
        """spatial_frac in SpatCalProduct is a copy, not an alias."""
        ros = _make_synthetic_rectified_order_set()
        result = build_calibration_products(ros)
        original = ros.rectified_orders[0].spatial_frac.copy()
        ros.rectified_orders[0].spatial_frac[:] = 0.0
        np.testing.assert_array_equal(
            result.spatcal_products[0].spatial_frac,
            original,
            err_msg="spatial_frac was not copied; mutation of source affected product",
        )


# ===========================================================================
# 5. Parametric axis-size tests
# ===========================================================================


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 4), (64, 32), (128, 128)])
def test_output_shapes_parametric(n_spectral, n_spatial):
    """WaveCalProduct and SpatCalProduct have correct shapes for various sizes."""
    ros = _make_synthetic_rectified_order_set(
        n_spectral=n_spectral, n_spatial=n_spatial
    )
    result = build_calibration_products(ros)
    for wcp, scp in zip(result.wavecal_products, result.spatcal_products):
        assert wcp.wavelength_um.shape == (n_spectral,)
        assert wcp.spatial_frac.shape == (n_spatial,)
        assert wcp.rectified_flux.shape == (n_spatial, n_spectral)
        assert scp.spatial_frac.shape == (n_spatial,)
        assert scp.detector_rows.shape == (n_spatial,)


# ===========================================================================
# 6. Error handling
# ===========================================================================


class TestBuildCalibrationProductsErrors:
    """Error paths for build_calibration_products."""

    def test_raises_on_empty_rectified_order_set(self):
        """ValueError if rectified_orders has no orders."""
        empty_ros = RectifiedOrderSet(mode="test")
        with pytest.raises(ValueError, match="empty"):
            build_calibration_products(empty_ros)

    def test_raises_on_inconsistent_axis_sizes(self):
        """ValueError if flux shape is inconsistent with spatial_frac/wavelength_um."""
        inconsistent_ros = _make_inconsistent_rectified_order_set()
        with pytest.raises(ValueError, match="inconsistent"):
            build_calibration_products(inconsistent_ros)

    def test_raises_on_empty_mode_string(self):
        """ValueError is not raised for an empty mode string (mode is not validated)."""
        ros = _make_synthetic_rectified_order_set()
        ros.mode = ""
        result = build_calibration_products(ros)
        assert result.mode == ""


# ===========================================================================
# 7. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1CalibrationProductsSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset.

    The chain is:
      1. Flat-order tracing (Stage 1)
      2. Arc-line tracing (Stage 2)
      3. Provisional wavelength mapping (Stage 3)
      4. Coefficient-surface refinement (Stage 5)
      5. Rectification indices (Stage 6)
      6. Rectified orders (Stage 7)
      7. Calibration products (Stage 8)
    """

    @pytest.fixture(scope="class")
    def h1_calibration_products(self):
        """CalibrationProductSet built from the real H1 calibration chain."""
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

        # Stage 1: flat tracing
        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES,
            col_range=(650, 1550),
        )
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))

        # Stage 2: arc tracing
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)

        # Stage 3: provisional wavelength mapping
        wci = read_wavecalinfo("H1")
        ll = read_line_list("H1")
        wav_map = fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )

        # Stage 5: coefficient-surface refinement
        surface = fit_refined_coefficient_surface(wav_map, disp_degree=3)

        # Stage 6: rectification indices
        rect_indices = build_rectification_indices(geom, surface, wav_map=wav_map)

        # Stage 7: rectified orders
        with fits.open(_H1_FLAT_FILES[0]) as hdul:
            detector_image = hdul[0].data.astype(float)
        rectified = build_rectified_orders(detector_image, rect_indices)

        # Stage 8: calibration products
        cal_products = build_calibration_products(rectified)

        return cal_products, rectified

    def test_returns_calibration_product_set(self, h1_calibration_products):
        """build_calibration_products returns a CalibrationProductSet."""
        cal_products, _ = h1_calibration_products
        assert isinstance(cal_products, CalibrationProductSet)

    def test_mode_is_h1(self, h1_calibration_products):
        """mode is 'H1'."""
        cal_products, _ = h1_calibration_products
        assert cal_products.mode == "H1"

    def test_n_orders_matches_rectified(self, h1_calibration_products):
        """n_orders matches the rectified order set."""
        cal_products, rectified = h1_calibration_products
        assert cal_products.n_orders == rectified.n_orders

    def test_n_orders_positive(self, h1_calibration_products):
        """n_orders > 0."""
        cal_products, _ = h1_calibration_products
        assert cal_products.n_orders > 0

    def test_wavelength_axes_monotonic(self, h1_calibration_products):
        """wavelength_um is monotonic for every wavecal product."""
        cal_products, _ = h1_calibration_products
        for wcp in cal_products.wavecal_products:
            diffs = np.diff(wcp.wavelength_um)
            assert np.all(diffs > 0) or np.all(diffs < 0), (
                f"Order {wcp.order}: wavelength axis is not monotonic"
            )

    def test_spatial_axes_in_unit_interval(self, h1_calibration_products):
        """spatial_frac values are in [0, 1] for every product."""
        cal_products, _ = h1_calibration_products
        for wcp in cal_products.wavecal_products:
            assert np.all(wcp.spatial_frac >= 0.0)
            assert np.all(wcp.spatial_frac <= 1.0)
        for scp in cal_products.spatcal_products:
            assert np.all(scp.spatial_frac >= 0.0)
            assert np.all(scp.spatial_frac <= 1.0)

    def test_orders_list_matches_rectified(self, h1_calibration_products):
        """orders list matches the rectified order set's orders list."""
        cal_products, rectified = h1_calibration_products
        assert cal_products.orders == rectified.orders

    def test_get_order_works_for_all_orders(self, h1_calibration_products):
        """get_order succeeds for every order in the calibration product set."""
        cal_products, _ = h1_calibration_products
        for order in cal_products.orders:
            wcp, scp = cal_products.get_order(order)
            assert wcp.order == order
            assert scp.order == order
