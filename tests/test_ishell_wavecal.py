"""
Tests for the iSHELL wavecal and rectification pipeline.

Coverage:
  - Successful wavecal object construction from stored J/H/K calibrations.
  - Polynomial wavelength solution accuracy.
  - RectificationMap creation from populated geometry.
  - Basic rectification behaviour on synthetic inputs.
  - Failure cases for malformed or inconsistent calibration inputs.
  - Arc-line centroid measurement from stored plane-1 data (fit_arc_line_centroids).
  - Centroid-based centerline wavelength solution (build_geometry_from_arc_lines).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyspextool.instruments.ishell.calibrations import (
    FlatInfo,
    LineList,
    LineListEntry,
    WaveCalInfo,
    read_flatinfo,
    read_line_list,
    read_wavecalinfo,
)
from pyspextool.instruments.ishell.geometry import (
    OrderGeometry,
    OrderGeometrySet,
    RectificationMap,
    build_order_geometry_set,
)
from pyspextool.instruments.ishell.wavecal import (
    build_geometry_from_arc_lines,
    build_geometry_from_wavecalinfo,
    build_rectification_maps,
    fit_arc_line_centroids,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

# Modes exercised in parameterised tests – one representative for each band
REPRESENTATIVE_MODES = ["J0", "H1", "K1"]

# All supported modes for broader coverage
ALL_MODES = [
    "J0", "J1", "J2", "J3",
    "H1", "H2", "H3",
    "K1", "K2", "K3", "Kgas",
]


def _make_synthetic_flatinfo(
    mode: str = "K1",
    orders: list[int] | None = None,
    n_orders: int = 3,
    plate_scale: float = 0.125,
    slit_height_arcsec: float = 5.0,
) -> FlatInfo:
    """Construct a minimal synthetic FlatInfo for use in unit tests."""
    if orders is None:
        orders = list(range(233, 233 + n_orders))
    no = len(orders)
    # Each order spans 200 columns starting at 0
    xranges = np.array([[i * 210, i * 210 + 199] for i in range(no)], dtype=int)
    # Simple flat-at-row edge polys
    edge_coeffs = np.array(
        [[[50.0 + i * 40, 0.0, 0.0, 0.0, 0.0],
          [80.0 + i * 40, 0.0, 0.0, 0.0, 0.0]]
         for i in range(no)],
        dtype=float,
    )
    image = np.zeros((2048, 2048), dtype=np.int16)
    return FlatInfo(
        mode=mode,
        orders=orders,
        rotation=5,
        plate_scale_arcsec=plate_scale,
        slit_height_pixels=slit_height_arcsec / plate_scale,
        slit_height_arcsec=slit_height_arcsec,
        slit_range_pixels=(30, 60),
        resolving_power_pixel=23000.0,
        step=5,
        flat_fraction=0.85,
        comm_window=5,
        image=image,
        xranges=xranges,
        edge_coeffs=edge_coeffs,
        edge_degree=4,
    )


def _make_synthetic_wavecalinfo(
    mode: str = "K1",
    orders: list[int] | None = None,
    n_orders: int = 3,
    n_pixels: int = 200,
) -> WaveCalInfo:
    """Construct a minimal synthetic WaveCalInfo with known wavelength values."""
    if orders is None:
        orders = list(range(233, 233 + n_orders))
    no = len(orders)

    # Simple linear wavelength grid: each order gets a unique band
    # starting at 2.0 + i*0.01 µm with dispersion 5e-5 µm/pix
    data = np.full((no, 4, n_pixels), np.nan)
    xranges = np.zeros((no, 2), dtype=int)
    for i in range(no):
        x0 = i * 210
        x1 = x0 + n_pixels - 1
        cols = np.arange(n_pixels, dtype=float) + x0
        wavs = 2.0 + i * 0.01 + cols * 5e-5
        data[i, 0, :] = wavs
        xranges[i] = [x0, x1]

    return WaveCalInfo(
        mode=mode,
        n_orders=no,
        orders=orders,
        resolving_power=70000.0,
        data=data,
        linelist_name="K1_lines.dat",
        wcal_type="2DXD",
        home_order=250,
        disp_degree=3,
        order_degree=2,
        xranges=xranges,
    )


# ===========================================================================
# 1.  build_geometry_from_wavecalinfo – construction from stored calibrations
# ===========================================================================


class TestBuildGeometryFromWaveCalInfo:
    """Geometry population from stored WaveCalInfo and FlatInfo."""

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_returns_order_geometry_set(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        assert isinstance(geom, OrderGeometrySet)

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_mode_attribute(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        assert geom.mode == mode

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_n_orders_correct(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        assert geom.n_orders == fi.n_orders

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_orders_match_flatinfo(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        assert geom.orders == sorted(fi.orders)

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_has_wavelength_solution(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        assert geom.has_wavelength_solution()

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_has_tilt_placeholder(self, mode):
        """tilt_coeffs is set (placeholder zero) for all orders."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        assert geom.has_tilt()

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_has_spatcal(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        assert geom.has_spatcal()

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_wave_coeffs_shape(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        expected_degree = wci.disp_degree
        for g in geom.geometries:
            assert g.wave_coeffs is not None
            assert g.wave_coeffs.ndim == 1
            # degree + 1 terms
            assert len(g.wave_coeffs) == expected_degree + 1

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_tilt_coeffs_zero(self, mode):
        """tilt_coeffs is the placeholder [0.0] for all orders."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        for g in geom.geometries:
            assert g.tilt_coeffs is not None
            np.testing.assert_allclose(g.tilt_coeffs, [0.0])

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_spatcal_coeffs_linear(self, mode):
        """spatcal_coeffs should be [0.0, plate_scale]."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        for g in geom.geometries:
            assert g.spatcal_coeffs is not None
            assert len(g.spatcal_coeffs) == 2
            assert g.spatcal_coeffs[0] == pytest.approx(0.0)
            # The second coefficient should be the plate scale
            assert g.spatcal_coeffs[1] == pytest.approx(fi.plate_scale_arcsec)

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_plane0_values_in_band_range(self, mode):
        """Plane 0 of the stored data cube has values consistent with the
        expected J/H/K wavelength ranges (µm).

        This is a *structural* range check: the stored values happen to fall
        in the expected band for each mode.  It does NOT confirm that plane 0
        is formally documented as wavelengths – that semantic is inferred.
        """
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)

        # Rough band boundaries (µm) used only as a sanity range check
        band_limits = {
            "J": (1.0, 1.4),
            "H": (1.4, 1.9),
            "K": (1.85, 2.5),
        }
        band = mode[0]
        lo, hi = band_limits[band]
        for g in geom.geometries:
            col_mid = 0.5 * (g.x_start + g.x_end)
            val = np.polynomial.polynomial.polyval(col_mid, g.wave_coeffs)
            assert lo < val < hi, (
                f"Mode {mode}, order {g.order}: plane-0 fitted value {val:.4f} "
                f"outside expected band range [{lo}, {hi}] µm"
            )

    def test_mode_mismatch_raises_value_error(self):
        fi = read_flatinfo("J0")
        wci = read_wavecalinfo("K1")
        with pytest.raises(ValueError, match="mode"):
            build_geometry_from_wavecalinfo(wci, fi)

    def test_missing_edge_coeffs_raises_value_error(self):
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        # Remove edge_coeffs
        fi_bad = FlatInfo(
            mode=fi.mode, orders=fi.orders, rotation=fi.rotation,
            plate_scale_arcsec=fi.plate_scale_arcsec,
            slit_height_pixels=fi.slit_height_pixels,
            slit_height_arcsec=fi.slit_height_arcsec,
            slit_range_pixels=fi.slit_range_pixels,
            resolving_power_pixel=fi.resolving_power_pixel,
            step=fi.step, flat_fraction=fi.flat_fraction,
            comm_window=fi.comm_window, image=fi.image,
            xranges=fi.xranges,
            edge_coeffs=None,  # missing
        )
        with pytest.raises(ValueError, match="edge_coeffs"):
            build_geometry_from_wavecalinfo(wci, fi_bad)

    def test_missing_xranges_raises_value_error(self):
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        wci_bad = WaveCalInfo(
            mode=wci.mode, n_orders=wci.n_orders, orders=wci.orders,
            resolving_power=wci.resolving_power, data=wci.data,
            linelist_name=wci.linelist_name, wcal_type=wci.wcal_type,
            home_order=wci.home_order, disp_degree=wci.disp_degree,
            order_degree=wci.order_degree,
            xranges=None,  # missing
        )
        with pytest.raises(ValueError, match="xranges"):
            build_geometry_from_wavecalinfo(wci_bad, fi)

    def test_synthetic_wavelength_accuracy(self):
        """For a synthetic linear wavecal, fitted coefficients match exactly."""
        fi = _make_synthetic_flatinfo(mode="K1", n_orders=2)
        wci = _make_synthetic_wavecalinfo(mode="K1", n_orders=2, n_pixels=200)

        geom = build_geometry_from_wavecalinfo(wci, fi, dispersion_degree=1)
        for i, g in enumerate(geom.geometries):
            # Expected linear model: wave = (2.0 + i*0.01) + col * 5e-5
            # where col is the physical detector column (= array_index + x0).
            # The polynomial coefficient c0 is the value extrapolated to col=0,
            # i.e. c0 = 2.0 + i*0.01 (x0 contribution is absorbed by the fit).
            expected_c0 = 2.0 + i * 0.01
            expected_c1 = 5e-5
            np.testing.assert_allclose(g.wave_coeffs[0], expected_c0, rtol=1e-5)
            np.testing.assert_allclose(g.wave_coeffs[1], expected_c1, rtol=1e-5)

    def test_custom_dispersion_degree(self):
        """dispersion_degree parameter overrides wavecalinfo.disp_degree."""
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        geom = build_geometry_from_wavecalinfo(wci, fi, dispersion_degree=2)
        for g in geom.geometries:
            assert len(g.wave_coeffs) == 3  # degree 2 → 3 terms


# ===========================================================================
# 2.  build_rectification_maps – RectificationMap creation
# ===========================================================================


class TestBuildRectificationMaps:
    """RectificationMap creation from populated geometry."""

    @pytest.fixture()
    def k1_geom(self):
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        return build_geometry_from_wavecalinfo(wci, fi), fi

    def test_returns_list_of_maps(self, k1_geom):
        geom, fi = k1_geom
        maps = build_rectification_maps(geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        assert isinstance(maps, list)

    def test_one_map_per_order(self, k1_geom):
        geom, fi = k1_geom
        maps = build_rectification_maps(geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        assert len(maps) == geom.n_orders

    def test_all_maps_are_rectification_map(self, k1_geom):
        geom, fi = k1_geom
        maps = build_rectification_maps(geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        for m in maps:
            assert isinstance(m, RectificationMap)

    def test_order_numbers_match(self, k1_geom):
        geom, fi = k1_geom
        maps = build_rectification_maps(geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        for m, g in zip(maps, geom.geometries):
            assert m.order == g.order

    def test_output_shape_consistent(self, k1_geom):
        geom, fi = k1_geom
        maps = build_rectification_maps(geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        for m in maps:
            assert m.src_rows.shape == m.output_shape
            assert m.src_cols.shape == m.output_shape

    def test_wavelength_axis_matches_order_range(self, k1_geom):
        geom, fi = k1_geom
        maps = build_rectification_maps(geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        for m, g in zip(maps, geom.geometries):
            expected_n_spec = g.x_end - g.x_start + 1
            assert m.n_spectral == expected_n_spec

    def test_wavelengths_increasing(self, k1_geom):
        """Plane-0 fitted values increase with column for K1 stored data.

        This checks the structural property of the stored reference arrays
        (which are inferred to represent wavelengths); it is not a test of
        a live wavelength measurement.
        """
        geom, fi = k1_geom
        maps = build_rectification_maps(geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        for m in maps:
            assert np.all(np.diff(m.output_wavelengths_um) > 0), (
                f"Wavelengths not monotonically increasing for order {m.order}"
            )

    def test_spatial_axis_symmetric(self, k1_geom):
        geom, fi = k1_geom
        maps = build_rectification_maps(geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        for m in maps:
            arcsec = m.output_spatial_arcsec
            # Should span roughly ±half_slit with zero near centre
            assert arcsec[0] < 0 < arcsec[-1]

    def test_src_rows_within_detector(self, k1_geom):
        """src_rows should be predominantly within iSHELL detector row range.

        The spatial axis may extend slightly beyond [0, 2048) at the order
        edges (where the slit footprint clips the detector boundary).  We
        accept a small margin of one full slit height beyond the hard limits.
        """
        geom, fi = k1_geom
        maps = build_rectification_maps(
            geom, plate_scale_arcsec=fi.plate_scale_arcsec,
            slit_height_arcsec=fi.slit_height_arcsec,
        )
        slit_pix = fi.slit_height_arcsec / fi.plate_scale_arcsec
        for m in maps:
            assert np.all(m.src_rows >= -slit_pix), (
                f"Order {m.order}: src_rows well below 0"
            )
            assert np.all(m.src_rows < 2048 + slit_pix), (
                f"Order {m.order}: src_rows well above 2047"
            )

    def test_no_wave_solution_raises_value_error(self):
        """build_rectification_maps must raise ValueError without wave solution."""
        g = OrderGeometry(
            order=233, x_start=10, x_end=50,
            bottom_edge_coeffs=np.array([20.0, 0.0]),
            top_edge_coeffs=np.array([40.0, 0.0]),
        )
        geom = OrderGeometrySet(mode="K1", geometries=[g])
        with pytest.raises(ValueError, match="wavelength solution"):
            build_rectification_maps(geom, plate_scale_arcsec=0.125)

    def test_custom_slit_height(self, k1_geom):
        geom, fi = k1_geom
        maps_default = build_rectification_maps(geom, plate_scale_arcsec=0.125)
        maps_narrow = build_rectification_maps(
            geom, plate_scale_arcsec=0.125, slit_height_arcsec=2.0
        )
        # Narrower slit → fewer spatial pixels
        for md, mn in zip(maps_default, maps_narrow):
            assert mn.n_spatial <= md.n_spatial

    def test_custom_n_spatial(self, k1_geom):
        geom, fi = k1_geom
        target_n = 20
        maps = build_rectification_maps(
            geom, plate_scale_arcsec=0.125, n_spatial=target_n
        )
        for m in maps:
            assert m.n_spatial == target_n

    def test_synthetic_zero_tilt_src_cols_equal_output_cols(self):
        """With zero tilt, src_cols should equal the output column index."""
        fi = _make_synthetic_flatinfo(mode="K1", n_orders=1)
        wci = _make_synthetic_wavecalinfo(mode="K1", n_orders=1, n_pixels=50)
        geom = build_geometry_from_wavecalinfo(wci, fi, dispersion_degree=1)

        maps = build_rectification_maps(
            geom, plate_scale_arcsec=0.125, n_spatial=5
        )
        m = maps[0]
        g = geom.geometries[0]
        cols_out = np.arange(g.x_start, g.x_end + 1, dtype=float)
        # Each column in the output should map back to that same column
        for j, c in enumerate(cols_out):
            np.testing.assert_allclose(
                m.src_cols[:, j], c,
                atol=1e-10,
                err_msg=f"Zero-tilt: src_col mismatch at output col {c}",
            )


# ===========================================================================
# 3.  _rectify_orders – rectification of synthetic images
# ===========================================================================


class TestRectifyOrders:
    """Tests for ishell._rectify_orders."""

    @pytest.fixture()
    def rectify_func(self):
        from pyspextool.instruments.ishell.ishell import _rectify_orders
        return _rectify_orders

    def test_empty_geometry_returns_copy(self, rectify_func):
        """Empty geometry → output equals input."""
        img = np.arange(100, dtype=float).reshape(10, 10)
        geom = OrderGeometrySet(mode="K1")
        result = rectify_func(img, geom)
        np.testing.assert_array_equal(result, img)

    def test_empty_geometry_output_is_copy_not_view(self, rectify_func):
        img = np.ones((10, 10))
        geom = OrderGeometrySet(mode="K1")
        result = rectify_func(img, geom)
        # Modifying result must not affect input
        result[0, 0] = 999.0
        assert img[0, 0] == pytest.approx(1.0)

    def test_output_shape_matches_input(self, rectify_func):
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        geom = build_geometry_from_wavecalinfo(wci, fi)
        img = np.random.rand(2048, 2048).astype(float)
        result = rectify_func(img, geom)
        assert result.shape == img.shape

    def test_no_wavelength_solution_raises(self, rectify_func):
        g = OrderGeometry(
            order=233, x_start=10, x_end=50,
            bottom_edge_coeffs=np.array([20.0, 0.0]),
            top_edge_coeffs=np.array([40.0, 0.0]),
        )
        geom = OrderGeometrySet(mode="K1", geometries=[g])
        img = np.zeros((100, 100))
        with pytest.raises(ValueError, match="wavelength solution"):
            rectify_func(img, geom)

    def test_zero_tilt_identity_within_order(self, rectify_func):
        """With placeholder zero tilt, output pixels inside the order equal
        the bilinearly-interpolated input (identity mapping)."""
        fi = _make_synthetic_flatinfo(mode="K1", n_orders=1)
        wci = _make_synthetic_wavecalinfo(mode="K1", n_orders=1, n_pixels=100)
        geom = build_geometry_from_wavecalinfo(wci, fi, dispersion_degree=1)

        # Place a recognisable pattern in the order region
        img = np.zeros((200, 700))
        g = geom.geometries[0]
        x0, x1 = g.x_start, g.x_end
        r_bot = int(np.ceil(g.eval_bottom_edge(float(x0))))
        r_top = int(np.floor(g.eval_top_edge(float(x0))))
        img[r_bot:r_top + 1, x0:x1 + 1] = 1.0

        result = rectify_func(img, geom)

        # Within the order interior the rectified values should be ≈ 1.0
        # (identity transformation when tilt = 0)
        interior_rows = slice(r_bot + 1, r_top)  # avoid edges where NaN boundary may occur
        interior_cols = slice(x0 + 1, x1)
        patch = result[interior_rows, interior_cols]
        valid = ~np.isnan(patch)
        assert valid.sum() > 0, "No valid pixels found in order interior"
        np.testing.assert_allclose(
            patch[valid], 1.0, atol=0.02,
            err_msg="Zero-tilt rectification must preserve pixel values",
        )

    def test_pixels_outside_orders_are_nan(self, rectify_func):
        """Pixels outside all order footprints must be NaN in the output."""
        fi = _make_synthetic_flatinfo(mode="K1", n_orders=1)
        wci = _make_synthetic_wavecalinfo(mode="K1", n_orders=1, n_pixels=100)
        geom = build_geometry_from_wavecalinfo(wci, fi, dispersion_degree=1)

        img = np.ones((200, 700))
        result = rectify_func(img, geom)

        # Pixel (0, 0) is far from the order and must be NaN
        assert np.isnan(result[0, 0])

    def test_real_k1_geometry_runs_without_error(self, rectify_func):
        """Smoke test: resampling a random image with K1 geometry completes
        without error and produces some non-NaN output (structural test only;
        no physical correctness is implied with the placeholder zero-tilt)."""
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        geom = build_geometry_from_wavecalinfo(wci, fi)
        img = np.random.rand(2048, 2048).astype(float)
        result = rectify_func(img, geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        assert result.shape == (2048, 2048)
        # Some pixels should be non-NaN (the order footprints)
        assert np.sum(~np.isnan(result)) > 0


# ===========================================================================
# 4.  Integration: build_geometry_from_wavecalinfo → _rectify_orders pipeline
# ===========================================================================


class TestIntegrationPipeline:
    """End-to-end pipeline tests (calibration reader → wavecal → rectification)."""

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_full_pipeline_produces_valid_output(self, mode):
        """Structural smoke test: read cals → build geometry (placeholder tilt)
        → build maps → resample.  Verifies the pipeline runs without error and
        produces an output with the correct shape and some non-NaN pixels.
        No physical correctness of wavelengths or rectification is asserted."""
        from pyspextool.instruments.ishell.ishell import _rectify_orders

        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        maps = build_rectification_maps(
            geom,
            plate_scale_arcsec=fi.plate_scale_arcsec,
            slit_height_arcsec=fi.slit_height_arcsec,
        )
        assert len(maps) == geom.n_orders

        img = np.random.rand(2048, 2048).astype(float)
        result = _rectify_orders(img, geom, plate_scale_arcsec=fi.plate_scale_arcsec)
        assert result.shape == img.shape
        assert np.sum(~np.isnan(result)) > 0

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_all_modes_geometry_build_succeeds(self, mode):
        """build_geometry_from_wavecalinfo must succeed for every supported mode
        and produce an OrderGeometrySet with all structural fields populated
        (wave_coeffs from plane 0, placeholder tilt, linear spatcal)."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        geom = build_geometry_from_wavecalinfo(wci, fi)
        assert geom.has_wavelength_solution()
        assert geom.has_tilt()      # placeholder zero tilt is set
        assert geom.has_spatcal()   # placeholder linear spatcal is set
        assert geom.n_orders > 0


# ===========================================================================
# 5.  Calibration reader backward compatibility
# ===========================================================================


class TestCalibrationsBackwardCompatibility:
    """New optional fields must not break existing calibration reader tests."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_read_flatinfo_xranges_available(self, mode):
        """FlatInfo.xranges must be populated (not None) for all modes."""
        fi = read_flatinfo(mode)
        assert fi.xranges is not None
        assert fi.xranges.shape == (fi.n_orders, 2)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_read_flatinfo_edge_coeffs_available(self, mode):
        """FlatInfo.edge_coeffs must be populated for all modes."""
        fi = read_flatinfo(mode)
        assert fi.edge_coeffs is not None
        assert fi.edge_coeffs.ndim == 3
        assert fi.edge_coeffs.shape[0] == fi.n_orders
        assert fi.edge_coeffs.shape[1] == 2

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_read_wavecalinfo_xranges_available(self, mode):
        """WaveCalInfo.xranges must be populated for all modes."""
        wci = read_wavecalinfo(mode)
        assert wci.xranges is not None
        assert wci.xranges.shape == (wci.n_orders, 2)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_build_order_geometry_set_from_flatinfo(self, mode):
        """build_order_geometry_set must work with the new FlatInfo fields."""
        fi = read_flatinfo(mode)
        assert fi.xranges is not None
        assert fi.edge_coeffs is not None
        geom_set = build_order_geometry_set(
            mode=mode,
            orders=fi.orders,
            x_ranges=fi.xranges,
            edge_coeffs=fi.edge_coeffs,
        )
        assert geom_set.n_orders == fi.n_orders
        assert geom_set.orders == sorted(fi.orders)

    def test_flatinfo_without_optional_fields_still_valid(self):
        """FlatInfo constructed without optional fields must still work."""
        fi = FlatInfo(
            mode="K1",
            orders=[233, 234],
            rotation=5,
            plate_scale_arcsec=0.125,
            slit_height_pixels=40.0,
            slit_height_arcsec=5.0,
            slit_range_pixels=(30, 60),
            resolving_power_pixel=23000.0,
            step=5,
            flat_fraction=0.85,
            comm_window=5,
            image=np.zeros((2048, 2048), dtype=np.int16),
        )
        assert fi.xranges is None
        assert fi.edge_coeffs is None
        assert fi.edge_degree is None
        assert fi.n_orders == 2

    def test_wavecalinfo_without_xranges_still_valid(self):
        """WaveCalInfo without xranges must still pass basic validation."""
        wci = WaveCalInfo(
            mode="K1",
            n_orders=2,
            orders=[233, 234],
            resolving_power=70000.0,
            data=np.zeros((2, 4, 100)),
            linelist_name="K1_lines.dat",
            wcal_type="2DXD",
            home_order=250,
            disp_degree=3,
            order_degree=2,
        )
        assert wci.xranges is None
        assert wci.n_pixels == 100


# ===========================================================================
# 6.  fit_arc_line_centroids – arc-line centroid measurement from stored data
# ===========================================================================


def _make_synthetic_line_list(
    mode: str = "K1",
    orders: list[int] | None = None,
    n_lines_per_order: int = 6,
    start_wav: float = 2.20,
    disp: float = 5e-4,
) -> LineList:
    """Construct a synthetic LineList for use in centroid-measurement tests."""
    if orders is None:
        orders = list(range(233, 233 + 3))
    entries = []
    for o_idx, order in enumerate(orders):
        for j in range(n_lines_per_order):
            wav = start_wav + o_idx * 0.01 + j * disp * 50
            entries.append(
                LineListEntry(
                    order=order,
                    wavelength_um=wav,
                    species="Ar I",
                    fit_window_angstrom=1.0,
                    fit_type="G",
                    fit_n_terms=5,
                )
            )
    return LineList(mode=mode, entries=entries)


class TestFitArcLineCentroids:
    """Tests for arc-line centroid measurement from stored plane-1 data."""

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_returns_dict_keyed_by_order(self, mode):
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        result = fit_arc_line_centroids(wci, ll)
        assert isinstance(result, dict)
        # Every order in wavecalinfo must appear as a key
        for order in wci.orders:
            assert order in result

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_results_are_lists_of_triples(self, mode):
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        result = fit_arc_line_centroids(wci, ll)
        for order, measurements in result.items():
            assert isinstance(measurements, list)
            for item in measurements:
                assert len(item) == 3, (
                    f"Order {order}: expected (centroid, wavelength, snr) triple, got {item}"
                )

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_centroids_within_column_range(self, mode):
        """Measured centroids must fall within the valid detector column range."""
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        result = fit_arc_line_centroids(wci, ll)
        for i, order in enumerate(wci.orders):
            x0, x1 = int(wci.xranges[i, 0]), int(wci.xranges[i, 1])
            for centroid, wav, snr in result[order]:
                assert x0 <= centroid <= x1, (
                    f"Order {order}: centroid {centroid:.2f} outside [{x0},{x1}]"
                )

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_snr_above_threshold(self, mode):
        """All accepted centroids must have SNR >= snr_min."""
        snr_min = 5.0
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        result = fit_arc_line_centroids(wci, ll, snr_min=snr_min)
        for order, measurements in result.items():
            for centroid, wav, snr in measurements:
                assert snr >= snr_min, (
                    f"Order {order}: accepted centroid with SNR {snr:.1f} < {snr_min}"
                )

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_wavelengths_match_line_list(self, mode):
        """Returned wavelengths must match the input line list wavelengths."""
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        result = fit_arc_line_centroids(wci, ll)
        known_wavs = {e.wavelength_um for e in ll.entries}
        for order, measurements in result.items():
            for centroid, wav, snr in measurements:
                assert wav in known_wavs, (
                    f"Order {order}: returned wavelength {wav} not in line list"
                )

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_at_least_one_order_with_measurements(self, mode):
        """At least one order must have accepted centroid measurements."""
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        result = fit_arc_line_centroids(wci, ll)
        total = sum(len(v) for v in result.values())
        assert total > 0, f"Mode {mode}: no arc lines identified"

    def test_missing_xranges_raises_value_error(self):
        """fit_arc_line_centroids must raise ValueError when xranges is None."""
        wci = read_wavecalinfo("K1")
        ll = read_line_list("K1")
        wci_bad = WaveCalInfo(
            mode=wci.mode, n_orders=wci.n_orders, orders=wci.orders,
            resolving_power=wci.resolving_power, data=wci.data,
            linelist_name=wci.linelist_name, wcal_type=wci.wcal_type,
            home_order=wci.home_order, disp_degree=wci.disp_degree,
            order_degree=wci.order_degree,
            xranges=None,
        )
        with pytest.raises(ValueError, match="xranges"):
            fit_arc_line_centroids(wci_bad, ll)

    def test_empty_line_list_returns_empty_results(self):
        """An empty line list should return empty result lists for all orders."""
        wci = read_wavecalinfo("K1")
        empty_ll = LineList(mode="K1", entries=[])
        result = fit_arc_line_centroids(wci, empty_ll)
        for order, measurements in result.items():
            assert measurements == [], (
                f"Order {order}: expected empty list for empty line list"
            )

    def test_very_high_snr_threshold_rejects_all(self):
        """An extremely high SNR threshold must result in no accepted lines."""
        wci = read_wavecalinfo("K1")
        ll = read_line_list("K1")
        result = fit_arc_line_centroids(wci, ll, snr_min=1e9)
        total = sum(len(v) for v in result.values())
        assert total == 0, (
            f"Expected no accepted lines at SNR=1e9, got {total}"
        )

    def test_centroids_near_predicted_wavelength(self):
        """Measured centroid positions (evaluated against the plane-0 wavelength
        grid) must be close to the known line wavelengths from the line list.

        This verifies that centroids are located near the correct spectral
        features, not at unrelated peaks in the plane-1 data.
        """
        mode = "K1"
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        result = fit_arc_line_centroids(wci, ll, snr_min=10.0)

        for i, order in enumerate(wci.orders):
            measurements = result[order]
            if not measurements:
                continue
            # Build wavelength function from plane 0 for this order
            wav_arr = wci.data[i, 0, :]
            valid = ~np.isnan(wav_arr)
            x0 = int(wci.xranges[i, 0])
            cols_v = np.arange(int(valid.sum()), dtype=float) + x0
            wav_v = wav_arr[valid]

            for centroid, known_wav, snr in measurements:
                # Interpolate wavelength at centroid from the reference grid
                measured_wav = float(np.interp(centroid, cols_v, wav_v))
                # Allow ±0.005 µm (5 nm) tolerance
                assert abs(measured_wav - known_wav) < 0.005, (
                    f"Order {order}: centroid at {centroid:.1f} px → wav "
                    f"{measured_wav:.6f} µm; expected {known_wav:.6f} µm"
                )


# ===========================================================================
# 7.  build_geometry_from_arc_lines – centroid-based centerline wavelength solution
# ===========================================================================


class TestBuildGeometryFromArcLines:
    """Tests for the centroid-based centerline wavelength solution."""

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_returns_order_geometry_set(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        geom = build_geometry_from_arc_lines(wci, fi, ll)
        assert isinstance(geom, OrderGeometrySet)

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_mode_attribute(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        geom = build_geometry_from_arc_lines(wci, fi, ll)
        assert geom.mode == mode

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_n_orders_correct(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        geom = build_geometry_from_arc_lines(wci, fi, ll)
        assert geom.n_orders == fi.n_orders

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_has_wavelength_solution(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        geom = build_geometry_from_arc_lines(wci, fi, ll)
        assert geom.has_wavelength_solution()

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_wave_coeffs_in_band_range(self, mode):
        """Wavelength solution at mid-order column must fall in the correct band."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        geom = build_geometry_from_arc_lines(wci, fi, ll)

        band_limits = {
            "J": (1.0, 1.4),
            "H": (1.4, 1.9),
            "K": (1.85, 2.5),
        }
        band = mode[0]
        lo, hi = band_limits[band]
        for g in geom.geometries:
            col_mid = 0.5 * (g.x_start + g.x_end)
            val = float(np.polynomial.polynomial.polyval(col_mid, g.wave_coeffs))
            assert lo < val < hi, (
                f"Mode {mode}, order {g.order}: wavelength {val:.4f} µm "
                f"outside band [{lo}, {hi}]"
            )

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_tilt_is_placeholder_zero(self, mode):
        """Tilt coefficients must be placeholder zero – the stored 1-D
        centerline data cannot provide tilt measurement (see module
        docstring for limitation)."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        geom = build_geometry_from_arc_lines(wci, fi, ll)
        for g in geom.geometries:
            assert g.tilt_coeffs is not None
            np.testing.assert_allclose(
                g.tilt_coeffs, [0.0],
                err_msg=(
                    f"Order {g.order}: tilt_coeffs must be placeholder zero "
                    "(tilt requires 2-D arc images; stored 1-D centerline data insufficient)"
                ),
            )

    @pytest.mark.parametrize("mode", REPRESENTATIVE_MODES)
    def test_spatcal_coeffs_linear(self, mode):
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        geom = build_geometry_from_arc_lines(wci, fi, ll)
        for g in geom.geometries:
            assert g.spatcal_coeffs is not None
            assert len(g.spatcal_coeffs) == 2
            assert g.spatcal_coeffs[0] == pytest.approx(0.0)
            assert g.spatcal_coeffs[1] == pytest.approx(fi.plate_scale_arcsec)

    def test_mode_mismatch_raises_value_error(self):
        fi = read_flatinfo("J0")
        wci = read_wavecalinfo("K1")
        ll = read_line_list("K1")
        with pytest.raises(ValueError, match="mode"):
            build_geometry_from_arc_lines(wci, fi, ll)

    def test_missing_edge_coeffs_raises_value_error(self):
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        ll = read_line_list("K1")
        fi_bad = FlatInfo(
            mode=fi.mode, orders=fi.orders, rotation=fi.rotation,
            plate_scale_arcsec=fi.plate_scale_arcsec,
            slit_height_pixels=fi.slit_height_pixels,
            slit_height_arcsec=fi.slit_height_arcsec,
            slit_range_pixels=fi.slit_range_pixels,
            resolving_power_pixel=fi.resolving_power_pixel,
            step=fi.step, flat_fraction=fi.flat_fraction,
            comm_window=fi.comm_window, image=fi.image,
            xranges=fi.xranges,
            edge_coeffs=None,
        )
        with pytest.raises(ValueError, match="edge_coeffs"):
            build_geometry_from_arc_lines(wci, fi_bad, ll)

    def test_fallback_to_bootstrap_when_few_lines(self):
        """Orders with fewer accepted lines than min_lines_per_order must
        fall back to the plane-0 bootstrap and emit RuntimeWarning."""
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        ll = read_line_list("K1")
        # Use an absurdly high SNR threshold so nothing is accepted → all fallback
        with pytest.warns(RuntimeWarning, match="falling back"):
            geom = build_geometry_from_arc_lines(wci, fi, ll, snr_min=1e9)
        # Result must still be a valid geometry set
        assert isinstance(geom, OrderGeometrySet)
        assert geom.has_wavelength_solution()

    def test_measured_orders_differ_from_bootstrap(self):
        """For orders with enough arc lines, the centroid-based wave_coeffs must
        differ from the bootstrap (plane-0) coefficients.

        Both are derived from the same stored data but via different fitting
        procedures: the centroid-based path fits measured positions from plane-1
        data while the bootstrap fits the plane-0 wavelength array directly.
        """
        mode = "K1"
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)

        geom_boot = build_geometry_from_wavecalinfo(wci, fi)
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            geom_meas = build_geometry_from_arc_lines(wci, fi, ll, snr_min=3.0)

        # Find an order that was fitted from measured centroids
        centroids = fit_arc_line_centroids(wci, ll, snr_min=3.0)
        measured_orders = {o for o, v in centroids.items() if len(v) >= 4}

        if not measured_orders:
            pytest.skip("No orders with enough measured centroids for K1 at snr_min=3")

        any_differ = False
        for g_boot, g_meas in zip(geom_boot.geometries, geom_meas.geometries):
            if g_boot.order in measured_orders:
                # Coefficients need not be identical (different fitting approach)
                if not np.allclose(g_boot.wave_coeffs, g_meas.wave_coeffs, rtol=1e-12):
                    any_differ = True
                    break

        assert any_differ, (
            "Measured and bootstrap wave_coeffs are identical for all measured "
            "orders; the measured path should produce different coefficients."
        )

    def test_custom_dispersion_degree(self):
        """dispersion_degree parameter overrides wavecalinfo.disp_degree."""
        fi = read_flatinfo("K1")
        wci = read_wavecalinfo("K1")
        ll = read_line_list("K1")
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            geom = build_geometry_from_arc_lines(wci, fi, ll, dispersion_degree=2)
        for g in geom.geometries:
            assert len(g.wave_coeffs) == 3  # degree 2 → 3 terms

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_all_modes_succeed(self, mode):
        """build_geometry_from_arc_lines must complete without error for
        every supported J/H/K mode and return a valid OrderGeometrySet
        with a centerline wavelength solution."""
        fi = read_flatinfo(mode)
        wci = read_wavecalinfo(mode)
        ll = read_line_list(mode)
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            geom = build_geometry_from_arc_lines(wci, fi, ll)
        assert geom.has_wavelength_solution()
        assert geom.has_tilt()
        assert geom.n_orders > 0

    def test_synthetic_measured_wave_coeffs(self):
        """For a synthetic arc spectrum with known Gaussian lines, the
        centroid-based wavelength solution must recover the correct wavelengths
        to high accuracy."""
        import scipy.stats

        # Build a synthetic wavecalinfo with plane-0 wavelength grid and
        # plane-1 arc spectrum containing narrow Gaussian lines
        n_orders = 2
        n_pixels = 400
        orders = [233, 234]
        x0 = 50
        disp = 5e-4  # µm/pixel

        data = np.full((n_orders, 4, n_pixels), np.nan)
        xranges = np.zeros((n_orders, 2), dtype=int)

        # Known line positions: 3 lines per order at fixed pixel offsets
        line_pixel_offsets = [80, 200, 320]  # pixel offsets into the order
        known_wavs = []

        for i, order in enumerate(orders):
            cols = np.arange(n_pixels, dtype=float) + x0
            wavs = 2.20 + i * 0.05 + cols * disp
            data[i, 0, :] = wavs

            # Build synthetic arc spectrum with Gaussian lines
            arc = np.random.default_rng(42 + i).normal(0, 0.5, n_pixels)
            for pix_offset in line_pixel_offsets:
                amplitude = 200.0
                sigma = 1.5
                arc += amplitude * np.exp(
                    -0.5 * ((np.arange(n_pixels) - pix_offset) / sigma) ** 2
                )

            data[i, 1, :] = arc
            xranges[i] = [x0, x0 + n_pixels - 1]

            order_wavs = [float(wavs[p]) for p in line_pixel_offsets]
            known_wavs.append(order_wavs)

        wci = WaveCalInfo(
            mode="K1",
            n_orders=n_orders,
            orders=orders,
            resolving_power=70000.0,
            data=data,
            linelist_name="K1_lines.dat",
            wcal_type="2DXD",
            home_order=250,
            disp_degree=1,
            order_degree=2,
            xranges=xranges,
        )

        # Build a LineList with the known lines
        entries = []
        for i, order in enumerate(orders):
            for wav in known_wavs[i]:
                entries.append(
                    LineListEntry(
                        order=order,
                        wavelength_um=wav,
                        species="Ar I",
                        fit_window_angstrom=1.0,
                        fit_type="G",
                        fit_n_terms=5,
                    )
                )
        ll = LineList(mode="K1", entries=entries)

        # Measure centroids
        result = fit_arc_line_centroids(wci, ll, snr_min=5.0)

        for i, order in enumerate(orders):
            measurements = result[order]
            assert len(measurements) >= 2, (
                f"Order {order}: expected >= 2 measured centroids, got {len(measurements)}"
            )
            for centroid, wav, snr in measurements:
                # expected_pix is the column number (= array_index + x0)
                expected_pix = (wav - (2.20 + i * 0.05)) / disp
                # Centroid should be within ±1 pixel of the true position
                assert abs(centroid - expected_pix) < 1.0, (
                    f"Order {order}: centroid {centroid:.2f} != expected "
                    f"{expected_pix:.2f} for line at {wav:.6f} µm"
                )
