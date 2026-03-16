"""
Tests for the provisional spectral-extraction scaffold
(extracted_spectra.py).

Coverage:
  - ExtractedOrderSpectrum: construction and field access.
  - ExtractedSpectrumSet: construction, get_order, orders, n_orders.
  - extract_rectified_orders on synthetic data:
      * returns correct type and mode,
      * n_orders matches rectified order set,
      * sum vs mean behaviour,
      * shape consistency (n_spectral matches wavelength axis),
      * wavelength-axis propagation,
      * NaN handling (all-NaN column → NaN in output),
  - Error handling:
      * empty rectified order set → ValueError,
      * invalid extraction method → ValueError,
      * non-2-D flux image → ValueError.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * number of extracted orders matches rectified order set,
      * wavelength axes are monotonic,
      * extracted flux arrays are finite where expected,
      * output shapes are correct.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from pyspextool.instruments.ishell.extracted_spectra import (
    ExtractedOrderSpectrum,
    ExtractedSpectrumSet,
    extract_rectified_orders,
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
# Synthetic-data helpers
# ---------------------------------------------------------------------------

_NROWS = 200
_NCOLS = 800
_N_ORDERS = 3
_N_SPECTRAL = 32
_N_SPATIAL = 16

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


def _make_synthetic_rectification_index_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
) -> RectificationIndexSet:
    """Build a RectificationIndexSet from the synthetic order parameters."""
    index_orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        cr = float(p["center_row"])
        hw = float(p["half_width"])
        col_start = float(p["col_start"])
        col_end = float(p["col_end"])

        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        output_wavs = np.linspace(wav_start, wav_end, n_spectral)
        output_spatial_frac = np.linspace(0.0, 1.0, n_spatial)

        src_cols = np.linspace(col_start, col_end, n_spectral)
        bottom = cr - hw
        top = cr + hw
        src_rows = (
            bottom + output_spatial_frac[:, np.newaxis] * (top - bottom)
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


def _make_synthetic_detector_image(
    n_rows: int = _NROWS, n_cols: int = _NCOLS
) -> np.ndarray:
    """Return a constant-value detector image (all pixels = 1.0)."""
    return np.ones((n_rows, n_cols), dtype=float)


def _make_synthetic_rectified_order_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
    fill_value: float = 2.0,
) -> RectifiedOrderSet:
    """Build a RectifiedOrderSet with constant-value flux images.

    Each order's flux array is filled with *fill_value*.  This makes it
    easy to verify sum (fill_value * n_spatial) and mean (fill_value)
    extraction results.
    """
    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=np.linspace(wav_start, wav_end, n_spectral),
                spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                flux=np.full((n_spatial, n_spectral), fill_value),
                source_image_shape=(_NROWS, _NCOLS),
            )
        )
    return RectifiedOrderSet(
        mode="H1_test",
        rectified_orders=orders,
        source_image_shape=(_NROWS, _NCOLS),
    )


# ===========================================================================
# 1. ExtractedOrderSpectrum: construction
# ===========================================================================


class TestExtractedOrderSpectrumConstruction:
    """Unit tests for ExtractedOrderSpectrum construction and field access."""

    def _make_minimal(self) -> ExtractedOrderSpectrum:
        n = 32
        return ExtractedOrderSpectrum(
            order=311,
            wavelength_um=np.linspace(1.64, 1.69, n),
            flux=np.ones(n),
            method="sum",
            n_spatial_used=16,
        )

    def test_construction(self):
        """ExtractedOrderSpectrum can be constructed."""
        sp = self._make_minimal()
        assert sp.order == 311

    def test_n_spectral(self):
        """n_spectral matches length of wavelength_um."""
        sp = self._make_minimal()
        assert sp.n_spectral == 32

    def test_variance_default_none(self):
        """variance defaults to None."""
        sp = self._make_minimal()
        assert sp.variance is None

    def test_method_stored(self):
        """method is stored correctly."""
        sp = self._make_minimal()
        assert sp.method == "sum"

    def test_n_spatial_used_stored(self):
        """n_spatial_used is stored correctly."""
        sp = self._make_minimal()
        assert sp.n_spatial_used == 16

    def test_flux_shape(self):
        """flux shape matches n_spectral."""
        sp = self._make_minimal()
        assert sp.flux.shape == (sp.n_spectral,)


# ===========================================================================
# 2. ExtractedSpectrumSet: construction and interface
# ===========================================================================


class TestExtractedSpectrumSetConstruction:
    """Unit tests for ExtractedSpectrumSet construction and interface."""

    def _make_set(self) -> ExtractedSpectrumSet:
        n = 32
        spectra = [
            ExtractedOrderSpectrum(
                order=311 + i * 4,
                wavelength_um=np.linspace(1.55 + i * 0.05, 1.59 + i * 0.05, n),
                flux=np.ones(n) * (i + 1),
                method="sum",
                n_spatial_used=16,
            )
            for i in range(3)
        ]
        return ExtractedSpectrumSet(mode="H1_test", spectra=spectra)

    def test_mode_stored(self):
        """mode is stored correctly."""
        ss = self._make_set()
        assert ss.mode == "H1_test"

    def test_n_orders(self):
        """n_orders matches the number of spectra."""
        ss = self._make_set()
        assert ss.n_orders == 3

    def test_orders_list(self):
        """orders returns the list of order numbers in storage order."""
        ss = self._make_set()
        assert ss.orders == [311, 315, 319]

    def test_get_order_existing(self):
        """get_order returns the correct spectrum for a known order."""
        ss = self._make_set()
        sp = ss.get_order(311)
        assert sp.order == 311

    def test_get_order_missing_raises(self):
        """get_order raises KeyError for an unknown order."""
        ss = self._make_set()
        with pytest.raises(KeyError):
            ss.get_order(999)

    def test_empty_set(self):
        """An empty ExtractedSpectrumSet is valid."""
        ss = ExtractedSpectrumSet(mode="H1_test")
        assert ss.n_orders == 0
        assert ss.orders == []


# ===========================================================================
# 3. extract_rectified_orders: synthetic data
# ===========================================================================


class TestExtractRectifiedOrdersSynthetic:
    """Synthetic-data tests for extract_rectified_orders."""

    def test_returns_extracted_spectrum_set(self):
        """extract_rectified_orders returns an ExtractedSpectrumSet."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros)
        assert isinstance(result, ExtractedSpectrumSet)

    def test_mode_propagated(self):
        """mode is propagated from the rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros)
        assert result.mode == ros.mode

    def test_n_orders_matches(self):
        """n_orders matches the input rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros)
        assert result.n_orders == ros.n_orders

    def test_orders_list_matches(self):
        """orders list matches the input rectified order set."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros)
        assert result.orders == ros.orders

    def test_sum_output_shape(self):
        """Extracted flux has shape (n_spectral,) for sum method."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros, method="sum")
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.flux.shape == (ro.n_spectral,)

    def test_mean_output_shape(self):
        """Extracted flux has shape (n_spectral,) for mean method."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros, method="mean")
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.flux.shape == (ro.n_spectral,)

    def test_sum_values_constant_image(self):
        """sum extraction of a constant-flux image equals fill * n_spatial."""
        fill = 3.0
        ros = _make_synthetic_rectified_order_set(
            n_spectral=_N_SPECTRAL, n_spatial=_N_SPATIAL, fill_value=fill
        )
        result = extract_rectified_orders(ros, method="sum")
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            expected = fill * ro.n_spatial
            np.testing.assert_allclose(
                sp.flux,
                expected,
                rtol=1e-12,
                err_msg=f"Order {sp.order}: sum extraction value incorrect",
            )

    def test_mean_values_constant_image(self):
        """mean extraction of a constant-flux image equals the fill value."""
        fill = 7.0
        ros = _make_synthetic_rectified_order_set(
            n_spectral=_N_SPECTRAL, n_spatial=_N_SPATIAL, fill_value=fill
        )
        result = extract_rectified_orders(ros, method="mean")
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            np.testing.assert_allclose(
                sp.flux,
                fill,
                rtol=1e-12,
                err_msg=f"Order {sp.order}: mean extraction value incorrect",
            )

    def test_sum_vs_mean_ratio(self):
        """sum result equals mean result * n_spatial for uniform flux."""
        fill = 5.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        sum_result = extract_rectified_orders(ros, method="sum")
        mean_result = extract_rectified_orders(ros, method="mean")
        for sp_sum, sp_mean, ro in zip(
            sum_result.spectra, mean_result.spectra, ros.rectified_orders
        ):
            np.testing.assert_allclose(
                sp_sum.flux,
                sp_mean.flux * ro.n_spatial,
                rtol=1e-12,
                err_msg=f"Order {sp_sum.order}: sum != mean * n_spatial",
            )

    def test_wavelength_axis_propagated(self):
        """wavelength_um is copied from the rectified order."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            np.testing.assert_array_equal(
                sp.wavelength_um,
                ro.wavelength_um,
                err_msg=f"Order {sp.order}: wavelength_um mismatch",
            )

    def test_wavelength_axis_is_copy(self):
        """wavelength_um in the output is a copy, not an alias."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros)
        original = ros.rectified_orders[0].wavelength_um.copy()
        ros.rectified_orders[0].wavelength_um[:] = 0.0
        np.testing.assert_array_equal(
            result.spectra[0].wavelength_um,
            original,
            err_msg="wavelength_um was not copied; mutation of source affected output",
        )

    def test_method_stored_sum(self):
        """method field is 'sum' when sum method is used."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros, method="sum")
        for sp in result.spectra:
            assert sp.method == "sum"

    def test_method_stored_mean(self):
        """method field is 'mean' when mean method is used."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros, method="mean")
        for sp in result.spectra:
            assert sp.method == "mean"

    def test_variance_is_none(self):
        """variance is None in this scaffold stage."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros)
        for sp in result.spectra:
            assert sp.variance is None

    def test_n_spatial_used_all_valid(self):
        """n_spatial_used equals n_spatial when no NaN rows are present."""
        ros = _make_synthetic_rectified_order_set()
        result = extract_rectified_orders(ros)
        for sp, ro in zip(result.spectra, ros.rectified_orders):
            assert sp.n_spatial_used == ro.n_spatial

    def test_nan_column_produces_nan_in_sum(self):
        """A spectral column that is all-NaN produces NaN in the sum output."""
        ros = _make_synthetic_rectified_order_set()
        # Set the first spectral column of the first order to all NaN.
        ros.rectified_orders[0].flux[:, 0] = np.nan
        result = extract_rectified_orders(ros, method="sum")
        assert np.isnan(result.spectra[0].flux[0]), \
            "Expected NaN at the all-NaN spectral column in sum output"
        # Other columns should be finite.
        assert np.all(np.isfinite(result.spectra[0].flux[1:]))

    def test_nan_column_produces_nan_in_mean(self):
        """A spectral column that is all-NaN produces NaN in the mean output."""
        ros = _make_synthetic_rectified_order_set()
        ros.rectified_orders[0].flux[:, 0] = np.nan
        result = extract_rectified_orders(ros, method="mean")
        assert np.isnan(result.spectra[0].flux[0]), \
            "Expected NaN at the all-NaN spectral column in mean output"
        assert np.all(np.isfinite(result.spectra[0].flux[1:]))

    def test_partial_nan_row_excluded_from_mean(self):
        """A single NaN spatial row is excluded from the mean without NaN output."""
        ros = _make_synthetic_rectified_order_set(fill_value=4.0)
        # Zero-out one spatial row: remaining rows still have fill=4.0.
        ros.rectified_orders[0].flux[0, :] = np.nan
        result = extract_rectified_orders(ros, method="mean")
        # Mean of the remaining (_N_SPATIAL - 1) rows of 4.0 is still 4.0.
        np.testing.assert_allclose(
            result.spectra[0].flux,
            4.0,
            rtol=1e-12,
        )

    def test_partial_nan_row_excluded_from_sum(self):
        """A single NaN spatial row is excluded from the sum."""
        fill = 2.0
        ros = _make_synthetic_rectified_order_set(fill_value=fill)
        ros.rectified_orders[0].flux[0, :] = np.nan
        result = extract_rectified_orders(ros, method="sum")
        expected = fill * (_N_SPATIAL - 1)
        np.testing.assert_allclose(result.spectra[0].flux, expected, rtol=1e-12)

    def test_n_spatial_used_with_nan_row(self):
        """n_spatial_used decreases when a spatial row is all-NaN."""
        ros = _make_synthetic_rectified_order_set()
        ros.rectified_orders[0].flux[0, :] = np.nan
        result = extract_rectified_orders(ros)
        assert result.spectra[0].n_spatial_used == _N_SPATIAL - 1


# ===========================================================================
# 4. Parametric shape-consistency tests
# ===========================================================================


@pytest.mark.parametrize("n_spectral,n_spatial", [(8, 4), (64, 32), (128, 64)])
def test_output_shapes_parametric(n_spectral, n_spatial):
    """Extracted spectra have correct shapes for various (n_spectral, n_spatial)."""
    ros = _make_synthetic_rectified_order_set(
        n_spectral=n_spectral, n_spatial=n_spatial
    )
    for method in ("sum", "mean"):
        result = extract_rectified_orders(ros, method=method)
        for sp in result.spectra:
            assert sp.flux.shape == (n_spectral,), (
                f"method={method}: flux shape {sp.flux.shape} != ({n_spectral},)"
            )
            assert sp.wavelength_um.shape == (n_spectral,)


# ===========================================================================
# 5. Error handling
# ===========================================================================


class TestExtractRectifiedOrdersErrors:
    """Error paths for extract_rectified_orders."""

    def test_raises_on_empty_rectified_order_set(self):
        """ValueError if rectified_orders has no orders."""
        empty_ros = RectifiedOrderSet(mode="test")
        with pytest.raises(ValueError, match="empty"):
            extract_rectified_orders(empty_ros)

    def test_raises_on_invalid_method(self):
        """ValueError if method is not 'sum' or 'mean'."""
        ros = _make_synthetic_rectified_order_set()
        with pytest.raises(ValueError, match="method"):
            extract_rectified_orders(ros, method="optimal")

    def test_raises_on_non_2d_flux(self):
        """ValueError if a flux array is not 2-D."""
        ros = _make_synthetic_rectified_order_set()
        # Replace first order's flux with a 1-D array.
        ros.rectified_orders[0].flux = np.ones(_N_SPECTRAL)
        with pytest.raises(ValueError, match="2-D"):
            extract_rectified_orders(ros)

    def test_raises_on_invalid_method_empty_string(self):
        """ValueError if method is an empty string."""
        ros = _make_synthetic_rectified_order_set()
        with pytest.raises(ValueError, match="method"):
            extract_rectified_orders(ros, method="")


# ===========================================================================
# 6. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestH1ExtractedSpectraSmokeTest:
    """End-to-end smoke test using the real H1 iSHELL calibration dataset.

    The chain is:
      1. Flat-order tracing (Stage 1)
      2. Arc-line tracing (Stage 2)
      3. Provisional wavelength mapping (Stage 3)
      4. Coefficient-surface refinement (Stage 5)
      5. Rectification indices (Stage 6)
      6. Rectified orders (Stage 7)
      7. Extracted spectra (Stage 10)
    """

    @pytest.fixture(scope="class")
    def h1_extracted_spectra(self):
        """ExtractedSpectrumSet built from the real H1 calibration chain."""
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

        # Stage 10: extracted spectra
        extracted = extract_rectified_orders(rectified, method="sum")

        return extracted, rectified

    def test_returns_extracted_spectrum_set(self, h1_extracted_spectra):
        """extract_rectified_orders returns an ExtractedSpectrumSet."""
        extracted, _ = h1_extracted_spectra
        assert isinstance(extracted, ExtractedSpectrumSet)

    def test_mode_is_h1(self, h1_extracted_spectra):
        """mode is 'H1'."""
        extracted, _ = h1_extracted_spectra
        assert extracted.mode == "H1"

    def test_n_orders_matches_rectified(self, h1_extracted_spectra):
        """n_orders matches the rectified order set."""
        extracted, rectified = h1_extracted_spectra
        assert extracted.n_orders == rectified.n_orders

    def test_n_orders_positive(self, h1_extracted_spectra):
        """n_orders > 0."""
        extracted, _ = h1_extracted_spectra
        assert extracted.n_orders > 0

    def test_wavelength_axes_monotonic(self, h1_extracted_spectra):
        """wavelength_um is monotonic for every extracted spectrum."""
        extracted, _ = h1_extracted_spectra
        for sp in extracted.spectra:
            diffs = np.diff(sp.wavelength_um)
            assert np.all(diffs > 0) or np.all(diffs < 0), (
                f"Order {sp.order}: wavelength axis is not monotonic"
            )

    def test_output_shapes_correct(self, h1_extracted_spectra):
        """flux shape equals wavelength_um shape for every spectrum."""
        extracted, _ = h1_extracted_spectra
        for sp in extracted.spectra:
            assert sp.flux.shape == sp.wavelength_um.shape, (
                f"Order {sp.order}: flux shape {sp.flux.shape} != "
                f"wavelength_um shape {sp.wavelength_um.shape}"
            )

    def test_flux_not_all_nan(self, h1_extracted_spectra):
        """At least some flux values are finite for each order."""
        extracted, _ = h1_extracted_spectra
        for sp in extracted.spectra:
            n_finite = np.sum(np.isfinite(sp.flux))
            assert n_finite > 0, (
                f"Order {sp.order}: all extracted flux values are NaN"
            )

    def test_n_spatial_used_positive(self, h1_extracted_spectra):
        """n_spatial_used > 0 for every extracted spectrum."""
        extracted, _ = h1_extracted_spectra
        for sp in extracted.spectra:
            assert sp.n_spatial_used > 0, (
                f"Order {sp.order}: n_spatial_used == 0"
            )

    def test_orders_list_matches_rectified(self, h1_extracted_spectra):
        """orders list matches the rectified order set."""
        extracted, rectified = h1_extracted_spectra
        assert extracted.orders == rectified.orders

    def test_get_order_works_for_all_orders(self, h1_extracted_spectra):
        """get_order succeeds for every order in the extracted spectrum set."""
        extracted, _ = h1_extracted_spectra
        for order in extracted.orders:
            sp = extracted.get_order(order)
            assert sp.order == order

    def test_mean_extraction_consistent_shape(self, h1_extracted_spectra):
        """mean extraction produces the same shape as sum extraction."""
        _, rectified = h1_extracted_spectra
        mean_result = extract_rectified_orders(rectified, method="mean")
        sum_result = extract_rectified_orders(rectified, method="sum")
        for sp_mean, sp_sum in zip(mean_result.spectra, sum_result.spectra):
            assert sp_mean.flux.shape == sp_sum.flux.shape
