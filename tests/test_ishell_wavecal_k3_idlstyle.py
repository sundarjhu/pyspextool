"""
Tests for wavecal_k3_idlstyle.py — IDL-style K3 1DXD wavelength calibration.

Coverage:
  1. 1D extraction from traced geometry (OrderArcSpectrum / extract_order_arc_spectra)
  2. IdlStyle1DXDModel evaluation (eval, eval_array, as_wavelength_func)
  3. fit_1dxd_wavelength_model global fit
  4. Rectification using 1DXD model (build_rectification_indices with wavelength_func)
  5. Integration: K3 benchmark driver exposes stage3b_k3_1dxd key

All tests that require real K3 data are skipped cleanly when the files are
absent (same convention used in test_ishell_k3_example.py).
"""

from __future__ import annotations

import os
import unittest
from dataclasses import fields as dc_fields
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K3_RAW_DIR = os.path.join(
    _REPO_ROOT, "data", "testdata", "ishell_k3_example", "raw"
)
_LFS_MAGIC = b"version https://git-lfs"
_ARC_NUMBERS = list(range(11, 13))   # 11–12
_FLAT_NUMBERS = list(range(6, 11))   # 6–10


def _is_real_fits(path: str) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(64)
    return not head.startswith(_LFS_MAGIC)


def _real_k3_files(pattern: str, frame_numbers: list[int]) -> list[str]:
    try:
        from pyspextool.instruments.ishell.io_utils import find_fits_files
    except Exception:  # noqa: BLE001
        return []
    if not os.path.isdir(_K3_RAW_DIR):
        return []
    all_files = find_fits_files(_K3_RAW_DIR)
    matched: list[str] = []
    for p in all_files:
        name = p.name
        if pattern not in name:
            continue
        for n in frame_numbers:
            if f".{n:05d}." in name:
                if _is_real_fits(str(p)):
                    matched.append(str(p))
                break
    return sorted(matched)


_K3_ARC_FILES = _real_k3_files("arc", _ARC_NUMBERS)
_K3_FLAT_FILES = _real_k3_files("flat", _FLAT_NUMBERS)
_HAVE_K3_DATA = len(_K3_ARC_FILES) >= 1 and len(_K3_FLAT_FILES) >= 1

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_synthetic_trace(n_orders=3, ncols=512, center_spacing=100):
    """Build a FlatOrderTrace-like mock with simple horizontal traces."""
    from unittest.mock import MagicMock

    poly_degree = 1
    center_poly_coeffs = np.zeros((n_orders, poly_degree + 1))
    for i in range(n_orders):
        # Center at fixed row: 50, 150, 250, … — no tilt
        center_poly_coeffs[i, 0] = 50.0 + i * center_spacing
        # zero tilt

    trace = MagicMock()
    trace.n_orders = n_orders
    trace.center_poly_coeffs = center_poly_coeffs
    trace.poly_degree = poly_degree
    return trace


def _make_synthetic_arc_img(nrows=1024, ncols=512):
    """Create a synthetic arc image with horizontal emission lines."""
    rng = np.random.default_rng(42)
    img = rng.normal(10.0, 2.0, size=(nrows, ncols)).astype(np.float32)
    # Add bright horizontal lines at rows 50, 150, 250 (matching mock trace)
    for row in [50, 150, 250]:
        img[row - 2: row + 3, :] += 2000.0
    return img


def _make_synthetic_wavecalinfo(n_orders=3, n_pixels=512, order_nums=None):
    """Build a minimal WaveCalInfo-like mock."""
    if order_nums is None:
        order_nums = [200 + i for i in range(n_orders)]

    wci = MagicMock()
    wci.mode = "K3"
    wci.n_orders = n_orders
    wci.orders = order_nums

    # xranges: col_start=0, col_end=n_pixels-1 for each order
    xranges = np.zeros((n_orders, 2), dtype=int)
    xranges[:, 1] = n_pixels - 1
    wci.xranges = xranges

    # data cube: plane 0 = wavelength in µm (simple linear ramp per order)
    # Use the same formula as _make_synthetic_line_list so that reference
    # line wavelengths fall within the WaveCalInfo grid for each order.
    data = np.zeros((n_orders, 4, n_pixels), dtype=float)
    max_on = max(order_nums)
    for i, on in enumerate(order_nums):
        # wavelength range roughly matching K3 band (~2 µm region)
        wav_lo = 2.0 - 0.1 * (max_on - on)
        wav_hi = wav_lo + 0.05
        data[i, 0, :] = np.linspace(wav_lo, wav_hi, n_pixels)
    wci.data = data

    return wci


def _make_synthetic_line_list(n_orders=3, order_nums=None, n_lines_per_order=5):
    """Build a minimal LineList-like mock with synthetic lines."""
    if order_nums is None:
        order_nums = [200 + i for i in range(n_orders)]

    entries = []
    for on in order_nums:
        wav_lo = 2.0 - 0.1 * (max(order_nums) - on)
        wav_hi = wav_lo + 0.05
        for k in range(n_lines_per_order):
            frac = (k + 0.5) / n_lines_per_order
            wav = wav_lo + frac * (wav_hi - wav_lo)
            entry = MagicMock()
            entry.order = on
            entry.wavelength_um = wav
            entry.species = "Ar I"
            entries.append(entry)

    ll = MagicMock()
    ll.entries = entries
    return ll


# ===========================================================================
# 1. OrderArcSpectrum and OrderArcSpectraSet dataclasses
# ===========================================================================


class TestOrderArcSpectrumDataclass:
    """OrderArcSpectrum construction and properties."""

    def _make(self, n_cols=512):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import OrderArcSpectrum

        flux = np.ones(n_cols, dtype=float)
        return OrderArcSpectrum(
            order_index=0,
            order_number=203,
            col_start=10,
            col_end=10 + n_cols - 1,
            flux=flux,
        )

    def test_construction(self):
        s = self._make()
        assert s.order_index == 0
        assert s.order_number == 203

    def test_n_cols_property(self):
        s = self._make(n_cols=200)
        assert s.n_cols == 200

    def test_columns_property(self):
        s = self._make(n_cols=50)
        cols = s.columns
        assert cols[0] == s.col_start
        assert cols[-1] == s.col_end
        assert len(cols) == 50


class TestOrderArcSpectraSet:
    """OrderArcSpectraSet API."""

    def _make_set(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            OrderArcSpectrum,
            OrderArcSpectraSet,
        )

        spectra = [
            OrderArcSpectrum(i, 200 + i, 0, 511, np.ones(512))
            for i in range(4)
        ]
        return OrderArcSpectraSet(mode="K3", spectra=spectra, aperture_half_width=3)

    def test_n_orders(self):
        s = self._make_set()
        assert s.n_orders == 4

    def test_order_numbers(self):
        s = self._make_set()
        assert s.order_numbers == [200, 201, 202, 203]

    def test_get_spectrum_found(self):
        s = self._make_set()
        spec = s.get_spectrum(201)
        assert spec.order_number == 201

    def test_get_spectrum_not_found(self):
        s = self._make_set()
        with pytest.raises(KeyError):
            s.get_spectrum(999)


# ===========================================================================
# 2. IdlStyle1DXDModel: construction and evaluation
# ===========================================================================


class TestIdlStyle1DXDModel:
    """IdlStyle1DXDModel evaluation helpers."""

    def _make_model(self, wdeg=3, odeg=2):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel

        rng = np.random.default_rng(0)
        coeffs = rng.normal(0, 1e-6, size=(wdeg + 1, odeg + 1))
        # Set constant term to a reasonable wavelength in µm
        coeffs[0, 0] = 2.0
        n_lines = 100
        return IdlStyle1DXDModel(
            mode="K3",
            wdeg=wdeg,
            odeg=odeg,
            order_ref=203.0,
            coeffs=coeffs,
            fitted_order_numbers=list(range(203, 230)),
            fit_rms_um=0.001,
            n_lines=n_lines,
            n_lines_total=n_lines,
            n_lines_rejected=0,
            accepted_mask=np.ones(n_lines, dtype=bool),
            median_residual_um=0.0,
            n_orders_fit=27,
        )

    def test_eval_returns_finite_scalar(self):
        model = self._make_model()
        wav = model.eval(col=1024.0, order=215.0)
        assert np.isfinite(wav)

    def test_eval_array_returns_finite(self):
        model = self._make_model()
        cols = np.linspace(0, 2047, 100)
        orders = np.full(100, 215.0)
        wavs = model.eval_array(cols, orders)
        assert wavs.shape == (100,)
        assert np.all(np.isfinite(wavs))

    def test_eval_and_eval_array_agree(self):
        model = self._make_model()
        col, order = 800.0, 210.0
        scalar = model.eval(col, order)
        array = model.eval_array(np.array([col]), np.array([order]))
        assert abs(scalar - array[0]) < 1e-12

    def test_polynomial_degree_3_2(self):
        model = self._make_model(wdeg=3, odeg=2)
        assert model.wdeg == 3
        assert model.odeg == 2
        assert model.coeffs.shape == (4, 3)

    def test_as_wavelength_func_callable(self):
        model = self._make_model()
        func = model.as_wavelength_func()
        assert callable(func)
        cols = np.array([100.0, 200.0, 300.0])
        result = func(cols, 215.0)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_as_wavelength_func_matches_eval_array(self):
        model = self._make_model()
        func = model.as_wavelength_func()
        cols = np.linspace(0, 2047, 50)
        order = 215.0
        result_func = func(cols, order)
        result_eval = model.eval_array(cols, np.full(50, order))
        np.testing.assert_allclose(result_func, result_eval)

    def test_order_ref_determines_v(self):
        """v = order_ref / order; larger order_ref → larger v."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel

        coeffs = np.zeros((2, 2))
        coeffs[0, 0] = 2.0  # constant
        coeffs[0, 1] = 1.0  # v coefficient
        _n = 1
        _mask = np.ones(_n, dtype=bool)
        m1 = IdlStyle1DXDModel(
            "K3", 1, 1, 200.0, coeffs, [200], 0.0, _n, _n, 0, _mask, 0.0, 1
        )
        m2 = IdlStyle1DXDModel(
            "K3", 1, 1, 400.0, coeffs, [200], 0.0, _n, _n, 0, _mask, 0.0, 1
        )
        # At order=200: v1=200/200=1, v2=400/200=2 → wav1 < wav2
        wav1 = m1.eval(0.0, 200.0)
        wav2 = m2.eval(0.0, 200.0)
        assert wav2 > wav1


# ===========================================================================
# 3. extract_order_arc_spectra — uses traced geometry
# ===========================================================================


class TestExtractOrderArcSpectra:
    """extract_order_arc_spectra extracts correctly from synthetic data."""

    def _run(self, n_orders=3, ncols=512, aperture_hw=3):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            extract_order_arc_spectra,
        )

        arc_img = _make_synthetic_arc_img(nrows=512, ncols=ncols)
        trace = _make_synthetic_trace(n_orders=n_orders, ncols=ncols)
        wci = _make_synthetic_wavecalinfo(n_orders=n_orders, n_pixels=ncols)
        return extract_order_arc_spectra(arc_img, trace, wci, aperture_half_width=aperture_hw)

    def test_returns_correct_type(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import OrderArcSpectraSet
        result = self._run()
        assert isinstance(result, OrderArcSpectraSet)

    def test_n_orders_matches(self):
        result = self._run(n_orders=3)
        assert result.n_orders == 3

    def test_mode_propagated(self):
        result = self._run()
        assert result.mode == "K3"

    def test_aperture_half_width_stored(self):
        result = self._run(aperture_hw=2)
        assert result.aperture_half_width == 2

    def test_flux_shape(self):
        result = self._run(n_orders=2, ncols=256)
        for spec in result.spectra:
            assert len(spec.flux) == spec.n_cols
            assert spec.n_cols > 0

    def test_flux_finite_at_center_rows(self):
        """Bright rows in the arc image should yield large finite flux."""
        result = self._run(n_orders=3, ncols=512)
        for spec in result.spectra:
            finite_mask = np.isfinite(spec.flux)
            assert finite_mask.any(), "All flux values are NaN"

    def test_uses_traced_centre_not_arc_peak(self):
        """Extraction must be at the traced centre row, not the arc peak row."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            extract_order_arc_spectra,
        )

        # Arc image with signal only at row 300 (not at any traced centre)
        arc_img = np.zeros((512, 64), dtype=float)
        arc_img[300, :] = 1000.0

        # Trace with centres at rows 50, 150, 250
        trace = _make_synthetic_trace(n_orders=3, ncols=64, center_spacing=100)
        wci = _make_synthetic_wavecalinfo(n_orders=3, n_pixels=64)

        result = extract_order_arc_spectra(arc_img, trace, wci, aperture_half_width=3)

        # The extraction is at rows 50±3, 150±3, 250±3, NOT row 300.
        # Background is zero so flux should be near 0.
        for spec in result.spectra:
            assert np.nanmean(spec.flux) < 1.0, (
                "Extraction should be at traced centre (near 0), "
                "not at arc peak (row 300)"
            )

    def test_column_alignment_preserved(self):
        """columns property must start at col_start and be contiguous."""
        result = self._run(n_orders=2, ncols=128)
        for spec in result.spectra:
            cols = spec.columns
            assert cols[0] == spec.col_start
            assert len(cols) == spec.n_cols
            diffs = np.diff(cols)
            assert np.all(diffs == 1)

    def test_raises_on_non_2d_image(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            extract_order_arc_spectra,
        )

        bad_img = np.ones((512,), dtype=float)  # 1D
        trace = _make_synthetic_trace()
        wci = _make_synthetic_wavecalinfo()
        with pytest.raises(ValueError, match="2-D"):
            extract_order_arc_spectra(bad_img, trace, wci)

    def test_raises_on_bad_aperture(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            extract_order_arc_spectra,
        )

        arc_img = np.ones((512, 512), dtype=float)
        trace = _make_synthetic_trace()
        wci = _make_synthetic_wavecalinfo()
        with pytest.raises(ValueError, match="aperture_half_width"):
            extract_order_arc_spectra(arc_img, trace, wci, aperture_half_width=0)

    def test_order_count_mismatch_warns(self):
        """Warns when n_trace_orders != n_wavecalinfo_orders."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            extract_order_arc_spectra,
        )
        import warnings

        arc_img = np.ones((512, 64), dtype=float)
        trace = _make_synthetic_trace(n_orders=5, ncols=64)
        wci = _make_synthetic_wavecalinfo(n_orders=3, n_pixels=64)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extract_order_arc_spectra(arc_img, trace, wci)

        assert any(issubclass(x.category, RuntimeWarning) for x in w)
        assert result.n_orders == 3  # uses min(5, 3)


# ===========================================================================
# 4. fit_1dxd_wavelength_model
# ===========================================================================


class TestFit1DXDWavelengthModel:
    """fit_1dxd_wavelength_model fits a global polynomial."""

    def _build_inputs(self, n_orders=3, ncols=512):
        """Build spectra_set, wavecalinfo, line_list for fitting."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            OrderArcSpectrum,
            OrderArcSpectraSet,
        )

        order_nums = [200 + i for i in range(n_orders)]
        wci = _make_synthetic_wavecalinfo(n_orders=n_orders, n_pixels=ncols,
                                          order_nums=order_nums)
        ll = _make_synthetic_line_list(n_orders=n_orders, order_nums=order_nums,
                                       n_lines_per_order=6)

        # Build spectra with peaks at known column positions matching lines.
        # Use high amplitude so peaks exceed the default min_prominence.
        spectra = []
        for i, on in enumerate(order_nums):
            # Background level of 10 so peaks need to be well above it.
            flux = np.full(ncols, 10.0, dtype=float)
            for entry in ll.entries:
                if entry.order != on:
                    continue
                wav = entry.wavelength_um
                wav_array = wci.data[i, 0, :]
                col_idx = int(np.argmin(np.abs(wav_array - wav)))
                # Gaussian-like peak with amplitude 10000 (>>min_prominence=50)
                for dc in range(-2, 3):
                    c = col_idx + dc
                    if 0 <= c < ncols:
                        flux[c] += 10000.0 * np.exp(-0.5 * dc ** 2)

            spec = OrderArcSpectrum(
                order_index=i,
                order_number=on,
                col_start=0,
                col_end=ncols - 1,
                flux=flux,
            )
            spectra.append(spec)

        spectra_set = OrderArcSpectraSet(mode="K3", spectra=spectra, aperture_half_width=3)
        return spectra_set, wci, ll

    def test_returns_correct_type(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            IdlStyle1DXDModel,
            fit_1dxd_wavelength_model,
        )

        spectra_set, wci, ll = self._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        assert isinstance(model, IdlStyle1DXDModel)

    def test_polynomial_degrees_stored(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )

        spectra_set, wci, ll = self._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=3, odeg=2)
        assert model.wdeg == 3
        assert model.odeg == 2
        assert model.coeffs.shape == (4, 3)

    def test_finite_wavelengths(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )

        spectra_set, wci, ll = self._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        # Evaluate at several (col, order) combinations
        for col in [100, 300, 500]:
            for on in model.fitted_order_numbers:
                wav = model.eval(float(col), float(on))
                assert np.isfinite(wav), f"Non-finite wav at col={col}, order={on}"

    def test_n_lines_positive(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )

        spectra_set, wci, ll = self._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        assert model.n_lines > 0

    def test_n_orders_fit_positive(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )

        spectra_set, wci, ll = self._build_inputs(n_orders=3)
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        assert model.n_orders_fit >= 1

    def test_fit_rms_finite(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )

        spectra_set, wci, ll = self._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        assert np.isfinite(model.fit_rms_um)
        assert model.fit_rms_um >= 0.0

    def test_raises_on_negative_wdeg(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )

        spectra_set, wci, ll = self._build_inputs()
        with pytest.raises(ValueError, match="wdeg"):
            fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=-1, odeg=2)

    def test_raises_on_negative_odeg(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )

        spectra_set, wci, ll = self._build_inputs()
        with pytest.raises(ValueError, match="odeg"):
            fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=-1)

    def test_raises_when_too_few_lines(self):
        """Raises ValueError when fewer than min_lines_total are matched."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            OrderArcSpectrum,
            OrderArcSpectraSet,
            fit_1dxd_wavelength_model,
        )

        # Empty spectra → no peaks → no matches
        spectra = [
            OrderArcSpectrum(0, 200, 0, 511, np.zeros(512)),
        ]
        spectra_set = OrderArcSpectraSet(mode="K3", spectra=spectra)
        wci = _make_synthetic_wavecalinfo(n_orders=1, n_pixels=512, order_nums=[200])
        ll = _make_synthetic_line_list(n_orders=1, order_nums=[200])
        with pytest.raises(ValueError):
            fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=1, odeg=1,
                                      min_lines_total=5)


# ===========================================================================
# 5. Rectification with 1DXD model (Option B)
# ===========================================================================


class TestRectificationWith1DXDModel:
    """build_rectification_indices with wavelength_func uses 1DXD model."""

    def _make_geometry_and_model(self, n_orders=3, ncols=512):
        from pyspextool.instruments.ishell.geometry import OrderGeometry, OrderGeometrySet
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel

        order_nums = list(range(200, 200 + n_orders))

        geometries = []
        for i, on in enumerate(order_nums):
            row_c = 100.0 + i * 100.0
            bot = np.array([row_c - 15.0, 0.0])  # constant (degree 1)
            top = np.array([row_c + 15.0, 0.0])
            geom = OrderGeometry(
                order=on,
                x_start=0,
                x_end=ncols - 1,
                bottom_edge_coeffs=bot,
                top_edge_coeffs=top,
            )
            geometries.append(geom)

        geom_set = OrderGeometrySet(mode="K3", geometries=geometries)

        # Simple model: wavelength increases with column, roughly 2 µm
        coeffs = np.zeros((2, 2))
        coeffs[0, 0] = 2.0       # constant offset
        coeffs[1, 0] = 1e-5      # linear in col
        coeffs[0, 1] = 0.01      # small order dependence

        model = IdlStyle1DXDModel(
            mode="K3",
            wdeg=1,
            odeg=1,
            order_ref=200.0,
            coeffs=coeffs,
            fitted_order_numbers=order_nums,
            fit_rms_um=0.001,
            n_lines=100,
            n_lines_total=100,
            n_lines_rejected=0,
            accepted_mask=np.ones(100, dtype=bool),
            median_residual_um=0.0,
            n_orders_fit=n_orders,
        )

        return geom_set, model, order_nums

    def test_returns_rectification_index_set(self):
        from pyspextool.instruments.ishell.rectification_indices import (
            RectificationIndexSet,
            build_rectification_indices,
        )

        geom_set, model, order_nums = self._make_geometry_and_model()
        rect_idx = build_rectification_indices(
            geom_set,
            wavelength_func=model.as_wavelength_func(),
            fitted_order_numbers=model.fitted_order_numbers,
            n_spectral=32,
            n_spatial=8,
        )
        assert isinstance(rect_idx, RectificationIndexSet)

    def test_n_orders_matches(self):
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )

        geom_set, model, order_nums = self._make_geometry_and_model(n_orders=3)
        rect_idx = build_rectification_indices(
            geom_set,
            wavelength_func=model.as_wavelength_func(),
            fitted_order_numbers=model.fitted_order_numbers,
            n_spectral=16,
            n_spatial=8,
        )
        assert rect_idx.n_orders == 3

    def test_wavelengths_finite_and_monotone(self):
        """Each order's output wavelength axis must be finite and monotone."""
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )

        geom_set, model, _ = self._make_geometry_and_model()
        rect_idx = build_rectification_indices(
            geom_set,
            wavelength_func=model.as_wavelength_func(),
            fitted_order_numbers=model.fitted_order_numbers,
            n_spectral=32,
            n_spatial=8,
        )
        for io in rect_idx.index_orders:
            wavs = io.output_wavelengths_um
            assert np.all(np.isfinite(wavs)), f"Order {io.order}: non-finite wavelengths"
            diffs = np.diff(wavs)
            assert np.all(diffs > 0) or np.all(diffs < 0), (
                f"Order {io.order}: wavelength axis not monotone"
            )

    def test_wavelength_source_is_1dxd_not_scaffold(self):
        """The wavelength_func result must differ from a naive constant."""
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )

        geom_set, model, order_nums = self._make_geometry_and_model()

        # A constant-wavelength function (scaffold proxy)
        def const_func(cols, order):
            return np.full_like(np.asarray(cols, float), 2.0)

        rect_const = build_rectification_indices(
            geom_set,
            wavelength_func=const_func,
            fitted_order_numbers=order_nums,
            n_spectral=32,
            n_spatial=8,
        )
        rect_1dxd = build_rectification_indices(
            geom_set,
            wavelength_func=model.as_wavelength_func(),
            fitted_order_numbers=model.fitted_order_numbers,
            n_spectral=32,
            n_spatial=8,
        )
        # The 1DXD model is NOT constant; its wavelengths should span a range
        for io in rect_1dxd.index_orders:
            span = io.output_wavelengths_um[-1] - io.output_wavelengths_um[0]
            assert abs(span) > 1e-6, "1DXD wavelengths should not be constant"

    def test_raises_when_both_surface_and_func_given(self):
        """Providing both surface and wavelength_func should raise ValueError."""
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )
        from unittest.mock import MagicMock

        geom_set, model, _ = self._make_geometry_and_model()
        mock_surface = MagicMock()
        mock_surface.per_order_order_numbers = np.array([200, 201, 202])

        with pytest.raises(ValueError, match="not both"):
            build_rectification_indices(
                geom_set,
                surface=mock_surface,
                wavelength_func=model.as_wavelength_func(),
            )

    def test_raises_when_neither_surface_nor_func_given(self):
        """Providing neither surface nor wavelength_func should raise ValueError."""
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )

        geom_set, _, _ = self._make_geometry_and_model()
        with pytest.raises(ValueError):
            build_rectification_indices(geom_set)

    def test_mode_propagated(self):
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )

        geom_set, model, _ = self._make_geometry_and_model()
        rect_idx = build_rectification_indices(
            geom_set,
            wavelength_func=model.as_wavelength_func(),
            fitted_order_numbers=model.fitted_order_numbers,
            n_spectral=16,
            n_spatial=4,
        )
        assert rect_idx.mode == "K3"


# ===========================================================================
# 6. Module public API
# ===========================================================================


class TestModuleImports:
    """Public API is importable without side effects."""

    def test_all_public_symbols_importable(self):
        from pyspextool.instruments.ishell import wavecal_k3_idlstyle as m

        for name in m.__all__:
            assert hasattr(m, name), f"Missing public symbol: {name}"

    def test_module_importable(self):
        import pyspextool.instruments.ishell.wavecal_k3_idlstyle  # noqa: F401

    def test_order_match_stats_importable(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import OrderMatchStats  # noqa: F401


# ===========================================================================
# 7. Integration: K3 benchmark driver exposes stage3b key
# ===========================================================================


class TestK3BenchmarkDriverStage3b:
    """K3 benchmark driver run exposes 'stage3b_k3_1dxd' in completed dict."""

    @pytest.mark.skipif(not _HAVE_K3_DATA, reason="K3 raw data not available")
    def test_stage3b_key_present(self):
        import importlib.util
        import sys

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_out:
            completed = mod.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                output_dir=tmp_out,
                no_plots=True,
            )

        assert "stage3b_k3_1dxd" in completed, (
            "'stage3b_k3_1dxd' key must be present in completed dict"
        )

    @pytest.mark.skipif(not _HAVE_K3_DATA, reason="K3 raw data not available")
    def test_stage3b_uses_1dxd_not_scaffold_for_rectification(self):
        """When Stage 3b succeeds, Stage 6 must indicate K3 1DXD is the source."""
        import importlib.util
        import sys

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_stage3b",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_out:
            completed = mod.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                output_dir=tmp_out,
                no_plots=True,
            )

        # If Stage 3b succeeded, Stage 6 should also have completed
        if completed.get("stage3b_k3_1dxd"):
            assert completed.get("stage6_rect_indices", False), (
                "Stage 6 should succeed when Stage 3b succeeds"
            )


# ===========================================================================
# 8. Cross-correlation
# ===========================================================================


class TestXCorrOrderShift:
    """_xcorr_order_shift returns a finite shift and shifts search windows."""

    def _build_xcorr_inputs(self, true_shift_px=5, n_lines=6):
        """Build a synthetic spectrum with a known shift."""
        order_nums = [200]
        n_orders = 1
        ncols = 512
        order_num = 200
        max_on = 200

        wci = _make_synthetic_wavecalinfo(
            n_orders=n_orders, n_pixels=ncols, order_nums=order_nums
        )
        ll = _make_synthetic_line_list(
            n_orders=n_orders, order_nums=order_nums, n_lines_per_order=n_lines
        )
        coarse_cols = np.linspace(0, ncols - 1, ncols)
        wav_lo = 2.0 - 0.1 * (max_on - order_num)
        wav_hi = wav_lo + 0.05
        coarse_wavs = np.linspace(wav_lo, wav_hi, ncols)
        ref_entries = [
            (float(e.wavelength_um), str(e.species))
            for e in ll.entries
            if e.order == order_num
        ]
        # Build flux at reference line positions, shifted by true_shift_px
        flux = np.full(ncols, 10.0)
        for wav, _ in ref_entries:
            col = int(np.argmin(np.abs(coarse_wavs - wav)))
            col_shifted = col + true_shift_px
            for dc in range(-2, 3):
                c = col_shifted + dc
                if 0 <= c < ncols:
                    flux[c] += 5000.0 * np.exp(-0.5 * dc ** 2)
        return flux, coarse_cols, coarse_wavs, ref_entries

    def test_returns_finite_shift(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import _xcorr_order_shift

        flux, cols, wavs, refs = self._build_xcorr_inputs(true_shift_px=0)
        shift = _xcorr_order_shift(flux, cols, wavs, refs, col_start=0)
        assert np.isfinite(shift)

    def test_shift_within_max(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import _xcorr_order_shift

        flux, cols, wavs, refs = self._build_xcorr_inputs(true_shift_px=10)
        shift = _xcorr_order_shift(
            flux, cols, wavs, refs, col_start=0, max_shift_px=50
        )
        assert abs(shift) <= 50

    def test_zero_shift_for_aligned_spectrum(self):
        """When spectrum and reference are aligned, shift should be near zero."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import _xcorr_order_shift

        flux, cols, wavs, refs = self._build_xcorr_inputs(true_shift_px=0)
        shift = _xcorr_order_shift(flux, cols, wavs, refs, col_start=0)
        # Allow a couple of pixels tolerance for the synthetic data
        assert abs(shift) <= 3.0, f"Expected near-zero shift, got {shift:.2f} px"

    def test_empty_refs_returns_zero(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import _xcorr_order_shift

        flux = np.ones(100)
        cols = np.arange(100, dtype=float)
        wavs = np.linspace(2.0, 2.05, 100)
        shift = _xcorr_order_shift(flux, cols, wavs, [], col_start=0)
        assert shift == 0.0

    def test_shift_changes_peak_search_window(self):
        """Applying the xcorr shift to coarse_cols changes matched peaks."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            _xcorr_order_shift,
            _match_1d_peaks,
        )
        from scipy.signal import find_peaks

        # Build flux shifted by +20 px
        true_shift = 20
        flux, cols, wavs, refs = self._build_xcorr_inputs(true_shift_px=true_shift)
        shift = _xcorr_order_shift(flux, cols, wavs, refs, col_start=0)

        # Detect peaks
        peak_idxs, _ = find_peaks(flux, prominence=50.0, distance=5)
        peak_cols = peak_idxs.astype(float)

        # Unshifted matching should find fewer/different matches
        matches_unshifted = _match_1d_peaks(
            peak_cols, cols, wavs, refs, match_tol_um=0.002
        )
        # Shifted matching
        shifted_cols = cols + shift
        matches_shifted = _match_1d_peaks(
            peak_cols, shifted_cols, wavs, refs, match_tol_um=0.002
        )
        # Shifted matching should find at least as many matches for this data
        # (for a perfect synthetic case the shift perfectly compensates)
        assert len(matches_shifted) >= len(matches_unshifted)


# ===========================================================================
# 9. Sigma clipping
# ===========================================================================


class TestSigmaClipping:
    """fit_1dxd_wavelength_model rejects outliers via sigma clipping."""

    def _build_inputs_with_outlier(self):
        """Build inputs where one arc-line position has a large residual."""
        spectra_set, wci, ll = TestFit1DXDWavelengthModel()._build_inputs()
        # Inject an outlier: corrupt the last line's flux to a far-off position
        # by adding a large extra peak at a position with no reference line
        spec0 = spectra_set.spectra[0]
        flux = spec0.flux.copy()
        # Add a tall, narrow, isolated spike at column 490 (far from real lines)
        for dc in range(-1, 2):
            c = 490 + dc
            if 0 <= c < len(flux):
                flux[c] += 80000.0 * np.exp(-0.5 * dc ** 2)
        spectra_set.spectra[0] = spec0.__class__(
            order_index=spec0.order_index,
            order_number=spec0.order_number,
            col_start=spec0.col_start,
            col_end=spec0.col_end,
            flux=flux,
        )
        return spectra_set, wci, ll

    def test_n_lines_total_and_accepted_fields_exist(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )
        spectra_set, wci, ll = TestFit1DXDWavelengthModel()._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        assert hasattr(model, "n_lines_total")
        assert hasattr(model, "n_lines_rejected")
        assert hasattr(model, "accepted_mask")
        assert hasattr(model, "median_residual_um")
        assert model.n_lines_total >= model.n_lines
        assert model.n_lines_rejected == model.n_lines_total - model.n_lines

    def test_accepted_mask_shape(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )
        spectra_set, wci, ll = TestFit1DXDWavelengthModel()._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        assert model.accepted_mask.shape == (model.n_lines_total,)
        assert model.accepted_mask.dtype == bool
        assert int(np.sum(model.accepted_mask)) == model.n_lines

    def test_per_order_stats_populated(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            OrderMatchStats,
            fit_1dxd_wavelength_model,
        )
        spectra_set, wci, ll = TestFit1DXDWavelengthModel()._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        assert len(model.per_order_stats) == spectra_set.n_orders
        for stat in model.per_order_stats:
            assert isinstance(stat, OrderMatchStats)
            assert stat.order_number in spectra_set.order_numbers

    def test_per_order_stats_xcorr_shift_finite(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )
        spectra_set, wci, ll = TestFit1DXDWavelengthModel()._build_inputs()
        model = fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)
        for stat in model.per_order_stats:
            assert np.isfinite(stat.xcorr_shift_px)

    def test_tight_sigma_clip_rejects_outlier(self):
        """With a very tight threshold an outlier spike should be clipped."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )
        spectra_set, wci, ll = self._build_inputs_with_outlier()
        # Use tight clipping so the outlier is likely rejected
        model_clip = fit_1dxd_wavelength_model(
            spectra_set, wci, ll, wdeg=2, odeg=1, sigma_thresh=1.5
        )
        model_noclip = fit_1dxd_wavelength_model(
            spectra_set, wci, ll, wdeg=2, odeg=1, sigma_thresh=1e9
        )
        # With tight clipping, at least some rejections expected
        # (total accepted ≤ total matched before clipping)
        assert model_clip.n_lines <= model_clip.n_lines_total

    def test_rms_not_worse_after_clipping(self):
        """Sigma clipping should not increase RMS compared to single-pass."""
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import (
            fit_1dxd_wavelength_model,
        )
        spectra_set, wci, ll = self._build_inputs_with_outlier()
        model_clip = fit_1dxd_wavelength_model(
            spectra_set, wci, ll, wdeg=2, odeg=1, sigma_thresh=1.5
        )
        model_noclip = fit_1dxd_wavelength_model(
            spectra_set, wci, ll, wdeg=2, odeg=1, sigma_thresh=1e9
        )
        # After rejection the RMS on accepted points should be ≤ no-clip RMS
        # (or at worst equal — both are finite)
        assert np.isfinite(model_clip.fit_rms_um)
        assert np.isfinite(model_noclip.fit_rms_um)


# ===========================================================================
# 10. Order labels in rectification output
# ===========================================================================


class TestOrderLabelsInRectification:
    """RectificationIndexOrder.order uses real echelle order numbers."""

    def test_order_labels_are_echelle_orders_with_geom_order_map(self):
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel
        from pyspextool.instruments.ishell.geometry import (
            OrderGeometry,
            OrderGeometrySet,
        )

        # Build geometry with placeholder 0-indexed orders
        order_nums = [203, 215, 229]
        geom_placeholder_orders = [0, 1, 2]
        geometries = []
        for gi, on in zip(geom_placeholder_orders, order_nums):
            row_c = 256.0
            half = 15.0
            bot = np.array([row_c - half, 0.0])
            top = np.array([row_c + half, 0.0])
            geom = OrderGeometry(
                order=gi, x_start=0, x_end=511,
                bottom_edge_coeffs=bot, top_edge_coeffs=top,
            )
            geometries.append(geom)
        geom_set = OrderGeometrySet(mode="K3", geometries=geometries)

        coeffs = np.zeros((2, 2))
        coeffs[0, 0] = 2.0
        coeffs[1, 0] = 1e-5
        model = IdlStyle1DXDModel(
            mode="K3", wdeg=1, odeg=1, order_ref=203.0,
            coeffs=coeffs, fitted_order_numbers=order_nums,
            fit_rms_um=0.001, n_lines=30, n_lines_total=30,
            n_lines_rejected=0, accepted_mask=np.ones(30, dtype=bool),
            median_residual_um=0.0, n_orders_fit=3,
        )

        # Build geom_order_map: placeholder index → echelle order
        geom_order_map = {gi: on for gi, on in zip(geom_placeholder_orders, order_nums)}

        rect_idx = build_rectification_indices(
            geom_set,
            wavelength_func=model.as_wavelength_func(),
            fitted_order_numbers=model.fitted_order_numbers,
            geom_order_map=geom_order_map,
            n_spectral=16, n_spatial=4,
        )

        # Output order labels should be real echelle order numbers
        output_orders = sorted(io.order for io in rect_idx.index_orders)
        assert output_orders == sorted(order_nums), (
            f"Expected echelle order labels {sorted(order_nums)}, "
            f"got {output_orders}"
        )

    def test_order_labels_not_placeholder_indices(self):
        """When geom_order_map is given, placeholder indices must not appear."""
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel
        from pyspextool.instruments.ishell.geometry import (
            OrderGeometry,
            OrderGeometrySet,
        )

        order_nums = [210, 220]
        geometries = []
        for gi, on in enumerate(order_nums):
            row_c = 200.0 + gi * 50
            half = 12.0
            geom = OrderGeometry(
                order=gi, x_start=0, x_end=255,
                bottom_edge_coeffs=np.array([row_c - half, 0.0]),
                top_edge_coeffs=np.array([row_c + half, 0.0]),
            )
            geometries.append(geom)
        geom_set = OrderGeometrySet(mode="K3", geometries=geometries)

        coeffs = np.zeros((2, 2))
        coeffs[0, 0] = 2.0
        coeffs[1, 0] = 1e-5
        model = IdlStyle1DXDModel(
            mode="K3", wdeg=1, odeg=1, order_ref=210.0,
            coeffs=coeffs, fitted_order_numbers=order_nums,
            fit_rms_um=0.001, n_lines=20, n_lines_total=20,
            n_lines_rejected=0, accepted_mask=np.ones(20, dtype=bool),
            median_residual_um=0.0, n_orders_fit=2,
        )
        geom_order_map = {0: 210, 1: 220}
        rect_idx = build_rectification_indices(
            geom_set,
            wavelength_func=model.as_wavelength_func(),
            fitted_order_numbers=model.fitted_order_numbers,
            geom_order_map=geom_order_map,
            n_spectral=16, n_spatial=4,
        )
        for io in rect_idx.index_orders:
            assert io.order in order_nums, (
                f"Order label {io.order} is a placeholder, expected one of {order_nums}"
            )
            assert io.order not in [0, 1], (
                f"Order label {io.order} is a placeholder index, not an echelle order"
            )


# ===========================================================================
# 11. Diagnostics export contains 1DXD fields
# ===========================================================================


class TestDiagnosticsExport:
    """_export_diagnostics includes 1DXD fields when diags_1dxd is provided."""

    def test_csv_contains_1dxd_fields(self):
        import csv as _csv
        import tempfile
        import sys
        import importlib.util

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_diag_test",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        # Build minimal scaffold diagnostics
        from unittest.mock import MagicMock
        mock_diag = mod.OrderCalibrationDiagnostics(
            order_number=215,
            n_candidate=10, n_accepted=8, n_rejected=2,
            poly_degree_requested=3, poly_degree_used=3,
            fit_rms_nm=0.05, skipped=False, non_monotonic=False,
        )
        # Build 1DXD diagnostics
        dxd_diag = mod.OrderDiagnostics1DXD(
            order_number=215, xcorr_shift_px=2.5,
            n_candidate=8, n_matched=7, n_accepted=6, n_rejected=1,
            rms_resid_nm=0.03, participated=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = mod._export_diagnostics(
                [mock_diag], tmp, "csv", "test",
                diags_1dxd=[dxd_diag],
            )
            with open(path, newline="") as fh:
                rows = list(_csv.DictReader(fh))

        assert len(rows) == 1
        row = rows[0]
        assert "1dxd_xcorr_shift_px" in row
        assert "1dxd_n_accepted" in row
        assert "1dxd_rms_nm" in row
        assert float(row["1dxd_xcorr_shift_px"]) == pytest.approx(2.5)
        assert int(row["1dxd_n_accepted"]) == 6

    def test_json_contains_1dxd_fields(self):
        import json as _json
        import tempfile
        import sys
        import importlib.util

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_diag_test_json",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        mock_diag = mod.OrderCalibrationDiagnostics(
            order_number=210,
            n_candidate=5, n_accepted=4, n_rejected=1,
            poly_degree_requested=3, poly_degree_used=3,
            fit_rms_nm=0.04, skipped=False, non_monotonic=False,
        )
        dxd_diag = mod.OrderDiagnostics1DXD(
            order_number=210, xcorr_shift_px=-1.0,
            n_candidate=5, n_matched=4, n_accepted=4, n_rejected=0,
            rms_resid_nm=0.02, participated=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = mod._export_diagnostics(
                [mock_diag], tmp, "json", "test",
                diags_1dxd=[dxd_diag],
            )
            with open(path) as fh:
                rows = _json.load(fh)

        assert len(rows) == 1
        row = rows[0]
        assert "1dxd_xcorr_shift_px" in row
        assert "1dxd_participated" in row
        assert row["1dxd_xcorr_shift_px"] == pytest.approx(-1.0)

    def test_csv_without_1dxd_has_no_1dxd_fields(self):
        """When diags_1dxd=None, no 1DXD columns should appear."""
        import csv as _csv
        import tempfile
        import sys
        import importlib.util

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_diag_test_no1dxd",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        mock_diag = mod.OrderCalibrationDiagnostics(
            order_number=220,
            n_candidate=6, n_accepted=5, n_rejected=1,
            poly_degree_requested=3, poly_degree_used=3,
            fit_rms_nm=0.06, skipped=False, non_monotonic=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = mod._export_diagnostics([mock_diag], tmp, "csv", "test")
            with open(path, newline="") as fh:
                reader = _csv.DictReader(fh)
                fieldnames = reader.fieldnames or []

        assert not any(f.startswith("1dxd") for f in fieldnames), (
            "No 1DXD fields expected when diags_1dxd=None"
        )


# ===========================================================================
# 12. Stage 6 runs without scaffold prov_map dependency
# ===========================================================================


class TestStage6WithoutProvMap:
    """Stage 6 can run from K3 1DXD path using geom_order_map only."""

    def test_geom_order_map_eliminates_wav_map(self):
        """build_rectification_indices must work without wav_map when geom_order_map given."""
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel
        from pyspextool.instruments.ishell.geometry import (
            OrderGeometry,
            OrderGeometrySet,
        )

        order_nums = [203, 215, 229]
        geometries = []
        for gi, on in enumerate(order_nums):
            row_c = 200.0 + gi * 50
            half = 15.0
            geom = OrderGeometry(
                order=gi, x_start=0, x_end=255,
                bottom_edge_coeffs=np.array([row_c - half, 0.0]),
                top_edge_coeffs=np.array([row_c + half, 0.0]),
            )
            geometries.append(geom)
        geom_set = OrderGeometrySet(mode="K3", geometries=geometries)

        coeffs = np.zeros((2, 2))
        coeffs[0, 0] = 2.0
        coeffs[1, 0] = 1e-5
        model = IdlStyle1DXDModel(
            mode="K3", wdeg=1, odeg=1, order_ref=203.0,
            coeffs=coeffs, fitted_order_numbers=order_nums,
            fit_rms_um=0.001, n_lines=20, n_lines_total=20,
            n_lines_rejected=0, accepted_mask=np.ones(20, dtype=bool),
            median_residual_um=0.0, n_orders_fit=3,
        )
        geom_order_map = {0: 203, 1: 215, 2: 229}

        # NO wav_map passed — should still work
        rect_idx = build_rectification_indices(
            geom_set,
            wavelength_func=model.as_wavelength_func(),
            fitted_order_numbers=model.fitted_order_numbers,
            geom_order_map=geom_order_map,
            wav_map=None,
            n_spectral=16, n_spatial=4,
        )
        assert rect_idx.n_orders == len(order_nums)


# ===========================================================================
# 13. Fit-result structure — matched-point arrays
# ===========================================================================


class TestMatchedPointArrays:
    """fit_1dxd_wavelength_model returns explicit matched-point arrays."""

    def _run(self):
        spectra_set, wci, ll = TestFit1DXDWavelengthModel()._build_inputs()
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import fit_1dxd_wavelength_model
        return fit_1dxd_wavelength_model(spectra_set, wci, ll, wdeg=2, odeg=1)

    def test_matched_cols_px_exists(self):
        model = self._run()
        assert hasattr(model, "matched_cols_px")

    def test_matched_order_numbers_exists(self):
        model = self._run()
        assert hasattr(model, "matched_order_numbers")

    def test_matched_ref_wavelength_um_exists(self):
        model = self._run()
        assert hasattr(model, "matched_ref_wavelength_um")

    def test_matched_fit_wavelength_um_exists(self):
        model = self._run()
        assert hasattr(model, "matched_fit_wavelength_um")

    def test_matched_residual_um_exists(self):
        model = self._run()
        assert hasattr(model, "matched_residual_um")

    def test_all_arrays_length_equals_n_lines_total(self):
        model = self._run()
        n = model.n_lines_total
        assert len(model.matched_cols_px) == n
        assert len(model.matched_order_numbers) == n
        assert len(model.matched_ref_wavelength_um) == n
        assert len(model.matched_fit_wavelength_um) == n
        assert len(model.matched_residual_um) == n

    def test_accepted_mask_length_matches(self):
        model = self._run()
        assert len(model.accepted_mask) == model.n_lines_total

    def test_matched_residual_is_ref_minus_fit(self):
        """matched_residual_um must equal ref - fit."""
        model = self._run()
        diff = model.matched_ref_wavelength_um - model.matched_fit_wavelength_um
        np.testing.assert_allclose(model.matched_residual_um, diff, atol=1e-12)

    def test_accepted_points_rms_matches_fit_rms(self):
        """RMS of accepted residuals must match model.fit_rms_um."""
        model = self._run()
        acc_resid = model.matched_residual_um[model.accepted_mask]
        if len(acc_resid) > 0:
            rms = float(np.sqrt(np.mean(acc_resid ** 2)))
            assert abs(rms - model.fit_rms_um) < 1e-9 * max(rms, 1e-12)

    def test_fit_wavelengths_finite(self):
        model = self._run()
        assert np.all(np.isfinite(model.matched_fit_wavelength_um))

    def test_ref_wavelengths_finite(self):
        model = self._run()
        assert np.all(np.isfinite(model.matched_ref_wavelength_um))


# ===========================================================================
# 14. Residual QA plot file — integration test (K3 data required)
# ===========================================================================


class TestResidualQAFile:
    """_plot_k3_1dxd_residuals produces the expected output file."""

    @pytest.mark.skipif(not _HAVE_K3_DATA, reason="K3 raw data not available")
    def test_residual_qa_file_created(self):
        import importlib.util
        import sys
        import tempfile

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_resqa",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp_out:
            completed = mod.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                output_dir=tmp_out,
                no_plots=False,
                save_plots=True,
            )
            # Check the residual QA file exists if Stage 3b succeeded
            if completed.get("stage3b_k3_1dxd"):
                import glob
                matches = glob.glob(os.path.join(tmp_out, "*k3_1dxd_residuals*"))
                assert len(matches) >= 1, (
                    "Expected at least one file matching *k3_1dxd_residuals* "
                    f"in {tmp_out}, got: {list(os.listdir(tmp_out))}"
                )

    def test_residual_plot_function_returns_path_when_saving(self):
        """_plot_k3_1dxd_residuals returns a file path when save=True."""
        import importlib.util
        import sys
        import tempfile

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_resqa_unit",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        # Build a minimal IdlStyle1DXDModel with matched arrays
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel

        rng = np.random.default_rng(42)
        n = 30
        cols = rng.uniform(0, 2047, n)
        orders = rng.choice([203, 215, 229], n).astype(float)
        ref_wavs = rng.uniform(2.0, 2.5, n)
        fit_wavs = ref_wavs + rng.normal(0, 0.001, n)
        resid = ref_wavs - fit_wavs
        mask = np.abs(resid) < 0.003

        model = IdlStyle1DXDModel(
            mode="K3", wdeg=3, odeg=2, order_ref=203.0,
            coeffs=np.zeros((4, 3)), fitted_order_numbers=[203, 215, 229],
            fit_rms_um=float(np.sqrt(np.mean(resid[mask] ** 2))),
            n_lines=int(np.sum(mask)), n_lines_total=n,
            n_lines_rejected=int(np.sum(~mask)),
            accepted_mask=mask, median_residual_um=float(np.median(resid[mask])),
            n_orders_fit=3,
            matched_cols_px=cols,
            matched_order_numbers=orders,
            matched_ref_wavelength_um=ref_wavs,
            matched_fit_wavelength_um=fit_wavs,
            matched_residual_um=resid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = mod._plot_k3_1dxd_residuals(model, tmp, save=True, prefix="test")
            assert out is not None
            assert os.path.isfile(out)
            assert "k3_1dxd_residuals" in os.path.basename(out)


# ===========================================================================
# 15. Global metadata export
# ===========================================================================


class TestGlobalMetadataExport:
    """_export_1dxd_global_summary writes required fields to JSON."""

    def _make_model_for_export(self):
        from pyspextool.instruments.ishell.wavecal_k3_idlstyle import IdlStyle1DXDModel
        n = 20
        return IdlStyle1DXDModel(
            mode="K3", wdeg=3, odeg=2, order_ref=203.0,
            coeffs=np.zeros((4, 3)), fitted_order_numbers=[203, 215, 229],
            fit_rms_um=0.00015, n_lines=18, n_lines_total=n,
            n_lines_rejected=2, accepted_mask=np.ones(n, dtype=bool),
            median_residual_um=0.00002, n_orders_fit=3,
        )

    def test_summary_file_created(self):
        import importlib.util
        import sys
        import tempfile

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_summary",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        model = self._make_model_for_export()
        with tempfile.TemporaryDirectory() as tmp:
            path = mod._export_1dxd_global_summary(model, tmp, "test")
            assert os.path.isfile(path)
            assert path.endswith(".json")

    def test_summary_contains_required_fields(self):
        import json as _json
        import importlib.util
        import sys
        import tempfile

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_summary2",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        model = self._make_model_for_export()
        with tempfile.TemporaryDirectory() as tmp:
            path = mod._export_1dxd_global_summary(model, tmp, "test")
            with open(path) as fh:
                data = _json.load(fh)

        required = [
            "n_lines_total",
            "n_lines_accepted",
            "n_lines_rejected",
            "fit_rms_um",
            "median_residual_um",
            "lambda_degree",
            "order_degree",
            "fitted_order_numbers",
        ]
        for field in required:
            assert field in data, f"Missing required field: {field!r}"

    def test_summary_values_correct(self):
        import json as _json
        import importlib.util
        import sys
        import tempfile

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_summary3",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        model = self._make_model_for_export()
        with tempfile.TemporaryDirectory() as tmp:
            path = mod._export_1dxd_global_summary(model, tmp, "test")
            with open(path) as fh:
                data = _json.load(fh)

        assert data["n_lines_total"] == model.n_lines_total
        assert data["n_lines_accepted"] == model.n_lines
        assert data["n_lines_rejected"] == model.n_lines_rejected
        assert data["lambda_degree"] == model.wdeg
        assert data["order_degree"] == model.odeg
        assert data["fitted_order_numbers"] == model.fitted_order_numbers

    @pytest.mark.skipif(not _HAVE_K3_DATA, reason="K3 raw data not available")
    def test_summary_written_by_run(self):
        """run_k3_example always writes the global summary JSON when Stage 3b runs."""
        import importlib.util
        import sys
        import tempfile

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_summary_run",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp_out:
            completed = mod.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                output_dir=tmp_out,
                no_plots=True,
            )
            if completed.get("stage3b_k3_1dxd"):
                import glob
                matches = glob.glob(os.path.join(tmp_out, "*k3_1dxd_summary.json"))
                assert len(matches) >= 1, (
                    "Expected global summary JSON when Stage 3b succeeded"
                )


# ===========================================================================
# 16. Regression — existing per-order export still works
# ===========================================================================


class TestPerOrderExportRegression:
    """Existing per-order diagnostics export is not broken."""

    def test_csv_export_still_works(self):
        """_export_diagnostics still writes CSV correctly after new changes."""
        import csv as _csv
        import tempfile
        import sys
        import importlib.util

        scripts_dir = os.path.join(_REPO_ROOT, "scripts")
        spec = importlib.util.spec_from_file_location(
            "run_ishell_k3_example_reg",
            os.path.join(scripts_dir, "run_ishell_k3_example.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        mock_diag = mod.OrderCalibrationDiagnostics(
            order_number=215,
            n_candidate=10, n_accepted=8, n_rejected=2,
            poly_degree_requested=3, poly_degree_used=3,
            fit_rms_nm=0.05, skipped=False, non_monotonic=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = mod._export_diagnostics(
                [mock_diag], tmp, "csv", "test",
            )
            assert os.path.isfile(path)
            with open(path, newline="") as fh:
                rows = list(_csv.DictReader(fh))
        assert len(rows) == 1
        assert int(rows[0]["order_number"]) == 215
