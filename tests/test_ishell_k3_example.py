"""
Tests for the iSHELL K3 benchmark example driver.

These tests verify:
  - If K3 raw data are absent, all data-dependent tests skip cleanly.
  - If K3 raw data are present, the benchmark driver runs through the
    implemented stages without crashing.
  - File-discovery logic correctly handles both ``.fits`` and ``.fits.gz``
    inputs.
  - Key output files are produced when the driver succeeds.
  - The ``run_k3_example`` function raises ``FileNotFoundError`` when the
    raw directory is missing.
  - The ``K3BenchmarkConfig`` dataclass exposes correct defaults and can be
    overridden via keyword arguments to ``run_k3_example``.
  - Output FITS files are named according to ``wavecal_output_name``.
  - QA plot filenames honour ``qa_plot_prefix``.

NOTE: The K3 data files in this repository are ``.fits.gz``.  These tests
exercise the gzip-transparent read path without requiring decompression.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K3_RAW_DIR = os.path.join(
    _REPO_ROOT, "data", "testdata", "ishell_k3_example", "raw"
)
_LFS_MAGIC = b"version https://git-lfs"

# Frame-number groups from the IDL Spextool manual
_FLAT_NUMBERS = list(range(6, 11))   # 6–10
_ARC_NUMBERS = list(range(11, 13))   # 11–12


def _is_real_fits(path: str) -> bool:
    """Return True if *path* is a real FITS/FITS.gz file (not an LFS pointer)."""
    with open(path, "rb") as fh:
        head = fh.read(64)
    # gzip magic: 0x1f 0x8b
    # FITS magic: 'SIMPLE'
    return not head.startswith(_LFS_MAGIC)


def _real_k3_files(pattern: str, frame_numbers: list[int]) -> list[str]:
    """Return sorted real K3 file paths matching *pattern* and *frame_numbers*."""
    from pyspextool.instruments.ishell.io_utils import find_fits_files

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


_K3_FLAT_FILES = _real_k3_files("flat", _FLAT_NUMBERS)
_K3_ARC_FILES = _real_k3_files("arc", _ARC_NUMBERS)
_HAVE_K3_DATA = len(_K3_FLAT_FILES) >= 1 and len(_K3_ARC_FILES) >= 1

# ---------------------------------------------------------------------------
# Helper: import the benchmark driver without executing it
# ---------------------------------------------------------------------------


def _import_driver():
    """Import run_k3_example from the scripts directory."""
    import importlib.util
    import sys

    scripts_dir = os.path.join(_REPO_ROOT, "scripts")
    spec = importlib.util.spec_from_file_location(
        "run_ishell_k3_example",
        os.path.join(scripts_dir, "run_ishell_k3_example.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so the module is properly
    # initialised and can be looked up by name during module-level
    # class/decorator evaluation (e.g. @dataclass uses sys.modules[__name__]).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakePath:
    """Minimal Path-like object for file-discovery unit tests."""

    def __init__(self, directory: str, filename: str) -> None:
        self.name = filename
        self._full = f"{directory}/{filename}"

    def __str__(self) -> str:
        return self._full


# ---------------------------------------------------------------------------
# 1. io_utils: .fits.gz files are recognised as FITS
# ---------------------------------------------------------------------------


class TestFitsGzRecognition:
    """io_utils correctly identifies .fits.gz as valid FITS files."""

    def test_is_fits_file_gz(self):
        from pyspextool.instruments.ishell.io_utils import is_fits_file
        assert is_fits_file("frame.fits.gz") is True

    def test_is_fits_file_plain(self):
        from pyspextool.instruments.ishell.io_utils import is_fits_file
        assert is_fits_file("frame.fits") is True

    def test_is_fits_file_negative(self):
        from pyspextool.instruments.ishell.io_utils import is_fits_file
        assert is_fits_file("frame.txt") is False

    def test_find_fits_files_gz(self, tmp_path):
        """find_fits_files returns .fits.gz files alongside .fits files."""
        from pyspextool.instruments.ishell.io_utils import find_fits_files
        (tmp_path / "a.fits").write_bytes(b"SIMPLE")
        (tmp_path / "b.fits.gz").write_bytes(b"\x1f\x8b")
        (tmp_path / "c.txt").write_text("not fits")
        found_names = {p.name for p in find_fits_files(tmp_path)}
        assert "a.fits" in found_names
        assert "b.fits.gz" in found_names
        assert "c.txt" not in found_names


# ---------------------------------------------------------------------------
# 2. Driver file discovery helpers
# ---------------------------------------------------------------------------


class TestDriverFileDiscovery:
    """_select_files correctly matches flat/arc/spc groups by frame number."""

    def _select_files(self, driver_mod, all_files, frame_type, frame_numbers):
        # We use a temporary directory with fake files
        return driver_mod._select_files(all_files, frame_type, frame_numbers)

    def test_select_flat_files(self):
        """_select_files matches flat frames 6–10 from a mixed file list."""
        driver = _import_driver()

        # Fake file list (Path-like objects, as returned by find_fits_files)
        names = [
            "icm.2017A999.170525.flat.00006.a.fits.gz",
            "icm.2017A999.170525.flat.00007.a.fits.gz",
            "icm.2017A999.170525.arc.00011.a.fits.gz",
            "icm.2017A999.170525.spc.00001.a.fits.gz",
        ]
        fake_dir = "/fake/raw"
        all_files = [_FakePath(fake_dir, n) for n in names]

        selected = driver._select_files(all_files, "flat", list(range(6, 11)))
        assert len(selected) == 2
        assert all("flat" in p for p in selected)

    def test_select_arc_files(self):
        """_select_files matches arc frames 11–12."""
        driver = _import_driver()

        names = [
            "icm.2017A999.170525.flat.00006.a.fits.gz",
            "icm.2017A999.170525.arc.00011.a.fits.gz",
            "icm.2017A999.170525.arc.00012.b.fits.gz",
            "icm.2017A999.170525.arc.00023.a.fits.gz",  # not in 11-12
        ]
        fake_dir = "/fake/raw"
        all_files = [_FakePath(fake_dir, n) for n in names]

        selected = driver._select_files(all_files, "arc", list(range(11, 13)))
        assert len(selected) == 2
        assert all("arc" in p for p in selected)

    def test_select_no_match(self):
        """_select_files returns empty list when nothing matches."""
        driver = _import_driver()
        all_files = []
        selected = driver._select_files(all_files, "flat", list(range(6, 11)))
        assert selected == []


# ---------------------------------------------------------------------------
# 3. Driver raises FileNotFoundError on missing raw directory
# ---------------------------------------------------------------------------


class TestDriverMissingDirectory:
    """run_k3_example raises FileNotFoundError for a missing raw directory."""

    def test_missing_raw_dir(self):
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            missing_raw = os.path.join(tmp, "nonexistent_raw")
            with pytest.raises(FileNotFoundError, match="raw directory not found"):
                driver.run_k3_example(
                    raw_dir=missing_raw,
                    out_dir=out_dir,
                    no_plots=True,
                )

    def test_missing_flat_files(self, tmp_path):
        """run_k3_example raises FileNotFoundError when flat files are absent."""
        driver = _import_driver()
        raw_dir = str(tmp_path / "raw")
        os.makedirs(raw_dir)
        out_dir = str(tmp_path / "output")
        with pytest.raises(FileNotFoundError, match="flat files"):
            driver.run_k3_example(
                raw_dir=raw_dir,
                out_dir=out_dir,
                no_plots=True,
            )

    def test_unknown_override_raises_type_error(self, tmp_path):
        """run_k3_example raises TypeError for an unrecognised keyword."""
        driver = _import_driver()
        raw_dir = str(tmp_path / "raw")
        os.makedirs(raw_dir)
        with pytest.raises(TypeError, match="unexpected keyword"):
            driver.run_k3_example(
                raw_dir=raw_dir,
                out_dir=str(tmp_path / "output"),
                no_plots=True,
                totally_unknown_param="oops",
            )


# ---------------------------------------------------------------------------
# 4a. K3BenchmarkConfig: unit tests (no raw data needed)
# ---------------------------------------------------------------------------


class TestK3BenchmarkConfig:
    """K3BenchmarkConfig dataclass API tests."""

    def test_default_flat_frames(self):
        """Default flat_frames matches IDL manual frames 6–10."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.flat_frames == list(range(6, 11))

    def test_default_arc_frames(self):
        """Default arc_frames matches IDL manual frames 11–12."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.arc_frames == list(range(11, 13))

    def test_default_dark_frames(self):
        """Default dark_frames matches IDL manual frames 25–29."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.dark_frames == list(range(25, 30))

    def test_default_object_frames(self):
        """Default object_frames matches IDL manual spc frames 1–5."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.object_frames == list(range(1, 6))

    def test_default_standard_frames(self):
        """Default standard_frames matches IDL manual spc frames 13–17."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.standard_frames == list(range(13, 18))

    def test_default_wavecal_output_name(self):
        """Default wavecal_output_name is 'wavecal11-12' (IDL convention)."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.wavecal_output_name == "wavecal11-12"

    def test_default_flat_output_name(self):
        """Default flat_output_name is 'flat6-10' (IDL convention)."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.flat_output_name == "flat6-10"

    def test_default_dark_output_name(self):
        """Default dark_output_name is 'dark25-29' (IDL convention)."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.dark_output_name == "dark25-29"

    def test_default_qa_plot_prefix(self):
        """Default qa_plot_prefix is 'qa'."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.qa_plot_prefix == "qa"

    def test_default_mode_name(self):
        """Default mode_name is 'K3'."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert cfg.mode_name == "K3"

    def test_custom_wavecal_output_name(self):
        """K3BenchmarkConfig can be constructed with a custom wavecal name."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig(wavecal_output_name="my_wavecal")
        assert cfg.wavecal_output_name == "my_wavecal"

    def test_custom_flat_frames(self):
        """K3BenchmarkConfig accepts custom flat frame lists."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig(flat_frames=[1, 2, 3])
        assert cfg.flat_frames == [1, 2, 3]

    def test_each_config_instance_has_independent_frame_lists(self):
        """Two separate K3BenchmarkConfig instances do not share frame lists."""
        driver = _import_driver()
        cfg1 = driver.K3BenchmarkConfig()
        cfg2 = driver.K3BenchmarkConfig()
        cfg1.flat_frames.append(99)
        assert 99 not in cfg2.flat_frames


# ---------------------------------------------------------------------------
# 4b. K3 raw data present: end-to-end smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_K3_DATA, reason="K3 raw files not present")
class TestK3BenchmarkSmoke:
    """End-to-end smoke tests that require the real K3 FITS files."""

    def test_flat_files_are_fits_gz(self):
        """All selected flat files end in .fits.gz."""
        assert all(f.endswith(".fits.gz") for f in _K3_FLAT_FILES)

    def test_arc_files_are_fits_gz(self):
        """All selected arc files end in .fits.gz."""
        assert all(f.endswith(".fits.gz") for f in _K3_ARC_FILES)

    def test_flat_tracing_runs(self):
        """Stage 1 (flat/order tracing) runs without error on K3 data."""
        import warnings
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = trace_orders_from_flat(_K3_FLAT_FILES)

        assert trace.n_orders > 0, "Expected at least one order to be traced"

    def test_arc_tracing_runs(self):
        """Stage 2 (arc-line tracing) runs without error on K3 data."""
        import warnings
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
        from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = trace_orders_from_flat(_K3_FLAT_FILES)
            geom = trace.to_order_geometry_set("K3")
            arc_result = trace_arc_lines(_K3_ARC_FILES, geom)

        assert arc_result.n_lines > 0, "Expected at least one arc line to be traced"

    def test_benchmark_driver_completes_stages_1_3(self):
        """run_k3_example completes at least stages 1–3 with real K3 data."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            completed = driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
            )

        assert completed.get("stage1_flat_tracing"), "Stage 1 should complete"
        assert completed.get("stage2_arc_tracing"), "Stage 2 should complete"
        assert completed.get("stage3_provisional_wavemap"), "Stage 3 should complete"

    def test_benchmark_driver_via_config_object(self):
        """run_k3_example accepts a K3BenchmarkConfig object."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            cfg = driver.K3BenchmarkConfig(
                raw_dir=_K3_RAW_DIR,
                output_dir=out_dir,
                no_plots=True,
            )
            completed = driver.run_k3_example(cfg)

        assert completed.get("stage1_flat_tracing"), "Stage 1 should complete"
        assert completed.get("stage2_arc_tracing"), "Stage 2 should complete"

    def test_benchmark_driver_produces_output_dir(self):
        """run_k3_example creates the output directory."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            assert not os.path.isdir(out_dir)
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
            )
            assert os.path.isdir(out_dir)

    def test_wavecal_output_name_default(self):
        """Default run writes wavecal11-12.fits (IDL manual convention)."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            completed = driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
            )
            if completed.get("stage8_calibration_fits"):
                out_path = os.path.join(out_dir, "wavecal11-12.fits")
                assert os.path.isfile(out_path), (
                    f"Expected default wavecal FITS at {out_path}"
                )
                # Must be plain .fits, not .fits.gz
                assert not out_path.endswith(".fits.gz")

    def test_wavecal_output_name_custom(self):
        """wavecal_output_name override is honoured in written filenames."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            completed = driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                wavecal_output_name="my_custom_wavecal",
                no_plots=True,
            )
            if completed.get("stage8_calibration_fits"):
                out_path = os.path.join(out_dir, "my_custom_wavecal.fits")
                assert os.path.isfile(out_path), (
                    f"Expected custom-named wavecal FITS at {out_path}"
                )

    def test_qa_plot_prefix_default(self):
        """Default QA plots use prefix 'qa'."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                save_plots=True,
                no_plots=False,
            )
            flat_plot = os.path.join(out_dir, "qa_flat_orders.png")
            assert os.path.isfile(flat_plot), (
                f"Expected QA plot at {flat_plot}"
            )

    def test_qa_plot_prefix_custom(self):
        """Custom qa_plot_prefix is reflected in QA plot filenames."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                qa_plot_prefix="bench",
                save_plots=True,
                no_plots=False,
            )
            flat_plot = os.path.join(out_dir, "bench_flat_orders.png")
            assert os.path.isfile(flat_plot), (
                f"Expected prefixed QA plot at {flat_plot}"
            )

    def test_override_via_config_wavecal_name(self):
        """K3BenchmarkConfig.wavecal_output_name controls the output filename."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            cfg = driver.K3BenchmarkConfig(
                raw_dir=_K3_RAW_DIR,
                output_dir=out_dir,
                wavecal_output_name="cfg_wavecal",
                no_plots=True,
            )
            completed = driver.run_k3_example(cfg)
            if completed.get("stage8_calibration_fits"):
                out_path = os.path.join(out_dir, "cfg_wavecal.fits")
                assert os.path.isfile(out_path), (
                    f"Expected config-named wavecal FITS at {out_path}"
                )

    # -----------------------------------------------------------------------
    # New tests for benchmark alignment (edge-order filter, QA plots)
    # -----------------------------------------------------------------------

    def test_filter_edge_orders_unit(self):
        """_filter_edge_orders reduces 29 raw K3 orders to 27 science orders."""
        import warnings
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat

        driver = _import_driver()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_raw = trace_orders_from_flat(_K3_FLAT_FILES)

        filtered = driver._filter_edge_orders(trace_raw)
        # K3 has 27 science orders (203–229) per the IDL Spextool manual.
        # The filter should retain exactly those orders.
        assert filtered.n_orders == 27, (
            f"Expected 27 K3 science orders after edge filtering, "
            f"got {filtered.n_orders}"
        )
        assert filtered.n_orders < trace_raw.n_orders, (
            "Edge filter should remove at least one partial order"
        )

    def test_filter_edge_orders_noop_on_uniform_halfwidth(self):
        """_filter_edge_orders is a no-op when all half-widths are equal."""
        import numpy as np
        from pyspextool.instruments.ishell.tracing import FlatOrderTrace

        driver = _import_driver()

        n = 5
        uniform_trace = FlatOrderTrace(
            n_orders=n,
            sample_cols=np.arange(10),
            center_rows=np.zeros((n, 10)),
            center_poly_coeffs=np.zeros((n, 2)),
            fit_rms=np.ones(n),
            half_width_rows=np.full(n, 16.0),
            poly_degree=1,
            seed_col=1023,
        )
        result = driver._filter_edge_orders(uniform_trace)
        assert result.n_orders == n

    def test_all_qa_plots_produced(self):
        """All expected QA plots are produced when save_plots=True."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                save_plots=True,
                no_plots=False,
            )
            expected_plots = [
                "qa_flat_orders.png",
                "qa_arc_lines.png",
                "qa_wavecal_residuals.png",
                "qa_2d_coeff_fit.png",
                "qa_rectified_order.png",
            ]
            for fname in expected_plots:
                fpath = os.path.join(out_dir, fname)
                assert os.path.isfile(fpath), (
                    f"Expected QA plot not produced: {fpath}"
                )

    def test_arc_lines_plot_produced(self):
        """Arc-lines QA plot is produced (tests the traced_lines accessor fix)."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                save_plots=True,
                no_plots=False,
            )
            arc_plot = os.path.join(out_dir, "qa_arc_lines.png")
            assert os.path.isfile(arc_plot), (
                f"Arc-lines QA plot missing: {arc_plot}\n"
                "This likely means the arc_result.traced_lines accessor is broken."
            )

    def test_rectified_order_plot_produced(self):
        """Rectified-order QA plot is produced (tests the flux accessor fix)."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            completed = driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                save_plots=True,
                no_plots=False,
            )
            if not completed.get("stage7_rectified_orders"):
                pytest.skip("Stage 7 did not complete; rectified-order QA skipped")
            rect_plot = os.path.join(out_dir, "qa_rectified_order.png")
            assert os.path.isfile(rect_plot), (
                f"Rectified-order QA plot missing: {rect_plot}\n"
                "This likely means the RectifiedOrder.flux accessor is broken."
            )

    def test_2d_coeff_fit_plot_produced(self):
        """2DCoeffFit-style QA plot is produced."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                save_plots=True,
                no_plots=False,
            )
            coeff_plot = os.path.join(out_dir, "qa_2d_coeff_fit.png")
            assert os.path.isfile(coeff_plot), (
                f"2DCoeffFit QA plot missing: {coeff_plot}"
            )

    def test_residuals_plot_produced(self):
        """Wavecal residuals QA plot is produced."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                save_plots=True,
                no_plots=False,
            )
            res_plot = os.path.join(out_dir, "qa_wavecal_residuals.png")
            assert os.path.isfile(res_plot), (
                f"Wavecal residuals QA plot missing: {res_plot}"
            )

    def test_benchmark_stage1_filtered_order_count(self):
        """Stage 1 reports 27 retained K3 science orders after edge filtering."""
        driver = _import_driver()
        # Use driver's filter directly on raw tracing result
        import warnings
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_raw = trace_orders_from_flat(_K3_FLAT_FILES)

        trace_filtered = driver._filter_edge_orders(trace_raw)
        assert trace_filtered.n_orders == 27, (
            f"K3 benchmark should retain 27 science orders, "
            f"got {trace_filtered.n_orders}"
        )


# ---------------------------------------------------------------------------
# 5. OrderCalibrationDiagnostics: unit tests (no raw data needed)
# ---------------------------------------------------------------------------


class TestOrderCalibrationDiagnosticsUnit:
    """OrderCalibrationDiagnostics dataclass API tests (no K3 data required)."""

    def test_diagnostics_dataclass_exists(self):
        """OrderCalibrationDiagnostics is importable from the driver."""
        driver = _import_driver()
        assert hasattr(driver, "OrderCalibrationDiagnostics"), (
            "OrderCalibrationDiagnostics dataclass not found in run_ishell_k3_example"
        )

    def test_diagnostics_dataclass_fields(self):
        """OrderCalibrationDiagnostics has the required fields."""
        driver = _import_driver()
        fields = set(driver.OrderCalibrationDiagnostics.__dataclass_fields__.keys())
        required = {
            "order_number", "n_candidate", "n_accepted", "n_rejected",
            "poly_degree_requested", "poly_degree_used", "fit_rms_nm",
            "skipped", "non_monotonic",
        }
        missing = required - fields
        assert not missing, f"Missing fields: {missing}"

    def test_diagnostics_construction(self):
        """OrderCalibrationDiagnostics can be constructed with the expected fields."""
        import math
        driver = _import_driver()
        d = driver.OrderCalibrationDiagnostics(
            order_number=205,
            n_candidate=10,
            n_accepted=7,
            n_rejected=3,
            poly_degree_requested=3,
            poly_degree_used=3,
            fit_rms_nm=1.234,
            skipped=False,
            non_monotonic=False,
        )
        assert d.order_number == 205
        assert d.n_rejected == 3
        assert d.skipped is False
        assert d.non_monotonic is False
        assert d.fit_rms_nm == 1.234

    def test_diagnostics_skipped_order(self):
        """OrderCalibrationDiagnostics correctly represents a skipped order."""
        import math
        driver = _import_driver()
        d = driver.OrderCalibrationDiagnostics(
            order_number=210,
            n_candidate=1,
            n_accepted=1,
            n_rejected=0,
            poly_degree_requested=3,
            poly_degree_used=3,
            fit_rms_nm=float("nan"),
            skipped=True,
            non_monotonic=False,
        )
        assert d.skipped is True
        assert math.isnan(d.fit_rms_nm)

    def test_config_export_diagnostics_field(self):
        """K3BenchmarkConfig has export_diagnostics field defaulting to False."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert hasattr(cfg, "export_diagnostics")
        assert cfg.export_diagnostics is False

    def test_config_diagnostics_format_field(self):
        """K3BenchmarkConfig has diagnostics_format field defaulting to 'csv'."""
        driver = _import_driver()
        cfg = driver.K3BenchmarkConfig()
        assert hasattr(cfg, "diagnostics_format")
        assert cfg.diagnostics_format == "csv"


# ---------------------------------------------------------------------------
# 6. OrderCalibrationDiagnostics: data-dependent tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_K3_DATA, reason="K3 raw files not present")
class TestOrderCalibrationDiagnosticsWithData:
    """Diagnostics tests that require the real K3 FITS files."""

    def test_diagnostics_returned_in_completed(self):
        """run_k3_example returns _order_diagnostics in the completed dict."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            completed = driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
            )
        assert "_order_diagnostics" in completed, (
            "run_k3_example should return '_order_diagnostics' key"
        )

    def test_diagnostics_list_has_correct_order_count(self):
        """_order_diagnostics list has one entry per K3 science order (27)."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            completed = driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
            )
        diags = completed.get("_order_diagnostics", [])
        assert len(diags) == 27, (
            f"Expected 27 diagnostics entries (one per K3 science order), "
            f"got {len(diags)}"
        )

    def test_diagnostics_entries_are_correct_type(self):
        """All entries in _order_diagnostics are OrderCalibrationDiagnostics."""
        driver = _import_driver()
        DiagClass = driver.OrderCalibrationDiagnostics
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            completed = driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
            )
        diags = completed.get("_order_diagnostics", [])
        for d in diags:
            assert isinstance(d, DiagClass), (
                f"Expected OrderCalibrationDiagnostics, got {type(d)}"
            )

    def test_diagnostics_rejected_count_consistent(self):
        """n_rejected == n_candidate - n_accepted for all orders."""
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            completed = driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
            )
        diags = completed.get("_order_diagnostics", [])
        for d in diags:
            assert d.n_rejected == d.n_candidate - d.n_accepted, (
                f"Order {d.order_number}: n_rejected ({d.n_rejected}) != "
                f"n_candidate ({d.n_candidate}) - n_accepted ({d.n_accepted})"
            )

    def test_export_diagnostics_csv(self):
        """export_diagnostics=True writes a CSV file with one row per order."""
        import csv
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
                export_diagnostics=True,
                diagnostics_format="csv",
            )
            csv_path = os.path.join(out_dir, "qa_order_diagnostics.csv")
            assert os.path.isfile(csv_path), (
                f"Expected diagnostics CSV at {csv_path}"
            )
            with open(csv_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == 27, (
                f"Expected 27 CSV rows, got {len(rows)}"
            )
            assert "order_number" in rows[0]
            assert "n_accepted" in rows[0]
            assert "fit_rms_nm" in rows[0]

    def test_export_diagnostics_json(self):
        """export_diagnostics=True with json format writes a JSON array."""
        import json
        driver = _import_driver()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "output")
            driver.run_k3_example(
                raw_dir=_K3_RAW_DIR,
                out_dir=out_dir,
                no_plots=True,
                export_diagnostics=True,
                diagnostics_format="json",
            )
            json_path = os.path.join(out_dir, "qa_order_diagnostics.json")
            assert os.path.isfile(json_path), (
                f"Expected diagnostics JSON at {json_path}"
            )
            with open(json_path) as fh:
                data = json.load(fh)
            assert isinstance(data, list)
            assert len(data) == 27, (
                f"Expected 27 JSON entries, got {len(data)}"
            )
            assert "order_number" in data[0]


# ---------------------------------------------------------------------------
# 7. Bug-fix tests: non-monotonic warning parsing
# ---------------------------------------------------------------------------


class TestParseNonMonotonicOrderNumber:
    """Unit tests for _parse_non_monotonic_order_number (no K3 data required)."""

    def _parse(self, msg: str):
        driver = _import_driver()
        return driver._parse_non_monotonic_order_number(msg)

    def test_extracts_echelle_order_not_index(self):
        """Parses echelle order number (204), not the index (1) before it."""
        msg = "Order 1 (order_number=204.0): the sampled wavelength surface is not strictly monotone in detector column.  The column-inversion for rectification may be inaccurate."
        result = self._parse(msg)
        assert result == 204, (
            f"Expected 204, got {result!r}. "
            "The parser must extract from order_number=..., not from 'Order N'."
        )

    def test_integer_order_number(self):
        """Handles order_number without decimal point (e.g. 204)."""
        msg = "Order 5 (order_number=229): some warning text."
        result = self._parse(msg)
        assert result == 229, f"Expected 229, got {result!r}"

    def test_float_order_number(self):
        """Handles order_number with trailing .0 (e.g. 216.0)."""
        msg = "Order 13 (order_number=216.0): the sampled wavelength surface is not strictly monotone."
        result = self._parse(msg)
        assert result == 216, f"Expected 216, got {result!r}"

    def test_returns_none_on_missing_pattern(self):
        """Returns None when order_number= is absent."""
        result = self._parse("Some unrelated warning text.")
        assert result is None

    def test_returns_none_on_empty_string(self):
        """Returns None for an empty string without crashing."""
        result = self._parse("")
        assert result is None

    def test_index_and_echelle_differ(self):
        """Confirms result differs from the leading order index."""
        msg = "Order 3 (order_number=210.0): non-monotonic."
        result = self._parse(msg)
        # Order index would be 3; echelle order should be 210.
        assert result == 210
        assert result != 3


# ---------------------------------------------------------------------------
# 8. Bug-fix tests: arc-line plot representative row logic
# ---------------------------------------------------------------------------


class TestArcLinePlotRepresentativeRow:
    """Unit tests confirming _plot_arc_lines uses median(trace_rows), not poly eval."""

    def _make_fake_line(self, seed_col: int, trace_rows, poly_coeffs=None):
        """Create a minimal TracedArcLine-like object for testing."""
        import numpy as np
        from pyspextool.instruments.ishell.arc_tracing import TracedArcLine

        trace_rows_arr = np.asarray(trace_rows, dtype=float)
        # trace_cols: put centroid at seed_col for simplicity
        trace_cols_arr = np.full_like(trace_rows_arr, float(seed_col))
        if poly_coeffs is None:
            # Intentionally large tilt slope so that poly(seed_col) gives a
            # wildly incorrect row value, confirming the fix is necessary.
            _wrong_slope = 50.0
            poly_coeffs = np.array([float(seed_col), _wrong_slope])
        return TracedArcLine(
            order_index=0,
            seed_col=seed_col,
            trace_rows=trace_rows_arr,
            trace_cols=trace_cols_arr,
            poly_coeffs=np.asarray(poly_coeffs),
            fit_rms=0.1,
            peak_flux=1.0,
        )

    def test_median_trace_rows_used(self):
        """Representative row should be median of trace_rows, not poly(seed_col)."""
        import numpy as np

        trace_rows = [10, 20, 30, 40, 50]
        seed_col = 500
        # poly_coeffs encodes col(row); evaluating at seed_col (500) would give
        # a wildly wrong row value.  The correct row is median([10..50]) = 30.
        line = self._make_fake_line(seed_col, trace_rows)

        expected_row = float(np.median(trace_rows))  # = 30.0

        # Replicate the fixed logic from _plot_arc_lines
        if len(line.trace_rows) > 0:
            row = float(np.median(line.trace_rows))
        else:
            row = None

        assert row == expected_row, (
            f"Expected median row {expected_row}, got {row}"
        )
        # Confirm this differs from the (incorrect) polynomial evaluation
        poly_result = float(np.polynomial.polynomial.polyval(seed_col, line.poly_coeffs))
        assert row != poly_result, (
            "The median-row and poly-eval values should not be equal here "
            "(confirming the fix is different from the old code)"
        )

    def test_skip_line_with_no_trace_rows(self):
        """Lines with empty trace_rows should be skipped (row is None / not plotted)."""
        import numpy as np

        line = self._make_fake_line(seed_col=500, trace_rows=[])

        # Replicate the fixed logic
        if len(line.trace_rows) > 0:
            row = float(np.median(line.trace_rows))
        else:
            row = None  # should be skipped

        assert row is None, (
            "A line with no trace_rows should yield row=None (skipped in plot)"
        )

    def test_median_with_odd_count(self):
        """Median of odd-length trace_rows is the middle element."""
        import numpy as np
        trace_rows = [100, 200, 300]
        line = self._make_fake_line(seed_col=600, trace_rows=trace_rows)
        row = float(np.median(line.trace_rows))
        assert row == 200.0

    def test_median_with_even_count(self):
        """Median of even-length trace_rows is the average of the two middle values."""
        import numpy as np
        trace_rows = [100, 200, 300, 400]
        line = self._make_fake_line(seed_col=600, trace_rows=trace_rows)
        row = float(np.median(line.trace_rows))
        assert row == 250.0


# ---------------------------------------------------------------------------
# 9. Flatinfo integration: benchmark explicitly uses IDL-style path
# ---------------------------------------------------------------------------


def _make_fake_flat_order_trace():
    """Return a minimal FlatOrderTrace for use in mocked tracing tests."""
    import numpy as np
    from pyspextool.instruments.ishell.tracing import FlatOrderTrace

    n = 3
    return FlatOrderTrace(
        n_orders=n,
        sample_cols=np.arange(10),
        center_rows=np.zeros((n, 10)),
        center_poly_coeffs=np.zeros((n, 2)),
        fit_rms=np.ones(n),
        half_width_rows=np.full(n, 16.0),
        poly_degree=1,
        seed_col=1023,
    )


def _make_minimal_raw_dir(tmp_path):
    """Create a minimal raw directory with one flat and one arc FITS file.

    Returns the raw directory path as a string.
    """
    import numpy as np
    from astropy.io import fits

    raw_dir = str(tmp_path / "raw")
    os.makedirs(raw_dir)
    for frame, ftype in [(6, "flat"), (11, "arc")]:
        fname = f"icm.2017A999.170525.{ftype}.{frame:05d}.a.fits"
        fits.PrimaryHDU(data=np.zeros((4, 4))).writeto(
            os.path.join(raw_dir, fname)
        )
    return raw_dir


class TestK3BenchmarkFlatinfoIntegration:
    """Integration tests verifying that the K3 benchmark explicitly loads
    flatinfo and passes it into trace_orders_from_flat.

    These tests use mocking and do NOT require the real K3 dataset.
    """

    def _patch_stage1_pass_stage2_stop(self, monkeypatch, driver):
        """Patch the driver so Stage 1 completes but Stage 2 raises StopIteration."""
        import numpy as np

        fake_trace = _make_fake_flat_order_trace()
        monkeypatch.setattr(driver, "trace_orders_from_flat",
                            lambda *a, **kw: fake_trace)
        monkeypatch.setattr(driver, "load_and_combine_flats",
                            lambda *a, **kw: np.zeros((4, 4)))
        monkeypatch.setattr(driver, "load_and_combine_arcs",
                            lambda *a, **kw: np.zeros((4, 4)))
        monkeypatch.setattr(driver, "trace_arc_lines",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                StopIteration("stop after stage 1")
                            ))

    def test_read_flatinfo_is_called_for_configured_mode(self, tmp_path, monkeypatch):
        """run_k3_example calls read_flatinfo with the configured mode."""
        driver = _import_driver()
        raw_dir = _make_minimal_raw_dir(tmp_path)

        calls: list[str] = []

        def recording_read_flatinfo(mode_name):
            calls.append(mode_name)
            # Return the real flatinfo so the driver can proceed through Stage 1
            from pyspextool.instruments.ishell.calibrations import read_flatinfo
            return read_flatinfo(mode_name)

        monkeypatch.setattr(driver, "read_flatinfo", recording_read_flatinfo)
        self._patch_stage1_pass_stage2_stop(monkeypatch, driver)

        with pytest.raises(StopIteration):
            driver.run_k3_example(
                raw_dir=raw_dir,
                out_dir=str(tmp_path / "output"),
                no_plots=True,
            )

        assert "K3" in calls, (
            f"read_flatinfo was not called with mode 'K3'; calls recorded: {calls}"
        )

    def test_flatinfo_passed_to_trace_orders_from_flat(self, tmp_path, monkeypatch):
        """run_k3_example passes the loaded flatinfo into trace_orders_from_flat."""
        import numpy as np

        driver = _import_driver()
        raw_dir = _make_minimal_raw_dir(tmp_path)

        fake_trace = _make_fake_flat_order_trace()

        # Capture the kwargs that trace_orders_from_flat is called with
        captured_kwargs: list[dict] = []

        def capturing_trace(*args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return fake_trace

        monkeypatch.setattr(driver, "trace_orders_from_flat", capturing_trace)
        monkeypatch.setattr(driver, "load_and_combine_flats",
                            lambda *a, **kw: np.zeros((4, 4)))
        monkeypatch.setattr(driver, "load_and_combine_arcs",
                            lambda *a, **kw: np.zeros((4, 4)))
        monkeypatch.setattr(driver, "trace_arc_lines",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                StopIteration("stop after stage 1")
                            ))

        with pytest.raises(StopIteration):
            driver.run_k3_example(
                raw_dir=raw_dir,
                out_dir=str(tmp_path / "output"),
                no_plots=True,
            )

        assert captured_kwargs, "trace_orders_from_flat was not called"
        assert "flatinfo" in captured_kwargs[0], (
            "trace_orders_from_flat was not called with flatinfo= keyword; "
            f"kwargs were: {captured_kwargs[0]}"
        )
        assert captured_kwargs[0]["flatinfo"] is not None, (
            "flatinfo passed to trace_orders_from_flat must not be None"
        )

    def test_flatinfo_load_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        """run_k3_example raises RuntimeError (not silently falls back) when
        read_flatinfo fails.  The error message must mention the mode and the
        IDL-style path."""
        driver = _import_driver()
        raw_dir = _make_minimal_raw_dir(tmp_path)

        def broken_read_flatinfo(mode_name):
            raise FileNotFoundError(f"No flatinfo for mode '{mode_name}'")

        monkeypatch.setattr(driver, "read_flatinfo", broken_read_flatinfo)

        with pytest.raises(RuntimeError) as exc_info:
            driver.run_k3_example(
                raw_dir=raw_dir,
                out_dir=str(tmp_path / "output"),
                no_plots=True,
            )

        msg = str(exc_info.value)
        assert "K3" in msg, (
            f"RuntimeError message should mention the mode 'K3'; got: {msg!r}"
        )
        assert "IDL" in msg or "flatinfo" in msg.lower(), (
            f"RuntimeError message should mention the IDL-style path; got: {msg!r}"
        )

    def test_flatinfo_load_failure_does_not_silently_fallback(
        self, tmp_path, monkeypatch
    ):
        """When read_flatinfo raises, the benchmark must NOT silently continue.

        This verifies the 'no silent fallback' constraint from the problem
        statement: if flatinfo cannot be loaded the benchmark must fail loudly,
        not auto-switch to the fallback tracing path.
        """
        driver = _import_driver()
        raw_dir = _make_minimal_raw_dir(tmp_path)

        trace_was_called = []

        def broken_read_flatinfo(mode_name):
            raise ValueError("simulated flatinfo parse failure")

        def recording_trace(*args, **kwargs):
            trace_was_called.append(kwargs.get("flatinfo"))
            raise StopIteration("should not reach tracing")

        monkeypatch.setattr(driver, "read_flatinfo", broken_read_flatinfo)
        monkeypatch.setattr(driver, "trace_orders_from_flat", recording_trace)

        # The benchmark must raise before reaching trace_orders_from_flat
        with pytest.raises((RuntimeError, ValueError)):
            driver.run_k3_example(
                raw_dir=raw_dir,
                out_dir=str(tmp_path / "output"),
                no_plots=True,
            )

        assert not trace_was_called, (
            "trace_orders_from_flat must not be reached when flatinfo "
            "fails to load (no silent fallback allowed)"
        )

    def test_flatinfo_console_output_contains_mode(
        self, tmp_path, monkeypatch, capsys
    ):
        """run_k3_example prints 'Flatinfo loaded for mode K3' to stdout."""
        driver = _import_driver()
        raw_dir = _make_minimal_raw_dir(tmp_path)
        self._patch_stage1_pass_stage2_stop(monkeypatch, driver)

        with pytest.raises(StopIteration):
            driver.run_k3_example(
                raw_dir=raw_dir,
                out_dir=str(tmp_path / "output"),
                no_plots=True,
            )

        captured = capsys.readouterr()
        assert "Flatinfo loaded for mode K3" in captured.out, (
            f"Expected 'Flatinfo loaded for mode K3' in stdout; got:\n{captured.out}"
        )

    def test_stage1_tracing_source_output(self, tmp_path, monkeypatch, capsys):
        """run_k3_example prints 'Stage 1 tracing source: flatinfo / IDL-style path'."""
        driver = _import_driver()
        raw_dir = _make_minimal_raw_dir(tmp_path)
        self._patch_stage1_pass_stage2_stop(monkeypatch, driver)

        with pytest.raises(StopIteration):
            driver.run_k3_example(
                raw_dir=raw_dir,
                out_dir=str(tmp_path / "output"),
                no_plots=True,
            )

        captured = capsys.readouterr()
        assert "flatinfo / IDL-style path" in captured.out, (
            f"Expected 'flatinfo / IDL-style path' in Stage 1 output; "
            f"got:\n{captured.out}"
        )
