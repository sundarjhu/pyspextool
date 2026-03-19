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
