"""
Tests for the FITS file-discovery and path-handling utilities (io_utils.py).

Coverage:
  - is_fits_file: recognises .fits and .fits.gz; rejects other extensions.
  - strip_fits_suffix: removes .fits and .fits.gz; leaves unrecognised
    suffixes unchanged.
  - ensure_fits_suffix: plain paths get .fits appended; .fits paths are
    unchanged; .fits.gz paths are rewritten to .fits.
  - find_fits_files: discovers both .fits and .fits.gz in a directory;
    returns a sorted list; ignores non-FITS files.
  - split_fits_path: returns correct (stem, suffix) pairs; raises
    ValueError for non-FITS paths.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from pyspextool.instruments.ishell.io_utils import (
    ensure_fits_suffix,
    find_fits_files,
    is_fits_file,
    split_fits_path,
    strip_fits_suffix,
)


# ---------------------------------------------------------------------------
# is_fits_file
# ---------------------------------------------------------------------------


class TestIsFitsFile:
    def test_plain_fits_str(self):
        assert is_fits_file("frame.fits") is True

    def test_plain_fits_path(self):
        assert is_fits_file(Path("frame.fits")) is True

    def test_compressed_fits_str(self):
        assert is_fits_file("frame.fits.gz") is True

    def test_compressed_fits_path(self):
        assert is_fits_file(Path("/data/obs/frame.fits.gz")) is True

    def test_full_path_plain(self):
        assert is_fits_file("/data/obs/frame.fits") is True

    def test_full_path_compressed(self):
        assert is_fits_file("/data/obs/frame.fits.gz") is True

    def test_txt_extension(self):
        assert is_fits_file("frame.txt") is False

    def test_no_extension(self):
        assert is_fits_file("frame") is False

    def test_gz_only(self):
        assert is_fits_file("frame.gz") is False

    def test_fits_embedded_not_suffix(self):
        # '.fits' inside the stem should not count
        assert is_fits_file("frame.fits.bak") is False


# ---------------------------------------------------------------------------
# strip_fits_suffix
# ---------------------------------------------------------------------------


class TestStripFitsSuffix:
    def test_plain_fits(self):
        assert strip_fits_suffix("frame.fits") == "frame"

    def test_compressed_fits(self):
        assert strip_fits_suffix("frame.fits.gz") == "frame"

    def test_full_path_plain(self):
        result = strip_fits_suffix("/data/obs/frame.fits")
        assert result == "/data/obs/frame"

    def test_full_path_compressed(self):
        result = strip_fits_suffix("/data/obs/frame.fits.gz")
        assert result == "/data/obs/frame"

    def test_path_object_plain(self):
        result = strip_fits_suffix(Path("/data/obs/frame.fits"))
        assert result == "/data/obs/frame"

    def test_path_object_compressed(self):
        result = strip_fits_suffix(Path("/data/obs/frame.fits.gz"))
        assert result == "/data/obs/frame"

    def test_unrecognised_suffix_unchanged(self):
        assert strip_fits_suffix("frame.txt") == "frame.txt"

    def test_no_suffix_unchanged(self):
        assert strip_fits_suffix("frame") == "frame"

    def test_returns_str(self):
        result = strip_fits_suffix(Path("frame.fits"))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ensure_fits_suffix
# ---------------------------------------------------------------------------


class TestEnsureFitsSuffix:
    def test_plain_fits_unchanged(self):
        assert ensure_fits_suffix("frame.fits") == "frame.fits"

    def test_compressed_rewritten(self):
        assert ensure_fits_suffix("frame.fits.gz") == "frame.fits"

    def test_no_suffix_appended(self):
        assert ensure_fits_suffix("frame") == "frame.fits"

    def test_full_path_plain_unchanged(self):
        assert ensure_fits_suffix("/out/result.fits") == "/out/result.fits"

    def test_full_path_compressed_rewritten(self):
        assert ensure_fits_suffix("/out/result.fits.gz") == "/out/result.fits"

    def test_full_path_no_suffix_appended(self):
        assert ensure_fits_suffix("/out/result") == "/out/result.fits"

    def test_path_object_input(self):
        assert ensure_fits_suffix(Path("frame.fits.gz")) == "frame.fits"

    def test_returns_str(self):
        result = ensure_fits_suffix(Path("frame.fits"))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# find_fits_files
# ---------------------------------------------------------------------------


class TestFindFitsFiles:
    def test_finds_plain_fits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.fits").touch()
            results = find_fits_files(tmpdir)
            assert len(results) == 1
            assert results[0].name == "a.fits"

    def test_finds_compressed_fits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "b.fits.gz").touch()
            results = find_fits_files(tmpdir)
            assert len(results) == 1
            assert results[0].name == "b.fits.gz"

    def test_finds_both_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.fits").touch()
            (Path(tmpdir) / "b.fits.gz").touch()
            results = find_fits_files(tmpdir)
            names = [p.name for p in results]
            assert "a.fits" in names
            assert "b.fits.gz" in names
            assert len(results) == 2

    def test_ignores_non_fits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.fits").touch()
            (Path(tmpdir) / "readme.txt").touch()
            (Path(tmpdir) / "data.csv").touch()
            results = find_fits_files(tmpdir)
            assert len(results) == 1
            assert results[0].name == "a.fits"

    def test_returns_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "z.fits").touch()
            (Path(tmpdir) / "a.fits.gz").touch()
            (Path(tmpdir) / "m.fits").touch()
            results = find_fits_files(tmpdir)
            names = [p.name for p in results]
            assert names == sorted(names)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = find_fits_files(tmpdir)
            assert results == []

    def test_mixed_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "flat.fits").touch()
            (Path(tmpdir) / "arc.fits.gz").touch()
            (Path(tmpdir) / "dark.fits").touch()
            (Path(tmpdir) / "log.txt").touch()
            (Path(tmpdir) / "config.yaml").touch()
            results = find_fits_files(tmpdir)
            assert len(results) == 3
            suffixes = {p.name.split(".", 1)[1] for p in results}
            assert suffixes <= {"fits", "fits.gz"}

    def test_returns_path_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.fits").touch()
            results = find_fits_files(tmpdir)
            assert all(isinstance(p, Path) for p in results)


# ---------------------------------------------------------------------------
# split_fits_path
# ---------------------------------------------------------------------------


class TestSplitFitsPath:
    def test_plain_fits(self):
        stem, suffix = split_fits_path("frame.fits")
        assert stem == "frame"
        assert suffix == ".fits"

    def test_compressed_fits(self):
        stem, suffix = split_fits_path("frame.fits.gz")
        assert stem == "frame"
        assert suffix == ".fits.gz"

    def test_full_path_plain(self):
        stem, suffix = split_fits_path("/data/obs/frame.fits")
        assert stem == "/data/obs/frame"
        assert suffix == ".fits"

    def test_full_path_compressed(self):
        stem, suffix = split_fits_path("/data/obs/frame.fits.gz")
        assert stem == "/data/obs/frame"
        assert suffix == ".fits.gz"

    def test_path_object_plain(self):
        stem, suffix = split_fits_path(Path("/data/obs/frame.fits"))
        assert stem == "/data/obs/frame"
        assert suffix == ".fits"

    def test_path_object_compressed(self):
        stem, suffix = split_fits_path(Path("/data/obs/frame.fits.gz"))
        assert stem == "/data/obs/frame"
        assert suffix == ".fits.gz"

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Not a recognised FITS file path"):
            split_fits_path("frame.txt")

    def test_no_suffix_raises_value_error(self):
        with pytest.raises(ValueError):
            split_fits_path("frame")

    def test_stem_is_str(self):
        stem, _ = split_fits_path(Path("frame.fits"))
        assert isinstance(stem, str)

    def test_suffix_is_str(self):
        _, suffix = split_fits_path(Path("frame.fits.gz"))
        assert isinstance(suffix, str)


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify that strip_fits_suffix and ensure_fits_suffix are inverse."""

    @pytest.mark.parametrize("name", ["frame.fits", "frame.fits.gz"])
    def test_ensure_after_strip_gives_fits(self, name):
        stripped = strip_fits_suffix(name)
        result = ensure_fits_suffix(stripped)
        assert result.endswith(".fits")
        assert not result.endswith(".gz")

    @pytest.mark.parametrize("name", ["frame.fits", "frame.fits.gz"])
    def test_is_fits_file_consistent_with_split(self, name):
        assert is_fits_file(name)
        stem, suffix = split_fits_path(name)
        assert suffix in (".fits", ".fits.gz")
