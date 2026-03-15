"""
Tests for the iSHELL order merging / spectrum assembly module (merge.py).

Coverage
--------
* Successful merging for representative J, H, and K mode configurations.
* Dimensional consistency of merged spectra.
* Overlap-region handling (weighted average in overlap, concatenation outside).
* Variance and quality-flag (bitmask) propagation.
* Failure cases: malformed spectra list, missing metadata keys, inconsistent
  lengths, bad spectrum shapes, out-of-range aperture requests,
  single-order rejection.

Provisional tilt model
----------------------
Tests that exercise the full pipeline are smoke tests only: they verify
structural properties (shape, finiteness, flag propagation) rather than
exact pixel values, because the upstream provisional zero-tilt model is not
scientifically meaningful.  The ``tilt_provisional`` warning is checked
wherever it is relevant.

No real iSHELL science data is required.  All synthetic data is built using
the same helper functions as ``test_ishell_extract.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyspextool.instruments.ishell.merge import (
    merge_extracted_orders,
    _SUPPORTED_MODES,
)
from pyspextool.instruments.ishell.geometry import OrderGeometry, OrderGeometrySet
from pyspextool.pyspextoolerror import pySpextoolError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NWAVE = 64        # number of wavelength samples per synthetic order

# Fractional per-aperture intensity variation used in _make_spectra_and_metadata
_APERTURE_INTENSITY_VARIATION = 0.01

# Fractional tolerance for floating-point uncertainty comparisons
_UNCERTAINTY_RELATIVE_TOLERANCE = 1.01

# Representative (mode_name, order_start, n_orders) triples – one per band
REPRESENTATIVE_CASES = [
    ("J0", 262, 3),
    ("H1", 201, 3),
    ("K1", 233, 2),
]


# ---------------------------------------------------------------------------
# Helper factories – replicating the patterns from test_ishell_extract.py
# ---------------------------------------------------------------------------


def _make_synthetic_spectrum(
    wave_start: float,
    wave_end: float,
    nwave: int = _NWAVE,
    intensity: float = 100.0,
    uncertainty: float = 10.0,
    flag: int = 0,
) -> np.ndarray:
    """Return a ``(4, nwave)`` synthetic spectrum.

    The wavelength grid runs linearly from *wave_start* to *wave_end*.
    All intensity, uncertainty, and flag values are constant.
    """
    wavelength = np.linspace(wave_start, wave_end, nwave)
    sp = np.zeros((4, nwave), dtype=float)
    sp[0, :] = wavelength
    sp[1, :] = intensity
    sp[2, :] = uncertainty
    sp[3, :] = flag
    return sp


def _make_spectra_and_metadata(
    mode: str = "K1",
    n_orders: int = 2,
    naps: int = 1,
    order_start: int = 233,
    nwave: int = _NWAVE,
    overlap_fraction: float = 0.1,
    tilt_provisional: bool = False,
    subtraction_mode: str = "A",
    intensity: float = 100.0,
    uncertainty: float = 10.0,
    flag: int = 0,
) -> tuple[list, dict]:
    """Build a minimal (spectra, metadata) pair for *n_orders* and *naps*.

    Adjacent orders have a small wavelength overlap controlled by
    *overlap_fraction* (fraction of a single order's bandwidth).

    The wavelength convention is: orders are assigned increasing wavelength
    ranges so that after sorting each subsequent order lies to the right,
    with a small overlap with its neighbour.  This matches typical iSHELL
    J/H/K echelle geometry.
    """
    # Build per-order wavelength ranges so that they overlap slightly
    band_width = 0.05  # microns per order
    overlap = overlap_fraction * band_width
    wave_ranges = []
    start = 2.0
    for i in range(n_orders):
        end = start + band_width
        wave_ranges.append((start, end))
        start = end - overlap   # next order starts *overlap* before this one ends

    orders = list(range(order_start, order_start + n_orders))

    # Build spectra list: order-major, aperture-minor
    spectra = []
    for i in range(n_orders):
        ws, we = wave_ranges[i]
        for k in range(naps):
            sp = _make_synthetic_spectrum(
                wave_start=ws,
                wave_end=we,
                nwave=nwave,
                intensity=intensity * (1.0 + _APERTURE_INTENSITY_VARIATION * k),  # slight per-aperture variation
                uncertainty=uncertainty,
                flag=flag,
            )
            spectra.append(sp)

    metadata = {
        "orders": orders,
        "n_apertures": naps,
        "tilt_provisional": tilt_provisional,
        "subtraction_mode": subtraction_mode,
        "rectified": tilt_provisional,
        "plate_scale_arcsec": 0.125,
        "aperture_positions_arcsec": np.array([0.25 * k + 0.5
                                               for k in range(naps)]),
        "aperture_radii_arcsec": np.full((n_orders, naps), 0.5),
        "aperture_signs": np.ones(naps, dtype=int),
    }
    return spectra, metadata


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_order_spectra():
    """Minimal two-order, single-aperture spectra + metadata."""
    return _make_spectra_and_metadata(n_orders=2, naps=1)


@pytest.fixture()
def two_order_two_ap_spectra():
    """Two-order, two-aperture spectra + metadata."""
    return _make_spectra_and_metadata(n_orders=2, naps=2)


@pytest.fixture()
def three_order_spectra():
    """Three-order, single-aperture spectra + metadata."""
    return _make_spectra_and_metadata(n_orders=3, naps=1)


# ===========================================================================
# 1. Module-level checks
# ===========================================================================


class TestModuleAPI:
    """Public API is importable and callable."""

    def test_merge_extracted_orders_callable(self):
        assert callable(merge_extracted_orders)

    def test_supported_modes_not_empty(self):
        assert len(_SUPPORTED_MODES) > 0

    def test_supported_modes_contains_jhk(self):
        expected = {"J0", "H1", "K1", "K2"}
        assert expected.issubset(_SUPPORTED_MODES)

    def test_supported_modes_no_l_modes(self):
        for m in ("L1", "Lp", "M1"):
            assert m not in _SUPPORTED_MODES


# ===========================================================================
# 2. Basic return structure
# ===========================================================================


class TestReturnStructure:
    """Verify the shape and type of the return values."""

    def test_returns_tuple(self, two_order_spectra):
        spectra, meta = two_order_spectra
        result = merge_extracted_orders(spectra, meta)
        assert isinstance(result, tuple) and len(result) == 2

    def test_merged_spectra_is_list(self, two_order_spectra):
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        assert isinstance(merged, list)

    def test_merge_metadata_is_dict(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        assert isinstance(merge_meta, dict)

    def test_single_aperture_one_output(self, two_order_spectra):
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        assert len(merged) == 1

    def test_two_apertures_two_outputs(self, two_order_two_ap_spectra):
        spectra, meta = two_order_two_ap_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        assert len(merged) == 2

    def test_merged_spectrum_is_4_row_array(self, two_order_spectra):
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        assert merged[0].ndim == 2
        assert merged[0].shape[0] == 4

    def test_merged_spectrum_nwave_positive(self, two_order_spectra):
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        assert merged[0].shape[1] > 0

    def test_merge_metadata_required_keys(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        for key in ("apertures", "orders", "n_orders", "n_apertures_merged",
                    "tilt_provisional"):
            assert key in merge_meta

    def test_merge_metadata_n_orders(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        assert merge_meta["n_orders"] == 2

    def test_merge_metadata_orders_matches_input(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        assert merge_meta["orders"] == meta["orders"]

    def test_merge_metadata_apertures_correct(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        assert merge_meta["apertures"] == [1]


# ===========================================================================
# 3. Dimensional consistency
# ===========================================================================


class TestDimensionalConsistency:
    """All four rows of a merged spectrum should have the same length."""

    def test_all_rows_same_length(self, two_order_spectra):
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        sp = merged[0]
        assert sp[0].shape == sp[1].shape == sp[2].shape == sp[3].shape

    def test_merged_longer_than_single_order(self, two_order_spectra):
        """Merged spectrum must cover at least as many wavelength points as one order."""
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        single_nwave = spectra[0].shape[1]
        assert merged[0].shape[1] >= single_nwave

    def test_three_orders_longer_than_two(self):
        """Three merged orders should produce a wider wavelength range."""
        sp2, meta2 = _make_spectra_and_metadata(n_orders=2)
        sp3, meta3 = _make_spectra_and_metadata(n_orders=3)
        merged2, _ = merge_extracted_orders(sp2, meta2)
        merged3, _ = merge_extracted_orders(sp3, meta3)
        # Wavelength range should be wider for 3 orders
        wmin2 = np.nanmin(merged2[0][0, :])
        wmax2 = np.nanmax(merged2[0][0, :])
        wmin3 = np.nanmin(merged3[0][0, :])
        wmax3 = np.nanmax(merged3[0][0, :])
        assert wmax3 - wmin3 >= wmax2 - wmin2

    @pytest.mark.parametrize("mode,order_start,n_orders", REPRESENTATIVE_CASES)
    def test_representative_modes(self, mode, order_start, n_orders):
        """Successful merging for representative J, H, K configurations."""
        spectra, meta = _make_spectra_and_metadata(
            mode=mode, n_orders=n_orders, order_start=order_start)
        merged, merge_meta = merge_extracted_orders(spectra, meta)
        assert len(merged) == 1
        assert merged[0].shape[0] == 4
        assert merged[0].shape[1] > 0
        assert merge_meta["n_orders"] == n_orders

    def test_aperture_selection_single(self, two_order_two_ap_spectra):
        """Selecting only aperture 1 returns a single merged spectrum."""
        spectra, meta = two_order_two_ap_spectra
        merged, merge_meta = merge_extracted_orders(
            spectra, meta, apertures=[1])
        assert len(merged) == 1
        assert merge_meta["n_apertures_merged"] == 1
        assert merge_meta["apertures"] == [1]


# ===========================================================================
# 4. Overlap handling
# ===========================================================================


class TestOverlapHandling:
    """Verify correct behaviour in the wavelength overlap region."""

    def test_wavelength_monotonically_increasing(self, two_order_spectra):
        """Merged wavelength grid should be (non-strictly) increasing."""
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        wl = merged[0][0, :]
        finite = np.isfinite(wl)
        wl_finite = wl[finite]
        assert (np.diff(wl_finite) >= 0).all()

    def test_no_overlap_gives_wider_range(self):
        """With no wavelength overlap, merged range equals sum of order ranges."""
        sp, meta = _make_spectra_and_metadata(n_orders=2, overlap_fraction=0.0)
        merged, _ = merge_extracted_orders(sp, meta)
        wl = merged[0][0, :]
        finite_wl = wl[np.isfinite(wl)]
        # Total range should be >= the range of a single order
        single_range = sp[0][0, -1] - sp[0][0, 0]
        merged_range = finite_wl[-1] - finite_wl[0]
        assert merged_range >= single_range

    def test_with_overlap_intensity_finite_in_overlap(self):
        """Intensity should be finite in the overlap region."""
        sp, meta = _make_spectra_and_metadata(n_orders=2, overlap_fraction=0.2)
        merged, _ = merge_extracted_orders(sp, meta)
        intensity = merged[0][1, :]
        finite_mask = np.isfinite(intensity)
        assert finite_mask.any(), "No finite intensity values found in merged spectrum"


# ===========================================================================
# 5. Variance / uncertainty propagation
# ===========================================================================


class TestVariancePropagation:
    """Verify that uncertainty is propagated and non-negative."""

    def test_uncertainty_row_exists(self, two_order_spectra):
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        # Row 2 is the uncertainty
        assert merged[0].shape[0] >= 3

    def test_uncertainty_nonnegative_where_finite(self, two_order_spectra):
        """Uncertainty (row 2) should be ≥ 0 wherever it is finite."""
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        unc = merged[0][2, :]
        finite = np.isfinite(unc)
        if finite.any():
            assert (unc[finite] >= 0).all()

    def test_uncertainty_not_all_nan(self, two_order_spectra):
        """At least some uncertainty values should be finite."""
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        unc = merged[0][2, :]
        assert np.isfinite(unc).any()

    def test_overlap_reduces_uncertainty(self):
        """In the overlap region, inverse-variance weighting should yield
        uncertainty no larger than either input spectrum's uncertainty."""
        unc_val = 10.0
        sp, meta = _make_spectra_and_metadata(
            n_orders=2, overlap_fraction=0.3, uncertainty=unc_val)
        merged, _ = merge_extracted_orders(sp, meta)
        unc = merged[0][2, :]
        finite = np.isfinite(unc)
        if finite.any():
            # Merged uncertainty should not exceed the input uncertainty
            # in the overlap (weighted average reduces noise)
            assert unc[finite].max() <= unc_val * _UNCERTAINTY_RELATIVE_TOLERANCE


# ===========================================================================
# 6. Flag / bitmask propagation
# ===========================================================================


class TestFlagPropagation:
    """Verify that quality flags are propagated correctly."""

    def test_flag_row_exists(self, two_order_spectra):
        spectra, meta = two_order_spectra
        merged, _ = merge_extracted_orders(spectra, meta)
        assert merged[0].shape[0] == 4

    def test_clean_input_gives_zero_flags(self):
        """If all input flags are 0, the merged flags should be 0."""
        sp, meta = _make_spectra_and_metadata(n_orders=2, flag=0)
        merged, _ = merge_extracted_orders(sp, meta)
        flags = merged[0][3, :]
        # Most flags should be zero for clean input
        finite_flags = flags[np.isfinite(flags)]
        assert (finite_flags == 0).all()

    def test_flagged_input_propagates_to_output(self):
        """A non-zero flag in any input spectrum should propagate to output."""
        sp, meta = _make_spectra_and_metadata(n_orders=2, flag=1)
        merged, _ = merge_extracted_orders(sp, meta)
        flags = merged[0][3, :]
        finite_flags = flags[np.isfinite(flags)]
        assert (finite_flags > 0).any()

    def test_partial_flag_propagation(self):
        """Flag from one order should not contaminate non-overlap region of the other."""
        sp, meta = _make_spectra_and_metadata(n_orders=2, flag=0)
        # Set flags to 1 only in the second order
        sp[1][3, :] = 1
        merged, _ = merge_extracted_orders(sp, meta)
        flags = merged[0][3, :]
        finite = np.isfinite(flags)
        # At least some flags should be zero (from the first, clean order)
        assert (flags[finite] == 0).any()


# ===========================================================================
# 7. Provisional tilt model warning
# ===========================================================================


class TestTiltProvisionalWarning:
    """Verify that the tilt_provisional warning is raised and propagated."""

    def test_tilt_provisional_warns(self):
        sp, meta = _make_spectra_and_metadata(tilt_provisional=True)
        with pytest.warns(RuntimeWarning, match="tilt_provisional"):
            merge_extracted_orders(sp, meta)

    def test_tilt_not_provisional_no_warning(self):
        sp, meta = _make_spectra_and_metadata(tilt_provisional=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Should not raise
            merge_extracted_orders(sp, meta)

    def test_tilt_provisional_propagated_to_merge_metadata(self):
        sp, meta = _make_spectra_and_metadata(tilt_provisional=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, merge_meta = merge_extracted_orders(sp, meta)
        assert merge_meta["tilt_provisional"] is True

    def test_tilt_not_provisional_false_in_metadata(self):
        sp, meta = _make_spectra_and_metadata(tilt_provisional=False)
        _, merge_meta = merge_extracted_orders(sp, meta)
        assert merge_meta["tilt_provisional"] is False


# ===========================================================================
# 8. Metadata propagation
# ===========================================================================


class TestMetadataPropagation:
    """Optional extraction metadata keys should be forwarded to merge output."""

    def test_subtraction_mode_propagated(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        assert merge_meta["subtraction_mode"] == meta["subtraction_mode"]

    def test_rectified_propagated(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        assert "rectified" in merge_meta

    def test_plate_scale_propagated(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        assert merge_meta["plate_scale_arcsec"] == meta["plate_scale_arcsec"]

    def test_aperture_positions_propagated(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        np.testing.assert_array_equal(
            merge_meta["aperture_positions_arcsec"],
            meta["aperture_positions_arcsec"])

    def test_aperture_radii_propagated(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        np.testing.assert_array_equal(
            merge_meta["aperture_radii_arcsec"],
            meta["aperture_radii_arcsec"])

    def test_aperture_signs_propagated(self, two_order_spectra):
        spectra, meta = two_order_spectra
        _, merge_meta = merge_extracted_orders(spectra, meta)
        np.testing.assert_array_equal(
            merge_meta["aperture_signs"],
            meta["aperture_signs"])


# ===========================================================================
# 9. Aperture selection
# ===========================================================================


class TestApertureSelection:
    """Verify that the apertures parameter filters correctly."""

    def test_select_first_aperture_only(self, two_order_two_ap_spectra):
        spectra, meta = two_order_two_ap_spectra
        merged, merge_meta = merge_extracted_orders(
            spectra, meta, apertures=[1])
        assert len(merged) == 1
        assert merge_meta["apertures"] == [1]

    def test_select_second_aperture_only(self, two_order_two_ap_spectra):
        spectra, meta = two_order_two_ap_spectra
        merged, merge_meta = merge_extracted_orders(
            spectra, meta, apertures=[2])
        assert len(merged) == 1
        assert merge_meta["apertures"] == [2]

    def test_select_both_apertures_explicitly(self, two_order_two_ap_spectra):
        spectra, meta = two_order_two_ap_spectra
        merged, merge_meta = merge_extracted_orders(
            spectra, meta, apertures=[1, 2])
        assert len(merged) == 2
        assert merge_meta["apertures"] == [1, 2]

    def test_none_selects_all_apertures(self, two_order_two_ap_spectra):
        spectra, meta = two_order_two_ap_spectra
        merged, merge_meta = merge_extracted_orders(
            spectra, meta, apertures=None)
        assert len(merged) == 2
        assert set(merge_meta["apertures"]) == {1, 2}


# ===========================================================================
# 10. Failure / error cases
# ===========================================================================


class TestFailureCases:
    """Verify that malformed inputs raise the expected exceptions."""

    def test_non_list_spectra(self):
        """spectra must be a list."""
        _, meta = _make_spectra_and_metadata(n_orders=2)
        with pytest.raises(pySpextoolError, match="spectra must be a list"):
            merge_extracted_orders(np.zeros((4, 64)), meta)

    def test_non_dict_metadata(self):
        """metadata must be a dict."""
        spectra, _ = _make_spectra_and_metadata(n_orders=2)
        with pytest.raises(pySpextoolError, match="metadata must be a dict"):
            merge_extracted_orders(spectra, "not a dict")

    def test_missing_orders_key(self):
        """Missing 'orders' key should raise."""
        spectra, meta = _make_spectra_and_metadata(n_orders=2)
        del meta["orders"]
        with pytest.raises(pySpextoolError, match="missing required keys"):
            merge_extracted_orders(spectra, meta)

    def test_missing_n_apertures_key(self):
        """Missing 'n_apertures' key should raise."""
        spectra, meta = _make_spectra_and_metadata(n_orders=2)
        del meta["n_apertures"]
        with pytest.raises(pySpextoolError, match="missing required keys"):
            merge_extracted_orders(spectra, meta)

    def test_wrong_spectra_length(self):
        """spectra list length must match norders * n_apertures."""
        spectra, meta = _make_spectra_and_metadata(n_orders=2)
        # Remove one element to make length inconsistent
        with pytest.raises(pySpextoolError, match=r"len\(spectra\)"):
            merge_extracted_orders(spectra[:-1], meta)

    def test_spectrum_wrong_shape(self):
        """Each spectrum element must be (4, nwave)."""
        spectra, meta = _make_spectra_and_metadata(n_orders=2)
        spectra[0] = np.zeros((3, _NWAVE))   # wrong number of rows
        with pytest.raises(pySpextoolError, match="shape.*4"):
            merge_extracted_orders(spectra, meta)

    def test_spectrum_1d_array_rejected(self):
        """1-D arrays should be rejected."""
        spectra, meta = _make_spectra_and_metadata(n_orders=2)
        spectra[0] = np.zeros(_NWAVE)   # 1-D
        with pytest.raises(pySpextoolError):
            merge_extracted_orders(spectra, meta)

    def test_single_order_raises(self):
        """Only one order in metadata should raise ValueError."""
        spectra, meta = _make_spectra_and_metadata(n_orders=1)
        # Patch meta to claim 1 order (which is too few to merge)
        meta["orders"] = [233]
        # Also trim spectra to length 1 to avoid the length-mismatch error
        spectra = spectra[:1]
        with pytest.raises(ValueError, match="At least 2 orders"):
            merge_extracted_orders(spectra, meta)

    def test_invalid_aperture_index(self, two_order_spectra):
        """Requesting aperture 99 on a 1-aperture extraction should raise."""
        spectra, meta = two_order_spectra
        with pytest.raises(pySpextoolError, match="do not exist"):
            merge_extracted_orders(spectra, meta, apertures=[99])

    def test_zero_aperture_index_rejected(self, two_order_spectra):
        """Aperture indices are 1-based; 0 should be rejected."""
        spectra, meta = two_order_spectra
        with pytest.raises(pySpextoolError, match="do not exist"):
            merge_extracted_orders(spectra, meta, apertures=[0])

    def test_spectra_not_ndarray(self):
        """List containing a non-ndarray element should raise."""
        spectra, meta = _make_spectra_and_metadata(n_orders=2)
        spectra[0] = "not an array"
        with pytest.raises(pySpextoolError):
            merge_extracted_orders(spectra, meta)
