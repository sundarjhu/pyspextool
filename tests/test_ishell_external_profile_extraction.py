"""
Tests for the external-profile extraction workflow (external_profile_extraction.py).

Coverage:
  - ExternalProfileExtractionResult: construction, convenience properties.
  - extract_with_external_profile on synthetic data:
      * basic extraction runs using external templates (single order),
      * multi-order extraction runs correctly,
      * results differ from empirical extraction in a noisy case,
      * fallback profile source works for orders without a template,
      * partial template set with fallback extracts all orders,
      * missing template raises ValueError when no fallback provided,
      * spatial dimension mismatch raises ValueError,
      * spectral dimension mismatch raises ValueError,
      * mode mismatch raises ValueError,
      * empty rectified order set raises ValueError,
      * invalid fallback_profile_source raises ValueError,
      * bookkeeping fields (external_profile_applied, template_n_frames_used)
        are populated correctly.
  - Integration test:
      * build templates using Stage 21, apply using this module,
      * verify extraction completes and outputs are valid,
      * verify external_profile_applied is True for all orders.
  - Smoke test on real H1 calibration data (skipped when LFS files absent):
      * build templates from a calibration dataset (rectified from flat frames),
      * apply templates to a distinct science dataset (small-noise perturbation
        of the calibration image, simulating a separate science exposure),
      * verify no crashes, correct shapes, finite flux/variance values,
        external_profile_applied is True for all orders.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyspextool.instruments.ishell.external_profile_extraction import (
    ExternalProfileExtractionResult,
    extract_with_external_profile,
)
from pyspextool.instruments.ishell.profile_templates import (
    ExternalProfileTemplate,
    ExternalProfileTemplateSet,
    ProfileTemplateDefinition,
    build_external_profile_template,
)
from pyspextool.instruments.ishell.rectified_orders import (
    RectifiedOrder,
    RectifiedOrderSet,
)
from pyspextool.instruments.ishell.weighted_optimal_extraction import (
    WeightedExtractionDefinition,
    WeightedExtractedSpectrumSet,
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

_N_SPECTRAL = 32
_N_SPATIAL = 20
_N_ORDERS = 3

_ORDER_PARAMS = [
    {"order_number": 311},
    {"order_number": 315},
    {"order_number": 320},
]


def _make_rectified_order_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
    fill_value: float = 2.0,
    mode: str = "H1_test",
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.0,
) -> RectifiedOrderSet:
    """Build a RectifiedOrderSet with constant (or noisy) flux images."""
    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        flux = np.full((n_spatial, n_spectral), fill_value)
        if noise_scale > 0.0 and rng is not None:
            flux = flux + rng.normal(0.0, noise_scale, size=(n_spatial, n_spectral))
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=np.linspace(wav_start, wav_end, n_spectral),
                spatial_frac=np.linspace(0.0, 1.0, n_spatial),
                flux=flux,
                source_image_shape=(200, 800),
            )
        )
    return RectifiedOrderSet(
        mode=mode,
        rectified_orders=orders,
        source_image_shape=(200, 800),
    )


def _make_gaussian_profile_set(
    n_spectral: int = _N_SPECTRAL,
    n_spatial: int = _N_SPATIAL,
    center_frac: float = 0.5,
    sigma_frac: float = 0.1,
    mode: str = "H1_test",
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.0,
) -> RectifiedOrderSet:
    """Build a RectifiedOrderSet with a Gaussian spatial profile."""
    spatial = np.linspace(0.0, 1.0, n_spatial)
    profile_1d = np.exp(-0.5 * ((spatial - center_frac) / sigma_frac) ** 2)
    profile_1d = profile_1d / profile_1d.sum()

    orders = []
    for idx, p in enumerate(_ORDER_PARAMS):
        wav_start = 1.55 + idx * 0.05
        wav_end = wav_start + 0.04
        flux = np.outer(profile_1d, np.ones(n_spectral)) * 100.0
        if noise_scale > 0.0 and rng is not None:
            flux = flux + rng.normal(0.0, noise_scale, size=(n_spatial, n_spectral))
        orders.append(
            RectifiedOrder(
                order=p["order_number"],
                order_index=idx,
                wavelength_um=np.linspace(wav_start, wav_end, n_spectral),
                spatial_frac=spatial.copy(),
                flux=flux,
                source_image_shape=(200, 800),
            )
        )
    return RectifiedOrderSet(
        mode=mode,
        rectified_orders=orders,
        source_image_shape=(200, 800),
    )


def _build_templates(
    rectified_set: RectifiedOrderSet,
    definition: ProfileTemplateDefinition | None = None,
) -> ExternalProfileTemplateSet:
    """Build an ExternalProfileTemplateSet from a RectifiedOrderSet."""
    if definition is None:
        definition = ProfileTemplateDefinition()
    return build_external_profile_template(rectified_set, definition)


def _default_extraction_def(
    n_spatial: int = _N_SPATIAL,
) -> WeightedExtractionDefinition:
    """Return a sensible extraction definition for synthetic data."""
    return WeightedExtractionDefinition(
        center_frac=0.5,
        radius_frac=0.45,
    )


# ===========================================================================
# 1. ExternalProfileExtractionResult: construction and properties
# ===========================================================================


class TestExternalProfileExtractionResult:
    """Tests for ExternalProfileExtractionResult construction."""

    def test_construction_minimal(self):
        """ExternalProfileExtractionResult can be constructed with a spectrum set."""
        ros = _make_rectified_order_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)
        assert isinstance(result, ExternalProfileExtractionResult)
        assert isinstance(result.spectra, WeightedExtractedSpectrumSet)

    def test_orders_passthrough(self):
        """orders property returns the same order list as the spectrum set."""
        ros = _make_rectified_order_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)
        assert result.orders == result.spectra.orders

    def test_n_orders_passthrough(self):
        """n_orders returns the correct count."""
        ros = _make_rectified_order_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)
        assert result.n_orders == _N_ORDERS


# ===========================================================================
# 2. Extraction correctness: synthetic tests
# ===========================================================================


class TestExtractionWithExternalProfile:
    """Correctness tests using synthetic rectified order sets."""

    def test_basic_single_order(self):
        """Extraction runs for a single order with a matching template."""
        ros = _make_rectified_order_set()
        # Use only order 311.
        single_ros = RectifiedOrderSet(
            mode=ros.mode,
            rectified_orders=[ros.rectified_orders[0]],
            source_image_shape=ros.source_image_shape,
        )
        templates = _build_templates(single_ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(single_ros, extraction_def, templates)

        assert result.n_orders == 1
        sp = result.spectra.spectra[0]
        assert sp.order == 311
        assert sp.flux.shape == (_N_SPECTRAL,)
        assert sp.variance.shape == (_N_SPECTRAL,)

    def test_multi_order_extraction(self):
        """Extraction handles all three orders correctly."""
        ros = _make_rectified_order_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)

        assert result.n_orders == _N_ORDERS
        for order_num in [311, 315, 320]:
            sp = result.spectra.get_order(order_num)
            assert sp.flux.shape == (_N_SPECTRAL,)
            assert sp.variance.shape == (_N_SPECTRAL,)

    def test_wavelength_preserved(self):
        """Wavelength axis is preserved from the rectified order."""
        ros = _make_rectified_order_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)

        for ro, sp in zip(ros.rectified_orders, result.spectra.spectra):
            np.testing.assert_array_equal(ro.wavelength_um, sp.wavelength_um)

    def test_profile_source_used_is_external(self):
        """profile_source_used field on each spectrum must be 'external'."""
        ros = _make_rectified_order_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)

        for sp in result.spectra.spectra:
            assert sp.profile_source_used == "external"

    def test_bookkeeping_all_applied(self):
        """All orders should have external_profile_applied=True when all templates present."""
        ros = _make_rectified_order_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)

        for order_num in ros.orders:
            assert result.external_profile_applied[order_num] is True
            assert result.template_n_frames_used[order_num] == 1

    def test_bookkeeping_n_frames_used(self):
        """template_n_frames_used reflects n_frames_used from the template."""
        rng = np.random.default_rng(42)
        ros1 = _make_rectified_order_set(rng=rng, noise_scale=0.1)
        ros2 = _make_rectified_order_set(rng=rng, noise_scale=0.1)
        templates = build_external_profile_template([ros1, ros2], ProfileTemplateDefinition())
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros1, extraction_def, templates)

        for order_num in ros1.orders:
            assert result.template_n_frames_used[order_num] == 2

    def test_results_differ_from_empirical_in_noisy_case(self):
        """External profile extraction should differ from empirical in a noisy scenario."""
        rng = np.random.default_rng(0)
        # Calibration frames: clean Gaussian profile.
        cal_ros = _make_gaussian_profile_set(rng=rng, noise_scale=0.0)
        templates = _build_templates(cal_ros)

        # Science frame: same profile but with significant noise.
        sci_ros = _make_gaussian_profile_set(rng=rng, noise_scale=5.0)
        extraction_def = _default_extraction_def()

        result_external = extract_with_external_profile(
            sci_ros, extraction_def, templates
        )
        # Empirical extraction on the noisy science frame.
        from pyspextool.instruments.ishell.weighted_optimal_extraction import (
            extract_weighted_optimal,
        )

        empirical_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.45,
            profile_source="empirical",
        )
        result_empirical = extract_weighted_optimal(sci_ros, empirical_def)

        # The two results should not be identical because the external profile
        # was built from noiseless data.
        for order_num in sci_ros.orders:
            ext_flux = result_external.spectra.get_order(order_num).flux
            emp_flux = result_empirical.get_order(order_num).flux
            # They can differ at individual columns; check not identical.
            assert not np.allclose(ext_flux, emp_flux, equal_nan=True), (
                f"Order {order_num}: external and empirical fluxes are "
                "unexpectedly identical in a noisy case."
            )

    def test_output_flux_finite_for_clean_data(self):
        """For clean data with a well-defined profile, flux should be finite."""
        ros = _make_gaussian_profile_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)

        for sp in result.spectra.spectra:
            # At least some spectral columns should be finite.
            assert np.any(np.isfinite(sp.flux)), (
                f"Order {sp.order}: no finite flux values."
            )

    def test_variance_positive_for_clean_data(self):
        """Variance entries should be positive (or NaN) for clean data."""
        ros = _make_gaussian_profile_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)

        for sp in result.spectra.spectra:
            finite_var = sp.variance[np.isfinite(sp.variance)]
            assert np.all(finite_var > 0), (
                f"Order {sp.order}: some finite variance values are ≤ 0."
            )


# ===========================================================================
# 3. Fallback behaviour
# ===========================================================================


class TestFallbackBehavior:
    """Tests for the fallback_profile_source parameter."""

    def _make_partial_template_set(
        self,
        ros: RectifiedOrderSet,
        include_orders: list[int],
    ) -> ExternalProfileTemplateSet:
        """Build a template set containing only a subset of orders."""
        partial_ros = RectifiedOrderSet(
            mode=ros.mode,
            rectified_orders=[
                ro for ro in ros.rectified_orders if ro.order in include_orders
            ],
            source_image_shape=ros.source_image_shape,
        )
        return _build_templates(partial_ros)

    def test_fallback_empirical(self):
        """Orders without a template use 'empirical' when fallback is set."""
        ros = _make_rectified_order_set()
        # Only include a template for the first order.
        partial_templates = self._make_partial_template_set(ros, [311])
        extraction_def = _default_extraction_def()

        result = extract_with_external_profile(
            ros, extraction_def, partial_templates,
            fallback_profile_source="empirical"
        )

        assert result.n_orders == _N_ORDERS
        assert result.external_profile_applied[311] is True
        assert result.external_profile_applied[315] is False
        assert result.external_profile_applied[320] is False
        assert result.template_n_frames_used[315] == 0
        assert result.template_n_frames_used[320] == 0

    def test_fallback_smoothed_empirical(self):
        """Orders without a template use 'smoothed_empirical' when fallback is set."""
        ros = _make_rectified_order_set()
        partial_templates = self._make_partial_template_set(ros, [311])
        extraction_def = _default_extraction_def()

        result = extract_with_external_profile(
            ros, extraction_def, partial_templates,
            fallback_profile_source="smoothed_empirical"
        )

        assert result.n_orders == _N_ORDERS
        assert result.external_profile_applied[311] is True

    def test_fallback_order_has_correct_profile_source(self):
        """Orders that fell back record the correct profile_source_used."""
        ros = _make_rectified_order_set()
        partial_templates = self._make_partial_template_set(ros, [311])
        extraction_def = _default_extraction_def()

        result = extract_with_external_profile(
            ros, extraction_def, partial_templates,
            fallback_profile_source="empirical"
        )

        # Order 311 used external template.
        assert result.spectra.get_order(311).profile_source_used == "external"
        # Orders 315 and 320 fell back to empirical.
        assert result.spectra.get_order(315).profile_source_used == "empirical"
        assert result.spectra.get_order(320).profile_source_used == "empirical"

    def test_all_fallback_produces_results(self):
        """Even an empty template set works if fallback is configured."""
        ros = _make_rectified_order_set()
        # Build empty template set (same mode, no templates).
        empty_templates = ExternalProfileTemplateSet(
            mode=ros.mode,
            templates=[],
        )
        extraction_def = _default_extraction_def()

        result = extract_with_external_profile(
            ros, extraction_def, empty_templates,
            fallback_profile_source="empirical"
        )

        assert result.n_orders == _N_ORDERS
        for order_num in ros.orders:
            assert result.external_profile_applied[order_num] is False


# ===========================================================================
# 4. Error cases
# ===========================================================================


class TestErrorCases:
    """Tests for ValueError and other error conditions."""

    def test_empty_rectified_order_set_raises(self):
        """Empty rectified_orders raises ValueError."""
        ros = _make_rectified_order_set()
        empty_ros = RectifiedOrderSet(
            mode=ros.mode,
            rectified_orders=[],
            source_image_shape=(200, 800),
        )
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()

        with pytest.raises(ValueError, match="empty"):
            extract_with_external_profile(empty_ros, extraction_def, templates)

    def test_mode_mismatch_raises(self):
        """Mode mismatch between rectified_orders and profile_templates raises ValueError."""
        ros = _make_rectified_order_set(mode="H1_test")
        templates = _build_templates(ros)
        # Build a different-mode set.
        other_ros = _make_rectified_order_set(mode="K1_test")

        extraction_def = _default_extraction_def()

        with pytest.raises(ValueError, match="[Mm]ode"):
            extract_with_external_profile(other_ros, extraction_def, templates)

    def test_missing_template_no_fallback_raises(self):
        """Missing template raises ValueError when no fallback is provided."""
        ros = _make_rectified_order_set()
        # Template set contains only one order.
        partial_ros = RectifiedOrderSet(
            mode=ros.mode,
            rectified_orders=[ros.rectified_orders[0]],
            source_image_shape=ros.source_image_shape,
        )
        partial_templates = _build_templates(partial_ros)
        extraction_def = _default_extraction_def()

        with pytest.raises(ValueError, match="[Nn]o template"):
            extract_with_external_profile(ros, extraction_def, partial_templates)

    def test_invalid_fallback_source_raises(self):
        """Invalid fallback_profile_source raises ValueError."""
        ros = _make_rectified_order_set()
        templates = _build_templates(ros)
        extraction_def = _default_extraction_def()

        with pytest.raises(ValueError, match="fallback_profile_source"):
            extract_with_external_profile(
                ros, extraction_def, templates,
                fallback_profile_source="psffit"  # invalid
            )

    def test_spatial_dimension_mismatch_raises(self):
        """Template with wrong spatial dimension raises ValueError."""
        ros = _make_rectified_order_set(n_spatial=_N_SPATIAL)
        # Build templates from a set with a different spatial dimension.
        other_ros = _make_rectified_order_set(n_spatial=_N_SPATIAL + 5)
        templates = _build_templates(other_ros)
        extraction_def = _default_extraction_def()

        with pytest.raises(ValueError, match="spatial dimension"):
            extract_with_external_profile(ros, extraction_def, templates)

    def test_spectral_dimension_mismatch_raises(self):
        """Template with wrong spectral dimension raises ValueError."""
        ros = _make_rectified_order_set(n_spectral=_N_SPECTRAL)
        other_ros = _make_rectified_order_set(n_spectral=_N_SPECTRAL + 4)
        templates = _build_templates(other_ros)
        extraction_def = _default_extraction_def()

        with pytest.raises(ValueError, match="spectral dimension"):
            extract_with_external_profile(ros, extraction_def, templates)


# ===========================================================================
# 5. Integration test: build templates (Stage 21) then apply (Stage 22)
# ===========================================================================


class TestIntegration:
    """Integration tests: Stage 21 → Stage 22 pipeline."""

    def test_stage21_to_stage22_pipeline(self):
        """Build templates with build_external_profile_template and apply here."""
        rng = np.random.default_rng(10)
        cal_ros = _make_gaussian_profile_set(rng=rng, noise_scale=0.1)
        template_def = ProfileTemplateDefinition(
            combine_method="median",
            normalize_profile=True,
            smooth_sigma=0.5,
        )
        templates = build_external_profile_template(cal_ros, template_def)

        sci_ros = _make_gaussian_profile_set(rng=rng, noise_scale=0.5)
        extraction_def = _default_extraction_def()

        result = extract_with_external_profile(sci_ros, extraction_def, templates)

        assert isinstance(result, ExternalProfileExtractionResult)
        assert result.n_orders == _N_ORDERS

        for order_num in sci_ros.orders:
            sp = result.spectra.get_order(order_num)
            assert sp.flux.shape == (_N_SPECTRAL,)
            assert sp.variance.shape == (_N_SPECTRAL,)
            assert sp.profile_source_used == "external"

    def test_multi_frame_template_then_apply(self):
        """Build multi-frame template then apply; verify n_frames_used."""
        rng = np.random.default_rng(99)
        frames = [
            _make_gaussian_profile_set(rng=rng, noise_scale=0.05)
            for _ in range(3)
        ]
        templates = build_external_profile_template(frames, ProfileTemplateDefinition())

        sci_ros = _make_gaussian_profile_set(rng=rng, noise_scale=0.5)
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(sci_ros, extraction_def, templates)

        for order_num in sci_ros.orders:
            assert result.template_n_frames_used[order_num] == 3

    def test_integration_result_shapes_valid(self):
        """All output shapes are well-formed after the full Stage 21→22 pipeline."""
        ros = _make_gaussian_profile_set()
        templates = build_external_profile_template(ros, ProfileTemplateDefinition())
        extraction_def = _default_extraction_def()
        result = extract_with_external_profile(ros, extraction_def, templates)

        assert result.n_orders == _N_ORDERS
        for sp in result.spectra.spectra:
            assert sp.flux.ndim == 1
            assert sp.variance.ndim == 1
            assert len(sp.flux) == len(sp.wavelength_um)
            assert len(sp.variance) == len(sp.wavelength_um)


# ===========================================================================
# 6. Smoke test: real H1 calibration data
# ===========================================================================


@pytest.mark.skipif(
    not _HAVE_H1_DATA,
    reason="Real H1 calibration FITS files not available (LFS not pulled).",
)
class TestSmoke:
    """Smoke tests using real iSHELL H1 calibration data.

    Demonstrates the realistic two-dataset workflow:

    * **Calibration dataset** — rectified orders built from the H1 flat frame,
      used to build the external profile template (Stage 21).
    * **Science dataset** — a lightly perturbed copy of the same detector
      image, simulating a separate science exposure.  Used for extraction
      (Stage 22).

    Using two distinct datasets ensures the test exercises the intended
    separation between template-building and extraction.  (In practice the
    templates would come from a high-SNR standard-star or flat exposure taken
    at a different time.)

    Chain:
      1. Flat-order tracing (Stage 1)
      2. Arc-line tracing (Stage 2)
      3. Provisional wavelength mapping (Stage 3)
      4. Coefficient-surface refinement (Stage 5)
      5. Rectification indices (Stage 6)
      6. Rectified orders — calibration image (Stage 7)
      7. Rectified orders — science image (perturbed copy, Stage 7)
      8. External profile template building from calibration set (Stage 21)
      9. External-profile extraction of science set (Stage 22, this module)
    """

    @pytest.fixture(scope="class")
    def h1_calibration_to_science_fixture(self):
        """Run the H1 chain and return (templates, cal_rectified, sci_rectified).

        ``cal_rectified`` is built from the first flat frame.
        ``sci_rectified`` is built from a small-noise-perturbed copy of the
        same detector image, acting as a stand-in for an independent science
        exposure.
        """
        import astropy.io.fits as fits

        from pyspextool.instruments.ishell.arc_tracing import trace_arc_lines
        from pyspextool.instruments.ishell.calibrations import (
            read_line_list,
            read_wavecalinfo,
        )
        from pyspextool.instruments.ishell.rectification_indices import (
            build_rectification_indices,
        )
        from pyspextool.instruments.ishell.rectified_orders import (
            build_rectified_orders,
        )
        from pyspextool.instruments.ishell.tracing import trace_orders_from_flat
        from pyspextool.instruments.ishell.wavecal_2d import (
            fit_provisional_wavelength_map,
        )
        from pyspextool.instruments.ishell.wavecal_2d_refine import (
            fit_refined_coefficient_surface,
        )

        # ------------------------------------------------------------------
        # Stages 1–6: shared calibration chain
        # ------------------------------------------------------------------

        # Stage 1
        flat_trace = trace_orders_from_flat(
            _H1_FLAT_FILES, col_range=(650, 1550)
        )
        geom = flat_trace.to_order_geometry_set("H1", col_range=(650, 1550))

        # Stage 2
        arc_result = trace_arc_lines(_H1_ARC_FILES, geom)

        # Stage 3
        wci = read_wavecalinfo("H1")
        ll = read_line_list("H1")
        wav_map = fit_provisional_wavelength_map(
            arc_result, wci, ll, dispersion_degree=3
        )

        # Stage 5
        surface = fit_refined_coefficient_surface(wav_map, disp_degree=3)

        # Stage 6
        rect_indices = build_rectification_indices(geom, surface, wav_map=wav_map)

        # ------------------------------------------------------------------
        # Stage 7 (calibration): rectified orders from the first flat frame
        # ------------------------------------------------------------------
        with fits.open(_H1_FLAT_FILES[0]) as hdul:
            cal_image = hdul[0].data.astype(float)
        cal_rectified = build_rectified_orders(cal_image, rect_indices)

        # ------------------------------------------------------------------
        # Stage 7 (science): rectified orders from a lightly perturbed copy
        # of the same detector image, simulating a separate science exposure.
        # A 1 % Gaussian noise perturbation is sufficient to make the two
        # datasets distinct while keeping flux levels realistic.
        # ------------------------------------------------------------------
        rng = np.random.default_rng(seed=12345)
        sci_image = cal_image + rng.normal(
            scale=0.01 * np.nanmedian(np.abs(cal_image)),
            size=cal_image.shape,
        )
        sci_rectified = build_rectified_orders(sci_image, rect_indices)

        # ------------------------------------------------------------------
        # Stage 21: build external profile templates from the calibration set
        # ------------------------------------------------------------------
        template_def = ProfileTemplateDefinition(
            combine_method="median",
            normalize_profile=True,
            smooth_sigma=0.0,
        )
        templates = build_external_profile_template(cal_rectified, template_def)

        return templates, cal_rectified, sci_rectified

    def test_h1_calibration_to_science_workflow(self, h1_calibration_to_science_fixture):
        """Stage 22 extraction of a science frame using calibration-frame templates."""
        templates, _cal, sci_rectified = h1_calibration_to_science_fixture
        extraction_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.45,
        )
        result = extract_with_external_profile(sci_rectified, extraction_def, templates)
        assert isinstance(result, ExternalProfileExtractionResult)

    def test_external_profile_applied_all_orders(self, h1_calibration_to_science_fixture):
        """external_profile_applied is True for every science order."""
        templates, _cal, sci_rectified = h1_calibration_to_science_fixture
        extraction_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.45,
        )
        result = extract_with_external_profile(sci_rectified, extraction_def, templates)
        for order_num in sci_rectified.orders:
            assert result.external_profile_applied[order_num] is True, (
                f"Order {order_num}: expected external_profile_applied=True."
            )

    def test_correct_n_orders(self, h1_calibration_to_science_fixture):
        """Number of output spectra matches number of science rectified orders."""
        templates, _cal, sci_rectified = h1_calibration_to_science_fixture
        extraction_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.45,
        )
        result = extract_with_external_profile(sci_rectified, extraction_def, templates)
        assert result.n_orders == sci_rectified.n_orders

    def test_correct_shapes(self, h1_calibration_to_science_fixture):
        """Each output spectrum has the expected shape."""
        templates, _cal, sci_rectified = h1_calibration_to_science_fixture
        extraction_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.45,
        )
        result = extract_with_external_profile(sci_rectified, extraction_def, templates)
        for sp in result.spectra.spectra:
            n_spec = sci_rectified.get_order(sp.order).n_spectral
            assert sp.flux.shape == (n_spec,)
            assert sp.variance.shape == (n_spec,)

    def test_reasonable_flux(self, h1_calibration_to_science_fixture):
        """At least some spectral channels have finite flux per order."""
        templates, _cal, sci_rectified = h1_calibration_to_science_fixture
        extraction_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.45,
        )
        result = extract_with_external_profile(sci_rectified, extraction_def, templates)
        for sp in result.spectra.spectra:
            assert np.any(np.isfinite(sp.flux)), (
                f"Order {sp.order}: all flux values are NaN."
            )

    def test_variance_positive(self, h1_calibration_to_science_fixture):
        """Finite variance values are positive per order."""
        templates, _cal, sci_rectified = h1_calibration_to_science_fixture
        extraction_def = WeightedExtractionDefinition(
            center_frac=0.5,
            radius_frac=0.45,
        )
        result = extract_with_external_profile(sci_rectified, extraction_def, templates)
        for sp in result.spectra.spectra:
            finite_var = sp.variance[np.isfinite(sp.variance)]
            assert np.all(finite_var > 0), (
                f"Order {sp.order}: non-positive finite variance."
            )
