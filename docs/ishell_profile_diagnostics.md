# iSHELL Profile Diagnostics — Stage 23 Scaffold

> **Important: These diagnostics are not sufficient for science validation.**

This document describes **Stage 23** of the iSHELL 2DXD reduction scaffold:
a diagnostic-only layer for evaluating external spatial-profile template
quality, comparing external vs empirical extraction behaviour, and detecting
potential template leakage.

---

## Overview

Stage 21 (`profile_templates.py`) can build profile templates from calibration
frames, and Stage 22 (`external_profile_extraction.py`) applies those templates
to science frames.  However, the pipeline previously had no tools to:

- evaluate template quality
- compare external vs empirical extraction
- detect when a template may have been built from the same data

Stage 23 (`profile_diagnostics.py`) fills that gap with a **diagnostic-only**
layer.  It does **not** change any extraction behaviour, reject templates, or
make pipeline decisions.

---

## What each metric represents

### 1. Template quality (`ProfileDiagnostics`)

Per-order metrics computed directly from an `ExternalProfileTemplate`.

| Metric | Description |
|---|---|
| `finite_fraction` | Fraction of profile pixels that are finite (range [0, 1]). Near 1.0 is good. Near 0.0 means severe masking or NaN contamination. |
| `peak_spatial_index` | Row index of the spatial maximum in the collapsed (per-column median) profile.  `-1` if no finite values exist. |
| `peak_spatial_frac` | Fractional slit position of the peak.  Expect near the true target position.  `NaN` if no finite values. |
| `colsum_min` | Minimum column sum across all finite spectral columns.  For a normalized template this should be near 1.0. |
| `colsum_max` | Maximum column sum.  For a normalized template near 1.0; deviation indicates scale variation. |
| `colsum_median` | Median column sum.  Most useful summary for normalization checks. |
| `roughness` | Mean absolute difference between adjacent finite spatial-axis values in the collapsed 1-D profile.  Lower is smoother.  `NaN` with fewer than 2 finite values. |
| `n_frames_used` | Number of calibration frames used to build this template.  More frames generally means a more stable profile. |
| `is_normalized_like` | `True` if all finite column sums are within `1e-2` of 1.0.  Indicates that the profile was column-normalized. |

**Example: good template**

```
order=311  finite_fraction=1.00  peak_spatial_frac=0.50
colsum_median=1.00  roughness=0.002  n_frames_used=3  is_normalized_like=True
```

**Example: noisy template**

```
order=311  finite_fraction=0.95  peak_spatial_frac=0.48
colsum_median=1.01  roughness=0.08  n_frames_used=1  is_normalized_like=True
```

High roughness and `n_frames_used=1` suggest this template would benefit from
more calibration frames.

**Example: suspicious (leakage-like) template**

```
order=311  finite_fraction=1.00  peak_spatial_frac=0.50
colsum_median=1.00  roughness=0.001  n_frames_used=1  is_normalized_like=True
```

Extremely low roughness combined with a perfect profile match may warrant
checking whether the template was built from the science data.

---

### 2. External vs empirical comparison (`ExternalVsEmpiricalDiagnostics`)

Per-order comparison between extraction with an external profile template and
extraction with the empirical profile derived from the science frame.

| Metric | Description |
|---|---|
| `flux_l2_difference` | Normalised L2 (RMS) difference between external and empirical flux vectors.  Near 0 means both methods agree well. |
| `median_abs_flux_difference` | Median absolute difference between finite pairs of the two flux vectors.  Robust to outliers. |
| `finite_fraction_flux` | Fraction of wavelength bins where the external extraction produced a finite flux value. |
| `finite_fraction_variance` | Fraction of wavelength bins where the external extraction produced a finite variance estimate. |
| `external_profile_applied` | `True` if the external template was used (vs fallback). |
| `template_n_frames_used` | Number of calibration frames in the template (`0` for fallback orders). |

Large `flux_l2_difference` values may indicate:
- the template is miscentred (star moved on slit between calibration and science)
- the template is from a different observing mode
- the science frame has substantially different spatial structure than the template

---

### 3. Template leakage detection (`TemplateLeakageDiagnostics`)

Heuristic metrics that quantify the similarity between the empirical profile
of the science frame and the external template profile.

| Metric | Description |
|---|---|
| `profile_correlation` | Pearson correlation between science empirical and external template profiles (aperture rows only).  Near 1.0 means nearly identical shapes. |
| `profile_l2_difference` | Shape-normalised L2 difference after unit-norm normalisation of both profiles.  Near 0.0 means identical shapes. |
| `flux_image_correlation` | Pearson correlation between the science flux image (flattened) and the template-derived model image. |
| `possible_template_leakage` | Heuristic flag: `True` when `profile_correlation > 0.98` **AND** `profile_l2_difference < 0.05`. |

The `possible_template_leakage` flag is set using two simple thresholds:

- `LEAKAGE_CORRELATION_THRESHOLD = 0.98`
- `LEAKAGE_L2_THRESHOLD = 0.05`

These are documented constants in `profile_diagnostics.py`.  They are *not*
tuned for science use and must not be used for automated decisions.

---

## How to interpret leakage flags cautiously

**A positive `possible_template_leakage` flag does not prove leakage.**

High profile correlation can occur when:
- The target genuinely fills the slit in a consistent manner across observations.
- The calibration frame happens to have a similar profile (e.g. a standard star
  with similar spatial extent as the science target).
- The noise level is low enough that both profiles are very smooth Gaussians.

**A negative flag does not certify template independence.**

If the template and science data are nearly identical but differ by a small
noise perturbation, the correlation may still be very high.

**Always verify template provenance separately.**  The metadata field
`n_frames_used` and the observation log should be consulted to confirm that
the template was built from calibration data independent of the science target.

---

## Thresholds are heuristic only

The leakage thresholds (`0.98` and `0.05`) are:

- Simple constants that are easy to understand.
- Not calibrated against real iSHELL data.
- Not validated for any particular science programme.
- Subject to revision in a future non-scaffold stage.

Do **not** rely on these thresholds to automatically accept or reject templates.
They are for human inspection only.

---

## Usage

```python
from pyspextool.instruments.ishell.profile_diagnostics import (
    compute_profile_diagnostics,
    compare_external_vs_empirical,
    compute_leakage_diagnostics,
    run_full_profile_diagnostics,
)
from pyspextool.instruments.ishell.weighted_optimal_extraction import (
    WeightedExtractionDefinition,
)

extraction_def = WeightedExtractionDefinition(
    center_frac=0.5,
    radius_frac=0.4,
)

# 1. Template quality metrics
tmpl_diags = compute_profile_diagnostics(templates)
for d in tmpl_diags.diagnostics:
    print(f"Order {d.order}: finite_fraction={d.finite_fraction:.3f}, "
          f"roughness={d.roughness:.4f}, "
          f"is_normalized_like={d.is_normalized_like}")

# 2. External vs empirical comparison
cmp_diags = compare_external_vs_empirical(
    science_rectified, extraction_def, templates
)
for d in cmp_diags.diagnostics:
    print(f"Order {d.order}: l2_diff={d.flux_l2_difference:.4f}")

# 3. Leakage heuristics
leak_diags = compute_leakage_diagnostics(
    science_rectified, templates, extraction_def
)
for d in leak_diags.diagnostics:
    print(f"Order {d.order}: corr={d.profile_correlation:.4f}, "
          f"l2={d.profile_l2_difference:.4f}, "
          f"leakage={d.possible_template_leakage}")

# 4. All at once
result = run_full_profile_diagnostics(
    science_rectified, extraction_def, templates
)
print(result.template_diagnostics)
print(result.comparison_diagnostics)
print(result.leakage_diagnostics)
```

---

## Files

| File | Purpose |
|---|---|
| `src/pyspextool/instruments/ishell/profile_diagnostics.py` | Stage 23 module |
| `tests/test_ishell_profile_diagnostics.py` | Tests for Stage 23 |
| `docs/ishell_profile_diagnostics.md` | This document |
| `docs/ishell_profile_templates.md` | Stage 21 documentation |
| `docs/ishell_external_profile_extraction.md` | Stage 22 documentation |

---

## What remains missing for science-quality validation

This scaffold implements diagnostic *numbers* but the following are explicitly
**out of scope**:

- **Automatic template rejection** — no template is ever rejected on the basis
  of these metrics.
- **Automatic fallback switching** — the pipeline does not switch to empirical
  extraction when leakage is detected.
- **Science-quality thresholds** — the leakage thresholds are heuristic
  constants, not calibrated to any science programme.
- **Centroid alignment checks** — no check for spatial offset between template
  centroid and science-frame centroid.
- **Template staleness detection** — no date/time comparison between the
  template and the science frame.
- **PSF fitting** — no parametric PSF model; all profiles are nanmedian
  estimates.
- **Wavelength-dependent profile checks** — the template is spectrally
  constant; no per-column profile variation is checked.
- **Cross-order consistency checks** — each order is diagnosed independently.

These limitations will need to be addressed in a future stage for the pipeline
to be suitable for science-quality extraction.

> **These diagnostics are not sufficient for science validation.**
