# iSHELL Profile Warnings — Stage 24 / 25 Scaffold

> **Warnings are informational and do not imply automatic failure.**

This document describes **Stage 24** and **Stage 25** of the iSHELL 2DXD
reduction scaffold.  Stage 24 introduced a diagnostic reporting and warning
layer; Stage 25 extends it with user-configurable warning policies.

---

## Overview

Stage 23 produces numeric diagnostic metrics (finite fractions, roughness
values, correlation coefficients, etc.) but provides no standard way to
interpret or report them.  Stage 24 fills that gap with a lightweight
warning layer.

This layer:

- converts diagnostic metrics into named warning codes with clear messages
- provides structured programmatic access to issues via `ProfileWarningSet`
- produces human-readable text output via `format_profile_warnings()`
- integrates with Stage 23 via a single entry point `run_diagnostics_with_warnings()`

**Important constraints:**

- Does **not** change extraction behaviour.
- Does **not** automatically reject templates.
- Does **not** automatically switch profile sources.
- Does **not** introduce any pipeline decision logic.

This stage is about **surfacing information**, not acting on it.

---

## Warning Codes

### Template quality warnings

These are generated from `ProfileDiagnosticsSet` (Stage 23 template metrics).

#### `LOW_FINITE_FRACTION`

| Field | Value |
|---|---|
| Severity | `warning` |
| Threshold | `finite_fraction < 0.9` |
| Metric | `ProfileDiagnostics.finite_fraction` |

**Physical meaning:**  
A significant fraction of pixels in the profile array are non-finite (NaN or
Inf).  This can result from bad-pixel masking, edge effects, or NaN
contamination during template construction.

**How to interpret:**  
Values well below 0.9 may indicate that the template covers only part of the
slit or that a large number of calibration pixels were masked.  Check the
calibration frame quality and bad-pixel mask.

**Does not mean:** The template is definitely unusable.  Some NaN contamination
is acceptable, especially near slit edges.

---

#### `NOT_NORMALIZED`

| Field | Value |
|---|---|
| Severity | `warning` |
| Threshold | `is_normalized_like is False` |
| Metric | `ProfileDiagnostics.colsum_median` |

**Physical meaning:**  
The profile template is not column-normalized (i.e. the sum of profile values
along the spatial axis is not close to 1.0 for each wavelength column).
Optimal extraction algorithms typically assume a normalized spatial profile.

**How to interpret:**  
If the template was built without the `normalize_profile=True` option in Stage
21, this warning is expected.  If normalization was requested and this warning
fires, there may be an issue with the template construction.

**Does not mean:** The extraction will fail.  Some extraction configurations
handle non-normalized profiles, but the user should verify this is intentional.

---

#### `HIGH_ROUGHNESS`

| Field | Value |
|---|---|
| Severity | `info` |
| Threshold | `roughness > 0.05` |
| Metric | `ProfileDiagnostics.roughness` |

**Physical meaning:**  
The collapsed 1-D spatial profile has high spatial-frequency noise.  Roughness
is the mean absolute difference between adjacent spatial-axis values in the
per-column-median profile.

**How to interpret:**  
High roughness often results from a template built from a single calibration
frame with significant read noise.  Using more frames (higher `n_frames_used`)
typically reduces roughness through averaging.  Smoothing (`smooth_sigma > 0`)
in Stage 21 also reduces roughness.

**Does not mean:** The extraction is inaccurate.  Mild roughness may not
significantly affect flux estimates, though it can increase variance.

---

### External vs empirical warnings

These are generated from `ExternalVsEmpiricalDiagnosticsSet` (Stage 23
comparison metrics), and are only produced when a comparison diagnostic set
is provided.

#### `LARGE_FLUX_DIFFERENCE`

| Field | Value |
|---|---|
| Severity | `warning` |
| Threshold | `flux_l2_difference > 0.2` |
| Metric | `ExternalVsEmpiricalDiagnostics.flux_l2_difference` |

**Physical meaning:**  
The external-profile extraction and the empirical-profile extraction produce
substantially different flux vectors for the same order.  The metric is the
normalised L2 (RMS) difference between the two flux arrays.

**How to interpret:**  
Large differences may indicate:

- The template is spatially offset from the science target (e.g. telescope
  moved between calibration and science).
- The template was built from a different mode or instrument configuration.
- The science target has a very different spatial profile than the calibration
  source.

A large `flux_l2_difference` is a signal to investigate template provenance,
not to automatically reject the extraction.

---

#### `LOW_FINITE_FLUX`

| Field | Value |
|---|---|
| Severity | `warning` |
| Threshold | `finite_fraction_flux < 0.8` |
| Metric | `ExternalVsEmpiricalDiagnostics.finite_fraction_flux` |

**Physical meaning:**  
A large fraction of wavelength bins in the external extraction produced
non-finite flux values.  This suggests that the extraction failed for many
columns.

**How to interpret:**  
This can occur when the template profile does not overlap with the aperture
definition, when the template contains many NaNs, or when the science frame
itself has extensive bad pixels.  Check the aperture definition and template
coverage.

---

### Leakage warnings

These are generated from `TemplateLeakageDiagnosticsSet` (Stage 23 leakage
metrics), and are only produced when a leakage diagnostic set is provided.

#### `POSSIBLE_TEMPLATE_LEAKAGE`

| Field | Value |
|---|---|
| Severity | `severe` |
| Threshold | `profile_correlation > policy.leakage_corr_min AND profile_l2_difference < policy.leakage_l2_max` |
| Metric | `TemplateLeakageDiagnostics.profile_correlation` / `profile_l2_difference` |

**Physical meaning:**  
The external template profile is suspiciously similar to the empirical profile
derived from the science frame itself.  This is a heuristic indicator that the
template may have been built from the same data that is being extracted
("leakage"), which would inflate the signal-to-noise ratio of the extracted
spectrum.

The warning fires when **both** conditions are satisfied simultaneously:

```
profile_correlation > policy.leakage_corr_min  (default 0.98)
profile_l2_difference < policy.leakage_l2_max  (default 0.05)
```

Both thresholds are configurable via `ProfileWarningPolicy` (Stage 25).

**How to interpret:**  
Verify that the template was built from calibration data independent of the
science frame.  Check:

- The observation log to confirm that the template and science frames are
  from different exposures.
- The `n_frames_used` field to confirm that multiple frames were combined.
- The template construction provenance (Stage 21 inputs).

**Important caveats:**

- A positive flag does **not** prove leakage.  High correlation can occur
  naturally when both profiles are smooth, centred Gaussians with low noise.
- A negative flag does **not** certify independence.  Slightly perturbed
  copies of the same data may not trigger the flag.

This warning is labelled `severe` because the potential science impact (biased
flux estimates) is significant if leakage is real.  However, "severe" is an
informational label — it does **not** trigger any automatic action.

---

## Severity Levels

| Level | Meaning |
|---|---|
| `info` | Mild diagnostic indication; may not affect science quality |
| `warning` | Potentially significant issue worth investigating |
| `severe` | High-impact issue if real; warrants careful manual verification |

**All levels are informational.** Severity does not cause automatic rejection
or fallback switching.

---

## Thresholds

All thresholds are module-level constants in `profile_warnings.py`:

```python
THRESHOLD_LOW_FINITE_FRACTION = 0.9
THRESHOLD_HIGH_ROUGHNESS      = 0.05
THRESHOLD_LARGE_FLUX_DIFFERENCE = 0.2
THRESHOLD_LOW_FINITE_FLUX     = 0.8
```

These thresholds are:

- Simple scalar constants that are easy to understand and inspect.
- Not calibrated against real iSHELL science data.
- Not validated for any particular observing programme.
- Subject to revision in a future non-scaffold stage.

Do **not** rely on these thresholds for autonomous science-quality decisions.

---

## Usage

### Recommended entry point

```python
from pyspextool.instruments.ishell.profile_warnings import (
    run_diagnostics_with_warnings,
    format_profile_warnings,
)
from pyspextool.instruments.ishell.weighted_optimal_extraction import (
    WeightedExtractionDefinition,
)

extraction_def = WeightedExtractionDefinition(center_frac=0.5, radius_frac=0.4)

# Run all diagnostics and generate warnings in one call
result = run_diagnostics_with_warnings(
    science_rectified_orders,
    extraction_def,
    profile_templates,
)

# Print human-readable report
print(format_profile_warnings(result.warnings))

# Programmatic access
if result.warnings.has_severe():
    print("Severe warnings present — please review manually.")

for order in result.warnings.orders:
    order_warnings = result.warnings.get_order(order)
    for w in order_warnings:
        print(f"Order {order}: [{w.severity.upper()}] {w.code}: {w.message}")
```

### Manual warning generation

```python
from pyspextool.instruments.ishell.profile_warnings import (
    generate_profile_warnings,
    format_profile_warnings,
)
from pyspextool.instruments.ishell.profile_diagnostics import (
    compute_profile_diagnostics,
    compare_external_vs_empirical,
    compute_leakage_diagnostics,
)

# Compute diagnostics manually
tmpl_diags = compute_profile_diagnostics(templates)
cmp_diags  = compare_external_vs_empirical(science_ros, extraction_def, templates)
leak_diags = compute_leakage_diagnostics(science_ros, templates, extraction_def)

# Generate warnings
ws = generate_profile_warnings(
    tmpl_diags,
    comparison_diag_set=cmp_diags,
    leakage_diag_set=leak_diags,
)

# Filter by severity
severe_ws = ws.filter_by_severity("severe")
print(f"Severe warnings: {len(severe_ws)}")

# Format output
text = format_profile_warnings(ws)
print(text)
```

### Example output

```
Order 311:
  [SEVERE] POSSIBLE_TEMPLATE_LEAKAGE: Template profile is suspiciously similar to the science frame profile (correlation=0.9953, l2_diff=0.0023). Verify that the template was not built from the science data.
  [WARNING] NOT_NORMALIZED: Template column sums deviate from 1 (colsum_median=1.1200). Profile may not be column-normalized.

Order 315:
  [INFO] HIGH_ROUGHNESS: Template spatial profile is rough (roughness=0.0623, threshold=0.05). Consider using more calibration frames.
```

---

## API Reference

### `ProfileWarning`

A single warning for one order.

| Field | Type | Description |
|---|---|---|
| `order` | `int` | Echelle order number |
| `code` | `str` | Machine-readable warning code |
| `severity` | `str` | `"info"`, `"warning"`, or `"severe"` |
| `message` | `str` | Human-readable explanation |
| `metric_value` | `float` or `None` | Numeric metric that triggered the warning |
| `threshold` | `float` or `None` | Threshold compared against `metric_value` |
| `context` | `dict` | Additional diagnostic context |

### `ProfileWarningSet`

| Method / Property | Description |
|---|---|
| `warnings` | All warnings (list copy) |
| `orders` | Sorted list of distinct order numbers |
| `n_orders` | Number of distinct orders with warnings |
| `get_order(order)` | Warnings for a specific order (empty list if none) |
| `filter_by_severity(severity)` | New set with only that severity level |
| `has_severe()` | `True` if any warning has `severity="severe"` |
| `__len__()` | Total number of warnings |

### `DiagnosticsWithWarnings`

Result of `run_diagnostics_with_warnings()`.

| Field | Type | Description |
|---|---|---|
| `profile_diagnostics` | `ProfileDiagnosticsSet` | Stage 23 template metrics |
| `comparison_diagnostics` | `ExternalVsEmpiricalDiagnosticsSet` | Stage 23 comparison metrics |
| `leakage_diagnostics` | `TemplateLeakageDiagnosticsSet` | Stage 23 leakage metrics |
| `warnings` | `ProfileWarningSet` | Stage 24 generated warnings |

---

## Files

| File | Purpose |
|---|---|
| `src/pyspextool/instruments/ishell/profile_warnings.py` | Stage 24 module |
| `tests/test_ishell_profile_warnings.py` | Tests for Stage 24 |
| `docs/ishell_profile_warnings.md` | This document |
| `docs/ishell_profile_diagnostics.md` | Stage 23 documentation |
| `docs/ishell_profile_templates.md` | Stage 21 documentation |
| `docs/ishell_external_profile_extraction.md` | Stage 22 documentation |

---

## What remains missing for science-quality automation

This scaffold provides human-readable reporting but explicitly omits the
following (out of scope for Stage 24):

- **Automatic template rejection** — no template is ever rejected based
  on warning codes.
- **Automatic fallback switching** — the pipeline does not switch to
  empirical extraction when warnings are present.
- **Science-quality threshold calibration** — all thresholds are simple
  heuristic constants, not calibrated to any iSHELL science programme.
- **Cross-order consistency checks** — each order is warned independently;
  no cross-order patterns are detected.
- **Temporal staleness detection** — no check for time elapsed between
  template construction and science observation.
- **Centroid alignment warnings** — no check for spatial offset between
  the template centroid and the science-frame centroid.
- **PSF-based quality metrics** — all profile quality is assessed via
  simple scalar statistics, not parametric PSF fitting.

These limitations will need to be addressed in a future stage for the
pipeline to be suitable for science-quality autonomous decisions.

> **Warnings are informational and do not imply automatic failure.**

---

## Configurable Warning Policies (Stage 25)

Stage 25 introduces `ProfileWarningPolicy`, a dataclass that allows users to
tune warning thresholds and enable/disable individual warning types without
changing any extraction behaviour.

> **Changing thresholds does not make the results science-quality.**
> All thresholds remain heuristic constants for human inspection only.

### Policy fields

| Field | Type | Default | Description |
|---|---|---|---|
| `finite_fraction_min` | `float` | `0.9` | Minimum finite-pixel fraction; `LOW_FINITE_FRACTION` fires below this |
| `roughness_max` | `float` | `0.05` | Maximum profile roughness; `HIGH_ROUGHNESS` fires above this |
| `flux_l2_diff_max` | `float` | `0.2` | Maximum normalised L2 flux difference; `LARGE_FLUX_DIFFERENCE` fires above this |
| `finite_flux_min` | `float` | `0.8` | Minimum finite-flux fraction; `LOW_FINITE_FLUX` fires below this |
| `leakage_corr_min` | `float` | `0.98` | `POSSIBLE_TEMPLATE_LEAKAGE` fires when `profile_correlation > leakage_corr_min` **and** `profile_l2_difference < leakage_l2_max` |
| `leakage_l2_max` | `float` | `0.05` | `POSSIBLE_TEMPLATE_LEAKAGE` fires when `profile_correlation > leakage_corr_min` **and** `profile_l2_difference < leakage_l2_max` |
| `enable_low_finite_fraction` | `bool` | `True` | Set `False` to suppress `LOW_FINITE_FRACTION` entirely |
| `enable_not_normalized` | `bool` | `True` | Set `False` to suppress `NOT_NORMALIZED` entirely |
| `enable_high_roughness` | `bool` | `True` | Set `False` to suppress `HIGH_ROUGHNESS` entirely |
| `enable_large_flux_difference` | `bool` | `True` | Set `False` to suppress `LARGE_FLUX_DIFFERENCE` entirely |
| `enable_low_finite_flux` | `bool` | `True` | Set `False` to suppress `LOW_FINITE_FLUX` entirely |
| `enable_leakage_warning` | `bool` | `True` | Set `False` to suppress `POSSIBLE_TEMPLATE_LEAKAGE` entirely |

**Validation rules:**

- All threshold fields must be finite numbers.
- Fraction fields (`finite_fraction_min`, `finite_flux_min`,
  `leakage_corr_min`) must be in `[0, 1]`.
- Non-fraction, non-negative fields (`roughness_max`, `flux_l2_diff_max`,
  `leakage_l2_max`) must be `>= 0`.

### How thresholds affect warnings

Each numeric threshold is a simple scalar boundary:

- **Lower** `finite_fraction_min` or `finite_flux_min` → fewer
  `LOW_FINITE_FRACTION` / `LOW_FINITE_FLUX` warnings (higher tolerance for
  NaN contamination).
- **Higher** `finite_fraction_min` or `finite_flux_min` → more warnings
  (stricter).
- **Higher** `roughness_max` or `flux_l2_diff_max` → fewer `HIGH_ROUGHNESS` /
  `LARGE_FLUX_DIFFERENCE` warnings (higher tolerance for spatial noise /
  extraction disagreement).
- **Lower** `roughness_max` or `flux_l2_diff_max` → more warnings (stricter).
- **Lower** `leakage_corr_min` → `POSSIBLE_TEMPLATE_LEAKAGE` fires at lower
  profile correlations (more sensitive to potential leakage).
- **Higher** `leakage_corr_min` → warning only fires at very high correlations
  (fewer false positives for smooth, similar profiles).
- **Higher** `leakage_l2_max` → warning fires even when profiles differ more
  in L2 norm (more sensitive).
- **Lower** `leakage_l2_max` → warning only fires when profiles are nearly
  identical in L2 norm (fewer false positives).

The `enable_*` flags take priority: when `False`, the corresponding warning
code is never emitted regardless of the threshold value.

### Example configurations

```python
from pyspextool.instruments.ishell.profile_warnings import (
    ProfileWarningPolicy,
    generate_profile_warnings,
    format_profile_warnings,
    run_diagnostics_with_warnings,
)

# Default policy — reproduces Stage 24 behaviour exactly
result = run_diagnostics_with_warnings(
    science_ros, extraction_def, profile_templates
)
print(format_profile_warnings(result.warnings))

# Strict policy — more sensitive, surfaces borderline issues
strict_policy = ProfileWarningPolicy.strict()
result = run_diagnostics_with_warnings(
    science_ros, extraction_def, profile_templates, policy=strict_policy
)

# Conservative policy — fewer warnings, focus on clear problems
conservative_policy = ProfileWarningPolicy.conservative()
result = run_diagnostics_with_warnings(
    science_ros, extraction_def, profile_templates, policy=conservative_policy
)

# Custom policy — raise finite-fraction sensitivity only
custom_policy = ProfileWarningPolicy(finite_fraction_min=0.95)
ws = generate_profile_warnings(tmpl_diags, policy=custom_policy)

# Disable noisy warning types
quiet_policy = ProfileWarningPolicy(
    enable_high_roughness=False,
    enable_not_normalized=False,
)
ws = generate_profile_warnings(tmpl_diags, policy=quiet_policy)

# Filter formatted output to severe warnings only
text = format_profile_warnings(ws, include_info=False, include_warning=False)

# Flat (non-grouped) output
text = format_profile_warnings(ws, group_by_order=False)
```

### Preset policies

Two convenience presets are provided:

- `ProfileWarningPolicy.strict()` — lower thresholds, surfaces more warnings.
  Useful during initial pipeline validation.
- `ProfileWarningPolicy.conservative()` — higher thresholds, fewer warnings.
  Useful when focusing on clear problems only.

Neither preset changes extraction behaviour.  They are convenience
constructors that return `ProfileWarningPolicy` instances.

### Formatting options

`format_profile_warnings()` accepts four additional keyword arguments:

| Parameter | Default | Effect |
|---|---|---|
| `include_info` | `True` | When `False`, omit `info`-level lines |
| `include_warning` | `True` | When `False`, omit `warning`-level lines |
| `include_severe` | `True` | When `False`, omit `severe`-level lines |
| `group_by_order` | `True` | When `False`, produce flat output without `Order N:` headers |

Default values preserve Stage 24 output exactly.

### Stage 25 constraints

Stage 25 does **not** implement:

- automatic fallback switching based on policy
- template rejection based on policy
- pipeline decision logic of any kind
- adaptive thresholds derived from data

All behaviour remains explicit, user-controlled, and transparent.
