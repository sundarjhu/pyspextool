# Stage 22: External Profile Extraction Workflow

This document describes **Stage 22** of the iSHELL 2DXD reduction scaffold:
a clean, reusable interface for applying external spatial-profile templates
to weighted optimal extraction across all echelle orders.

## Purpose

Stage 20 introduced `profile_source="external"` support inside
`extract_weighted_optimal()`.  Stage 21 introduced
`build_external_profile_template()` which constructs an
`ExternalProfileTemplateSet` from one or more calibration frames.

However, there was no workflow that connected the two.  Users had to:

1. Call `build_external_profile_template()` manually.
2. Iterate over orders, extract each template, and call
   `extract_weighted_optimal()` per order with the right `external_profile`.

Stage 22 provides that workflow via a single function:

```python
extract_with_external_profile(
    rectified_orders,
    extraction_def,
    profile_templates,
)
```

---

## API Reference

### `extract_with_external_profile()`

```python
from pyspextool.instruments.ishell.external_profile_extraction import (
    extract_with_external_profile,
)

result = extract_with_external_profile(
    rectified_orders,        # RectifiedOrderSet  (Stage 7)
    extraction_def,          # WeightedExtractionDefinition  (Stage 17–20)
    profile_templates,       # ExternalProfileTemplateSet  (Stage 21)
    *,
    variance_image=None,     # optional per-pixel variance array
    variance_model=None,     # optional VarianceModelDefinition  (Stage 14)
    mask=None,               # optional bad-pixel mask  (True = bad)
    fallback_profile_source=None,  # "empirical" | "smoothed_empirical" | None
)
```

**Returns:** `ExternalProfileExtractionResult`

---

### `ExternalProfileExtractionResult`

```python
@dataclass
class ExternalProfileExtractionResult:
    spectra: WeightedExtractedSpectrumSet          # one spectrum per order
    external_profile_applied: dict[int, bool]      # order → True if template used
    template_n_frames_used: dict[int, int]         # order → n_frames_used (0 if fallback)
```

Convenience properties forward to the wrapped `WeightedExtractedSpectrumSet`:

- `.orders` — list of order numbers
- `.n_orders` — number of orders

---

## Example Workflow

### Step 1: Build a profile template (Stage 21)

```python
import numpy as np
from pyspextool.instruments.ishell.profile_templates import (
    ProfileTemplateDefinition,
    build_external_profile_template,
)

# calibration_rectified is a RectifiedOrderSet built from flat / standard frames
template_def = ProfileTemplateDefinition(
    combine_method="median",   # robust combination across frames
    normalize_profile=True,    # columns sum to 1.0
    smooth_sigma=0.5,          # optional Gaussian smoothing (spatial pixels)
)

templates = build_external_profile_template(
    calibration_rectified,     # one or more RectifiedOrderSet objects
    template_def,
)
```

### Step 2: Apply the template to science frames (Stage 22)

```python
from pyspextool.instruments.ishell.external_profile_extraction import (
    extract_with_external_profile,
)
from pyspextool.instruments.ishell.weighted_optimal_extraction import (
    WeightedExtractionDefinition,
)

extraction_def = WeightedExtractionDefinition(
    center_frac=0.5,
    radius_frac=0.45,
    reject_outliers=True,      # iterative sigma-clipping (Stage 18)
    sigma_clip=5.0,
)

result = extract_with_external_profile(
    science_rectified,     # RectifiedOrderSet for the science target
    extraction_def,
    templates,
)
```

### Step 3: Access results

```python
# All orders
for sp in result.spectra.spectra:
    print(f"Order {sp.order}: flux shape {sp.flux.shape}")

# Single order
sp = result.spectra.get_order(311)
print(sp.wavelength_um)
print(sp.flux)
print(sp.variance)

# Bookkeeping
print(result.external_profile_applied)
# {311: True, 315: True, 320: True}

print(result.template_n_frames_used)
# {311: 3, 315: 3, 320: 3}  (if 3 calibration frames were stacked)
```

---

## Fallback Behavior

By default, every order in `rectified_orders` must have a matching template;
if any order is missing a template a `ValueError` is raised:

```
ValueError: No template found for order(s) [315] in profile_templates
(available: [311, 320]).  Provide a template for every science order or set
fallback_profile_source to 'empirical' / 'smoothed_empirical'.
```

Set `fallback_profile_source` to allow partial template coverage:

```python
result = extract_with_external_profile(
    science_rectified,
    extraction_def,
    partial_templates,            # only covers some orders
    fallback_profile_source="empirical",
)

# Check which orders used the external template
for order_num, applied in result.external_profile_applied.items():
    if applied:
        print(f"Order {order_num}: external profile (n_frames={result.template_n_frames_used[order_num]})")
    else:
        print(f"Order {order_num}: fell back to empirical")
```

Valid `fallback_profile_source` values:

| Value | Behavior |
|---|---|
| `None` (default) | Missing template → `ValueError` |
| `"empirical"` | Per-row median from science aperture |
| `"smoothed_empirical"` | Empirical + Gaussian smoothing |

---

## Validation Performed

Before extraction begins, this module checks:

1. `rectified_orders` is not empty.
2. `rectified_orders.mode == profile_templates.mode` (mode compatibility).
3. Every order in `rectified_orders` has a template, **or** `fallback_profile_source` is set.
4. `fallback_profile_source` is a valid profile-source string.
5. For each order with a template:
   - Template spatial dimension matches rectified order spatial dimension.
   - Template spectral dimension (if 2-D) matches rectified order spectral dimension.

---

## Limitations

This is a **scaffold implementation**.  The following are intentionally
**out of scope**:

- **No PSF fitting** — profiles are flat empirical estimates derived from
  per-row nanmedian, not parametric Gaussian or Moffat models.
- **No centroid alignment** — the spatial center of the template is not
  adjusted to match the centroid in the science frame.  If the star moved
  on the slit between the calibration and science observations, the
  template will be miscentered.
- **No wavelength-dependent profile warping** — the template profile is
  spectrally constant (wavelength axis is not used for warping).
- **No cross-order interpolation** — each order uses its own independent
  template.  There is no mechanism to transfer a profile from one order to
  a neighboring order.
- **No science-quality template construction** — the template is a simple
  median stack.  Bad-pixel interpolation, weighted combination, or
  sigma-clipped stacking are not implemented.
- **No template staleness check** — the function does not warn if the
  template was built from data taken in a different observing condition.
- **No telluric correction or flux calibration** — those belong to later
  pipeline stages.

---

## Differences from Science-Quality Approaches

| Property | This scaffold | Science-quality |
|---|---|---|
| Profile source | External per-row nanmedian | Parametric PSF fit |
| Wavelength dependence | None (spectrally constant) | Wavelength-dependent |
| Centroid tracking | None | Sub-pixel per wavelength |
| Inter-order profile | Independent | Cross-order transfer |
| Frame weighting | Equal weight (Stage 21) | SNR or inverse-variance |
| Template alignment | None | Spatial shift + warp |
| Bad-pixel handling | NaN masking only | Model-based interpolation |
| Slit curvature | Ignored (scaffold constraint) | Corrected |

---

## Files

| File | Purpose |
|---|---|
| `src/pyspextool/instruments/ishell/external_profile_extraction.py` | Stage 22 module (this stage) |
| `src/pyspextool/instruments/ishell/profile_templates.py` | Stage 21: template builder |
| `src/pyspextool/instruments/ishell/weighted_optimal_extraction.py` | Stages 17–20: extraction engine |
| `tests/test_ishell_external_profile_extraction.py` | Tests for Stage 22 |
| `docs/ishell_profile_templates.md` | Stage 21 documentation |
| `docs/ishell_weighted_optimal_extraction.md` | Stages 17–20 documentation |
