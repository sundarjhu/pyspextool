# iSHELL Profile Templates — Stage 21 Scaffold

## Overview

`profile_templates.py` implements **Stage 21** of the iSHELL 2DXD reduction
scaffold: an external spatial-profile template builder that constructs a
profile array suitable for direct use with `profile_source="external"` in the
Stage-20 weighted optimal extractor (`weighted_optimal_extraction.py`).

---

## Motivation

Stage 20 (`weighted_optimal_extraction.py`) added support for
`profile_source="external"`, allowing a caller to supply a pre-built 2-D
spatial profile array instead of estimating the profile from the science frame
being extracted.  External profiles are useful when:

- The science frame is too faint to derive a reliable empirical profile.
- A high-SNR calibration frame (e.g. a bright standard star or flat) is
  available and should be used to constrain the profile shape.
- Multiple dithered or stacked exposures are available and can be combined
  to build a more stable profile estimate.

Before Stage 21, there was no helper in the repository that actually *built*
such an external profile from rectified order images.  Stage 21 fills that gap
at the scaffold level.

---

## What the template builder does

The main entry point is:

```python
from pyspextool.instruments.ishell.profile_templates import (
    ProfileTemplateDefinition,
    build_external_profile_template,
)

definition = ProfileTemplateDefinition(
    combine_method="median",   # "median" or "mean"
    normalize_profile=True,    # sum each column to 1.0
    smooth_sigma=0.0,          # Gaussian smoothing σ in spatial pixels
    min_finite_fraction=0.5,   # minimum finite fraction required per pixel
)

template_set = build_external_profile_template(
    rectified_order_sets,      # one RectifiedOrderSet or a list
    definition,
    mask=None,                 # optional 2-D boolean bad-pixel mask
)
```

For each echelle order that appears in all input sets, the builder:

1. **Stacks** the rectified flux images from all input sets along a new axis,
   giving a `(n_frames, n_spatial, n_spectral)` array.

2. **Applies the mask** (if supplied) to each flux image before stacking:
   pixels marked `True` in the mask are set to `NaN`.

3. **Combines** the stack using the per-pixel `nanmedian` or `nanmean`
   (controlled by `combine_method`).  Pixels that are finite in fewer than
   `min_finite_fraction` of frames are set to `NaN` in the combined image.

4. **Estimates the profile** from the combined image using a per-row
   `nanmedian` across the wavelength axis (equivalent to the `"global_median"`
   profile mode used in Stages 12 and 17).  The resulting 1-D profile vector
   is broadcast across all spectral columns, giving a `(n_spatial, n_spectral)`
   profile array.  Negative values are clipped to zero.

5. **Smooths** the profile along the spatial axis using a Gaussian kernel of
   standard deviation `smooth_sigma` pixels (same kernel as the
   `"smoothed_empirical"` profile source in Stage 20).  Smoothing is skipped
   when `smooth_sigma=0.0`.

6. **Normalizes** each spectral column of the profile to sum to 1.0
   (when `normalize_profile=True`).  Columns with a zero or NaN spatial sum
   are set to `NaN`.

---

## How profiles are combined across frames

When a single `RectifiedOrderSet` is supplied, the profile is derived directly
from that single frame — no multi-frame combination is performed, though the
interface is identical.

When multiple sets are supplied:

- All sets must share the same `mode` string (e.g. `"H1"`).
- Only orders whose number appears in **all** sets are included in the output.
- The flux images for each common order are stacked pixel-by-pixel.
- The chosen combine method (`median` or `mean`) is applied column-by-column
  across frames using `numpy.nanmedian` / `numpy.nanmean`, so that NaN values
  from bad pixels or rectification boundaries are handled robustly.

The `min_finite_fraction` parameter provides an additional quality gate:
pixels that are finite in fewer than `min_finite_fraction × n_frames` of the
input frames are masked out in the combined image.  This prevents noisy or
unreliable pixels from contributing to the combined profile.

### Median vs mean

- **Median** (default): more robust to outliers and cosmic rays; recommended
  when individual frames may have unmasked artifacts.
- **Mean**: preserves flux levels more faithfully; appropriate when frames
  have been cleaned and an unbiased average is desired.

---

## How this relates to Stage 20 external profiles

The per-order `profile` array stored in each `ExternalProfileTemplate` has
shape `(n_spatial, n_spectral)` and can be supplied directly as
`external_profile` in `WeightedExtractionDefinition` with
`profile_source="external"`:

```python
from pyspextool.instruments.ishell.weighted_optimal_extraction import (
    WeightedExtractionDefinition,
    extract_weighted_optimal,
)

template = template_set.get_order(311)

ext_def = WeightedExtractionDefinition(
    center_frac=0.5,
    radius_frac=0.2,
    profile_source="external",
    external_profile=template.profile,
)
result = extract_weighted_optimal(science_rectified_orders, ext_def)
```

The extraction code accepts the full-order profile and slices the aperture
rows internally (the same row selection that is applied to the flux image).

---

## How this differs from real science-quality profile modeling

This scaffold is intentionally simple.  Key differences from a full
science-quality profile model are:

| Property | This scaffold | Science-quality approach |
|---|---|---|
| Profile estimation | Per-row nanmedian → 1-D vector | Parametric PSF fit (e.g. Gaussian, Moffat) |
| Wavelength dependence | Profile is wavelength-independent (broadcast) | Wavelength-dependent PSF model |
| Spatial centroid | No tracking; raw fractional slit coordinates | Sub-pixel centroid tracking per wavelength |
| Inter-order profile | Each order built independently | Profile transfer across orders |
| Frame weighting | Equal weight per frame | SNR-weighted or inverse-variance weighted |
| Bad-pixel interpolation | NaN masking; no replacement | Model-based replacement of bad pixels |
| Slit curvature | Ignored | Corrected before profiling |

---

## What remains intentionally unimplemented

The following capabilities are explicitly **out of scope** for Stage 21:

- **PSF fitting** — no parametric PSF model (Gaussian, Moffat, empirical kernel).
- **Order-to-order profile transfer** — each order is built independently.
- **Wavelength-dependent centroid alignment** — no centroid tracking along
  the dispersion axis.
- **Cross-night calibration logic** — all input frames are treated as equally
  valid regardless of observing date or conditions.
- **Science-quality template construction** — this scaffold is intended for
  pipeline plumbing and early testing only.
- **Telluric correction** — not applicable at this stage.
- **Merged-spectrum products** — order merging belongs to a later stage.
- **Source masking** — the `mask_sources` field in `ProfileTemplateDefinition`
  is reserved for a future implementation that would mask bright point sources
  before combining; it currently has no effect.

---

## Public API

### `ProfileTemplateDefinition`

Dataclass storing template-builder parameters.

| Field | Type | Default | Description |
|---|---|---|---|
| `combine_method` | `str` | `"median"` | How to combine frames: `"median"` or `"mean"`. |
| `normalize_profile` | `bool` | `True` | Normalize each column to spatial sum = 1.0. |
| `smooth_sigma` | `float` | `0.0` | Gaussian smoothing σ in spatial pixels (0 = no smoothing). |
| `min_finite_fraction` | `float` | `0.5` | Minimum fraction of frames required to be finite per pixel. |
| `mask_sources` | `bool` | `True` | Reserved; currently has no effect. |

### `ExternalProfileTemplate`

Per-order profile container.

| Field | Type | Description |
|---|---|---|
| `order` | `int` | Echelle order number. |
| `wavelength_um` | `ndarray` | Wavelength axis in µm. |
| `spatial_frac` | `ndarray` | Spatial axis in fractional slit coordinates [0, 1]. |
| `profile` | `ndarray` | 2-D spatial profile, shape `(n_spatial, n_spectral)`. |
| `n_frames_used` | `int` | Number of input frames that contained this order. |
| `source_mode` | `str` | iSHELL observing mode (e.g. `"H1"`). |
| `profile_smoothed` | `bool` | Whether Gaussian smoothing was applied. |
| `finite_fraction` | `float` | Fraction of finite pixels across the input stack. |

Properties: `n_spectral`, `n_spatial`, `shape`.

### `ExternalProfileTemplateSet`

Collection container.

| Field | Type | Description |
|---|---|---|
| `mode` | `str` | iSHELL observing mode. |
| `templates` | `list[ExternalProfileTemplate]` | One per order. |

Properties: `orders`, `n_orders`.
Methods: `get_order(order)`.

### `build_external_profile_template`

```python
build_external_profile_template(
    rectified_order_sets,   # RectifiedOrderSet or list[RectifiedOrderSet]
    definition,             # ProfileTemplateDefinition
    *,
    mask=None,              # optional 2-D boolean bad-pixel mask
) -> ExternalProfileTemplateSet
```

Raises `ValueError` on:
- Empty input list.
- Input sets with different modes.
- No orders common to all input sets.
- Mask with incompatible shape.

---

## Pipeline position

Stage 21 sits between the rectified-order generation (Stage 7) and the
weighted-optimal extraction (Stages 17–20):

```
Stage 7: rectified_orders.py → RectifiedOrderSet
         ↓
Stage 21: profile_templates.py → ExternalProfileTemplateSet
         ↓ (template.profile)
Stage 20: weighted_optimal_extraction.py (profile_source="external")
         ↓
WeightedExtractedSpectrumSet
```

The template set can be built from the same data used for extraction (single
frame) or from a separate, higher-SNR calibration frame or stack.
Templates are typically built from calibration or stacked frames, not the
same data being extracted; using independent calibration data ensures that the
profile estimate is not contaminated by the noise of the science target.
