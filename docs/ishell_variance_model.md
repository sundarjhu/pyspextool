# iSHELL Variance Model (Stage 14) — Scaffold Documentation

> **Scaffold status.** This module is a first provisional implementation.
> It uses a simplified noise model that is not suitable for science-quality
> uncertainty estimation.  See [Limitations](#limitations) below.

## Overview

`variance_model.py` implements Stage 14 of the iSHELL 2DXD scaffold: a
provisional per-pixel variance-image generator.

The variance image produced here can be passed directly to the variance
propagation mechanisms already present in `aperture_extraction.py` (Stage 11)
and `optimal_extraction.py` (Stage 12) via their `variance_image` argument.

---

## Provisional noise-model formula

The scaffold uses a simplified CCD-style noise model.  All quantities are
in **ADU** (the same units as the raw detector image).

```
poisson_var = max(image, 0) / gain_e_per_adu        [if include_poisson=True]
read_var    = (read_noise_electron / gain_e_per_adu)**2  [if include_read_noise=True]

variance    = poisson_var + read_var
variance    = maximum(variance, minimum_variance)
```

### Poisson (shot) noise

Shot noise arises from the discrete nature of photon detection.  For a
signal of `S` electrons the variance in electrons is `S`.  Converting to
ADU:

```
poisson_var_adu = S_adu / gain = image / gain_e_per_adu
```

If `clip_negative_flux=True` (the default), negative pixel values are
clipped to zero before this calculation.  This prevents negative image
values (e.g. from background over-subtraction or read-out artefacts) from
reducing the Poisson variance below zero.

### Read noise

Read noise is a constant per-pixel contribution from the detector
electronics.  It is expressed in electrons (`read_noise_electron`) and
converted to ADU² variance:

```
read_var_adu = (read_noise_electron / gain_e_per_adu)**2
```

### Minimum-variance floor

A small positive floor (`minimum_variance`) is applied to every pixel
after summing the two terms.  This prevents downstream divisions by zero
when the variance image is used as a weight.

---

## Units

| Quantity               | Units     |
|------------------------|-----------|
| Raw detector image     | ADU       |
| `gain_e_per_adu`       | e⁻ / ADU  |
| `read_noise_electron`  | e⁻ (RMS)  |
| Output variance image  | ADU²      |

The output variance image is in **ADU²**.  Extraction stages that operate in
ADU (the current scaffold) should pass the variance image directly.

---

## How to use the result

### Generating a variance image

```python
from pyspextool.instruments.ishell.variance_model import (
    VarianceModelDefinition,
    build_variance_image,
)

defn = VarianceModelDefinition(
    read_noise_electron=10.0,   # detector read noise in e-
    gain_e_per_adu=2.0,         # detector gain in e-/ADU
)

variance_product = build_variance_image(detector_image, defn, mode="H1")
variance_image = variance_product.variance_image   # shape (n_rows, n_cols)
```

### Passing to aperture extraction

```python
from pyspextool.instruments.ishell.aperture_extraction import (
    ApertureDefinition,
    extract_with_aperture,
)

ap = ApertureDefinition(center_frac=0.5, radius_frac=0.2)
result = extract_with_aperture(
    rectified_orders,
    ap,
    method="sum",
    variance_image=variance_image,   # <-- pass here
)
```

### Passing to optimal extraction

```python
from pyspextool.instruments.ishell.optimal_extraction import (
    OptimalExtractionDefinition,
    extract_optimal,
)

opt_defn = OptimalExtractionDefinition(center_frac=0.5, radius_frac=0.2)
result = extract_optimal(
    rectified_orders,
    opt_defn,
    variance_image=variance_image,   # <-- pass here
)
```

### Unit-variance placeholder

If no noise model is available, use `build_unit_variance_image` to obtain a
valid `VarianceImageProduct` with all variances equal to 1.0:

```python
from pyspextool.instruments.ishell.variance_model import build_unit_variance_image

placeholder = build_unit_variance_image(detector_image)
```

---

## How this differs from a real detector-noise model

| Feature                        | This scaffold | Real model |
|-------------------------------|:---:|:---:|
| Shot (Poisson) noise           | ✓   | ✓   |
| Read noise                     | ✓   | ✓   |
| Dark current                   | ✗   | ✓   |
| Flat-field uncertainty         | ✗   | ✓   |
| Bad-pixel masking              | ✗   | ✓   |
| Correlated / inter-pixel noise | ✗   | ✓   |
| Covariance propagation         | ✗   | ✓   |
| Non-linearity correction       | ✗   | ✓   |
| Multiple-read noise model      | ✗   | ✓   |

---

## Limitations

This is an explicitly provisional scaffold.  The following limitations apply:

1. **No dark current.**  Thermally generated electrons are not modelled.

2. **No flat-field noise.**  The flat-field correction introduces its own
   noise that is not propagated here.

3. **No bad-pixel masking.**  Bad pixels receive the same treatment as
   valid pixels.  Downstream stages must handle bad pixels independently.

4. **No correlated noise.**  Inter-pixel correlations (e.g. from the
   detector IPC, persistence, or charge diffusion) are ignored.

5. **No covariance propagation.**  The output is a diagonal (per-pixel)
   variance.  No covariance terms are tracked.

6. **NaN propagation.**  NaN pixels in the input image produce NaN variance
   values.  No special treatment is applied.

7. **Signal-only estimate.**  The Poisson term uses the raw pixel value as a
   proxy for the true signal.  No sky-subtracted or bias-corrected signal is
   used.

---

## Public API summary

| Symbol                    | Type       | Description                                  |
|---------------------------|------------|----------------------------------------------|
| `VarianceModelDefinition` | dataclass  | Noise-model parameters (gain, read noise, …) |
| `VarianceImageProduct`    | dataclass  | Output container with variance image         |
| `build_variance_image`    | function   | Main entry point                             |
| `build_unit_variance_image` | function | Convenience unit-variance constructor        |
