# iSHELL Telluric Correction — Developer Guide

**Scope:** NASA IRTF iSHELL spectrograph, J / H / K modes only.
L, Lp, and M modes are explicitly out of scope.

---

## 1. Overview

The telluric-correction stage operates on **merged 1-D spectra** produced by
the iSHELL merge stage (see `docs/ishell_merge_guide.md`).  It implements the
Spextool **xtellcor** Vega-model method:

> An A0 V standard star is observed close in time and airmass to the science
> target.  A high-resolution Kurucz Vega model is modified to match the
> observed A0 V spectrum (accounting for radial velocity, reddening, and
> magnitude).  The ratio of the modified Vega model to the observed A0 V
> standard is the **telluric correction spectrum**.  Multiplying the science
> spectrum by this correction removes most of the atmospheric absorption while
> simultaneously providing a first-order flux calibration.

The public entry point is:

```python
from pyspextool.instruments.ishell.telluric import correct_telluric

corrected, correction, metadata = correct_telluric(
    science_spectrum,          # (4, nwave) merged science spectrum
    standard_spectrum,         # (4, nwave_std) merged A0 V standard spectrum
    vega_wavelength,           # Vega model wavelengths (µm)
    vega_fluxdensity,          # Vega model flux density (erg s⁻¹ cm⁻² Å⁻¹)
    vega_continuum,            # Vega model continuum
    vega_fitted_continuum,     # Vega model fitted continuum
    slit_width_arcsec=0.375,
    standard_bmag=9.8,
    standard_vmag=9.5,
)
```

---

## 2. Input / Output format

### Merged 1-D spectrum (4, nwave)

Both `science_spectrum` and `standard_spectrum` are `(4, nwave)` NumPy arrays:

| Row | Content                                      |
|-----|----------------------------------------------|
| 0   | Wavelength (µm), monotonically increasing    |
| 1   | Intensity (flux density)                     |
| 2   | Uncertainty                                  |
| 3   | Spectral quality flag (integer bitmask, 0 = clean) |

These are exactly the arrays returned by
`pyspextool.instruments.ishell.merge.merge_extracted_orders`.

### Vega model

The Vega model is the Kurucz synthetic spectrum at resolving power 50 000
(stored as `Vega50000.fits` in the pySpextool data directory).  It must be
loaded by the caller before passing to `correct_telluric`.  The model provides
four columns:

| Column               | Units               |
|----------------------|---------------------|
| `wavelength`         | µm                  |
| `flux density`       | erg s⁻¹ cm⁻² Å⁻¹    |
| `continuum flux density`        | erg s⁻¹ cm⁻² Å⁻¹ |
| `fitted continuum flux density` | erg s⁻¹ cm⁻² Å⁻¹ |

### Output

| Return value          | Shape           | Description                                      |
|-----------------------|-----------------|--------------------------------------------------|
| `corrected_spectrum`  | `(4, nwave)`    | Corrected science on the science wavelength grid |
| `correction_spectrum` | `(4, nwave_std)`| Correction on the standard wavelength grid       |
| `metadata`            | `dict`          | Correction parameters (see below)                |

Metadata keys:

| Key                  | Type     | Description                                        |
|----------------------|----------|----------------------------------------------------|
| `"kernel_method"`    | `str`    | Always `"ip"` for iSHELL                          |
| `"slit_width_arcsec"`| `float`  | Slit width used                                    |
| `"ip_coefficients"`  | `(3,)`   | IP coefficients `[c0, c1, c2]`                    |
| `"shift_applied"`    | `float`  | Sub-pixel shift in pixels (0.0 if none applied)   |
| `"ew_scale"`         | `float`  | Global EW scale factor                            |
| `"tilt_provisional"` | `bool`   | Whether the upstream tilt model is provisional    |
| `"intensity_unit"`   | `str`    | Flux density unit of the output                   |

---

## 3. Algorithm detail

### Step 1 — Build the IP kernel

iSHELL is a high-resolution echelle spectrograph (R ≈ 75 000 for the 0.375″
slit).  For such instruments the **deconvolution** method used by lower-
resolution instruments cannot reliably extract the instrument profile from a
single hydrogen line; therefore iSHELL uses the **IP method exclusively**
(iSHELL Spextool Manual v10jan2020, §6.2).

The IP is parameterised as an error-function profile
(see `pyspextool.telluric.core.make_instrument_profile`):

```
P(x) = C × { erf[(x + c1 − c0) / c2] − erf[(x − c1 − c0) / c2] }
```

where `x` is in standard-star pixel units, `C` normalises the profile to unit
sum, and `(c0, c1, c2)` are the slit-width-specific coefficients stored in
`IP_coefficients.dat`.

The kernel is built at the median dispersion of the **merged** standard
spectrum and spans 10 FWHMs of the slit projected onto the detector
(10 × `slit_width_pix` × `standard_dispersion` / `vega_dispersion`).

#### iSHELL plate scale

iSHELL plate scale: **0.125 arcsec/pixel** (Manual Table 4, PLTSCALE).  
For a 0.375″ slit: `slit_width_pix = 0.375 / 0.125 = 3.0 pixels`.

#### IP coefficient file

Coefficients are stored in
`src/pyspextool/instruments/ishell/IP_coefficients.dat`:

```
# slit_width_arcsec  c0  c1  c2
0.375  0.0  0.90702904  1.6253079
0.750  0.0  2.7027089   1.3422317
1.500  0.0  5.6978779   2.4753528
4.000  0.0 15.1570104   2.4753528   # scaled estimate; needs real measurement
```

The 4.000″ entry is a scaled placeholder.  All other values are derived from
the iSHELL Spextool heritage data.

### Step 2 — Modify the Vega model

`pyspextool.telluric.core.modify_kuruczvega` performs:

1. Extract the Vega model pixels covering the standard's wavelength range
   (plus kernel half-width on each side for edge handling).
2. Normalise the Vega model line profile: `(Vega_fd / Vega_continuum − 1) × ew_scale`.
3. Convolve with the IP kernel to match the instrument resolution.
4. Shift to the standard's radial velocity: `λ_shifted = λ × (1 + v_r / c)`.
5. Interpolate onto the standard's wavelength grid.
6. Redden to match the standard's B−V colour using the `dust_extinction` G23
   extinction law with `R_V = 3.1`.
7. Scale to the standard's observed V-band magnitude.
8. Convert to the requested flux density units.

The EW scale factors are provided as a constant array (`ew_scale` parameter)
because hydrogen-line EW fitting is not performed (see Section 5).

### Step 3 — Construct the correction spectrum

```
correction(λ) = Vega_model_modified(λ) / A0V_observed(λ)
```

Uncertainty propagation:

```
σ_correction² = (Vega_modified / A0V²)² × σ_A0V²
```

### Step 4 — Apply the correction

The correction spectrum is interpolated from the standard's wavelength grid
onto the science wavelength grid using `linear_interp1d`.  The science
spectrum is then multiplied:

```
science_corrected(λ) = science_observed(λ) × correction(λ)
```

Uncertainty propagation in quadrature:

```
σ_corrected² = (correction × σ_science)² + (science × σ_correction)²
```

Quality flags are combined via bitwise OR.  Science pixels outside the
wavelength coverage of the standard are flagged with bit 0.

### Step 5 — Residual wavelength shift (optional)

A small sub-pixel shift of the correction spectrum relative to the science
spectrum can be found by minimising the RMS of the corrected spectrum over a
user-specified wavelength region (`shift_wavelength_range`).  The shift is
found by `pyspextool.telluric.core.find_shift`, which evaluates the RMS over a
grid of ±1.5 pixel shifts and fits a 2nd-order polynomial around the minimum.

Enable with:

```python
correct_telluric(
    ...
    apply_shift=True,
    shift_wavelength_range=[1.20, 1.30],   # µm
)
```

The shift value is returned in `metadata["shift_applied"]` (pixels).

---

## 4. The role of the A0 V standard

An A0 V star is used as the telluric standard because its near-IR spectrum is
well described by the Kurucz Vega model, which is itself an A0 V star.
Because the telluric correction divides out both the standard's intrinsic
absorption and the atmospheric absorption simultaneously, the result is a
nearly model-independent correction with only the Vega model's H-line
morphology imposed on the science spectrum.

The standard should be:
* Observed close in time and airmass to the science target.
* Spectral type A0 V (or close: A0–A1 V).
* Have known B- and V-band magnitudes.

---

## 5. Known limitation — hydrogen-line EW fitting

In the original Spextool xtellcor, the equivalent widths of prominent
hydrogen absorption lines in the Vega model are optimised to match those of
the observed A0 V standard.  This minimises residual H-line artefacts in the
corrected spectrum.

**iSHELL does not perform this step.**

Hydrogen lines (Paschen series, Brackett series) in iSHELL spectra often span
multiple echelle orders.  On merged 1-D spectra the line profiles are
broadened and distorted by the order-merge process, making per-line EW
optimisation unreliable.

Consequence: **residual H-line absorption artefacts may remain in the
corrected spectrum**, particularly at:
- Paβ (1.282 µm) in J-band
- Paα (1.875 µm) in H/K-band
- Brγ (2.166 µm) in K-band

These residuals can be reduced by:
1. Using a standard star whose H-line EWs are very well matched to Vega.
2. Manually masking the affected wavelength regions post-correction.
3. Implementing a dedicated EW-fitting step for individual H lines once
   the upstream 2-D tilt solution is finalised.

---

## 6. Provisional tilt model

The upstream iSHELL rectification currently uses a **placeholder zero-tilt
model**: orders are rectified as if all echelle tilt angles are zero.
Spectral-line curvature is not corrected at this stage.

Merged spectra are structurally valid and suitable for pipeline development
and testing, but are **not science-quality** until a proper 2-D tilt solution
is available (pending the finalisation of the 2DXD tracing and tilt-fitting
routines).

When the merge metadata carries `tilt_provisional=True`, pass it through to
`correct_telluric`:

```python
corrected, correction, meta = correct_telluric(
    ...,
    tilt_provisional=merge_metadata["tilt_provisional"],
)
```

A `RuntimeWarning` is emitted so that downstream code can detect and log the
provisional status.

---

## 7. Example usage

```python
from astropy.io import fits
from pyspextool.instruments.ishell.merge import merge_extracted_orders
from pyspextool.instruments.ishell.telluric import correct_telluric
from pyspextool.setup_utils import mishu   # pooch helper for data files

# --- Load the Vega model --------------------------------------------------
vega_path = mishu.fetch("Vega50000.fits")
with fits.open(vega_path) as hdul:
    vdata = hdul[1].data
vega_wavelength       = vdata["wavelength"]
vega_fluxdensity      = vdata["flux density"]
vega_continuum        = vdata["continuum flux density"]
vega_fitted_continuum = vdata["fitted continuum flux density"]

# --- Merge the extracted spectra (from extract stage) --------------------
# Science target
science_merged, sci_meta = merge_extracted_orders(
    science_spectra_per_order, science_extraction_metadata
)
# A0 V standard
standard_merged, std_meta = merge_extracted_orders(
    standard_spectra_per_order, standard_extraction_metadata
)

# --- Apply telluric correction --------------------------------------------
# One aperture each
corrected, correction, meta = correct_telluric(
    science_spectrum   = science_merged[0],    # aperture 0
    standard_spectrum  = standard_merged[0],   # aperture 0
    vega_wavelength    = vega_wavelength,
    vega_fluxdensity   = vega_fluxdensity,
    vega_continuum     = vega_continuum,
    vega_fitted_continuum = vega_fitted_continuum,
    slit_width_arcsec  = 0.375,
    standard_bmag      = 9.83,
    standard_vmag      = 9.80,
    standard_rv        = 0.0,           # km/s (add barycentric correction if known)
    apply_shift        = True,
    shift_wavelength_range = [1.20, 1.30],  # µm, a clean telluric region
    tilt_provisional   = sci_meta["tilt_provisional"],
    intensity_unit     = "W m-2 um-1",
)

# corrected.shape == (4, nwave_science)
# meta["shift_applied"]   # sub-pixel wavelength shift in pixels
# meta["tilt_provisional"] # True if upstream data is provisional
```

---

## 8. File inventory

| File                                                         | Role                              |
|--------------------------------------------------------------|-----------------------------------|
| `src/pyspextool/instruments/ishell/telluric.py`              | Main implementation                |
| `src/pyspextool/instruments/ishell/IP_coefficients.dat`      | IP profile coefficients            |
| `src/pyspextool/instruments/ishell/telluric_modeinfo.dat`    | Mode metadata (method = ip)        |
| `src/pyspextool/instruments/ishell/telluric_shiftinfo.dat`   | Default shift ranges (TBD)         |
| `src/pyspextool/telluric/core.py`                            | Core algorithm functions (shared)  |
| `tests/test_ishell_telluric.py`                              | Unit / integration tests           |
| `docs/ishell_telluric_guide.md`                              | This document                      |

---

## 9. References

* Cushing, M. C. et al. (2004) *PASP* **116**, 362. — Original Spextool paper.
* Vacca, W. D., Cushing, M. C., & Rayner, J. T. (2003) *PASP* **115**, 389.
  — xtellcor Vega-model telluric correction method.
* iSHELL Spextool Manual v10jan2020, Cushing et al. — iSHELL-specific
  constraints (deconvolution unavailable, IP method, §6.2).
