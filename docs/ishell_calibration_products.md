# iSHELL Provisional Calibration Products (Stage 8)

This document describes the **Stage 8** module of the iSHELL 2DXD
reduction scaffold:
`src/pyspextool/instruments/ishell/calibration_products.py`.

---

## What the provisional calibration products represent

Stage 8 extracts the wavelength and spatial coordinate axes from the
rectified order images produced by Stage 7, and packages them into
structured, typed calibration-product containers.

**WaveCalProduct** (one per echelle order)

- `wavelength_um` – the wavelength axis of the rectified grid, in µm.
  This is the uniformly spaced vector that was defined by the Stage-5
  coefficient surface and carried through Stages 6 and 7 unchanged.
- `spatial_frac` – the fractional spatial axis from the rectified grid,
  in [0, 1].
- `rectified_flux` – a reference to the rectified flux image so that
  downstream stages can access the calibration axes and the corresponding
  data in one place.

**SpatCalProduct** (one per echelle order)

- `spatial_frac` – the fractional slit-position axis in [0, 1].  0.0 is
  the bottom edge of the order on the detector; 1.0 is the top edge.
- `detector_rows` – a mapping from spatial fraction to detector row.  At
  this scaffold stage this is a provisional proxy derived from the
  spatial axis itself; a later stage will populate this with the physical
  row coordinates from the rectification index arrays.

**CalibrationProductSet**

- Collects all `WaveCalProduct` and `SpatCalProduct` objects for a single
  mode.
- Provides `orders`, `n_orders`, and `get_order(order_number)` for
  convenient access.

---

## How the products relate to the rectified orders

The rectified orders produced by Stage 7 define a 2-D grid:

```
(spatial_frac axis) × (wavelength_um axis)  →  flux[j, i]
```

Stage 8 lifts the two coordinate axes out of the `RectifiedOrder` objects
and wraps them in named calibration containers.  No new information is
introduced; the axes are copied directly from the Stage-7 output.

The relationship is:

```
Stage 7 RectifiedOrder.wavelength_um  →  WaveCalProduct.wavelength_um
Stage 7 RectifiedOrder.spatial_frac   →  WaveCalProduct.spatial_frac
                                          SpatCalProduct.spatial_frac
Stage 7 RectifiedOrder.flux           →  WaveCalProduct.rectified_flux
```

---

## How this differs from final Spextool calibration files

| Aspect | Stage 8 (this module) | Final Spextool calibration files |
|---|---|---|
| Format | Python dataclasses in memory | FITS files on disk |
| Wavelength accuracy | Provisional (polynomial surface, no sigma-clipping) | Science-quality (refined, IDL-coefficient compatible) |
| Spatial axis | Fractional slit position [0, 1] | Physical arcseconds |
| Detector-row mapping | Proxy from spatial_frac | Physical row coordinates |
| Tilt/curvature | Ignored (scaffold simplification) | Corrected |
| FITS keywords | None | Full WCS + Spextool metadata |
| Intended use | Pipeline scaffold development and validation | Science-quality spectral extraction |

Stage 8 is an intermediate representation that makes the calibration axes
available in a structured form for downstream scaffold stages.  It does
**not** produce FITS files and is **not** suitable for science-quality
extraction.

---

## Intentional limitations (scaffold constraints)

The following are explicit scaffold simplifications, documented in
`docs/ishell_scaffold_constraints.md`, that are **not** to be fixed in
this stage:

- No tilt or curvature correction in the wavelength or spatial axes.
- Spatial axis in fractional units only; no conversion to arcseconds.
- Provisional wavelength solution (no sigma-clipping, no refinement).
- No FITS file output.
- No spectral extraction.

---

## Usage example

```python
from pyspextool.instruments.ishell.calibration_products import (
    build_calibration_products,
)

# Assuming `rectified` is a RectifiedOrderSet from Stage 7:
cal_products = build_calibration_products(rectified)

print(f"Mode: {cal_products.mode}")
print(f"Number of orders: {cal_products.n_orders}")

for order in cal_products.orders:
    wcp, scp = cal_products.get_order(order)
    print(
        f"Order {order}: "
        f"wavelength range [{wcp.wavelength_um.min():.4f}, "
        f"{wcp.wavelength_um.max():.4f}] µm, "
        f"n_spectral={wcp.n_spectral}, n_spatial={wcp.n_spatial}"
    )
```

---

## What is still missing for full 2DXD calibration products

The following steps are required before these products can be used for
science-quality calibration:

1. **Physical spatial calibration** – converting `spatial_frac` to
   arcseconds using the plate scale and slit geometry.
2. **Detector-row mapping** – populating `SpatCalProduct.detector_rows`
   from the physical `src_rows` in the rectification index arrays.
3. **Tilt and curvature correction** – accounting for the tilt and
   curvature of arc lines across the spatial direction.
4. **Wavelength refinement** – sigma-clipping and re-fitting the
   wavelength solution against a science-quality line list.
5. **FITS file output** – writing calibration products to FITS files with
   appropriate WCS keywords and Spextool metadata headers.
6. **IDL-coefficient compatibility** – ensuring the wavelength polynomial
   coefficients are compatible with the legacy IDL Spextool format.
