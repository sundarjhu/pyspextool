# iSHELL Extraction Product Model

This document describes the intended data model for extraction-related
products in the iSHELL 2DXD scaffold.

## Motivation

Several scaffold stages produce per-order spectra:

- whole-slit extraction
- aperture extraction
- optimal extraction (future)
- order merging (future)

To prevent incompatible result containers from proliferating,
future extraction stages should converge on a **shared per-order
spectrum product model**.

## Design guidance

Extraction-related modules should:

- operate on **per-order spectral products**
- preserve the wavelength axis
- allow additional metadata to be attached without breaking
  downstream stages

Typical fields may include:

- order number
- wavelength axis
- extracted flux
- variance (optional)
- mask (optional)
- extraction metadata (aperture, background, etc.)

## Scaffold constraint

New extraction stages **should not introduce incompatible
result containers** if the existing structure can be reused.

Instead, they should extend or adapt the existing extraction
product structure.

This keeps later stages such as:

- optimal extraction
- variance propagation
- order stitching

architecturally simple.
