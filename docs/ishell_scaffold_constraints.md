# iSHELL 2DXD Scaffold Constraints

These constraints apply to the current development scaffold for the
iSHELL 2DXD calibration pipeline.

They are **intentional simplifications** and must not be “fixed” or
expanded unless explicitly requested in a later stage.

## Rectification indices (Stage 6)

The current rectification-index scaffold intentionally:

- ignores arc-line tilt
- ignores arc-line curvature
- uses fractional spatial coordinates in the range [0, 1]
- uses raw detector column units (not normalized)

These are scaffold simplifications.

## Wavelength model

The wavelength solution produced by the scaffold:

- is provisional
- is not sigma-clipped
- is not IDL-coefficient compatible
- uses provisional order-number assignment

This is acceptable for scaffold development.

## Rectification mapping

The rectification indices currently provide only a geometric mapping:

(rectified wavelength, spatial fraction) → (detector row, detector column)

They do **not** yet include:

- tilt correction
- curvature correction
- physical spatial calibration
- detector-image interpolation

Those steps belong to later stages.

## Important rule

Future scaffold stages must **consume the existing outputs**
without retroactively modifying earlier stages.
