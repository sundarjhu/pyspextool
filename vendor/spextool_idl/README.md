# Spextool IDL Reference (Vendor Copy)

This directory contains a **local vendor copy** of the Spextool IDL codebase used as a **reference for porting iSHELL reduction logic to Python**.

## Purpose

- Provide direct access to the **IDL implementation of Spextool**, which is the authoritative source for:
  - flat-field generation
  - order location / tracing
  - wavelength calibration (1DXD / 2DXD)
  - rectification logic
- Enable systematic comparison between:
  - IDL Spextool (source of truth for algorithms)
  - pySpextool (Python architecture and interfaces)
- Support accurate, non-speculative translation of iSHELL reduction procedures.

## Important Notes

- This directory is **NOT part of the Python package**.
- None of the files here are imported or executed by the Python code.
- This is strictly a **read-only reference tree** for development.

## Scope

The contents may include:
- `.pro` source files (IDL procedures)
- configuration files used by Spextool
- minimal supporting resources required to understand the code

Large calibration data products (e.g., FITS files) are included only if necessary.

## Version / Provenance

- Source: [Specify where you obtained the Spextool IDL package]
- Version: [If known]
- Date added: [YYYY-MM-DD]

## Usage Guidelines

- When modifying Python code in `pyspextool`, developers should:
  1. Identify the corresponding IDL procedure in this directory
  2. Read and understand the IDL implementation
  3. Port the **algorithmic behavior**, not just the interface
- Avoid guessing algorithms when a corresponding IDL implementation exists.

## Exclusions

- This directory should not be modified except to:
  - update the reference version
  - add missing IDL files required for tracing dependencies

## Licensing

- This code is included for **development and reference purposes only**.
- Users must ensure compliance with the original Spextool licensing terms.