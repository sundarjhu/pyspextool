"""
Provisional detector-noise / variance-image scaffold for iSHELL 2DXD reduction.

This module implements the **fourteenth stage** of the iSHELL 2DXD reduction
scaffold: a first provisional variance-image generator based on a simple
detector-noise model.

.. note::
    This is a **scaffold implementation**.  It uses a simplified noise model
    and is intentionally not a science-quality uncertainty estimate.  See
    ``docs/ishell_variance_model.md`` for a full list of assumptions and
    limitations.

Pipeline stage summary
----------------------
.. list-table::
   :header-rows: 1

   * - Stage
     - Module
     - Purpose
   * - 1
     - ``tracing.py``
     - Flat-order tracing
   * - 2
     - ``arc_tracing.py``
     - Arc-line tracing
   * - 3
     - ``wavecal_2d.py``
     - Per-order provisional wavelength mapping
   * - 4
     - ``wavecal_2d_surface.py``
     - Provisional global wavelength surface
   * - 5
     - ``wavecal_2d_refine.py``
     - Coefficient-surface refinement
   * - 6
     - ``rectification_indices.py``
     - Rectification-index generation
   * - 7
     - ``rectified_orders.py``
     - Rectified order images
   * - 8
     - ``calibration_products.py``
     - Provisional calibration product containers
   * - 9
     - ``calibration_fits.py``
     - FITS calibration writer
   * - 10
     - ``extracted_spectra.py``
     - Whole-slit provisional extraction
   * - 11
     - ``aperture_extraction.py``
     - Aperture-aware extraction
   * - 12
     - ``optimal_extraction.py``
     - Profile-weighted extraction
   * - **14**
     - **``variance_model.py``**
     - **Provisional variance-image generation (this module)**

What this module does
---------------------
Generates a per-pixel variance image (in ADU²) from a raw detector image
using a simple CCD-style noise model:

* **Poisson (shot) noise** – proportional to the signal in electrons.
  Negative pixel values are clipped to zero before contributing Poisson
  variance (controlled by ``clip_negative_flux``).

* **Read noise** – a constant per-pixel floor contributed by detector
  read-out electronics, expressed in electrons.

The provisional formula (in ADU²) is::

    poisson_var = max(image, 0) / gain_e_per_adu     [if include_poisson]
    read_var    = (read_noise_electron / gain_e_per_adu)**2  [if include_read_noise]
    variance    = poisson_var + read_var
    variance    = maximum(variance, minimum_variance)

What this module does NOT do (by design)
-----------------------------------------
* **No dark-current model** – dark current is not included.
* **No flat-field uncertainty** – flat-field noise is not propagated.
* **No bad-pixel masking** – bad pixels receive the same treatment as
  valid pixels.
* **No correlated noise** – inter-pixel correlations are ignored.
* **No covariance propagation** – only diagonal (per-pixel) variance.
* **No full detector electronics model** – only read noise and shot
  noise are included.
* **No science-quality uncertainty model** – this scaffold is intended
  for plumbing and early testing only.

Units
-----
The variance image is in **ADU²** (the same units as the raw detector
image squared).  Downstream extraction stages that operate in ADU should
pass the variance image directly.

Public API
----------
- :func:`build_variance_image` – main entry point.
- :func:`build_unit_variance_image` – convenience unit-variance constructor.
- :class:`VarianceModelDefinition` – noise-model parameters.
- :class:`VarianceImageProduct` – output product container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "VarianceModelDefinition",
    "VarianceImageProduct",
    "build_variance_image",
    "build_unit_variance_image",
]


# ---------------------------------------------------------------------------
# Noise-model definition
# ---------------------------------------------------------------------------


@dataclass
class VarianceModelDefinition:
    """Parameters for the provisional detector-noise model.

    All values describe **detector-level** quantities.  The resulting
    variance image is in the same units as the raw detector image squared
    (ADU²).

    Parameters
    ----------
    read_noise_electron : float
        RMS read noise in electrons (e⁻).  Must be ≥ 0.
    gain_e_per_adu : float
        Detector gain in electrons per ADU.  Must be > 0.
    include_poisson : bool, optional
        If ``True`` (default), add Poisson (shot) noise contribution.
    include_read_noise : bool, optional
        If ``True`` (default), add read-noise contribution.
    minimum_variance : float, optional
        Lower floor applied to every pixel in the output variance image.
        Must be > 0.  Defaults to ``1e-10``.
    clip_negative_flux : bool, optional
        If ``True`` (default), negative pixel values are clipped to zero
        before computing Poisson variance.  This prevents negative values
        from reducing the variance below the read-noise floor.

    Notes
    -----
    At least one of *include_poisson* or *include_read_noise* should be
    ``True`` to produce a meaningful variance image.  If both are
    ``False``, the output variance is ``minimum_variance`` everywhere.
    """

    read_noise_electron: float
    gain_e_per_adu: float
    include_poisson: bool = True
    include_read_noise: bool = True
    minimum_variance: float = 1e-10
    clip_negative_flux: bool = True

    def __post_init__(self) -> None:
        """Validate noise-model parameters."""
        if self.read_noise_electron < 0.0:
            raise ValueError(
                f"read_noise_electron must be >= 0; "
                f"got {self.read_noise_electron!r}."
            )
        if self.gain_e_per_adu <= 0.0:
            raise ValueError(
                f"gain_e_per_adu must be > 0; got {self.gain_e_per_adu!r}."
            )
        if self.minimum_variance <= 0.0:
            raise ValueError(
                f"minimum_variance must be > 0; got {self.minimum_variance!r}."
            )


# ---------------------------------------------------------------------------
# Output product
# ---------------------------------------------------------------------------


@dataclass
class VarianceImageProduct:
    """Provisional variance image produced by the detector-noise scaffold.

    Parameters
    ----------
    variance_image : ndarray, shape (n_rows, n_cols)
        Per-pixel variance in ADU².
    source_image_shape : tuple[int, int]
        Shape ``(n_rows, n_cols)`` of the raw detector image from which the
        variance was computed.
    definition : VarianceModelDefinition
        The noise-model parameters used to generate this product.
    mode : str or None, optional
        Instrument mode label (e.g. ``"H1"``), if known.  Purely
        informational.

    Notes
    -----
    This is a scaffold product.  It does not include bad-pixel masks,
    covariance terms, or dark-current contributions.
    """

    variance_image: npt.NDArray[np.floating]
    source_image_shape: tuple[int, int]
    definition: VarianceModelDefinition
    mode: Optional[str] = None

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the variance image ``(n_rows, n_cols)``."""
        return self.variance_image.shape  # type: ignore[return-value]

    @property
    def finite_fraction(self) -> float:
        """Fraction of pixels with finite (non-NaN, non-Inf) variance values."""
        return np.isfinite(self.variance_image).sum() / self.variance_image.size


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def build_variance_image(
    image: npt.ArrayLike,
    definition: VarianceModelDefinition,
    *,
    mode: Optional[str] = None,
) -> VarianceImageProduct:
    """Build a provisional per-pixel variance image from a detector image.

    Uses a simple CCD-style noise model::

        poisson_var = max(image, 0) / gain_e_per_adu     [if include_poisson]
        read_var    = (read_noise_electron / gain_e_per_adu)**2  [if include_read_noise]
        variance    = poisson_var + read_var
        variance    = maximum(variance, minimum_variance)

    Parameters
    ----------
    image : array-like, shape (n_rows, n_cols)
        Raw detector image in ADU.  Must be 2-D.
    definition : VarianceModelDefinition
        Noise-model parameters (gain, read noise, flags).
    mode : str or None, optional
        Instrument mode label passed through to the output product.

    Returns
    -------
    VarianceImageProduct
        Provisional variance image in ADU² with the same shape as *image*.

    Raises
    ------
    ValueError
        If *image* is not 2-D.

    Notes
    -----
    The returned variance is always ≥ ``definition.minimum_variance``.

    NaN pixels in *image* propagate to the variance image.  If a pixel is
    NaN the Poisson term is also NaN (since ``max(NaN, 0)`` is NaN in
    NumPy), so the full variance for that pixel will be NaN.

    This function does **not** perform bad-pixel masking, dark-current
    correction, or flat-field uncertainty propagation.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(
            f"image must be 2-D; got shape {img.shape!r}."
        )

    source_shape: tuple[int, int] = img.shape  # type: ignore[assignment]

    variance = np.zeros(source_shape, dtype=float)

    if definition.include_poisson:
        if definition.clip_negative_flux:
            signal = np.maximum(img, 0.0)
        else:
            signal = img
        variance = variance + signal / definition.gain_e_per_adu

    if definition.include_read_noise:
        read_var = (definition.read_noise_electron / definition.gain_e_per_adu) ** 2
        variance = variance + read_var

    variance = np.maximum(variance, definition.minimum_variance)

    return VarianceImageProduct(
        variance_image=variance,
        source_image_shape=source_shape,
        definition=definition,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_unit_variance_image(
    image: npt.ArrayLike,
    *,
    mode: Optional[str] = None,
) -> VarianceImageProduct:
    """Build a unit-variance image (variance = 1 everywhere).

    Useful as a placeholder when no noise model is available or when
    downstream stages require a variance image argument but weighting
    is not needed.

    Parameters
    ----------
    image : array-like, shape (n_rows, n_cols)
        Reference array whose shape is used to construct the variance image.
        Must be 2-D.  Pixel values are not used.
    mode : str or None, optional
        Instrument mode label passed through to the output product.

    Returns
    -------
    VarianceImageProduct
        Variance image filled with ones, with the same shape as *image*.

    Raises
    ------
    ValueError
        If *image* is not 2-D.

    Notes
    -----
    The associated :class:`VarianceModelDefinition` uses
    ``read_noise_electron=1``, ``gain_e_per_adu=1``,
    ``include_poisson=False``, and ``minimum_variance=1.0`` so that
    :attr:`VarianceImageProduct.definition` is self-consistent.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(
            f"image must be 2-D; got shape {img.shape!r}."
        )

    unit_def = VarianceModelDefinition(
        read_noise_electron=1.0,
        gain_e_per_adu=1.0,
        include_poisson=False,
        include_read_noise=False,
        minimum_variance=1.0,
    )

    source_shape: tuple[int, int] = img.shape  # type: ignore[assignment]
    variance = np.ones(source_shape, dtype=float)

    return VarianceImageProduct(
        variance_image=variance,
        source_image_shape=source_shape,
        definition=unit_def,
        mode=mode,
    )
