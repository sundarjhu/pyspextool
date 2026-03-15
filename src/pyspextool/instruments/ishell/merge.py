"""
iSHELL order merging / spectrum assembly for J/H/K modes.

Scope
-----
J, H, and K modes only (J0–J3, H1–H3, K1–K3, Kgas).
L, Lp, and M modes are explicitly out of scope.

Overview
--------
This module provides the integration point between the iSHELL spectral
extraction stage and the generic pySpextool order-merging engine
(:func:`~pyspextool.merge.core.merge_spectra`).

Given the output of
:func:`~pyspextool.instruments.ishell.extract.extract_spectra` — a list of
per-order ``(4, nwave)`` spectra and an extraction-metadata dict — this
module produces an assembled spectrum for each requested aperture by
iteratively merging the individual echelle orders using the generic
:func:`~pyspextool.merge.core.merge_spectra` function.

What is generic vs iSHELL-specific
------------------------------------
* **Generic** — all spectral merging arithmetic (weighted averaging in
  overlap regions, wavelength-grid concatenation, uncertainty propagation,
  bitmask combination via OR) is performed by
  :func:`~pyspextool.merge.core.merge_spectra`.
* **iSHELL-specific** — translating the per-order list returned by
  :func:`~pyspextool.instruments.ishell.extract.extract_spectra` into the
  anchor/add pairs expected by the generic merging engine; sorting orders
  by wavelength; and propagating iSHELL extraction metadata into the merge
  output.

Provisional tilt model
-----------------------
The upstream extraction uses a placeholder zero-tilt rectification model.
Spectral-line curvature is **not** corrected at this stage.  This module
faithfully propagates the ``'tilt_provisional'`` flag from the extraction
metadata so that callers can detect when the upstream data are not yet
science-quality.

See ``docs/ishell_merge_guide.md`` for a full description of the pipeline
and what remains provisional pending a proper 2-D tilt solution.

Public API
----------
:func:`merge_extracted_orders`
    Accept the output of
    :func:`~pyspextool.instruments.ishell.extract.extract_spectra`,
    merge the individual echelle orders into a single assembled spectrum
    per aperture, and return the merged spectra together with metadata.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyspextool.io.check import check_parameter
from pyspextool.merge.core import merge_spectra
from pyspextool.pyspextoolerror import pySpextoolError

__all__ = [
    "merge_extracted_orders",
    "_SUPPORTED_MODES",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Supported iSHELL observing modes for J/H/K bands.
#: L, Lp, and M modes are out of scope for this module.
_SUPPORTED_MODES: frozenset[str] = frozenset(
    ["J0", "J1", "J2", "J3", "H1", "H2", "H3", "K1", "K2", "K3", "Kgas"]
)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def merge_extracted_orders(
    spectra: list,
    metadata: dict,
    apertures: Sequence[int] | None = None,
) -> tuple[list, dict]:
    """Merge order-by-order iSHELL extracted spectra into assembled spectra.

    This function is the primary integration point between the iSHELL
    extraction stage and the generic pySpextool merging engine.  It accepts
    the two-element tuple returned by
    :func:`~pyspextool.instruments.ishell.extract.extract_spectra` and
    produces one assembled ``(4, nwave_merged)`` spectrum per requested
    aperture.

    .. warning::
        **Provisional tilt model.**  If ``metadata['tilt_provisional']`` is
        ``True``, the upstream rectification used a placeholder zero-tilt
        model and spectral-line curvature has **not** been corrected.  The
        merged spectra are structurally valid and suitable for pipeline
        development and testing, but are not science-quality until a proper
        2-D tilt solution is available.  See ``docs/ishell_merge_guide.md``
        for details.

    Parameters
    ----------
    spectra : list of ndarray
        Per-order extracted spectra as returned by
        :func:`~pyspextool.instruments.ishell.extract.extract_spectra`.
        Each element is a ``(4, nwave)`` array where:

        * Row 0 — wavelength (microns).
        * Row 1 — intensity (DN s⁻¹).
        * Row 2 — uncertainty (DN s⁻¹).
        * Row 3 — spectral quality flag (0 = clean).

        The list has ``norders * n_apertures`` elements.  The ordering is
        aperture-minor, order-major: element ``i * naps + k`` corresponds to
        order index ``i`` and aperture index ``k``.

    metadata : dict
        Extraction metadata dict as returned by
        :func:`~pyspextool.instruments.ishell.extract.extract_spectra`.
        Must contain keys:

        * ``'orders'`` — list of echelle order numbers (length ``norders``).
        * ``'n_apertures'`` — number of apertures per order.

        Optional but propagated if present:

        * ``'tilt_provisional'`` — bool flag for provisional tilt model.
        * ``'subtraction_mode'``, ``'rectified'``, ``'plate_scale_arcsec'``,
          ``'aperture_positions_arcsec'``, ``'aperture_radii_arcsec'``,
          ``'aperture_signs'``.

    apertures : sequence of int or None, optional
        1-based aperture indices to merge.  ``None`` (default) merges all
        apertures.  For example, pass ``apertures=[1]`` to assemble only the
        first aperture of a multi-aperture extraction.

    Returns
    -------
    merged_spectra : list of ndarray, length n_merged_apertures
        Each element is a ``(4, nwave_merged)`` array containing the
        assembled spectrum for one aperture:

        * Row 0 — wavelength (microns), monotonically increasing.
        * Row 1 — intensity (DN s⁻¹).
        * Row 2 — uncertainty (DN s⁻¹).
        * Row 3 — spectral quality flag (integer bitmask, 0 = clean).

    merge_metadata : dict
        Metadata describing the merge, including:

        * ``'apertures'`` — list of 1-based aperture indices that were merged.
        * ``'orders'`` — list of echelle order numbers that were merged.
        * ``'n_orders'`` — number of orders merged.
        * ``'n_apertures_merged'`` — number of apertures in the output.
        * ``'tilt_provisional'`` — bool, propagated from *metadata*.
        * ``'subtraction_mode'`` — propagated from *metadata* if present.
        * ``'rectified'`` — propagated from *metadata* if present.
        * ``'plate_scale_arcsec'`` — propagated from *metadata* if present.
        * ``'aperture_positions_arcsec'`` — propagated from *metadata* if
          present.
        * ``'aperture_radii_arcsec'`` — propagated from *metadata* if present.
        * ``'aperture_signs'`` — propagated from *metadata* if present.

    Raises
    ------
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If *spectra* is not a list.
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If *metadata* is not a dict or is missing required keys
        (``'orders'``, ``'n_apertures'``).
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If the length of *spectra* is inconsistent with
        ``norders * n_apertures`` as reported in *metadata*.
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If any element of *spectra* is not a 2-D array with 4 rows.
    :class:`~pyspextool.pyspextoolerror.pySpextoolError`
        If *apertures* requests an aperture index that does not exist.
    ValueError
        If *metadata* contains fewer than 2 orders (nothing to merge).

    Examples
    --------
    Merge all apertures from a single A-mode extraction:

    >>> from pyspextool.instruments.ishell.extract import extract_spectra
    >>> from pyspextool.instruments.ishell.merge import merge_extracted_orders
    >>> spectra, meta = extract_spectra(preprocess_result, geometry)
    >>> merged, merge_meta = merge_extracted_orders(spectra, meta)
    >>> merged[0].shape   # (4, nwave_merged) for the first aperture
    (4, 8192)
    >>> merge_meta['tilt_provisional']
    True

    Merge only aperture 1 from a two-aperture A-B extraction:

    >>> merged, merge_meta = merge_extracted_orders(
    ...     spectra, meta, apertures=[1])
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(spectra, list):
        raise pySpextoolError(
            "merge_extracted_orders: spectra must be a list of ndarray "
            "(output of extract_spectra).  Got "
            f"{type(spectra).__name__}.")

    if not isinstance(metadata, dict):
        raise pySpextoolError(
            "merge_extracted_orders: metadata must be a dict "
            "(output of extract_spectra).  Got "
            f"{type(metadata).__name__}.")

    required_meta_keys = {'orders', 'n_apertures'}
    missing_keys = required_meta_keys - set(metadata.keys())
    if missing_keys:
        raise pySpextoolError(
            "merge_extracted_orders: metadata is missing required keys: "
            f"{sorted(missing_keys)}.  Ensure it is the output of "
            "extract_spectra().")

    orders = list(metadata['orders'])
    norders = len(orders)
    naps = int(metadata['n_apertures'])

    if norders < 2:
        raise ValueError(
            f"merge_extracted_orders: metadata['orders'] contains only "
            f"{norders} order(s).  At least 2 orders are required to merge.  "
            "To assemble a single-order extraction, use the spectra directly "
            "without merging.")

    expected_len = norders * naps
    if len(spectra) != expected_len:
        raise pySpextoolError(
            f"merge_extracted_orders: len(spectra) = {len(spectra)} but "
            f"metadata indicates norders={norders} × n_apertures={naps} = "
            f"{expected_len}.  Ensure spectra and metadata come from the "
            "same extract_spectra() call.")

    # Validate each spectrum element
    for idx, sp in enumerate(spectra):
        if not isinstance(sp, np.ndarray) or sp.ndim != 2 or sp.shape[0] != 4:
            raise pySpextoolError(
                f"merge_extracted_orders: spectra[{idx}] must be a 2-D "
                "ndarray with shape (4, nwave).  Got "
                f"{type(sp).__name__} with shape "
                f"{getattr(sp, 'shape', 'unknown')}.")

    # Resolve requested apertures (1-based)
    possible_apertures = list(range(1, naps + 1))
    if apertures is None:
        apertures_to_merge = possible_apertures
    else:
        apertures_to_merge = [int(a) for a in apertures]
        invalid = [a for a in apertures_to_merge
                   if a not in possible_apertures]
        if invalid:
            raise pySpextoolError(
                f"merge_extracted_orders: requested aperture(s) {invalid} "
                f"do not exist.  Available apertures: {possible_apertures}.")

    # ------------------------------------------------------------------
    # Warn if tilt model is provisional
    # ------------------------------------------------------------------
    tilt_provisional = bool(metadata.get('tilt_provisional', False))
    if tilt_provisional:
        warnings.warn(
            "merge_extracted_orders: metadata['tilt_provisional'] is True.  "
            "The upstream rectification was performed with a provisional "
            "zero-tilt model; spectral-line curvature has not been corrected.  "
            "Merged spectra are structurally valid but not science-quality "
            "pending a proper 2-D tilt solution.  "
            "See docs/ishell_merge_guide.md for details.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Sort orders by median wavelength (ascending)
    # ------------------------------------------------------------------
    # Build (order_index, median_wavelength) pairs using aperture 0
    # (all apertures share the same wavelength grid per order).
    order_median_wavelengths = []
    for i in range(norders):
        sp = spectra[i * naps + 0]   # first aperture of this order
        wl = sp[0, :]
        finite_wl = wl[np.isfinite(wl)]
        med_wl = float(np.median(finite_wl)) if len(finite_wl) > 0 else np.nan
        order_median_wavelengths.append(med_wl)

    sorted_order_indices = sorted(
        range(norders), key=lambda i: order_median_wavelengths[i]
    )

    # ------------------------------------------------------------------
    # Merge orders for each requested aperture
    # ------------------------------------------------------------------
    merged_spectra = []
    for ap in apertures_to_merge:
        k = ap - 1   # 0-based aperture index

        # Anchor: first order in sorted wavelength order
        i0 = sorted_order_indices[0]
        anchor_sp = spectra[i0 * naps + k]

        merged_wavelength = anchor_sp[0, :].astype(float)
        merged_intensity = anchor_sp[1, :].astype(float)
        merged_uncertainty = anchor_sp[2, :].astype(float)
        merged_bitmask = anchor_sp[3, :].astype(np.uint8)

        # Iteratively add subsequent orders
        for sort_pos in range(1, norders):
            i = sorted_order_indices[sort_pos]
            add_sp = spectra[i * naps + k]

            add_wavelength = add_sp[0, :].astype(float)
            add_intensity = add_sp[1, :].astype(float)
            add_uncertainty = add_sp[2, :].astype(float)
            add_bitmask = add_sp[3, :].astype(np.uint8)

            result = merge_spectra(
                merged_wavelength,
                merged_intensity,
                add_wavelength,
                add_intensity,
                anchor_uncertainty=merged_uncertainty,
                anchor_bitmask=merged_bitmask,
                add_uncertainty=add_uncertainty,
                add_bitmask=add_bitmask,
            )

            merged_wavelength = result['wavelength']
            merged_intensity = result['intensity']
            # merge_spectra returns None for uncertainty/bitmask if
            # either input was None; fall back gracefully.
            if result['uncertainty'] is not None:
                merged_uncertainty = result['uncertainty']
            else:
                merged_uncertainty = np.full_like(merged_wavelength, np.nan)

            if result['bitmask'] is not None:
                merged_bitmask = result['bitmask'].astype(np.uint8)
            else:
                merged_bitmask = np.zeros(len(merged_wavelength),
                                          dtype=np.uint8)

        assembled = np.vstack((
            merged_wavelength,
            merged_intensity,
            merged_uncertainty,
            merged_bitmask.astype(float),
        ))
        merged_spectra.append(assembled)

    # ------------------------------------------------------------------
    # Build merge metadata
    # ------------------------------------------------------------------
    merge_metadata: dict = {
        'apertures': apertures_to_merge,
        'orders': orders,
        'n_orders': norders,
        'n_apertures_merged': len(apertures_to_merge),
        'tilt_provisional': tilt_provisional,
    }
    # Propagate optional keys from extraction metadata
    for key in ('subtraction_mode', 'rectified', 'plate_scale_arcsec',
                'aperture_positions_arcsec', 'aperture_radii_arcsec',
                'aperture_signs'):
        if key in metadata:
            merge_metadata[key] = metadata[key]

    return merged_spectra, merge_metadata
