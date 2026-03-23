"""IDL vs Python comparison harness for iSHELL flat-order tracing.

This module provides a reproducible comparison path between the Python
implementation of iSHELL flat-order tracing (:func:`trace_orders_from_flat`
in ``src/pyspextool/instruments/ishell/tracing.py``) and the IDL
implementation (``mc_findorders.pro`` in ``vendor/spextool_idl/``).

.. important::

   This module contains **no tracing algorithm changes**.  It only provides
   comparison infrastructure.  Behavioral fixes are applied separately after
   a concrete mismatch is demonstrated.

=============================================================================
DELIVERABLE 1 – Required IDL Outputs and Source Routines
=============================================================================

The IDL caller chain is::

    mc_ishellcals2dxd.pro
      → mc_readflatinfo()      reads <MODE>_flatinfo.fits → flatinfo struct
      → mc_adjustguesspos()    cross-corr shift → adjusted guesspos / xranges
      → mc_findorders()        PRIMARY: produces *edgecoeffs* and *xranges*

**Primary comparison targets** from ``mc_findorders.pro``::

    edgecoeffs : fltarr(degree+1, 2, norders)
        edgecoeffs[*, 0, i] = bottom-edge polynomial for order i
        edgecoeffs[*, 1, i] = top-edge polynomial for order i
        Coefficient convention: IDL poly() order – constant term first.
    xranges    : intarr(2, norders)
        xranges[0, i] = first column where both edges are on-detector
        xranges[1, i] = last  column where both edges are on-detector

**Optional secondary targets** (need light instrumentation inside IDL loop)::

    sample_bot_rows : fltarr(nscols, norders)   per-order sampled bottom edges
    sample_top_rows : fltarr(nscols, norders)   per-order sampled top edges
    sample_cols     : intarr(nscols)            sample column positions

Python equivalents in :class:`~pyspextool.instruments.ishell.tracing.FlatOrderTrace`::

    bot_poly_coeffs : ndarray (n_orders, poly_degree+1)  ≈ edgecoeffs[*, 0, :]ᵀ
    top_poly_coeffs : ndarray (n_orders, poly_degree+1)  ≈ edgecoeffs[*, 1, :]ᵀ
    order_xranges   : ndarray (n_orders, 2)              ≈ xranges.T

=============================================================================
DELIVERABLE 2 – Interchange Format (NPZ schema)
=============================================================================

**Format**: NumPy NPZ (binary, portable, NumPy-native)

**Required NPZ arrays** (stored with Python index ordering after transposing
from IDL column-major conventions):

.. code-block:: none

    bot_edge_coeffs : ndarray, shape (n_orders, poly_degree+1), dtype float64
        IDL source : edgecoeffs[:, 0, :].T
        Bottom-edge polynomial coefficients; constant term first (index 0).

    top_edge_coeffs : ndarray, shape (n_orders, poly_degree+1), dtype float64
        IDL source : edgecoeffs[:, 1, :].T
        Top-edge polynomial coefficients; constant term first (index 0).

    xranges         : ndarray, shape (n_orders, 2), dtype int64
        IDL source : xranges.T
        Per-order valid column range: column 0 = x_start, column 1 = x_end.

**Metadata** (stored as 0-d ndarrays):

.. code-block:: none

    n_orders    : int   – number of traced orders
    poly_degree : int   – polynomial degree
    mode        : str   – iSHELL mode name, e.g. ``"K3"``

**Optional secondary arrays**:

.. code-block:: none

    sample_bot_rows : ndarray, shape (n_orders, n_sample), float32/float64
    sample_top_rows : ndarray, shape (n_orders, n_sample), float32/float64
    sample_cols     : ndarray, shape (n_sample,), int32/int64

**Expected file locations**::

    tests/test_data/idl_reference/ishell_<MODE>_tracing_reference.npz
    e.g. tests/test_data/idl_reference/ishell_K3_tracing_reference.npz

=============================================================================
DELIVERABLE 3 – IDL Export Instructions
=============================================================================

Add the following IDL procedure to the iSHELL calibration pipeline (or run
as a standalone script after ``mc_findorders`` returns).  This exports the
primary outputs as a FITS table that can then be converted to NPZ by
:func:`fits_to_npz_reference` in this module.

.. code-block:: idl

    ; -----------------------------------------------------------------------
    ; save_tracing_reference_fits.pro
    ;
    ; Call AFTER mc_findorders returns edgecoeffs and xranges.
    ; edgecoeffs : fltarr(degree+1, 2, norders)
    ; xranges    : intarr(2, norders)
    ; mode       : string, e.g. 'K3'
    ; outdir     : output directory path string
    ; -----------------------------------------------------------------------
    pro save_tracing_reference_fits, edgecoeffs, xranges, mode, outdir
      s       = size(edgecoeffs, /DIMEN)
      degree  = s[0] - 1
      norders = s[2]
    
      ; Transpose to Python (row = order) convention
      bot = transpose(reform(edgecoeffs[*, 0, *], degree+1, norders))
      top = transpose(reform(edgecoeffs[*, 1, *], degree+1, norders))
      xr  = transpose(fix(xranges))
    
      fname = outdir + '/ishell_' + mode + '_tracing_reference.fits'
      mwrfits, bot, fname, /CREATE
      mwrfits, top, fname
      mwrfits, xr,  fname
    
      hdr = headfits(fname, EXTEN=0)
      sxaddpar, hdr, 'MODE',    mode,    'iSHELL observing mode'
      sxaddpar, hdr, 'NORDERS', norders, 'Number of traced orders'
      sxaddpar, hdr, 'POLYDEG', degree,  'Edge polynomial degree'
      modfits, fname, 0, hdr
    end

After producing the FITS file, convert to NPZ with::

    from tests.test_ishell_tracing_idl_compare import fits_to_npz_reference
    fits_to_npz_reference(
        'tests/test_data/idl_reference/ishell_K3_tracing_reference.fits',
        'tests/test_data/idl_reference/ishell_K3_tracing_reference.npz',
        mode='K3',
    )

=============================================================================
DELIVERABLE 5 – Completion Status
=============================================================================

**Status**: Harness prepared; actual IDL comparison pending IDL reference data.

The comparison infrastructure (load, run, compare) is fully implemented and
validated against a synthetic round-trip.  The real IDL-vs-Python comparison
can be completed as soon as the IDL operator runs ``save_tracing_reference_fits``
and places the resulting NPZ file at the path documented above.

=============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import numpy.polynomial.polynomial as npp
import pytest
from astropy.io import fits

from pyspextool.instruments.ishell.tracing import (
    FlatOrderTrace,
    trace_orders_from_flat,
)

# ---------------------------------------------------------------------------
# Path conventions
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_IDL_REF_DIR = os.path.join(_REPO_ROOT, "tests", "test_data", "idl_reference")

# Git LFS pointer prefix (used to skip stub files in test data)
_LFS_MAGIC = b"version https://git-lfs"

# Tolerance for polynomial-position comparison (pixels).
# This is the *decisive* acceptance threshold: the max absolute deviation
# between Python and IDL edge-polynomial positions evaluated over the
# overlapping valid column range.
#
# 0.5 px is chosen as the natural sub-pixel accuracy criterion for this
# comparison.  The Sobel-COM edge detector computes a weighted centroid
# over a small window (2–5 pixels); the theoretical precision of a
# centre-of-mass estimate on a well-sampled gradient peak is ≪1 pixel.
# Using half a pixel (0.5 px) as the threshold enforces that any systematic
# drift between the IDL (float32) and Python (float64) implementations must
# be well within the detector noise floor of ≈1 pixel, while still
# comfortably clearing rounding differences introduced by the float32 vs
# float64 arithmetic paths.
_POSITION_TOL_PX = 0.5

# Tolerance for polynomial-coefficient comparison.
# This is looser since coefficient sensitivity depends on the normalisation.
_COEFF_TOL = 0.5

# Tolerance for xrange comparison (integer columns).
_XRANGE_TOL_COLS = 2


# ---------------------------------------------------------------------------
# NPZ schema sentinel
# ---------------------------------------------------------------------------

#: Required array names, expected ndim, and target dtype for each key in the
#: reference NPZ.  Used by :func:`load_idl_reference` to validate the file and
#: apply consistent dtype conversion.
IDL_REF_NPZ_SCHEMA: dict[str, int] = {
    "bot_edge_coeffs": 2,   # shape (n_orders, poly_degree+1)
    "top_edge_coeffs": 2,   # shape (n_orders, poly_degree+1)
    "xranges": 2,           # shape (n_orders, 2)
}

#: Target dtype for each required array when loaded from the reference NPZ.
IDL_REF_NPZ_DTYPES: dict[str, type] = {
    "bot_edge_coeffs": float,
    "top_edge_coeffs": float,
    "xranges": int,
}

#: Optional array names in the reference NPZ.
IDL_REF_NPZ_OPTIONAL: set[str] = {
    "sample_bot_rows",
    "sample_top_rows",
    "sample_cols",
}


# ---------------------------------------------------------------------------
# Comparison result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OrderComparisonResult:
    """Per-order comparison result between Python and IDL outputs.

    Attributes
    ----------
    order_index : int
        Zero-based order index.
    bot_coeff_max_abs_diff : float
        Max absolute coefficient difference for the bottom-edge polynomial.
        **Diagnostic only** — coefficient vectors can differ while the
        evaluated polynomial positions agree over the valid column range,
        because the differences cancel when the polynomial is summed.
        Use ``bot_pos_max_diff`` as the decisive acceptance metric.
    top_coeff_max_abs_diff : float
        Max absolute coefficient difference for the top-edge polynomial.
        **Diagnostic only** — same caveat as ``bot_coeff_max_abs_diff``.
    xstart_diff : int
        ``python_xstart - idl_xstart`` (signed, in columns).
    xend_diff : int
        ``python_xend - idl_xend`` (signed, in columns).
    has_valid_overlap : bool
        ``True`` if the Python and IDL valid column ranges overlap, i.e.
        ``max(py_xstart, idl_xstart) <= min(py_xend, idl_xend)``.
        When ``False``, the position-difference fields below are NaN
        and the order is treated as a domain disagreement failure.
    bot_pos_max_diff : float
        **Primary acceptance metric.**  Max absolute position difference
        (pixels) for the bottom-edge polynomial evaluated at every column
        in the overlapping valid range.  NaN when ``has_valid_overlap`` is
        ``False``.
    top_pos_max_diff : float
        **Primary acceptance metric.**  Max absolute position difference
        (pixels) for the top-edge polynomial over the overlapping valid
        range.  NaN when ``has_valid_overlap`` is ``False``.
    center_pos_max_diff : float
        Max absolute position difference (pixels) for the derived centre
        polynomial ``(bot + top) / 2`` over the overlapping valid range.
        NaN when ``has_valid_overlap`` is ``False``.
    """

    order_index: int
    bot_coeff_max_abs_diff: float
    top_coeff_max_abs_diff: float
    xstart_diff: int
    xend_diff: int
    has_valid_overlap: bool
    bot_pos_max_diff: float
    top_pos_max_diff: float
    center_pos_max_diff: float


@dataclass
class TracingComparisonSummary:
    """Summary of a Python-vs-IDL tracing comparison.

    Attributes
    ----------
    n_orders_python : int
    n_orders_idl : int
    n_orders_match : bool
    per_order : list of OrderComparisonResult
    max_bot_pos_diff : float
        Maximum ``bot_pos_max_diff`` across all orders (pixels).
    max_top_pos_diff : float
        Maximum ``top_pos_max_diff`` across all orders (pixels).
    max_center_pos_diff : float
        Maximum ``center_pos_max_diff`` across all orders (pixels).
    max_xstart_diff : int
        Maximum ``|xstart_diff|`` across all orders (columns).
    max_xend_diff : int
        Maximum ``|xend_diff|`` across all orders (columns).
    agreement_acceptable : bool
        ``True`` if all per-order differences are within the tolerances
        ``_POSITION_TOL_PX`` and ``_XRANGE_TOL_COLS``.
    """

    n_orders_python: int
    n_orders_idl: int
    n_orders_match: bool
    per_order: list[OrderComparisonResult]
    max_bot_pos_diff: float
    max_top_pos_diff: float
    max_center_pos_diff: float
    max_xstart_diff: int
    max_xend_diff: int
    agreement_acceptable: bool


# ---------------------------------------------------------------------------
# Helper: load IDL reference NPZ
# ---------------------------------------------------------------------------


def load_idl_reference(npz_path: str) -> dict:
    """Load and validate an IDL reference NPZ file.

    The file must contain at least the arrays listed in
    :data:`IDL_REF_NPZ_SCHEMA`.  Metadata scalars (``n_orders``,
    ``poly_degree``, ``mode``) are optional but recommended.

    Parameters
    ----------
    npz_path : str
        Path to the NPZ file.

    Returns
    -------
    dict
        Validated reference data.  Keys include at minimum
        ``"bot_edge_coeffs"``, ``"top_edge_coeffs"``, and ``"xranges"``.
        If metadata keys are absent they are inferred from array shapes.

    Raises
    ------
    FileNotFoundError
        If *npz_path* does not exist.
    ValueError
        If required arrays are missing or have wrong dimensions.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"IDL reference file not found: {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)
    ref: dict = {}

    # Validate required arrays
    for key, expected_ndim in IDL_REF_NPZ_SCHEMA.items():
        if key not in raw:
            raise ValueError(
                f"IDL reference NPZ is missing required array '{key}'. "
                f"Required keys: {list(IDL_REF_NPZ_SCHEMA)}"
            )
        arr = raw[key]
        if arr.ndim != expected_ndim:
            raise ValueError(
                f"IDL reference array '{key}' has ndim={arr.ndim}, "
                f"expected {expected_ndim}."
            )
        ref[key] = arr.astype(IDL_REF_NPZ_DTYPES[key])

    # Load optional arrays
    for key in IDL_REF_NPZ_OPTIONAL:
        if key in raw:
            ref[key] = raw[key]

    # Infer / load metadata
    n_orders_inferred = ref["bot_edge_coeffs"].shape[0]
    poly_degree_inferred = ref["bot_edge_coeffs"].shape[1] - 1

    ref["n_orders"] = int(raw["n_orders"]) if "n_orders" in raw else n_orders_inferred
    ref["poly_degree"] = (
        int(raw["poly_degree"]) if "poly_degree" in raw else poly_degree_inferred
    )
    ref["mode"] = str(raw["mode"]) if "mode" in raw else "unknown"

    # Sanity checks
    if ref["bot_edge_coeffs"].shape != ref["top_edge_coeffs"].shape:
        raise ValueError(
            "bot_edge_coeffs and top_edge_coeffs must have the same shape; "
            f"got {ref['bot_edge_coeffs'].shape} vs {ref['top_edge_coeffs'].shape}"
        )
    if ref["xranges"].shape[0] != n_orders_inferred or ref["xranges"].shape[1] != 2:
        raise ValueError(
            f"xranges must have shape (n_orders, 2); got {ref['xranges'].shape}"
        )

    return ref


# ---------------------------------------------------------------------------
# Helper: FITS reference → NPZ converter
# ---------------------------------------------------------------------------


def fits_to_npz_reference(
    fits_path: str,
    npz_path: str,
    mode: str = "unknown",
) -> None:
    """Convert an IDL-generated FITS reference file to the NPZ schema.

    The FITS file is expected to have three image extensions (as produced by
    the IDL ``save_tracing_reference_fits`` procedure documented in the module
    docstring):

    - Extension 1: ``bot_edge_coeffs`` shape ``(n_orders, poly_degree+1)``
    - Extension 2: ``top_edge_coeffs`` shape ``(n_orders, poly_degree+1)``
    - Extension 3: ``xranges``         shape ``(n_orders, 2)``

    The primary header may contain ``MODE``, ``NORDERS``, and ``POLYDEG``
    keywords; any that are present are used; absent ones are inferred.

    Parameters
    ----------
    fits_path : str
        Path to the IDL-generated FITS reference file.
    npz_path : str
        Path where the output NPZ file will be written.
    mode : str, optional
        iSHELL mode override (used only if the FITS header lacks ``MODE``).
    """
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        mode_hdr = str(hdr.get("MODE", mode))
        n_orders_hdr = int(hdr.get("NORDERS", 0))
        poly_degree_hdr = int(hdr.get("POLYDEG", 0))

        bot = np.array(hdul[1].data, dtype=float)
        top = np.array(hdul[2].data, dtype=float)
        xr = np.array(hdul[3].data, dtype=int)

    n_orders = n_orders_hdr if n_orders_hdr > 0 else bot.shape[0]
    poly_degree = poly_degree_hdr if poly_degree_hdr > 0 else (bot.shape[1] - 1)

    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez(
        npz_path,
        bot_edge_coeffs=bot,
        top_edge_coeffs=top,
        xranges=xr,
        n_orders=np.array(n_orders),
        poly_degree=np.array(poly_degree),
        mode=np.array(mode_hdr),
    )


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def compare_tracing_to_idl(
    py_result: FlatOrderTrace,
    idl_ref: dict,
) -> TracingComparisonSummary:
    """Compare Python tracing output to IDL reference outputs.

    **Decisive metric**: the max absolute polynomial-position difference
    evaluated at every column in the overlapping valid column range
    (``bot_pos_max_diff`` / ``top_pos_max_diff`` in
    :class:`OrderComparisonResult`).  An order passes when both values are
    ≤ ``_POSITION_TOL_PX``.

    **Diagnostic fields**: ``bot_coeff_max_abs_diff`` /
    ``top_coeff_max_abs_diff`` are recorded for debugging but are *not* used
    in the acceptance decision.  Coefficient vectors can differ while the
    evaluated polynomial positions agree over the valid column range, so
    coefficient differences alone are not reliable as a pass/fail criterion.

    **No-overlap case**: when the Python and IDL valid column ranges do not
    overlap for an order, ``has_valid_overlap`` is ``False``, position diffs
    are NaN, and ``agreement_acceptable`` is set to ``False`` for the overall
    summary.

    Parameters
    ----------
    py_result : FlatOrderTrace
        Result from :func:`trace_orders_from_flat`.  Must have
        ``bot_poly_coeffs``, ``top_poly_coeffs``, and ``order_xranges``
        populated (i.e. produced by the full tracing path, not constructed
        directly).
    idl_ref : dict
        Loaded reference data from :func:`load_idl_reference`.

    Returns
    -------
    TracingComparisonSummary
        Structured comparison result with per-order details and aggregate
        statistics.

    Raises
    ------
    ValueError
        If ``py_result`` is missing edge or xrange data.
    """
    if py_result.bot_poly_coeffs is None:
        raise ValueError(
            "py_result.bot_poly_coeffs is None; "
            "FlatOrderTrace must be produced by trace_orders_from_flat, "
            "not constructed directly."
        )
    if py_result.top_poly_coeffs is None:
        raise ValueError("py_result.top_poly_coeffs is None.")
    if py_result.order_xranges is None:
        raise ValueError("py_result.order_xranges is None.")

    n_py = py_result.n_orders
    n_idl = idl_ref["n_orders"]

    idl_bot = idl_ref["bot_edge_coeffs"]  # (n_orders, poly_degree+1)
    idl_top = idl_ref["top_edge_coeffs"]
    idl_xr = idl_ref["xranges"]          # (n_orders, 2)

    # Compare order-by-order for the orders present in both results.
    n_compare = min(n_py, n_idl)
    per_order: list[OrderComparisonResult] = []

    for i in range(n_compare):
        py_bot = py_result.bot_poly_coeffs[i]
        py_top = py_result.top_poly_coeffs[i]
        py_xr = py_result.order_xranges[i]

        ib = idl_bot[i]
        it = idl_top[i]
        ixr = idl_xr[i]

        # Pad shorter coefficient array with zeros to match length.
        n_py_c = len(py_bot)
        n_idl_c = len(ib)
        if n_py_c < n_idl_c:
            py_bot = np.pad(py_bot, (0, n_idl_c - n_py_c))
            py_top = np.pad(py_top, (0, n_idl_c - n_py_c))
        elif n_idl_c < n_py_c:
            ib = np.pad(ib, (0, n_py_c - n_idl_c))
            it = np.pad(it, (0, n_py_c - n_idl_c))

        bot_coeff_diff = float(np.max(np.abs(py_bot - ib)))
        top_coeff_diff = float(np.max(np.abs(py_top - it)))

        # Evaluate polynomials over the intersection of valid column ranges.
        x_start = int(max(py_xr[0], ixr[0]))
        x_end = int(min(py_xr[1], ixr[1]))
        has_overlap = x_end >= x_start

        if has_overlap:
            cols = np.arange(x_start, x_end + 1, dtype=float)
            # numpy.polynomial.polynomial.polyval uses same convention as IDL poly()
            py_bot_pos = npp.polyval(cols, py_bot)
            py_top_pos = npp.polyval(cols, py_top)
            idl_bot_pos = npp.polyval(cols, ib)
            idl_top_pos = npp.polyval(cols, it)

            bot_pos_diff = float(np.max(np.abs(py_bot_pos - idl_bot_pos)))
            top_pos_diff = float(np.max(np.abs(py_top_pos - idl_top_pos)))

            py_cen_pos = (py_bot_pos + py_top_pos) / 2.0
            idl_cen_pos = (idl_bot_pos + idl_top_pos) / 2.0
            cen_pos_diff = float(np.max(np.abs(py_cen_pos - idl_cen_pos)))
        else:
            bot_pos_diff = float("nan")
            top_pos_diff = float("nan")
            cen_pos_diff = float("nan")

        per_order.append(
            OrderComparisonResult(
                order_index=i,
                bot_coeff_max_abs_diff=bot_coeff_diff,
                top_coeff_max_abs_diff=top_coeff_diff,
                xstart_diff=int(py_xr[0]) - int(ixr[0]),
                xend_diff=int(py_xr[1]) - int(ixr[1]),
                has_valid_overlap=has_overlap,
                bot_pos_max_diff=bot_pos_diff,
                top_pos_max_diff=top_pos_diff,
                center_pos_max_diff=cen_pos_diff,
            )
        )

    # Aggregate statistics
    finite_bot = [r.bot_pos_max_diff for r in per_order if np.isfinite(r.bot_pos_max_diff)]
    finite_top = [r.top_pos_max_diff for r in per_order if np.isfinite(r.top_pos_max_diff)]
    finite_cen = [r.center_pos_max_diff for r in per_order if np.isfinite(r.center_pos_max_diff)]

    max_bot = float(max(finite_bot)) if finite_bot else float("nan")
    max_top = float(max(finite_top)) if finite_top else float("nan")
    max_cen = float(max(finite_cen)) if finite_cen else float("nan")
    max_xs = max(abs(r.xstart_diff) for r in per_order) if per_order else 0
    max_xe = max(abs(r.xend_diff) for r in per_order) if per_order else 0

    acceptable = (
        n_py == n_idl
        and all(r.has_valid_overlap for r in per_order)
        and (not finite_bot or max_bot <= _POSITION_TOL_PX)
        and (not finite_top or max_top <= _POSITION_TOL_PX)
        and max_xs <= _XRANGE_TOL_COLS
        and max_xe <= _XRANGE_TOL_COLS
    )

    return TracingComparisonSummary(
        n_orders_python=n_py,
        n_orders_idl=n_idl,
        n_orders_match=(n_py == n_idl),
        per_order=per_order,
        max_bot_pos_diff=max_bot,
        max_top_pos_diff=max_top,
        max_center_pos_diff=max_cen,
        max_xstart_diff=max_xs,
        max_xend_diff=max_xe,
        agreement_acceptable=acceptable,
    )


# ---------------------------------------------------------------------------
# Helper: run Python tracing with flatinfo parameters
# ---------------------------------------------------------------------------


def run_python_tracing(
    flat_files: list[str],
    idl_ref: dict,
    *,
    flatinfo=None,
    guess_rows: list[float] | None = None,
    seed_col: int | None = None,
    col_range: tuple[int, int] | None = None,
    slit_height_range: tuple[float, float] | None = None,
    n_sample_cols: int = 40,
    ybuffer: int = 1,
) -> FlatOrderTrace:
    """Run Python tracing with parameters consistent with an IDL reference run.

    When *flatinfo* is provided, all tracing parameters (fraction, com window,
    slit height range, step, polynomial degree) are read from it, matching the
    IDL side which reads the same parameters from the flatinfo FITS file.

    When *flatinfo* is ``None``, the caller must supply *guess_rows* and the
    remaining keyword arguments.

    Parameters
    ----------
    flat_files : list of str
        FITS flat-field file paths (same set used for the IDL reference run).
    idl_ref : dict
        Loaded reference data from :func:`load_idl_reference`; used only for
        *poly_degree* when neither *flatinfo* nor explicit degree is available.
    flatinfo : optional
        :class:`~pyspextool.instruments.ishell.calibrations.FlatInfo` object.
        When provided, overrides all tracing parameter keywords.
    guess_rows : list of float, optional
        Required when *flatinfo* is ``None``.
    seed_col : int, optional
    col_range : tuple of int, optional
    slit_height_range : tuple of float, optional
    n_sample_cols : int, optional
    ybuffer : int, optional

    Returns
    -------
    FlatOrderTrace
    """
    return trace_orders_from_flat(
        flat_files,
        flatinfo=flatinfo,
        guess_rows=guess_rows,
        seed_col=seed_col,
        col_range=col_range,
        slit_height_range=slit_height_range,
        n_sample_cols=n_sample_cols,
        poly_degree=idl_ref.get("poly_degree", 3),
        ybuffer=ybuffer,
    )


# ---------------------------------------------------------------------------
# Synthetic round-trip fixture helpers (always available – no IDL data needed)
# ---------------------------------------------------------------------------

_NROWS = 256
_NCOLS = 512
_SYN_N_ORDERS = 5
_SYN_SPACING = 40
_SYN_FIRST_CENTER = 60
_SYN_HALF_WIDTH = 10
_SYN_TILT = 0.01
_SYN_GUESS_ROWS: list[float] = [
    float(_SYN_FIRST_CENTER + k * _SYN_SPACING) for k in range(_SYN_N_ORDERS)
]


def _make_synthetic_flat(seed: int = 42) -> np.ndarray:
    """Gaussian-profile synthetic flat, 256×512, 5 orders."""
    rng = np.random.default_rng(seed)
    rows = np.arange(_NROWS, dtype=float)
    cols = np.arange(_NCOLS, dtype=float)
    flat = rng.normal(0.0, 50.0, size=(_NROWS, _NCOLS)).astype(np.float32)
    flat += 200.0
    for k in range(_SYN_N_ORDERS):
        c0 = float(_SYN_FIRST_CENTER + k * _SYN_SPACING)
        for j, col in enumerate(cols):
            center = c0 + _SYN_TILT * col
            sigma = _SYN_HALF_WIDTH / 2.355
            flat[:, j] += 10000.0 * np.exp(-0.5 * ((rows - center) / sigma) ** 2)
    return flat


def _write_fits_flat(flat: np.ndarray, path: str) -> None:
    primary = fits.PrimaryHDU(data=flat)
    ped = fits.ImageHDU(data=np.zeros_like(flat), name="SUM_PED")
    sam = fits.ImageHDU(data=np.zeros_like(flat), name="SUM_SAM")
    fits.HDUList([primary, ped, sam]).writeto(path, overwrite=True)


def _make_synthetic_idl_reference(
    py_result: FlatOrderTrace,
) -> dict:
    """Construct a synthetic IDL-like reference from a Python result.

    This helper fabricates a reference that is *identical* to the Python
    output so the round-trip comparison trivially passes.  Its purpose is
    to validate the comparison pipeline itself, not to test for IDL agreement.
    """
    assert py_result.bot_poly_coeffs is not None
    assert py_result.top_poly_coeffs is not None
    assert py_result.order_xranges is not None

    return {
        "bot_edge_coeffs": py_result.bot_poly_coeffs.copy(),
        "top_edge_coeffs": py_result.top_poly_coeffs.copy(),
        "xranges": py_result.order_xranges.copy().astype(int),
        "n_orders": py_result.n_orders,
        "poly_degree": py_result.poly_degree,
        "mode": "synthetic",
    }


# ---------------------------------------------------------------------------
# Tests – always-on (no external data required)
# ---------------------------------------------------------------------------


class TestIdlComparisonSchema:
    """Validate the NPZ schema definition and loader contract."""

    def test_schema_dict_has_required_keys(self):
        assert "bot_edge_coeffs" in IDL_REF_NPZ_SCHEMA
        assert "top_edge_coeffs" in IDL_REF_NPZ_SCHEMA
        assert "xranges" in IDL_REF_NPZ_SCHEMA

    def test_schema_ndim_values_are_correct(self):
        assert IDL_REF_NPZ_SCHEMA["bot_edge_coeffs"] == 2
        assert IDL_REF_NPZ_SCHEMA["top_edge_coeffs"] == 2
        assert IDL_REF_NPZ_SCHEMA["xranges"] == 2

    def test_load_idl_reference_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_idl_reference(str(tmp_path / "nonexistent.npz"))

    def test_load_idl_reference_raises_on_missing_key(self, tmp_path):
        p = tmp_path / "bad.npz"
        np.savez(str(p), bot_edge_coeffs=np.zeros((3, 4)))
        with pytest.raises(ValueError, match="missing required array"):
            load_idl_reference(str(p))

    def test_load_idl_reference_raises_on_wrong_ndim(self, tmp_path):
        p = tmp_path / "bad_ndim.npz"
        np.savez(
            str(p),
            bot_edge_coeffs=np.zeros(12),  # 1-D, should be 2-D
            top_edge_coeffs=np.zeros((3, 4)),
            xranges=np.zeros((3, 2), dtype=int),
        )
        with pytest.raises(ValueError, match="ndim"):
            load_idl_reference(str(p))

    def test_load_idl_reference_valid_file(self, tmp_path):
        n_orders, degree = 3, 3
        p = tmp_path / "ref.npz"
        np.savez(
            str(p),
            bot_edge_coeffs=np.ones((n_orders, degree + 1)),
            top_edge_coeffs=np.ones((n_orders, degree + 1)) * 2,
            xranges=np.array([[100, 400]] * n_orders),
            n_orders=np.array(n_orders),
            poly_degree=np.array(degree),
            mode=np.array("K3"),
        )
        ref = load_idl_reference(str(p))
        assert ref["n_orders"] == n_orders
        assert ref["poly_degree"] == degree
        assert ref["mode"] == "K3"
        assert ref["bot_edge_coeffs"].shape == (n_orders, degree + 1)
        assert ref["xranges"].dtype in (int, np.int32, np.int64, np.intp)

    def test_load_idl_reference_infers_metadata_when_absent(self, tmp_path):
        n_orders, degree = 4, 2
        p = tmp_path / "no_meta.npz"
        np.savez(
            str(p),
            bot_edge_coeffs=np.zeros((n_orders, degree + 1)),
            top_edge_coeffs=np.zeros((n_orders, degree + 1)),
            xranges=np.zeros((n_orders, 2), dtype=int),
        )
        ref = load_idl_reference(str(p))
        assert ref["n_orders"] == n_orders
        assert ref["poly_degree"] == degree
        assert ref["mode"] == "unknown"


class TestCompareTracingToIdl:
    """Validate the compare_tracing_to_idl() helper using synthetic data."""

    @pytest.fixture()
    def syn_result_and_ref(self, tmp_path):
        """Run Python tracing on synthetic flat and create matching reference."""
        flat = _make_synthetic_flat()
        fpath = str(tmp_path / "syn_flat.fits")
        _write_fits_flat(flat, fpath)

        py_result = trace_orders_from_flat(
            [fpath],
            guess_rows=_SYN_GUESS_ROWS,
        )
        idl_ref = _make_synthetic_idl_reference(py_result)
        return py_result, idl_ref

    def test_round_trip_order_count_matches(self, syn_result_and_ref):
        py_result, idl_ref = syn_result_and_ref
        summary = compare_tracing_to_idl(py_result, idl_ref)
        assert summary.n_orders_match, (
            f"n_orders mismatch: python={summary.n_orders_python}, "
            f"idl={summary.n_orders_idl}"
        )

    def test_round_trip_zero_position_diff(self, syn_result_and_ref):
        """Identical Python and IDL data → zero position differences."""
        py_result, idl_ref = syn_result_and_ref
        summary = compare_tracing_to_idl(py_result, idl_ref)
        assert summary.max_bot_pos_diff == pytest.approx(0.0, abs=1e-10), (
            f"Expected zero bot_pos_diff for round-trip; got {summary.max_bot_pos_diff}"
        )
        assert summary.max_top_pos_diff == pytest.approx(0.0, abs=1e-10)
        assert summary.max_center_pos_diff == pytest.approx(0.0, abs=1e-10)

    def test_round_trip_zero_xrange_diff(self, syn_result_and_ref):
        py_result, idl_ref = syn_result_and_ref
        summary = compare_tracing_to_idl(py_result, idl_ref)
        assert summary.max_xstart_diff == 0
        assert summary.max_xend_diff == 0

    def test_round_trip_agreement_acceptable(self, syn_result_and_ref):
        py_result, idl_ref = syn_result_and_ref
        summary = compare_tracing_to_idl(py_result, idl_ref)
        assert summary.agreement_acceptable, (
            "Round-trip comparison (Python == IDL) must report agreement_acceptable=True"
        )

    def test_per_order_results_count(self, syn_result_and_ref):
        py_result, idl_ref = syn_result_and_ref
        summary = compare_tracing_to_idl(py_result, idl_ref)
        assert len(summary.per_order) == min(
            summary.n_orders_python, summary.n_orders_idl
        )

    def test_round_trip_has_valid_overlap(self, syn_result_and_ref):
        """Round-trip (identical xranges) → has_valid_overlap=True for every order."""
        py_result, idl_ref = syn_result_and_ref
        summary = compare_tracing_to_idl(py_result, idl_ref)
        for r in summary.per_order:
            assert r.has_valid_overlap, (
                f"Order {r.order_index}: expected has_valid_overlap=True "
                "for round-trip comparison with identical xranges."
            )

    def test_no_overlap_sets_flag_and_nan_diffs(self, syn_result_and_ref):
        """When IDL xranges don't overlap with Python xranges, has_valid_overlap
        is False and position diffs are NaN; agreement_acceptable is False."""
        py_result, idl_ref = syn_result_and_ref
        # Shift all IDL xranges completely off the Python valid range
        idl_ref_bad = {k: v.copy() if isinstance(v, np.ndarray) else v
                       for k, v in idl_ref.items()}
        idl_ref_bad["xranges"] = idl_ref["xranges"].copy()
        # Place IDL xranges far to the right of the Python xranges
        idl_ref_bad["xranges"][:, 0] = idl_ref["xranges"][:, 1] + 100
        idl_ref_bad["xranges"][:, 1] = idl_ref["xranges"][:, 1] + 200
        summary = compare_tracing_to_idl(py_result, idl_ref_bad)
        for r in summary.per_order:
            assert not r.has_valid_overlap, (
                f"Order {r.order_index}: expected has_valid_overlap=False."
            )
            assert np.isnan(r.bot_pos_max_diff), (
                f"Order {r.order_index}: expected NaN bot_pos_max_diff."
            )
            assert np.isnan(r.top_pos_max_diff), (
                f"Order {r.order_index}: expected NaN top_pos_max_diff."
            )
        assert not summary.agreement_acceptable, (
            "No-overlap case must produce agreement_acceptable=False."
        )

    def test_detects_large_position_error(self, syn_result_and_ref):
        """Injecting a large coefficient shift → agreement_acceptable=False."""
        py_result, idl_ref = syn_result_and_ref
        # Corrupt one order's bottom-edge constant term by 50 pixels
        idl_ref_bad = {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in idl_ref.items()
        }
        idl_ref_bad["bot_edge_coeffs"] = idl_ref["bot_edge_coeffs"].copy()
        idl_ref_bad["bot_edge_coeffs"][0, 0] += 50.0
        summary = compare_tracing_to_idl(py_result, idl_ref_bad)
        assert not summary.agreement_acceptable

    def test_detects_wrong_order_count(self, syn_result_and_ref):
        py_result, idl_ref = syn_result_and_ref
        idl_ref_bad = dict(idl_ref)
        idl_ref_bad["n_orders"] = idl_ref["n_orders"] + 1
        summary = compare_tracing_to_idl(py_result, idl_ref_bad)
        assert not summary.n_orders_match

    def test_raises_when_py_result_has_no_edge_coeffs(self, tmp_path):
        """compare_tracing_to_idl raises ValueError when edges are None."""
        dummy = FlatOrderTrace(
            n_orders=2,
            sample_cols=np.arange(10),
            center_rows=np.ones((2, 10)),
            center_poly_coeffs=np.zeros((2, 4)),
            fit_rms=np.ones(2),
            half_width_rows=np.full(2, 10.0),
            poly_degree=3,
            seed_col=5,
            bot_poly_coeffs=None,  # missing
            top_poly_coeffs=None,
        )
        idl_ref = {
            "bot_edge_coeffs": np.zeros((2, 4)),
            "top_edge_coeffs": np.zeros((2, 4)),
            "xranges": np.array([[100, 400], [100, 400]]),
            "n_orders": 2,
            "poly_degree": 3,
            "mode": "K3",
        }
        with pytest.raises(ValueError, match="bot_poly_coeffs is None"):
            compare_tracing_to_idl(dummy, idl_ref)


class TestFitsToNpzConverter:
    """Validate the FITS → NPZ reference converter."""

    def test_round_trip_fits_to_npz(self, tmp_path):
        n_orders, degree = 3, 3
        bot = np.arange(n_orders * (degree + 1), dtype=float).reshape(n_orders, degree + 1)
        top = bot + 100.0
        xr = np.array([[50, 400], [50, 400], [50, 400]], dtype=int)

        # Build a mock FITS file matching the IDL save format
        primary = fits.PrimaryHDU()
        primary.header["MODE"] = "K3"
        primary.header["NORDERS"] = n_orders
        primary.header["POLYDEG"] = degree
        hdul = fits.HDUList([
            primary,
            fits.ImageHDU(data=bot, name="BOT"),
            fits.ImageHDU(data=top, name="TOP"),
            fits.ImageHDU(data=xr.astype(np.int32), name="XRANGES"),
        ])
        fits_path = str(tmp_path / "ref.fits")
        npz_path = str(tmp_path / "ref.npz")
        hdul.writeto(fits_path)

        fits_to_npz_reference(fits_path, npz_path, mode="K3")

        ref = load_idl_reference(npz_path)
        np.testing.assert_array_almost_equal(ref["bot_edge_coeffs"], bot)
        np.testing.assert_array_almost_equal(ref["top_edge_coeffs"], top)
        np.testing.assert_array_equal(ref["xranges"], xr)
        assert ref["mode"] == "K3"
        assert ref["n_orders"] == n_orders
        assert ref["poly_degree"] == degree


class TestNpzSaveLoad:
    """Validate that an NPZ written with the documented schema round-trips."""

    def test_save_and_load_npz(self, tmp_path):
        n_orders, degree = 5, 3
        bot = np.random.default_rng(0).random((n_orders, degree + 1))
        top = np.random.default_rng(1).random((n_orders, degree + 1))
        xr = np.tile([100, 1900], (n_orders, 1))

        p = str(tmp_path / "test_ref.npz")
        np.savez(
            p,
            bot_edge_coeffs=bot,
            top_edge_coeffs=top,
            xranges=xr,
            n_orders=np.array(n_orders),
            poly_degree=np.array(degree),
            mode=np.array("K3"),
        )

        ref = load_idl_reference(p)
        np.testing.assert_array_almost_equal(ref["bot_edge_coeffs"], bot)
        np.testing.assert_array_almost_equal(ref["top_edge_coeffs"], top)
        np.testing.assert_array_equal(ref["xranges"], xr)
        assert ref["n_orders"] == n_orders
        assert ref["poly_degree"] == degree
        assert ref["mode"] == "K3"


# ---------------------------------------------------------------------------
# Tests – real IDL comparison (skipped when reference NPZ is absent)
# ---------------------------------------------------------------------------


def _find_idl_reference_npz() -> str | None:
    """Return the path to any available IDL reference NPZ, or None."""
    if not os.path.isdir(_IDL_REF_DIR):
        return None
    for fname in sorted(os.listdir(_IDL_REF_DIR)):
        if fname.startswith("ishell_") and fname.endswith("_tracing_reference.npz"):
            return os.path.join(_IDL_REF_DIR, fname)
    return None


_IDL_REF_PATH = _find_idl_reference_npz()
_HAVE_IDL_REF = _IDL_REF_PATH is not None


@pytest.mark.skipif(
    not _HAVE_IDL_REF,
    reason=(
        "No IDL reference NPZ found in tests/test_data/idl_reference/. "
        "To enable this test, run the IDL save_tracing_reference_fits procedure "
        "documented in the module docstring of test_ishell_tracing_idl_compare.py, "
        "convert to NPZ with fits_to_npz_reference(), and place the result at "
        "tests/test_data/idl_reference/ishell_<MODE>_tracing_reference.npz."
    ),
)
class TestRealIdlComparison:
    """Compare Python tracing against real IDL outputs.

    These tests are skipped unless an IDL reference NPZ is present at
    ``tests/test_data/idl_reference/ishell_<MODE>_tracing_reference.npz``.

    When the NPZ is present, the tests assert:
    - Python and IDL agree on the number of traced orders.
    - Per-order bottom and top edge polynomial positions agree within
      ``_POSITION_TOL_PX`` pixels over the valid column range.
    - Per-order xranges agree within ``_XRANGE_TOL_COLS`` columns.

    If any assertion fails (CASE B in the problem statement), the test output
    reports the per-order breakdown to identify which order and which stage
    shows drift.
    """

    @pytest.fixture(scope="class")
    def idl_ref(self):
        return load_idl_reference(_IDL_REF_PATH)

    @pytest.fixture(scope="class")
    def comparison_summary(self, idl_ref):
        """Load the flatinfo for the reference mode and run Python tracing."""
        from pyspextool.instruments.ishell.calibrations import read_flatinfo

        mode = idl_ref["mode"]

        # Locate the flatinfo FITS for the mode
        data_dir = os.path.join(
            _REPO_ROOT,
            "src",
            "pyspextool",
            "instruments",
            "ishell",
            "data",
        )
        flatinfo_path = os.path.join(data_dir, f"{mode}_flatinfo.fits")
        if not os.path.isfile(flatinfo_path):
            pytest.skip(f"flatinfo FITS not found: {flatinfo_path}")

        flatinfo = read_flatinfo(flatinfo_path)

        # Locate flat FITS files for this mode (look in test data tree)
        flat_dir = os.path.join(
            _REPO_ROOT,
            "data",
            "testdata",
            f"ishell_{mode.lower()}_calibrations",
            "raw",
        )
        if not os.path.isdir(flat_dir):
            pytest.skip(f"Raw flat directory not found: {flat_dir}")

        from pyspextool.instruments.ishell.io_utils import is_fits_file

        flat_files = sorted(
            os.path.join(flat_dir, f)
            for f in os.listdir(flat_dir)
            if "flat" in f.lower() and is_fits_file(f)
        )
        flat_files = [
            p for p in flat_files
            if not open(p, "rb").read(64).startswith(_LFS_MAGIC)
        ]
        if not flat_files:
            pytest.skip(f"No real flat FITS files found in {flat_dir}")

        py_result = run_python_tracing(flat_files, idl_ref, flatinfo=flatinfo)
        return compare_tracing_to_idl(py_result, idl_ref)

    # --- Primary comparison assertions ---

    def test_order_count_matches(self, comparison_summary):
        s = comparison_summary
        assert s.n_orders_match, (
            f"Order count mismatch: Python traced {s.n_orders_python} orders, "
            f"IDL produced {s.n_orders_idl} orders."
        )

    def test_bottom_edge_position_within_tolerance(self, comparison_summary):
        s = comparison_summary
        for r in s.per_order:
            assert r.has_valid_overlap, (
                f"Order {r.order_index}: Python and IDL xranges do not overlap "
                "(NO COLUMN OVERLAP). Python xrange differs from IDL xrange "
                "by more than the full valid domain — cannot evaluate position "
                "agreement. Check xstart_diff and xend_diff."
            )
            assert r.bot_pos_max_diff <= _POSITION_TOL_PX, (
                f"Order {r.order_index}: bottom-edge position deviation "
                f"{r.bot_pos_max_diff:.3f} px exceeds tolerance "
                f"{_POSITION_TOL_PX} px."
            )

    def test_top_edge_position_within_tolerance(self, comparison_summary):
        s = comparison_summary
        for r in s.per_order:
            assert r.has_valid_overlap, (
                f"Order {r.order_index}: Python and IDL xranges do not overlap "
                "(NO COLUMN OVERLAP). Cannot evaluate top-edge position agreement."
            )
            assert r.top_pos_max_diff <= _POSITION_TOL_PX, (
                f"Order {r.order_index}: top-edge position deviation "
                f"{r.top_pos_max_diff:.3f} px exceeds tolerance "
                f"{_POSITION_TOL_PX} px."
            )

    def test_xranges_within_tolerance(self, comparison_summary):
        s = comparison_summary
        for r in s.per_order:
            assert abs(r.xstart_diff) <= _XRANGE_TOL_COLS, (
                f"Order {r.order_index}: xstart_diff={r.xstart_diff} cols "
                f"exceeds tolerance {_XRANGE_TOL_COLS} cols."
            )
            assert abs(r.xend_diff) <= _XRANGE_TOL_COLS, (
                f"Order {r.order_index}: xend_diff={r.xend_diff} cols "
                f"exceeds tolerance {_XRANGE_TOL_COLS} cols."
            )

    def test_agreement_acceptable_overall(self, comparison_summary):
        s = comparison_summary
        if not s.agreement_acceptable:
            lines = [
                "CASE B: Python–IDL tracing agreement is NOT acceptable.",
                f"  Max bot-edge position diff: {s.max_bot_pos_diff:.3f} px  "
                f"(tolerance {_POSITION_TOL_PX} px)",
                f"  Max top-edge position diff: {s.max_top_pos_diff:.3f} px",
                f"  Max xstart diff:            {s.max_xstart_diff} cols  "
                f"(tolerance {_XRANGE_TOL_COLS} cols)",
                f"  Max xend   diff:            {s.max_xend_diff} cols",
                "  Per-order breakdown:",
            ]
            for r in s.per_order:
                overlap_tag = "" if r.has_valid_overlap else " [NO COLUMN OVERLAP]"
                lines.append(
                    f"    order {r.order_index}{overlap_tag}: "
                    f"bot_pos={r.bot_pos_max_diff:.3f} px, "
                    f"top_pos={r.top_pos_max_diff:.3f} px, "
                    f"xstart_diff={r.xstart_diff}, "
                    f"xend_diff={r.xend_diff}"
                )
            pytest.fail("\n".join(lines))
