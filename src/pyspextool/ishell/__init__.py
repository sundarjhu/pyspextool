"""
pyspextool.ishell — iSHELL 2DXD diagnostic and utility tools.

This sub-package collects functions used to inspect and diagnose
raw iSHELL calibration data (flat fields, arc frames) before order
tracing and full spectral extraction are carried out.
"""

from pyspextool.ishell.diagnostics import (
    plot_flat_orders,
    plot_detector_cross_section,
    plot_arc_frame,
)

__all__ = [
    "plot_flat_orders",
    "plot_detector_cross_section",
    "plot_arc_frame",
]
