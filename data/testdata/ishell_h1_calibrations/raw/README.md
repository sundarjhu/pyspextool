# H1 Raw Calibration Data for iSHELL Arc-Tracing Tests

## Status

**No raw 2-D arc-lamp or flat-lamp FITS frames are currently present in this
directory.**

The arc-tracing scaffold
(`src/pyspextool/instruments/ishell/arc_tracing.py`) and the provisional
2DXD wavelength-mapping scaffold
(`src/pyspextool/instruments/ishell/wavecal_2dxd.py`) are designed to
eventually use real 2-D raw arc FITS frames from this directory.  Until
those frames are available, the code falls back to the packaged
`H1_wavecalinfo.fits` plane-1 data as a 1-D centerline arc-spectrum proxy.

## Expected contents (when available)

When real H1 calibration frames are obtained, this directory should contain:

| Filename pattern | Description |
|-----------------|-------------|
| `arc_H1_*.fits`  | Raw 2-D ThAr arc-lamp frames, 2048 × 2048 detector image |
| `flat_H1_*.fits` | Raw 2-D flat-lamp frames, 2048 × 2048 detector image |

These files should be raw (dark-subtracted and linearity-corrected, but not
flat-fielded) iSHELL H1 mode (1.47–1.83 µm) frames taken at the IRTF.

## How to use real frames (future)

Once real frames are available, call:

```python
from pyspextool.instruments.ishell.arc_tracing import (
    trace_arc_lines_from_2d_image
)
from pyspextool.instruments.ishell.calibrations import (
    read_flatinfo, read_wavecalinfo, read_line_list
)

wci = read_wavecalinfo("H1")
fi  = read_flatinfo("H1")
ll  = read_line_list("H1")

# Load your real arc image
from astropy.io import fits
arc_image = fits.getdata("data/testdata/ishell_h1_calibrations/raw/arc_H1_001.fits")

# NOTE: trace_arc_lines_from_2d_image() is currently a stub (NotImplementedError).
# It needs to be implemented once real frames are available.
arc_traces = trace_arc_lines_from_2d_image(arc_image, fi, ll, wci)
```

## Current proxy path

```python
from pyspextool.instruments.ishell.arc_tracing import (
    build_arc_trace_result_from_wavecalinfo
)
from pyspextool.instruments.ishell.calibrations import (
    read_flatinfo, read_wavecalinfo, read_line_list
)
from pyspextool.instruments.ishell.wavecal_2dxd import fit_provisional_2dxd

wci = read_wavecalinfo("H1")
fi  = read_flatinfo("H1")
ll  = read_line_list("H1")

# Uses packaged H1_wavecalinfo.fits plane-1 as arc spectrum proxy
arc_traces = build_arc_trace_result_from_wavecalinfo(wci, ll, fi)
result = fit_provisional_2dxd(arc_traces, wci, fi, ll)
```
