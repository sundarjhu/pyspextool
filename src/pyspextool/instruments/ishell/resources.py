from importlib.resources import files
from astropy.io import fits


DATA_PACKAGE = "pyspextool.instruments.ishell.data"


def get_linearity_cube():
    """
    Load the iSHELL detector linearity correction cube.

    Raises
    ------
    RuntimeError
        If the calibration file is missing or replaced by a Git LFS pointer.
    """

    path = files(DATA_PACKAGE) / "ishell_lincorr_CDS.fits"

    if not path.exists():
        raise RuntimeError(
            "Missing iSHELL linearity calibration file "
            "(ishell_lincorr_CDS.fits). "
            "If you cloned the repository, run `git lfs pull`."
        )

    # Detect Git LFS pointer file
    with open(path, "rb") as f:
        start = f.read(64)

    if start.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            "The iSHELL linearity calibration file is a Git LFS pointer.\n"
            "Install Git LFS and run:\n\n"
            "    git lfs pull\n"
        )

    return fits.open(path)