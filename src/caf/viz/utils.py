"""Miscellaneous utility functions for visualisation."""

##### IMPORTS #####

import logging
import pathlib
import re

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_PACKAGE_PATH = pathlib.Path(__file__).parent
_STYLE_SUFFIX = "mplstyle"

##### CLASSES & FUNCTIONS #####


def normalise_name(name: str) -> str:
    """Convert name to lowercase and replace spaces with underscore."""
    name = name.lower().strip()
    return re.sub(r"\s+", "_", name)


def style_names() -> list[str]:
    """List current matplotlib styles available in CAF.viz.

    Examples
    --------
    Listing all available caf.viz styles.
    >>> from caf.viz import utils
    >>> utils.style_names()
    ['tfn']

    Using the TfN style.
    >>> from matplotlib import pyplot as plt
    >>> plt.style.use("caf.viz.tfn")
    """
    styles = _PACKAGE_PATH.glob(f"**/*.{_STYLE_SUFFIX}")
    return [i.stem for i in styles]
