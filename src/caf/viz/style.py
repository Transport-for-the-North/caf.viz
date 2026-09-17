"""Contains the constants to be used for TfN styles and plotting."""

# Built-Ins
import dataclasses
import logging
import pathlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Third Party

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

NAVY = "#0d0f3d"
TEAL = "#00dec6"
PURPLE = "#7317de"
PINK = "#e50080"
WHITE = "#ffffff"
ORANGE = "#f15a29"
GREY = "#5d898c"
ICE = "#66ffff"
YELLOW = "#e9e623"
DARK_GREEN = "#1f3529"
GREEN = "#a0ca2a"

PRIMARY = (NAVY, TEAL, PURPLE, PINK, WHITE)
NPR = (TEAL, NAVY)
STRATEGIC_RAIL = (PINK, NAVY)
INT = (TEAL, NAVY)
SMART = (ORANGE, GREY)
MRN = (PURPLE, GREY)
FREIGHT = (YELLOW, DARK_GREEN)
NIP = (GREEN, DARK_GREEN)


@dataclasses.dataclass
class LogoInput:
    """Definition for parameters for to adding a logo to a plot."""

    logo_path: pathlib.Path
    logo_zoom: float = 0.4
    logo_alpha: float = 1.0
    logo_pad: float = 0.02

    def add_logo(self, ax: plt.Axes) -> plt.Axes:
        """Add the logo to the given axes.

        Parameters
        ----------
        ax : plt.Axes
            The axes to which the logo will be added.

        Returns
        -------
        plt.Axes
            The axes with the logo added.
        """
        if not self.logo_path.exists():
            raise FileNotFoundError(f"logo_path does not exist: {self.logo_path}")
        logo_img = mpimg.imread(self.logo_path)
        logo_box = OffsetImage(logo_img, zoom=self.logo_zoom, alpha=self.logo_alpha)
        logo_artist = AnnotationBbox(
            logo_box,
            (1 - self.logo_pad, self.logo_pad),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(1, 0),
            zorder=10,
        )
        ax.add_artist(logo_artist)
        return ax
