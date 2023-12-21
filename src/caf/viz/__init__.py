from matplotlib import pyplot as _plt

from . import _version

__version__ = _version.get_versions()["version"]

_plt.style.use("caf.viz.tfn")

# Aliases
from caf.viz.xy_plot import XYPlotType, axes_plot_xy, plot_xy
