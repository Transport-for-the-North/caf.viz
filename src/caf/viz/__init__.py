"""CAF package containing functionality for transport related visualisations."""
# TODO(MB) Write a more detailed description of the package
from matplotlib import pyplot as _plt

from ._version import __version__

_plt.style.use("caf.viz.tfn")

# Aliases
from caf.viz.xy_plot import XYPlotType, axes_plot_xy, plot_xy
