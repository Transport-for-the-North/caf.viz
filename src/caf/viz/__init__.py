"""Visualisation functionality and tools for transport related data."""

from matplotlib import pyplot as plt

from caf.viz import tfn_constants
from caf.viz.demand_plot import plot_matrix

from ._version import __version__

# Set TfN style
plt.style.use("caf.viz.tfn")
