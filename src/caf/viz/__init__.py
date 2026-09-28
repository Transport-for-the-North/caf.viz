"""Visualisation functionality and tools for transport related data."""

from matplotlib import pyplot as plt

from caf.viz import style
from caf.viz.matrix import plot_matrix

from ._version import __version__

# Set TfN style
plt.style.use("caf.viz.tfn")
