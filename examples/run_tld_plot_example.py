"""
Trip Length Distribution Plot
=====================================

This example demonstrates how to create a Trip Length Distribution (TLD) plot using the
:func:`~caf.viz.xy_plot.plot_tld` function from the :mod:`caf.viz.xy_plot` module.
"""

# %%
# The following are the required imports for this example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from caf.toolkit import cost_utils

from caf.viz.xy_plot import plot_tld

plt.style.use("caf.viz.tfn")

# %%
# First, simulate some trip length data to generate a realistic trip length distribution
np.random.seed(42)
n = 1000
trip_lengths = np.random.lognormal(mean=2.5, sigma=1, size=n)
bin_edges = np.linspace(0, trip_lengths.max(), 20)
trips, bin_edges = np.histogram(trip_lengths, bins=bin_edges)

weighted_avgs = []
for i in range(len(bin_edges) - 1):
    mask = (trip_lengths >= bin_edges[i]) & (trip_lengths < bin_edges[i + 1])

    if mask.sum() > 0:
        weighted_avgs.append(trip_lengths[mask].mean())
    else:
        weighted_avgs.append((bin_edges[i] + bin_edges[i + 1]) / 2)

# %%
# Next, create the CostDistribution object from the simulated trip length data.
cost_dist = pd.DataFrame(
    {
        "min": bin_edges[:-1],
        "max": bin_edges[1:],
        "avg": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "trips": trips,
        "weighted_avg": weighted_avgs,
    }
)
cd = cost_utils.CostDistribution(cost_dist, weighted_avg_col="weighted_avg")


# %%
# Finally, generate the TLD plot.
ax = plot_tld(
    cd,
    title="Example Trip Length Distribution",
    xlabel="Trip Length (km)",
    ylabel="Number of Trips",
)
plt.show()
