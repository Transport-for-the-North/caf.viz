"""
Trip Length Distribution Plot Example
=====================================

This example demonstrates how to create a Trip Length Distribution (TLD) plot using the `plot_tld` function from the `caf.viz.xy_plot` module. 
"""

# %%
# The following are the required imports for this example
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from caf.toolkit.cost_utils import CostDistribution
from caf.viz.xy_plot import plot_tld


# %%
# First, we will simulate some trip length data to generate a realistic trip length distribution
np.random.seed(42) # Set a seed for reproducibility
n = 1000 # Number of trips to simulate
trip_lengths = np.random.lognormal(mean=2.5, sigma=1, size=n) # Simulate trip lengths using a lognormal distribution to create a realistic skewed distribution of trip lengths
bin_edges = np.linspace(0, trip_lengths.max(), 20) # Force bins to start at 0 and end at the max trip length, with 20 bins in between
trips, bin_edges = np.histogram(trip_lengths, bins=bin_edges) # Get the number of trips in each bin and the bin edges from the histogram

# Manually calculate the weighted average for each bin
weighted_avgs = [] 
for i in range(len(bin_edges) - 1):
    mask = (trip_lengths >= bin_edges[i]) & (trip_lengths < bin_edges[i+1]) # Select trips in the current bin

    if mask.sum() > 0: # If there are trips in the bin, calculate the mean of those trips
        weighted_avgs.append(trip_lengths[mask].mean()) 
    else: # If there are no trips in the bin, use the midpoint of the bin as the weighted average
        weighted_avgs.append((bin_edges[i] + bin_edges[i+1]) / 2) 

# %% 
# Next, we will create the CostDistribution object from the simulated trip length data.
cost_dist = pd.DataFrame({
    "min": bin_edges[:-1],
    "max": bin_edges[1:],
    "avg": (bin_edges[:-1] + bin_edges[1:]) / 2,
    "trips": trips,
    "weighted_avg": weighted_avgs
})
cd = CostDistribution(cost_dist, weighted_avg_col="weighted_avg")


# %% Finally, we will create the TLD plot using the `plot_tld` function by passing the CostDistribution object and customising the title and axis labels.
ax = plot_tld(
    cd,
    title="Example Trip Length Distribution",
    xlabel="Trip Length (km)",
    ylabel="Number of Trips"
)
plt.show()
