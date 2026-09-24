"""
Plot Travel Demand Matrix
=========================

This example demonstrates how to plot travel matrices using :func:`caf.viz.plot_matrix`
as a graph of flows between the zones.
"""

# %%

import geodatasets
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import caf.viz as cviz
from caf.viz import demand_plot

# %%
# Get Atlanta polygons from :mod:`geodatasets` to use as example zones.
zones = gpd.read_file(geodatasets.get_path("GeoDa Atlanta"))
print(f"Loaded dataset with {len(zones):,} rows and {len(zones.columns):,} columns")

# %%
# Generate random matrix for plotting.
rng = np.random.default_rng(103470)
matrix = pd.DataFrame(
    {
        "o": np.repeat(zones.index, len(zones)),
        "d": np.tile(zones.index, len(zones)),
        "trips": rng.normal(scale=100, size=len(zones) ** 2),
    }
)
print(
    f"Generated random matrix ({len(matrix):}, {len(matrix.columns)})",
    f"with values from {matrix['trips'].min():,.1f} - {matrix['trips'].max():,.1f}",
)

# %%
# Plot a matrix with only positive values using the default parameters and only
# displaying (approximately) the largest 5% of values.
fig, ax = cviz.plot_matrix(
    zones,
    matrix.abs(),
    matrix["trips"].quantile(0.95),
)
fig  # noqa: B018

# %%
# Plot the matrix with positive and negative values.
fig, ax = cviz.plot_matrix(
    zones,
    matrix,
    matrix["trips"].quantile(0.95),
    plot_title="Negative and Positive Demand",
    legend_title="Legend",
)
fig  # noqa: B018


# %%
# Plot the matrix with default direction arrows.
fig, ax = cviz.plot_matrix(
    zones,
    matrix,
    matrix["trips"].quantile(0.95),
    direction_inputs=demand_plot.DirectionInputs(),
    plot_title="Random Matrix Plot with Directions",
)
fig  # noqa: B018

# %%
# Explicitly close all figures once they are done with.
plt.close("all")
