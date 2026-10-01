"""
Plot Travel Demand Matrix
=========================

This example demonstrates how to plot travel matrices using :func:`caf.viz.plot_matrix`
as a graph of flows between the zones.
"""

# %%

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import caf.viz as cviz
from caf.viz import _datasets, matrix

# %%
# Get UK authority boundaries polygons to use as example zones.
zones, attr = _datasets.fetch_dataset(_datasets.Datasets.ONS_UTLA)
print(f"Loaded dataset with {len(zones):,} rows and {len(zones.columns):,} columns\n{attr}")

# %%
# Convert CRS to British National Grid for plotting examples.
zones = zones.to_crs(epsg=27700)

# %%
# Generate random matrix for plotting.
rng = np.random.default_rng(103470)
demand_matrix = pd.DataFrame(
    {
        "o": np.repeat(zones.index, len(zones)),
        "d": np.tile(zones.index, len(zones)),
        "trips": rng.normal(scale=100, size=len(zones) ** 2),
    }
)

# %%
# Use the approximate distance between the zones to scale the trips value.
centroids = zones.centroid
centroids.name = "centroid"

demand_matrix = demand_matrix.merge(
    centroids, left_on="o", right_index=True, validate="m:1"
).merge(centroids, left_on="d", right_index=True, validate="m:1", suffixes=("-o", "-d"))
demand_matrix["distance"] = gpd.GeoSeries(demand_matrix["centroid-o"]).distance(
    demand_matrix["centroid-d"]
)

long = demand_matrix["distance"] > demand_matrix["distance"].quantile(0.1)
demand_matrix.loc[long, "trips"] = demand_matrix.loc[long, "trips"] / 10

demand_matrix = demand_matrix.drop(columns=["centroid-o", "centroid-d", "distance"])
print(
    f"Generated random matrix ({len(demand_matrix):}, {len(demand_matrix.columns)})",
    f"with values from {demand_matrix['trips'].min():,.1f} - ",
    f"{demand_matrix['trips'].max():,.1f}",
)

# %%
# Plot a matrix with only positive values using the default parameters and only
# displaying (approximately) the largest 1% of values.
fig, ax = cviz.plot_matrix(
    zones,
    demand_matrix.abs(),
    demand_matrix["trips"].quantile(0.95),
)

# %%
# Plot the matrix with positive and negative values.
fig, ax = cviz.plot_matrix(
    zones,
    demand_matrix,
    demand_matrix["trips"].quantile(0.95),
    plot_title="Negative and Positive Demand",
    legend_title="Legend",
)

# %%
# Plot the matrix with default direction arrows.
fig, ax = cviz.plot_matrix(
    zones,
    demand_matrix,
    demand_matrix["trips"].quantile(0.99),
    direction_inputs=matrix.DirectionInputs(),
    plot_title="Random Matrix Plot with Directions",
)

# %%
# Explicitly close all figures once they are done with.
plt.close("all")
