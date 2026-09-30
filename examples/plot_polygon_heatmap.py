"""
Plot Polygon Heatmap
====================

Example showing how to use :func:`caf.viz.mapping.heatmap_figure` to generate
Polygon heatmaps of spatial data.
"""

# %%
# Import the :mod:`caf.viz.mapping` module and packages for generating / loading example data,
# private :mod:`caf.viz._datasets` module is used to get example GeoSpatial data.
import numpy as np
from shapely import geometry

from caf.viz import _datasets, mapping

# %%
# Get UK Upper-tier/unitary authorities to use as example polygons.
geodata, attr = _datasets.fetch_dataset(_datasets.Datasets.ONS_UTLA)
YEAR = 2025
mask = (geodata["start"].isna() | (geodata["start"] <= YEAR)) & (
    geodata["end"].isna() | (geodata["end"] > YEAR)
)
geodata = geodata.loc[mask]
print(
    f"Loaded dataset with {len(geodata):,} rows and {len(geodata.columns):,} columns\n{attr}"
)

# %%
# Convert to the British National Grid CRS for plotting.
geodata = geodata.to_crs(27700)

# %%
# Insert column of random data for heatmap plotting.
rng = np.random.default_rng()
geodata["value"] = np.abs(rng.normal(100, 50, size=len(geodata)))
print(
    f"Generated {len(geodata):,} random values for plotting,"
    f" {geodata['value'].min():.1f} - {geodata['value'].max():.1f}"
)

# %%
# Plot a heatmap with defined bin edges.
fig = mapping.heatmap_figure(
    geodata,
    "value",
    "Example UK Authorities with Defined Bins",
    bins=[100, 500, 1000],
)

# %%
# Plot a heatmap with bins 8 bins generated based on the input data.
fig = mapping.heatmap_figure(
    geodata,
    "value",
    "Example UK Authorities with Generated Bins",
    n_bins=8,
)

# %%
# Plot a heatmap with a sub-plot containing a zoomed in area.
fig = mapping.heatmap_figure(
    geodata,
    "value",
    "Example UK Authorities with Generated Bins & Zoomed Sub-Plot",
    zoomed_bounds=mapping.Extent(297400, 347300, 541000, 658000),
    n_bins=8,
)

# %%
# Plot a heatmap with another Polygon as the boundary.
boundary = geometry.Polygon(
    [
        [316231, 385575],
        [320059, 459347],
        [285117, 515784],
        [313416, 563161],
        [398241, 670853],
        [426965, 632406],
        [456268, 532029],
        [495635, 513204],
        [527752, 471537],
        [549597, 407400],
        [443887, 377371],
        [341587, 366190],
    ]
)
fig = mapping.heatmap_figure(
    geodata,
    "value",
    "Example UK Authorities with a Polygon Boundary",
    n_bins=8,
    polygon_boundary=boundary,
    zoomed_bounds=mapping.Extent(*boundary.bounds),
)

# %%
# Plot a heatmap with positive and negative values
mask = rng.choice([-1, 1], size=len(geodata))
if mask.min() == mask.max():
    mask[: len(mask) // 2] = -1
    mask[len(mask) // 2 :] = 1

geodata["negatives"] = geodata["value"] * mask
fig = mapping.heatmap_figure(
    geodata,
    "negatives",
    "Example UK Authorities with Positive and Negative Values",
    n_bins=8,
    polygon_boundary=boundary,
    zoomed_bounds=mapping.Extent(*boundary.bounds),
    positive_negative_colormaps=True,
)
