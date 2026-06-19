"""
Plot Polygon Heatmap
====================

Example showing how to use :func:`caf.viz.mapping.heatmap_figure` to generate
Polygon heatmaps of spatial data.
"""

# %%
# Import the :mod:`caf.viz.mapping` module and packages for generating / loading example data,
# :mod:`geodatasets` is used to get example GeoSpatial data.
import geodatasets
import geopandas as gpd
import numpy as np
from shapely import geometry

from caf.viz import mapping

# %%
# Get European NUTS zones to use as example polygons.
path = geodatasets.get_path("eurostat.nuts_rg_10m_2024_3035")
geodata = gpd.read_file(path)
print(f"Loaded dataset with {len(geodata):,} rows and {len(geodata.columns):,} columns")

# %%
# Insert column of random data for heatmap plotting.
rng = np.random.default_rng()
geodata["value"] = np.abs(rng.normal(100, 50, size=len(geodata)))
print(
    f"Generated {len(geodata):,} random values for plotting,"
    f" {geodata['value'].min():.1f} - {geodata['value'].max():.1f}"
)

# %%
# Clip data to mainland Europe for more zoomed in maps.
clipped = geodata.clip_by_rect(2362632, 1386467, 5220063, 4744847)
geodata = geodata.loc[clipped.index]
geodata.geometry = clipped
print(f"{len(geodata):,} features after clipping")

# %%
# Plot a heatmap with defined bin edges.
fig = mapping.heatmap_figure(
    geodata,
    "value",
    "Example NUTS Zones with Defined Bins",
    bins=[100, 500, 1000],
)

# %%
# Plot a heatmap with bins 8 bins generated based on the input data.
fig = mapping.heatmap_figure(
    geodata,
    "value",
    "Example NUTS Zones with Generated Bins",
    n_bins=8,
)

# %%
# Plot a heatmap with a sub-plot containing a zoomed in area.
fig = mapping.heatmap_figure(
    geodata,
    "value",
    "Example NUTS Zones with Generated Bins & Zoomed Sub-Plot",
    zoomed_bounds=mapping.Extent(4006001, 3024499, 4465735, 3866698),
    n_bins=8,
)

# %%
# Plot a heatmap with another Polygon as the boundary.
boundary = geometry.Polygon(
    [
        [3764449, 3208962],
        [4226005, 2888032],
        [4088719, 2632553],
        [4185228, 2228950],
        [3797698, 2172508],
        [3520316, 2231879],
        [3354327, 2369427],
        [3134785, 2976605],
        [3764449, 3208962],
    ]
)
fig = mapping.heatmap_figure(
    geodata,
    "value",
    "Example NUTS Zones with a Polygon Boundary",
    n_bins=8,
    polygon_boundary=boundary,
    zoomed_bounds=mapping.Extent(*boundary.bounds),
)
