"""
Plot Simple Map
===============

Example for :func:`~caf.viz.mapping.simple_map` which can map basic geometry layers
with an OpenStreetMap background.
"""

# %%
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from shapely import geometry

from caf.viz import mapping

# %%
# Use the TfN Matplotlib style for the plots.
plt.style.use("caf.viz.tfn")

# %%
# Define an example Polygon roughly outlining North England.
boundary = gpd.GeoDataFrame(
    {"Name": ["North England"]},
    geometry=[
        geometry.Polygon(
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
    ],
    crs="EPSG:27700",
)

# %%
# Generate random points to use as sample data.
rng = np.random.default_rng(1684187)
size = 100
points = gpd.GeoDataFrame(
    {"Value": rng.random(size) * 100},
    geometry=boundary.sample_points(size, rng=rng).explode(ignore_index=True),
)
points.head()

# %%
# Create a dictionary of layers ready to be mapped, where the key is used for the map legend.
# The :class:`LayerData` objects contain the geometry data to be mapped and styling options.
layers: dict[str, mapping.LayerData] = {}
layers["Boundary"] = mapping.LayerData(
    boundary, mapping.LayerOptions(edgecolor="red", facecolor="#ffffff00", size=2)
)
layers["Points"] = mapping.LayerData(points, mapping.LayerOptions(color="blue", size=3))


# %%
# Create the map with OpenStreetMap background.
fig = mapping.simple_map(layers, "Example Map of Points")


# %%
# Explicitly closing all figures.
plt.close("all")
