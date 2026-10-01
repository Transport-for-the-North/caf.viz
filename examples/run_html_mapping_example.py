"""
HTML mapping example
====================

This is a code example showing how to create an interactive map using the
:mod:`caf.viz.web.mapping` module.
This example shows two ways to create an interactive html map:

- A single map with the desired datasets, and
- A split map consisting of an overview map with the split geometries which link to
  individual maps showing the datasets for each split geometry.

The example uses UK boundaries from Office National Statistics (ONS).
"""

# %%
# Imports
# -------
import os
import pathlib

import geopandas as gpd
import numpy as np
from shapely import geometry

from caf.viz import _datasets
from caf.viz.web import mapping

# %%
# Get data
# --------
# Load the UK Upper-tier/unitary authorities to use as example polygons.
authorities, attr = _datasets.fetch_dataset(_datasets.Datasets.ONS_UTLA)
YEAR = 2025
mask = (authorities["start"].isna() | (authorities["start"] <= YEAR)) & (
    authorities["end"].isna() | (authorities["end"] > YEAR)
)
authorities = authorities.loc[mask]
print(
    f"Loaded dataset with {len(authorities):,} rows and"
    f" {len(authorities.columns):,} columns\n{attr}"
)

# %%
# Prepare Dataset
# ---------------
# Prepare dataset for mapping as a :class:`~caf.viz.web.mapping.MapData` object, which
# includes the data, the color column to use, and various mapping options in a
# :class:`~caf.viz.web.mapping.ExploreOptions` object.
mapping_datasets = {
    "Authorities": mapping.MapData(
        data=authorities.to_crs(f"EPSG:{mapping.MAP_CRS_EPSG}"),
        color_column="areacd",
        options=mapping.ExploreOptions(tooltip=["areanm", "areacd"], show_legend=False),
    )
}

# %%
# Map a Single Layer
# ------------------
# :func:`~caf.viz.web.mapping.map_datasets` will create a :class:`folium.Map` object from the
# datasets with OpenStreetMap background, it can be saved to a standalone HTML file
# with :meth:`folium.Map.save`.
mapping.map_datasets(datasets=mapping_datasets)

# %%
# Polygon Mask
# ------------
# Use a custom :class:`shapely.geometry.Polygon` (or another :class:`~geopandas.GeoDataFrame`)
# to filter the map layer(s).
boundary = geometry.Polygon(
    [
        [-3.260189, 53.360387],
        [-3.221733, 54.023903],
        [-3.776543, 54.524274],
        [-3.353580, 54.955543],
        [-2.029728, 55.930742],
        [-1.573794, 55.584558],
        [-1.128841, 54.680185],
        [-0.524591, 54.505141],
        [-0.046691, 54.123826],
        [0.256121, 53.541937],
        [-1.343080, 53.291492],
        [-2.875676, 53.189576],
    ]
)
mapping.map_datasets(mapping_datasets, mask=boundary, mask_name="North England")

# %%
# Multiple Layers
# ---------------
# Multiple datasets can be passed to :func:`~caf.viz.web.mapping.map_datasets` and they
# will appear as separate layers in the map which can be toggled on and off.
#
# .. note::
#   This example generates some random points for plotting.

points = authorities.sample_points(20).explode().to_frame(name="geometry")
points = points.join(authorities["areanm"])

rng = np.random.default_rng()
points["value"] = rng.random(len(points)) * 100

# %%
# Add the random points to the map datasets. Adding lots of data to a single
# map can make them unclear and slow to load, see :ref:`Create a Split Map`
# below for how to split the map into sections to show more details.
mapping_datasets["Points"] = mapping.MapData(
    points.to_crs(epsg=mapping.MAP_CRS_EPSG),
    "value",
    options=mapping.ExploreOptions(tooltip=["areanm", "value"]),
)

mapping.map_datasets(mapping_datasets)

# %%
# Create a Split Map
# ------------------
# Create a map which is split across multiple HTML files to allow for more detailed
# data on each subset map, see `Split Map Overview <split_map.html>`_.
#
# .. note::
#     HTML file is written to generated documentation folder, so it's deployed with
#     documentation, normally this can be saved anywhere.

readthedocs_path = os.getenv("READTHEDOCS_OUTPUT")
subpath = "html/_generated/examples/split_map.html"
if readthedocs_path is None:
    split_map_path = pathlib.Path("../docs/build") / subpath
else:
    split_map_path = pathlib.Path(readthedocs_path) / subpath
split_map_path.parent.mkdir(exist_ok=True, parents=True)

# %%
# Use the data loaded previously to product the set of maps. An overview map with the split
# data is saved to the ``output_path``, the lower level HTML files with the more detailed
# data are saved in a "Split Maps" sub-folder and linked to from the overview.

mapping_datasets.pop("Authorities")

mapping.produce_map_set(
    output_path=split_map_path,
    datasets=mapping_datasets,
    split=authorities,
    split_name_column="areanm",
    filter_zone_gpd=gpd.GeoSeries([boundary]).set_crs(epsg=4326),
)
