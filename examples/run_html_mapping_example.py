"""
HTML mapping example
====================

This is a code example showing how to create an interactive map using the :mod:`caf.viz.web.mapping`
module. This example shows two ways to create an interactive html map:

- A single map with the desired datasets, and
- A split map consisting of an overview map with the split geometries which link to
  individual maps showing the datasets for each split geometry.

The example uses the NUTS dataset from Eurostat and the cities dataset from Natural Earth,
both of which are available in geodatasets.

"""

# %%
# Imports
# -------
import os
import pathlib

import geopandas as gpd
from geodatasets import get_path

from caf.viz.web import mapping

# %%
# Define constants
# ----------------
# These are constants used to select the desired data from the geodatasets package.
COUNTRY_CODE = 0
REGION_CODE = 3

# %%
# Load data
# ---------
# Load the datasets to be mapped as :class:`geopandas.GeoDataFrame` objects.
path_to_data = get_path("eurostat.nuts_rg_10m_2024_3035")
europe = gpd.read_file(path_to_data)

path_to_data = get_path("naturalearth.cities")
cities = gpd.read_file(path_to_data)

europe_countries = europe[europe["LEVL_CODE"] == COUNTRY_CODE]

# %%
# Prepare datasets for mapping
# ----------------------------
# Prepare datasets for mapping as a :class:`~caf.viz.web.mapping.MapData` object, which
# includes the data, the color column to use, and various mapping options in a
# :class:`~caf.viz.web.mapping.ExploreOptions` object.
datasets = {"Countries": europe_countries, "Cities": cities}

color_column = {"Countries": None, "Cities": "natscale"}

tooltip = {"Countries": ["NUTS_NAME", "NAME_ENGL", "CAPT"], "Cities": ["name", "natscale"]}

options = {
    "Countries": mapping.ExploreOptions(
        tooltip=tooltip["Countries"],
        show_legend=False,
        style={"fillOpacity": 0.4, "fillColor": "grey", "color": "black"},
    ),
    "Cities": mapping.ExploreOptions(tooltip=tooltip["Cities"], show_legend=True),
}

mapping_datasets = {}
for name, data in datasets.items():
    mapping_datasets[name] = mapping.MapData(
        data=data.to_crs(f"EPSG:{mapping.MAP_CRS_EPSG}"),
        color_column=color_column[name],
        options=options[name],
    )

# Prepare mask
if europe_countries.crs != mapping.MAP_CRS_EPSG:
    filter_zones = europe_countries.to_crs(f"EPSG:{mapping.MAP_CRS_EPSG}")
else:
    filter_zones = europe_countries

# %%
# Create a Single Map
# -------------------
# :func:`~caf.viz.web.mapping.map_datasets` will create a :class:`folium.Map` object from the
# datasets with OpenStreetMap background, it can be saved to a standalone HTML file
# with :meth:`folium.Map.save`.
mapping.map_datasets(datasets=mapping_datasets, mask=filter_zones, mask_name="Europe")

# %%
# Create a Split Map
# ------------------
# Create a map which is split across multiple HTML files to allow for more detailed
# data on each subset map, see `Split Map Overview <split_map.html>`_.
#
# .. note::
#     HTML file is written to generated documentation folder, so it's deployed with
#     documentation, normally this can be saved anywhere.

READTHEDOCS_OUTPUT = os.getenv("READTHEDOCS_OUTPUT")
if READTHEDOCS_OUTPUT is None:
    PATH_TO_SAVE_SPLIT_MAP = pathlib.Path("split_map.html")
else:
    PATH_TO_SAVE_SPLIT_MAP = (
        pathlib.Path(READTHEDOCS_OUTPUT) / "html/_generated/examples/split_map.html"
    )
PATH_TO_SAVE_SPLIT_MAP.parent.mkdir(exist_ok=True, parents=True)

europe_regions = europe[europe["LEVL_CODE"] == REGION_CODE]

mapping_datasets = {
    "Regions": mapping.MapData(
        data=europe_regions.to_crs(f"EPSG:{mapping.MAP_CRS_EPSG}"),
        color_column="NAME_ENGL",
        options=mapping.ExploreOptions(
            tooltip=["NUTS_NAME", "NAME_ENGL", "CAPT"],
            show_legend=True,
            style={"fillOpacity": 0.4, "fillColor": "grey", "color": "black"},
        ),
    )
}

# Check map crs for split geometries
if europe_countries.crs != mapping.MAP_CRS_EPSG:
    europe_countries = europe_countries.to_crs(f"EPSG:{mapping.MAP_CRS_EPSG}")

mapping.produce_map_set(
    output_path=PATH_TO_SAVE_SPLIT_MAP,
    datasets=mapping_datasets,
    split=europe_countries,
    split_name_column="NAME_ENGL",
)
