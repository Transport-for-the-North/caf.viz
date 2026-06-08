"""
Example
===========

This is a code example which is displayed but **not** run.

"""

# %%
# IMPORTS #
import pathlib

import geopandas as gpd
from geodatasets import get_path

from caf.viz.web import mapping

# %%
# CONSTANTS #
COUNTRY_CODE = 0
REGION_CODE = 3

# PATHS
PATH_TO_SAVE_MAP = pathlib.Path(r"map.html")
PATH_TO_SAVE_SPLIT_MAP = pathlib.Path(r"split_map.html")

# %%
# DATA #
path_to_data = get_path("eurostat.nuts_rg_10m_2024_3035")
europe = gpd.read_file(path_to_data)

path_to_data = get_path("naturalearth.cities")
cities = gpd.read_file(path_to_data)

europe_countries = europe[europe["LEVL_CODE"] == COUNTRY_CODE]

# %%
# PREPARE DATA #
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
# CREATE MAP (single map) #
m = mapping.map_datasets(datasets=mapping_datasets, mask=filter_zones, mask_name="Europe")


# %%
# SAVE MAP (OPTIONAL) #
m.save(PATH_TO_SAVE_MAP)

# %%
# CREATE MAP (split map) #
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

# Check map crs
if europe_countries.crs != mapping.MAP_CRS_EPSG:
    europe_countries = europe_countries.to_crs(f"EPSG:{mapping.MAP_CRS_EPSG}")

split_m = mapping.produce_map_set(
    output_path=PATH_TO_SAVE_SPLIT_MAP,
    datasets=mapping_datasets,
    split=europe_countries,
    split_name_column="NAME_ENGL",
)
