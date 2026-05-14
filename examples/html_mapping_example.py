"""
HTML mapping example
=======

This is a code example which is displayed but **not** run.

"""

# %%
import geopandas as gpd
from geodatasets import get_path

path_to_data = get_path("eurostat.nuts_rg_10m_2024_3035")
europe = gpd.read_file(path_to_data)
path_to_data = get_path("naturalearth.cities")
cities = gpd.read_file(path_to_data)


# %%
cities

# %%
europe

# %%
from caf.viz.web import mapping

## Prepare data for mapping ##
datasets = {
    "Cities": cities
}

color_column = {
    "Cities": "natscale"
}

tooltip = {
    "Cities": ["name", "natscale"]
}

mapping_datasets = {}
for name, data in datasets.items():
    mapping_datasets[name] = mapping.MapData(
        data=data,
        color_column=color_column[name],
        options=mapping.ExploreOptions(
            tooltip=tooltip[name],
            show_legend=True
        )
    )

# %%
if europe.crs != mapping.MAP_CRS_EPSG:
    filter_zones = europe.to_crs(f"EPSG:{mapping.MAP_CRS_EPSG}")[["NAME_ENGL", "geometry"]]
else:
    filter_zones = europe[["NAME_ENGL", "geometry"]]
filter_polygon = filter_zones.dissolve(by="NAME_ENGL")

m = mapping.map_datasets(
    datasets=mapping_datasets,
    mask=filter_polygon.geometry,
    mask_name="NAME_ENGL"
)

m

m.save("path/to/save/map.html")