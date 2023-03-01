# -*- coding: utf-8 -*-
"""
Created on: 27/02/2023
Updated on:

Original author: Isaac Scott
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
from pathlib import Path
from typing import Optional
from PIL import Image

# Third Party
import geopandas as gpd
import contextily as cx
from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches, lines
from adjustText import adjust_text

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
NAVY = "#0d0f0d"
TEAL = "#00dec6"
PURPLE = "#7317de"
PINK = "#e50080"
WHITE = "#ffffff"
ORANGE = "#f15a29"
GREY = "#5d898c"
ICE = "66ffff"
YELLOW = "#e9e623"
DARK_GREEN = "#1f3529"
GREEN = "#a0ca2a"

PRIMARY = (NAVY, TEAL, PURPLE, PINK, WHITE)
NPR = (TEAL, NAVY)
STRATEGIC_RAIL = (PINK, NAVY)
INT = (TEAL, NAVY)
SMART = (ORANGE, GREY)
MRN = (PURPLE, GREY)
FREIGHT = (YELLOW, DARK_GREEN)
NIP = (GREEN, DARK_GREEN)

tfn_boundary = Path(
    r"Y:\Data Strategy\GIS Shapefiles\TfN Boundary\Transport_for_the_north_boundary_2020_generalised.shp"
)
tfn_logo_large = Path(r"C:\Users\IsaacScott\Projects\caf.viz\src\caf\TFN_logo.png")


# # # CLASSES # # #
@dataclass
class Layer:
    path: Path
    zorder: int
    name: str
    label: str = None
    alpha: float = 1
    linewidth: float = 1
    marker: str = "o"
    markersize: int = 4


@dataclass
class MonoLayer(Layer):
    colour: str = None


@dataclass
class CatLayer(Layer):
    column: str = None
    colours: list[str] = None


# # # FUNCTIONS # # #
def plotter(mono_layers: list[MonoLayer], cat_layers: list[CatLayer], crs):
    texts=None
    fig, ax = plt.subplots(figsize=(10,10))
    bound = gpd.read_file(tfn_boundary)
    # logo = plt.imread(tfn_logo_large)
    bound.plot(ax=ax, color=GREY)
    # newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor="N", zorder=5)
    # newax.axis("off")
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, crs=crs)
    for attributes in cat_layers:
        layer = gpd.read_file(attributes.path).to_crs(crs)
        layer = gpd.clip(layer, bound)
        for i, j in zip(layer[attributes.column].unique(), attributes.colours):
            gdf = layer[layer[attributes.column] == i]
            gdf.plot(
                ax=ax,
                label=f"{i} {attributes.name}",
                color=j,
                zorder=attributes.zorder,
                alpha=attributes.alpha,
                linewidth=attributes.linewidth,
                marker=attributes.marker,
                markersize=attributes.markersize
            )

    for attributes in mono_layers:
        layer = gpd.read_file(attributes.path).to_crs(crs)
        layer = gpd.clip(layer, bound)
        if attributes.label:
            layer.plot(
                ax=ax,
                color=attributes.colour,
                label=attributes.name,
                zorder=attributes.zorder,
                alpha=attributes.alpha,
                linewidth=attributes.linewidth,
                marker=attributes.marker,
                markersize=attributes.markersize,
            )
            texts = [ax.text(x, y, label) for x, y, label in zip(layer.geometry.x, layer.geometry.y, layer[attributes.label])]


        else:
            layer.plot(
                ax=ax,
                label=attributes.name,
                color=attributes.colour,
                zorder=attributes.zorder,
                alpha=attributes.alpha,
                linewidth=attributes.linewidth,
                marker=attributes.marker,
                markersize=attributes.markersize,
            )

    ax.axis("off")
    ax.legend()
    print("debugging")
    return fig, ax, texts


if __name__ == "__main__":
    mrn = MonoLayer(
        name="Major Road Network",
        path=Path(r"I:\Data\MPD_MRN\MRN shapefile\mrnpaths.shp"), zorder=1, colour=PURPLE
    )
    srn = MonoLayer(
        name="Strategic Road Network",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\UK SRN 2016\UK SRN 2016\UK Strategic Road Network (20160929).shp"
        ),
        zorder=2,
        colour=NAVY,
        linewidth=1.5
    )
    eco_centres = CatLayer(
        name="Economic Centres",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Economics\Economics\Key Economic Centres.shp"
        ),
        zorder=3,
        column="Timeframe",
        colours=[TEAL, WHITE]
    )
    ent_zones = MonoLayer(
        name="Enterprise Zones",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Economics\Economics\Enterprise Zones.shp"
        ),
        zorder=2,
        colour=PINK,
    )
    cities = MonoLayer(
        name='_nolegend_',
        label='City',
        path=Path(r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\STP\STP\TfN Cities.shp"),
        zorder=10,
        markersize=0,
        colour=GREY
    )
    ports = MonoLayer(
        name='TfN Ports',
        path=Path(r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Freight\Freight\Ports (TfN).shp"),
        zorder=3,
        marker="D",
        colour=NAVY
    )
    airports = MonoLayer(
        name='TfN Airports',
        path=Path(r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Freight\Freight\Airports (TfN).shp"),
        zorder=3,
        marker="s",
        colour=NAVY
    )


    fig, ax, texts = plotter(mono_layers=[cities, mrn, srn, ent_zones,ports, airports], cat_layers=[eco_centres], crs="EPSG:27700")
    adjust_text(texts)
    fig.savefig('mrn.png')
    print('debugging')
