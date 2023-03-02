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

# Third Party
import geopandas as gpd
import contextily as cx
from dataclasses import dataclass
from fiona import collection
from caf.toolkit.config_base import BaseConfig
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
from pydantic import validator

from adjustText import adjust_text
import pandas as pd

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
class Layer(BaseConfig):
    """
    Base class for storing info about layers to be plotted.

    Parameters
    ----------
    path(Path): Path to the shapefile to be plotted.
    zorder(int): Zorder parameter determining order of layers to plot. Lower
        numbers are plotted below higher numbers.
    name(str): The name of the layer in the plot legend. '_nolegend_' will
        prevent a legend item being added for this layer.
    label(str): The name of the column from the shapefile to be used for
        plot annotations.
    alpha(float): Alpha of layer plot i.e. transparency. 0 is transparent and
        1 is opaque.
    linewidth(float): Linewidth of linefeatures being plotted. Defaults to 1.
    marker(str): Choose marker for point features. See matplotlib documentation
        for options here.
    markersize(int): Size of point feature markers.
    clip(bool): Choose whether the layer should be clipped to the TfN boundary.
        Defaults to True.
    """

    path: Path
    zorder: int
    name: str
    label: str = None
    alpha: float = 1
    linewidth: float = 1
    marker: str = "o"
    markersize: int = 4
    clip: bool = True

    @validator("path")
    def path_exists(cls, v):
        if os.path.isfile(v) is False:
            raise ValueError(
                f"The path provided for {v} does not exist."
                "If this path is on a network drive make sure you are connected"
            )
        return v

    @validator("label")
    def column_exists(cls, v, values):
        if v is not None:
            with collection(values["path"]) as source:
                schema = source.schema
                if v not in schema["properties"].keys():
                    raise ValueError(
                        f"The label provided, {v}, does not appear in the "
                        f"chosen shapefile."
                    )
        return v


class MonoLayer(Layer):
    """
    Class for parameters for plotting simple layers. These are layers where all
    features are the same colour. Inherits from 'Layer'.

    Parameters
    ----------
    colour(str): The colour for the features.
    """

    colour: str = None


class CatLayer(Layer):
    """
    Class for parameters for plotting categorical layers. These are layers with
    colours decided by simple categories in shapefile columns. Inherits from
    'Layer'.

    Parameters
    ----------
    column(str): The column in the layer for the plot to be categorised by.
    colours(list[str]): The colours you want for the layer. The number of
        colours must equal the number of unique values in the chosen column.
        There is no validator check for this as it would need to load in the
        entire shapefile.
    """

    column: str = None
    colours: list[str] = None


class HeatmapLayer(Layer):
    """
    Class for parameters for plotting heatmap layers. These will use custom
    colourmaps of user defined colours.

    Parameters
    ----------
    colours(str): Colours to go into the colour map. Usually two colours.
    column(str): Column for the heatmap to be made from.
    colour_labels(list[str]): Labels to go with colours in the plot legend.
        The order of these will map to the order of the colours.
    """

    colours: list[str] = None
    column: str = None
    colour_labels: list[str] = None

    @validator("colour_labels")
    def lengths_match(cls, v, values):
        if len(v) != len(values["colours"]):
            raise ValueError("The lengths of 'colours' and 'colour_labels' must be the same.")


def read_layer(clip: bool, crs: str, path: Path, bound: gpd.GeoDataFrame):
    layer = gpd.read_file(path)
    if layer.crs is None:
        layer.set_crs(crs, inplace=True)
    else:
        layer.to_crs(crs, inplace=True)
    if clip:
        layer = layer.clip(bound)
    return layer


def heatmap_plot(
    attributes: HeatmapLayer, crs: str, bound: gpd.GeoDataFrame, custom_handles: list, ax, fig
):
    layer = read_layer(attributes.clip, crs, attributes.path, bound)
    cmap = LinearSegmentedColormap.from_list("", attributes.colours)
    layer.plot(
        ax=ax,
        label=attributes.name,
        column=attributes.column,
        cmap=cmap,
        zorder=attributes.zorder,
        alpha=attributes.alpha,
        marker=attributes.marker,
        markersize=attributes.markersize,
    )
    handles = [
        mpatches.Patch(color=i, label=j)
        for i, j in zip(attributes.colours, attributes.colour_labels)
    ]
    custom_handles += handles
    return custom_handles


def mono_plot(
    attributes: MonoLayer,
    crs: str,
    bound: gpd.GeoDataFrame,
    custom_handles: list,
    ax,
    fig,
    all_texts: list[tuple[float, float, str]],
):
    layer = read_layer(attributes.clip, crs, attributes.path, bound)
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
    if "Polygon" in layer.geom_type.to_list():
        handle = mpatches.Patch(color=attributes.colour, label=attributes.name)
        custom_handles.append(handle)
    if attributes.label:
        texts = [
            (x, y, label)
            for x, y, label in zip(
                layer.geometry.centroid.x,
                layer.geometry.centroid.y,
                layer[attributes.label],
            )
            if pd.isnull(label) == False
        ]
        all_texts += texts
    return custom_handles, all_texts


def cat_plot(
    attributes: CatLayer,
    crs: str,
    bound: gpd.GeoDataFrame,
    custom_handles: list,
    ax,
    fig,
    all_texts: list[tuple[float, float, str]],
):
    layer = read_layer(attributes.clip, crs, attributes.path, bound)
    for i, j in zip(layer[attributes.column].unique(), attributes.colours):
        gdf = layer[layer[attributes.column] == i]
        gdf.plot(
            ax=ax,
            label=i,
            color=j,
            zorder=attributes.zorder,
            alpha=attributes.alpha,
            linewidth=attributes.linewidth,
            marker=attributes.marker,
            markersize=attributes.markersize,
        )
        if "Polygon" in gdf.geom_type.to_list():
            handle = mpatches.Patch(color=j, label=i)
            custom_handles.append(handle)
    if attributes.label:
        texts = [
            (x, y, label)
            for x, y, label in zip(
                layer.geometry.centroid.x,
                layer.geometry.centroid.y,
                layer[attributes.label],
            )
            if pd.isnull(label) == False
        ]
        all_texts += texts
    return custom_handles, all_texts


# # # FUNCTIONS # # #
def plotter(
    crs="EPSG:27700",
    mono_layers: list[MonoLayer] = [],
    cat_layers: list[CatLayer] = [],
    heatmap_layers: list[HeatmapLayer] = [],
    legend_title: str = None,
):
    custom_handles = []
    all_texts = []
    fig, ax = plt.subplots(figsize=(10, 10))
    bound = gpd.read_file(tfn_boundary, crs=crs)
    bound.plot(ax=ax, color=GREY)
    # cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, crs=crs)
    for attributes in heatmap_layers:
        custom_handles = heatmap_plot(attributes, crs, bound, custom_handles, ax, fig)
    for attributes in cat_layers:
        custom_handles, all_texts = cat_plot(
            attributes, crs, bound, custom_handles, ax, fig, all_texts
        )
    for attributes in mono_layers:
        custom_handles, all_texts = mono_plot(
            attributes, crs, bound, custom_handles, ax, fig, all_texts
        )
    handles, _ = ax.get_legend_handles_labels()
    custom_handles += handles
    if legend_title:
        legend = ax.legend(handles=custom_handles, title=legend_title)
    else:
        legend = ax.legend(handles=custom_handles)
    bbox = legend.get_bbox_to_anchor().transformed(fig.transFigure)
    all_texts.append((320000, 330000, "Contains OS data © Crown copyright 2022"))
    final_texts = [ax.text(i[0], i[1], i[2]) for i in all_texts]
    adjust_text(final_texts, objects=[bbox])
    ax.axis("off")
    print("debugging")
    return fig, ax


if __name__ == "__main__":
    mrn = MonoLayer(
        name="Major Road Network",
        path=Path(r"I:\Data\MPD_MRN\MRN shapefile\mrnpaths.shp"),
        zorder=1,
        colour=PURPLE,
    )
    srn = MonoLayer(
        name="Strategic Road Network",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\UK SRN 2016\UK SRN 2016\UK Strategic Road Network (20160929).shp"
        ),
        zorder=2,
        colour=NAVY,
        linewidth=1.5,
    )
    eco_centres = CatLayer(
        name="Economic Centres",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Economics\Economics\Key Economic Centres.shp"
        ),
        zorder=3,
        column="Timeframe",
        colours=[GREEN, WHITE],
    )
    ent_zones = MonoLayer(
        name="Enterprise Zones",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Economics\Economics\Enterprise Zones.shp"
        ),
        zorder=2,
        colour=ORANGE,
    )
    cities = MonoLayer(
        name="_nolegend_",
        label="City",
        path=Path(r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\STP\STP\TfN Cities.shp"),
        zorder=10,
        markersize=0,
        colour=GREY,
    )
    ports = MonoLayer(
        name="TfN Ports",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Freight\Freight\Ports (TfN).shp"
        ),
        zorder=3,
        marker="D",
        colour=TEAL,
        markersize=5,
    )
    airports = MonoLayer(
        name="TfN Airports",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Freight\Freight\Airports (TfN).shp"
        ),
        zorder=3,
        marker="s",
        colour=PINK,
        markersize=5,
    )
    SDC = CatLayer(
        name="Strategic Development Corridors",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP Shapes\Development Corridors (2)\Development Corridors.shp"
        ),
        zorder=2,
        colours=[NAVY, TEAL, PINK],
        alpha=0.5,
        column="Mode",
        label="Letter",
        clip=False,
    )
    # UNESCO = MonoLayer(
    #     name="UNESCO Sites",
    #     path=Path(r"E:\stp\world_heritage_comb.shp"),
    #     colour=PURPLE,
    #     zorder=5,
    #     alpha=0.5,
    #     label="Name",
    #     clip=False,
    # )
    parks = MonoLayer(
        name="National Parks",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP2\Visitor economy\Boundary Layers\National_Parks_England.shp"
        ),
        colour=TEAL,
        label="name",
        clip=False,
        zorder=2,
    )
    AONB = MonoLayer(
        name="Areas of Outstanding\nNatural Beauty England",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP2\Visitor economy\Boundary Layers\Areas_of_Outstanding_Natural_Beauty__England____Natural_England.shp"
        ),
        colour=PINK,
        clip=False,
        label="NAME",
        zorder=3,
    )
    no_data = MonoLayer(
        name="Other Significant\nVisitor Attractions",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP2\Visitor economy\Visitor Attractions Shapefiles\Attractions_with_no_data.shp"
        ),
        colour=NAVY,
        zorder=6,
        label="id",
    )
    free = MonoLayer(
        name="Free Attractions\n(>200,000 Annual Visitors)",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP2\Visitor economy\Workspace\Free_Attractions_Greater_Than_200000.shp"
        ),
        colour=PINK,
        zorder=6,
        label="id",
    )
    paid = MonoLayer(
        name="Paid Attractions\n(>200,000 Annual Visitors)",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP2\Visitor economy\Workspace\Paid_Attractions_Greater_Than_200000.shp"
        ),
        colour=TEAL,
        zorder=7,
        label="id",
    )
    coastal = MonoLayer(
        name="Major Coastal Destinations",
        path=Path(
            r"Y:\Data Strategy\GIS Shapefiles\STP2\Visitor economy\Workspace\Coastal_Tourist_Destinations.shp"
        ),
        colour=PURPLE,
        zorder=8,
        label="Place Name",
    )
    warehouses = HeatmapLayer(
        name="Warehouse Density",
        path=Path(r"Y:\Freight\13 Freight Analysis\NDR Business Floorspace\QGIS\LAD_NDR.shp"),
        colours=[YELLOW, DARK_GREEN],
        column="density",
        zorder=1,
        colour_labels=["Low", "High"],
    )
    fig, ax = plotter(cat_layers=[SDC],mono_layers=[cities], crs="EPSG:27700")
    fig.savefig("sdc.png")
    # fig, ax = plotter(
    #     mono_layers=[cities, mrn, srn, ent_zones, ports, airports],
    #     cat_layers=[eco_centres],
    #     crs="EPSG:27700",
    # )
    # fig.savefig("mrn.png")
    # fig, ax = plotter(mono_layers=[UNESCO, AONB, parks, paid, free, no_data, coastal], crs="EPSG:27700")
    # fig.savefig("tourism.png")
    # fig, ax = plotter(
    #     heatmap_layers=[warehouses],
    #     mono_layers=[mrn, airports, ports, cities],
    #     legend_title="Warehouse Density",
    # )
    # fig.savefig("freight.png")
    print("debugging")
