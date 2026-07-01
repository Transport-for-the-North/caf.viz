"""Functionality for creating html maps from spatial datasets."""

import dataclasses
import logging
import pathlib
import re
import warnings
from collections.abc import Mapping
from typing import NamedTuple, Self

import folium
import geopandas as gpd
import pandas as pd
import tqdm
from branca.element import Element, MacroElement, Template
from shapely import geometry

##### CONSTANTS #####

LOG = logging.getLogger(__name__)

MAP_CRS_EPSG = 4326

# Textbox variables
TITLE = "How to use the map"
TEXT_SPLIT = """
    <p>Click on a subset area, then click the link that pops up to view the data for that area
     (Ctrl click to open in a new tab).</p>
    <p>You can turn layers on/off in the right-hand corner
    and you can drag the legend boxes to another position on the map.</p>
    """
TEXT_NOSPLIT = """
    <p>You can turn layers on/off in the right-hand corner
    and you can drag the legend boxes to another position on the map.</p>
    """
HEAD = """
    {% macro header(this, kwargs) %}
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

    <script>
        $( function() {
            // Toggle collapse/expand on button click
            $( "#textbox-toggle" ).click( function() {
                $( "#textbox-content" ).slideToggle( 300 );
                $( this ).find( "i" ).toggleClass( "fa-chevron-up fa-chevron-down" );
            });
        });
        </script>

    <style type='text/css'>
    .textbox {
        position: absolute;
        z-index:9999;
        border-radius:4px;
        background: rgba( 28, 25, 56, 0.25 );
        box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
        backdrop-filter: blur( 4px );
        -webkit-backdrop-filter: blur( 4px );
        border: 4px solid rgba( 215, 164, 93, 0.2 );
        padding: 10px;
        font-size:12px;
        left: 20px;
        bottom: 20px;
        color: black;
    }
    .textbox .textbox-title {
        color: black;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 20px;
    }
    .textbox-toggle-btn {
        background: none;
        border: none;
        color: black;
        cursor: pointer;
        font-size: 16px;
        padding: 0;
        display: flex;
        align-items: center;
    }
    .textbox-toggle-btn:hover {
        opacity: 0.7;
    }
    .textbox-content {
        transition: all 0.3s ease;
    }
    /* Continuous colorbar (Branca) */
    .legend.leaflet-control:not(:empty) {
        background: white;
        padding: 8px 10px;
        border: 1px solid #999;
        border-radius: 4px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }
    </style>
    {% endmacro %}
    """

##### CLASSES & FUNCTIONS #####


class Bounds(NamedTuple):
    """Bounding box coordinates."""

    min_x: int | float
    min_y: int | float
    max_x: int | float
    max_y: int | float

    def __add__(self, value: object) -> Self:
        """Create a bounding box which contain both."""
        if not isinstance(value, self.__class__):
            raise TypeError(f"value should be {self.__class__} not {type(value)}")

        return self.__class__(
            min_x=min(self.min_x, value.min_x),
            min_y=min(self.min_y, value.min_y),
            max_x=max(self.max_x, value.max_x),
            max_y=max(self.max_y, value.max_y),
        )


@dataclasses.dataclass
class ExploreOptions:
    """Options for MapData."""

    show_legend: bool = True
    tooltip: bool | list[str] = False
    popup: bool | str | list[str] = False
    style: dict = dataclasses.field(default_factory=dict)
    highlight_style: dict = dataclasses.field(default_factory=dict)
    legend_title: str | None = None
    cmap: str | list[str] = "viridis"
    show: bool = True
    categorical: bool | None = None
    """Whether the legend should be categorical or not."""

    def infer_categorical(self, data: pd.Series | None) -> bool:
        """Infer categorical value if it isn't given.

        Returns
        -------
        bool
            - Attribute value if not None; or
            - False if `data` is None or dtype of `data`
              is numeric; or
            - True for any other dtypes (including boolean).
        """
        if self.categorical is not None:
            return self.categorical
        if data is None:
            return False

        # is_numeric_dtype returns true for bool so checking that first
        if pd.api.types.is_bool_dtype(data):
            return True
        return not pd.api.types.is_numeric_dtype(data)


@dataclasses.dataclass()
class MapData:
    """Data to plot on a map."""

    data: gpd.GeoDataFrame
    color_column: str | None = None
    to_filter: bool = True
    options: ExploreOptions = dataclasses.field(default_factory=lambda: ExploreOptions())  # noqa: PLW0108 # pylint: disable=W0108


def _clean_filename(name: str, replacement: str = "_") -> str:
    # replace characters not allowed in filenames
    cleaned = re.sub(r'[<>:"\'.,/\\|?*]', replacement, name)
    # collapse repeated underscores/spaces, trim edges
    cleaned = re.sub(r"[_\s]+", "_", cleaned).strip("_")
    return cleaned[:255]  # keep it under common filesystem limits


def _explore(
    map_: folium.Map,
    data: gpd.GeoDataFrame,
    data_column: str | None,
    name: str,
    *,
    options: ExploreOptions,
) -> None:
    if options.legend_title is None and data_column is not None:
        legend = {"caption": f"{name} - {data_column.replace('_', ' ')}"}
    elif options.legend_title is not None and data_column is not None:
        legend = {"caption": options.legend_title}
    else:
        legend = {}

    data.explore(
        data_column,
        categorical=options.infer_categorical(
            None if data_column is None else data[data_column]
        ),
        cmap=options.cmap,
        legend=options.show_legend,
        m=map_,
        tooltip=options.tooltip,
        popup=options.popup,
        tiles=None,
        name=name,
        show=options.show,
        legend_kwds=legend,
        popup_kwds={"labels": False},
        style_kwds=options.style,
        highlight_kwds=options.highlight_style,
    )


def check_output_path(output_path: pathlib.Path) -> pathlib.Path:
    """Check that the output path is valid and adjust if necessary."""
    if output_path.is_dir():
        output_path = output_path / "Map.html"
    elif output_path.suffix.lower() != ".html":
        output_path = output_path.with_suffix(".html")
    return output_path


def check_mask(mask: gpd.GeoDataFrame | gpd.GeoSeries) -> geometry.Polygon:
    """Check that the mask is a single polygon and of the right CRS."""
    if mask.crs != f"EPSG:{MAP_CRS_EPSG}":
        mask = mask.to_crs(f"EPSG:{MAP_CRS_EPSG}")
    return mask.union_all()


def map_datasets(
    datasets: Mapping[str, MapData],
    mask: geometry.Polygon
    | geometry.MultiPolygon
    | gpd.GeoDataFrame
    | gpd.GeoSeries
    | None = None,
    mask_name: str | None = None,
    textbox_text: str = TEXT_NOSPLIT,
    output_path: pathlib.Path | None = None,
) -> pathlib.Path | folium.Map:
    """Produce single HTML map including all datasets.

    A single interactive map will be produced, filtered to the mask if provided.
    A textbox is included with instructions on how to use the map, which can be customised with
    the textbox_text parameter.

    Parameters
    ----------
    datasets : Mapping[str, MapData]
        Datasets must be provided as a dictionary of name to MapData.
        MapData includes the GeoDataFrame, an optional color column to plot, and extra options
         for plotting (ExploreOptions).
    mask : geometry.Polygon | geometry.MultiPolygon | gpd.GeoDataFrame | gpd.GeoSeries  | None,
      optional
        Mask to be used to filter data for mapping.
        Must be a single geometry object (Polygon or MultiPolygon) of the correct CRS
          (EPSG:4326).
        If it is a GeoDataFrame/GeoSeries, it will be unioned to create the mask geometry and
          CRS will be adjusted if necessary.
        By default None.
    mask_name : str | None, optional
        Name of the mask (filtering) layer.
        By default None.
    textbox_text : str, optional
        Text to go into the foldable textbox in bottom left of map.
        By default TEXT_NOSPLIT.
    output_path : pathlib.Path | None, optional
        Output path to write the HTML map.
        If a directory is provided, the map will be written to a file called "Map.html" in
          that directory.
        If None, the map object will be returned instead.

    Returns
    -------
    pathlib.Path | folium.Map
        Output path of the created HTML map or the map object itself.
    """
    # Check output path and mask
    if output_path is not None:
        output_path = check_output_path(output_path)
    if mask is not None and not isinstance(mask, (geometry.Polygon, geometry.MultiPolygon)):
        mask = check_mask(mask)

    map_ = folium.Map(tiles="OpenStreetMap", prefer_canvas=True)

    if mask is not None:
        folium.GeoJson(
            mask,
            name=mask_name,
            zoom_on_click=False,
            style_function=lambda _: {"color": "black", "fill": True, "fillOpacity": "0.2"},
        ).add_to(map_)

    for name, details in datasets.items():
        data = details.data.copy()
        if data.crs != MAP_CRS_EPSG:
            data = data.to_crs(MAP_CRS_EPSG)
        if mask is not None and details.to_filter:
            data = _filter_data(data, mask, name, mask_name)

            if len(data) == 0:
                warnings.warn(
                    f"{name} dataset contains 0 rows after filtering, "
                    f"filtered with {type(mask)} with bounds {mask.bounds}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

        _explore(map_, data, details.color_column, name, options=details.options)
        LOG.debug("Created %s layer with %s features", name, f"{len(data):,}")

    # Add CSS (on Header)
    macro = MacroElement()
    macro._template = Template(HEAD)  # pylint: disable=W0212
    map_.get_root().add_child(macro)

    body = f"""
        <div id='textbox' class='textbox'>
                <div class='textbox-title'>
                    {TITLE}
                    <button id="textbox-toggle" class="textbox-toggle-btn">
                        <i class="fas fa-chevron-up"></i>
                    </button>
                </div>
                <div id="textbox-content" class="textbox-content">
                    {textbox_text}
                </div>
            </div>"""

    # Add body
    body = Element(body, "textbox")
    map_.get_root().html.add_child(body)  # type: ignore[attr-defined]

    folium.LayerControl(collapsed=False).add_to(map_)

    # Fit map to bounds of mask or last dataset
    bounds = Bounds(*mask.bounds) if mask else Bounds(*data.union_all().bounds)
    map_.fit_bounds([[bounds.min_y, bounds.min_x], [bounds.max_y, bounds.max_x]])

    if output_path is None:
        return map_
    map_.save(output_path)
    LOG.debug("Written %s", output_path)
    return output_path


def _filter_data(
    data: gpd.GeoDataFrame,
    filter_: geometry.Polygon,
    name: str,
    filter_name: str | None,
) -> gpd.GeoDataFrame:
    """Filter data on a polygon mask."""
    before = len(data)
    if before != 0:
        data = data.loc[data.intersects(filter_)]
        LOG.debug(
            "Filtered %s with %s %s (%s) rows remaining",
            name,
            filter_name,
            f"{len(data):,}",
            f"{len(data) / before:.0%}",
        )
    return data


def _load_map_split(
    split: gpd.GeoDataFrame,
    split_name_column: str,
    filter_zone: gpd.GeoDataFrame | None = None,
) -> gpd.GeoSeries:
    """Load split geometries to split the maps on, filtered on a specific area if given."""
    if split.crs != f"EPSG:{MAP_CRS_EPSG}":
        split = split.to_crs(f"EPSG:{MAP_CRS_EPSG}")
    if filter_zone is not None:
        filter_zone = check_mask(filter_zone)
        split["centroid"] = split.geometry.centroid
        # filter the split zones to centroids to avoid including neighbouring zones
        split = split[split["centroid"].within(filter_zone)]

    if split_name_column not in split.columns:
        if split.index.name == split_name_column:
            split = split.reset_index()
        else:
            raise KeyError(
                f"split_name_column '{split_name_column}' not found in split columns "
                f"{list(split.columns)} "
                f"or index name '{split.index.name}'"
            )

    split = split[[split_name_column, split.geometry.name]]
    data = split.to_crs(MAP_CRS_EPSG)
    series = data.set_index(split_name_column).squeeze()

    if not isinstance(series, gpd.GeoSeries):
        raise TypeError(f"expected GeoSeries not {type(series)}")
    return series


def produce_map_set(
    output_path: pathlib.Path,
    datasets: dict[str, MapData],
    split: gpd.GeoDataFrame,
    split_name_column: str,
    filter_zone_gpd: gpd.GeoDataFrame | None = None,
) -> None:
    """Produce HTML maps for datasets, split into regions.

    A set of maps will be produced, one for each geometry in the split GeoDataFrame, filtered
     to the filter_zone_gpd if provided.
    The initial overview map will include links to the split maps, which will be stored in a
      separate folder.
    A textbox is included with instructions on how to use the map.

    Parameters
    ----------
    output_path : pathlib.Path
        Output path to write the HTML map.
        If a directory is provided, the overview map will be written to a file called
         "Overview Map.html" in that directory.
    datasets : dict[str, MapData]
        Datasets must be provided as a dictionary of name to MapData.
        MapData includes the GeoDataFrame, an optional color column to plot, and extra options
         for plotting (ExploreOptions).
    split : gpd.GeoDataFrame
        GeoDataFrame containing the geometries to split the map into.
    split_name_column : str
        Name of the column containing the names of the split geometries.
    filter_zone_gpd : gpd.GeoDataFrame | None, optional
        GeoDataFrame or GeoSeries containing the geometry to filter the split geometries.
    """
    if output_path.is_dir():
        output_path = output_path / "Overview Map.html"

    split_folder = pathlib.Path(output_path.parent, "Split Maps")
    split_folder.mkdir(exist_ok=True)
    split["split_zone"] = split[split_name_column].map(_clean_filename)
    split_geom = _load_map_split(
        split=split, split_name_column="split_zone", filter_zone=filter_zone_gpd
    )

    # Make overview map with links to split maps
    overview_data = gpd.GeoDataFrame(geometry=split_geom).reset_index()
    overview_data["popup_text"] = (
        '<a href="'
        + split_folder.name
        + "/"
        + overview_data["split_zone"]
        + '.html">'
        + overview_data["split_zone"]
        + "</a>"
    )
    overview_data["popup_text_submap"] = (
        '<a href="'
        + overview_data["split_zone"]
        + '.html">'
        + overview_data["split_zone"]
        + "</a>"
    )

    overview_geom = {
        "Subset Areas": MapData(
            data=overview_data,
            color_column=None,
            to_filter=False,
            options=ExploreOptions(
                tooltip=["split_zone"],
                popup="popup_text",
                style={"fillOpacity": 0.4, "fillColor": "grey", "color": "black"},
            ),
        )
    }
    map_datasets(overview_geom, textbox_text=TEXT_SPLIT, output_path=output_path)

    datasets = {
        "Subset Areas": MapData(
            data=overview_data,
            color_column=None,
            to_filter=False,
            options=ExploreOptions(
                tooltip=["split_zone"],
                popup="popup_text_submap",
                style={"fillOpacity": 0.1, "color": "black", "fillColor": "grey"},
                highlight_style={"fillOpacity": 0.5},
            ),
        )
    } | datasets
    for map_name, geom in tqdm.tqdm(
        split_geom.items(),
        desc=f"Producing {split_folder.name} maps",
        total=len(split_geom),
        dynamic_ncols=True,
    ):
        map_datasets(
            datasets,
            mask=geom,
            mask_name=map_name,
            textbox_text=TEXT_SPLIT,
            output_path=split_folder / f"{map_name}.html",
        )
    LOG.info("Written %s maps to %s", len(split_geom), split_folder)
