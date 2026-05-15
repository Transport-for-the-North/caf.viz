"""Static heatmaps for polygon based zone systems with GeoPandas."""

##### IMPORTS #####

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import re
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

import mapclassify
import numpy as np
import pandas as pd
from matplotlib import cm, patches
from matplotlib import pyplot as plt
from shapely import geometry

if TYPE_CHECKING:
    import geopandas as gpd


##### CONSTANTS #####
LOG = logging.getLogger(__name__)


##### CLASSES #####
@dataclasses.dataclass
class CustomCmap:
    """Store information about a custom colour map."""

    bin_categories: pd.Series
    colours: pd.DataFrame
    legend_elements: list[patches.Patch]

    def __add__(self, other: CustomCmap) -> CustomCmap:
        """Return new CustomCmap with the attributes from `self` and `other` concatenated."""
        if not isinstance(other, CustomCmap):
            raise TypeError(f"other should be a CustomCmap not {type(other)}")
        return CustomCmap(
            pd.concat([self.bin_categories, other.bin_categories], verify_integrity=True),
            pd.concat([self.colours, other.colours], verify_integrity=True),
            self.legend_elements + other.legend_elements,
        )


class Bounds(NamedTuple):
    """Coordinates for geospatial extent."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int


@dataclasses.dataclass
class _LegendLabelValues:
    lower: float
    upper: float
    lower_formatted: str
    upper_formatted: str


##### FUNCTIONS #####
def _extract_legend_values(text: str) -> _LegendLabelValues | None:
    """Extract numbers and formatted strings from legend label."""
    number = r"\d+(?:\.\d*)?"
    units = r"%?"
    pattern = (
        rf"(?P<lower>{number})"
        rf"(?P<lower_units>{units})"
        r"[,\s]+"  # separator
        rf"(?P<upper>{number})"
        rf"(?P<upper_units>{units})"
    )

    match: re.Match | None = re.search(pattern, text)
    if match is None:
        return match

    return _LegendLabelValues(
        lower=float(match.group("lower")),
        upper=float(match.group("upper")),
        lower_formatted="{}{}".format(match.group("lower"), match.group("lower_units")),
        upper_formatted="{}{}".format(match.group("upper"), match.group("upper_units")),
    )


def _colormap_classify(
    data: pd.Series,
    cmap_name: str,
    n_bins: int = 5,
    label_fmt: str = "{:.0f}",
    bins: list[int | float] | None = None,
    nan_colour: tuple[float, float, float, float] | None = None,
) -> CustomCmap:
    """Calculate a NaturalBreaks colour map."""

    def make_label(lower: float, upper: float) -> str:
        if lower == -np.inf:
            return "< " + label_fmt.format(upper)
        if upper == np.inf:
            return "> " + label_fmt.format(lower)
        return label_fmt.format(lower) + " - " + label_fmt.format(upper)

    finite = data.dropna()
    if finite.empty:
        # Return empty colour map
        return CustomCmap(pd.Series(dtype=float), pd.DataFrame(dtype=float), [])

    if bins is not None:
        mc_bins = mapclassify.UserDefined(finite, bins)
    else:
        mc_bins = _mapclassify_natural(finite, k=n_bins)

    bin_categories = pd.Series(mc_bins.yb, index=finite.index)

    cmap = cm.get_cmap(cmap_name, mc_bins.k)
    # Cmap produces incorrect results if given floats instead of int so
    # bin_categories can't contain Nans until after colours are calculated
    colours = pd.DataFrame(
        cmap(bin_categories.astype(int)), index=bin_categories.index, columns=iter("RGBA")
    )

    bin_categories = bin_categories.reindex_like(data)
    colours = colours.reindex(bin_categories.index)

    if nan_colour is None:
        colours.loc[bin_categories.isna(), :] = np.nan
    else:
        colours.loc[bin_categories.isna(), :] = nan_colour

    min_bin = np.min(finite)
    if min_bin > mc_bins.bins[0]:
        if mc_bins.bins[0] > 0:
            min_bin = 0
        else:
            min_bin = -np.inf

    bins = [min_bin, *mc_bins.bins]
    labels = [make_label(i, j) for i, j in itertools.pairwise(bins)]
    legend = [
        patches.Patch(fc=i, label=j, ls="")
        for i, j in zip(cmap(range(mc_bins.k)), labels, strict=True)
    ]

    if nan_colour is not None:
        legend.append(patches.Patch(fc=nan_colour, label="Missing Values", ls=""))

    return CustomCmap(bin_categories, colours, legend)


def _mapclassify_natural(
    y: np.ndarray,
    k: int = 5,  # pylint: disable=invalid-name
    initial: int = 10,
) -> mapclassify.NaturalBreaks:
    """Try smaller values of k on error of NaturalBreaks.

    Parameters
    ----------
    y : np.ndarray
        (n,1), values to classify
    k : int, optional, default 5
        number of classes required
    initial : int, default 10
        Number of initial solutions generated with different centroids.
        Best of initial results is returned.

    Returns
    -------
    mapclassify.NaturalBreaks
    """
    while True:
        try:
            return mapclassify.NaturalBreaks(y, k, initial)
        except ValueError:
            if k <= 2:  # noqa: PLR2004
                raise
            k -= 1


def _add_poly_boundary(
    ax: plt.Axes, area: geometry.MultiPolygon | geometry.Polygon
) -> patches.Polygon:
    """Add analytical area boundary to a map.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add boundary to.
    area : Union[geometry.MultiPolygon, geometry.Polygon]
        Analytical area polygon of boundary to add.

    Returns
    -------
    patches.Polygon
        The patch added to the axes.

    Raises
    ------
    TypeError
        If `area` isn't a Polygon or MultiPolygon.
    """
    if isinstance(area, geometry.Polygon):
        polygons = geometry.MultiPolygon([area])
    elif isinstance(area, geometry.MultiPolygon):
        polygons = area
    else:
        raise TypeError(f"unexpected type ({type(area)}) for area")

    legend_patch = None
    for i, poly in enumerate(polygons.geoms):
        patch = patches.Polygon(
            poly.exterior.coords,
            ec="red",
            fill=False,
            linewidth=2,
            label="Boundary" if i == 0 else None,
            zorder=2,
        )
        if i == 0:
            legend_patch = patch
        ax.add_patch(patch)

    if legend_patch is None:
        raise ValueError("no polygons plotted")

    return legend_patch


def heatmap_figure(
    geodata: gpd.GeoDataFrame,
    column_name: str,
    title: str,
    bins: list[int | float] | None = None,
    n_bins: int = 5,
    polygon_boundary: geometry.Polygon | geometry.MultiPolygon = None,
    positive_negative_colormaps: bool = False,
    legend_label_fmt: str = "{:.1%}",
    legend_title: str | None = None,
    zoomed_bounds: Bounds | None = None,
    missing_kwds: dict[str, Any] | None = None,
    annotation: str | None = None,
) -> plt.Figure:
    """Create a heatmap with GeoSpatial Polygon data.

    Parameters
    ----------
    geodata
        GeoSpatial data to be plotted, designed for use with Polygon
        geometries but other geometry types should work.
    column_name
        Name of column used to determine heatmap colours.
    title
        Figure title.
    bins
        Optional bin edges to use for colouring, if not provided
        bin edges will be generated based on `n_bins`.
    n_bins
        Number of bins to generate (default 5), ignored if `bins` is provided.
    polygon_boundary
        Polygon to overlay on heatmap.
    positive_negative_colormaps
        If True colour use separate colour maps for positive and negative values.
    legend_label_fmt
        Number format for legend, default "{:.1%}".
    legend_title
        Optional legend title, `column_name` is used if not given.
    zoomed_bounds
        Optional bounding box to zoom to in a sub-plot.
    missing_kwds
        Keyword arguments for styling any NaN values.
    annotation
        Optional annotation to add to the bottom of the figure.

    Returns
    -------
    plt.Figure
        Figure with heatmap plotted, with 2 Axes if `zoomed_bounds`
        is given.
    """
    legend_kwargs = dict(title_fontsize="large", fontsize="medium")

    ncols = 1 if zoomed_bounds is None else 2

    fig, axes = plt.subplots(
        1, ncols, figsize=(20, 15), frameon=False, constrained_layout=True
    )
    if ncols == 1:
        axes: list[plt.Axes] = [axes]  # type: ignore[no-redef]

    fig.suptitle(title, fontsize="xx-large", backgroundcolor="white")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_axis_off()
        if polygon_boundary is not None:
            _add_poly_boundary(ax, polygon_boundary)
    if polygon_boundary is not None:
        axes[0].legend(**legend_kwargs, loc="upper right")

    kwargs = dict(
        column=column_name,
        cmap="viridis_r",
        scheme="NaturalBreaks",
        k=7,
        legend_kwds=dict(
            title=legend_title or str(column_name).title(),
            **legend_kwargs,
            loc="upper right",
        ),
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",
        },
        linewidth=0.0,
        edgecolor="black",
    )

    if missing_kwds is not None:
        kwargs["missing_kwds"].update(missing_kwds)  # type: ignore[attr-defined]

    if positive_negative_colormaps:
        # Calculate, and apply, separate colormaps for positive and negative values
        negative_cmap = _colormap_classify(
            geodata.loc[geodata[column_name] <= 0, column_name],
            "PuBu_r",
            label_fmt=legend_label_fmt,
            n_bins=n_bins,
            bins=list(filter(lambda x: x <= 0, bins)) if bins is not None else bins,
        )
        positive_cmap = _colormap_classify(
            geodata.loc[
                (geodata[column_name] > 0) | (geodata[column_name].isna()), column_name
            ],
            "YlGn",
            label_fmt=legend_label_fmt,
            n_bins=n_bins,
            bins=list(filter(lambda x: x > 0, bins)) if bins is not None else bins,
            nan_colour=(1.0, 0.0, 0.0, 1.0),
        )
        cmap = negative_cmap + positive_cmap
        # Update colours index to be the same order as geodata
        cmap.colours = cmap.colours.reindex(geodata.index)

        for ax in axes:
            geodata.plot(
                ax=ax,
                color=cmap.colours.values,
                linewidth=kwargs["linewidth"],
                edgecolor=kwargs["edgecolor"],
                missing_kwds=kwargs["missing_kwds"],
            )
        axes[ncols - 1].legend(
            handles=cmap.legend_elements,
            **kwargs.pop("legend_kwds"),  # type: ignore[arg-type]
        )

    else:
        if bins:
            kwargs["scheme"] = "UserDefined"
            bins = sorted(bins)
            max_ = np.max(geodata[column_name].values)
            if bins[-1] < max_:
                bins[-1] = math.ceil(max_)
            kwargs["classification_kwds"] = {"bins": bins}
            del kwargs["k"]

        # If the quatiles scheme throws a warning then use FisherJenksSampled
        warnings.simplefilter("error", category=UserWarning)
        try:
            geodata.plot(ax=axes[0], legend=zoomed_bounds is None, **kwargs)
            if zoomed_bounds is not None:
                geodata.plot(ax=axes[1], legend=True, **kwargs)
        except UserWarning:
            kwargs["scheme"] = "FisherJenksSampled"
            geodata.plot(ax=axes[0], legend=zoomed_bounds is None, **kwargs)
            if zoomed_bounds is not None:
                geodata.plot(ax=axes[1], legend=True, **kwargs)
        finally:
            warnings.simplefilter("default", category=UserWarning)

        # Format legend text
        legend = axes[ncols - 1].get_legend()
        for label in legend.get_texts():
            # Don't attempt to rename the label
            # if it isn't in the expected format
            values = _extract_legend_values(label.get_text())
            if values is None:
                continue

            if values.lower == -np.inf:
                label.set_text(f"< {values.upper_formatted}")
            elif values.upper == np.inf:
                label.set_text(f"> {values.lower_formatted}")
            else:
                label.set_text(f"{values.lower_formatted} - {values.upper_formatted}")

    if zoomed_bounds is not None:
        axes[1].set_xlim(zoomed_bounds.min_x, zoomed_bounds.max_x)
        axes[1].set_ylim(zoomed_bounds.min_y, zoomed_bounds.max_y)

    if annotation is not None:
        axes[ncols - 1].annotate(
            annotation,
            xy=(0.9, 0.01),
            xycoords="figure fraction",
            bbox=dict(boxstyle="square", fc="white"),
        )
    return fig
