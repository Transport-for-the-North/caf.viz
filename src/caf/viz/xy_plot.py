# -*- coding: utf-8 -*-
"""Functionality for plotting of general X / Y data."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import enum
import logging
import warnings
from typing import Sequence

# Third Party
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import axes, figure, ticker
from pydantic import dataclasses
from scipy import stats

# Local Imports
from caf.viz import subplots, utils

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### FUNCTIONS & CLASSES #####


@dataclasses.dataclass(config={"arbitrary_types_allowed": True})
class BasicData:
    """Basic data for XY plots."""

    data: pd.DataFrame
    x_column: str
    y_column: str
    x_label: str | None = None
    y_label: str | None = None
    title: str | None = None
    auto_label: bool = True


@dataclasses.dataclass
class CmapData:
    """Colormap parameters."""

    column: str
    label: str | None = None
    auto_label: bool = True
    cmap: str = mpl.rcParams.get("image.cmap")

    def get_label(self) -> str | None:
        """Return label for colorbar."""
        if self.label is not None:
            return self.label
        if self.auto_label:
            return self.column
        return None


class XYPlotType(enum.Enum):
    """Types of 2D XY plots."""

    SCATTER = "scatter"
    SCATTER_DENSITY = "scatter_density"
    HEXBIN = "hexbin"

    @classmethod
    def _missing_(cls, value) -> XYPlotType | None:
        normal = utils.normalise_name(str(value))
        for member in cls:
            if member.value == normal:
                return member

        raise ValueError(
            f"'{value}' is not a valid XYPlotType should be one of: "
            + ", ".join(f"'{i.value}'" for i in cls)
        )


def hexbin(
    fig: figure.Figure,
    ax: axes.Axes,
    data: BasicData,
    cmap: CmapData | None = None,
    gridsize: int = 50,
    colorbar: bool = True,
) -> None:
    # TODO(MB) docstring
    weights = None
    if cmap is not None:
        if cmap.column not in data.data.columns:
            raise KeyError(f"colormap data column ('{cmap.column}') not present")
        weights = cmap.column

    hb = ax.hexbin(
        data.x_column,
        data.y_column,
        C=weights,
        data=data.data,
        mincnt=1,
        gridsize=gridsize,
        linewidths=0.1,
    )

    if not colorbar:
        return

    if cmap is None:
        label = "Count"
    else:
        label = cmap.get_label()

    fig.colorbar(hb, ax=ax, label=label)


def scatter(
    fig: figure.Figure,
    ax: axes.Axes,
    data: BasicData,
    density: bool = False,
    cmap: CmapData | None = None,
    colorbar: bool = True,
    # TODO(MB) Add keyword arguments to pass to `ax.scatter`
) -> None:
    # TODO(MB) docstring
    xy_data = data.data[[data.x_column, data.y_column]].values.T
    z_data = None

    if cmap is not None and density:
        raise ValueError(
            "cmap and density shouldn't both be given,"
            " unsure which to use for colouring scatter plot"
        )

    if cmap is not None:
        z_data = data.data[cmap.column]
    elif density:
        # Calculate point density
        kernel = stats.gaussian_kde(xy_data)
        z_data: np.ndarray = kernel(xy_data)

    if z_data is not None:
        # Sort the points by density, so that the densest points are plotted last
        idx = z_data.argsort()
        z_data = z_data[idx]
        xy_data = np.take(xy_data, idx, axis=1)

    points = ax.scatter(xy_data[0], xy_data[1], c=z_data)

    if not colorbar or z_data is None:
        return None

    if cmap is not None:
        fig.colorbar(points, ax=ax, label=cmap.get_label())
    else:
        fig.colorbar(points, ax=ax, label="Density", ticks=ticker.NullLocator())


def axes_plot_xy(
    fig: figure.Figure,
    ax: axes.Axes,
    type_: XYPlotType,
    data: BasicData,
    cmap: CmapData | None = None,
    **plot_kwargs,
) -> None:
    # TODO(MB) docstring
    # TODO(MB) Validate columns are present in data

    if type_ == XYPlotType.HEXBIN:
        hexbin(fig, ax, data, cmap, **plot_kwargs)

    elif type_ in (XYPlotType.SCATTER, XYPlotType.SCATTER_DENSITY):
        if len(data.data) > 1_000_000:
            warnings.warn(
                "scatter plots may take a long time with "
                "a lot of data, try hexbin plots instead",
                RuntimeWarning,
            )
        scatter(
            fig,
            ax,
            data,
            density=type_ == XYPlotType.SCATTER_DENSITY,
            cmap=cmap,
            **plot_kwargs,
        )

    else:
        raise NotImplementedError(f"unknown plot type {type_}")

    if data.title is not None:
        ax.set_title(data.title)

    if data.x_label is not None:
        ax.set_xlabel(data.x_label)
    elif data.auto_label:
        ax.set_xlabel(data.x_column)

    if data.y_label is not None:
        ax.set_ylabel(data.y_label)
    elif data.auto_label:
        ax.set_ylabel(data.y_column)


def plot_xy(
    data: pd.DataFrame,
    x_column: str | Sequence[str],
    y_column: str | Sequence[str],
    type_: XYPlotType = XYPlotType.SCATTER,
    title: str | None = None,
    weight_column: None | str | Sequence[str] = None,
) -> figure.Figure:
    # TODO(MB) Docstring
    # TODO(MB) Add functionality to share a colorbar
    # TODO(MB) Add option to define subplot parameters and
    # switch to subplotter if nrows and ncols are given
    type_ = XYPlotType(type_)

    if isinstance(x_column, str) and isinstance(y_column, str):
        x_column, y_column = (x_column,), (y_column,)

    x_column, y_column = tuple(x_column), tuple(y_column)
    if len(x_column) != len(y_column):
        raise ValueError(
            f"number of x column names ({len(x_column)}) and "
            f"y column names ({len(y_column)}) should be the same"
        )

    if weight_column is not None:
        if isinstance(weight_column, str):
            weight_column = (weight_column,)
        weight_column = tuple(weight_column)
        if len(weight_column) != len(x_column):
            raise ValueError(
                f"number of weight column names ({len(weight_column)}) should "
                f"be the same as the x and y column names ({len(x_column)})"
            )

    plot_data = []
    for i, (x, y) in enumerate(zip(x_column, y_column)):
        plot_data.append(
            dict(
                type_=type_,
                data=BasicData(data, x, y),
                cmap=None if weight_column is None else CmapData(weight_column[i]),
            )
        )

    fig = subplots.grid_plot(axes_plot_xy, plot_data)
    if title is not None:
        fig.suptitle(title)

    return fig
