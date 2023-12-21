# -*- coding: utf-8 -*-
"""Functionality for plotting of general X / Y data."""

##### IMPORTS #####

from __future__ import annotations

import enum
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib import axes, figure, ticker
from scipy import stats

from caf.viz import subplots, utils

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### FUNCTIONS & CLASSES #####


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
    fig: figure.Figure, ax: axes.Axes, data: pd.DataFrame, x: str, y: str, z: str | None = None
) -> None:
    # TODO(MB) docstring
    weights = None
    if z is not None:
        weights = data[z]

    hb = ax.hexbin(data[x], data[y], C=weights, mincnt=1, gridsize=50, linewidths=0.1)
    fig.colorbar(
        hb,
        ax=ax,
        label="Count" if z is None else z,
        ticks=ticker.MaxNLocator(integer=True) if z is None else None,
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)


def scatter(
    fig: figure.Figure,
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    density: bool = False,
    z: str | None = None,
) -> None:
    # TODO(MB) docstring
    xy_data = data[[x, y]].values.T
    z_data = None

    if z is not None and density:
        raise ValueError(
            "z value and density shouldn't both be given,"
            " unsure which to use for colouring scatter plot"
        )

    if z is not None:
        z_data = data[z]
    elif density:
        # Calculate point density
        kernel = stats.gaussian_kde(xy_data)
        z_data: np.ndarray = kernel(xy_data)

    if z_data is not None:
        # Sort the points by density, so that the densest points are plotted last
        idx = z_data.argsort()
        z_data = z_data[idx]
        xy_data = np.take(xy_data, idx, axis=1)

    points = ax.scatter(
        xy_data[0], xy_data[1], s=3, c=z_data, cmap=None if z_data is None else "viridis"
    )
    if z is not None:
        fig.colorbar(points, ax=ax, label=z)
    elif z_data is not None:
        fig.colorbar(points, ax=ax, label="Density", ticks=ticker.NullLocator())

    ax.set_xlabel(x)
    ax.set_ylabel(y)


def axes_plot_xy(
    fig: figure.Figure,
    ax: axes.Axes,
    type_: XYPlotType,
    data: pd.DataFrame,
    x: str,
    y: str,
    z: str | None = None,
) -> None:
    # TODO(MB) docstring
    # TODO(MB) Validate columns are present in data

    if type_ == XYPlotType.HEXBIN:
        hexbin(fig, ax, data, x, y, z)
    elif type_ in (XYPlotType.SCATTER, XYPlotType.SCATTER_DENSITY):
        scatter(fig, ax, data, x, y, density=type_ == XYPlotType.SCATTER_DENSITY, z=z)
    else:
        raise NotImplementedError(f"unknown plot type {type_}")


def plot_xy(
    data: pd.DataFrame,
    x_column: str | Sequence[str],
    y_column: str | Sequence[str],
    type_: XYPlotType = XYPlotType.SCATTER,
    title: str | None = None,
    weight_column: None | str | Sequence[str] = None,
) -> figure.Figure:
    # TODO(MB) Docstring
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
                data=data,
                x=x,
                y=y,
                z=None if weight_column is None else weight_column[i],
            )
        )

    fig = subplots.grid_plot(axes_plot_xy, plot_data)
    if title is not None:
        fig.suptitle(title)

    return fig
