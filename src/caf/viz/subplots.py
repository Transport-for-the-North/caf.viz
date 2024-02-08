# -*- coding: utf-8 -*-
"""Functionality for helping creating figures with subplots."""

##### IMPORTS #####
from __future__ import annotations

# Built-Ins
import itertools
import logging
import math
import sys
from typing import Any, Callable, Concatenate, TypeAlias

# Third Party
import numpy as np
from matplotlib import axes, figure
from matplotlib import pyplot as plt

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_FIGURE_WIDTH_FACTOR = 5
_FIGURE_HEIGHT_FACTOR = 4

# Python 3.10 raises an error when Concatenate ends with ellipsis,
# although this should be allowed according to the Python docs
if sys.version_info.major == 3 and sys.version_info.minor > 10:
    PlotFunction: TypeAlias = Callable[Concatenate[figure.Figure, axes.Axes, ...], None]

##### FUNCTIONS & CLASSES #####


def subplotter(
    func: PlotFunction,
    plot_args: list[dict[str, Any]],
    nrows: int,
    ncols: int,
    **subplots_kwargs,
) -> figure.Figure:
    """Plot subplots in a grid, axes plots are created using `func`.

    A roughly square grid of subplots will be created, with a figure
    size calculated based on the number of subplots in the grid.

    Parameters
    ----------
    func : (figure.Figure, axes.Axes, ...) -> None
        Function to plot data onto the individual subplot axes,
        this will be call once with a single axes for each set of
        arguments in `plot_args`.
    plot_args : list[dict[str, Any]]
        List of arguments for plotting `func`, the length of
        this defines the number of subplot axes generated.
    nrows : int
        Number of rows of subplots to create.
    ncols : int
        Number of columns of subplots to create.
    **subplots_kwargs : Keyword arguments
        Any other keyword arguments are passed to `plt.subplots`.
        This cannot contain the parameter: 'squeeze'.

    Returns
    -------
    figure.Figure
        The plotted figure.

    Raises
    ------
    ValueError
        - If length of `plot_args` is 0, or not enough Axes are created
        to contain all the subplots i.e. ncols * nrows < len(plot_args).
        - If one of 'squeeze' is given in `subplots_kwargs`.
    """
    if len(plot_args) == 0:
        raise ValueError("no arguments given for plotting")

    n_subplots = len(plot_args)
    if nrows * ncols < n_subplots:
        raise ValueError(f"not enough Axes rows and columns to fit all {n_subplots} subplots")

    if "squeeze" in subplots_kwargs:
        raise ValueError("squeeze cannot be defined in `subplot_kwargs` when using subplotter")

    subplots_kwargs = subplots_kwargs.copy()
    figsize = subplots_kwargs.pop(
        "figsize",
        (math.ceil(_FIGURE_WIDTH_FACTOR * ncols), math.ceil(_FIGURE_HEIGHT_FACTOR * nrows)),
    )

    axs: list[list[axes.Axes]]
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        figsize=figsize,
        **subplots_kwargs,
    )

    grid_iter = itertools.product(range(nrows), range(ncols))
    for (i, j), args in itertools.zip_longest(grid_iter, plot_args):
        if args is None:
            # Turn off axes if no more data to plot
            axs[i][j].set_axis_off()
            continue

        func(fig, axs[i][j], **args)

    return fig


def grid_plot(
    func: PlotFunction, plot_args: list[dict[str, Any]], **subplots_kwargs
) -> figure.Figure:
    """Plot subplots in a grid, axes plots are created using `func`.

    A roughly square grid of subplots will be created, with a figure
    size calculated based on the number of subplots in the grid.

    Parameters
    ----------
    func : (figure.Figure, axes.Axes, ...) -> None
        Function to plot data onto the individual subplot axes,
        this will be call once with a single axes for each set of
        arguments in `plot_args`.
    plot_args : list[dict[str, Any]]
        List of arguments for plotting `func`, the length of
        this defines the number of subplot axes generated.
    **subplots_kwargs : Keyword arguments
        Any other keyword arguments are passed to `plt.subplots`.
        This cannot contain the parameters:
        'squeeze', 'nrows' or 'ncols'.

    Returns
    -------
    figure.Figure
        The plotted figure.

    Raises
    ------
    ValueError
        - If length of `plot_args` is 0.
        - If one of 'squeeze', 'nrows' or 'ncols' is given in `subplots_kwargs`.
    """
    if len(plot_args) == 0:
        raise ValueError("no arguments given for plotting")

    n_subplots = len(plot_args)
    ncols = int(np.ceil(np.sqrt(n_subplots)))
    nrows = int(np.ceil(n_subplots / ncols))

    for param in ("squeeze", "nrows", "ncols"):
        if param in subplots_kwargs:
            raise ValueError(
                f"{param} cannot be defined in `subplot_kwargs` when using grid_plot"
            )

    return subplotter(func, plot_args, nrows=nrows, ncols=ncols, **subplots_kwargs)
