# -*- coding: utf-8 -*-
"""Functionality for helping creating figures with subplots."""

##### IMPORTS #####

import itertools
import logging
import math
from typing import Any, Callable, Concatenate

import numpy as np
from matplotlib import axes, figure
from matplotlib import pyplot as plt

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_FIGURE_WIDTH_FACTOR = 4
_FIGURE_HEIGHT_FACTOR = 3.5


##### FUNCTIONS & CLASSES #####


def grid_plot(
    func: Callable[Concatenate[figure.Figure, axes.Axes, ...], None],
    plot_args: list[dict[str, Any]],
    **figure_kwargs,
) -> figure.Figure:
    """Plot subplots in a grid, axes plots are created using `func`.

    A roughly square grid of subplots will be created, with a figure
    size calculated based on the number of subplots in the grid.

    Parameters
    ----------
    func : Callable[Concatenate[figure.Figure, axes.Axes, ...], None]
        Function to plot data onto the individual subplot axes,
        this will be call once with a single axes for each set of
        arguments in `plot_args`.
    plot_args : list[dict[str, Any]]
        List of arguments for plotting `func`, the length of
        this defines the number of subplot axes generated.
    **figure_kwargs : Keyword arguments
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
        - If one of 'squeeze', 'nrows' or 'ncols' is given in `figure_kwargs`.
    """
    if len(plot_args) == 0:
        raise ValueError("no arguments given for plotting")

    n_subplots = len(plot_args)
    ncols = int(np.ceil(np.sqrt(n_subplots)))
    nrows = int(np.ceil(n_subplots / ncols))

    for param in ("squeeze", "nrows", "ncols"):
        if param in figure_kwargs:
            raise ValueError(
                f"{param} cannot be defined in `figure_kwargs` when using grid_plot"
            )

    figure_kwargs = figure_kwargs.copy()
    figsize = figure_kwargs.pop(
        "figsize",
        (math.ceil(_FIGURE_WIDTH_FACTOR * ncols), math.ceil(_FIGURE_HEIGHT_FACTOR * nrows)),
    )

    axs: list[list[axes.Axes]]
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        figsize=figsize,
        **figure_kwargs,
    )

    grid_iter = itertools.product(range(nrows), range(ncols))
    for (i, j), args in itertools.zip_longest(grid_iter, plot_args):
        if args is None:
            # Turn off axes if no more data to plot
            axs[i][j].set_axis_off()
            continue

        func(fig, axs[i][j], **args)

    return fig
