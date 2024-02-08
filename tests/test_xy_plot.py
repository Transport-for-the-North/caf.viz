# -*- coding: utf-8 -*-
"""Tests for the `xy_plot` module."""

##### IMPORTS #####

# Built-Ins
import itertools
import pathlib
from typing import Iterator

# Third Party
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf

# Local Imports
from caf.viz import xy_plot

##### CONSTANTS #####


##### CLASSES & FUNCTIONS #####


@pytest.fixture(name="random_data")
def fix_random_data() -> tuple[pd.DataFrame, Iterator[tuple[str, str]]]:
    """Generate DataFrame containing 10 random x / y columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 10 pairs of x / y columns containing random data,
        with columns: X1, Y1... X9, Y9.
    Iterator[tuple[str, str]]
        Infinite generator which returns the names of a pair of x / y
        columns, will repeat previous column names after 9.
    """
    data = {}
    columns = []
    rng = np.random.default_rng()
    for col in range(10):
        names = (f"X{col}", f"Y{col}")
        columns.append(names)

        rand_data = rng.normal(10_000, 5000, 1000)
        data[names[0]] = rand_data
        data[names[1]] = rand_data + (rng.random(1000) * 10_000)

    return pd.DataFrame(data), itertools.cycle(columns)


@pytest.mark.integration
@pytest.mark.parametrize("n_subplots", [1, 6, 10, 23])
def test_plot(
    tmp_path: pathlib.Path,
    random_data: tuple[pd.DataFrame, Iterator[tuple[str, str]]],
    n_subplots: int,
):
    """Test `plot_xy` with different number of subplots for all plot types."""
    plt.style.use("caf.viz.tfn")

    data, column_iter = random_data
    x_columns: list[str] = []
    y_columns: list[str] = []

    while len(x_columns) < n_subplots:
        x, y = next(column_iter)
        x_columns.append(x)
        y_columns.append(y)

    out_file = tmp_path / "test_plots.pdf"
    with backend_pdf.PdfPages(out_file) as pdf:
        for type_ in xy_plot.XYPlotType:
            fig = xy_plot.plot_xy(
                data,
                x_columns,
                y_columns,
                type_,
                f"Test {type_.value.strip().replace('_', ' ').title()} Plots",
            )
            pdf.savefig(fig)
            plt.close(fig)

    assert out_file.is_file(), f"'{out_file.name}' not created"
