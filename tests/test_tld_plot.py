import pandas as pd
from caf.toolkit.cost_utils import CostDistribution
from caf.viz.xy_plot import plot_tld
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def make_cd():
    cost_dist = pd.DataFrame({
        "min": [0, 5, 10, 15],
        "max": [5, 10, 15, 20],
        "avg": [2.5, 7.5, 12.5, 17.5],
        "trips": [100, 200, 150, 50]
    })
    return CostDistribution(cost_dist)

def make_skewed_cd():
    cost_dist = pd.DataFrame({
        "min": [0, 5, 10, 15],
        "max": [5, 10, 15, 20],
        "avg": [2.5, 7.5, 12.5, 17.5],
        "trips": [100, 200, 150, 50],
        "weighted_avg": [4.0, 9.0, 14.0, 19.0]
    })
    return CostDistribution(cost_dist, weighted_avg_col="weighted_avg")


def test_plot_tld():
    cd = make_cd()

    ax = plot_tld(cd)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_plot_tld.png")  # Save the figure to check it visually
    assert ax is not None


def test_zero_trips():
    cd = make_cd()
    cd.trip_vals[:] = 0

    ax = plot_tld(cd)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_zero_trips.png")  # Save the figure to check it visually
    assert ax is not None

def test_uneven_bins():
    cost_dist = pd.DataFrame({
        "min": [0, 5, 15],
        "max": [5, 15, 30],
        "avg": [2.5, 10, 22.5],
        "trips": [100, 200, 50]
    })
    cd = CostDistribution(cost_dist)

    ax = plot_tld(cd)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_uneven_bins.png")  # Save the figure to check it visually
    assert len(ax.patches) == 3  # Check that the number of bars matches the number of bins


def test_single_bin():
    cost_dist = pd.DataFrame({
        "min": [0],
        "max": [10],
        "avg": [5],
        "trips": [100]
    })
    cd = CostDistribution(cost_dist)

    ax = plot_tld(cd)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_single_bin.png")  # Save the figure to check it visually
    assert len(ax.patches) == 1  # Check that there is only one bar for the single bin

def test_all_trips_in_one_bin():
    cd = make_cd()
    cd.trip_vals[:] = [0, 0, 1000, 0]

    ax = plot_tld(cd)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_all_trips_in_one_bin.png")  # Save the figure to check it visually

    assert ax is not None

def test_no_weighted_avg_vals():
    cost_dist = pd.DataFrame({
        "min": [0, 5, 10],
        "max": [5, 10, 15],
        "avg": [2.5, 7.5, 12.5],
        "trips": [100, 200, 150]
    })
    cd = CostDistribution(cost_dist, weighted_avg_col=None)

    ax = plot_tld(cd)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_no_weighted_avg_vals.png")  # Save the figure to check it visually
    assert ax is not None

def test_many_bins():
    n = 100
    cost_dist = pd.DataFrame({
        "min": range(n),
        "max": range(1, n+1),
        "avg": [i + 0.5 for i in range(n)],
        "trips": np.random.randint(0, 100, size=n)
    })
    cd = CostDistribution(cost_dist)

    ax = plot_tld(cd)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_many_bins.png")  # Save the figure to check it visually
    assert len(ax.patches) == n  # Check that the number of bars matches the number of bins

def test_negative_trips():
    cd = make_cd()
    cd.trip_vals[0] = -10  # Set a negative trip count

    ax = plot_tld(cd)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_negative_trips.png")  # Save the figure to check it visually
    assert ax is not None

def test_weighted_avg_points():
    cd = make_skewed_cd()

    ax = plot_tld(cd, show_weighted_avg_line=True, show_weighted_avg_points=True)
    ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_weighted_avg_points.png")  # Save the figure to check it visually
    assert ax is not None


def test_existing_axis():
    cd = make_cd()
    fig, ax = plt.subplots()

    result_ax = plot_tld(cd, ax=ax)
    result_ax.figure.savefig(Path.cwd() / "tests" / "output" / "test_existing_axis.png")  # Save the figure to check it visually
    assert result_ax is ax