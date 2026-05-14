import pandas as pd
from caf.toolkit.cost_utils import CostDistribution
from caf.viz.xy_plot import plot_tld
from pathlib import Path

def make_cd():
    cost_dist = pd.DataFrame({
        "min": [0, 5, 10, 15],
        "max": [5, 10, 15, 20],
        "avg": [2.5, 7.5, 12.5, 17.5],
        "trips": [100, 200, 150, 50]
    })
    return CostDistribution(cost_dist)


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