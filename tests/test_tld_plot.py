"""Testing for the plot_tld function in caf.viz.xy_plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from caf.toolkit import cost_utils

from caf.viz.xy_plot import plot_tld


def get_output_path(filename: str) -> Path:
    """Get the test plot output path for a given filename."""
    output_dir = Path("tests/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def make_cd() -> cost_utils.CostDistribution:
    """Create a simple CostDistribution object for testing."""
    cost_dist = pd.DataFrame(
        {
            "min": [0, 5, 10, 15],
            "max": [5, 10, 15, 20],
            "avg": [2.5, 7.5, 12.5, 17.5],
            "trips": [100, 200, 150, 50],
        }
    )
    return cost_utils.CostDistribution(cost_dist)


def make_skewed_cd() -> cost_utils.CostDistribution:
    """Create a skewed CostDistribution object for testing."""
    cost_dist = pd.DataFrame(
        {
            "min": [0, 5, 10, 15],
            "max": [5, 10, 15, 20],
            "avg": [2.5, 7.5, 12.5, 17.5],
            "trips": [100, 200, 150, 50],
            "weighted_avg": [4.0, 9.0, 14.0, 19.0],
        }
    )
    return cost_utils.CostDistribution(cost_dist, weighted_avg_col="weighted_avg")


def test_plot_tld() -> None:
    """Test the basic functionality of the plot_tld function."""
    cd = make_cd()

    ax = plot_tld(cd)
    ax.figure.savefig(
        get_output_path("test_plot_tld.png")
    )  # Save the figure to check it visually
    assert ax is not None


def test_zero_trips() -> None:
    """Test the plot_tld function with zero trips."""
    cd = make_cd()
    cd.trip_vals[:] = 0

    ax = plot_tld(cd)
    ax.figure.savefig(
        get_output_path("test_zero_trips.png")
    )  # Save the figure to check it visually
    assert ax is not None


def test_uneven_bins() -> None:
    """Test the plot_tld function with uneven bin widths."""
    cost_dist = pd.DataFrame(
        {
            "min": [0, 5, 15],
            "max": [5, 15, 30],
            "avg": [2.5, 10, 22.5],
            "trips": [100, 200, 50],
        }
    )
    cd = cost_utils.CostDistribution(cost_dist)

    ax = plot_tld(cd)
    ax.figure.savefig(
        get_output_path("test_uneven_bins.png")
    )  # Save the figure to check it visually
    assert ax is not None


def test_single_bin() -> None:
    """Test the plot_tld function with a single bin."""
    cost_dist = pd.DataFrame({"min": [0], "max": [10], "avg": [5], "trips": [100]})
    cd = cost_utils.CostDistribution(cost_dist)

    ax = plot_tld(cd)
    ax.figure.savefig(
        get_output_path("test_single_bin.png")
    )  # Save the figure to check it visually
    assert ax is not None


def test_all_trips_in_one_bin() -> None:
    """Test the plot_tld function with all trips in one bin."""
    cd = make_cd()
    cd.trip_vals[:] = [0, 0, 1000, 0]

    ax = plot_tld(cd)
    ax.figure.savefig(
        get_output_path("test_all_trips_in_one_bin.png")
    )  # Save the figure to check it visually

    assert ax is not None


def test_no_weighted_avg_vals() -> None:
    """Test the plot_tld function with no weighted average values."""
    cost_dist = pd.DataFrame(
        {
            "min": [0, 5, 10],
            "max": [5, 10, 15],
            "avg": [2.5, 7.5, 12.5],
            "trips": [100, 200, 150],
        }
    )
    cd = cost_utils.CostDistribution(cost_dist, weighted_avg_col=None)

    ax = plot_tld(cd)
    ax.figure.savefig(
        get_output_path("test_no_weighted_avg_vals.png")
    )  # Save the figure to check it visually
    assert ax is not None


def test_many_bins() -> None:
    """Test the plot_tld function with many bins."""
    n = 100
    cost_dist = pd.DataFrame(
        {
            "min": range(n),
            "max": range(1, n + 1),
            "avg": [i + 0.5 for i in range(n)],
            "trips": np.random.randint(0, 100, size=n),
        }
    )
    cd = cost_utils.CostDistribution(cost_dist)

    ax = plot_tld(cd)
    ax.figure.savefig(
        get_output_path("test_many_bins.png")
    )  # Save the figure to check it visually
    assert len(ax.patches) == n  # Check that the number of bars matches the number of bins


def test_negative_trips() -> None:
    """Test the plot_tld function with negative trip counts."""
    cd = make_cd()
    cd.trip_vals[0] = -10  # Set a negative trip count

    ax = plot_tld(cd)
    ax.figure.savefig(
        get_output_path("test_negative_trips.png")
    )  # Save the figure to check it visually
    assert ax is not None


def test_weighted_avg_points() -> None:
    """Test the plot_tld function with weighted average points."""
    cd = make_skewed_cd()

    ax = plot_tld(cd, show_weighted_avg_line=True, show_weighted_avg_points=True)
    ax.figure.savefig(
        get_output_path("test_weighted_avg_points.png")
    )  # Save the figure to check it visually
    assert ax is not None


def test_existing_axis() -> None:
    """Test the plot_tld function with an existing axis."""
    cd = make_cd()
    _fig, ax = plt.subplots()

    result_ax = plot_tld(cd, ax=ax)
    result_ax.figure.savefig(
        get_output_path("test_existing_axis.png")
    )  # Save the figure to check it visually
    assert result_ax is ax
