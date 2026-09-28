"""Tests for general X / Y plotting functionality."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from caf.viz.xy_plot import XYPlotType, axes_plot_xy


@pytest.fixture
def plot_data() -> pd.DataFrame:
    """Create simple data for XY plotting tests."""
    return pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }
    )


def make_basic_data(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> SimpleNamespace:
    """Create the plotting-data attributes required by axes_plot_xy.

    Avoid constructing BasicData directly so this unit test remains focused
    on axes_plot_xy validation rather than Pydantic dataclass validation.
    """
    return SimpleNamespace(
        data=data,
        x_column=x_column,
        y_column=y_column,
        x_label=None,
        y_label=None,
        title=None,
        auto_label=True,
    )


@pytest.mark.parametrize("plot_type", [XYPlotType.SCATTER, XYPlotType.HEXBIN])
def test_axes_plot_xy_valid_columns(
    plot_data: pd.DataFrame,
    plot_type: XYPlotType,
) -> None:
    """Test plotting succeeds when both requested columns exist."""
    fig, ax = plt.subplots()
    try:
        axes_plot_xy(
            fig,
            ax,
            plot_type,
            make_basic_data(plot_data, "x", "y"),
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("plot_type", [XYPlotType.SCATTER, XYPlotType.HEXBIN])
@pytest.mark.parametrize(
    ("x_column", "y_column", "missing_columns"),
    [
        ("missing_x", "y", ("missing_x",)),
        ("x", "missing_y", ("missing_y",)),
        ("missing_x", "missing_y", ("missing_x", "missing_y")),
    ],
)
def test_axes_plot_xy_missing_columns(
    plot_data: pd.DataFrame,
    plot_type: XYPlotType,
    x_column: str,
    y_column: str,
    missing_columns: tuple[str, ...],
) -> None:
    """Test missing plotting columns raise a useful KeyError."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(KeyError) as exc_info:
            axes_plot_xy(
                fig,
                ax,
                plot_type,
                make_basic_data(plot_data, x_column, y_column),
            )
    finally:
        plt.close(fig)

    message = str(exc_info.value)
    assert "plot data column" in message
    for column in missing_columns:
        assert column in message
