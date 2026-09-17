"""Matrix plotting functionality."""

import dataclasses
import math
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from shapely.geometry import LineString

from caf.viz import style

MATRIX_PLOT_LINEWIDTHS: tuple[int, int, int] = (6, 3, 1)
MATRIX_PLOT_ALPHAS: tuple[float, float, float] = (0.03, 0.08, 0.2)


def _left_curve(geom: LineString, curve_ratio: float, n_points: int = 30) -> LineString:
    coords = list(geom.coords)
    if len(coords) < 2:  # noqa: PLR2004
        return geom
    x0, y0 = coords[0]
    x1, y1 = coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    chord = np.hypot(dx, dy)
    if chord == 0:
        return geom

    nx = -dy / chord
    ny = dx / chord
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    cx = mx + nx * chord * curve_ratio
    cy = my + ny * chord * curve_ratio

    t = np.linspace(0, 1, n_points)
    one_minus_t = 1 - t
    xs = one_minus_t**2 * x0 + 2 * one_minus_t * t * cx + t**2 * x1
    ys = one_minus_t**2 * y0 + 2 * one_minus_t * t * cy + t**2 * y1
    return LineString(zip(xs, ys, strict=True))


def _add_half_arrows(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    color: str,
    norm_fn: Callable,
    *,
    direction_min_normalized: float,
    direction_arrow_span_ratio: float,
    direction_arrow_offset_ratio: float,
    direction_arrow_scale: float,
    direction_arrow_alpha: float,
    is_negative: bool = False,
) -> plt.Axes:
    if gdf.empty:
        return ax

    for _, row in gdf.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue

        coords = np.array(geom.coords)
        if len(coords) < 3:  # noqa: PLR2004
            continue

        trip_val = row["trips"]
        if is_negative:
            norm_val = np.sqrt(norm_fn(abs(trip_val)))
        else:
            norm_val = np.sqrt(norm_fn(trip_val))
        if norm_val < direction_min_normalized:
            continue

        seg_idx = max(1, int(0.65 * (len(coords) - 1)))
        p0 = coords[seg_idx - 1]
        p1 = coords[seg_idx]
        tangent = p1 - p0
        t_len = np.hypot(tangent[0], tangent[1])
        if t_len == 0:
            continue
        tangent = tangent / t_len
        left = np.array([-tangent[1], tangent[0]])

        geom_len = max(geom.length, t_len)
        span = geom_len * direction_arrow_span_ratio
        offset = geom_len * direction_arrow_offset_ratio

        center = p1 + left * offset
        tail = center - tangent * (0.5 * span)
        head = center + tangent * (0.5 * span)

        ax.annotate(
            "",
            xy=head,
            xytext=tail,
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1,
                mutation_scale=direction_arrow_scale,
                alpha=direction_arrow_alpha * norm_val,
            ),
            zorder=3,
        )
    return ax


def _validate_line_settings(
    line_widths: int | list[int] | None, alphas: list[float] | float | None
) -> tuple[list[int], list[float]]:
    if line_widths is None:
        line_widths = list(MATRIX_PLOT_LINEWIDTHS)
    elif isinstance(line_widths, int):
        line_widths = [line_widths]
    if alphas is None:
        alphas = list(MATRIX_PLOT_ALPHAS)
    elif isinstance(alphas, float):
        alphas = [alphas]

    if len(line_widths) != len(alphas):
        raise ValueError("line_widths and alphas should be the same length.")
    return line_widths, alphas


def _plot_inters(
    ax: plt.Axes,
    inter_values: gpd.GeoDataFrame,
    line_widths: list[int],
    alphas: list[float],
    normalisation_fn: Callable,
    colour: str,
) -> plt.Axes:
    for lw, alpha in zip(line_widths, alphas, strict=True):
        if not inter_values.empty:
            alpha_plot = alpha * np.sqrt(normalisation_fn(inter_values["trips"]))

            inter_values.plot(
                ax=ax,
                color=colour,
                alpha=alpha_plot,
                linewidth=lw,
                zorder=1,
                rasterized=True,
            )
    return ax


def _plot_intras(
    ax: plt.Axes,
    intra_values: gpd.GeoDataFrame,
    line_widths: list[int],
    alphas: list[float],
    normalisation_fn: Callable,
    colour: str,
) -> plt.Axes:
    if not intra_values.empty:
        for lw, alpha in zip(line_widths, alphas, strict=True):
            intra_values.plot(
                ax=ax,
                color=colour,
                markersize=lw / 2,
                alpha=np.sqrt(normalisation_fn(intra_values["trips"])) * alpha,
                zorder=2,
                rasterized=True,
            )
    return ax


@dataclasses.dataclass
class DirectionInputs:
    """Definition for parameters for directional matrix plotting."""

    curve_left_ratio: float = 0.08
    direction_arrow_alpha: float = 0.8
    direction_arrow_scale: float = 9
    direction_min_normalized: float = 0.15
    direction_arrow_offset_ratio: float = 0.03
    direction_arrow_span_ratio: float = 0.12


def plot_matrix(  # noqa: PLR0913
    zones: gpd.GeoSeries,
    matrix: pd.DataFrame,
    demand_threshold: float,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    direction_inputs: DirectionInputs | None = None,
    logo: style.LogoInput | None = None,
    output_path: Path | None = None,
    plot_title: str | None = None,
    legend_title: str | None = None,
    total_title: str | None = None,
    line_widths: list[int] | int | None = None,
    alphas: list[float] | float | None = None,
    annotation_text: str = "Source: Transport for the North",
    positive_colour: str = style.TEAL,
    negative_colour: str = style.ORANGE,
    unit: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot matrix values spatially as flow lines between zones.

    Parameters
    ----------
    zones : gpd.GeoSeries
        Zoning system of the Matrix. Indices should be zone IDs.
    matrix : pd.DataFrame
        Matrix in long format, with columns in the following order: origin zone ID, destination
        zone ID, number of trips.
    demand_threshold : float
        Minimum number of trips required for a flow to be plotted.
    bounds : tuple[float, float, float, float] | None, optional
        Bounding box for the plot in the format (min x, min y, max x, max y).
        If None, no bounding is used for the plot.
    direction_inputs : DirectionInputs | None, optional
        Inputs controlling the appearance of directional arrows and curves, by default None
    logo : style.LogoInput | None, optional
        Logo to be added to the plot, by default None
    output_path : Path | None, optional
        Path to save the plot, by default None
    legend_title : str | None, optional
        Title for the plot legend, by default None
    total_title : str | None, optional
        Title for the total demand annotation, by default None
    line_widths : list[int] | int | None, optional
        Line widths for the flow lines, by default None
    alphas : list[float] | float | None, optional
        Transparency levels for the flow lines, by default None
    annotation_text : str, optional
        Text for the plot annotation, by default "Source: Transport for the North"
    positive_colour : str, optional
        Colour for positive flows, by default style.TEAL
    negative_colour : str, optional
        Colour for negative flows, by default style.ORANGE
    unit : str, optional
        Unit of measurement for the flows, by default ""

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes of the generated plot.
    """
    # multiple line width and alphas to stack plots to create a glow effect
    line_widths, alphas = _validate_line_settings(line_widths, alphas)

    centroids = zones.centroid
    o = centroids.copy()
    o.index.name = "o"
    o.name = "geometry"
    d = centroids.copy()
    d.index.name = "d"
    d.name = "geometry"
    matrix.columns = ["o", "d", "trips"]
    matrix = matrix.set_index(["o", "d"])
    line_matrix = matrix.join(o, how="right").join(d, rsuffix="_o", lsuffix="_d", how="right")

    total_demand = matrix["trips"].abs().sum()

    trunc_line = line_matrix.loc[
        (line_matrix["trips"] > demand_threshold) | (line_matrix["trips"] < -demand_threshold)
    ].copy()
    trunc_line.loc[:, "geometry"] = trunc_line.loc[:, "geometry_o"].combine(
        trunc_line.loc[:, "geometry_d"], lambda p1, p2: LineString([p1, p2])
    )
    trunc_line = gpd.GeoDataFrame(trunc_line[["trips", "geometry"]])
    intras = trunc_line.loc[
        trunc_line.index.get_level_values(0) == trunc_line.index.get_level_values(1)
    ].reset_index(level=0, drop=True)
    intras = centroids.to_frame().join(intras["trips"], how="right")
    # Split intras into positive and negative
    pos_intras = intras[intras["trips"] > 0] if not intras.empty else intras
    neg_intras = intras[intras["trips"] < 0] if not intras.empty else intras
    inters = trunc_line.loc[
        trunc_line.index.get_level_values(0) != trunc_line.index.get_level_values(1)
    ]

    inters_plot = inters.copy()
    if direction_inputs is not None and not inters_plot.empty:
        inters_plot["geometry"] = inters_plot["geometry"].apply(
            lambda g: _left_curve(g, direction_inputs.curve_left_ratio)
        )

    # trunc_line: GeoDataFrame with 'geometry' (LineString) and 'trips' columns
    # Split inter-zonal flows into positive and negative
    pos_inters = inters_plot[inters_plot["trips"] > 0]
    pos_inters = pos_inters.nlargest(3000, "trips")
    neg_inters = inters_plot[inters_plot["trips"] < 0]
    # ------------------------------------------------------------------
    # Robust normalisation using 95th percentile clipping
    # ------------------------------------------------------------------

    all_inters = pd.concat(
        [
            pos_inters["trips"] if not pos_inters.empty else pd.Series(dtype=float),
            (abs(neg_inters["trips"]) if not neg_inters.empty else pd.Series(dtype=float)),
        ]
    )

    vmax = np.percentile(all_inters, 99) if len(all_inters) else 1

    norm = Normalize(vmin=demand_threshold, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(8, 10), facecolor=style.NAVY)
    ax.set_facecolor(style.NAVY)
    ax.axis("off")

    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

    # Add geometry boundaries to plot
    zones.boundary.plot(ax=ax, color="#b0b8c0", linewidth=0.4, alpha=0.6, zorder=0)

    _plot_inters(
        ax,
        pos_inters,
        line_widths=line_widths,
        alphas=alphas,
        normalisation_fn=norm,
        colour=positive_colour,
    )
    _plot_inters(
        ax,
        neg_inters,
        line_widths=line_widths,
        alphas=alphas,
        normalisation_fn=norm,
        colour=negative_colour,
    )
    _plot_intras(
        ax,
        neg_intras,
        line_widths=line_widths,
        alphas=alphas,
        normalisation_fn=norm,
        colour=negative_colour,
    )
    _plot_intras(
        ax,
        pos_intras,
        line_widths=line_widths,
        alphas=alphas,
        normalisation_fn=norm,
        colour=positive_colour,
    )

    if direction_inputs is not None:
        _add_half_arrows(
            ax,
            pos_inters,
            positive_colour,
            norm,
            direction_min_normalized=direction_inputs.direction_min_normalized,
            direction_arrow_span_ratio=direction_inputs.direction_arrow_span_ratio,
            direction_arrow_offset_ratio=direction_inputs.direction_arrow_offset_ratio,
            direction_arrow_scale=direction_inputs.direction_arrow_scale,
            direction_arrow_alpha=direction_inputs.direction_arrow_alpha,
            is_negative=False,
        )
        _add_half_arrows(
            ax,
            neg_inters,
            negative_colour,
            norm,
            direction_min_normalized=direction_inputs.direction_min_normalized,
            direction_arrow_span_ratio=direction_inputs.direction_arrow_span_ratio,
            direction_arrow_offset_ratio=direction_inputs.direction_arrow_offset_ratio,
            direction_arrow_scale=direction_inputs.direction_arrow_scale,
            direction_arrow_alpha=direction_inputs.direction_arrow_alpha,
            is_negative=True,
        )

    # Add legend for demand (positive and negative)

    y_base = 0.88
    y_step = 0.04

    # --- Build legend values in correct order ---
    legend_vals = {
        "Min": 0 if math.isnan(inters["trips"].min()) else inters["trips"].min(),
        "Max": 0 if math.isnan(inters["trips"].max()) else inters["trips"].max(),
    }

    # --- Plot legend ---
    for i, (label, val) in enumerate(legend_vals.items()):
        y = y_base - y_step * i
        if val > 0:
            colour = positive_colour
            prefix = "+"
        else:
            colour = negative_colour
            prefix = ""

        alpha = np.sqrt(norm(abs(val)))
        ax.scatter([0.05], [y], s=40, color=colour, alpha=alpha, lw=0, transform=ax.transAxes)
        ax.text(
            0.08,
            y,
            f"{label} {prefix}{_round_nice(val)} {unit}",
            color="white",
            va="center",
            ha="left",
            fontsize=10,
            transform=ax.transAxes,
        )

    if legend_title is not None:
        ax.text(
            0.0005,
            y_base + y_step * 1.2,
            legend_title,
            color="white",
            fontsize=11,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    # Add title and source
    if plot_title is not None:
        ax.set_title(
            plot_title,
            color="white",
            fontsize=18,
            fontweight="bold",
            pad=15,
            loc="center",
        )
    if logo is not None:
        ax = logo.add_logo(ax)
    # Add total absolute magnitude demand text box
    ax.text(
        0.0005,
        0.7,
        f"{total_title}:\n{'+' if total_demand > 0 else ''}{total_demand:,.0f} {unit}",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        ha="left",
        va="bottom",
        zorder=20,
    )
    ax.text(
        1 - logo.logo_pad if logo is not None else 1,
        logo.logo_pad - 0.02 if logo is not None else 0.02,
        annotation_text,
        color="white",
        fontsize=9,
        ha="right",
        va="top",
        transform=ax.transAxes,
    )

    # Optionally add a PNG logo to the bottom-right corner.

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    plt.close(fig)
    return fig, ax


def _round_nice(val: float) -> int:
    if val == 0:
        return 0
    exp = int(np.floor(np.log10(abs(val))))
    base = np.round(val / 10**exp) * 10**exp
    return int(base)
