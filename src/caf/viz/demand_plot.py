import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.colors import Normalize, to_rgba
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from src.caf.viz import tfn_constants

def plot_demand(
    centroids: gpd.GeoSeries,
    id_col : str,
    zones: gpd.GeoSeries,
    matrix: pd.DataFrame,
    demand_threshold: float,
    lws: list[int],
    alphas: list[float],
    plot_nodes: bool = True,
    show_direction: bool = False,
    curve_left_ratio: float = 0.08,
    direction_arrow_alpha: float = 0.8,
    direction_arrow_scale: float = 9,
    direction_min_normalized: float = 0.15,
    direction_arrow_offset_ratio: float = 0.03,
    direction_arrow_span_ratio: float = 0.12,
    logo_path: Path = None,
    logo_zoom: float = 0.4,
    logo_alpha: float = 1.0,
    logo_pad: float = 0.02,
    output_path: Path = None,
    show: bool = False,
    return_fig: bool = False,
    plot_title: str = None,
    legend_title: str = None
):
    nodes = centroids.set_index(id_col)['geometry']
    o = nodes.copy()
    o.index.name = 'o'
    d = nodes.copy()
    d.index.name='d'

    matrix.columns = ['o', 'd', 'trips']
    matrix = matrix.set_index(['o', 'd'])
    line_matrix = matrix.join(o, how='right').join(d, rsuffix='_o', lsuffix='_d', how='right')

    trunc_line = line_matrix[
        (line_matrix['trips'] > demand_threshold) |
        (line_matrix['trips'] < -demand_threshold)
        ]
    trunc_line['geometry'] = trunc_line['geometry_o'].combine(
        trunc_line['geometry_d'],
        lambda p1, p2: LineString([p1, p2])
    )
    trunc_line = gpd.GeoDataFrame(trunc_line[['trips','geometry']])
    intras = trunc_line.loc[trunc_line.index.get_level_values(0) == trunc_line.index.get_level_values(1)].reset_index(level=0, drop=True)
    intras = nodes.to_frame().join(intras['trips'], how='right')
    # Split intras into positive and negative
    pos_intras = intras[intras['trips'] > 0] if not intras.empty else intras
    neg_intras = intras[intras['trips'] < 0] if not intras.empty else intras
    inters = trunc_line.loc[trunc_line.index.get_level_values(0) != trunc_line.index.get_level_values(1)]

    def left_curve(geom: LineString, curve_ratio: float, n_points: int = 30) -> LineString:
        coords = list(geom.coords)
        if len(coords) < 2:
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
        return LineString(zip(xs, ys))

    inters_plot = inters.copy()
    if show_direction and not inters_plot.empty:
        inters_plot['geometry'] = inters_plot['geometry'].apply(lambda g: left_curve(g, curve_left_ratio))

    # trunc_line: GeoDataFrame with 'geometry' (LineString) and 'trips' columns
    # Split inter-zonal flows into positive and negative
    pos_inters = inters_plot[inters_plot['trips'] > 0]
    neg_inters = inters_plot[inters_plot['trips'] < 0]
    # Normalizers for positive and negative
    max_pos = max(
        pos_inters['trips'].max() if not pos_inters.empty else 0,
        pos_intras['trips'].max() if not pos_intras.empty else 0
    )

    norm_pos = Normalize(vmin=0, vmax=max_pos if max_pos > 0 else 1, clip=True)
    max_neg = max(
        abs(neg_inters['trips'].min()) if not neg_inters.empty else 0,
        abs(neg_intras['trips'].min()) if not neg_intras.empty else 0
    )

    norm_neg = Normalize(vmin=0, vmax=max_neg if max_neg > 0 else 1, clip=True)

    fig, ax = plt.subplots(figsize=(8, 10), facecolor=tfn_constants.NAVY)
    ax.set_facecolor(tfn_constants.NAVY)
    ax.axis('off')

    # Add geometry boundaries to plot
    zones.boundary.plot(
        ax=ax,
        color="#b0b8c0",
        linewidth=0.4,
        alpha=0.2,
        zorder=0
    )

    # Colors for positive and negative flows
    POS_COLOR = tfn_constants.TEAL
    NEG_COLOR = tfn_constants.ORANGE  # Or another contrasting color

    # Plot positive flows with glow effect
    for lw, alpha in zip(lws, alphas):
        if not pos_inters.empty:
            alpha_plot = alpha * norm_pos(pos_inters['trips'])
            pos_inters.plot(ax=ax, color=POS_COLOR, alpha=alpha_plot, linewidth=lw, zorder=1, rasterized=True)

    # Plot negative flows with glow effect
    for lw, alpha in zip(lws, alphas):
        if not neg_inters.empty:
            alpha_plot = alpha * np.sqrt(norm_neg(abs(neg_inters['trips'])))
            neg_inters.plot(ax=ax, color=NEG_COLOR, alpha=alpha_plot, linewidth=lw, zorder=1, rasterized=True)

    # Optionally, plot nodes (intras) split by sign
    if not pos_intras.empty:
        for lw, alpha in zip(lws, alphas):
            pos_intras.plot(ax=ax, color=POS_COLOR, markersize=lw/2, alpha=norm_pos(pos_intras['trips']) * alpha, zorder=2, rasterized=True)
    if not neg_intras.empty:
        for lw, alpha in zip(lws, alphas):
            neg_intras.plot(ax=ax, color=NEG_COLOR, markersize=lw/2, alpha=norm_neg(abs(neg_intras['trips'])) * alpha, zorder=2, rasterized=True)

    if show_direction:
        def add_half_arrows(gdf, color, norm_fn, is_negative=False):
            if gdf.empty:
                return

            for _, row in gdf.iterrows():
                geom = row['geometry']
                if geom is None or geom.is_empty:
                    continue

                coords = np.array(geom.coords)
                if len(coords) < 3:
                    continue

                trip_val = row['trips']
                norm_val = np.sqrt(norm_fn(abs(trip_val))) if is_negative else norm_fn(trip_val)
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
                    '',
                    xy=head,
                    xytext=tail,
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        lw=1,
                        mutation_scale=direction_arrow_scale,
                        alpha=direction_arrow_alpha * norm_val,
                    ),
                    zorder=3,
                )

        add_half_arrows(pos_inters, POS_COLOR, norm_pos, is_negative=False)
        add_half_arrows(neg_inters, NEG_COLOR, norm_neg, is_negative=True)

    # Add legend for demand (positive and negative)
    def round_nice(val):
        if val == 0: return 0
        exp = int(np.floor(np.log10(abs(val))))
        base = np.round(val / 10**exp) * 10**exp
        return int(base)

    y_base = 0.88
    y_step = 0.04
    idx = 0

    # --- Build legend values in correct order ---
    legend_vals_pos = []
    legend_vals_neg = []

    if not pos_inters.empty:
        legend_vals_pos = np.linspace(
            pos_inters['trips'].min(),
            pos_inters['trips'].max(),
            3
        )

    if not neg_inters.empty:
        legend_vals_neg = np.linspace(
            neg_inters['trips'].min(),
            neg_inters['trips'].max(),
            3
        )

    # Combine: negatives first, then positives
    legend_vals = list(legend_vals_neg) + list(legend_vals_pos)

    # --- Plot legend ---
    idx = 0
    for val in legend_vals:

        y = y_base - y_step * idx
        nice_val = round_nice(val)

        if val >= 0:
            color = POS_COLOR
            alpha_val = norm_pos(val)
            label = f"+{nice_val}"
        else:
            color = NEG_COLOR
            alpha_val = np.sqrt(norm_neg(abs(val)))
            label = f"{nice_val}"

        ax.scatter(
            [0.05],
            [y],
            s=40,
            color=color,
            alpha=alpha_val,
            lw=0,
            transform=ax.transAxes
        )

        ax.text(
            0.08,
            y,
            label,
            color="white",
            va='center',
            ha='left',
            fontsize=12,
            transform=ax.transAxes
        )

        idx += 1

    if legend_title is not None:
        ax.text(
            0.05, y_base + y_step * 2, legend_title,
            color="white",
            fontsize=11,
            ha='left',
            va='top',
            transform=ax.transAxes
        )

    # Add title and source
    if plot_title is not None:
        ax.set_title(
            plot_title,
            color='white',
            fontsize=18,
            fontweight='bold',
            pad=15,
            loc='center'
        )

    ax.text(1- logo_pad, logo_pad - 0.02, "Source: Transport for the North", color='white', fontsize=9, ha='right', va='top', transform=ax.transAxes)

    # Optionally add a PNG logo to the bottom-right corner.
    if logo_path is not None:
        logo_path = Path(logo_path)
        if not logo_path.exists():
            raise FileNotFoundError(f"logo_path does not exist: {logo_path}")
        logo_img = mpimg.imread(logo_path)
        logo_box = OffsetImage(logo_img, zoom=logo_zoom, alpha=logo_alpha)
        logo_artist = AnnotationBbox(
            logo_box,
            (1 - logo_pad, logo_pad),
            xycoords='axes fraction',
            frameon=False,
            box_alignment=(1, 0),
            zorder=10,
        )
        ax.add_artist(logo_artist)

    if show:
        plt.show()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    if return_fig:
        return fig, ax
    return None

if __name__ == '__main__':
    # Inputs to change for Caf.Viz
    # Provide zone system nodes shape file
    gdf = gpd.read_file(r"G:\policy-builder\inputs\Zoning\NoHAM_zones_v3.8\NoHAM_Zones_v3.8_nodes_tfn.shp")
    # Provide node ID column name
    gdf_id_col = "ZONE ID_v3"
    # Provide zone system shapefile
    gdf_zone = gpd.read_file(r"G:\policy-builder\inputs\Zoning\NoHAM_zones_v3.8\NoHAM_Zones_v3.8_tfn.shp")

    # Provide path to demand file
    demand = pd.read_csv(r"G:\policy-builder\inputs\NoHAM_sqaure_matrices\base_2023_am_1_demand.csv")

    # Provide location of logo image
    logo = r"G:\policy-builder\CaS_Plots_Work\tfn-logo.png"
    # Provide a plot title
    title = "Policy Builder Demand Change"
    # Provide a plot legend title
    legend = "Demand change\n(trips)"

    # Provide output path
    output_path = r"E:\Policy Builder\Caf.Vis test\plot.pdf"

    # run plot_demand
    plot_demand(gdf, gdf_id_col, gdf_zone, demand,
        1, [9,6,3], [0.15,0.32,0.7], True, True, 0.08, 0.8, 9, 0.15, 0.03, 0.12, logo,
        plot_title=title, legend_title=legend, output_path=Path(output_path), show=False, return_fig=False)
