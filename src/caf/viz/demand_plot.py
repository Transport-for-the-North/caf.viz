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
    legend_title: str = None,
    total_title: str = None,
):
    nodes = centroids.set_index(id_col)['geometry']
    o = nodes.copy()
    o.index.name = 'o'
    d = nodes.copy()
    d.index.name='d'

    matrix.columns = ['o', 'd', 'trips']
    matrix = matrix.set_index(['o', 'd'])
    line_matrix = matrix.join(o, how='right').join(d, rsuffix='_o', lsuffix='_d', how='right')

    total_demand = matrix["trips"].abs().sum()

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
    pos_inters = pos_inters.nlargest(
        3000,
        "trips"
    )
    neg_inters = inters_plot[inters_plot['trips'] < 0]
    # ------------------------------------------------------------------
    # Robust normalisation using 95th percentile clipping
    # ------------------------------------------------------------------

    all_pos = pd.concat([
        pos_inters['trips'] if not pos_inters.empty else pd.Series(dtype=float),
        pos_intras['trips'] if not pos_intras.empty else pd.Series(dtype=float)
    ])

    all_neg = pd.concat([
        abs(neg_inters['trips']) if not neg_inters.empty else pd.Series(dtype=float),
        abs(neg_intras['trips']) if not neg_intras.empty else pd.Series(dtype=float)
    ])

    vmax_pos = np.percentile(all_pos, 99) if len(all_pos) else 1
    vmax_neg = np.percentile(all_neg, 99) if len(all_neg) else 1

    norm_pos = Normalize(
        vmin=demand_threshold,
        vmax=vmax_pos,
        clip=True
    )

    norm_neg = Normalize(
        vmin=demand_threshold,
        vmax=vmax_neg,
        clip=True
    )

    fig, ax = plt.subplots(figsize=(8, 10), facecolor=tfn_constants.NAVY)
    ax.set_facecolor(tfn_constants.NAVY)
    ax.axis('off')

    # Add geometry boundaries to plot
    zones.boundary.plot(
        ax=ax,
        color="#b0b8c0",
        linewidth=0.4,
        alpha=0.6,
        zorder=0
    )

    # Colors for positive and negative flows
    POS_COLOR = tfn_constants.TEAL
    NEG_COLOR = tfn_constants.ORANGE  # Or another contrasting color

    # Plot positive flows with glow effect
    for lw, alpha in zip(lws, alphas):
        if not pos_inters.empty:
            alpha_plot = (
                    alpha *
                    np.sqrt(
                        norm_pos(pos_inters['trips'])
                    )
            )

            pos_inters.plot(
                ax=ax,
                color=POS_COLOR,
                alpha=alpha_plot,
                linewidth=lw,
                zorder=1,
                rasterized=True
            )
    # Plot negative flows with glow effect
    for lw, alpha in zip(lws, alphas):
        if not neg_inters.empty:
            alpha_plot = (
                    alpha *
                    np.sqrt(
                        norm_neg(
                            abs(neg_inters['trips'])
                        )
                    )
            )

            neg_inters.plot(
                ax=ax,
                color=NEG_COLOR,
                alpha=alpha_plot,
                linewidth=lw,
                zorder=1,
                rasterized=True
            )

    # Optionally, plot nodes (intras) split by sign
    if not pos_intras.empty:
        for lw, alpha in zip(lws, alphas):
            pos_intras.plot(
                ax=ax,
                color=POS_COLOR,
                markersize=lw / 2,
                alpha=np.sqrt(
                    norm_pos(pos_intras['trips'])
                ) * alpha,
                zorder=2,
                rasterized=True
            )
    if not neg_intras.empty:
        for lw, alpha in zip(lws, alphas):
            neg_intras.plot(
                ax=ax,
                color=NEG_COLOR,
                markersize=lw / 2,
                alpha=np.sqrt(
                    norm_neg(
                        abs(neg_intras['trips'])
                    )
                ) * alpha,
                zorder=2,
                rasterized=True
            )
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
                if is_negative:
                    norm_val = np.sqrt(
                        norm_fn(abs(trip_val))
                    )
                else:
                    norm_val = np.sqrt(
                        norm_fn(trip_val)
                    )
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
            0.0005, y_base + y_step * 1.2, legend_title,
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

    # Add total absolute magnitude demand text box
    ax.text(
        0.0005, 0.7,
        f"{total_title}:\n{total_demand:,.0f}",
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        ha="left",
        va="bottom",
        zorder=20
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
    gdf = gpd.read_file(r"G:\policy-explorer\CaS_Plots_Work\Updates 070826\MSOAs_no_Externals_centroids.shp")
    # Provide node ID column name
    gdf_id_col = "Area"
    # Provide zone system shapefile
    gdf_zone = gpd.read_file(r"G:\policy-explorer\CaS_Plots_Work\Updates 070826\MSOAs_no_Externals.shp")

    # Provide path to demand file
    demand = pd.read_csv(r"E:\Policy Builder\Demand Matrices\Demand HWY MSOAs NO Externals.csv")

    # Provide location of logo image
    logo = r"G:\policy-explorer\CaS_Plots_Work\tfn-logo.png"
    # Provide a plot title
    title = "Policy Builder HWY Demand (WYCA Area Only)"
    # Provide a total title
    total_title = "Total Demand (trips)"
    # Provide a plot legend title
    legend = "Demand (trips):"

    # Provide output path
    output_path_1 = r"E:\Policy Builder\Caf.Vis test\Updated 082026\plot_msoas_no_externals_with_total.pdf"

    # Additional Inputs - SECTORIZED (Add if want to run otherwise set run_sectorised to 'No' in run_caf_vis_plot)
    # Provide matrix to sectorized matrix lookup
    # lookup = pd.read_csv(r"G:\policy-builder\inputs\Zoning\noham_3_8_to_noham_sector_v2_spatial.csv")
    # Provide zone system nodes shape file SECTORIZED
    # gdf_sector = gpd.read_file(r"G:\policy-builder\inputs\Zoning\NoHAM_sectors_v2\NoHAM_sector_system_V2_nodes_tfn.shp")
    # Provide node ID column name SECTORIZED
    # gdf_id_col_sector = "Code"
    # Provide zone system shapefileSECTORIZED
    # gdf_zone_sector = gpd.read_file(r"G:\policy-builder\inputs\Zoning\NoHAM_sectors_v2\NoHAM_sector_system_V2_tfn.shp")
    # Provide output path
    # output_path_2 = r"E:\Policy Builder\Caf.Vis test\plot_noham_3_8_sectors.pdf"