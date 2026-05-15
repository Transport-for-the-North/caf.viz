import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgba
from pathlib import Path
import tfn_constants

def plot_demand(
    centroids: gpd.GeoSeries,
    matrix_path: Path,
    demand_threshold: float,
    plot_nodes: bool = True,
    output_path: Path = None,
    show: bool = False,
    return_fig: bool = False,
):
    nodes = centroids.set_index('normits_id')['geometry']
    o = nodes.copy()
    o.index.name = 'o'
    d = nodes.copy()
    d.index.name='d'

    matrix = pd.read_csv(matrix_path, index_col=[0,1], names=['o','d','trips'])
    line_matrix = matrix.join(o, how='right').join(d, rsuffix='_o', lsuffix='_d', how='right')

    trunc_line = line_matrix[line_matrix['trips'] > demand_threshold]
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

    # trunc_line: GeoDataFrame with 'geometry' (LineString) and 'trips' columns
    # Split inter-zonal flows into positive and negative
    pos_inters = inters[inters['trips'] > 0]
    neg_inters = inters[inters['trips'] < 0]
    # Normalizers for positive and negative
    norm_pos = Normalize(vmin=0, vmax=pos_inters['trips'].max() if not pos_inters.empty else 1)
    norm_neg = Normalize(vmin=neg_inters['trips'].min() if not neg_inters.empty else -1, vmax=0)


    fig, ax = plt.subplots(figsize=(8, 10), facecolor=tfn_constants.NAVY)
    ax.set_facecolor(tfn_constants.NAVY)
    ax.axis('off')

    # Colors for positive and negative flows
    POS_COLOR = tfn_constants.TEAL
    NEG_COLOR = tfn_constants.ORANGE  # Or another contrasting color

    # Plot positive flows with glow effect
    for lw, alpha in zip([12, 8, 4], [0.1, 0.2, 0.6]):
        if not pos_inters.empty:
            alpha_plot = alpha * norm_pos(pos_inters['trips'])
            pos_inters.plot(ax=ax, color=POS_COLOR, alpha=alpha_plot, linewidth=lw, zorder=1)

    # Plot negative flows with glow effect
    for lw, alpha in zip([12, 8, 4], [0.1, 0.2, 0.6]):
        if not neg_inters.empty:
            alpha_plot = alpha * np.abs(norm_neg(neg_inters['trips']))
            neg_inters.plot(ax=ax, color=NEG_COLOR, alpha=alpha_plot, linewidth=lw, zorder=1)

    # Optionally, plot nodes (intras) split by sign
    if not pos_intras.empty:
        pos_intras.plot(ax=ax, color=POS_COLOR, markersize=3, alpha=norm_pos(pos_intras['trips']) * 0.5, zorder=2)
    if not neg_intras.empty:
        neg_intras.plot(ax=ax, color=NEG_COLOR, markersize=3, alpha=np.abs(norm_neg(neg_intras['trips'])) * 0.5, zorder=2)

    # Add legend for demand (positive and negative)
    def round_nice(val):
        if val == 0: return 0
        exp = int(np.floor(np.log10(abs(val))))
        base = np.round(val / 10**exp) * 10**exp
        return int(base)

    y_base = 0.95
    y_step = 0.05
    idx = 0
    # Positive legend
    if not pos_inters.empty:
        legend_vals_pos = np.linspace(pos_inters['trips'].min(), pos_inters['trips'].max(), 3)
        for val in legend_vals_pos:
            y = y_base - y_step * idx
            nice_val = round_nice(val)
            ax.scatter([0.05], [y], s=40, color=POS_COLOR, alpha=norm_pos(val), lw=0, transform=ax.transAxes)
            ax.text(0.08, y, f"+{nice_val}", color='white', va='center', ha='left', fontsize=12, transform=ax.transAxes)
            idx += 1
    # Negative legend
    if not neg_inters.empty:
        legend_vals_neg = np.linspace(neg_inters['trips'].min(), neg_inters['trips'].max(), 3)
        for val in legend_vals_neg:
            y = y_base - y_step * idx
            nice_val = round_nice(val)
            ax.scatter([0.05], [y], s=40, color=NEG_COLOR, alpha=np.abs(norm_neg(val)), lw=0, transform=ax.transAxes)
            ax.text(0.08, y, f"{nice_val}", color='white', va='center', ha='left', fontsize=12, transform=ax.transAxes)
            idx += 1
    ax.text(0.05, 0.99, "no. of commuters (pos/neg)", color='white', fontsize=11, ha='left', va='top', transform=ax.transAxes)

    # Add title and source
    ax.set_title("Normits Demand\nHighways AM commute", color='white', fontsize=18, fontweight='bold', pad=30, loc='center')
    ax.text(0.05, 0.02, "Source: Your Data\nMore info here", color='white', fontsize=9, ha='left', va='bottom', transform=ax.transAxes)

    if show:
        plt.show()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    if return_fig:
        return fig, ax
    return None

if __name__ == '__main__':
    gdf = gpd.read_file(r"Y:\Data Strategy\GIS Shapefiles\NorMITs 2024 zone system\NorMITs zone\v3.3\NorMITs_zoning_v3.3_node.shp")
    gdf = gdf[gdf['gor'].isin([1,2,3])]

    plot_demand(gdf, Path(r"I:\NorMITs Distribution\voa_gb_2023_uni\car\p1\gm_m3_p1_hb_fr_ts1_estTrip.csv.bz2"),
                1, True, Path(r"C:\Users\IsaacScott\projects\viz\plot.png"), True, False)