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


    # trunc_line: GeoDataFrame with 'geometry' (LineString) and 'trips' columns
    norm = Normalize(vmin=0, vmax=trunc_line['trips'].max())

    fig, ax = plt.subplots(figsize=(8, 10), facecolor=tfn_constants.NAVY)
    ax.set_facecolor(tfn_constants.NAVY)
    ax.axis('off')

    # Vectorized plotting of lines with glow effect
    for lw, alpha in zip([8, 4, 2], [0.15, 0.3, 0.9]):
        plotter = trunc_line.copy()
        colour = (0.5, 0.8, 1)
        plotter['alpha'] = alpha * norm(trunc_line['trips'])
        trunc_line.plot(ax=ax, color=tfn_constants.TEAL, alpha=plotter['alpha'], linewidth=lw, zorder=1)

    # Optionally, plot nodes if you have them
    # nodes.plot(ax=ax, color=tfn_constants.ICE, markersize=5, zorder=2)

    # Add legend for demand
    legend_vals = np.linspace(trunc_line['trips'][trunc_line['trips'] > 0].min(), trunc_line['trips'].max(), 4)
    def round_nice(val):
        if val == 0: return 0
        exp = int(np.floor(np.log10(val)))
        base = np.round(val / 10**exp) * 10**exp
        return int(base)
    for idx, val in enumerate(legend_vals):
        y = 0.95 - 0.05 * idx
        nice_val = round_nice(val)
        ax.scatter([0.05], [y], s=40, color=(0.7, 0.9, 1, norm(val)), lw=0, transform=ax.transAxes)
        ax.text(0.08, y, f"{nice_val}", color='white', va='center', ha='left', fontsize=12, transform=ax.transAxes)
    ax.text(0.05, 0.99, "no. of commuters", color='white', fontsize=11, ha='left', va='top', transform=ax.transAxes)

    # Add title and source
    ax.set_title("Normits Demand\nHighways AM commute", color='white', fontsize=18, fontweight='bold', pad=30, loc='center')
    ax.text(0.05, 0.02, "Source: Your Data\nMore info here", color='white', fontsize=9, ha='left', va='bottom', transform=ax.transAxes)


    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    if show:
        plt.show()
    if return_fig:
        return fig, ax
    return None

if __name__ == '__main__':
    gdf = gpd.read_file(r"Y:\Data Strategy\GIS Shapefiles\NorMITs 2024 zone system\NorMITs zone\v3.3\NorMITs_zoning_v3.3_node.shp")
    gdf = gdf[gdf['gor'].isin([1,2,3])]

    plot_demand(gdf, Path(r"I:\NorMITs Distribution\voa_gb_2023_uni\car\p1\gm_m3_p1_hb_fr_ts1_estTrip.csv.bz2"),
                1, Path(r"C:\Users\IsaacScott\projects\viz\plot.png"), True, False)