import matplotlib.pyplot as plt
import pandas as pd

from caf.viz import xy_plot

plt.style.use("caf.viz.tfn")
data = pd.DataFrame(
    [
        {"name": "Steve", "points": 8},
        {"name": "Bob", "points": 4},
        {"name": "Kieran", "points": 5},
        {"name": "Mathew", "points": 1},
        {"name": "Ben", "points": 5},
        {"name": "Isaac", "points": 9},
        {"name": "Amber", "points": 9},
        {"name": "James", "points": 7},
    ],
)

fig = xy_plot.plot_xy(data, "name", "points", xy_plot.XYPlotType.BAR, title="Mathew Loses")
fig.show()

