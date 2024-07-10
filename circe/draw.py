import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import numpy as np

def plot_connections(
    df_connections,
    chromosome,
    start,
    end,
    abs_threshold=0.4,
    sep=(':', '-'),
    transparency_by_score=True,
    width_by_score=True,
    figsize=(20, 6),
    ax=None,
    arc_params={},
    regions_params={}
    
):
    """
    Plot connections between regions in a chromosome window.

    Parameters
    ----------
    df_connections : pd.DataFrame
        DataFrame with columns ['Peak1', 'Peak2', 'score'].
    chromosome : str
        Chromosome name.
    start : int
        Start of the window.
    end : int
        End of the window.
    abs_threshold : float
        Minimum absolute value of the score to plot a connection.
    sep : tuple
        Tuple with two strings to separate the chromosome name from the start and end of the region.
    transparency_by_score : bool
        If True, the transparency of the arcs is proportional to the score.
    width_by_score : bool
        If True, the width of the arcs is proportional to the score.
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes
      Axes object.

    Returns
    -------
    None
        """

    df = df_connections.loc[df_connections['Peak1'].str.startswith(chromosome), :]
    df.loc[:, "Peak2"] = df["Peak2"].str.replace(sep[0], '-').str.replace(sep[1], '-')
    df.loc[:, "Peak1"] = df["Peak1"].str.replace(sep[0], '-').str.replace(sep[1], '-')

    if len(df)==0:
        raise ValueError(
            "Couldn't find connections with the parameter: chromosome={}".format(
                chromosome))

    lower_bound = df["Peak1"].str.split('-', expand=True)[2].astype(int).values
    upper_bound = df["Peak2"].str.split('-', expand=True)[1].astype(int).values
    df = df[((lower_bound >= start) & (lower_bound <= end))
            &
            ((upper_bound >= start) & (upper_bound <= end))]
    del upper_bound
    del lower_bound

    if len(df) == 0:
        raise ValueError(
            "Couldn't find connections with the parameter: chromosome={}, start={}, end={}".format(
                chromosome, start, end))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="#FFFFFF")

    df.loc[:, "Peak1_start"] = df["Peak1"].str.split(
        '-', expand=True)[1].astype(int)
    df.loc[:, "Peak1_end"] = df["Peak1"].str.split(
        '-', expand=True)[2].astype(int)
    df.loc[:, "Peak2_start"] = df["Peak2"].str.split(
        '-', expand=True)[1].astype(int)
    df.loc[:, "Peak2_end"] = df["Peak2"].str.split(
        '-', expand=True)[2].astype(int)

    coordinates = df.loc[:,
                         [
                             "Peak1_start",
                             "Peak1_end",
                             "Peak2_start",
                             "Peak2_end",
                             "score"
                             ]]
    max_height = np.abs(coordinates.iloc[:, 4]).max()
    print(max_height)

    for coordinate in coordinates.index:
        coords = coordinates.loc[coordinate, :].values
        if np.abs(coords[4]) < abs_threshold:
            continue

        center_arc = (coords[1] + coords[3] + coords[0] + coords[2])/4
        width_arc = (coords[3] + coords[2])/2 - (coords[1] + coords[0])/2
        height_arc = np.abs(coords[4]*2)
        alpha_arc = height_arc/(2*np.abs(max_height)) if transparency_by_score else 1
        lw_arc = height_arc*2/np.abs(max_height) if width_by_score else 4

        color = "#4584b6" if coords[4] > 0 else "#ffde57"

        ax.add_patch(
            Arc(
                (center_arc, 0),
                width=width_arc,
                height=height_arc,
                lw=lw_arc,
                edgecolor=color,
                facecolor='none',
                theta1=0,
                theta2=180,
                alpha=alpha_arc,
                **arc_params))

    regions = np.unique(np.concatenate([df["Peak1"].unique(), df["Peak2"].unique()]))
    for region in regions:
        start, end = region.split('-')[1:3]
        start, end = int(start), int(end)
        ax.add_patch(
            Rectangle(
                xy=(start, -0.05),
                height=0.03,
                width=end-start,
                facecolor="#646464",
                **regions_params))

    ax.set_xlim(coordinates.iloc[:, 0].min(), coordinates.iloc[:,3].max())
    ax.set_ylim(-0.05, np.abs(max_height * 1.1))
    ax.spines[['right', 'top', 'left']].set_visible(False)

    x_formatter = ScalarFormatter(useOffset=True, useMathText=True)
    x_formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_formatter('chr1-{x:1.0f}')

    ticks = [tick for tick in ax.get_yticks() if tick >= 0]
    ax.set_yticks(ticks)
    ax.plot(
        [coordinates.iloc[:, 0].min(), coordinates.iloc[:, 0].min()],
        [0, ticks[-1]],
        color="black"
    )
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=20)
    # ax.set_axis_off()

    # plt.show()
