import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from matplotlib.ticker import ScalarFormatter
import anndata
import pandas as pd
import numpy as np
import warnings
from .circe import extract_atac_links, subset_region


def plot_connections(
    connections,
    chromosome,
    start,
    end,
    abs_threshold=0.4,
    only_positive=False,
    sep=('_', '_'),
    ax_labels=True,
    transparency_by_score=True,
    width_by_score=True,
    figsize=(20, 6),
    ax=None,
    arc_params={},
    regions_params={},
    peak1_col=None,
    peak2_col=None,
    score_col=None,
    varp=None
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
    only_positive : bool
        If True, only positive scores are plotted.
    sep : tuple
        Tuple with two strings to separate the chromosome name from the start
        and end of the region.
    ax_labels : bool
        If True, the axis labels are shown.
    transparency_by_score : bool
        If True, the transparency of the arcs is proportional to the score.
    width_by_score : bool
        If True, the width of the arcs is proportional to the score.
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes
      Axes object.
    arc_params : dict
        Parameters for the Arc patches.
    regions_params : dict
        Parameters for the Rectangle patches.
    peak1_col : str
        Column name for the first peak.
    peak2_col : str
        Column name for the second peak.
    score_col : str
        Column name for the score.
    varp : str
        Column name for the variance partitioning score.

    Returns
    -------
    None
    """

    if isinstance(connections, anndata.AnnData):
        # Handle the AnnData case
        if peak1_col is not None or \
           peak2_col is not None or \
           score_col is not None:
            warnings.warn(
                "peak1_col, peak2_col, and score_col parameters are ignored" +
                "for AnnData input.", UserWarning)

        # Assign default values to the columns
        peak1_col = 'Peak1'
        peak2_col = 'Peak2'
        score_col = 'score'
        # Extract connections
        df_connections = extract_atac_links(
            connections,
            key=None,
            columns=[peak1_col, peak2_col, score_col]
        )
    elif isinstance(connections, pd.DataFrame):
        # Handle the DataFrame case
        if varp is not None:
            warnings.warn(
                "varp parameter is ignored for DataFrame input.", UserWarning)
        df_connections = connections
        # Validate columns
        peak1_col = 'Peak1' if peak1_col is None else peak1_col
        peak2_col = 'Peak2' if peak2_col is None else peak2_col
        score_col = 'score' if score_col is None else score_col
        required_columns = [peak1_col, peak2_col, score_col]
        for col in required_columns:
            if col not in df_connections.columns:
                raise ValueError(f"{col} must be present in the DataFrame.")
    else:
        raise ValueError("Input must be either an AnnData object" +
                         "or a pandas DataFrame.")

    df = df_connections.loc[df_connections[peak1_col].str.startswith(
        chromosome), :]
    df.loc[:, peak2_col] = df[peak2_col].str.replace(
        sep[0], '-').str.replace(sep[1], '-')
    df.loc[:, peak1_col] = df[peak1_col].str.replace(
        sep[0], '-').str.replace(sep[1], '-')

    if len(df) == 0:
        raise ValueError(
            "Couldn't find connections with the parameter:" +
            "chromosome={}".format(
                chromosome))

    lower_bound = df[peak1_col].str.split(
        '-', expand=True)[2].astype(int).values
    upper_bound = df[peak2_col].str.split(
        '-', expand=True)[1].astype(int).values
    df = df[((lower_bound >= start) & (lower_bound <= end))
            &
            ((upper_bound >= start) & (upper_bound <= end))]
    del upper_bound
    del lower_bound

    if len(df) == 0:
        raise ValueError(
            "Couldn't find connections with the parameter:" +
            "chromosome={}, start={}, end={}".format(
                chromosome, start, end))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="#FFFFFF")

    df.loc[:, "Peak1_start"] = df[peak1_col].str.split(
        '-', expand=True)[1].astype(int)
    df.loc[:, "Peak1_end"] = df[peak1_col].str.split(
        '-', expand=True)[2].astype(int)
    df.loc[:, "Peak2_start"] = df[peak2_col].str.split(
        '-', expand=True)[1].astype(int)
    df.loc[:, "Peak2_end"] = df[peak2_col].str.split(
        '-', expand=True)[2].astype(int)

    coordinates = df.loc[:,
                         [
                             "Peak1_start",
                             "Peak1_end",
                             "Peak2_start",
                             "Peak2_end",
                             score_col
                             ]]
    max_height = np.abs(coordinates.iloc[:, 4]).max()

    for coordinate in coordinates.index:
        coords = coordinates.loc[coordinate, :].values
        if only_positive:
            if coords[4] < abs_threshold or coords[4] == 0:
                continue
        elif np.abs(coords[4]) < abs_threshold or coords[4] == 0:
            continue

        center_arc = (coords[1] + coords[3] + coords[0] + coords[2])/4
        width_arc = (coords[3] + coords[2])/2 - (coords[1] + coords[0])/2
        height_arc = np.abs(coords[4]*2)
        alpha_arc = height_arc/(2*np.abs(max_height)) \
            if transparency_by_score else 1
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

    regions = np.unique(np.concatenate(
        [df[peak1_col].unique(), df[peak2_col].unique()]
        ))
    for region in regions:
        start, end = region.split('-')[1:3]
        start, end = int(start), int(end)
        ax.add_patch(
            Rectangle(
                xy=(start, -0.125*np.abs(max_height)),
                height=0.1*np.abs(max_height),
                width=end-start,
                facecolor="#646464",
                **regions_params))

    ax.set_xlim(coordinates.iloc[:, 0].min(), coordinates.iloc[:, 3].max())
    ax.set_ylim(-0.125*np.abs(max_height), np.abs(max_height * 1.1))
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
    ax.tick_params(axis='y', which='major', labelsize=15)
    # ax.set_axis_off()
    if ax_labels:
        ax.set_ylabel("Co-accessibility", fontsize=20)
        ax.set_xlabel("Coordinates", fontsize=20)

    # plt.show()


def plot_ccan(
    adata,
    ccan_module,
    ccan_col="CCAN",
    varp="atac_network",
    **kwargs
):
    """
    Plot the connections of a ccan module.
    Only connections between 2 regions of the ccan module are plotted.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated anndata object.
        Should contain both a varp object that contains the
        cis-coaccessibility network and an obs column with the
        ccan module names.
    ccan_module : str
        Name of the ccan module to plot.
    ccan_col : str
        Column name for the ccan module in adata.var.
    varp : str
        Key for the peak-to-peak coaccessibility network in adata.varp.
    **kwargs : dict
        Additional parameters for plot_connections.

    Returns
    -------
    None
    """

    if ccan_col not in adata.var.columns:
        raise ValueError(f"Column {ccan_col} not found in adata.var.")

    # Define the ccan module mask
    ccan_mask = adata.var[ccan_col] == ccan_module
    # Check if any regions are found
    if adata[:, ccan_mask].X.shape[1] == 0:
        raise ValueError(f"No regions found for ccan module {ccan_module}.")

    # Extract the window to plot
    start = adata[:, ccan_mask].var["start"].min()
    end = adata[:, ccan_mask].var["end"].max()
    chromosome = adata[:, ccan_mask].var["chromosome"].unique()

    # CCAN module should be on a single chromosome
    if len(chromosome) != 1:
        raise ValueError("Multiple chromosomes found in ccan module.")
    chromosome = chromosome[0]
    print("This CCAN module is on the chromosome: {}".format(chromosome))

    # Subset the adata object
    ccan_adata = subset_region(
        adata,
        chromosome=chromosome,
        start=start,
        end=end
    )

    df_connections = extract_atac_links(
        ccan_adata,
        key=None,
        columns=["Peak1", "Peak2", "score"]
    )

    # Replace by 0 all links that are not between regions in the ccan module
    df_connections.loc[
        ~df_connections["Peak1"].isin(adata.var_names[ccan_mask]), "score"] = 0
    df_connections.loc[
        ~df_connections["Peak2"].isin(adata.var_names[ccan_mask]), "score"] = 0

    # Extract and plot connections
    plot_connections(
        df_connections,
        chromosome,
        start-(end-start)/100,
        end+(end-start)/100,
        varp=varp,
        **kwargs
    )
