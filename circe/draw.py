import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter
import anndata
import pandas as pd
import numpy as np
import warnings
from .circe import extract_atac_links, subset_region


def plot_genes(
    genes,
    chromosome,
    start,
    end,
    gene_spacing=100_000,
    ax=None,
    fontsize=10,
    track_width=0.1,
    track_spacing=0.15,
    name_padding=0.5,
    y_lim_top=-0.15,
    chromosome_col=None,
    start_col=None,
    end_col=None,
    name_col=None,
    strand_col=None,
):
    """
    Plots gene positions on a genome track with minimal overlap.

    Parameters:
    - genes: DataFrame with columns [
        'chromosome', 'start', 'end', 'genename', 'strand']
    - chromosome: Chromosome name
    - start, end: Genomic coordinates defining the region of interest
    - gene_spacing: Minimum distance between genes
    - ax: Matplotlib axis object (optional, if integrating with other plots)
    - fontsize: Font size for gene names
    - track_width: Height of gene tracks
    - track_spacing: Vertical spacing between gene tracks
    - name_padding: Extra padding to avoid overlap
        between gene names and gene bodies
    - y_lim_top: Top limit for the y-axis
    - chromosome_col: Column name for the chromosome
    - start_col: Column name for the start position
    - end_col: Column name for the end position
    - name_col: Column name for the gene name
    - strand_col: Column name for the gene strand
    """

    chromosome_col = 'chromosome' if chromosome_col is None else chromosome_col
    start_col = 'start' if start_col is None else start_col
    end_col = 'end' if end_col is None else end_col
    name_col = 'genename' if name_col is None else name_col
    strand_col = 'strand' if strand_col is None else strand_col
    required_columns = [
        chromosome_col, start_col, end_col, name_col, strand_col]
    for col in required_columns:
        if col not in genes.columns:
            raise ValueError(f"{col} must be present in the DataFrame.")

    genes = genes.loc[genes[chromosome_col] == chromosome, :]

    if len(genes) == 0:
        raise ValueError(
            "Couldn't find connections with the parameter:" +
            "chromosome={}".format(
                chromosome))

    genes = genes[
        ((genes[start_col] >= start)
            * (genes[end_col] <= end))
        +
        ((genes[start_col] >= start)
            * (genes[end_col] <= end))]

    if len(genes) == 0:
        raise ValueError(
            "Couldn't find connections with the parameter:" +
            "chromosome={}, start={}, end={}".format(
                chromosome, start, end))

    # Sort genes by start position
    genes = genes.sort_values(by=start_col)

    genes[name_col] = genes[name_col].fillna("unknown")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))

    tracks = []  # Each track is a list of (gene_start, gene_end, name_width)
    gene_positions = []  # Store final positions for plotting

    for i in genes.index:
        name, g_start, g_end, strand = genes.loc[
            i, [name_col, start_col, end_col, strand_col]]
        # Approximate text width in data coordinates
        name_width = len(name) * fontsize * 0.15
        # Dynamic padding based on name size
        padding = name_width + name_padding * fontsize
        assigned = False

        for i, track in enumerate(tracks):
            if all(
                g[1] + g[2] + name_padding <= g_start-gene_spacing
                    or g[0] >= g_end+gene_spacing + padding
                    for g in track):
                track.append((g_start, g_end, padding))
                gene_positions.append((
                    name, g_start, g_end, strand, i, padding))
                assigned = True
                break

        if not assigned:
            tracks.append([(g_start, g_end, padding)])
            gene_positions.append(
                (name, g_start, g_end, strand, len(tracks) - 1, padding))

    # Plot genes with names on the side
    for name, g_start, g_end, strand, track, padding in gene_positions:
        y_pos = -(track+1) * (track_width + track_spacing) + y_lim_top
        color = "#00A08A" if strand == "+" else "#F2AD00"

        # Draw gene body
        ax.add_patch(patches.Rectangle(
            xy=(g_start, y_pos),
            height=track_width,
            width=g_end-g_start,
            color=color
        ))

        # Position gene name dynamically based on padding
        name_x = g_end + name_padding if strand == "+" else g_start - padding
        ax.text(
            name_x,
            y_pos+track_width/2,
            "  " + name if strand == "+" else name,
            va="center",
            ha="left" if strand == "+" else "right",
            fontsize=fontsize)

    ax.set_xlim(start, end)
    ax.set_ylim(-(len(tracks))*(track_width+track_spacing) + y_lim_top, 0)
    ax.set_xlabel("Genomic Coordinate")

    if ax is None:
        plt.show()


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
    adaptative_x_axis=True,
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
    connections : pd.DataFrame | anndata.AnnData
        DataFrame with columns ['Peak1', 'Peak2', 'score'].
        If an AnnData object is passed, the columns are inferred from the
        object. (varp slot with varp parameter as key)
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
        Key to extract atac network if anndata object passed.

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
            key=varp,
            columns=[peak1_col, peak2_col, score_col])

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
    df = df_connections.copy()
    df.loc[:, peak2_col] = df[peak2_col].str.replace(sep[0], '-').str.replace(
        sep[1], '-')
    df.loc[:, peak1_col] = df[peak1_col].str.replace(sep[0], '-').str.replace(
        sep[1], '-')
    df = df.loc[df[peak1_col].str.startswith(
        chromosome+'-'), :]

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
            patches.Arc(
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
        start_region, end_region = region.split('-')[1:3]
        start_region, end_region = int(start_region), int(end_region)
        ax.add_patch(
            patches.Rectangle(
                xy=(start_region, -0.125*np.abs(max_height)),
                height=0.1*np.abs(max_height),
                width=end_region-start_region,
                facecolor="#646464",
                **regions_params))
    if adaptative_x_axis is True:
        ax.set_xlim(coordinates.iloc[:, 0].min(), coordinates.iloc[:, 3].max())
    else:
        ax.set_xlim(start, end)
    ax.set_ylim(-0.125*np.abs(max_height), np.abs(max_height * 1.1))
    ax.spines[['right', 'top', 'left']].set_visible(False)

    x_formatter = ScalarFormatter(useOffset=True, useMathText=True)
    x_formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_formatter(f'{chromosome}-{{x:.0f}}')

    ticks = [tick for tick in ax.get_yticks() if tick >= 0]
    ax.set_yticks(ticks)
    if adaptative_x_axis is True:
        ax.plot(
            [coordinates.iloc[:, 0].min(), coordinates.iloc[:, 0].min()],
            [0, ticks[-1]],
            color="black"
        )
    else:
        ax.plot(
            [start, start],
            [0, ticks[-1]],
            color="black"
        )
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=15)
    # ax.set_axis_off()
    if ax_labels:
        ax.set_ylabel("Co-accessibility", fontsize=14)
        ax.set_xlabel("Coordinates", fontsize=14)


def plot_connections_genes(
    connections,
    genes,
    chromosome,
    start,
    end,
    ax=None,
    legend=True,
    peak1_col=None,
    peak2_col=None,
    score_col=None,
    varp=None,
    chromosome_col=None,
    start_col=None,
    end_col=None,
    name_col=None,
    strand_col=None,
    abs_threshold=0.4,
    only_positive=False,
    sep=('_', '_'),
    ax_labels=True,
    transparency_by_score=True,
    adaptative_x_axis=False,
    width_by_score=True,
    figsize=(20, 10),
    arc_params={},
    regions_params={},
    gene_spacing=100_000,
    fontsize_genes=10,
    track_width=0.1,
    track_spacing=0.15,
    name_padding=0.5,
    y_lim_top=-0.15,
):
    """
    Plot connections between regions in a chromosome window and gene positions.

    Parameters
    ----------
    connections : pd.DataFrame | anndata.AnnData
        DataFrame with columns ['Peak1', 'Peak2', 'score'].
        If an AnnData object is passed, the columns are inferred from the
        object. (varp slot with varp parameter as key)
    genes : pd.DataFrame
        DataFrame with columns [
            'chromosome', 'start', 'end', 'genename', 'strand']
    chromosome : str
        Chromosome name.
    start : int
        Start of the window.
    end : int
        End of the window.
    ax : matplotlib.axes.Axes
        Axes object.
    legend : bool
        If True, a legend is added to the plot.
    peak1_col : str
        Column name for the first peak.
    peak2_col : str
        Column name for the second peak.
    score_col : str
        Column name for the score.
    varp : str
        Key to extract atac network if anndata object passed.
    chromosome_col : str

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="#FFFFFF")

    plot_connections(
        connections=connections,
        chromosome=chromosome,
        start=start,
        end=end,
        ax=ax,
        abs_threshold=abs_threshold,
        only_positive=only_positive,
        sep=sep,
        ax_labels=ax_labels,
        transparency_by_score=transparency_by_score,
        adaptative_x_axis=adaptative_x_axis,
        width_by_score=width_by_score,
        arc_params=arc_params,
        regions_params=regions_params,
        peak1_col=peak1_col,
        peak2_col=peak2_col,
        score_col=score_col,
        varp=varp
        )

    max = ax.get_ylim()[1]

    plot_genes(
        genes=genes,
        chromosome=chromosome,
        start=start,
        end=end,
        ax=ax,
        gene_spacing=gene_spacing,
        fontsize=fontsize_genes,
        track_width=track_width,
        track_spacing=track_spacing,
        name_padding=name_padding,
        y_lim_top=y_lim_top,
        chromosome_col=chromosome_col,
        start_col=start_col,
        end_col=end_col,
        name_col=name_col,
        strand_col=strand_col,
    )

    ax.set_ylim(None, max)
    ax.yaxis.set_label_coords(
        x=-0.04,
        y=(ax.get_ylim()[1]/2 - ax.get_ylim()[0])
        / (ax.get_ylim()[1] - ax.get_ylim()[0]))

    if legend:
        # Add 2 legends, for gene body strand and co-accessibility sign
        coaccess_items = {
            "Positive edge": "#4584b6",
            "Negative edge": "#ffde57",
        }
        strand_items = {
            "+": "#00A08A",
            "-": "#F2AD00",
        }

        patch_strand = [
            patches.Patch(color=color, label=label)
            for label, color in strand_items.items()]
        patch_coaccess = [
            patches.Patch(color=color, label=label)
            for label, color in coaccess_items.items()]

        legend_strand = ax.legend(
            handles=patch_strand,
            loc="lower right",
            title="Strand")
        legend_coaccess = ax.legend(
            handles=patch_coaccess,
            loc="upper right",
            title="Co-accessibility")

        ax.add_artist(legend_strand)
        ax.add_artist(legend_coaccess)


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
