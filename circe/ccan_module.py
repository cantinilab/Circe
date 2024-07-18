import pandas  # For DataFrame manipulation
import networkx  # For graph creation and manipulation
import numpy  # For numerical operations
from .circe import extract_atac_links


def find_ccan_cutoff(connection_df, tolerance_digits=2, seed=42):
    """
    Find the optimal cutoff value for coaccess based on specified tolerance.

    This function iteratively searches for a cutoff value that results in
    a significant change in the number of co-accessible networks (CCANs).

    Parameters
    ----------
    connection_df : pandas.DataFrame
        A DataFrame containing the connections, must include a 'score' column.

    tolerance_digits : int
        The number of decimal places to consider for determining the cutoff.

    Returns
    -------
    float
        The optimal cutoff value rounded
        to the specified number of decimal places.
    """

    connection_df = connection_df[connection_df['score'] > 0]
    tolerance = 10 ** (-tolerance_digits)
    bottom = 0
    top = 1

    while (top - bottom) > tolerance:
        test_val = bottom + round((top - bottom) / 2, tolerance_digits + 1)
        ccan_num_test = number_of_ccans(connection_df, test_val, seed=seed)
        next_step = test_val

        while ccan_num_test != 0:
            next_step += (top - bottom) / 10
            ccan_num_test2 = number_of_ccans(
                connection_df, next_step, seed=seed)
            if ccan_num_test2 != ccan_num_test:
                break

        if ccan_num_test == 0:
            top = test_val
        elif ccan_num_test > ccan_num_test2:
            top = test_val
        else:
            bottom = test_val

    return round((top + bottom) / 2, tolerance_digits)


def number_of_ccans(connections_df, coaccess_cutoff, seed=42):
    """
    Count the number of co-accessible networks (CCANs)
    based on the Louvain method.

    This function creates a graph from the given
    connections and applies the Louvain method to identify communities,
    then counts the number of communities with more than two members.

    Parameters
    ----------
    connections_df : pandas.DataFrame
        A DataFrame containing the connections,
        must include a 'coaccess' column.

    coaccess_cutoff : float
        The cutoff value for coaccess used in graph construction.

    Returns
    -------
    int
        The number of co-accessible networks (CCANs)
        with more than two members.
    """

    # Create the graph using the provided connections and cutoff
    ccan_graph = make_ccan_graph(connections_df, coaccess_cutoff)
    if ccan_graph is None:
        return 0

    # Apply the Louvain method for community detection
    partition = {}
    partition = networkx.community.louvain_communities(
        ccan_graph, weight='score', seed=seed)
    partition = {node: comp_n
                 for comp_n in range(len(partition))
                 for node in partition[comp_n]}

    # Count communities with more than two members
    community_sizes = {k: 0 for k in set(partition.values())}
    for member in partition.values():
        community_sizes[member] += 1

    return sum(size > 2 for size in community_sizes.values())


def make_ccan_graph(
        connections_df,
        coaccess_cutoff,
        peak1_col="Peak1",
        peak2_col="Peak2",
        score_col="score"
):
    """
    Create an undirected graph from a DataFrame of connections.

    This function generates an undirected graph
    based on the provided edge list, using specified columns
    for source, target, and edge weights. It filters connections
    based on a specified coaccess cutoff.

    Parameters
    ----------
    connections_df : pandas.DataFrame or dict
        A DataFrame or dictionary containing the edge list with connections.
        Must include columns for the source, target, and edge weights.

    coaccess_cutoff : float
        The threshold value for filtering edges based on coaccess.
        Only connections with coaccess greater than this value
        will be included in the graph.

    source_col : str, optional
        The name of the column in the DataFrame
        representing the source nodes (default is "Peak1").

    target_col : str, optional
        The name of the column in the DataFrame
        representing the target nodes (default is "Peak2").

    weight_col : str, optional
        The name of the column in the DataFrame
        representing the edge weights (default is "score").

    Returns
    -------
    networkx.Graph
        The resulting undirected graph object.

    Raises
    ------
    ValueError
        If no connections meet the coaccess cutoff criteria.
    """

    connections_df = pandas.DataFrame(connections_df)

    # Filter connections based on coaccess cutoff
    edges_df = connections_df[
        (connections_df[score_col].notna()) &
        (connections_df[score_col] > coaccess_cutoff)
    ]

    nodes = numpy.unique(
        pandas.concat([edges_df[peak1_col], edges_df[peak2_col]]))

    self_edges = pandas.DataFrame(
        {
            i: {
                peak1_col: nodes[i],
                peak2_col: nodes[i],
                score_col: 0
                }
            for i in range(len(nodes))
        }
    ).transpose()

    edges_df = pandas.concat([edges_df, self_edges])
    # Create an undirected graph
    site_graph = networkx.from_pandas_edgelist(
        edges_df,
        source=peak1_col,
        target=peak2_col,
        edge_attr=score_col,
        create_using=networkx.Graph
    )

    return site_graph


def find_ccans(
    connections_df,
    peak1_col='Peak1',
    peak2_col='Peak2',
    score_col='score',
    coaccess_cutoff_override=None,
    tolerance_digits=2,
    verbose=True,
    seed=42
):
    """
    Generate co-accessible networks (CCANs) from connection data.

    Parameters
    ----------
    connections_df : pandas.DataFrame
        A DataFrame containing specified columns for peaks and scores.

    peak1_col : str
        Name of the column for the first peak.

    peak2_col : str
        Name of the column for the second peak.

    score_col : str
        Name of the column for the score.

    coaccess_cutoff_override : float or None
        Optional cutoff for coaccessibility. Must be between 0 and 1.

    tolerance_digits : int
        The number of decimal places to consider for determining the cutoff.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with Peaks and their corresponding CCANs.
    """

    # Assertions
    assert isinstance(connections_df, pandas.DataFrame), \
        "connections_df must be a DataFrame"
    required_columns = [peak1_col, peak2_col, score_col]
    for col in required_columns:
        assert col in connections_df.columns, \
            f"{col} must be present in the DataFrame"

    assert isinstance(tolerance_digits, (int, float)), \
        "tolerance_digits must be a number"

    if coaccess_cutoff_override is not None:
        assert isinstance(coaccess_cutoff_override, (int, float)), \
            "coaccess_cutoff_override must be a number or None"
        assert 0 <= coaccess_cutoff_override <= 1, \
            "coaccess_cutoff_override must be between 0 and 1 (or None)"

    # Determine coaccess cutoff
    if coaccess_cutoff_override is not None:
        coaccess_cutoff = coaccess_cutoff_override
    else:
        coaccess_cutoff = find_ccan_cutoff(
            connections_df,
            tolerance_digits,
            seed=seed)

    print(f"Coaccessibility cutoff used: {coaccess_cutoff}")

    # Generate CCAN graph
    ccan_graph = make_ccan_graph(
        connections_df,
        coaccess_cutoff=coaccess_cutoff,
        peak1_col=peak1_col,
        peak2_col=peak2_col,
        score_col=score_col)

    # Apply Louvain method for community detection
    partition = {}
    partition = networkx.community.louvain_communities(
        ccan_graph, weight='score', seed=seed)
    partition = {node: comp_n
                 for comp_n in range(len(partition))
                 for node in partition[comp_n]}

    # Create output DataFrame
    df = pandas.DataFrame({
        'Peak': partition.keys(),
        'CCAN': partition.values()
        })
    sizes = df['CCAN'].value_counts()
    # Keep only CCANs of size > 2
    df = df[df['CCAN'].isin(sizes[sizes > 2].index)]

    if verbose:
        print(f"Number of CCANs generated: {df['CCAN'].nunique()}")

    return df.reset_index(drop=True)


def add_ccans(
    atac,
    peak1_col='Peak1',
    peak2_col='Peak2',
    score_col='score',
    coaccess_cutoff_override=None,
    tolerance_digits=2,
    verbose=False,
    seed=42,
    ccan_col='CCAN'
):
    """
    Extract co-accessible networks (CCANs) from an AnnData object
    and add them to the `var` slot.

    Parameters
    ----------
    atac : anndata.AnnData
        The AnnData object containing the data
        and the connections in the `var` slot.

    peak1_col : str
        Name of the column for the first peak.

    peak2_col : str
        Name of the column for the second peak.

    score_col : str
        Name of the column for the score.

    coaccess_cutoff_override : float or None
        Optional cutoff for coaccessibility. Must be between 0 and 1.

    tolerance_digits : int
        The number of decimal places to consider for determining the cutoff.

    verbose : bool
        If True, prints additional information during the process.

    Returns
    -------
    anndata.AnnData
        The modified AnnData object with CCANs added to `var`.
    """

    # Extract the connections DataFrame from the AnnData object
    connections_df = extract_atac_links(atac)

    # Generate CCANs
    df_ccans = find_ccans(
        connections_df,
        peak1_col=peak1_col,
        peak2_col=peak2_col,
        score_col=score_col,
        coaccess_cutoff_override=coaccess_cutoff_override,
        tolerance_digits=tolerance_digits,
        verbose=verbose,
        seed=seed
    )

    # Map CCANs back to the var slot of the AnnData object
    atac.var[ccan_col] = None  # Initialize CCAN column
    atac.var.loc[atac.var_names.isin(df_ccans['Peak']), ccan_col] = \
        df_ccans[ccan_col].values

    return atac
