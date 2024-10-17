import pytest
import circe as ci
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from circe.ccan_module import find_ccan_cutoff, number_of_ccans, make_ccan_graph, find_ccans, add_ccans
from anndata import AnnData

print(ci)

# Create fake single-cell atac-seq data
nb_cells = 50
nb_chr = 1
nb_regions_per_chr = 100
between_reg = 2000
size_reg = 50

counts = []  # Create one DataFrame (cells x regions) per chromosome
for chr in range(1, nb_chr+1):
    counts.append(pd.DataFrame(
        np.random.randint(0, 100, size=(nb_cells, nb_regions_per_chr)),
        index=['Cell_'+j for j in map(str, range(nb_cells))],
        columns=['chr'+str(chr)+'_'+str(i)+'_'+str(i+size_reg)
                 for i in range(
                     1,
                     nb_regions_per_chr*between_reg+1,
                     between_reg)]))
atac = ad.AnnData(pd.concat(counts, axis=1))  # Create AnnData object from the dfs
distance_threshold = 50000


atac = ci.add_region_infos(atac)

# Compute network and add it directly in AnnData.varp attribute
ci.compute_atac_network(
    atac,
    window_size=distance_threshold,
    unit_distance=1000,
    distance_constraint=distance_threshold/2,
    n_samples=2,
    n_samples_maxtry=10,
    max_alpha_iteration=2
)

df_connections = ci.extract_atac_links(atac)


# Test find_ccan_cutoff function
def test_find_ccan_cutoff_basic():
    connection_df = pd.DataFrame({
        'Peak1': ['chr1_100_200', 'chr1_150_250'],
        'Peak2': ['chr1_300_400', 'chr1_350_450'],
        'score': [0.4, 0.8]
    })
    result = find_ccan_cutoff(connection_df, tolerance_digits=2)
    print(result)
    assert 0 <= result <= 1, "The cutoff value should be between 0 and 1"

def test_find_ccan_cutoff_no_positive_scores():
    connection_df = pd.DataFrame({
        'Peak1': ['chr1_100_200', 'chr1_150_250'],
        'Peak2': ['chr1_300_400', 'chr1_350_450'],
        'score': [-0.6, -0.5]
    })
    df_ccans = find_ccans(connection_df, coaccess_cutoff_override=0.8)
    assert df_ccans.empty, "No CCANs should be found when all scores are negative"

# Test number_of_ccans function
def test_number_of_ccans_basic():
    connection_df = pd.DataFrame({
        'Peak1': ['chr1_100_200', 'chr1_150_250'],
        'Peak2': ['chr1_300_400', 'chr1_350_450'],
        'score': [0.5, 0.8]
    })
    result = number_of_ccans(connection_df, 0.4)
    assert isinstance(result, int), "The result should be an integer representing the number of CCANs"
    assert result >= 0, "The number of CCANs should be non-negative"

# Test make_ccan_graph function
def test_make_ccan_graph():
    connection_df = pd.DataFrame({
        'Peak1': ['chr1_100_200', 'chr1_150_250'],
        'Peak2': ['chr1_300_400', 'chr1_350_450'],
        'score': [0.5, 0.8]
    })
    graph = make_ccan_graph(connection_df, coaccess_cutoff=0.4)
    assert isinstance(graph, nx.Graph), "The function should return a networkx Graph"
    assert len(graph.nodes()) > 0, "The graph should have at least one node"
    assert len(graph.edges()) > 0, "The graph should have at least one edge"

def test_make_ccan_graph_no_edges():
    connection_df = pd.DataFrame({
        'Peak1': ['chr1_100_200', 'chr1_150_250'],
        'Peak2': ['chr1_300_400', 'chr1_350_450'],
        'score': [0.1, 0.2]
    })
    graph = make_ccan_graph(connection_df, coaccess_cutoff=0.5)
    assert len(graph.edges()) == 0, "Graph should have no edges when cutoff is high"

# Test find_ccans function
def test_find_ccans_basic():
    connection_df = pd.DataFrame({
        'Peak1': ['chr1_100_200', 'chr1_150_250'],
        'Peak2': ['chr1_300_400', 'chr1_350_450'],
        'score': [0.5, 0.8]
    })
    df_ccans = find_ccans(connection_df)
    assert isinstance(df_ccans, pd.DataFrame), "The function should return a DataFrame"
    assert 'CCAN' in df_ccans.columns, "The DataFrame should contain a 'CCAN' column"

def test_find_ccans_with_override():
    connection_df = pd.DataFrame({
        'Peak1': ['chr1_100_200', 'chr1_150_250'],
        'Peak2': ['chr1_300_400', 'chr1_350_450'],
        'score': [0.5, 0.8]
    })
    df_ccans = find_ccans(connection_df, coaccess_cutoff_override=0.3)
    print(df_ccans)

# Test add_ccans function
def test_add_ccans_basic():
    
    adata_with_ccans = add_ccans(atac)
    assert 'CCAN' in adata_with_ccans.var.columns, "The 'CCAN' column should be added to the var slot of AnnData"


# Group 1: Input Validation and Error Handling

def test_plot_connections_empty_df():
    empty_df = pd.DataFrame(columns=["Peak1", "Peak2", "score"])
    fig, ax = plt.subplots(1)
    with pytest.raises(ValueError, match="Couldn't find connections"):
        ci.draw.plot_connections(empty_df, "chr1", 1e5, 3e5, ax=ax)

def test_plot_connections_missing_columns():
    atac_df = pd.DataFrame({
        "Peak1": ["chr1_100_200", "chr1_150_250"],
        "score": [0.5, 0.8]
    })
    fig, ax = plt.subplots(1)
    with pytest.raises(ValueError, match="Peak2 must be present"):
        ci.draw.plot_connections(atac_df, "chr1", 1e5, 3e5, ax=ax)

def test_plot_connections_invalid_boundaries():
    atac_df = pd.DataFrame({
        "Peak1": ["chr1_100_200", "chr1_150_250"],
        "Peak2": ["chr1_300_400", "chr1_350_450"],
        "score": [0.5, 0.8]
    })
    fig, ax = plt.subplots(1)
    with pytest.raises(ValueError, match="Couldn't find connections"):
        ci.draw.plot_connections(atac_df, "chr1", 500000, 600000, ax=ax)

def test_plot_connections_anndata_default_columns():

    fig, ax = plt.subplots(1)
    with pytest.raises(UserWarning):
        ci.draw.plot_connections(atac, "chr1", None, 10000, ax=ax)
    
    ci.draw.plot_connections(atac, "chr1", None, 10000, ax=ax)    


# Group 2: Parameter Testing and Variations

def test_plot_connections_threshold():
    atac_df = pd.DataFrame({
        "Peak1": ["chr1_100_200", "chr1_150_250"],
        "Peak2": ["chr1_300_400", "chr1_350_450"],
        "score": [0.2, 0.5]
    })
    fig, ax = plt.subplots(1)
    ci.draw.plot_connections(atac_df, "chr1", 100, 300, abs_threshold=0.3, ax=ax)
    assert len(ax.patches) == 2  # Only 1 connection should be plotted

def test_plot_connections_only_positive():
    atac_df = pd.DataFrame({
        "Peak1": ["chr1_100_200", "chr1_150_250"],
        "Peak2": ["chr1_300_400", "chr1_350_450"],
        "score": [-0.6, 0.8]
    })
    fig, ax = plt.subplots(1)
    ci.draw.plot_connections(atac_df, "chr1", 100, 300, only_positive=True, ax=ax)
    assert len(ax.patches) == 2  # Only 1 positive connection should be plotted

def test_plot_connections_custom_ax_labels():
    atac_df = pd.DataFrame({
        "Peak1": ["chr1_100_200", "chr1_150_250"],
        "Peak2": ["chr1_300_400", "chr1_350_450"],
        "score": [0.5, 0.8]
    })
    fig, ax = plt.subplots(1)
    ci.draw.plot_connections(atac_df, "chr1", 100, 300, ax_labels=False, ax=ax)
    assert ax.get_ylabel() == ''  # Axis labels should be turned off

# Group 3: Visual and Plot Customization

def test_plot_connections_custom_figsize():
    atac_df = pd.DataFrame({
        "Peak1": ["chr1_100_200", "chr1_150_250"],
        "Peak2": ["chr1_300_400", "chr1_350_450"],
        "score": [0.5, 0.8]
    })
    fig, ax = plt.subplots(1)
    ci.draw.plot_connections(atac_df, "chr1", 100, 300, figsize=(10, 4), ax=ax)


def test_plot_connections_transparency_by_score():
    atac_df = pd.DataFrame({
        "Peak1": ["chr1_100_200", "chr1_150_250"],
        "Peak2": ["chr1_300_400", "chr1_350_450"],
        "score": [0.1, 0.9]
    })
    fig, ax = plt.subplots(1)
    ci.draw.plot_connections(atac_df, "chr1", 100, 300, transparency_by_score=True, ax=ax)

def test_plot_connections_custom_arc_params():
    atac_df = pd.DataFrame({
        "Peak1": ["chr1_100_200", "chr1_150_250"],
        "Peak2": ["chr1_300_400", "chr1_350_450"],
        "score": [0.5, 0.8]
    })
    fig, ax = plt.subplots(1)
    ci.draw.plot_connections(atac_df, "chr1", 100, 300, arc_params={'lw': 5}, ax=ax)
    for patch in ax.patches:
        assert patch.get_linewidth() == 5  # Custom linewidth should be applied
