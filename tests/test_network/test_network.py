import pytest
import circe as ci
import circe.downloads
import anndata as ad
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

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


# Add region annotations in AnnData.var dataframe
def test_annotation_functions():
    # Does it 'inplace'
    ci.add_region_infos(atac, inplace=True)
    # Returns a new AnnData object
    ci.add_region_infos(atac, inplace=False)

    # Wrong name (number of elements)
    with pytest.raises(Exception) as ValueError:
        ci.add_region_infos(atac, sep=("-", "-"))


def test_network_atac():
    # Add region annotations in AnnData.var dataframe
    ci.add_region_infos(atac)

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

    # Test on different initilisation strategies
    ci.sliding_graphical_lasso(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=2,
        n_samples_maxtry=10,
        max_alpha_iteration=2,
        init_method="kendalltau"
    )
    ci.sliding_graphical_lasso(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=2,
        n_samples_maxtry=10,
        max_alpha_iteration=2,
        init_method="spearman"
    )
    ci.sliding_graphical_lasso(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=2,
        n_samples_maxtry=10,
        max_alpha_iteration=2,
        init_method="cov"
    )
    ci.sliding_graphical_lasso(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=2,
        n_samples_maxtry=10,
        max_alpha_iteration=2,
        init_method="corrcoef"
    )

    # Test on sparse matrix
    atac.X = sp.sparse.csr_matrix(atac.X)
    ci.compute_atac_network(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=2,
        n_samples_maxtry=10,
        max_alpha_iteration=50
    )

    # Test calculate alpha if chromosome sizes is given
    chromosome_sizes = {f"chr{i}": 10_000_000 for i in range(1, nb_chr+1)}
    ci.circe.average_alpha(
        atac,
        window_size=distance_threshold,
        unit_distance=800,
        distance_constraint=distance_threshold/2,
        n_samples=2,
        n_samples_maxtry=10,
        max_alpha_iteration=2,
        chromosomes_sizes=chromosome_sizes
    )

    # Test calculate alpha n_samples too high for number of window found
    ci.circe.average_alpha(
        atac,
        window_size=distance_threshold,
        unit_distance=800,
        distance_constraint=distance_threshold/2,
        n_samples=2000,
        n_samples_maxtry=1800,
        max_alpha_iteration=2,
    )

    # Error if only one region is given
    with pytest.raises(Exception) as ValueError:
        ci.compute_atac_network(atac[:, 1])

    # Error if only one sample is given
    with pytest.raises(Exception) as ValueError:
        ci.compute_atac_network(atac[1, :])

    # Extract from AnnData.varp the dataframe listing the edges (peak1 - peak2)
    # Names are sorted by alphabetical order (Peak1 < Peak2)
    ci.extract_atac_links(atac)
    ci.extract_atac_links(atac, key="atac_network")

    # Test error if many varp keys are present but no slot precised
    atac.varp['atac_network2'] = atac.varp['atac_network']
    with pytest.raises(Exception):
        ci.extract_atac_links(atac)

    # Test if many varp keys are present and good one given, on a dense and csc matrix formats
    atac.varp["atac_network2"] = sp.sparse.csc_matrix(atac.varp["atac_network2"])
    atac_df = ci.extract_atac_links(atac, key="atac_network2")

    atac.varp["atac_network2"] = atac.varp["atac_network2"].toarray()
    atac_df = ci.extract_atac_links(atac, key="atac_network2")

    # Test if key given is wrong
    with pytest.raises(Exception) as KeyError:
        ci.extract_atac_links(atac, key="wrong_key")

    ci.draw.plot_connections(
        atac_df,
        "chr1",
        1e5,
        3e5,
        sep=('_', '_'))

    fig, ax = plt.subplots(1)
    ci.draw.plot_connections(
        atac_df,
        "chr1",
        1e5,
        3e5,
        abs_threshold=0.0,
        sep=('_', '_'),
        ax=ax)

    # Test if wrong chromosome name is given
    with pytest.raises(Exception) as ValueError:
        ci.draw.plot_connections(atac_df, "chr100", 1e5, 3e5, sep=('_', '_'))

    # Test if wrong start or end column name is given
    with pytest.raises(Exception) as ValueError:
        ci.draw.plot_connections(atac_df, "chr1", 0, 0, sep=('_', '_'))

    # Download gene annotations
    genes_df = ci.downloads.download_genes()

    fig, ax = plt.subplots(1, figsize=(20, 6))
    ci.draw.plot_connections_genes(
        connections=atac_df,
        genes=genes_df,
        chromosome="chr1",
        start=50_000,
        end=350_000,
        gene_spacing=30_000,
        y_lim_top=-0.01,
        abs_threshold=0.0,
        track_spacing=0.01,
        track_width=0.01,
        legend=True,
        ax=ax
    )

    with pytest.raises(Exception) as ValueError:
        ci.draw.plot_connections_genes(
            connections=atac_df,
            genes=genes_df,
            chromosome="chr100",
            start=50_000,
            end=350_000,
        )

    with pytest.raises(Exception) as ValueError:
        ci.draw.plot_connections_genes(
            connections=atac,
            genes=genes_df[-5:],
            chromosome="chr1",
            start=350_000,
            end=50_000,
        )