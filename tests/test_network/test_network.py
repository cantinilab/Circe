import os
import pytest
import atacnet.pyquic
print(atacnet.pyquic)
print(os.getcwd())

import atacnet as an
print(an.pyquic)

import anndata as ad
import scipy as sp
import numpy as np
import pandas as pd

print(an)

# Create fake single-cell atac-seq data
nb_cells = 100
nb_chr = 3
nb_regions_per_chr = 100
between_reg = 2000
size_reg = 50

counts = []  # Create one DataFrame (cells x regions) per chromosome
for chr in range(nb_chr):
    counts.append(pd.DataFrame(np.random.randint(0,100, size=(nb_cells, nb_regions_per_chr)),
                        index=['Cell_'+j for j in map(str, range(nb_cells))],
                        columns=['chr'+str(chr)+'_'+str(i)+'_'+str(i+size_reg) for i in range(1, nb_regions_per_chr*between_reg+1, between_reg)]))
atac = ad.AnnData(pd.concat(counts, axis=1))  # Create AnnData object from the dataframes

distance_threshold = 50000


# Add region annotations in AnnData.var dataframe
def test_annotation_functions():
    # Does it 'inplace'
    an.add_region_infos(atac)
    # Returns a new AnnData object
    an.add_region_infos(atac, inplace=False)

    # Wrong name (number of elements)
    with pytest.raises(Exception) as ValueError:
        an.add_region_infos(atac, sep=("-", "-"))


def test_network_atac():
    # Add region annotations in AnnData.var dataframe
    an.add_region_infos(atac)

    # Compute network and add it directly in AnnData.varp attribute
    an.compute_atac_network(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=50,
        n_samples_maxtry=100,
        max_alpha_iteration=60
    )

    # Test on different initilisation strategies
    an.sliding_graphical_lasso(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=5,
        n_samples_maxtry=10,
        max_alpha_iteration=10,
        init_method="kendalltau"
    )
    an.sliding_graphical_lasso(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=5,
        n_samples_maxtry=10,
        max_alpha_iteration=10,
        init_method="spearman"
    )
    an.sliding_graphical_lasso(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=5,
        n_samples_maxtry=10,
        max_alpha_iteration=10,
        init_method="cov"
    )
    an.sliding_graphical_lasso(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=5,
        n_samples_maxtry=10,
        max_alpha_iteration=10,
        init_method="corrcoef"
    )

    # Test on sparse matrix
    atac.X = sp.sparse.csr_matrix(atac.X)
    an.compute_atac_network(
        atac,
        window_size=distance_threshold,
        unit_distance=1000,
        distance_constraint=distance_threshold/2,
        n_samples=50,
        n_samples_maxtry=100,
        max_alpha_iteration=60
    )

    # Test calculate alpha if chromosome sizes is given
    chromosome_sizes = {f"chr{i}": 10_000_000 for i in range(0, nb_chr)}
    an.atacnet.average_alpha(
        atac,
        window_size=distance_threshold,
        unit_distance=800,
        distance_constraint=distance_threshold/2,
        n_samples=50,
        n_samples_maxtry=100,
        max_alpha_iteration=60,
        chromosomes_sizes=chromosome_sizes
    )

    # Test calculate alpha n_samples too high for number of window found
    an.atacnet.average_alpha(
        atac,
        window_size=distance_threshold,
        unit_distance=800,
        distance_constraint=distance_threshold/2,
        n_samples=200,
        n_samples_maxtry=100,
        max_alpha_iteration=60,
    )

    # Extract from AnnData.varp the dataframe listing the edges (peak1 - peak2)
    # Names are sorted by alphabetical order (Peak1 < Peak2)
    an.extract_atac_links(atac)
    an.extract_atac_links(atac, key="atac_network")

    # Test error if many varp keys are present but no slot precised
    atac.varp['atac_network2'] = atac.varp['atac_network']
    with pytest.raises(Exception):
        an.extract_atac_links(atac)

    # Test if many varp keys are present and good one given
    an.extract_atac_links(atac, key="atac_network2")

    # Test if key given is wrong
    with pytest.raises(Exception) as KeyError:
        an.extract_atac_links(atac, key="wrong_key")
