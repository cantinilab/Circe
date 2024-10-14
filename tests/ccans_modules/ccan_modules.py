import pytest
import circe as ci
import anndata as ad
import numpy as np
import pandas as pd

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
