import sys
syspath = sys.path
sys.path  = [path for path in sys.path if "python" in path]
import atacnet as an
import anndata as ad
import numpy as np
import pandas as pd
import os
print(sys.path)
sys.path = syspath
print(an)
#os.chdir('atacnet/')

# Create fake single-cell atac-seq data
nb_cells = 300
nb_chr = 10
nb_regions_per_chr = 200
between_reg = 2000
size_reg = 50

counts = []  # Create one DataFrame (cells x regions) per chromosome
for chr in range(nb_chr):
    counts.append(pd.DataFrame(np.random.randint(0,100, size=(nb_cells, nb_regions_per_chr)),
                        index=['Cell_'+j for j in map(str, range(nb_cells))],
                        columns=['chr'+str(chr)+'_'+str(i)+'_'+str(i+size_reg) for i in range(1, nb_regions_per_chr*between_reg+1, between_reg)]))
atac = ad.AnnData(pd.concat(counts, axis=1))  # Create AnnData object from the dataframes

distance_threshold = 50000

def test_annotation_functions():
  # Add region annotations in AnnData.var dataframe
  an.add_region_infos(atac)


def test_network_atac():
    # Add region annotations in AnnData.var dataframe
    an.add_region_infos(atac)

    # Compute network and add it directly in AnnData.varp attribute
    an.compute_atac_network(
        atac, #metacells,
        window_size=distance_threshold,
        unit_distance = 1000,
        distance_constraint=distance_threshold/2,
        n_samples=50,
        n_samples_maxtry=100,
        max_alpha_iteration=60
    )

    # Extract from AnnData.varp the dataframe listing the edges (peak1 - peak2)
    # Names are sorted by alphabetical order (Peak1 < Peak2)
    an.extract_atac_links(atac)
