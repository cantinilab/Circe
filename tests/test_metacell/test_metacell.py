import os
import atacnet.pyquic
print(atacnet.pyquic)
print(os.getcwd())

import atacnet as an
print(an.pyquic)

import anndata as ad
import numpy as np
import pandas as pd
import atacnet.metacells

# Create fake single-cell atac-seq data
nb_cells = 100
nb_chr = 3
nb_regions_per_chr = 50
between_reg = 2000
size_reg = 50

counts = []  # Create one DataFrame (cells x regions) per chromosome
for chr in range(nb_chr):
    counts.append(pd.DataFrame(np.random.randint(0,100, size=(nb_cells, nb_regions_per_chr)),
                        index=['Cell_'+j for j in map(str, range(nb_cells))],
                        columns=['chr'+str(chr)+'_'+str(i)+'_'+str(i+size_reg) for i in range(1, nb_regions_per_chr*between_reg+1, between_reg)]))
atac = ad.AnnData(pd.concat(counts, axis=1))  # Create AnnData object from the dataframes

distance_threshold = 50000


def test_metacells():
    # Add region annotations in AnnData.var dataframe
    atacnet.metacells.compute_metacells(atac)
