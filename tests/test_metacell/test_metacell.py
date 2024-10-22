import anndata as ad
import numpy as np
import pandas as pd
import circe.metacells
import scipy as sp
import pytest

# Create fake single-cell atac-seq data
nb_cells = 30
nb_chr = 1
nb_regions_per_chr = 40
between_reg = 2000
size_reg = 50

counts = []  # Create one DataFrame (cells x regions) per chromosome
for chr in range(1, nb_chr+1):
    counts.append(pd.DataFrame(
        np.random.randint(0,100, size=(nb_cells, nb_regions_per_chr)),
        index=['Cell_'+j for j in map(str, range(nb_cells))],
        columns=['chr'+str(chr)+'_'+str(i)+'_'+str(i+size_reg)
                 for i in range(
                     1, nb_regions_per_chr*between_reg+1, between_reg)]))
atac = ad.AnnData(pd.concat(counts, axis=1))  # Create AnnData object from dfs
distance_threshold = 50000


def test_metacells():
    # Compute metacells
    circe.metacells.compute_metacells(atac, k=10)
    with pytest.raises(Exception) as ValueError:
        circe.metacells.compute_metacells(atac, dim_reduction="wrong_method")
    with pytest.raises(Exception) as ValueError:
        circe.metacells.compute_metacells(atac, projection="wrong_method")
    
    circe.metacells.compute_metacells(atac, k=10, method="sum", projection=None)

    # Compute metacells from sparse matrix
    atac.X = sp.sparse.csr_matrix(atac.X)
    circe.metacells.compute_metacells(atac, k=10, method="sum", projection=None)


def subset_regions():
    circe.metacells.subset_regions(atac, chr='chr1', start=1, end=1000)