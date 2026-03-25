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
        np.random.randint(0, 100, size=(nb_cells, nb_regions_per_chr)),
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

    # Empty region (column of zeros) should raise ValueError
    atac_empty = atac.copy()
    if sp.sparse.issparse(atac_empty.X):
        X = atac_empty.X.toarray()
    else:
        X = atac_empty.X.copy()
    X[:, 0] = 0
    atac_empty.X = X
    with pytest.raises(Exception, match="empty regions"):
        circe.metacells.compute_metacells(atac_empty, k=10)
    
    circe.metacells.compute_metacells(atac, k=10, method="sum", projection=None)

    # Compute metacells from sparse matrix
    atac.X = sp.sparse.csr_matrix(atac.X)
    circe.metacells.compute_metacells(atac, k=10, method="sum", projection=None)


def subset_regions():
    circe.metacells.subset_regions(atac, chr='chr1', start=1, end=1000)


def test_metacells_extra_branches():
    """Cover max_metacells early-exit (line 93) and overlap detection (lines 98-99)."""
    # Reset to dense in case previous test left it sparse
    if sp.sparse.issparse(atac.X):
        atac.X = atac.X.toarray()

    # max_metacells=1 forces the early break at line 93
    result = circe.metacells.compute_metacells(atac, k=10, max_metacells=1)
    assert result.n_obs == 1

    # max_overlap_metacells=0.0 makes every candidate fail the overlap check
    # (any shared cells → no_overlap=False, hitting lines 98-99)
    result2 = circe.metacells.compute_metacells(
        atac, k=10, max_overlap_metacells=0.0
    )
    assert result2.n_obs >= 1


def test_metacells_custom_obsm():
    """Custom dim_reduction via adata.obsm covers lines 68-70."""
    if sp.sparse.issparse(atac.X):
        atac.X = atac.X.toarray()
    # Pre-compute LSI and store under a custom key
    circe.metacells.lsi(atac)
    atac.obsm["my_lsi"] = atac.obsm["X_lsi"]
    result = circe.metacells.compute_metacells(
        atac, k=10, dim_reduction="my_lsi"
    )
    assert result.n_obs >= 1