import anndata as ad
import numpy as np
import pandas as pd

from circe.rank_correlation import _compute_ranks, kendalltau_correlation
from circe.quic_graph_lasso import QuicGraphicalLasso

import pytest


# Create fake single-cell atac-seq data
nb_cells = 50
nb_chr = 1
nb_regions_per_chr = 100
between_reg = 2000
size_reg = 50

counts = []  # Create one DataFrame (cells x regions) per chromosome
for chr in range(1, nb_chr+1):
    counts.append(pd.DataFrame(np.random.randint(0,100, size=(nb_cells, nb_regions_per_chr)),
                        index=['Cell_'+j for j in map(str, range(nb_cells))],
                        columns=['chr'+str(chr)+'_'+str(i)+'_'+str(i+size_reg) for i in range(1, nb_regions_per_chr*between_reg+1, between_reg)]))
atac = ad.AnnData(pd.concat(counts, axis=1))  # Create AnnData object from the dataframes

# test _compute_ranks and create a fake ground truth
_compute_ranks(atac.X, winsorize=True, truncation=None, verbose=True)
cov = np.cov(atac.X)

# Run model to get
graph_lasso_model = QuicGraphicalLasso(
    init_method="precomputed",
    tol=1e-4,
    max_iter=1e4,
    auto_scale=False,
    )

# Fit graphical lasso
graph_lasso_model.fit(cov)

# Test cov error implementations and if not implemented
graph_lasso_model.cov_error(
    comp_cov=cov,
    score_metric="frobenius")

graph_lasso_model.cov_error(
    comp_cov=cov,
    score_metric="spectral")

graph_lasso_model.cov_error(
    comp_cov=cov,
    score_metric="kl")

graph_lasso_model.cov_error(
    comp_cov=cov,
    score_metric="quadratic")

graph_lasso_model.cov_error(
    comp_cov=cov,
    score_metric="log_likelihood")

with pytest.raises(Exception) as NotImplementedError:
    graph_lasso_model.cov_error(
        comp_cov=cov,
        score_metric="not_implemented")


# Test grahical lasso with wrong lambda
with pytest.raises(Exception) as ValueError:
    graph_lasso_model = QuicGraphicalLasso(
        init_method="precomputed",
        tol=1e-4,
        max_iter=1e4,
        auto_scale=False,
        lam=np.array([1.0, 1.0])
        )
    graph_lasso_model.fit(cov)


# Test grahical lasso with wrong method
with pytest.raises(Exception) as ValueError:
    graph_lasso_model = QuicGraphicalLasso(
        init_method="wrong_method",
        tol=1e-4,
        max_iter=1e4,
        auto_scale=False,
        )
    graph_lasso_model.fit(cov)

# Run graphical lasso with 'path' method
graph_lasso_model = QuicGraphicalLasso(
    init_method="precomputed",
    tol=1e-4,
    max_iter=1e4,
    auto_scale=False,
    )
graph_lasso_model.fit(cov)

# Kendalltau correlation with additional params
kendalltau_correlation(cov, rowvar=True, weighted=True)