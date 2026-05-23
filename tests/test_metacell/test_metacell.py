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


def _strategy_test_atac(sparse_input=False):
    """Small deterministic AnnData with a custom embedding for strategy tests."""
    rng = np.random.default_rng(0)
    X = rng.integers(1, 10, size=(12, 8))
    if sparse_input:
        X = sp.sparse.csr_matrix(X)
    adata = ad.AnnData(X)
    adata.var_names = [f"chr1_{i}_{i + size_reg}"
                       for i in range(1, 8 * between_reg + 1, between_reg)]
    adata.obsm["test_lsi"] = np.column_stack([
        np.arange(adata.n_obs),
        np.arange(adata.n_obs) % 3,
    ])
    return adata


def _as_array(X):
    return X.toarray() if sp.sparse.issparse(X) else X


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


@pytest.mark.parametrize("strategy", ["inverse_index", "vectorized", "spatial"])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_metacells_strategies(strategy, sparse_input):
    """All selectable filtering strategies should run and return metacells."""
    result = circe.metacells.compute_metacells(
        _strategy_test_atac(sparse_input=sparse_input),
        k=4,
        dim_reduction="test_lsi",
        strategy=strategy,
        nn_algorithm="kd_tree",
    )
    assert result.n_obs >= 1
    assert result.n_vars == 8
    assert sp.sparse.issparse(result.X) is sparse_input


@pytest.mark.parametrize("sparse_input", [False, True])
def test_metacells_exact_strategies_match(sparse_input):
    """inverse_index and vectorized are exact and should produce same output."""
    inverse = circe.metacells.compute_metacells(
        _strategy_test_atac(sparse_input=sparse_input),
        k=4,
        dim_reduction="test_lsi",
        strategy="inverse_index",
        nn_algorithm="kd_tree",
    )
    vectorized = circe.metacells.compute_metacells(
        _strategy_test_atac(sparse_input=sparse_input),
        k=4,
        dim_reduction="test_lsi",
        strategy="vectorized",
        nn_algorithm="kd_tree",
    )
    assert inverse.obs_names.tolist() == vectorized.obs_names.tolist()
    assert sp.sparse.issparse(inverse.X) is sparse_input
    assert sp.sparse.issparse(vectorized.X) is sparse_input
    np.testing.assert_array_equal(_as_array(inverse.X), _as_array(vectorized.X))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("threshold_frac", [0.3, 0.5, 0.7, 0.9])
def test_inverse_index_matches_vectorized_random(seed, threshold_frac):
    """Direct equivalence on synthetic neighbour indices across seeds and
    overlap thresholds. Covers the counting-array implementation of
    _select_metacells_inverse_index against the reference vectorized strategy.
    """
    rng = np.random.default_rng(seed)
    n_obs, k = 400, 30
    idx = np.stack(
        [rng.choice(n_obs, size=k, replace=False) for _ in range(n_obs)])
    idx[:, 0] = np.arange(n_obs)
    threshold = threshold_frac * k
    a = circe.metacells._select_metacells_inverse_index(
        idx, k, threshold, n_obs, n_obs)
    b = circe.metacells._select_metacells_vectorized(
        idx, k, threshold, n_obs, n_obs)
    assert a == b


def test_metacells_invalid_strategy():
    with pytest.raises(ValueError, match="strategy must be one of"):
        circe.metacells.compute_metacells(
            _strategy_test_atac(),
            k=4,
            dim_reduction="test_lsi",
            strategy="wrong_strategy",
        )


def test_metacells_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        circe.metacells.compute_metacells(
            _strategy_test_atac(),
            k=4,
            dim_reduction="test_lsi",
            mode="wrong_mode",
        )


def test_metacells_smoothing_mode_keeps_every_cell():
    """smoothing mode bypasses the overlap filter; output has n_obs rows."""
    adata = _strategy_test_atac()
    result = circe.metacells.compute_metacells(
        adata, k=4, dim_reduction="test_lsi", mode="smoothing",
    )
    assert result.n_obs == adata.n_obs
    assert result.n_vars == adata.n_vars


def test_metacells_smoothing_respects_max_metacells_cap():
    adata = _strategy_test_atac()
    result = circe.metacells.compute_metacells(
        adata, k=4, dim_reduction="test_lsi",
        mode="smoothing", max_metacells=5,
    )
    assert result.n_obs == 5


def test_metacells_community_requires_max_metacells():
    with pytest.raises(ValueError, match="max_metacells"):
        circe.metacells.compute_metacells(
            _strategy_test_atac(), k=4,
            dim_reduction="test_lsi", mode="community",
        )


def test_metacells_community_mode_runs():
    """End-to-end community mode. Skipped if leidenalg/igraph unavailable."""
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    adata = _strategy_test_atac()
    result = circe.metacells.compute_metacells(
        adata, k=4, dim_reduction="test_lsi",
        mode="community", max_metacells=3,
    )
    assert 1 <= result.n_obs <= adata.n_obs
    assert result.n_vars == adata.n_vars


def test_metacells_nn_jobs_argument():
    result = circe.metacells.compute_metacells(
        _strategy_test_atac(),
        k=4,
        dim_reduction="test_lsi",
        nn_jobs=1,
    )
    assert result.n_obs >= 1
