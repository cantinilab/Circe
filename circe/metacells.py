import scipy as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import numpy as np
import anndata as ad
import scanpy as sc
from rich.progress import track

def compute_metacells_old(
        adata,
        k=50,
        max_overlap_metacells=0.9,
        max_metacells=None,
        dim_reduction='lsi',
        projection=None,
        method='mean',
        metric='cosine'
):
    """
    Compute metacells by suming/averaging expression of neighbouring cells.
    Neighbouring cells are identified from umap coordinates,
    itself obtained from LSI coordinates to reduce dimensionality to topics.
    (It has been implemented to be close to Cicero/Monocle3 code 
    from Trapnell lab).

    Parameters
    ----------
    adata : AnnData
        AnnData object
    k : int, optional
        Number of neighbours to consider.
        The default is 50.
    max_overlap_metacells : float, optional
        Maximum percentage of overlapping cells between two metacells
        in order to consider them as different.
        The default is 0.9.
    max_metacells : int, optional
        Maximum number of metacells to compute.
        The default is None, which means a metacell is created and centered on each cell.
    dim_reduction : str, optional
        Dimensionality reduction method to use to compute metacells.
        The default is 'lsi'.
    projection : str, optional
        Projection method to use to compute metacells.
        The default is None, which uses the dimensionality reduction
        coordinates directly (no UMAP projection).
    method : str, optional
        Method to use to compute metacells (mean or sum).
    metric : str, optional
        Distance type used to calculate distance between neighbors.
        The default is 'cosine'.

    Returns
    -------
    metacells_AnnData : AnnData
        AnnData object containing metacells.
    """
    # Verify not empty cells
    if adata.X.sum(axis=0).min() == 0:
        raise ValueError("adata.X contains empty regions." +
        "Please filter them before computing metacells." +
        "You can use sc.pp.filter_genes(adata, min_counts=1) to do so.")
    if dim_reduction == 'lsi':
        lsi(adata, tfidf_fn=tfidf_old)
        key_dim_reduction = f"X_{dim_reduction}"
    elif dim_reduction in adata.obsm.keys():
        key_dim_reduction = dim_reduction
        print(f"Using adata.obsm['{dim_reduction}'] to identify neighboring cells")
    else:
        raise ValueError(f"Only 'lsi' is implemented for now, and no adata.obsm['{dim_reduction}'] coordinates found.")

    if projection == 'umap':
        sc.pp.neighbors(adata, use_rep=key_dim_reduction, metric=metric)
        sc.tl.umap(adata)
        key_projection = 'X_umap'
    elif projection is None:
        key_projection = key_dim_reduction
    else:
        raise ValueError("Only 'umap' and None are implemented for now.")

    # Identify non-overlapping above a threshold metacells
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(adata.obsm[key_projection])
    distances, indices = nbrs.kneighbors(adata.obsm[key_projection])
    indices = [set(indice) for indice in indices]
    if max_metacells is None:
        max_metacells = len(indices)

    # Select metacells that doesn't overlap too much (percentage of same cells of origin  < max_overlap_metacells for each pair)
    metacells = [indices[0]]
    for idx, candidate in enumerate(track(indices[1:], description="Computing metacells..."), start=1):
        if idx >= max_metacells:
            break

        no_overlap = True
        for metacell in metacells:
            if len(metacell.intersection(candidate)) >= max_overlap_metacells * k:
                no_overlap = False
                break
        if no_overlap:
            metacells.append(candidate)

    # Sum expression of neighbors composing the metacell
    is_sparse = sp.sparse.issparse(adata.X)
    metacells_values = []
    for metacell in metacells:
        cell_indices = list(metacell)
        if method == 'mean':
            vals = adata.X[cell_indices].mean(axis=0)
            metacells_values.append(vals.A1 if is_sparse else vals)
        elif method == 'sum':
            vals = adata.X[cell_indices].sum(axis=0)
            metacells_values.append(vals.A1 if is_sparse else vals)

    # Create a new AnnData object from it
    metacells_AnnData = ad.AnnData(np.array(metacells_values))
    metacells_AnnData.var_names = adata.var_names
    metacells_AnnData.obs_names = [f"metacell_{i}" for i in range(
        len(metacells_values))]
    metacells_AnnData.var = adata.var.copy()
    metacells_AnnData.varp = adata.varp.copy()

    return metacells_AnnData


def _select_metacells_inverse_index(indices, k, threshold, max_metacells, n_obs):
    """
    Exact greedy selection of non-overlapping metacells using an inverse index.

    A cell -> kept-metacells index restricts each candidate's overlap test to
    the kept metacells that actually share cells with it. Each shared cell
    contributes the metacell's id once, so ``numpy.bincount`` of the gathered
    ids yields the overlap counts in C. Produces results identical to
    ``_select_metacells_vectorized``.

    Returns
    -------
    list of int
        Indices of the candidate cells kept as metacell centers.
    """
    inv_index = [[] for _ in range(n_obs)]
    kept = [0]
    for x in indices[0]:
        inv_index[x].append(0)

    for idx in track(range(1, n_obs),
                     description="Computing metacells (inverse index)..."):
        if idx >= max_metacells:
            break
        # Gather every kept metacell that shares a cell with this candidate;
        # an id appears once per shared cell, so bincount gives the overlaps.
        gathered = []
        for x in indices[idx]:
            gathered.extend(inv_index[x])
        if gathered:
            conflict = np.bincount(gathered).max() >= threshold
        else:
            conflict = False
        if not conflict:
            new_id = len(kept)
            kept.append(idx)
            for x in indices[idx]:
                inv_index[x].append(new_id)
    return kept


def _select_metacells_vectorized(indices, k, threshold, max_metacells, n_obs):
    """
    Exact greedy selection of non-overlapping metacells using a sparse
    membership matrix. The overlap of a candidate against all kept metacells
    is computed with a single sparse matrix product.

    Returns
    -------
    list of int
        Indices of the candidate cells kept as metacell centers.
    """
    membership = sp.sparse.csr_matrix(
        (np.ones(n_obs * k, dtype=np.int32),
         (np.repeat(np.arange(n_obs), k), indices.ravel())),
        shape=(n_obs, n_obs))
    kept = [0]
    for idx in track(range(1, n_obs),
                     description="Computing metacells (vectorized)..."):
        if idx >= max_metacells:
            break
        candidate = membership[idx]
        overlaps = membership[kept].dot(candidate.T)
        if overlaps.nnz == 0 or overlaps.max() < threshold:
            kept.append(idx)
    return kept


def _select_metacells_spatial(indices, k, threshold, max_metacells, n_obs):
    """
    Approximate greedy selection of non-overlapping metacells with spatial
    pruning.

    A candidate is only compared against kept metacells whose center cell is
    among the candidate's k nearest neighbours. When ``max_overlap_metacells``
    is high, two strongly-overlapping metacells almost always have mutually
    neighbouring centers, so the result is close to the exact selection -
    but it is NOT guaranteed identical.

    Returns
    -------
    list of int
        Indices of the candidate cells kept as metacell centers.
    """
    mark = np.zeros(n_obs, dtype=bool)
    center_to_metacell = {0: 0}
    kept = [0]
    for idx in track(range(1, n_obs),
                     description="Computing metacells (spatial)..."):
        if idx >= max_metacells:
            break
        candidate_cells = indices[idx]
        mark[candidate_cells] = True
        conflict = False
        for c in candidate_cells:
            j = center_to_metacell.get(int(c))
            if j is not None:
                overlap = mark[indices[kept[j]]].sum()
                if overlap >= threshold:
                    conflict = True
                    break
        mark[candidate_cells] = False
        if not conflict:
            center_to_metacell[idx] = len(kept)
            kept.append(idx)
    return kept


_FILTERING_STRATEGIES = {
    'inverse_index': _select_metacells_inverse_index,
    'vectorized': _select_metacells_vectorized,
    'spatial': _select_metacells_spatial,
}


def compute_metacells(
        adata,
        k=50,
        max_overlap_metacells=0.9,
        max_metacells=None,
        dim_reduction='lsi',
        projection=None,
        method='mean',
        metric='cosine',
        nn_algorithm='auto',
        strategy='inverse_index'
):
    """
    Compute metacells by suming/averaging expression of neighbouring cells.
    Neighbouring cells are identified from umap coordinates,
    itself obtained from LSI coordinates to reduce dimensionality to topics.
    (It has been implemented to be close to Cicero/Monocle3 code
    from Trapnell lab).

    This is an optimized reimplementation of :func:`compute_metacells_old`.
    The expression aggregation is vectorized with sparse matrix operations,
    the nearest-neighbour search strategy is configurable, and the
    non-overlap filtering can use several strategies (see ``strategy``).

    Parameters
    ----------
    adata : AnnData
        AnnData object
    k : int, optional
        Number of neighbours to consider.
        The default is 50.
    max_overlap_metacells : float, optional
        Maximum percentage of overlapping cells between two metacells
        in order to consider them as different.
        The default is 0.9.
    max_metacells : int, optional
        Maximum number of metacells to compute.
        The default is None, which means a metacell is created and centered on each cell.
    dim_reduction : str, optional
        Dimensionality reduction method to use to compute metacells.
        The default is 'lsi'.
    projection : str, optional
        Projection method to use to compute metacells.
        The default is None, which uses the dimensionality reduction
        coordinates directly (no UMAP projection).
    method : str, optional
        Method to use to compute metacells (mean or sum).
    metric : str, optional
        Distance type used to calculate distance between neighbors.
        The default is 'cosine'.
    nn_algorithm : str, optional
        Nearest-neighbour search strategy passed to
        :class:`sklearn.neighbors.NearestNeighbors` ('auto', 'ball_tree',
        'kd_tree' or 'brute'). The default is 'auto', which lets
        scikit-learn pick an appropriate strategy (the old implementation
        was hard-coded to 'kd_tree', which degrades in the ~20 dimensions
        of the LSI embedding).
    strategy : str, optional
        Non-overlap filtering strategy. The default is 'inverse_index'.

        - 'inverse_index' : exact, fastest. Uses a cell -> kept-metacells
          index so each candidate is only compared against metacells that
          actually share cells with it. Produces results identical to
          :func:`compute_metacells_old`.
        - 'vectorized' : exact. Computes overlaps with a sparse matrix
          product. Also identical to :func:`compute_metacells_old`.
        - 'spatial' : approximate, fast. Prunes comparisons using the
          nearest-neighbour structure of the metacell centers. Results are
          close to exact for a high ``max_overlap_metacells`` but NOT
          guaranteed identical.

    Returns
    -------
    metacells_AnnData : AnnData
        AnnData object containing metacells.
    """
    if strategy not in _FILTERING_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(_FILTERING_STRATEGIES)}.")
    # Verify not empty cells
    if adata.X.sum(axis=0).min() == 0:
        raise ValueError("adata.X contains empty regions." +
        "Please filter them before computing metacells." +
        "You can use sc.pp.filter_genes(adata, min_counts=1) to do so.")
    if dim_reduction == 'lsi':
        lsi(adata)
        key_dim_reduction = f"X_{dim_reduction}"
    elif dim_reduction in adata.obsm.keys():
        key_dim_reduction = dim_reduction
        print(f"Using adata.obsm['{dim_reduction}'] to identify neighboring cells")
    else:
        raise ValueError(f"Only 'lsi' is implemented for now, and no adata.obsm['{dim_reduction}'] coordinates found.")

    if projection == 'umap':
        sc.pp.neighbors(adata, use_rep=key_dim_reduction, metric=metric)
        sc.tl.umap(adata)
        key_projection = 'X_umap'
    elif projection is None:
        key_projection = key_dim_reduction
    else:
        raise ValueError("Only 'umap' and None are implemented for now.")

    if method not in ('mean', 'sum'):
        raise ValueError("method must be 'mean' or 'sum'.")

    # Identify the k nearest neighbours of each cell
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=nn_algorithm).fit(
        adata.obsm[key_projection])
    _, indices = nbrs.kneighbors(adata.obsm[key_projection])

    n_obs = adata.n_obs
    if max_metacells is None:
        max_metacells = n_obs

    # Greedy selection of metacells that don't overlap too much.
    threshold = max_overlap_metacells * k
    kept = _FILTERING_STRATEGIES[strategy](
        indices, k, threshold, max_metacells, n_obs)

    # Aggregate expression of the cells composing each metacell with a
    # single sparse matrix product (assignment @ adata.X).
    n_metacells = len(kept)
    member_indices = indices[kept]
    if method == 'mean':
        weights = np.full(n_metacells * k, 1.0 / k)
    else:  # 'sum'
        weights = np.ones(n_metacells * k, dtype=adata.X.dtype)
    assignment = sp.sparse.csr_matrix(
        (weights,
         (np.repeat(np.arange(n_metacells), k), member_indices.ravel())),
        shape=(n_metacells, n_obs))
    metacells_values = assignment @ adata.X

    # Create a new AnnData object from it
    metacells_AnnData = ad.AnnData(metacells_values)
    metacells_AnnData.var_names = adata.var_names
    metacells_AnnData.obs_names = [f"metacell_{i}" for i in range(n_metacells)]
    metacells_AnnData.var = adata.var.copy()
    metacells_AnnData.varp = adata.varp.copy()

    return metacells_AnnData


# LSI from scGLUE : https://github.com/gao-lab/GLUE/blob/master/scglue
def tfidf_old(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    # from scGLUE : https://github.com/gao-lab/GLUE/blob/master/scglue

    Original implementation, kept for use by :func:`compute_metacells_old`.

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if sp.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach).

    Optimized re-implementation of :func:`tfidf_old`, used by
    :func:`compute_metacells`. For a sparse input it computes the scaled
    nonzero values directly - applying the term-frequency (per-cell) and
    inverse-document-frequency (per-peak) factors to a single ``data``
    array - instead of allocating several intermediate sparse matrices.

    The per-cell / per-peak factors are computed exactly as in
    :func:`tfidf_old` (same dtype, same operations), so the result is
    bitwise-identical to :func:`tfidf_old`.

    Parameters
    ----------
    X
        Input matrix (sparse or dense)

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix (a new matrix; ``X`` is not modified)
    """
    if not sp.sparse.issparse(X):
        idf = X.shape[0] / X.sum(axis=0)
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

    X = X.tocsr()
    idf = np.asarray(X.shape[0] / X.sum(axis=0)).ravel()      # per peak
    inv_rowsum = np.asarray(1 / X.sum(axis=1)).ravel()        # per cell
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    new_data = X.data * inv_rowsum[rows]
    new_data *= idf[X.indices]
    return sp.sparse.csr_matrix(
        (new_data, X.indices.copy(), X.indptr.copy()), shape=X.shape)


def lsi(
        adata: ad.AnnData, n_components: int = 20,
        use_highly_variable: bool = None, tfidf_fn=tfidf, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    # from scGLUE : https://github.com/gao-lab/GLUE/blob/master/scglue

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    tfidf_fn
        TF-IDF function to apply. Defaults to the optimized :func:`tfidf`;
        :func:`compute_metacells_old` passes :func:`tfidf_old`.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf_fn(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
