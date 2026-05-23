import scipy as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import numpy as np
import anndata as ad
import scanpy as sc
from rich.progress import track

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

    # Reusable per-candidate overlap counter, indexed by kept-metacell id.
    # Avoids the per-iteration np.bincount allocation in the previous version
    # and allows early-exit once a metacell hits `threshold`. Reset only the
    # entries we touched (including on the break path).
    counts = np.zeros(max_metacells, dtype=np.int32)
    touched = []

    for idx in track(range(1, n_obs),
                     description="Computing metacells (inverse index)...",
                     update_period=0.2):
        if idx >= max_metacells:
            break
        conflict = False
        for x in indices[idx]:
            for metacell_id in inv_index[x]:
                if counts[metacell_id] == 0:
                    touched.append(metacell_id)
                counts[metacell_id] += 1
                if counts[metacell_id] >= threshold:
                    conflict = True
                    break
            if conflict:
                break
        for metacell_id in touched:
            counts[metacell_id] = 0
        touched.clear()
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
                     description="Computing metacells (vectorized)...",
                     update_period=0.2):
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
                     description="Computing metacells (spatial)...",
                     update_period=0.2):
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

_MODES = ('non_overlap', 'smoothing', 'community')


def _select_metacells_community(indices, k, max_metacells, n_obs):
    """
    Partition cells into ~max_metacells communities via Leiden on the k-NN
    graph and return the cell membership of each community.

    The resolution parameter is found by binary search (geometric mean on a
    log scale) so the community count lands close to ``max_metacells``. If
    the upper bound is hit without reaching the target, the bracket is
    widened adaptively up to resolution=10_000. Inspired by the SuperCell
    algorithm (Bilous et al.), reimplemented here using leidenalg.

    Note
    ----
    The actual number of communities may be smaller than ``max_metacells``
    if the graph does not support that many distinct partitions (this is a
    structural property of the data, not a bug). When the result differs
    from the target by more than 20%, a warning is emitted so the caller
    can audit. The exact count is always available via the returned list
    length and the resulting AnnData's ``n_obs``.

    Returns
    -------
    list of numpy.ndarray
        For each metacell, the indices of cells belonging to it.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mode='community' requires the 'leidenalg' and 'python-igraph' "
            "packages. Install them with `pip install leidenalg python-igraph` "
            "or `conda install -c conda-forge leidenalg python-igraph`."
        ) from exc
    import warnings

    rows = np.repeat(np.arange(n_obs), k)
    cols = indices.ravel()
    mask = rows != cols
    edges = np.stack([rows[mask], cols[mask]], axis=1).tolist()
    graph = ig.Graph(n=n_obs, edges=edges, directed=False)
    graph.simplify()

    def run(resolution):
        return leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=0,
        )

    lo, hi = 0.01, 100.0
    best_partition = None
    best_diff = None
    best_n = None
    max_iter = 12
    hi_cap = 10_000.0

    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5
        partition = run(mid)
        n_comm = len(set(partition.membership))
        diff = abs(n_comm - max_metacells)
        if best_partition is None or diff < best_diff:
            best_partition, best_diff, best_n = partition, diff, n_comm
        if n_comm == max_metacells:
            break
        if n_comm > max_metacells:
            hi = mid
        else:
            lo = mid
        # Widen the upper bound if we're saturating it without reaching the
        # target. Stop widening once we hit hi_cap.
        if lo >= hi * 0.5 and hi < hi_cap and n_comm < max_metacells:
            hi = min(hi * 10.0, hi_cap)

    drift = abs(best_n - max_metacells) / max_metacells
    if drift > 0.2:
        warnings.warn(
            f"mode='community' produced {best_n} communities but "
            f"{max_metacells} were requested ({drift:.0%} drift). The k-NN "
            f"graph may not support that many distinct partitions; try a "
            f"lower max_metacells or use a different dim_reduction.",
            RuntimeWarning,
            stacklevel=2,
        )

    membership = np.asarray(best_partition.membership)
    community_ids = np.unique(membership)
    return [np.where(membership == cid)[0] for cid in community_ids]


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
        nn_jobs=None,
        strategy='inverse_index',
        mode='non_overlap',
):
    """
    Compute metacells by suming/averaging expression of neighbouring cells.
    Neighbouring cells are identified from umap coordinates,
    itself obtained from LSI coordinates to reduce dimensionality to topics.
    (It has been implemented to be close to Cicero/Monocle3 code
    from Trapnell lab).

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
    nn_jobs : int, optional
        Number of parallel jobs to use for nearest-neighbour search, passed to
        :class:`sklearn.neighbors.NearestNeighbors`. The default is None,
        which uses scikit-learn's default. Use -1 to use all available CPUs.
    strategy : str, optional
        Non-overlap filtering strategy (only used when ``mode='non_overlap'``).
        The default is 'inverse_index'.

        - 'inverse_index' : exact, fastest. Uses a cell -> kept-metacells
          index so each candidate is only compared against metacells that
          actually share cells with it.
        - 'vectorized' : exact. Computes overlaps with a sparse matrix
          product.
        - 'spatial' : approximate, fast. Prunes comparisons using the
          nearest-neighbour structure of the metacell centers. Results are
          close to exact for a high ``max_overlap_metacells`` but NOT
          guaranteed identical.
    mode : str, optional
        Metacell construction regime. The default is 'non_overlap'.

        - 'non_overlap' : greedy k-NN metacells with overlap filtering (the
          existing behaviour). Output size depends on ``max_overlap_metacells``
          and ``max_metacells``.
        - 'smoothing' : keep every cell as a metacell and aggregate its k
          nearest neighbours. No overlap filtering. Use when the goal is to
          denoise the matrix without changing the cell count.
        - 'community' : SuperCell-style aggregation. Partitions cells into
          ``max_metacells`` Leiden communities on the k-NN graph and
          aggregates all cells of each community. Requires ``max_metacells``
          to be set. Use for true coarsening / dimensionality reduction.
          Needs the optional extra: ``pip install circe-py[community]``.
          The actual community count may be smaller than ``max_metacells``
          if the graph cannot be partitioned that finely; a warning is
          emitted when the result drifts more than 20% from the target.

    Returns
    -------
    metacells_AnnData : AnnData
        AnnData object containing metacells.
    """
    if strategy not in _FILTERING_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(_FILTERING_STRATEGIES)}.")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}.")
    if mode == 'community' and max_metacells is None:
        raise ValueError(
            "mode='community' requires max_metacells to be set explicitly.")
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
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm=nn_algorithm, n_jobs=nn_jobs).fit(
            adata.obsm[key_projection])
    _, indices = nbrs.kneighbors(adata.obsm[key_projection])

    n_obs = adata.n_obs
    if max_metacells is None:
        max_metacells = n_obs

    # Select metacells and build the (n_metacells x n_obs) assignment matrix.
    # Each row of `assignment` describes which input cells contribute to that
    # metacell and with what weight; `assignment @ adata.X` then aggregates
    # expression in one sparse matmul.
    if mode == 'community':
        members_per_metacell = _select_metacells_community(
            indices, k, max_metacells, n_obs)
        n_metacells = len(members_per_metacell)
        sizes = np.fromiter(
            (len(m) for m in members_per_metacell), dtype=np.int64,
            count=n_metacells)
        cols = np.concatenate(members_per_metacell)
        rows = np.repeat(np.arange(n_metacells), sizes)
        if method == 'mean':
            weights = np.repeat(1.0 / sizes, sizes)
        else:  # 'sum'
            weights = np.ones(cols.size, dtype=adata.X.dtype)
    else:
        if mode == 'smoothing':
            kept = list(range(min(n_obs, max_metacells)))
        else:  # 'non_overlap'
            threshold = max_overlap_metacells * k
            kept = _FILTERING_STRATEGIES[strategy](
                indices, k, threshold, max_metacells, n_obs)
        n_metacells = len(kept)
        cols = indices[kept].ravel()
        rows = np.repeat(np.arange(n_metacells), k)
        if method == 'mean':
            weights = np.full(n_metacells * k, 1.0 / k)
        else:  # 'sum'
            weights = np.ones(n_metacells * k, dtype=adata.X.dtype)

    assignment = sp.sparse.csr_matrix(
        (weights, (rows, cols)), shape=(n_metacells, n_obs))
    metacells_values = assignment @ adata.X

    # Create a new AnnData object from it
    metacells_AnnData = ad.AnnData(metacells_values)
    metacells_AnnData.var_names = adata.var_names
    metacells_AnnData.obs_names = [f"metacell_{i}" for i in range(n_metacells)]
    metacells_AnnData.var = adata.var.copy()
    metacells_AnnData.varp = adata.varp.copy()

    return metacells_AnnData


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach).

    For a sparse input it computes the scaled nonzero values directly -
    applying the term-frequency (per-cell) and inverse-document-frequency
    (per-peak) factors to a single ``data`` array - instead of allocating
    several intermediate sparse matrices.

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
        use_highly_variable: bool = None, **kwargs
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
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    # Cast to float32 before SVD: halves working memory and modestly speeds up
    # the dominant sparse matmuls inside randomized_svd. The 20 leading
    # components don't need float64 precision for LSI.
    if sp.sparse.issparse(X_norm):
        if X_norm.dtype != np.float32:
            X_norm = X_norm.astype(np.float32)
    else:
        X_norm = np.asarray(X_norm, dtype=np.float32)
    X_lsi = randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
