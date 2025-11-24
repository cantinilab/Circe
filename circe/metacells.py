import sklearn
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np
import anndata as ad
import scanpy as sc
from rich.progress import track
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

def _select_metacells(indices, k, max_overlap_metacells, max_metacells):
    """
    Select non-overlapping metacells using numpy arrays.
    This reduces memory overhead and speeds up overlap computation.
    
    Parameters
    ----------
    indices : ndarray
        Array of neighbor indices for each cell (n_cells x k)
    k : int
        Number of neighbors
    max_overlap_metacells : float
        Maximum overlap threshold
    max_metacells : int
        Maximum number of metacells to select
        
    Returns
    -------
    list
        List of selected metacell indices (as numpy arrays)
    """
    n_cells = len(indices)
    max_overlap_count = int(max_overlap_metacells * k)
    
    # Convert to list of sorted numpy arrays for faster operations
    indices_arrays = [np.sort(idx) for idx in indices]
    
    # Track selected metacells
    selected_metacells = [indices_arrays[0]]
    
    # Use a more efficient overlap check
    for i in track(range(1, n_cells), description="Computing metacells..."):
        if len(selected_metacells) >= max_metacells:
            break
        
        current = indices_arrays[i]
        is_non_overlapping = True
        
        # Check overlap with all selected metacells
        for selected in selected_metacells:
            # Fast intersection count using sorted arrays
            overlap = len(np.intersect1d(current, selected, assume_unique=True))
            if overlap >= max_overlap_count:
                is_non_overlapping = False
                break
        
        if is_non_overlapping:
            selected_metacells.append(current)
    
    return selected_metacells


def _aggregate_single_metacell(X, indices, method, is_sparse):
    """
    Aggregate a single metacell from cell expression matrix.
    
    Parameters
    ----------
    X : sparse or dense matrix
        Expression matrix
    indices : array
        Cell indices for this metacell
    method : str
        'mean' or 'sum'
    is_sparse : bool
        Whether X is sparse
        
    Returns
    -------
    ndarray
        Aggregated expression vector
    """
    if is_sparse:
        # Use sparse matrix slicing - much more memory efficient
        subset = X[indices, :]
        if method == 'mean':
            result = np.array(subset.mean(axis=0)).ravel()
        else:  # sum
            result = np.array(subset.sum(axis=0)).ravel()
    else:
        # Dense matrix
        subset = X[indices, :]
        if method == 'mean':
            result = np.mean(subset, axis=0)
        else:  # sum
            result = np.sum(subset, axis=0)
    
    return result


def _aggregate_metacells(X, metacell_indices, method, n_jobs=1):
    """
    Aggregate metacells with optional parallelization.
    
    This function avoids creating intermediate dense arrays and uses
    sparse matrix operations when possible.
    
    Parameters
    ----------
    X : sparse or dense matrix
        Expression matrix (cells x features)
    metacell_indices : list
        List of arrays, each containing cell indices for a metacell
    method : str
        'mean' or 'sum'
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    list
        List of aggregated expression vectors
    """
    is_sparse = sp.sparse.issparse(X)
    
    if n_jobs == 1:
        # Sequential processing
        metacells_values = []
        for indices in track(metacell_indices, description="Aggregating metacells..."):
            result = _aggregate_single_metacell(X, indices, method, is_sparse)
            metacells_values.append(result)
    else:
        # Parallel processing
        if n_jobs == -1:
            n_jobs = None  # Use all available cores
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(_aggregate_single_metacell, X, indices, method, is_sparse)
                for indices in metacell_indices
            ]
            metacells_values = [f.result() for f in track(futures, description="Aggregating metacells...")]
    
    return metacells_values


def compute_metacells(
        adata,
        k=50,
        max_overlap_metacells=0.9,
        max_metacells=None,
        dim_reduction='lsi',
        projection=None,
        method='mean',
        metric='cosine',
        n_jobs=1
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
        The default is 'umap'.
    method : str, optional
        Method to use to compute metacells (mean or sum).
    metric : str, optional
        Distance type used to calculate distance between neighbors.
        The default is 'cosine'.
    n_jobs : int, optional
        Number of parallel jobs for aggregation step. Use -1 for all cores.
        The default is 1 (no parallelization).

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
        lsi(adata)
        key_dim_reduction = f"X_{dim_reduction}"
        sc.pp.neighbors(adata, use_rep=key_dim_reduction, metric=metric)
    elif dim_reduction in adata.obsm.keys():
        key_dim_reduction = dim_reduction
        print("Using adata.obsm['{}'] to identify neighboring cells".format(dim_reduction))
        sc.pp.neighbors(adata, use_rep=dim_reduction, metric=metric)
    else:
        raise "Only 'lsi' is implemented for now, and no adata.obsm['{}'] coordinates found.".format(dim_reduction)

    if projection == 'umap':
        sc.tl.umap(adata)
        key_projection = 'X_umap'
    elif projection is None:
        key_projection = key_dim_reduction
    else:
        raise "Only 'umap' and None are implemented for now."

    # Identify non-overlapping above a threshold metacells
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(adata.obsm[key_projection])
    distances, indices = nbrs.kneighbors(adata.obsm[key_projection])
    
    if max_metacells is None:
        max_metacells = len(indices)

    # Select non-overlapping metacells
    metacell_indices = _select_metacells(
        indices, k, max_overlap_metacells, max_metacells
    )

    # Aggregate metacells
    metacells_values = _aggregate_metacells(
        adata.X, metacell_indices, method, n_jobs
    )

    # Create a new AnnData object from it
    metacells_AnnData = ad.AnnData(np.array(metacells_values))
    metacells_AnnData.var_names = adata.var_names
    metacells_AnnData.obs_names = [f"metacell_{i}" for i in range(
        len(metacells_values))]
    metacells_AnnData.var = adata.var.copy()
    metacells_AnnData.varp = adata.varp.copy()

    return metacells_AnnData


# LSI from scGLUE : https://github.com/gao-lab/GLUE/blob/master/scglue
def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    # from scGLUE : https://github.com/gao-lab/GLUE/blob/master/scglue

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
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
