import sklearn
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np
import anndata as ad
import scanpy as sc
from tqdm import tqdm

def compute_metacells(
        AnnData,
        k=50,
        max_overlap_metacells=0.9,
        max_metacells=5000,
        dim_reduction='lsi',
        projection='umap',
        method='mean'
):
    """
    Compute metacells by suming/averaging expression of neighbouring cells.
    Neighbouring cells are identified from umap coordinates,
    itself obtained from LSI coordinates to reduce dimensionality to topics.
    (It has been implemented to be close to Cicero/Monocle3 code 
    from Trapnell lab).

    Parameters
    ----------
    AnnData : AnnData
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
        The default is 5000.
    dim_reduction : str, optional
        Dimensionality reduction method to use to compute metacells.
        The default is 'lsi'.
    projection : str, optional
        Projection method to use to compute metacells.
        The default is 'umap'.
    method : str, optional
        Method to use to compute metacells (mean or sum).

    Returns
    -------
    metacells_AnnData : AnnData
        AnnData object containing metacells.
    """

    lsi(AnnData)
    if dim_reduction == 'lsi':
        sc.pp.neighbors(AnnData, use_rep="X_lsi", metric="cosine")
    else:
        raise "Only LSI is implemented for now."

    if projection == 'umap':
        sc.tl.umap(AnnData)
    else:
        raise "Only UMAP is implemented for now."

    # Identify non-overlapping above a threshold metacells
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(AnnData.obsm['X_umap'])
    distances, indices = nbrs.kneighbors(AnnData.obsm['X_umap'])
    indices = [set(indice) for indice in indices]

    # Select metacells that doesn't overlap too much (percentage of same cells of origin  < max_overlap_metacells for each pair)
    metacells = [indices[0]]
    iterations = 0
    for i in tqdm(indices[1:]):
        if iterations >= max_metacells-1:
            break

        no_overlap = True
        for metacell in metacells:
            if len(metacell.intersection(i)) >= max_overlap_metacells * k:
                no_overlap = False
                break
        if no_overlap:
            metacells.append(i)
        iterations += 1

    # Sum expression of neighbors composing the metacell
    metacells_values = []
    for metacell in metacells:
        if method == 'mean':
            if sp.sparse.issparse(AnnData.X):
                metacells_values.append(
                    np.array(np.mean([AnnData.X[i].toarray() for i in metacell], 0))[0]
                )
            else:
                metacells_values.append(
                    np.mean([AnnData.X[i] for i in metacell], 0)
                )
        elif method == 'sum':
            if sp.sparse.issparse(AnnData.X):
                metacells_values.append(
                    np.array(sum([AnnData.X[i].toarray() for i in metacell]))[0]
                )
            else:
                metacells_values.append(
                    sum([AnnData.X[i] for i in metacell])
                )

    # Create a new AnnData object from it
    metacells_AnnData = ad.AnnData(np.array(metacells_values))
    metacells_AnnData.var_names = AnnData.var_names
    metacells_AnnData.obs_names = [f"metacell_{i}" for i in range(len(metacells_values))]
    metacells_AnnData.var = AnnData.var.copy()
    metacells_AnnData.varp = AnnData.varp.copy()

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
