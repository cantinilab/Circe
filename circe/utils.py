import numpy as np
import pandas as pd
import anndata as ad
import scipy as sp

def cov_to_corr(cov_matrix, tol=1e-20):
    """
    Optimized version: Convert covariance matrix to correlation matrix,
    with a tolerance for small diagonal elements.

    Parameters
    ----------
    cov_matrix : np.array
        Covariance matrix.
    tol : float, optional
        Tolerance for diagonal elements. Default is 1e-20.

    Returns
    -------
    correlation_matrix : np.array
        Correlation matrix.
    """
    # Diagonal elements (standard deviations)
    d = np.sqrt(cov_matrix.diagonal())

    # Adjust small values in d to avoid instability
    d[d < tol] = 1

    # Calculate correlation matrix using broadcasting for efficiency
    correlation_matrix = cov_matrix / d[:, None] / d[None, :]

    # Set diagonal to 1
    np.fill_diagonal(correlation_matrix, 1)

    return correlation_matrix


def subset_region(adata: ad.AnnData, chromosome, start, end):
    """
    Subset anndata object on a specific region.

    Parameters
    ----------
    adata : anndata object
        anndata object with var_names as region names.
    chromosome : str
        Chromosome name.
    start : int
        Start position of the region.
    end : int
        End position of the region.

    Returns
    -------
    anndata : anndata object
        anndata object subsetted on the region defined by chr, start and end.
    """

    if len([True for i in adata.var.columns
            if i in ["chromosome", "start", "end"]]) < 3:
        raise KeyError(
            """
            'chr', 'start' and 'end' columns are not present in var.
            Please use 'add_region_infos' function to add these informations
            to your adata object.
            """
        )

    # subset per chromosome
    adata = adata[:, adata.var['chromosome'] == chromosome]
    # subset on region window
    adata = adata[:, (
        (start <= adata.var['start'])
        & (adata.var['start'] <= end)) + (
        (start <= adata.var['end'])
        & (adata.var['end'] <= end))]

    return adata


def add_region_infos(adata: ad.AnnData, sep=("_", "_"), inplace=False):
    """
    Get region informations from the var_names of adata object.
    e.g. chr1_12345_12346 -> 'chromosome' : chr1,
                             'start' : 12345,
                             'end' : 12346
    These info will be added to var of anndata object.
        adata.var['chromosome'] : chromosome
        adata.var['start'] : start position
        adata.var['end'] : end position

    Parameters
    ----------
    adata : anndata object
        anndata object with var_names as region names.
    sep : tuple, optional
        Separator of region names. The default is ('_', '_').

    Returns
    -------
    adata : anndata object
        anndata object with region informations in var.
    """
    # Check if user wants to modify anndata inplace or return a copy
    regions_list = adata.var_names

    # Replace sep[1] with sep[0] to make it easier to split
    regions_list = regions_list.str.replace(sep[1], sep[0])

    # Split region names
    regions_list = regions_list.str.split(sep[0]).tolist()

    # Check if all regions have the same number of elements
    if set([len(i) for i in regions_list]) != set([3]):
        raise ValueError(
            """
            Not all regions have the same number of elements.
            Check if sep is correct, it should be ({}, {}),
            with only one occurence each in region names.
            """.format(
                sep[0], sep[1]
            )
        )

    # Extract region informations from var_names
    region_infos = pd.DataFrame(
        regions_list, index=adata.var_names,
        columns=["chromosome", "start", "end"]
    )

    # Convert start and end to int
    region_infos["start"] = region_infos["start"].astype(int)
    region_infos["end"] = region_infos["end"].astype(int)

    # Add region informations to var
    adata.var["chromosome"] = region_infos["chromosome"]
    adata.var["start"] = region_infos["start"]
    adata.var["end"] = region_infos["end"]

    adata = sort_regions(adata)
    # Return anndata if inplace is False
    if inplace:
        pass
    else:
        return adata


def sort_regions(adata: ad.AnnData):
    """
    Sort regions by chromosome and start position.
    """
    ord_index = adata.var.sort_values(["chromosome", "start"]).index
    return adata[:, ord_index]


def extract_atac_links(
    adata: ad.AnnData,
    key=None,
    columns=('Peak1', 'Peak2', 'score')
):
    """
    Extract upper-triangular links (row < col) from adata.varp[key] into a DataFrame.
    Works natively with CSR/CSC. If another format is found, it is converted to CSR
    on the fly (without mutating adata).

    Parameters
    ----------
    adata : AnnData
        Object with var_names as variable names.
    key : str, optional
        Key from adata.varp. If None and only one key exists, that one is used.
    columns : tuple[str, str, str]
        Output column names (Peak1, Peak2, score).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns given by `columns`, sorted by descending score.
    """
    # ----- choose key -----
    if key is None:
        keys = list(adata.varp)
        if len(keys) == 1:
            key = keys[0]
        else:
            raise KeyError(
                "Several keys were found in adata.varp: {}. "
                "Please specify which key to use (arg 'key').".format(keys)
            )
    elif key not in adata.varp:
        raise KeyError(f"The key you provided ({key}) is not in adata.varp: {list(adata.varp)}")

    mat = adata.varp[key]

    # ----- ensure sparse, prefer CSR/CSC -----
    if not sp.sparse.issparse(mat):
        # fall back to CSR if a dense array sneaks in
        mat = sp.sparse.csr_matrix(mat)

    fmt = mat.getformat()
    if fmt == "csr":
        # Filter to strict upper triangle without converting to COO
        # (row < col) by using triu in CSR, then extract indices fast.
        u = sp.sparse.triu(mat, k=1, format="csr")
        indptr, indices, data = u.indptr, u.indices, u.data
        # build row indices aligned with data in O(nnz)
        rows = np.repeat(np.arange(u.shape[0], dtype=np.int64), np.diff(indptr))
        cols = indices

    elif fmt == "csc":
        # Do the same in CSC
        u = sp.sparse.triu(mat, k=1, format="csc")
        indptr, indices, data = u.indptr, u.indices, u.data
        # in CSC, indices are row indices; build column indices
        cols = np.repeat(np.arange(u.shape[1], dtype=np.int64), np.diff(indptr))
        rows = indices

    else:
        # Other formats: convert once to CSR (local copy), then proceed
        u = sp.sparse.triu(mat.tocsr(), k=1, format="csr")
        indptr, indices, data = u.indptr, u.indices, u.data
        rows = np.repeat(np.arange(u.shape[0], dtype=np.int64), np.diff(indptr))
        cols = indices

    # ----- build DataFrame -----
    df = pd.DataFrame({
        columns[0]: rows,
        columns[1]: cols,
        columns[2]: data
    })

    # map indices -> peak names (vectorized)
    var_names = np.asarray(adata.var_names)
    df[columns[0]] = var_names[df[columns[0]].to_numpy()]
    df[columns[1]] = var_names[df[columns[1]].to_numpy()]

    # sort by score desc and reset index
    df = df.sort_values(
        by=columns[2],
        ascending=False,
        kind="mergesort").reset_index(drop=True)
    return df
