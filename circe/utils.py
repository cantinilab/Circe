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
    columns=['Peak1', 'Peak2', 'score']
):
    """
    Extract links from adata.varp[key] and return them as a DataFrame.
    Since atac-networks scores are undirected, only one link is returned for
    each pair of regions.

    Parameters
    ----------
    adata : anndata object
        anndata object with var_names as variable names.
    key : str, optional
        key from adata.varp. The default is None.
        If None, and only one key is found in adata.varp, will use this key.
        Otherwise if several keys are found in adata.varp, will raise an error.
    columns : list, optional
        Columns names of the output DataFrame.
        The default is ['Peak1', 'Peak2', 'score'].

    Returns
    -------
    DataFrame
        DataFrame with columns names given by 'columns' parameter.
    """

    if key is None:  # if only one key (I guess often), no need to precise key
        # maybe replace by a default one later
        if len(list(adata.varp)) == 1:
            key = list(adata.varp)[0]
        else:
            raise KeyError(
                "Several keys were found in adata.varp: {}, ".format(
                    list(adata.varp)) +
                "please precise which keyword use (arg 'key'))"
            )
    else:
        if key not in list(adata.varp):
            raise KeyError("The key you provided ({}) is not in adata.varp: {}"
                           .format(key, list(adata.varp))
                           )

    # Convert to COO format if needed
    converted = False
    if isinstance(adata.varp[key], sp.sparse.csr_matrix):
        adata.varp[key] = adata.varp[key].tocoo()
        converted = True

    links = pd.DataFrame(
        [(row, col, data) for (row, col, data) in zip(
            [i for i in adata.varp[key].row],
            [i for i in adata.varp[key].col],
            adata.varp[key].data)
            if row < col],
        columns=columns
        ).sort_values(by=columns[2], ascending=False)

    links[columns[0]] = [adata.var_names[i] for i in links[columns[0]]]
    links[columns[1]] = [adata.var_names[i] for i in links[columns[1]]]
    links = links.reset_index(drop=True)

    # Convert back to CSR format if it was converted
    if converted:
        adata.varp[key] = adata.varp[key].tocsr()

    return links
