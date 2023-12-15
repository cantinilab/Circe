import numpy as np
import pandas as pd
from anndata import AnnData
import tqdm
import scipy as sp
import sklearn
from sklearn.preprocessing import normalize
from . import quic_graph_lasso
from functools import reduce


def add_region_infos(AnnData, sep=("_", "_"), inplace=True):
    """
    Get region informations from the var_names of AnnData object.
    e.g. chr1_12345_12346 -> 'chromosome' : chr1,
                             'start' : 12345,
                             'end' : 12346
    These info will be added to var of AnnData object.
        adata.var['chromosome'] : chromosome
        adata.var['start'] : start position
        adata.var['end'] : end position

    Parameters
    ----------
    AnnData : AnnData object
        AnnData object with var_names as region names.
    sep : tuple, optional
        Separator of region names. The default is ('_', '_').

    Returns
    -------
    AnnData : AnnData object
        AnnData object with region informations in var.
    """
    # Check if user wants to modify AnnData inplace or return a copy
    if inplace:
        pass
    else:
        AnnData = AnnData.copy()
    regions_list = AnnData.var_names

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
        regions_list, index=AnnData.var_names,
        columns=["chromosome", "start", "end"]
    )

    # Convert start and end to int
    region_infos["start"] = region_infos["start"].astype(int)
    region_infos["end"] = region_infos["end"].astype(int)

    # Add region informations to var
    AnnData.var["chromosome"] = region_infos["chromosome"]
    AnnData.var["start"] = region_infos["start"]
    AnnData.var["end"] = region_infos["end"]

    sort_regions(AnnData)
    # Return AnnData if inplace is False
    if inplace:
        pass
    else:
        return AnnData


def sort_regions(AnnData):
    """
    Sort regions by chromosome and start position.
    """
    AnnData.var.sort_values(["chromosome", "start"], inplace=True)
    return AnnData


def calc_penalty(alpha, distance, unit_distance=1000):
    with np.errstate(divide="ignore"):
        penalties = alpha * (1 - (unit_distance / distance) ** 0.75)
    penalties[~np.isfinite(penalties)] = 0
    return penalties


def get_distances_regions(AnnData):
    # Store start and end positions in two arrays
    m, n = np.meshgrid(AnnData.var["start"].values, AnnData.var["end"].values)
    # Get distance between start of region m and end of region n
    distance = np.abs(m - n)
    # Replace diagonal by 1
    distance = distance - (np.diag(distance)) * np.eye(distance.shape[0])
    return distance


def local_alpha(
    X,
    distances,
    maxit=100,
    s=0.75,
    distance_constraint=250000,
    distance_parameter_convergence=1e-22,
    max_elements=200,
    unit_distance=1000,
):
    # Check if there is more elements than max_elements
    if X.shape[1] > max_elements:
        raise ValueError(
            """There is more elements than max_elements.
                         You might want to take less regions for computational
                         time or increase max_elements."""
        )
    # Check if distance_constraint is not too high
    if (distances > distance_constraint).sum() <= 1:
        return "No long edges"

    starting_max = 2
    distance_parameter = 2
    distance_parameter_max = 2
    distance_parameter_min = 0

    for i in range(maxit):
        # Get covariance matrix
        cov = np.cov(X, rowvar=False)
        # Add small value to diagonal to enforce convergence is lasso ?
        cov = cov - (- 1e-4) * np.eye(len(cov))
        # Get penalties
        penalties = calc_penalty(
            distance_parameter, distances, unit_distance=unit_distance
        )

        # Initiating graphical lasso
        graph_lasso_model = quic_graph_lasso.QuicGraphicalLasso(
            init_method="precomputed", lam=penalties
        )

        # Fit graphical lasso
        results = graph_lasso_model.fit(cov).covariance_

        # Get proportion of far away/all region pairs that have a connection
        mask_distance = distances > distance_constraint
        far_edges = (
            results[mask_distance] != 0).sum() / len(results[mask_distance])

        near_edges = (results != 0).sum() / (results.shape[0] ** 2)
        # If far_edges is too high (not sparse enough after filtering),
        #  increase distance_parameter
        if far_edges > 0.05 or near_edges > 0.8 or distance_parameter == 0:
            distance_parameter_min = distance_parameter
        # If far_edges is too low (too sparse because of filtering),
        #  decrease distance_parameter
        else:
            distance_parameter_max = distance_parameter

        new_distance_parameter = (distance_parameter_max
                                  + distance_parameter_min) / 2
        # If new_distance_parameter is equal to starting_max,
        # double starting_max
        if new_distance_parameter == starting_max:
            new_distance_parameter = 2 * starting_max
            starting_max = new_distance_parameter
            distance_parameter_max = starting_max
        # Break the loop if distance_parameter is not changing
        if (
            abs(distance_parameter - new_distance_parameter)
            < distance_parameter_convergence
        ):
            break
        else:
            distance_parameter = new_distance_parameter

        # Print warning if maxit is reached
        if i == maxit - 1:
            print("maximum number of iterations hit")

    return distance_parameter


def average_alpha(
    AnnData,
    window_size=500000,
    unit_distance=1000,
    n_samples=100,
    n_samples_maxtry=500,
    max_alpha_iteration=100,
    s=0.75,
    distance_constraint=250000,
    distance_parameter_convergence=1e-22,
    max_elements=200,
):
    start_slidings = [0, int(window_size / 2)]

    idx_list = []
    for k in start_slidings:
        slide_results = {}
        slide_results["scores"] = np.array([])
        slide_results["idx"] = np.array([])
        slide_results["idy"] = np.array([])
        for chromosome in AnnData.var["chromosome"].unique():
            # Get start positions of windows
            window_starts = [
                i
                for i in range(
                    k,
                    AnnData.var["end"][
                        AnnData.var["chromosome"] == chromosome].max(),
                    window_size,
                )
            ]
            for start in window_starts:
                end = start + window_size
                # Get global indices of regions in the window
                idx = np.where(
                    (AnnData.var["chromosome"] == chromosome)
                    & (AnnData.var["start"] >= start)
                    & (AnnData.var["start"] <= end)
                )[0]
                if 0 < len(idx) < 200:
                    idx_list.append(idx)

    if len(idx_list) < n_samples_maxtry:
        random_windows = idx_list
    else:
        idx_list_indices = np.random.choice(len(idx_list), n_samples_maxtry, replace=True)
        random_windows = [idx_list[i] for i in idx_list_indices]

    alpha_list = []
    for window in tqdm.tqdm(random_windows):
        distances = get_distances_regions(
            AnnData[:, window]
            )

        alpha = local_alpha(
            X=AnnData[:, window].X,
            distances=distances,
            maxit=max_alpha_iteration,
            unit_distance=unit_distance,
            s=s,
            distance_constraint=distance_constraint,
            distance_parameter_convergence=distance_parameter_convergence,
            max_elements=max_elements,
        )
        if isinstance(alpha, int) or isinstance(alpha, float):
            alpha_list.append(alpha)
        else:
            pass
        if len(alpha_list) > n_samples:
            break

    if len(alpha_list) >= n_samples:
        alpha_list = np.random.choice(
            alpha_list,
            size=n_samples,
            replace=False)
    else:
        print(
            """
              only {} windows will be used to calculate optimal penalty,
              wasn't able to find {} windows with a non-null number
              of regions inferior to max_elements={},
              AND having long-range edges (>distance_constraint)
        .""".format(
                len(alpha_list), n_samples, max_elements
            )
        )

    alpha = np.mean(alpha_list)
    return alpha


def sliding_graphical_lasso(
    AnnData,
    window_size: int = 500_000,
    unit_distance=1_000,
    distance_constraint=250_000,
    s=0.75,
    max_alpha_iteration=100,
    distance_parameter_convergence=1e-22,
    max_elements=200,
    n_samples=100,
    n_samples_maxtry=500,
):
    """
    Extract sliding submatrix from a sparse correlation matrix.

    WARNING: might look generalised for many overlaps but is not yet at the
    end, that's why 'start_sliding' is hard coded as list of 2 values.
    """

    print("Calculating penalty coefficient alpha...")
    alpha = average_alpha(
        AnnData,
        window_size=window_size,
        unit_distance=unit_distance,
        n_samples=n_samples,
        n_samples_maxtry=n_samples_maxtry,
        max_alpha_iteration=max_alpha_iteration,
        s=s,
        distance_constraint=distance_constraint,
        distance_parameter_convergence=distance_parameter_convergence,
        max_elements=max_elements,
    )

    start_slidings = [0, int(window_size / 2)]

    results = {}
    regions_list = AnnData.var_names
    # Get global indices of regions
    map_indices = {regions_list[i]: i for i in range(len(regions_list))}

    print("Calculating co-accessibility scores...")
    for k in start_slidings:
        slide_results = {}
        slide_results["scores"] = np.array([])
        slide_results["idx"] = np.array([])
        slide_results["idy"] = np.array([])

        for chromosome in AnnData.var["chromosome"].unique():
            if k == 0:
                print(chromosome, "1/2")
            else:
                print(chromosome, "2/2")
            # Get start positions of windows
            window_starts = [
                i
                for i in range(
                    k,
                    AnnData.var["end"][
                        AnnData.var["chromosome"] == chromosome].max(),
                    window_size,
                )
            ]

            for start in tqdm.tqdm(window_starts):
                end = start + window_size
                # Get global indices of regions in the window
                idx = np.where(
                    (AnnData.var["chromosome"] == chromosome)
                    & (AnnData.var["start"] >= start)
                    & (AnnData.var["start"] <= end)
                )[0]

                # already global ?
                # Get global indices of regions in the window
                # idx = [map_indices[i] for i in regions_list[idx]]

                # Get submatrix
                window_accessibility = AnnData.X[:, idx].copy()
                window_scores = np.cov(window_accessibility, rowvar=False)
                window_scores = window_scores + 1e-4 * np.eye(
                    len(window_scores))

                distance = get_distances_regions(AnnData[:, idx])
                # Test if distance is negative
                if np.any(distance < 0):
                    raise ValueError(
                        """
                                     Distance between regions should be
                                     positive. You might have overlapping
                                     regions.
                                     """
                    )

                window_penalties = calc_penalty(
                    alpha,
                    distance=distance,
                    unit_distance=unit_distance)

                # Initiating graphical lasso
                graph_lasso_model = quic_graph_lasso.QuicGraphicalLasso(
                    init_method="precomputed", lam=window_penalties
                )

                # Fit graphical lasso
                graph_lasso_model.fit(window_scores)

                # Names of regions in the window
                window_region_names = AnnData.var_names[idx].copy()
                # convert to sparse matrix the results
                corrected_scores = sp.sparse.coo_matrix(
                    graph_lasso_model.covariance_)

                # Convert corrected_scores column
                # and row indices to global indices
                idx = [
                    map_indices[name]
                    for name in window_region_names[corrected_scores.row]
                ]
                idy = [
                    map_indices[name]
                    for name in window_region_names[corrected_scores.col]
                ]

                # Add the "sub" resuls to the global sparse matrix
                slide_results["scores"] = np.concatenate(
                    [slide_results["scores"], corrected_scores.data]
                )
                slide_results["idx"] = np.concatenate(
                    [slide_results["idx"], idx]
                    )
                slide_results["idy"] = np.concatenate(
                    [slide_results["idy"], idy]
                    )

            # Create sparse matrix
            results["window_" + str(k)] = sp.sparse.coo_matrix(
                (slide_results["scores"],
                 (slide_results["idx"], slide_results["idy"])),
                shape=(AnnData.X.shape[1], AnnData.X.shape[1]),
            )

    print("Averaging co-accessibility scores across windows...")
    sliding_keys = ["window_" + str(k) for k in start_slidings]

    k_positive_coords = []
    k_negative_coords = []
    for k in sliding_keys:
        k_positive_coords.append(
            {
                (x, y)
                for x, y, d in zip(
                    results[k].row,
                    results[k].col,
                    results[k].data)
                if d >= 0
            }
        )
        k_negative_coords.append(
            {
                (x, y)
                for x, y, d in zip(
                    results[k].row,
                    results[k].col,
                    results[k].data)
                if d <= 0
            }
        )

    # Get shared positive and negative coordinates intersection
    positive_coords = set.intersection(*k_positive_coords)
    negative_coords = set.intersection(*k_negative_coords)

    # Shared and same sign coordinates
    coords = pd.DataFrame(
        set.union(negative_coords, positive_coords), columns=["row", "col"]
    )

    # Add common signe values
    average = [
        sp.sparse.csr_matrix(
            (
                sp.sparse.csr_matrix(
                    sp.sparse.csr_matrix(results[k])[coords["row"],
                                                     coords["col"]]
                ).data,
                (coords["row"], coords["col"]),
            ),
            shape=results[k].shape,
        )
        for k in sliding_keys[:]
    ]

    average = reduce(lambda x, y: x + y, average)

    # Divide for mean value
    average = average / len(sliding_keys)

    # Add uni-measurement values
    # Get all shared coordinates
    l_all_coords = []
    for k in sliding_keys:
        l_all_coords.append(
            {(x, y) for x, y, d in zip(
                results[k].row,
                results[k].col,
                results[k].data)}
        )
    all_coords = set.intersection(*l_all_coords)

    for k in sliding_keys:
        # get all coordinates in a set
        k_coords = {
            (x, y) for x, y, d in zip(
                results[k].row,
                results[k].col,
                results[k].data)
        }
        # Substract all shared coordinates to this set
        k_coords = pd.DataFrame(
            k_coords.difference(all_coords),
            columns=["row", "col"])

        # Add values to the results
        k_single = sp.sparse.csr_matrix(
            (
                sp.sparse.csr_matrix(  # csr_matrix to re-store as sparse
                    # since accessed by coordinates might return unsparsed
                    sp.sparse.csr_matrix(  # csr_matrix to access by coords
                        results[k])[k_coords["row"], k_coords["col"]]
                ).data,
                (k_coords["row"], k_coords["col"]),
            ),
            shape=results[k].shape,
        )

        average = average + k_single

    # FIX/ADD Add a way of handling value with non maximal number of overlap
    # if more than 1 overlap with the windows

    return sp.sparse.coo_matrix(average)


# LSI from scGLUE : https://github.com/gao-lab/GLUE/blob/master/scglue



def tfidf(X: np.Array) -> np.Array:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

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
        adata: AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

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
