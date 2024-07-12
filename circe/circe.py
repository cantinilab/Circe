import numpy as np
import pandas as pd
from rich.progress import track, Progress
import time
import scipy as sp
from . import quic_graph_lasso
from functools import reduce
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")


def cov_to_corr(cov_matrix, tol=1e-20):
    """Convert covariance matrix to correlation matrix, with a tolerance for diagonal elements."""
    # Diagonal elements (variances)
    d = np.sqrt(cov_matrix.diagonal())

    # Apply tolerance: if a variance is less than tol, use it directly instead of normalizing to 1.
    # This avoids division by very small numbers which can lead to numerical instability.
    d_tol = np.where(d < tol, cov_matrix.diagonal(), d)

    # Outer product of the adjusted standard deviations vector
    d_matrix = np.outer(d_tol, d_tol)

    # Element-wise division of the covariance matrix by the d_matrix
    correlation_matrix = cov_matrix / d_matrix

    # Ensure the diagonal elements are 1 or the original diagonal value if it's below tolerance
    np.fill_diagonal(correlation_matrix, np.where(d < tol, cov_matrix.diagonal(), 1))

    return correlation_matrix


def subset_region(adata, chr, start, end):
    """
    Subset anndata object on a specific region.

    Parameters
    ----------
    adata : anndata object
        anndata object with var_names as region names.
    chr : str
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
    anndata = adata[:, adata.var['chromosome'] == chr]
    # subset on region window
    anndata = anndata[:, ((start <= anndata.var['start'])
                          & (anndata.var['start'] <= end)) +
                         ((start <= anndata.var['end'])
                          & (anndata.var['end'] <= end))]

    return anndata


def add_region_infos(anndata, sep=("_", "_"), inplace=False):
    """
    Get region informations from the var_names of anndata object.
    e.g. chr1_12345_12346 -> 'chromosome' : chr1,
                             'start' : 12345,
                             'end' : 12346
    These info will be added to var of anndata object.
        adata.var['chromosome'] : chromosome
        adata.var['start'] : start position
        adata.var['end'] : end position

    Parameters
    ----------
    anndata : anndata object
        anndata object with var_names as region names.
    sep : tuple, optional
        Separator of region names. The default is ('_', '_').

    Returns
    -------
    anndata : anndata object
        anndata object with region informations in var.
    """
    # Check if user wants to modify anndata inplace or return a copy
    if inplace:
        pass
    else:
        anndata = anndata.copy()
    regions_list = anndata.var_names

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
        regions_list, index=anndata.var_names,
        columns=["chromosome", "start", "end"]
    )

    # Convert start and end to int
    region_infos["start"] = region_infos["start"].astype(int)
    region_infos["end"] = region_infos["end"].astype(int)

    # Add region informations to var
    anndata.var["chromosome"] = region_infos["chromosome"]
    anndata.var["start"] = region_infos["start"]
    anndata.var["end"] = region_infos["end"]

    anndata = sort_regions(anndata)
    # Return anndata if inplace is False
    if inplace:
        pass
    else:
        return anndata


def sort_regions(anndata):
    """
    Sort regions by chromosome and start position.
    """
    ord_index = anndata.var.sort_values(["chromosome", "start"]).index
    return anndata[:, ord_index]


def compute_atac_network(
    anndata,
    window_size=500000,
    unit_distance=1000,
    distance_constraint=250000,
    s=0.75,
    max_alpha_iteration=100,
    distance_parameter_convergence=1e-22,
    max_elements=200,
    n_samples=100,
    n_samples_maxtry=500,
    key="atac_network",
    seed=42
):
    """
    Compute co-accessibility scores between regions in a sparse matrix, stored
    in the varp slot of the passed anndata object.
    Scores are computed using 'sliding_graphical_lasso'.

    1. First, the function calculates the optimal penalty coefficient alpha.
        Alpha is calculated by averaging alpha values from 'n_samples' windows,
        such as there's less than 5% of possible long-range edges
        (> distance_constraint) and less than 20% co-accessible regions
        (regardless of distance constraint) in each window.

    2. Then, it will calculate co-accessibility scores between regions in a
    sliding window of size 'window_size' and step 'window_size/2'.
        Results should be very similar to Cicero's results. There is a strong
        correlation between Cicero's co-accessibility scores and the ones
        calculated by this function. However, absolute values are not the same,
        because Cicero uses a different method to apply Graphical Lasso.

    3. Finally, it will average co-accessibility scores across windows.

    Parameters
    ----------
    anndata : anndata object
        anndata object with var_names as region names.
    window_size : int, optional
        Size of sliding window, in which co-accessible regions can be found.
        The default is 500000.
    unit_distance : int, optional
        Distance between two regions in the matrix, in base pairs.
        The default is 1000.
    distance_constraint : int, optional
        Distance threshold for defining long-range edges.
        It is used to fit the penalty coefficient alpha.
        The default is 250000.
    s : float, optional
        Parameter for penalizing long-range edges. The default is 0.75 and
        should probably not be changed, unless you know what you are doing.
    max_alpha_iteration : int, optional
        Maximum number of iterations to calculate optimal penalty coefficient.
        The default is 100.
    distance_parameter_convergence : float, optional
        Convergence parameter for alpha (penalty) coefficiant calculation.
        The default is 1e-22.
    max_elements : int, optional
        Maximum number of regions in a window. The default is 200.
    n_samples : int, optional
        Number of windows used to calculate optimal penalty coefficient alpha.
        The default is 100.
    n_samples_maxtry : int, optional
        Maximum number of windows to try to calculate optimal penalty
        coefficient alpha. Should be higher than n_samples. The default is 500.
    key : str, optional
        Key to store the results in adata.varp. The default is "atac_network".

    Returns
    -------
    None.
    """

    anndata.varp[key] = sliding_graphical_lasso(
        anndata=anndata,
        window_size=window_size,
        unit_distance=unit_distance,
        distance_constraint=distance_constraint,
        s=s,
        max_alpha_iteration=max_alpha_iteration,
        distance_parameter_convergence=distance_parameter_convergence,
        max_elements=max_elements,
        n_samples=n_samples,
        n_samples_maxtry=n_samples_maxtry,
        seed=seed
    )


def extract_atac_links(
    anndata,
    key=None,
    columns=['Peak1', 'Peak2', 'score']
):
    """
    Extract links from adata.varp[key] and return them as a DataFrame.
    Since atac-networks scores are undirected, only one link is returned for
    each pair of regions.

    Parameters
    ----------
    anndata : anndata object
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
        if len(list(anndata.varp)) == 1:
            key = list(anndata.varp)[0]
        else:
            raise "Several keys were found in adata.varp: {}, ".format(
                list(anndata.varp))\
                + "please precise which keyword use (arg 'key'))"
    else:
        if key not in list(anndata.varp):
            raise KeyError("The key you provided ({}) is not in adata.varp: {}"
                           .format(key, list(anndata.varp))
                           )

    # Convert to COO format if needed
    converted = False
    if isinstance(anndata.varp[key], sp.sparse.csr_matrix):
        anndata.varp[key] = anndata.varp[key].tocoo()
        converted = True

    links = pd.DataFrame(
        [(row, col, data) for (row, col, data) in zip(
            [i for i in anndata.varp[key].row],
            [i for i in anndata.varp[key].col],
            anndata.varp[key].data)
            if row < col],
        columns=columns
        ).sort_values(by=columns[2], ascending=False)

    links[columns[0]] = [anndata.var_names[i] for i in links[columns[0]]]
    links[columns[1]] = [anndata.var_names[i] for i in links[columns[1]]]

    # Convert back to CSR format if it was converted
    if converted:
        anndata.varp[key] = anndata.varp[key].tocsr()

    return links


def calc_penalty(alpha, distance, unit_distance=1000):
    """
    Calculate distance penalties for graphical lasso, based on the formula
    from Cicero's paper: alpha * (1 - (unit_distance / distance) ** 0.75).

    Non-finite and negative values are replaced by 0.

    Parameters
    ----------
    alpha : float
        Penalty coefficient.
    distance : array
        Distance between regions.
    unit_distance : int, optional
        Unit distance (in base pair) to divide distance by.
        The default is 1000 for 1kb (as in Cicero's paper).

    Returns
    -------
    penalties : np.array
        Penalty coefficients for graphical lasso.

    """
    with np.errstate(divide="ignore"):
        penalties = alpha * (1 - (unit_distance / distance) ** 0.75)
    penalties[~np.isfinite(penalties)] = 0
    penalties[penalties < 0] = 0
    return penalties


def get_distances_regions(anndata):
    """
    Get distances between regions, var_names from an anndata object.
    'add_region_infos' should be run before this function.

    Parameters
    ----------
    anndata : anndata object
        anndata object with var_names as region names.

    Returns
    -------
    distance : np.array
        Distance between regions.
    """

    # Store start and end positions in two arrays
    m, n = np.meshgrid((anndata.var["end"].values + anndata.var["start"].values)/2,
                       (anndata.var["end"].values + anndata.var["start"].values)/2)
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
    init_method="precomputed"
):
    """
    todo
    """
    # Check if there is more elements than max_elements
    if X.shape[1] > max_elements:
        raise ValueError(
            """There is more elements than max_elements.
                         You might want to take less regions for computational
                         time or increase max_elements."""
        )
    if sp.sparse.issparse(X):
        X = X.toarray()

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
            init_method=init_method,
            lam=penalties,
            tol=1e-4,
            max_iter=1e4,
            auto_scale=False,
            )

        # Fit graphical lasso
        results = graph_lasso_model.fit(cov).precision_

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
            # print("maximum number of iterations hit")
            pass
    return distance_parameter


def average_alpha(
    anndata,
    window_size=500000,
    unit_distance=1000,
    n_samples=100,
    n_samples_maxtry=500,
    max_alpha_iteration=100,
    s=0.75,
    distance_constraint=250000,
    distance_parameter_convergence=1e-22,
    max_elements=200,
    chromosomes_sizes=None,
    init_method="precomputed",
    seed=42
):
    """
    Calculate average alpha coefficient that determines the sparsity penalty
    term in the graphical lasso penalty used for the sliding graphical lasso.
    (The global penalty also uses distance between regions). 
    The alpha coefficient is calculated by averaging alpha values from
    'n_samples' windows, such as there's less than 5% of possible long-range
    edges (> distance_constraint) and less than 20% co-accessible regions in
    each window.

    Parameters
    ----------
    anndata : anndata object
        anndata object with var_names as region names.
    window_size : int, optional
        Size of the sliding window, where co-accessible regions can be found.
        The default is 500000.
    unit_distance : int, optional
        Unit distance (in base pair) to divide distance by.
        The default is 1000 and should be change carefully (in regards to
        the distance between regions).
    n_samples : int, optional
        Number of windows used to calculate average optimal penalty coefficient
        alpha. The default is 100.
    n_samples_maxtry : int, optional
        Maximum number of windows to try to calculate optimal penalty
        coefficient alpha. Should be higher than n_samples. The default is 500.
    max_alpha_iteration : int, optional
        Maximum number of iterations to converge optimal penalty coefficient.
        The default is 100.
    s : float, optional
        Parameter for penalizing long-range edges. The default is 0.75 and
        should probably not be changed, unless you know what you are doing.
    distance_constraint : int, optional
        Distance threshold for defining long-range edges. It is used to fit the
        penalty coefficient alpha. The default is 250000, and usually should
        be window_size/2.
    distance_parameter_convergence : float, optional
        Convergence parameter for alpha (penalty) coefficiant calculation.
        The default is 1e-22.
    max_elements : int, optional
        Maximum number of regions in a window. The default is 200.
    chromosomes_sizes : dict, optional
        Dictionary with chromosome names as keys and sizes as values.
        The default is None, and will use the maximum end position of each
        chromosome in anndata.var.
    init_method : str, optional
        Method to use to compute initial covariance matrix.
        The default is "precomputed".
        SHOULD BE CHANGED CAREFULLY.
    seed : int, optional
        Seed for random number generator. The default is 42.

    Returns
    -------
    alpha : float
        Average alpha coefficient.
    """
    start_slidings = [0, int(window_size / 2)]

    idx_list = []
    for k in start_slidings:
        slide_results = {}
        slide_results["scores"] = np.array([])
        slide_results["idx"] = np.array([])
        slide_results["idy"] = np.array([])
        for chromosome in anndata.var["chromosome"].unique():
            if chromosomes_sizes is None:
                chromosome_size = anndata.var["end"][
                    anndata.var["chromosome"] == chromosome].max()
            else:
                try:
                    chromosome_size = chromosomes_sizes[chromosome]
                except Warning:
                    print(
                        "{} not found as key in chromosome_size, ".format(
                            chromosome) +
                        " using max end position.")
                    chromosome_size = anndata.var["end"][
                        anndata.var["chromosome"] == chromosome].max()
            # Get start positions of windows
            window_starts = [
                i
                for i in range(
                    k,
                    chromosome_size,
                    window_size,
                )
            ]
            for start in window_starts:
                end = start + window_size
                # Get global indices of regions in the window
                idx = np.where(
                    (anndata.var["chromosome"] == chromosome)
                    & (
                        ((anndata.var["start"] > start)
                         & (anndata.var["start"] < end-1))
                        |
                        ((anndata.var["end"] > start)
                         & (anndata.var["end"] < end-1))
                      )
                    )[0]

                if 0 < len(idx) < 200:
                    idx_list.append(idx)

    if len(idx_list) < n_samples_maxtry:
        random_windows = idx_list
    else:
        rng = np.random.default_rng(seed=seed)
        idx_list_indices = rng.choice(
            len(idx_list),
            n_samples_maxtry,
            replace=False,
        )
        random_windows = [idx_list[i] for i in idx_list_indices]

    # While loop to calculate alpha until n_samples measures are obtained
    with Progress() as progress:
        bar_alpha = progress.add_task(
            "Calculating alpha", total=n_samples, auto_refresh=False)
        alpha_list = []

        while not progress.finished:
            for window in random_windows:
                # Calculate distances between regions
                distances = get_distances_regions(
                    anndata[:, window]
                    )

                # Calculate individual alpha
                alpha = local_alpha(
                    X=anndata[:, window].X,
                    distances=distances,
                    maxit=max_alpha_iteration,
                    unit_distance=unit_distance,
                    s=s,
                    distance_constraint=distance_constraint,
                    distance_parameter_convergence=distance_parameter_convergence,
                    max_elements=max_elements,
                    init_method=init_method
                )

                # Append alpha to alpha_list if it's a number and not None
                if isinstance(alpha, int) or isinstance(alpha, float):
                    alpha_list.append(alpha)
                    progress.update(bar_alpha, advance=1)
                    progress.refresh()
                else:
                    pass

                # Break the loop if n_samples is reached
                if len(alpha_list) >= n_samples:
                    time.sleep(0.001)
                    break
            # Break the while loop too if n_samples is reached
            break

    # Print warning if n_samples is not reached
    if len(alpha_list) < n_samples:
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

    # Calculate average alpha
    alpha = np.mean(alpha_list)
    return alpha


def sliding_graphical_lasso(
    anndata,
    window_size: int = 500_000,
    unit_distance=1_000,
    distance_constraint=250_000,
    s=0.75,
    max_alpha_iteration=100,
    distance_parameter_convergence=1e-22,
    max_elements=200,
    n_samples=100,
    n_samples_maxtry=500,
    init_method="precomputed",
    verbose=False,
    seed=42
):
    """
    Estimate co-accessibility scores between regions penalized on distance.
    The function uses graphical lasso to estimate the precision matrix of
    the co-accessibility scores. The function uses a sliding window approach.

    The function calculates an optimal penalty coefficient alpha
    for each window, based on the distance between regions in the window.
    The function then calculates co-accessibility scores between regions in
    each window using graphical lasso. The results are averaged across windows.

    WARNING: might look generalised for many overlaps but is not yet,
    that's why 'start_sliding' is hard coded as list of 2 values.

    Parameters
    ----------
    anndata : anndata object
        anndata object with var_names as region names.
    window_size : int, optional
        Size of the sliding window, where co-accessible regions can be found.
        The default is 500000.
    unit_distance : int, optional
        Distance between two regions in the matrix, in base pairs.
        The default is 1000.
    distance_constraint : int, optional
        Distance threshold for defining long-range edges.
        It is used to fit the penalty coefficient alpha.
        The default is 250000.
    s : float, optional
        Parameter for penalizing long-range edges. The default is 0.75 and
        should probably not be changed, unless you know what you are doing.
    max_alpha_iteration : int, optional
        Maximum number of iterations to calculate optimal penalty coefficient.
        The default is 100.
    distance_parameter_convergence : float, optional
        Convergence parameter for alpha (penalty) coefficiant calculation.
        The default is 1e-22.
    max_elements : int, optional
        Maximum number of regions in a window. The default is 200.
    n_samples : int, optional
        Number of windows used to calculate optimal penalty coefficient alpha.
        The default is 100.
    n_samples_maxtry : int, optional
        Maximum number of windows to try to calculate optimal penalty
        coefficient alpha. Should be higher than n_samples. The default is 500.
    init_method : str, optional
        Method to use to compute initial covariance matrix.
        The default is "precomputed".
        SHOULD BE CHANGED CAREFULLY.
    verbose : bool, optional
        Print alpha coefficient. The default is False.
    seed : int, optional
        Seed for random number generator. The default is 42.

    Returns
    -------
    results : dict
        Dictionary with keys as window names and values as sparse matrices 
        (csr) of co-accessibility scores.
    """

    # AnnData object should have more than 1 cell
    if anndata.X.shape[0] < 2 or anndata.X.shape[1] < 2:
        raise ValueError(
            """
            Not enough cells/regions in the AnnData object.
            You need at least 2 cells and 2 regions to calculate 
            co-accessibility.
            """
        )
    # print("Calculating penalty coefficient alpha...")
    alpha = average_alpha(
        anndata,
        window_size=window_size,
        unit_distance=unit_distance,
        n_samples=n_samples,
        n_samples_maxtry=n_samples_maxtry,
        max_alpha_iteration=max_alpha_iteration,
        s=s,
        distance_constraint=distance_constraint,
        distance_parameter_convergence=distance_parameter_convergence,
        max_elements=max_elements,
        init_method=init_method,
        seed=seed
    )
    if verbose:
        print("Alpha coefficient calculated : {}".format(alpha))

    start_slidings = [0, int(window_size / 2)]

    results = {}
    idx_ = {}
    idy_ = {}
    regions_list = anndata.var_names
    # Get global indices of regions
    map_indices = {regions_list[i]: i for i in range(len(regions_list))}

    for k in start_slidings:
        slide_results = {}
        slide_results["scores"] = np.array([])
        slide_results["idx"] = np.array([])
        slide_results["idy"] = np.array([])
        idx_["window_" + str(k)] = np.array([])
        idy_["window_" + str(k)] = np.array([])
#        if k == 0:
#            print("Starting to process chromosomes : {}".format(
#                anndata.var["chromosome"].unique()))
#        else:
#            print("Finishing to process chromosomes : {}".format(
#                anndata.var["chromosome"].unique()))
        for chromosome in track(
            anndata.var["chromosome"].unique(),
            description="Calculating co-accessibility: {}/2".format(
                1 if k == 0 else 2),):
            # Get start positions of windows
            window_starts = [
                i
                for i in range(
                    k,
                    anndata.var["end"][
                        anndata.var["chromosome"] == chromosome].max(),
                    window_size,
                )
            ]

            for start in window_starts:
                end = start + window_size
                # Get global indices of regions in the window
                idx = np.where(
                    (anndata.var["chromosome"] == chromosome)
                    & (
                        ((anndata.var["start"] > start)
                         & (anndata.var["start"] < end-1))
                        |
                        ((anndata.var["end"] > start)
                         & (anndata.var["end"] < end-1))
                      )
                    )[0]

                # Add to the list of all regions used to know how many
                # times each region is used
                x_, y_ = \
                    np.meshgrid(idx, idx)
                idx_["window_" + str(k)], idy_["window_" + str(k)] = \
                    np.concatenate([idx_["window_" + str(k)], x_.flatten()]), \
                    np.concatenate([idy_["window_" + str(k)], y_.flatten()])

                # already global ?
                # Get global indices of regions in the window
                # idx = [map_indices[i] for i in regions_list[idx]]

                if idx is None or len(idx) <= 1:
                    # print("Less than two regions in window")
                    continue

                # Get submatrix
                if sp.sparse.issparse(anndata.X):
                    window_accessibility = anndata.X[:, idx].toarray()
                    window_scores = np.cov(window_accessibility, rowvar=False)
                    window_scores = window_scores + 1e-4 * np.eye(
                        len(window_scores))

                else:
                    window_accessibility = anndata.X[:, idx].copy()
                    window_scores = np.cov(window_accessibility, rowvar=False)
                    window_scores = window_scores + 1e-4 * np.eye(
                        len(window_scores))

                distance = get_distances_regions(anndata[:, idx])

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
                    init_method=init_method,
                    lam=window_penalties,
                    tol=1e-4,
                    max_iter=1e4,
                    auto_scale=False,
                )

                # Fit graphical lasso
                graph_lasso_model.fit(window_scores)

                # Names of regions in the window
                window_region_names = anndata.var_names[idx].copy()

                # Transform to correlation matrix
                scores = cov_to_corr(graph_lasso_model.covariance_)

                # convert to sparse matrix the results
                corrected_scores = sp.sparse.coo_matrix(
                    scores)

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
                shape=(anndata.X.shape[1], anndata.X.shape[1]),
            )

    results = reconcile(results, idx_, idy_)

    print("Done !")
    return results


def reconcile(
    results_gl,
    idx_gl,
    idy_gl
):

    results_keys = list(results_gl.keys())
    print("Averaging co-accessibility scores across windows...")

    #################
    ### To keep entries contained in 2 windows

    # sum of values per non-null locations
    average = reduce(lambda x, y: x+y,
                     [results_gl[k] for k in results_keys])

    # Initiate divider depending on number of overlapping windows
    divider = sp.sparse.csr_matrix(
        ([1 for i in range(len(idx_gl[results_keys[0]]))],
         (idx_gl[results_keys[0]].astype(int),
          idy_gl[results_keys[0]].astype(int)))
    )
    for k in results_keys[1:]:
        divider = divider + sp.sparse.csr_matrix(
            ([1 for i in range(len(idx_gl[k]))],
             (idx_gl[k].astype(int),
              idy_gl[k].astype(int)))
        )

    # extract all values where there is no sign agreement between windows
    signs_disaggreeing = reduce(
        lambda x, y: sp.sparse.csr_matrix.multiply((x > 0), (y < 0)),
        [results_gl[k] for k in results_keys])
    signs_disaggreeing += reduce(
        lambda x, y: sp.sparse.csr_matrix.multiply((x < 0), (y > 0)),
        [results_gl[k] for k in results_keys])

    # Remove disagreeing values from average
    average = average - sp.sparse.csr_matrix.multiply(
        average, signs_disaggreeing)
    # Remove also disagreeing values from divider
    divider = sp.sparse.csr_matrix.multiply(divider, average.astype(bool).astype(int))

    # Delete the sign_disagreeing matrix
    del signs_disaggreeing

    # Divide the sum by number of values
    average.data = average.data/divider.data
    return average
