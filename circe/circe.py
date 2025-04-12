import logging
import numpy as np
import pandas as pd
from rich.progress import Progress
import time
import scipy as sp
from circe import quic_graph_lasso
from functools import reduce
import warnings
from typing import Union
from dask.distributed import LocalCluster, Client
from joblib import Parallel, delayed, parallel_config
import tqdm

warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")


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


def subset_region(adata, chromosome, start, end):
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


def add_region_infos(adata, sep=("_", "_"), inplace=False):
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
    if inplace:
        pass
    else:
        adata = adata.copy()
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


def sort_regions(adata):
    """
    Sort regions by chromosome and start position.
    """
    ord_index = adata.var.sort_values(["chromosome", "start"]).index
    return adata[:, ord_index]


def compute_atac_network(
    adata,
    window_size=None,
    unit_distance=1000,
    distance_constraint=None,
    s=None,
    organism=None,
    max_alpha_iteration=100,
    distance_parameter_convergence=1e-22,
    max_elements=200,
    n_samples=100,
    n_samples_maxtry=500,
    key="atac_network",
    seed=42,
    njobs=1,
    threads_per_worker=1,
    verbose=0
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
    adata : anndata object
        anndata object with var_names as region names.
    window_size : int, optional
        Size of sliding window, in which co-accessible regions can be found.
        The default is None and will be set to 500000 if organism is None.
        This parameter is organism specific.
    unit_distance : int, optional
        Distance between two regions in the matrix, in base pairs.
        The default is 1000.
    distance_constraint : int, optional
        Distance threshold for defining long-range edges.
        It is used to fit the penalty coefficient alpha.
        The default is None and will be set to 250000 if organism is None.
        This parameter is organism specific.
    s : float, optional
        Parameter for penalizing long-range edges. The default is None and
        will be set to 0.75 if organism is None. This parameter is organism
        specific.
    organism : str, optional
        Organism name. The default is None.
        If s, window_size and distance_constraint are None, will use
        organism-specific values.
        Otherwise, will use the values passed as arguments.
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
    seed : int, optional
        Seed for random number generator. The default is 42.
    njobs : int, optional
        Number of jobs to run in parallel. The default is 1.
    threads_per_worker : int, optional
        Number of threads per worker. The default is 1.
    verbose : int, optional
        Verbose level.
            0: no output at all
            1: tqdm progress bar
            2:detailed output
    Returns
    -------
    None.
    """

    adata.varp[key] = sliding_graphical_lasso(
        adata=adata,
        window_size=window_size,
        unit_distance=unit_distance,
        distance_constraint=distance_constraint,
        s=s,
        organism=organism,
        max_alpha_iteration=max_alpha_iteration,
        distance_parameter_convergence=distance_parameter_convergence,
        max_elements=max_elements,
        n_samples=n_samples,
        n_samples_maxtry=n_samples_maxtry,
        seed=seed,
        njobs=njobs,
        verbose=verbose
    )


def extract_atac_links(
    adata,
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


def calc_penalty(alpha, distance, unit_distance=1000, s=0.75):
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
    s : float, optional
        Parameter for penalizing long-range edges. The default is 0.75
        (Human/Mouse value). This parameter is organism specific.

    Returns
    -------
    penalties : np.array
        Penalty coefficients for graphical lasso.
    """
    with np.errstate(divide="ignore"):
        penalties = alpha * (1 - (unit_distance / distance) ** s)
    penalties[~np.isfinite(penalties)] = 0
    penalties[penalties < 0] = 0
    return penalties


def get_distances_regions(adata):
    """
    Get distances between regions, var_names from an anndata object.
    'add_region_infos' should be run before this function.

    Parameters
    ----------
    adata : anndata object
        anndata object with var_names as region names.

    Returns
    -------
    distance : np.array
        Distance between regions.
    """

    # Store start and end positions in two arrays
    m, n = np.meshgrid(
        (adata.var["end"].values + adata.var["start"].values)/2,
        (adata.var["end"].values + adata.var["start"].values)/2)
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
    Calculate optimal penalty coefficient alpha for a given window.
    The alpha coefficient is fitted based on the number of long-range edges
    (> distance_constraint) and short-range edges in the window.

    Parameters
    ----------
    X : np.array
        Matrix of regions in a window.
    distances : np.array
        Distance between regions in the window.
    maxit : int, optional
        Maximum number of iterations to converge alpha. The default is 100.
    s : float, optional
        Parameter for penalizing long-range edges. The default is 0.75
        (Human/Mouse value). This parameter is organism specific.
    distance_constraint : int, optional
        Distance threshold for defining long-range edges. It is used to fit the
        penalty coefficient alpha. The default is 250000 (Human/Mouse value).
        This parameter is organism specific and usually half of window_size.
    distance_parameter_convergence : float, optional
        Convergence parameter for alpha (penalty) coefficiant calculation.
        The default is 1e-22.
    max_elements : int, optional
        Maximum number of regions in a window. The default is 200.
    unit_distance : int, optional
        Unit distance (in base pair) to divide distance by.
        The default is 1000.
    init_method : str, optional
        Method to use to compute initial covariance matrix.
        The default is "precomputed".
        SHOULD BE CHANGED CAREFULLY.

    Returns
    -------
    distance_parameter : float
        Optimal penalty coefficient alpha.

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
            distance_parameter,
            distances,
            unit_distance=unit_distance,
            s=s
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
    adata,
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
    seed=42,
    verbose=False
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
    adata : anndata object
        anndata object with var_names as region names.
    window_size : int, optional
        Size of the sliding window, where co-accessible regions can be found.
        The default is 500000 (Human/Mouse value). This parameter is organism
        specific.
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
        Parameter for penalizing long-range edges. The default is 0.75
        (Human/Mouse value). This parameter is organism specific.
    distance_constraint : int, optional
        Distance threshold for defining long-range edges. It is used to fit the
        penalty coefficient alpha. The default is 250000 (Human/Mouse value).
        This parameter is organism specific and usually half of window_size.
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
    verbose : bool, optional
        Print alpha coefficient and number of windows used to
        calculate it if inferior to n_samples. The default is False.

    Returns
    -------
    alpha : float
        Average alpha coefficient.
    """
    start_slidings = [0, int(window_size / 2)]

    window_starts = []
    for k in start_slidings:
        slide_results = {}
        slide_results["scores"] = np.array([])
        slide_results["idx"] = np.array([])
        slide_results["idy"] = np.array([])
        for chromosome in adata.var["chromosome"].unique():
            if chromosomes_sizes is None:
                chromosome_size = adata.var["end"][
                    adata.var["chromosome"] == chromosome].max()
            else:
                try:
                    chromosome_size = chromosomes_sizes[chromosome]
                except Warning:
                    print(
                        "{} not found as key in chromosome_size, ".format(
                            chromosome) +
                        " using max end position.")
                    chromosome_size = adata.var["end"][
                        adata.var["chromosome"] == chromosome].max()
            # Get start positions of windows
            chr_window_starts = [
                (chromosome, i)
                for i in range(
                    k,
                    chromosome_size,
                    window_size,
                )
            ]
            window_starts += chr_window_starts

    random_windows = []
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(window_starts)
    while len(random_windows) < n_samples_maxtry:
        n_window_to_choose = n_samples_maxtry-len(random_windows)
        idx_windows = window_starts[0:n_window_to_choose]
        if len(idx_windows) == 0:
            break
        window_starts = window_starts[n_window_to_choose:]

        for chromosome, start in idx_windows:
            end = start + window_size
            # Get global indices of regions in the window
            idx = np.where(
                (adata.var["chromosome"] == chromosome)
                & (
                    ((adata.var["start"] > start)
                     & (adata.var["start"] < end-1))
                    |
                    ((adata.var["end"] > start)
                     & (adata.var["end"] < end-1))
                  )
                )[0]

            if 0 < len(idx) < 200:
                random_windows.append(idx)

    # While loop to calculate alpha until n_samples measures are obtained
    with Progress() as progress:
        bar_alpha = progress.add_task(
            "Calculating alpha", total=n_samples, auto_refresh=False)
        alpha_list = []

        while not progress.finished:
            for window in random_windows:
                # Calculate distances between regions
                distances = get_distances_regions(
                    adata[:, window]
                    )

                # Calculate individual alpha
                alpha = local_alpha(
                    X=adata[:, window].X,
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
        if verbose:
            warnings.warn(
                """
                only {} windows will be used to calculate optimal penalty,
                wasn't able to find {} windows with a non-null number
                of regions inferior to max_elements={},
                AND having long-range edges (>distance_constraint)
                .""".format(
                    len(alpha_list), n_samples, max_elements
                ), UserWarning
            )

    # Calculate average alpha
    alpha = np.mean(alpha_list)
    return alpha


def get_distances_regions_from_dataframe(df):
    """
    Get distances between regions, var_names from a dataframe object.
    'add_region_infos' should be run before this function.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with var_names as region names.

    Returns
    -------
    distance : np.array
        Distance between regions.
    """

    # Store start and end positions in two arrays
    m, n = np.meshgrid(
        (df["end"].values + df["start"].values)/2,
        (df["end"].values + df["start"].values)/2)
    # Get distance between start of region m and end of region n
    distance = np.abs(m - n)
    # Replace diagonal by 1
    distance = distance - (np.diag(distance)) * np.eye(distance.shape[0])
    return distance


def sliding_graphical_lasso(
    adata,
    window_size: Union[int, None] = None,
    unit_distance=1_000,
    distance_constraint=None,
    s=None,
    organism=None,
    max_alpha_iteration=100,
    distance_parameter_convergence=1e-22,
    max_elements=200,
    n_samples=100,
    n_samples_maxtry=500,
    init_method="precomputed",
    verbose=0,
    seed=42,
    njobs=1,
    threads_per_worker=1
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
    adata : AnnData object
        AnnData object with var_names as region names.
    window_size : int, optional
        Size of the sliding window, where co-accessible regions can be found.
        The default is None and will be set to 500000 if organism is None.
        This parameter is organism specific.
    unit_distance : int, optional
        Distance between two regions in the matrix, in base pairs.
        The default is 1000.
    distance_constraint : int, optional
        Distance threshold for defining long-range edges.
        It is used to fit the penalty coefficient alpha.
        The default is None and will be set to 250000 if organism is None.
        This parameter is organism specific.
    s : float, optional
        Parameter for penalizing long-range edges. The default is None and
        will be set to 0.75 if organism is None.
        This parameter is organism specific.
    organism : str, optional
        Organism name. The default is None.
        If s, window_size and distance_constraint are None, will use
        organism-specific values.
        Otherwise, will use the values passed as arguments.
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
    verbose : int, optional
        Verbose level.
            0: no output at all
            1: tqdm progress bar
            2:detailed output
    seed : int, optional
        Seed for random number generator. The default is 42.
    njobs : int, optional
        Number of jobs to run in parallel. The default is 1.
    threads_per_worker : int, optional
        Number of threads per worker. The default is 1.

    Returns
    -------
    results : dict
        Dictionary with keys as window names and values as sparse matrices
        (csr) of co-accessibility scores.
    """
    default_organism = "human"
    organism_values = {
        "human": {
            "window_size": 500_000,
            "distance_constraint": 250_000,
            "s": 0.75,
        },
        "mouse": {
            "window_size": 500_000,
            "distance_constraint": 250_000,
            "s": 0.75,
        },
        "drosophila": {
            "window_size": 100_000,
            "distance_constraint": 50_000,
            "s": 0.85,
        },
    }

    if organism is not None:
        if organism in organism_values.keys():
            if window_size is None:
                window_size = organism_values[organism]["window_size"]
            else:
                warnings.warn(
                    """
                    window_size is not None, using the value passed as param.
                    """, UserWarning)
            if distance_constraint is None:
                distance_constraint = organism_values[organism][
                    "distance_constraint"]
            else:
                warnings.warn(
                    """
                    distance_constraint is not None,
                    using the value passed as argument.
                    """, UserWarning)
            if s is None:
                s = organism_values[organism]["s"]
            else:
                warnings.warn(
                    """
                    s is not None, using the value passed as argument.
                    """, UserWarning)
        else:
            raise ValueError(
                """
                Organism not found in organism_values.
                Please keep organism=None or use one of the organisms:
                {}.
                """.format(
                    list(organism_values.keys())
                )
            )
    else:
        none_values = []
        if window_size is None:
            none_values.append("window_size")
            window_size = organism_values[default_organism]["window_size"]
        if distance_constraint is None:
            none_values.append("distance_constraint")
            distance_constraint = window_size / 2
        if s is None:
            none_values.append("s")
            s = organism_values[default_organism]["s"]
        if none_values:
            citation = "https://cole-trapnell-lab.github.io/cicero-release/docs_m3/#important-considerations-for-non-human-data"
            warnings.warn(
                """
                No organism, nor value passed for the parameters: {0},
                using default values.
                The default values are defined from human and mouse data,
                you might want to change them if you are working with
                another organisms.

                Default values used:
                {1}

                You can check how to define them in {2}.
                """.format(
                    none_values,
                    {key: value for key, value in organism_values[
                        "human"].items()
                        if key in none_values},
                    citation
                    )
            )

    # Check if distance_constraint is not too high
    if distance_constraint > window_size:
        raise ValueError(
            """
            distance_constraint should be lower than window_size.
            """
        )

    # Check if distance_constraint is not too high
    if distance_constraint is None:
        distance_constraint = window_size / 2

    # AnnData object should have more than 1 cell
    if adata.X.shape[0] < 2 or adata.X.shape[1] < 2:
        raise ValueError(
            """
            Not enough cells/regions in the AnnData object.
            You need at least 2 cells and 2 regions to calculate
            co-accessibility.
            """
        )

    alpha = average_alpha(
        adata,
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
        seed=seed,
    )
    if verbose >= 2:
        print("Alpha coefficient calculated : {}".format(alpha))

    # Custom log handler to capture log messages of distributed on memory
    class ListLogHandler(logging.Handler):
        def emit(self, record):
            log_messages.append(self.format(record))
    logger = logging.getLogger('distributed.worker.memory')
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # List to store log messages
    log_messages = []
    handler = ListLogHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    try:
        # TODO chr_progresses = Progress()
        with LocalCluster(
            n_workers=njobs,  # Number of workers (matches njobs from Joblib)
            processes=False,   # Use processes for isolation
            threads_per_worker=threads_per_worker,  # Single-threaded
        ) as cluster, Client(cluster) as client:   # (best for CPU-bound tasks)

            if verbose:
                # Optional: Monitor your computation with the Dask dashboard
                print(client.dashboard_link)

            # Configure joblib to use the default joblib parameters
            with parallel_config():
                chr_results = Parallel(n_jobs=njobs)(delayed(
                    chr_batch_graphical_lasso)(
                    adata.X[:, (adata.var["chromosome"] == chromosome).values],
                    adata.var.loc[adata.var["chromosome"] == chromosome, :],
                    chromosome,
                    alpha,
                    unit_distance,
                    window_size,
                    init_method,
                    max_elements,
                    n=n,
                    disable_tqdm=(verbose < 1),
                ) for n, chromosome in enumerate(
                    adata.var["chromosome"].unique()))

    except Exception as e:
        logger.warning("Exception occurred: %s", e)

    # Display collected log messages at the end of the script
    if verbose >= 2:
        print("Captured Warning Messages:")
        for message in log_messages:
            print(message)
    else:
        if len(log_messages) > 0:
            print(
                """
                Logger: {} warnings message have been returned by distributed
                about workers memory.\n
                It's usually expected, but you can display them with verbose=2
                """.format(len(log_messages)))

    full_results = sp.sparse.block_diag(chr_results, format="csr")
    return full_results


def chr_batch_graphical_lasso(
    chr_X,
    chr_var,
    chromosome,
    alpha,
    unit_distance,
    window_size,
    init_method,
    max_elements,
    n=0,
    disable_tqdm=False,
):

    results = {}
    idx_ = {}
    idy_ = {}
    map_indices = {region: i for i, region in enumerate(chr_var.index)}
    start_slidings = [0, int(window_size / 2)]

    for k in start_slidings:
        slide_results = {}
        slide_results["scores"] = np.array([])
        slide_results["idx"] = np.array([], dtype=int)
        slide_results["idy"] = np.array([], dtype=int)
        idx_["window_" + str(k)] = np.array([], dtype=int)
        idy_["window_" + str(k)] = np.array([], dtype=int)

        # Get start positions of windows
        window_starts = [
            i
            for i in range(
                k,
                chr_var["end"][
                    chr_var["chromosome"] == chromosome].max(),
                window_size,
            )
        ]

        idxs = []
        for start in window_starts:
            end = start + window_size
            # Get global indices of regions in the window
            idx = np.where(
                    ((chr_var["start"] > start)
                     & (chr_var["start"] < end-1))
                    |
                    ((chr_var["end"] > start)
                     & (chr_var["end"] < end-1))
                    )[0]
            if 1 < len(idx) <= max_elements:
                idxs.append(idx)
                x_, y_ = \
                    np.meshgrid(idxs[-1], idxs[-1])
                idx_["window_" + str(k)], idy_["window_" + str(k)] = \
                    np.concatenate([idx_["window_" + str(k)], x_.flatten()]), \
                    np.concatenate([idy_["window_" + str(k)], y_.flatten()])
        if idxs == []:
            results["window_" + str(k)] = sp.sparse.coo_matrix(
                (np.array([], dtype=int),
                 (np.array([], dtype=int),
                  np.array([], dtype=int))),
                shape=(chr_X.shape[1], chr_X.shape[1]),
            )
            continue
        # Use joblib.Parallel to run the function in parallel
        # on the different chromosomes
        parallel_lasso_results = Parallel(n_jobs=1, backend="threading")(
            delayed(single_graphical_lasso)(
                idx, chr_X, chr_var.iloc[idx, :],
                alpha,
                unit_distance,
                init_method,
                map_indices)
            for idx in tqdm.tqdm(
                idxs,
                position=n, leave=False,
                disable=disable_tqdm,
                desc=f"Processing chromosome: '{chromosome}'"))

        # Unpack the results and concatenate the arrays
        scores_list, idx_list, idy_list = zip(*parallel_lasso_results)

        # Concatenate the lists into the final result arrays
        slide_results = {
            "scores": np.concatenate([slide_results["scores"], *scores_list]),
            "idx": np.concatenate([slide_results["idx"], *idx_list]),
            "idy": np.concatenate([slide_results["idy"], *idy_list])
        }
        # Create sparse matrix
        results["window_" + str(k)] = sp.sparse.coo_matrix(
            (slide_results["scores"],
                (slide_results["idx"], slide_results["idy"])),
            shape=(chr_X.shape[1], chr_X.shape[1]),
        )
    return reconcile(results, idx_, idy_)


def single_graphical_lasso(
    idx,
    memmap_data,
    anndata_var,
    alpha,
    unit_distance,
    init_method, map_indices
):

    if idx is None or len(idx) <= 1:
        return np.array([], dtype=int), \
            np.array([], dtype=int), \
            np.array([], dtype=int)

    memmap_subset = memmap_data[:, idx]
    # Get submatrix
    if sp.sparse.issparse(memmap_subset):
        window_scores = np.cov(memmap_subset.toarray(), rowvar=False) + \
            1e-4 * np.eye(memmap_subset.shape[1])

    else:
        window_scores = np.cov(memmap_subset, rowvar=False) + \
            1e-4 * np.eye(memmap_subset.shape[1])

    distance = get_distances_regions_from_dataframe(anndata_var)

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

    # Transform to correlation matrix
    scores = sp.sparse.coo_matrix(
        cov_to_corr(graph_lasso_model.covariance_))

    # Convert corrected_scores column
    # and row indices to global indices
    idx = [
        map_indices[name]
        for name in anndata_var.index.values[scores.row]
    ]
    idy = [
        map_indices[name]
        for name in anndata_var.index.values[scores.col]
    ]
    return scores.data, idx, idy


def reconcile(
    results_gl,
    idx_gl,
    idy_gl
):

    results_keys = list(results_gl.keys())
    #################
    # To keep entries contained in 2 windows
    # sum of values per non-null locations
    average = reduce(lambda x, y: x+y,
                     [results_gl[k] for k in results_keys])

    # Initiate divider depending on number of overlapping windows
    divider = sp.sparse.csr_matrix(
        (np.ones(len(idx_gl[results_keys[0]])),
         (idx_gl[results_keys[0]],
          idy_gl[results_keys[0]])),
        shape=average.shape
    )
    for k in results_keys[1:]:
        divider = divider + sp.sparse.csr_matrix(
            ([1 for i in range(len(idx_gl[k]))],
             (idx_gl[k],
              idy_gl[k])),
            shape=average.shape
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
    divider = sp.sparse.csr_matrix.multiply(
        divider, average.astype(bool).astype(int))

    # Delete the sign_disagreeing matrix
    del signs_disaggreeing

    # Divide the sum by number of values
    average.data = average.data/divider.data
    return average
