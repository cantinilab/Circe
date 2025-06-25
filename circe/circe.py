import logging  # Visual settings
import warnings
from typing import Union
import tqdm
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
    )
import numpy as np  # Mathematical operations
import pandas as pd
import scipy as sp
import random
from functools import reduce
from circe import quic_graph_lasso  # Graphical Lasso implementation
from circe.metrics import cov_with_appended_zeros
from dask.distributed import Client, as_completed  # Parallel execution
from joblib import Parallel, delayed, parallel_config
import asyncio

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
    verbose=0,
    chromosomes_sizes=None
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
        The default is 0.
    chromosomes_sizes : dict, optional
        Dictionary with chromosome sizes. If None, will use the maximum
        end position of each chromosome in adata.var.
        The default is None.

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
        verbose=verbose,
        chromosomes_sizes=chromosomes_sizes,
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
    zrow,  # number of rows removed (0 if no rows filled with zeros)
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
    zrow : int
        Number of rows removed from X (0 if no rows filled with zeros).
        It will be used to correct covariance matrix once calculated from 
        only non-zero rows.
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
        cov = cov_with_appended_zeros(X, zrow, rowvar=False)
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


# ──────────────────────────────────────────────────────────────────────────────
#  Required extra dependency
#       pip install "dask[distributed]" rich

# ──────────────────────────────────────────────────────────────────────────────
#  Helper : one task = one window
#  (defined outside to make it picklable; receives **only scalars + adata**)
# ──────────────────────────────────────────────────────────────────────────────
def _alpha_task(X_window,
                zrow,  # number of rows removed (0 if no rows filled with 0s)
                chromosomes,          # 1-D array[str]  (len = n_peaks)
                starts,               # 1-D array[int]
                ends,                 # 1-D array[int]
                *,                    # -------- other keyword params --------
                max_alpha_iteration,
                unit_distance,
                s,
                distance_constraint,
                distance_parameter_convergence,
                max_elements,
                init_method):
    """
    Compute alpha for a single genomic window, sending only the data that
    matter to the worker (matrix slice + minimal metadata).
    """

    # ------------------------------------------------------------------
    # 1. Rebuild a tiny DataFrame just for distances
    # ------------------------------------------------------------------
    var_df = pd.DataFrame({
        "chromosome": chromosomes,
        "start":      starts,
        "end":        ends,
    })

    distances = get_distances_regions_from_dataframe(var_df)

    # ------------------------------------------------------------------
    # 2. Run local_alpha on the slice
    # ------------------------------------------------------------------
    alpha = local_alpha(
        X=X_window,
        zrow=zrow,
        distances=distances,
        maxit=max_alpha_iteration,
        unit_distance=unit_distance,
        s=s,
        distance_constraint=distance_constraint,
        distance_parameter_convergence=distance_parameter_convergence,
        max_elements=max_elements,
        init_method=init_method,
    )

    return alpha if isinstance(alpha, (int, float)) else None


def _build_payload(adata, window_idx):
    """
    Build the payload for one genomic window.

    Returns
    -------
    (X_window, chrom, start, end)   : tuple
        X_window  : sparse *or* dense array with empty columns removed
        chrom     : 1-D array of chromosome names (len = n_peaks_kept)
        start     : 1-D int array of start coordinates
        end       : 1-D int array of end coordinates
    None                             : if the window has < 2 non-zero columns
    """
    # 1. take the window slice (still sparse or dense, but thin -> n_cells × m)
    Xw = adata[:, window_idx].X

    # 2. drop all-zero columns
    Xw, zrows = _remove_null_rows(Xw)
    if Xw is None:
        return None

    # 3. trim metadata with the same mask
    chrom = adata.var["chromosome"].values[window_idx]
    start = adata.var["start"].values[window_idx]
    end = adata.var["end"].values[window_idx]

    return Xw, zrows, chrom, start, end


def _remove_null_rows(X):
    """"""
    if sp.sparse.issparse(X):
        nz_rows = np.flatnonzero(X.getnnz(axis=1))
        zrows = X.shape[0] - nz_rows.shape[0]
        if nz_rows.size < 2:
            return None, 0
        # sparse slice already copies the relevant data/indices/indptr
        X = X[nz_rows, :]
    else:                                # dense ndarray
        nz_rows = np.flatnonzero((X != 0).any(axis=1))
        if nz_rows.size < 2:
            return None, 0
        zrows = X.shape[0] - nz_rows.shape[0]
        X = X[nz_rows, :].copy()       # ensure independent buffer!
    return X, zrows


def quiet_dask(verbose: int):
    """
    verbose = 0   → completely mute WARNINGS from Dask
    verbose = 1   → keep WARNINGS (default)
    verbose ≥ 2   → keep everything (INFO / DEBUG)
    """
    if verbose == 0:
        level = logging.ERROR          # show only errors
    elif verbose == 1:
        level = logging.WARNING        # default
    else:                               # verbose >= 2
        level = logging.INFO           # or DEBUG
    for name in (
        "distributed.worker.state_machine",
        "distributed.nanny",
    ):
        logging.getLogger(name).setLevel(level)


# ──────────────────────────────────────────────────────────────────────────────
#  Main function (parallel version)
# ──────────────────────────────────────────────────────────────────────────────
def average_alpha(
    adata,
    window_size=500_000,
    unit_distance=1_000,
    n_samples=100,
    n_samples_maxtry=500,
    max_alpha_iteration=100,
    s=0.75,
    distance_constraint=250_000,
    distance_parameter_convergence=1e-22,
    max_elements=200,
    chromosomes_sizes=None,
    init_method="precomputed",
    seed=42,
    verbose=False,
    *,
    # NEW optional parameters for parallel execution
    client: Client | None = None,  # pass an existing Dask client or None
    n_workers: int = 1,
    threads_per_worker: int = 1,
):
    """"
    Estimate the **global sparsity‐penalty coefficient α** used by
    _sliding graphical lasso_ on scATAC-seq data.

    The function samples `n_samples` genomic windows, fits an “individual”
    α for each window (via :func:`local_alpha`) and returns their average.
    Windows that do not satisfy quality criteria (size < ``max_elements``,
    <5 % long-range edges, <20 % co-accessible regions) are skipped.

    Parallel implementation
    -----------------------
    • One genomic window = one task executed in a Dask cluster.  
    • The full :class:`anndata.AnnData` object is broadcast to workers.
    • Tasks stream back through :func:`dask.distributed.as_completed`; as soon
      as `n_samples` usable α’s are collected, the remaining tasks are
      cancelled.

    Parameters
    ----------
    adata : anndata.AnnData
        Input accessibility matrix with ``var`` containing at least the
        columns ``chromosome``, ``start`` and ``end`` (0-based, half-open).
    window_size : int, default 500_000
        Genomic size (bp) of the sliding window.
    unit_distance : int, default 1_000
        Unit (bp) used to rescale genomic distances prior to penalty
        weighting.
    n_samples : int, default 100
        Number of windows retained to compute the average α.
    n_samples_maxtry : int, default 500
        Maximum number of candidate windows to inspect in order to obtain
        `n_samples` valid ones.
    max_alpha_iteration : int, default 100
        Maximum iterations in the fixed-point search performed by
        :func:`local_alpha`.
    s : float, default 0.75
        Long-range penalty exponent (organism specific).
    distance_constraint : int, default 250_000
        Threshold (bp) above which an edge is considered long-range.
    distance_parameter_convergence : float, default 1e-22
        Convergence criterion for α.
    max_elements : int, default 200
        Upper bound on the number of regions (columns) allowed in a window.
    chromosomes_sizes : dict, optional
        Mapping ``{chromosome: size_in_bp}``.  
        By default the maximum ``end`` coordinate found in `adata.var`
        is used for each chromosome.
    init_method : {"precomputed", ...}, default "precomputed"
        Initialisation method forwarded to :func:`local_alpha`.
    seed : int, default 42
        Random seed used for the window shuffle.
    verbose : bool, default False
        Emit warnings when fewer than `n_samples` usable windows are found.

    Parallel-execution options
    --------------------------
    client : dask.distributed.Client, optional
        Existing Dask client / cluster.  When *None* (default) a **local**
        cluster is started with the resources below and shut down on exit.
    n_workers : int, default 8
        Number of worker processes in the auto-started local cluster (ignored
        if `client` is provided).
    threads_per_worker : int, default 1
        Number of OS threads per worker process.

    Returns
    -------
    alpha : float
        Mean sparsity-penalty coefficient across the selected windows.
        ``nan`` if no window satisfied the criteria.

    Warnings
    --------
    A :class:`UserWarning` is raised (when ``verbose=True``) if fewer than
    `n_samples` windows pass the filters.

    Dependencies
    ------------
    ``dask[distributed]``, ``rich`` and ``anndata >= 0.9`` must be available.
    """

    # ────────────────────────────────────────────────────────────────
    # 0. Build candidate windows (same logic as original function)
    # ────────────────────────────────────────────────────────────────
    if verbose:
        print("Extracting chromosome sizes...")
    start_slidings = [0, window_size // 2]
    window_starts = []
    for off in start_slidings:
        for chrom in adata.var["chromosome"].unique():
            chr_size = (
                chromosomes_sizes.get(chrom)
                if chromosomes_sizes is not None and chrom in chromosomes_sizes
                else adata.var["end"][adata.var["chromosome"] == chrom].max()
            )
            window_starts.extend((chrom, p)
                                 for p in range(off, chr_size, window_size))

    rng = random.Random(seed)
    rng.shuffle(window_starts)
    if verbose:
        print("Selecting {} genomic windows...".format(n_samples_maxtry))
    random_windows: list[np.ndarray] = []
    while len(random_windows) < n_samples_maxtry and window_starts:
        need = n_samples_maxtry - len(random_windows)
        batch, window_starts = window_starts[:need], window_starts[need:]
        for chrom, start in batch:
            end = start + window_size
            idx = np.where(
                (adata.var["chromosome"] == chrom)
                & (
                    ((adata.var["start"] > start) &
                     (adata.var["start"] < end - 1))
                    | ((adata.var["end"] > start) &
                       (adata.var["end"] < end - 1))
                )
            )[0]
            if 0 < len(idx) < max_elements:
                random_windows.append(idx)

    # Keep only windows larger than the distance constraint (i.e. with at least
    # one long-range edge)
    random_windows = [
        window for window in random_windows
        if (
            (adata.var.loc[adata.var_names[window[-1]], "end"]
                + adata.var.loc[adata.var_names[window[-1]], "start"])/2
            -
            (adata.var.loc[adata.var_names[window[0]], "end"]
                + adata.var.loc[adata.var_names[window[0]], "start"])/2
            > distance_constraint
            )]
    # ────────────────────────────────────────────────────────────────
    # 1. Dask client (local if none supplied)
    # ────────────────────────────────────────────────────────────────
    created_client = False

    # silence to be created dask instance
    quiet_dask(verbose)

    if client is None:
        client = Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit="0",
            timeout="60s",  # allow time for shutdown
        )
        if verbose:
            print("Created new Dask client with {} workers.".format(
                n_workers))
            print(client.dashboard_link)
        created_client = True

    # ``nullcontext`` does nothing, so the same ``with`` line works
    # for both cases
    try:
        # ────────────────────────────────────────────────────────────────
        # 2. Submit ALL windows immediately
        # ────────────────────────────────────────────────────────────────

        # ------------------------------------------------------------------
        # 0.  Build payloads on the client  (list-comprehension version)
        # ------------------------------------------------------------------
        if verbose:
            print("Building payloads for {} windows...".format(
                len(random_windows[:n_samples])))

        # rich progress bar options
        progress_columns = (
            "[progress.description]{task.description}",
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        # Use rich progress bar to show the progress of the payloads
        with Progress(
            *progress_columns,
            transient=False,
            auto_refresh=False
        ) as prog:

            payloads = Parallel(n_jobs=n_workers, verbose=0)(
                delayed(_build_payload)(adata, w) for w in
                prog.track(
                    random_windows[:n_samples],
                    description="Preparing {}".format(len(random_windows[:n_samples])) +
                    " random windows across the genome")
            )

        payloads = [p for p in payloads if p is not None]
        if verbose:
            print(len(payloads), "informative windows found.")

        if not payloads:                        # no informative windows
            raise RuntimeError("No informative windows found")

        # unzip the list of tuples into four parallel lists
        X_list, zrows, chrom_list, start_list, end_list = map(
            list, zip(*payloads))

        # 1. scatter only the big sparse matrices
        X_futures = client.scatter(X_list, broadcast=False)

        # 2. submit the computation tasks
        futures = [
            client.submit(
                _alpha_task,
                Xf, zrow, chrom, start, end,
                max_alpha_iteration=max_alpha_iteration,
                unit_distance=unit_distance,
                s=s,
                distance_constraint=distance_constraint,
                distance_parameter_convergence=distance_parameter_convergence,
                max_elements=max_elements,
                init_method=init_method,
            )
            for Xf, zrow, chrom, start, end in zip(
                X_futures,
                zrows,  # this is the number of rows removed
                chrom_list,
                start_list,
                end_list)
        ]

        # ────────────────────────────────────────────────────────────────
        # 3. Collect results until n_samples reached
        # ────────────────────────────────────────────────────────────────
        alpha_list: list[float] = []
        with Progress(*progress_columns, transient=False) as prog:
            bar = prog.add_task(
                "Calculating alpha",
                total=n_samples, auto_refresh=False)

            for fut in as_completed(futures):
                alpha = fut.result()
                if alpha is not None:
                    alpha_list.append(alpha)
                    prog.update(bar, advance=1)
                    prog.refresh()
                else:
                    print("Window skipped (no long edges" +
                          " or too many elements).")

                if len(alpha_list) >= n_samples:
                    # cancel leftovers and break
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
            if verbose:
                print("Calculating alpha over {} windows.".format(
                    len(alpha_list)))
    finally:
        if created_client:  # only shutdown if we created it
            try:
                client.close(timeout="120s")          # wait long, but finite
            except asyncio.TimeoutError:
                pass                                  # ignore late worker
        pass
    # ────────────────────────────────────────────────────────────────
    # 4. Clean-up & return
    # ────────────────────────────────────────────────────────────────
    if len(alpha_list) < n_samples and verbose:
        warnings.warn(
            f"only {len(alpha_list)} windows" +
            f" were usable (requested {n_samples}).",
            UserWarning,
        )
    return float(np.mean(alpha_list)) if alpha_list else np.nan


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
    threads_per_worker=1,
    chromosomes_sizes: Union[dict, None] = None
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
    chromosomes_sizes : dict, optional
        Dictionary with chromosome sizes. If None, will use the maximum
        end position of each chromosome in adata.var.
        The default is None.

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
        n_workers=njobs,
        threads_per_worker=1,
        verbose=verbose,
        chromosomes_sizes=chromosomes_sizes,
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
        print("Calculating co-accessibility scores...")
        # Configure joblib to use the default joblib parameters
        with parallel_config(n_jobs=njobs):
            chr_results = Parallel(n_jobs=njobs, verbose=verbose)(delayed(
                chr_batch_graphical_lasso)(
                adata[:, (adata.var["chromosome"] == chromosome).values].X,
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

    # Concatenate results from all chromosomes
    if verbose == 0:
        print("Concatenating results from all chromosomes...")
    else:
        print("Concatenating results from all chromosomes as a csr_matrix...")

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
    njobs=1,
    disable_tqdm=False,
    
):

    results = {}
    idx_ = {}
    idy_ = {}
    start_slidings = [0, int(window_size / 2)]

    map_indices = "global_idx"
    if map_indices in chr_var:
        # if it already exists, check that it contains the same numbering
        old = chr_var[map_indices].to_numpy()
        new = np.arange(len(chr_var), dtype=np.int64)
        if not np.array_equal(old, new):
            raise ValueError(f"{map_indices} already exists and differs. "
                             "Choose another name or delete the column first.")
        # identical → nothing to do
    else:
        chr_var.loc[:, map_indices] = np.arange(len(chr_var), dtype=np.int64)

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
        # inside one chromosome
        parallel_lasso_results = [
            single_graphical_lasso(
                idx,
                *_remove_null_rows(chr_X[:, idx]),
                anndata_var=chr_var.iloc[idx, :],
                alpha=alpha,
                unit_distance=unit_distance,
                init_method=init_method,
                map_indices=map_indices)
            for idx in tqdm.tqdm(
                idxs,
                position=n, leave=False,
                disable=disable_tqdm,
                desc=f"Processing chromosome: '{chromosome}'")]

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

    # Reconcile results from all windows
    return reconcile(results, idx_, idy_)


def single_graphical_lasso(
    idx,
    X,
    zrow,
    anndata_var,
    alpha,
    unit_distance,
    init_method,
    map_indices
):

    if idx is None or len(idx) <= 1 or X is None:
        return np.array([], dtype=int), \
            np.array([], dtype=int), \
            np.array([], dtype=int)

    # Get submatrix
    if sp.sparse.issparse(X):
        window_scores = cov_with_appended_zeros(
            X.toarray(), zrow, rowvar=False) + \
            1e-4 * np.eye(X.shape[1])
    else:
        window_scores = cov_with_appended_zeros(X, zrow, rowvar=False) + \
            1e-4 * np.eye(X.shape[1])

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
        anndata_var.loc[name, map_indices]
        for name in anndata_var.index.values[scores.row]
    ]
    idy = [
        anndata_var.loc[name, map_indices]
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
