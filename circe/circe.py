import warnings

from circe.utils import resolve_organism_params

warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")


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
    chromosomes_sizes=None,
):
    """
    Compute co-accessibility scores between regions in a sparse matrix, stored
    in the varp slot of the passed anndata object.

    Uses graphical lasso with distance penalties.

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

    # Resolve organism params ONCE at top level
    window_size, distance_constraint, s = resolve_organism_params(
        organism, window_size, distance_constraint, s
    )

    from circe.graphical_lasso import sliding_graphical_lasso
    adata.varp[key] = sliding_graphical_lasso(
        adata=adata,
        window_size=window_size,
        unit_distance=unit_distance,
        distance_constraint=distance_constraint,
        s=s,
        organism=None,  # Already resolved
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
