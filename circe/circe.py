import warnings
from typing import Union
import numpy as np
import scipy as sp
import anndata as ad

from circe.utils import ORGANISM_DEFAULTS, reconcile

warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")


def compute_atac_network(
    adata,
    method='graphical_lasso',
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
    # VAE-specific parameters (ignored when method='graphical_lasso')
    epochs=50,
    batch_size=32,
    hidden_layer=None,
    latent_dim=None,
    metric='pearson',
):
    """
    Compute co-accessibility scores between regions in a sparse matrix, stored
    in the varp slot of the passed anndata object.

    Two methods are available:
    - 'graphical_lasso': Uses graphical lasso with distance penalties (default)
    - 'vae': Uses VAE latent embeddings with correlation or cosine similarity

    Parameters
    ----------
    adata : anndata object
        anndata object with var_names as region names.
    method : str, optional
        Method to use: 'graphical_lasso' or 'vae'. Default is 'graphical_lasso'.
    window_size : int, optional
        Size of sliding window, in which co-accessible regions can be found.
        The default is None and will be set to 500000 if organism is None.
        This parameter is organism specific.
    unit_distance : int, optional
        Distance between two regions in the matrix, in base pairs.
        The default is 1000. Only used with 'graphical_lasso' method.
    distance_constraint : int, optional
        Distance threshold for defining long-range edges.
        It is used to fit the penalty coefficient alpha.
        The default is None and will be set to 250000 if organism is None.
        This parameter is organism specific. Only used with 'graphical_lasso' method.
    s : float, optional
        Parameter for penalizing long-range edges. The default is None and
        will be set to 0.75 if organism is None. This parameter is organism
        specific. Only used with 'graphical_lasso' method.
    organism : str, optional
        Organism name. The default is None.
        If s, window_size and distance_constraint are None, will use
        organism-specific values.
        Otherwise, will use the values passed as arguments.
    max_alpha_iteration : int, optional
        Maximum number of iterations to calculate optimal penalty coefficient.
        The default is 100. Only used with 'graphical_lasso' method.
    distance_parameter_convergence : float, optional
        Convergence parameter for alpha (penalty) coefficiant calculation.
        The default is 1e-22. Only used with 'graphical_lasso' method.
    max_elements : int, optional
        Maximum number of regions in a window. The default is 200.
    n_samples : int, optional
        Number of windows used to calculate optimal penalty coefficient alpha.
        The default is 100. Only used with 'graphical_lasso' method.
    n_samples_maxtry : int, optional
        Maximum number of windows to try to calculate optimal penalty
        coefficient alpha. Should be higher than n_samples. The default is 500.
        Only used with 'graphical_lasso' method.
    key : str, optional
        Key to store the results in adata.varp. The default is "atac_network".
    seed : int, optional
        Seed for random number generator. The default is 42.
        Only used with 'graphical_lasso' method.
    njobs : int, optional
        Number of jobs to run in parallel. The default is 1.
    threads_per_worker : int, optional
        Number of threads per worker. The default is 1.
        Only used with 'graphical_lasso' method.
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
    epochs : int, optional
        Number of VAE training epochs. Default is 50.
        Only used with 'vae' method.
    batch_size : int, optional
        Batch size for VAE training. Default is 32.
        Only used with 'vae' method.
    hidden_layer : int, optional
        Hidden layer dimension for VAE. Auto-determined if None.
        Only used with 'vae' method.
    latent_dim : int, optional
        Latent dimension for VAE. Auto-determined if None.
        Only used with 'vae' method.
    metric : str, optional
        Metric to use for computing similarity: 'pearson' or 'cosine'.
        Default is 'pearson'. Only used with 'vae' method.

    Returns
    -------
    None.
    """

    if method == 'vae':
        from circe.latent_network import compute_latent_network
        
        # Set default window_size if needed
        if window_size is None:
            if organism is not None and organism in ORGANISM_DEFAULTS:
                window_size = ORGANISM_DEFAULTS[organism]["window_size"]
            else:
                window_size = ORGANISM_DEFAULTS["human"]["window_size"]
        
        adata.varp[key] = compute_latent_network(
            adata=adata,
            window_size=window_size,
            max_elements=max_elements,
            epochs=epochs,
            batch_size=batch_size,
            hidden_layer=hidden_layer,
            latent_dim=latent_dim,
            verbose=verbose,
            metric=metric,
        )
    else:
        from circe.gl_network import sliding_graphical_lasso
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
