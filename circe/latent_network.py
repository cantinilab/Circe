"""VAE-based latent co-accessibility computation."""

import numpy as np
import scipy as sp
import scipy.sparse
import anndata as ad
import tqdm
from typing import Union, Optional
from joblib import Parallel, delayed, parallel_config

from circe.circe import reconcile


def compute_latent_network(
    adata: ad.AnnData,
    window_size: int = 500_000,
    max_elements: int = 200,
    epochs: int = 50,
    batch_size: int = 32,
    hidden_layer: Optional[int] = None,
    latent_dim: Optional[int] = None,
    verbose: int = 0,
    njobs: int = 1,
    chromosomes_sizes: Optional[dict] = None,
) -> scipy.sparse.csr_matrix:
    """
    Compute co-accessibility scores using VAE latent embeddings.

    Parameters
    ----------
    adata : AnnData
        AnnData object with peaks in var and cells in obs.
    window_size : int
        Size of sliding window in base pairs.
    max_elements : int
        Maximum number of peaks in a window.
    epochs : int
        Number of VAE training epochs.
    batch_size : int
        Batch size for VAE training.
    hidden_layer : int, optional
        Hidden layer dimension. Auto-determined if None.
    latent_dim : int, optional
        Latent dimension. Auto-determined if None.
    verbose : int
        Verbosity level (0, 1, or 2).
    njobs : int
        Number of parallel jobs for chromosome processing.
    chromosomes_sizes : dict, optional
        Dictionary mapping chromosome names to sizes.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of co-accessibility scores.
    """
    # Import TensorFlow only when needed
    try:
        import tensorflow as tf
        from circe.vae import VAE
    except ImportError:
        raise ImportError(
            "TensorFlow is required for VAE method. "
            "Install with: pip install tensorflow"
        )

    # Configure TensorFlow threading
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    if verbose >= 1:
        print("Preprocessing data...")
    
    # Convert to dense array if sparse
    if sp.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.asarray(adata.X)
    
    # Log2 transform (FAVA-style)
    X = np.log2(1 + X)
    
    # Transpose to (n_peaks, n_cells) for VAE
    X = X.T
    
    # Per-row (per-peak) min-max normalization to [0, 1] for sigmoid decoder
    row_min = X.min(axis=1, keepdims=True)
    row_max = X.max(axis=1, keepdims=True)
    X_norm = (X - row_min) / (row_max - row_min + 1e-8)
    X_norm = np.nan_to_num(X_norm)
    
    # Determine architecture dimensions
    original_dim = X_norm.shape[1]  # n_cells
    
    if hidden_layer is None:
        if original_dim >= 2000:
            hidden_layer = 1000
        elif original_dim >= 500:
            hidden_layer = 500
        else:
            hidden_layer = max(50, original_dim // 2)
    
    if latent_dim is None:
        if hidden_layer >= 1000:
            latent_dim = 100
        elif hidden_layer >= 500:
            latent_dim = 50
        else:
            latent_dim = max(5, hidden_layer // 10)
    
    if verbose >= 1:
        print(f"Training VAE (hidden={hidden_layer}, latent={latent_dim})...")
    
    # Train VAE
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.001)
    x_train = x_test = np.array(X_norm, dtype=np.float32)
    
    vae = VAE(
        opt, x_train, x_test, batch_size,
        original_dim, hidden_layer, latent_dim, epochs
    )
    
    # Extract latent embeddings for all peaks
    if verbose >= 1:
        print("Extracting latent embeddings...")
    
    encoder_outputs = vae.encoder.predict(x_test, batch_size=batch_size, verbose=0)
    # encoder_outputs is list of [z_mean, z_log_var, z], each shape (n_peaks, latent_dim)
    # Concatenate all three outputs like FAVA
    x_test_encoded = np.stack(encoder_outputs, axis=0)  # (3, n_peaks, latent_dim)
    latent_embeddings = x_test_encoded.transpose(1, 0, 2).reshape(x_test.shape[0], -1)
    # latent_embeddings shape: (n_peaks, 3*latent_dim)
    
    # Process each chromosome
    if verbose >= 1:
        print("Computing correlations within windows...")
    
    with parallel_config(n_jobs=njobs):
        chr_results = Parallel(n_jobs=njobs, verbose=verbose)(
            delayed(chr_latent_correlation)(
                adata[:, (adata.var["chromosome"] == chromosome).values].X,
                adata.var.loc[adata.var["chromosome"] == chromosome, :].copy(),
                chromosome,
                latent_embeddings[adata.var["chromosome"] == chromosome],
                window_size,
                max_elements,
                n=n,
                disable_tqdm=(verbose < 1),
            ) for n, chromosome in enumerate(adata.var["chromosome"].unique())
        )
    
    # Concatenate results from all chromosomes
    if verbose >= 1:
        print("Concatenating results from all chromosomes...")
    
    full_results = sp.sparse.block_diag(chr_results, format="csr")
    return full_results


def chr_latent_correlation(
    chr_X,
    chr_var,
    chromosome,
    chr_latent_embeddings,
    window_size,
    max_elements,
    n=0,
    disable_tqdm=False,
):
    """
    Compute correlations within sliding windows for a single chromosome.

    Parameters
    ----------
    chr_X : array
        Chromosome data matrix (not used, kept for API consistency).
    chr_var : DataFrame
        Chromosome var metadata.
    chromosome : str
        Chromosome name.
    chr_latent_embeddings : array
        Latent embeddings for peaks in this chromosome (n_peaks, 3*latent_dim).
    window_size : int
        Window size in base pairs.
    max_elements : int
        Maximum peaks per window.
    n : int
        Position for tqdm display.
    disable_tqdm : bool
        Disable progress bar.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse correlation matrix for this chromosome.
    """
    results = {}
    idx_ = {}
    idy_ = {}
    start_slidings = [0, int(window_size / 2)]
    
    # Create mapping indices
    map_indices = "global_idx"
    if map_indices in chr_var.columns:
        old = chr_var[map_indices].to_numpy()
        new = np.arange(len(chr_var), dtype=np.int64)
        if not np.array_equal(old, new):
            raise ValueError(
                f"{map_indices} already exists and differs. "
                "Choose another name or delete the column first."
            )
    else:
        chr_var.loc[:, map_indices] = np.arange(len(chr_var), dtype=np.int64)
    
    for k in start_slidings:
        slide_results = {
            "scores": np.array([]),
            "idx": np.array([], dtype=int),
            "idy": np.array([], dtype=int),
        }
        idx_["window_" + str(k)] = np.array([], dtype=int)
        idy_["window_" + str(k)] = np.array([], dtype=int)
        
        # Get window start positions
        chr_max = chr_var["end"].max()
        window_starts = list(range(k, chr_max, window_size))
        
        idxs = []
        for start in window_starts:
            end = start + window_size
            # Get indices of peaks in this window
            idx = np.where(
                ((chr_var["start"] > start) & (chr_var["start"] < end - 1))
                | ((chr_var["end"] > start) & (chr_var["end"] < end - 1))
            )[0]
            
            if 1 < len(idx) <= max_elements:
                idxs.append(idx)
                x_, y_ = np.meshgrid(idxs[-1], idxs[-1])
                idx_["window_" + str(k)] = np.concatenate([
                    idx_["window_" + str(k)], x_.flatten()
                ])
                idy_["window_" + str(k)] = np.concatenate([
                    idy_["window_" + str(k)], y_.flatten()
                ])
        
        if not idxs:
            results["window_" + str(k)] = sp.sparse.coo_matrix(
                (np.array([], dtype=int),
                 (np.array([], dtype=int), np.array([], dtype=int))),
                shape=(len(chr_var), len(chr_var)),
            )
            continue
        
        # Compute correlations for each window
        scores_list = []
        idx_list = []
        idy_list = []
        
        for idx in tqdm.tqdm(
            idxs,
            position=n,
            leave=False,
            disable=disable_tqdm,
            desc=f"Processing chromosome: '{chromosome}'"
        ):
            # Get latent embeddings for peaks in this window
            window_embeddings = chr_latent_embeddings[idx]
            
            # Compute Pearson correlation matrix
            corr_matrix = np.corrcoef(window_embeddings)
            
            # Extract upper triangle (avoid duplicates and diagonal)
            n_peaks = len(idx)
            upper_idx = np.triu_indices(n_peaks, k=1)
            scores = corr_matrix[upper_idx]
            
            # Map back to global indices
            global_idx = chr_var[map_indices].values
            global_i = global_idx[idx[upper_idx[0]]]
            global_j = global_idx[idx[upper_idx[1]]]
            
            scores_list.append(scores)
            idx_list.append(global_i)
            idy_list.append(global_j)
        
        # Concatenate results
        slide_results = {
            "scores": np.concatenate([slide_results["scores"], *scores_list]),
            "idx": np.concatenate([slide_results["idx"], *idx_list]),
            "idy": np.concatenate([slide_results["idy"], *idy_list]),
        }
        
        # Create sparse matrix
        results["window_" + str(k)] = sp.sparse.coo_matrix(
            (slide_results["scores"], (slide_results["idx"], slide_results["idy"])),
            shape=(len(chr_var), len(chr_var)),
        )
    
    # Reconcile results from all windows
    return reconcile(results, idx_, idy_)

