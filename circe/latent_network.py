"""VAE-based latent co-accessibility computation."""

import numpy as np
import scipy as sp
import scipy.sparse
import anndata as ad
from typing import Optional
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from circe.circe import reconcile


# -----------------------------------------------------------------------------
# Private helper functions
# -----------------------------------------------------------------------------

def _preprocess_data(adata):
    """Convert to dense, log2 transform, transpose, and normalize."""
    X = adata.X.toarray() if sp.sparse.issparse(adata.X) else np.asarray(adata.X)
    X = np.log2(1 + X).T  # Transpose to (n_peaks, n_cells)
    
    # Per-row min-max normalization to [0, 1]
    row_min, row_max = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    return np.nan_to_num((X - row_min) / (row_max - row_min + 1e-8))


def _determine_architecture(original_dim, hidden_layer=None, latent_dim=None):
    """Determine hidden layer and latent dimensions based on input size."""
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
    
    return hidden_layer, latent_dim


def _extract_latent_embeddings(vae, x_data, batch_size):
    """Extract and reshape latent embeddings from VAE encoder."""
    encoder_outputs = vae.encoder.predict(x_data, batch_size=batch_size, verbose=0)
    # encoder_outputs is [z_mean, z_log_var, z], each shape (n_peaks, latent_dim)
    x_encoded = np.stack(encoder_outputs, axis=0)  # (3, n_peaks, latent_dim)
    return x_encoded.transpose(1, 0, 2).reshape(x_data.shape[0], -1)


def _compute_window_indices(chr_var, window_size, max_elements, k):
    """Compute peak indices for each window in a sliding window pass."""
    chr_max = chr_var['end'].max()
    
    idxs = []
    mesh_idx = []
    mesh_idy = []
    
    for start in range(k, chr_max, window_size):
        end = start + window_size
        idx = np.where(
            ((chr_var['start'] > start) & (chr_var['start'] < end - 1))
            | ((chr_var['end'] > start) & (chr_var['end'] < end - 1))
        )[0]
        
        if 1 < len(idx) <= max_elements:
            idxs.append(idx)
            x_, y_ = np.meshgrid(idx, idx)
            mesh_idx.append(x_.flatten())
            mesh_idy.append(y_.flatten())
    
    return idxs, mesh_idx, mesh_idy


def _compute_window_correlation(idx, chr_latent_embeddings, global_idx):
    """Compute Pearson correlations for peaks in a single window."""
    window_embeddings = chr_latent_embeddings[idx]
    corr_matrix = np.corrcoef(window_embeddings)
    
    upper_idx = np.triu_indices(len(idx), k=1)
    scores = corr_matrix[upper_idx]
    global_i = global_idx[idx[upper_idx[0]]]
    global_j = global_idx[idx[upper_idx[1]]]
    
    return scores, global_i, global_j


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def compute_latent_network(
    adata: ad.AnnData,
    window_size: int = 500_000,
    max_elements: int = 200,
    epochs: int = 50,
    batch_size: int = 32,
    hidden_layer: Optional[int] = None,
    latent_dim: Optional[int] = None,
    verbose: int = 0,
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
            'TensorFlow is required for VAE method. '
            'Install with: pip install tensorflow'
        )

    # Configure TensorFlow threading
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    if verbose >= 1:
        print('Preprocessing data...')
    X_norm = _preprocess_data(adata)
    
    original_dim = X_norm.shape[1]  # n_cells
    hidden_layer, latent_dim = _determine_architecture(original_dim, hidden_layer, latent_dim)
    
    if verbose >= 1:
        print(f'Training VAE (hidden={hidden_layer}, latent={latent_dim})...')
    
    # Train VAE
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.001)
    x_train = np.array(X_norm, dtype=np.float32)
    vae = VAE(opt, x_train, x_train, batch_size, original_dim, hidden_layer, latent_dim, epochs)
    
    if verbose >= 1:
        print('Extracting latent embeddings...')
    latent_embeddings = _extract_latent_embeddings(vae, x_train, batch_size)
    
    # Process each chromosome with progress bar
    chromosomes = list(adata.var['chromosome'].unique())
    progress_columns = (
        '[progress.description]{task.description}',
        BarColumn(),
        '[progress.percentage]{task.percentage:>3.0f}%',
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    
    chr_results = []
    with Progress(*progress_columns, transient=False, disable=(verbose < 1)) as prog:
        task = prog.add_task('Processing chromosomes', total=len(chromosomes))
        for chromosome in chromosomes:
            chr_mask = (adata.var['chromosome'] == chromosome).values
            result = chr_latent_correlation(
                adata.var.loc[chr_mask, :].copy(),
                chr_latent_embeddings=latent_embeddings[chr_mask],
                window_size=window_size,
                max_elements=max_elements,
            )
            chr_results.append(result)
            prog.update(task, advance=1)
        prog.refresh()
    
    if verbose >= 1:
        print('Concatenating results...')
    
    return sp.sparse.block_diag(chr_results, format='csr')


def chr_latent_correlation(chr_var, chr_latent_embeddings, window_size, max_elements):
    """
    Compute correlations within sliding windows for a single chromosome.

    Parameters
    ----------
    chr_var : DataFrame
        Chromosome var metadata.
    chr_latent_embeddings : array
        Latent embeddings for peaks in this chromosome (n_peaks, 3*latent_dim).
    window_size : int
        Window size in base pairs.
    max_elements : int
        Maximum peaks per window.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse correlation matrix for this chromosome.
    """
    results = {}
    idx_ = {}
    idy_ = {}
    start_slidings = [0, window_size // 2]
    
    # Create mapping indices (chr_var is already a copy, safe to assign directly)
    global_idx = np.arange(len(chr_var), dtype=np.int64)
    
    for k in start_slidings:
        window_key = f'window_{k}'
        
        # Compute window indices
        idxs, mesh_idx, mesh_idy = _compute_window_indices(chr_var, window_size, max_elements, k)
        
        # Store mesh indices
        idx_[window_key] = np.concatenate(mesh_idx) if mesh_idx else np.array([], dtype=int)
        idy_[window_key] = np.concatenate(mesh_idy) if mesh_idy else np.array([], dtype=int)
        
        if not idxs:
            results[window_key] = sp.sparse.coo_matrix(
                ([], ([], [])), shape=(len(chr_var), len(chr_var)), dtype=float
            )
            continue
        
        # Compute correlations for each window
        scores_list = []
        idx_list = []
        idy_list = []
        
        for idx in idxs:
            scores, global_i, global_j = _compute_window_correlation(
                idx, chr_latent_embeddings, global_idx
            )
            scores_list.append(scores)
            idx_list.append(global_i)
            idy_list.append(global_j)
        
        # Create sparse matrix
        results[window_key] = sp.sparse.coo_matrix(
            (np.concatenate(scores_list),
             (np.concatenate(idx_list), np.concatenate(idy_list))),
            shape=(len(chr_var), len(chr_var)),
        )
    
    return reconcile(results, idx_, idy_)
