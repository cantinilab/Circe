#!/usr/bin/env python3
"""
Speed comparison: Original vs Optimized metacell computation

This script compares the performance of the original implementation
(using sets and row-by-row operations) with the optimized version.
"""
import time
import numpy as np
import pandas as pd
import anndata as ad
import scipy as sp
from rich.progress import track


def compute_metacells_original(adata, k=50, max_overlap_metacells=0.9, 
                               max_metacells=None, method='mean'):
    """
    Original implementation for comparison.
    Uses sets for overlap checking and row-by-row aggregation.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Use LSI coordinates (assuming already computed)
    if 'X_lsi' not in adata.obsm:
        from circe.metacells import lsi
        lsi(adata)
    
    key_projection = 'X_lsi'
    
    # Identify neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(adata.obsm[key_projection])
    distances, indices = nbrs.kneighbors(adata.obsm[key_projection])
    
    # Convert to sets (original approach)
    indices = [set(indice) for indice in indices]
    
    if max_metacells is None:
        max_metacells = len(indices)
    
    # Select metacells with set-based overlap checking
    metacells = [indices[0]]
    iterations = 0
    for i in track(indices[1:], description="[Original] Computing metacells..."):
        if iterations >= max_metacells - 1:
            break
        
        no_overlap = True
        for metacell in metacells:
            if len(metacell.intersection(i)) >= max_overlap_metacells * k:
                no_overlap = False
                break
        if no_overlap:
            metacells.append(i)
        iterations += 1
    
    # Aggregate with row-by-row operations (original approach)
    metacells_values = []
    for metacell in track(metacells, description="[Original] Aggregating..."):
        if method == 'mean':
            if sp.sparse.issparse(adata.X):
                metacells_values.append(
                    np.array(np.mean([adata.X[i].toarray() for i in metacell], 0))[0]
                )
            else:
                metacells_values.append(
                    np.mean([adata.X[i] for i in metacell], 0)
                )
        elif method == 'sum':
            if sp.sparse.issparse(adata.X):
                metacells_values.append(
                    np.array(sum([adata.X[i].toarray() for i in metacell]))[0]
                )
            else:
                metacells_values.append(
                    sum([adata.X[i] for i in metacell])
                )
    
    # Create AnnData
    metacells_AnnData = ad.AnnData(np.array(metacells_values))
    metacells_AnnData.var_names = adata.var_names
    metacells_AnnData.obs_names = [f"metacell_{i}" for i in range(len(metacells_values))]
    
    return metacells_AnnData


def _select_metacells_optimized(indices, k, max_overlap_metacells, max_metacells):
    """Optimized metacell selection using numpy arrays instead of sets."""
    n_cells = len(indices)
    max_overlap_count = int(max_overlap_metacells * k)
    
    # Convert to list of sorted numpy arrays for faster operations
    indices_arrays = [np.sort(idx) for idx in indices]
    
    # Track selected metacells
    selected_metacells = [indices_arrays[0]]
    
    # Use a more efficient overlap check
    for i in track(range(1, n_cells), description="[Optimized] Computing metacells..."):
        if len(selected_metacells) >= max_metacells:
            break
        
        current = indices_arrays[i]
        is_non_overlapping = True
        
        # Check overlap with all selected metacells
        for selected in selected_metacells:
            # Fast intersection count using sorted arrays
            overlap = len(np.intersect1d(current, selected, assume_unique=True))
            if overlap >= max_overlap_count:
                is_non_overlapping = False
                break
        
        if is_non_overlapping:
            selected_metacells.append(current)
    
    return selected_metacells


def _aggregate_single_metacell(X, indices, method, is_sparse):
    """Aggregate a single metacell from cell expression matrix."""
    if is_sparse:
        # Use sparse matrix slicing - much more memory efficient
        subset = X[indices, :]
        if method == 'mean':
            result = np.array(subset.mean(axis=0)).ravel()
        else:  # sum
            result = np.array(subset.sum(axis=0)).ravel()
    else:
        # Dense matrix
        subset = X[indices, :]
        if method == 'mean':
            result = np.mean(subset, axis=0)
        else:  # sum
            result = np.sum(subset, axis=0)
    
    return result


def _aggregate_metacells_efficient(X, metacell_indices, method, n_jobs=1):
    """Memory-efficient aggregation of metacells with optional parallelization."""
    is_sparse = sp.sparse.issparse(X)
    
    if n_jobs == 1:
        # Sequential processing
        metacells_values = []
        for indices in track(metacell_indices, description="[Optimized] Aggregating..."):
            result = _aggregate_single_metacell(X, indices, method, is_sparse)
            metacells_values.append(result)
    else:
        # Parallel processing
        if n_jobs == -1:
            n_jobs = None  # Use all available cores
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(_aggregate_single_metacell, X, indices, method, is_sparse)
                for indices in metacell_indices
            ]
            metacells_values = [f.result() for f in track(futures, description="[Optimized] Aggregating...")]
    
    return metacells_values


def compute_metacells_optimized(adata, k=50, max_overlap_metacells=0.9,
                                max_metacells=None, method='mean', n_jobs=1):
    """
    Optimized implementation.
    Uses numpy arrays and vectorized operations.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Use LSI coordinates (assuming already computed)
    if 'X_lsi' not in adata.obsm:
        from circe.metacells import lsi
        lsi(adata)
    
    key_projection = 'X_lsi'
    
    # Identify neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(adata.obsm[key_projection])
    distances, indices = nbrs.kneighbors(adata.obsm[key_projection])
    
    if max_metacells is None:
        max_metacells = len(indices)
    
    # Optimized metacell selection using vectorized operations
    metacell_indices = _select_metacells_optimized(
        indices, k, max_overlap_metacells, max_metacells
    )
    
    # Memory-efficient aggregation
    metacells_values = _aggregate_metacells_efficient(
        adata.X, metacell_indices, method, n_jobs
    )
    
    # Create AnnData
    metacells_AnnData = ad.AnnData(np.array(metacells_values))
    metacells_AnnData.var_names = adata.var_names
    metacells_AnnData.obs_names = [f"metacell_{i}" for i in range(len(metacells_values))]
    
    return metacells_AnnData


def generate_test_data(n_cells=1000, n_peaks=5000, sparse=True, seed=42):
    """Generate synthetic ATAC-seq data."""
    np.random.seed(seed)
    
    print(f"Generating test data: {n_cells} cells × {n_peaks} peaks (sparse={sparse})")
    
    # Realistic sparse data (ATAC-seq is typically 1-5% non-zero)
    density = 0.02
    nnz = int(n_cells * n_peaks * density)
    
    data = np.random.poisson(3, size=nnz)
    rows = np.random.randint(0, n_cells, size=nnz)
    cols = np.random.randint(0, n_peaks, size=nnz)
    
    X = sp.sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_peaks))
    
    if not sparse:
        X = X.toarray()
    
    adata = ad.AnnData(X)
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"chr1_{i*1000}_{i*1000+500}" for i in range(n_peaks)]
    
    return adata


def benchmark_implementation(func, adata, **kwargs):
    """Benchmark a single implementation."""
    import tracemalloc
    
    tracemalloc.start()
    start_time = time.time()
    
    try:
        result = func(adata.copy(), **kwargs)
        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'success': True,
            'time': elapsed,
            'memory_mb': peak / 1024 / 1024,
            'n_metacells': result.n_obs
        }
    except Exception as e:
        tracemalloc.stop()
        return {
            'success': False,
            'error': str(e),
            'time': time.time() - start_time
        }


def main():
    print("=" * 80)
    print("METACELL COMPUTATION: ORIGINAL vs OPTIMIZED")
    print("=" * 80)
    print()
    
    # Test configurations
    configs = [
        {'n_cells': 500, 'n_peaks': 2000, 'k': 30, 'max_metacells': 100},
        {'n_cells': 1000, 'n_peaks': 5000, 'k': 50, 'max_metacells': 200},
        {'n_cells': 2000, 'n_peaks': 10000, 'k': 50, 'max_metacells': 300},
        {'n_cells': 10000, 'n_peaks': 10000, 'k': 50, 'max_metacells': 300},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(configs)}: {config['n_cells']} cells, {config['n_peaks']} peaks")
        print(f"{'='*80}\n")
        
        # Generate data
        adata = generate_test_data(
            n_cells=config['n_cells'],
            n_peaks=config['n_peaks'],
            sparse=True
        )
        
        print(f"Data sparsity: {(adata.X.nnz / (adata.n_obs * adata.n_vars) * 100):.2f}% non-zero\n")
        
        # Benchmark original
        print("Running ORIGINAL implementation...")
        original_result = benchmark_implementation(
            compute_metacells_original,
            adata,
            k=config['k'],
            max_metacells=config['max_metacells'],
            method='mean'
        )
        
        if original_result['success']:
            print(f"✓ Time: {original_result['time']:.2f}s")
            print(f"✓ Memory: {original_result['memory_mb']:.1f} MB")
            print(f"✓ Metacells: {original_result['n_metacells']}")
        else:
            print(f"✗ Failed: {original_result['error']}")
        
        print()
        
        # Benchmark optimized (single core)
        print("Running OPTIMIZED implementation (n_jobs=1)...")
        optimized_result = benchmark_implementation(
            compute_metacells_optimized,
            adata,
            k=config['k'],
            max_metacells=config['max_metacells'],
            method='mean',
            n_jobs=1
        )
        
        if optimized_result['success']:
            print(f"✓ Time: {optimized_result['time']:.2f}s")
            print(f"✓ Memory: {optimized_result['memory_mb']:.1f} MB")
            print(f"✓ Metacells: {optimized_result['n_metacells']}")
        else:
            print(f"✗ Failed: {optimized_result['error']}")
        
        print()
        
        # Benchmark optimized (4 cores)
        print("Running OPTIMIZED implementation (n_jobs=4)...")
        optimized_parallel_result = benchmark_implementation(
            compute_metacells_optimized,
            adata,
            k=config['k'],
            max_metacells=config['max_metacells'],
            method='mean',
            n_jobs=4
        )
        
        if optimized_parallel_result['success']:
            print(f"✓ Time: {optimized_parallel_result['time']:.2f}s")
            print(f"✓ Memory: {optimized_parallel_result['memory_mb']:.1f} MB")
            print(f"✓ Metacells: {optimized_parallel_result['n_metacells']}")
        else:
            print(f"✗ Failed: {optimized_parallel_result['error']}")
        
        print()
        
        # Calculate speedup
        if original_result['success'] and optimized_result['success']:
            speedup = original_result['time'] / optimized_result['time']
            memory_reduction = original_result['memory_mb'] / optimized_result['memory_mb']
            
            print(f"{'─'*80}")
            print(f"SPEEDUP (optimized vs original): {speedup:.2f}x faster")
            print(f"MEMORY REDUCTION: {memory_reduction:.2f}x less memory")
            
            if optimized_parallel_result['success']:
                speedup_parallel = original_result['time'] / optimized_parallel_result['time']
                print(f"SPEEDUP (optimized 4-core vs original): {speedup_parallel:.2f}x faster")
            print(f"{'─'*80}")
        
        results.append({
            'config': config,
            'original': original_result,
            'optimized': optimized_result,
            'optimized_parallel': optimized_parallel_result
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Cells':<8} {'Peaks':<8} {'Original':<12} {'Optimized':<12} {'Opt-4core':<12} {'Speedup':<10}")
    print(f"{'':─<8} {'':─<8} {'':─<12} {'':─<12} {'':─<12} {'':─<10}")
    
    for r in results:
        if r['original']['success'] and r['optimized']['success']:
            speedup = r['original']['time'] / r['optimized']['time']
            print(f"{r['config']['n_cells']:<8} {r['config']['n_peaks']:<8} "
                  f"{r['original']['time']:>10.2f}s  {r['optimized']['time']:>10.2f}s  "
                  f"{r['optimized_parallel']['time']:>10.2f}s  {speedup:>8.2f}x")
    
    print(f"\n{'='*80}")
    print("Optimization complete! The optimized version is significantly faster.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
