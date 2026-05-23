"""
Step-level timing of compute_metacells at 100k x 100k.

Runs the full pipeline once with strategy='spatial', printing the time spent in
each phase (LSI/TF-IDF, nearest neighbours, overlap filtering, aggregation).
The phases are timed in-line by recreating compute_metacells' logic here so we
can instrument it without changing the library code.

Usage:
    python profile_metacells_steps.py
"""

from pathlib import Path
import time

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from circe.metacells import (
    tfidf, _FILTERING_STRATEGIES, _select_metacells_community,
)


SCRIPT_DIR = Path(__file__).resolve().parent

N_CELLS = 100_000
N_FEATURES = 75_000
DENSITY = 0.01
N_CLUSTERS = 10
K = 50
MAX_OVERLAP = 0.9
MODES = ("non_overlap", "smoothing", "community")
COMMUNITY_MAX_METACELLS = 10_000
NON_OVERLAP_STRATEGY = "spatial"
NN_ALGORITHM = "auto"
NN_JOBS = -1
SEED = 42


def create_test_data(n_cells, n_features, density, n_clusters, seed):
    """Block-structured sparse synthetic ATAC-like matrix."""
    print(f"\nGenerating data: {n_cells} cells x {n_features} features, "
          f"{n_clusters} clusters, density {density:.2%}")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    cell_cluster = rng.integers(0, n_clusters, size=n_cells)
    feature_bounds = np.linspace(0, n_features, n_clusters + 1, dtype=int)
    target_nnz_per_cluster = int(
        round(density * n_cells * n_features / n_clusters))

    rows, cols = [], []
    for cluster in range(n_clusters):
        cells = np.where(cell_cluster == cluster)[0]
        f0, f1 = feature_bounds[cluster], feature_bounds[cluster + 1]
        n_block_cells, n_block_features = len(cells), f1 - f0
        block_size = n_block_cells * n_block_features
        if block_size == 0:
            continue
        n_nz = min(block_size, target_nnz_per_cluster)
        picked = rng.choice(block_size, size=n_nz, replace=False)
        rows.append(cells[picked // n_block_features])
        cols.append(f0 + picked % n_block_features)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.coo_matrix(
        (data, (rows, cols)), shape=(n_cells, n_features),
        dtype=np.float32).tocsr()

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"peak_{i}" for i in range(n_features)]

    while True:
        shape_before = adata.shape
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
        if adata.shape == shape_before:
            break

    elapsed = time.perf_counter() - t0
    sparsity = adata.X.nnz / (adata.n_obs * adata.n_vars) * 100
    print(f"Generated in {elapsed:.1f}s | shape {adata.shape} | "
          f"{sparsity:.2f}% non-zero")
    return adata


def run_shared_steps(adata, k, nn_algorithm, nn_jobs):
    """LSI + nearest neighbours: identical across all three modes."""
    timings = {}
    use_hv = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_hv else adata

    t0 = time.perf_counter()
    X = tfidf(adata_use.X)
    timings["  lsi.tfidf"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_norm = normalize(X, norm="l1")
    timings["  lsi.l1_normalize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_norm = np.log1p(X_norm * 1e4)
    timings["  lsi.log1p_scale"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_lsi = randomized_svd(X_norm, 20, random_state=0)[0]
    timings["  lsi.randomized_svd"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    timings["  lsi.center_scale"] = time.perf_counter() - t0
    adata.obsm["X_lsi"] = X_lsi

    t0 = time.perf_counter()
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm=nn_algorithm, n_jobs=nn_jobs
    ).fit(adata.obsm["X_lsi"])
    _, indices = nbrs.kneighbors(adata.obsm["X_lsi"])
    timings["nearest_neighbors"] = time.perf_counter() - t0

    return timings, indices


def run_mode_steps(mode, indices, X, k, max_overlap, n_obs,
                   strategy, community_max):
    """Selection + aggregation for one mode, with sub-timings."""
    timings = {}

    if mode == "community":
        t0 = time.perf_counter()
        members_per_metacell = _select_metacells_community(
            indices, k, community_max, n_obs)
        timings["selection"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_metacells = len(members_per_metacell)
        sizes = np.fromiter(
            (len(m) for m in members_per_metacell),
            dtype=np.int64, count=n_metacells)
        cols = np.concatenate(members_per_metacell)
        rows = np.repeat(np.arange(n_metacells), sizes)
        weights = np.repeat(1.0 / sizes, sizes)
        timings["  agg.weights_indices"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        if mode == "smoothing":
            kept = list(range(n_obs))
        else:  # non_overlap
            kept = _FILTERING_STRATEGIES[strategy](
                indices, k, max_overlap * k, n_obs, n_obs)
        timings["selection"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_metacells = len(kept)
        cols = indices[kept].ravel()
        rows = np.repeat(np.arange(n_metacells), k)
        weights = np.full(n_metacells * k, 1.0 / k)
        timings["  agg.weights_indices"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    assignment = sp.csr_matrix(
        (weights, (rows, cols)), shape=(n_metacells, n_obs))
    timings["  agg.csr_construct"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = assignment @ X
    timings["  agg.sparse_matmul"] = time.perf_counter() - t0

    return timings, n_metacells


def _print_timings(label, timings, n_metacells):
    total = sum(timings.values())
    header = f"\n[{label}]"
    if n_metacells is not None:
        header += f" kept {n_metacells} metacells"
    header += f" | total {total:8.3f}s"
    print(header)
    print("-" * 56)
    for name, value in timings.items():
        share = 100.0 * value / total if total else 0.0
        print(f"  {name:<24s} {value:8.3f}s  {share:5.1f}%")


def main():
    print("=" * 72)
    print("CIRCE metacells: step-level profile across modes (LSI+NN shared)")
    print("=" * 72)
    print(f"k={K} | max_overlap={MAX_OVERLAP} | nn_algorithm={NN_ALGORITHM} | "
          f"nn_jobs={NN_JOBS}")
    print(f"modes={MODES} | non_overlap strategy={NON_OVERLAP_STRATEGY} | "
          f"community max_metacells={COMMUNITY_MAX_METACELLS}")

    adata = create_test_data(
        N_CELLS, N_FEATURES, DENSITY, N_CLUSTERS, SEED)

    print("\nRunning shared steps (LSI + nearest neighbours)...")
    shared, indices = run_shared_steps(adata, K, NN_ALGORITHM, NN_JOBS)
    _print_timings("shared (LSI + NN)", shared, n_metacells=None)

    n_obs = adata.n_obs
    X = adata.X
    per_mode = {}
    for mode in MODES:
        print(f"\nRunning mode='{mode}'...")
        timings, n_metacells = run_mode_steps(
            mode, indices, X, K, MAX_OVERLAP, n_obs,
            NON_OVERLAP_STRATEGY, COMMUNITY_MAX_METACELLS)
        _print_timings(f"mode={mode}", timings, n_metacells)
        per_mode[mode] = (timings, n_metacells)

    shared_total = sum(shared.values())
    print("\n" + "=" * 72)
    print("TOTAL PER MODE (shared LSI/NN + mode-specific selection/aggregation)")
    print("=" * 72)
    print(f"{'mode':<14s} {'n_metacells':>12s} {'shared':>10s} "
          f"{'mode-only':>11s} {'total':>10s}")
    for mode in MODES:
        timings, n_metacells = per_mode[mode]
        mode_total = sum(timings.values())
        print(f"{mode:<14s} {n_metacells:>12d} "
              f"{shared_total:>9.2f}s {mode_total:>10.2f}s "
              f"{shared_total + mode_total:>9.2f}s")


if __name__ == "__main__":
    main()
