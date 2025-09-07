How does CIRCE work?
===============

Intuition
---------
Accessible regulatory regions that tend to be co-accessible across cells are candidates for
cis-regulatory interactions. CIRCE estimates these relationships using a graphical lasso model
that **penalizes long-range pairs** according to genomic distance and computes within a local
**sliding window** (e.g., ±500 kb) to keep problems tractable.

Main steps
----------
1. **Dimensionality & neighbors**: compute LSI; optionally use UMAP/t-SNE only for visualization.
   Nearest neighbors are determined in LSI space to avoid distortions from low-dim embeddings.
2. **Metacells (optional)**: group similar cells to reduce sparsity and scale (trade-offs discussed below).
3. **Distance-aware graphical lasso**: within each genomic window of peaks, estimate a sparse
   inverse covariance with a **power-law distance penalty**. The penalty strength is chosen by
   sampling windows so that long-range pairs remain sparse.
4. **Networks (CCANs)**: transform co-accessibility to a graph and call modules with Louvain.

Implementation notes
--------------------
- Uses a QUIC-based graphical model implementation that supports pairwise penalties.
- Stores results in the input ``AnnData``; co-accessibility is kept as a sparse matrix.
- Designed for scverse compatibility and parallel CPU execution.

Practical defaults
------------------
- Window size: ~500 kb for human/mouse cis interactions.
- Distance-decay exponent: species-specific (e.g., ~0.75 human/mouse; ~0.85 Drosophila).
- Penalty scaling: chosen by random-window search to ensure sparsity of long-range edges.
