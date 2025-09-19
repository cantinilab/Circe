Overview and algorithm
===================================

What is CIRCE?
--------------
CIRCE predicts **cis-regulatory interactions** (co-accessibility links between peaks) from
single-cell ATAC-seq. It follows the algorithm of Cicero [1] but is implemented natively in Python,
integrates with the scverse/AnnData stack, and is engineered for speed and scale.

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

The algorithm in 10 lines
-------------------------
#. **Inputs**: a sparse peak × cell matrix ``X`` in ``AnnData`` (``adata.X``) and genomic
   coordinates for peaks in ``adata.var`` (chrom, start, end).
#. **Optional metacells**: group similar cells to reduce sparsity/size (trade-off: slight
   performance drop vs single-cell inputs).
#. **LSI + neighbors**: compute LSI on TF–IDF-like transforms; determine neighbors in **LSI
   space** (not in UMAP/t‑SNE) to avoid metric distortion.
#. **Genomic windowing**: split each chromosome into overlapping windows (e.g., ±500 kb) so
   each optimization is small and locality-aware.
#. **Empirical covariance**: for peaks inside a window, compute a covariance surrogate across
   cells (or metacells); optionally neighbor-aggregate counts to stabilize estimates.
#. **Distance‑aware graphical lasso**: estimate a sparse precision matrix ``Θ`` by minimizing a
   penalized likelihood, where **penalties grow with genomic distance** so long‑range pairs are
   discouraged:

   .. math::

      \min_{\Theta \succ 0} \; \tfrac{1}{2} \mathrm{tr}(S\Theta) - \log\det\Theta
      + \sum_{i<j} \lambda_{ij} |\Theta_{ij}|, \quad
      \lambda_{ij} = \lambda_0 \cdot f(d_{ij}), \; f'(d) \ge 0.

   Here, :math:`S` is the (regularized) covariance and :math:`d_{ij}` is genomic distance.
   CIRCE uses a fast QUIC-like solver that supports pairwise penalties.
#. **Co‑accessibility scores**: convert entries of ``Θ`` (e.g., partial correlations) into
   signed scores for peak pairs within the window.
#. **Merge across windows**: because windows overlap, the same pair may be estimated multiple
   times; keep the **max |score|** or an average within a distance cap.
#. **Export & visualize**: store the sparse link matrix in ``adata.varp`` and provide a tidy
   edge table (Peak1, Peak2, score) for filtering/plots.
#. **CCANs (visualize)**: build a graph with edges weighted by co‑accessibility, then detect modules
   (Louvain/Leiden) → **cis‑co‑accessible modules**.


Implementation notes
--------------------
- Uses a QUIC-based graphical model implementation that supports pairwise penalties.
- Stores results in the input ``AnnData``; co-accessibility is kept as a sparse matrix.
- Designed for scverse compatibility and parallel CPU execution.
- Penalty scaling: chosen by random-window search to ensure sparsity of long-range edges; a short calibration on
  sampled windows chooses :math:`\lambda_0` to keep **long‑range edges sparse**


Notes & limitations
-------------------
- Co‑accessibility is **not causality**; validate with orthogonal assays (e.g., PC‑HiC).
- Metacells reduce runtime/sparsity but can slightly degrade validation; single‑cell inputs
  performed best in our benchmarks.
- Long ranges (>500 Kb) are intentionally penalized and very long ranges (> 500Kb) are fully ignored with default parameters; tune if your biology requires larger spans.


Practical defaults
------------------
**When inferring the co-acceessible networks, default parameters exist for human, mouse and drosophila.** If you are working with another organism, or want to tune parameters, you can however modify the following parameters:
- **Window size**: ~500 kb (human/mouse) is a good starting point.
- **Distance-decay exponent**: species-specific (e.g., ~0.75 human/mouse; ~0.85 Drosophila).
- **Long-range cutoff**: pairs beyond this distance are ignored (e.g., typically, half of the windows size).

Otherwise recommandations are:
**Metacells**: optional, but recommended for large datasets
- **Neighbors**: build in LSI space; UMAP/t‑SNE is only for visualization since it does not preserve nicely distance ([Chari et al., 2023](https://doi.org/10.1371/journal.pcbi.1011288)).
- **Matrix format**: prefer **CSC** for very large atlases (fast column slicing in windows).

Output at a glance
------------------
- **Links**: sparse symmetric matrix in ``adata.varp[<key>]``; each nonzero is a peak‑pair score.
- **Networks**: CCAN membership and per‑module summaries (size, density, representative peaks).
- **Tables**: exportable dataframe with (Peak1, Peak2, score, distance, chromosome, window_id).
- **Visualizations**: built-in plotting functions for module and locus‑specific views, including gene
  annotations if provided.




Sources
-------------------
[1] Pliner et al., *Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data*, Mol Cell, 2018.