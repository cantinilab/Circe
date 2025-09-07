Overview and algorithm
===================================

What is CIRCE?
--------------
CIRCE predicts **cis-regulatory interactions** (co-accessibility links between peaks) from
single-cell ATAC-seq. It follows the algorithm of Cicero [1] but is implemented natively in Python,
integrates with the scverse/AnnData stack, and is engineered for speed and scale.

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

Practical defaults
------------------
- **Window size**: ~500 kb (human/mouse) is a good starting point.
- **Penalty schedule**: distance‑increasing (often close to a power‑law); a short calibration on
  sampled windows chooses :math:`\lambda_0` to keep **long‑range edges sparse**.
- **Neighbors**: build in LSI space; UMAP/t‑SNE is only for visualization.
- **Matrix format**: prefer **CSC** for very large atlases (fast column slicing in windows).

Output at a glance
------------------
- **Links**: sparse symmetric matrix in ``adata.varp[<key>]``; each nonzero is a peak‑pair score.
- **Networks**: CCAN membership and per‑module summaries (size, density, representative peaks).
- **Tables**: exportable dataframe with (Peak1, Peak2, score, distance, chromosome, window_id).

Notes & limitations
-------------------
- Co‑accessibility is **not causality**; validate with orthogonal assays (e.g., PC‑HiC).
- Metacells reduce runtime/sparsity but can slightly degrade validation; single‑cell inputs
  performed best in our benchmarks.
- Long ranges (>500 Kb) are intentionally penalized and very long ranges (> 500Kb) are fully ignored with default parameters; tune if your biology requires larger spans.



Sources
-------------------
[1] Pliner et al., *Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data*, Mol Cell, 2018.