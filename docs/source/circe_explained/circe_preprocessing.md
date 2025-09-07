Preprocessing choices
=================================

Single-cell vs metacells
------------------------
- **Single-cell inputs** typically yield **better validation** versus PC-HiC interactions.
- **Metacells** can reduce compute and sparsity but showed a small AUC drop in the tested defaults.

Binarization & count correction
-------------------------------
- Simple **binarization** of counts did not notably improve predictions over raw counts.
- The traditional **Cicero count correction** decreased performance in benchmarks.

Neighbors space
---------------
- Build neighbors in **LSI space** rather than low-dimensional UMAP/t-SNE to preserve distances.

Matrix format tips
------------------
- For very large atlases, store your peak-by-cell matrix as **CSC** to accelerate column extraction.
