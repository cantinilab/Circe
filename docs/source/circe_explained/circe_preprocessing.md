# 🔧 Extended Preprocessing Guidance for CIRCE
Here are some tips and observations from our benchmarks on preprocessing choices for CIRCE ([Trimbour et al., 2025](https://www.biorxiv.org/content/10.1101/2025.09.23.678054v2)).


## 1. Count treatment: raw counts, binarization, count correction

Use **raw counts** (peak-by-cell matrix) by default. This preserves quantitative information about accessibility that seems to help CIRCE’s co-accessibility inference.

**Binarization can be tried** — especially if you want to reduce biases from highly accessible peaks or differences in coverage — but be aware that in the CIRCE context it did not yield benefit.

**Avoid applying Cicero’s “count correction”** (or other heavy total-count normalization) prior to co-accessibility inference with CIRCE, unless you have a very strong reason and you benchmark performance (because in the published report it degraded results).


**Additional nuance / recommendation:**
Before analysis your dataset, consider doing quality filtering and cell QC (remove low-quality cells or potential doublets), which is standard in scATAC-seq preprocessing workflows. Several scATAC best-practice resources recommend filtering based on metrics such as fragment counts, fraction of reads in peaks, TSS enrichment, etc.
[You can check guidelines here](https://www.sc-best-practices.org/chromatin_accessibility/introduction.html)

### Notes on Cicero-Style Preprocessing and Why We Avoid It

Cicero’s recommended preprocessing pipeline historically includes **binarization** and a **normalization step** based on total accessibility counts per cell. Our evaluations, supported by published analyses, show that neither step is beneficial for CIRCE.

#### Binarization
Previous studies demonstrate that binarizing scATAC-seq counts does **not** improve data quality, statistical fit, or downstream biological interpretation.
- **Martens et al., 2023**  
  *“Here we show that binarization is an unnecessary step that neither improves goodness of fit, clustering, cell type identification nor batch integration.”*  
  DOI: https://doi.org/10.1038/s41592-023-02112-6

This aligns with our benchmarks: **binarized counts did not outperform raw counts** for co-accessibility inference.

#### Total-count normalization
Cicero-style correction also divides accessibility by a per-cell total count. However, this assumption — that every cell should have the same total accessibility — is unsupported biologically and methodologically.

- **Kwok et al., 2025**  
  *“Dividing by total count is a sound strategy for bulk sequencing [...]. However, in scATAC-seq data [...] the total count of each cell is different. Therefore, after TF (Ed.: Term Frequency transformation) transformation, the largest variation between cells will naturally be due to their denominators, that is, the total counts per cell or sequencing depth.”*  
  DOI: https://doi.org/10.1186/s13059-025-03735-y


## 2. Input type: single-cell vs metacells / pseudobulk

**Single-cell inputs:** As already noted, single-cell resolution yielded better validation against promoter-capture Hi-C (PC-HiC) interactions in the benchmarks with CIRCE. ([Trimbour et al., 2025](https://www.biorxiv.org/content/10.1101/2025.09.23.678054v2))

**Metacells:** Aggregating cells (into metacells) can reduce computational cost and mitigate sparsity, but you still need to generate enough of them or it may lead to decreased AUC.

**When to use metacells:** 
<br> a. If the dataset is extremely large (many tens/hundreds of thousands of cells) and memory/compute is limiting — or if individual cells are very sparse — metacells may be acceptable. But you should re-evaluate performance (e.g., link recovery, network robustness) compared to a single-cell run.
<br> b. If you need both **negative and positive co-accessibility scores**, and that you observe a **very high proportion of positive score** and very few negative co-accessibility scores. It typically indicate that you have a very high level of sparsity and metacells helps correcting this bias.

#### Metacell algorithms: Dimensionality reduction & neighbor graph construction (“neighbors space”)

To identify neighboring cells (i.e. define cell proximity / similarity), using LSI (latent semantic indexing) space rather than low-dimensional nonlinear embeddings such as UMAP or t-SNE works better to preserve relative distances. 

Rationale: Nonlinear embeddings (UMAP/t-SNE) are optimized for visualization and can distort distances in a way that may not reflect true similarity structure relevant for co-accessibility. LSI (or other linear/distance-preserving dimensionality reduction) tends to be more robust for neighbor-graph building.

Recommendation: Use LSI (or PCA / other linear methods) for constructing the neighbor graph. Avoid using UMAP or t-SNE for that purpose.

Tip / extension: Depending on dataset size and complexity, you might consider exploring different numbers of LSI components (e.g. carry out a small sweep: 50, 100, 200 LSI dims) to see how stable the inferred networks / CCANs are. This can help choose a setting robust to technical noise and overfitting.

## 3. AnnData matrix format choice

CIRCE can handle both sparse and dense matrices. A dense matrix will be **faster to process**, since the graphical lasso model ultimately needs dense matrix chunks.
On large atlases, you should store your peak-by-cell matrix in **CSC format** ([Compressed Sparse Column](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html)) to accelerate column extraction (i.e. per-cell operations) in CIRCE.

## 4. Benchmarking new metacells/preprocessing strategies

Our benchmark was limited to a comparison of Cicero and CIRCE's standard practices.

If you want to test your own preprocessing or metacells strategy, you can have a look at our [benchmark pipeline](https://www.github.com/cantinilab/circe_reproducibility) that we used to compare the different methods. :)

You can then simply add the Snakemake rule corresponding to your own method, and compare it to the other methods and the PC-HiC data considered there as the ground truth.
_Don't hesitate to open a GitHub issue there if you need additional guidance._

## Summary
- Binarization: **no demonstrated benefit** for scATAC or CIRCE.  
- Total-count normalization: **methodologically unsound** for sparse single-cell chromatin data.  
- Recommended: **use raw counts, without Cicero-style correction**.
- Metacells: Only if you need to correct extra-sparse data