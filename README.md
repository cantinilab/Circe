<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cantinilab/circe/main/logo_dark_theme.svg" width="600">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/cantinilab/circe/main/logo.svg" width="600">
    <img alt="Circe logo" src="https://raw.githubusercontent.com/cantinilab/circe/main/logo.svg" width="600">
  </picture>
</p>

-----------------
# CIRCE: Cis-regulatory interactions between chromatin regions
[![Unit_Tests](https://github.com/cantinilab/circe/actions/workflows/codecov.yaml/badge.svg)](https://github.com/cantinilab/circe/actions/workflows/codecov.yaml)
[![Wheels](https://github.com/cantinilab/Circe/actions/workflows/wheels.yml/badge.svg)](https://github.com/cantinilab/Circe/actions/workflows/wheels.yml)
[![codecov](https://codecov.io/gh/cantinilab/circe/graph/badge.svg?token=0OIFAP28D7)](https://codecov.io/gh/cantinilab/circe)
[![PyPI version](https://img.shields.io/pypi/v/circe-py?color=blue)](https://img.shields.io/pypi/v/circe-py)
[![Downloads](https://static.pepy.tech/badge/circe-py/month)](https://pepy.tech/project/circe-py)

## Description
This repository contains a Python package for inferring **co-accessibility networks from single-cell ATAC-seq data**, using [skggm](https://www.github.com/skggm/skggm) for the graphical lasso and [scanpy](https://www.github.com/theislab/scanpy) for data processing.

You can check our preprint here for more details! 😊<br>
https://doi.org/10.1101/2025.09.23.678054

While updating the pre-processing, CIRCE's algorithm is based on the pipeline and hypotheses presented in the manuscript "Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data" by Pliner et al. (2018). This original R package [Cicero](https://cole-trapnell-lab.github.io/cicero-release/) is available [here](https://www.github.com/cole-trapnell-lab/cicero-release).




## Installation
The package can be installed using pip:
```
pip install circe-py
```
 and from GitHub
```
pip install "git+https://github.com/cantinilab/circe.git"
```

## Minimal example
```
import anndata as ad
import circe as ci

# Load the data
atac = ad.read_h5ad('atac_data.h5ad')
atac = ci.add_region_infos(atac)

# Compute the co-accessibility network
ci.compute_atac_network(atac)

# Extract the network and find CCANs modules
circe_network = ci.extract_atac_links(atac)
ccans_module = ci.find_ccans(atac)
```

### Visualisation
```
fig, ax = plt.subplots(1, figsize = (20, 6))
genes_df = ci.downloads.download_genes()

ci.draw.plot_connections_genes(
    connections=atac,  # Main parameters
    genes=genes_df,
    chromosome="chr1",
    start=50_000,
    end=300_000,
    gene_spacing=30_000,
    abs_threshold=0.0,
    y_lim_top=-0.01,   # Visual parameters
    track_spacing=0.01,
    track_width=0.01,
    ax=ax
)
```
<img src="https://github.com/cantinilab/circe/raw/main/Figures/circe_figure_genes_access.png" align="center"/>

## Usage
You can go check out our documentation for more examples! <br> https://circe.readthedocs.io/<br>
The documentation is still in building, so don't hesitate to open any issues or requests you might have in this repo. 😊

## Benchmark & Comparison to Cicero R package
<br> Metacalls computation might create differences, but scores will be identical when applied to the same metacalls (cf comparison plots below). It should run significantly faster than Cicero _(e.g., running time of 5 sec instead of 17 min for the dataset 2)_.
<br> *On the same metacells obtained from the Cicero code.*

All tests run in the preprint can be found in the [circe benchmark repo](https://github.com/cantinilab/Circe_reproducibility).

### Real dataset 2 - subsample of 10x PBMC (2021)
- Pearson correlation coefficient: 0.999958
- Spearman correlation coefficient: 0.999911
<img src="https://github.com/cantinilab/circe/raw/main/Figures/correlation_real_dataset2.png" align="center" width="480"/>

Performance on real dataset 2:
- Runtime: ~100x faster
- Memory usage: ~5x less
<img src="https://github.com/cantinilab/circe/raw/main/Figures/perf_real_dataset2.png" align="center" width="480"/>

### Coming:
- Gene activity calculation

## Citation
>  Trimbour R., Saez Rodriguez J., Cantini L. (2025). CIRCE: a scalable Python package to predict cis-regulatory DNA interactions from single-cell chromatin accessibility data.
bioRxiv, 2025.09.23.678054, doi: https://doi.org/10.1101/2025.09.23.678054 


