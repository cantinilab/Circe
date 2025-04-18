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
This repo contains a python package for inferring **co-accessibility networks from single-cell ATAC-seq data**, using [skggm](https://www.github.com/skggm/skggm) for the graphical lasso and [scanpy](https://www.github.com/theislab/scanpy) for data processing.

It is based on the pipeline and hypotheses presented in the manuscript "Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data" by Pliner et al. (2018). This R package [Cicero](https://cole-trapnell-lab.github.io/cicero-release/) is available [here](https://www.github.com/cole-trapnell-lab/cicero-release).
## Installation
The package can be installed using pip:
```
pip install circe-py
```
 and from github
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

## Comparison to Cicero R package
<br> Metacalls computation might create differences, but scores will be identical applied to the same metacalls (cf comparison plots below). It should run significantly faster than Cicero _(e.g.: running time of 5 sec instead of 17 min for the dataset 2)_.

_If you have any suggestion, don't hesitate ! This package is still a work in progress :)_
<br> *On the same metacells obtained from Cicero code.*

All tests can be found in the [circe benchmark repo](https://github.com/cantinilab/Circe_reproducibility)

### Real dataset 2 - subsample of 10x PBMC (2021)
- Pearson correlation coefficient: 0.999958
- Spearman correlation coefficient: 0.999911
<img src="https://github.com/cantinilab/circe/raw/main/Figures/correlation_real_dataset2.png" align="center" width="480"/>

Performance on real dataset 2:
- Runtime: ~100x faster
- Memory usage: ~5x less
<img src="https://github.com/cantinilab/circe/raw/main/Figures/perf_real_dataset2.png" align="center" width="480"/>

### Coming:

- ~~_**Calculate metacells !**_~~
- ~~_Add stats on similarity on large datasets._~~
- ~~_Add stats on runtime, memory usage._~~
- _Implement the multithreading use. Should speed up even more._
- ~~_Fix seed for reproducibility._~~

## Usage
It is currently developped to work with AnnData objects. Check Example1.ipynb for a simple usage example.

## Citation
Trimbour Rémi (2025). Circe: Co-accessibility network from ATAC-seq data in python (based on Cicero package). Package version 0.3.6.

