# atac-networks


## Description
This repo contains a python package for inferring co-accessibility networks from single-cell ATAC-seq data, using [skggm](https://www.github.com/skggm/skggm) for the graphical lasso and [scanpy](https://www.github.com/theislab/scanpy) for data processing.

It is based on the pipeline and hypotheses presented in the manuscript "Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data" by Pliner et al. (2018). The original R package [Cicero](https://cole-trapnell-lab.github.io/cicero-release/) is available [here](https://www.github.com/cole-trapnell-lab/cicero-release).

Results may vary between both packages, notably due to the different implementations of graphical lasso.

## Comparison to Cicero R package
<img src="Figures/correlation_toy_dataset1.jpg" align="center" width="1000"/>
_Add stats on similarity on large datasets._
_Add stats on runtime, memory usage.
This package can be run on multiple cores.

## Installation
The package can be installed using pip **(in editable mode -e)**:

```
pip install -e "git+https://github.com/r-trimbour/atac-networks.git#egg=atac-networks"
```

 or by installing it from a local clone
```
git clone https://github.com/r-trimbour/atac-networks.git
pip install -e "atac-networks"
```

## Usage
It is currently developped to work with AnnData objects. Check Example1.ipynb for a simple usage example.

