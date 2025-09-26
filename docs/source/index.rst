.. role:: raw-html(raw)
    :format: html

.. circe documentation master file, created by
   sphinx-quickstart on Thu Sep  4 18:21:17 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../../logo.svg
   :alt: CIRCE-logo Remi-Trimbour 2024
   :align: center

CIRCE: Cis-regulatory interactions between chromatin regions
============================================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:
   :caption: CIRCE explained

   circe_explained/*

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:
   :caption: Examples

   examples/*

.. image:: https://github.com/cantinilab/circe/actions/workflows/codecov.yaml/badge.svg
   :target: https://github.com/cantinilab/circe/actions/workflows/codecov.yaml
   :alt: Unit Tests

.. image:: https://github.com/cantinilab/Circe/actions/workflows/wheels.yml/badge.svg
   :target: https://github.com/cantinilab/Circe/actions/workflows/wheels.yml
   :alt: Wheels

.. image:: https://codecov.io/gh/cantinilab/circe/graph/badge.svg?token=0OIFAP28D7
   :target: https://codecov.io/gh/cantinilab/circe
   :alt: codecov

.. image:: https://img.shields.io/pypi/v/circe-py?color=blue
   :target: https://img.shields.io/pypi/v/circe-py
   :alt: PyPI version

.. image:: https://static.pepy.tech/badge/circe-py/month
   :target: https://pepy.tech/project/circe-py
   :alt: Downloads


CIRCE is a Python package for inferring **co-accessibility networks from
single-cell ATAC-seq data**, using `skggm <https://www.github.com/skggm/skggm>`_
for the graphical lasso and `scanpy <https://www.github.com/theislab/scanpy>`_ for data processing.

It is based on the pipeline and hypotheses presented in the manuscript
*Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data*
by Pliner et al. (2018).
The original R package Cicero is available
`here <https://www.github.com/cole-trapnell-lab/cicero-release>`_.

You can check our preprint here for more details! 😊 :raw-html:`<br />`
https://doi.org/10.1101/2025.09.23.678054

.. note::
   In case you encounter any trouble, check out the `CIRCE GitHub repo <https://github.com/cantinilab/Circe>`_.

Installation
------------

The package can be installed using pip::

   pip install circe-py

or directly from GitHub::

   pip install "git+https://github.com/cantinilab/circe.git"


Minimal example
---------------

.. code-block:: python

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


Visualisation
-------------

.. code-block:: python

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

.. image:: https://github.com/cantinilab/circe/raw/main/Figures/circe_figure_genes_access.png
   :align: center


Bencmark & comparison to the Cicero R package
------------------------------
All tests run in the preprint can be found in the  `CIRCE benchmark repo <https://github.com/cantinilab/Circe_reproducibility>`_..

Metacells computation might create differences, but scores will be identical when applied to the same metacalls (cf comparison plots below).
It should run significantly faster than Cicero (e.g., running time of 5 sec instead of 17 min for dataset 2).
*On the same metacells obtained from the Cicero code.*


Real dataset 2 - subsample of 10x PBMC (2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Pearson correlation coefficient: 0.999958
- Spearman correlation coefficient: 0.999911

.. image:: https://github.com/cantinilab/circe/raw/main/Figures/correlation_real_dataset2.png
   :align: center
   :width: 480

Performance on real dataset 2:

- Runtime: ~100x faster
- Memory usage: ~5x less

.. image:: https://github.com/cantinilab/circe/raw/main/Figures/perf_real_dataset2.png
   :align: center
   :width: 480


Coming
------

- Gene activity
- Complete integration in HuMMuS GRN inference pipeline


Citation
--------

Trimbour R., Saez Rodriguez J., Cantini L. (2025). CIRCE: a scalable Python package to predict cis-regulatory DNA interactions from single-cell chromatin accessibility data.
bioRxiv, 2025.09.23.678054, doi: https://doi.org/10.1101/2025.09.23.678054 


.. toctree::
   :maxdepth: 1
   :caption: Contents

   examples/*
