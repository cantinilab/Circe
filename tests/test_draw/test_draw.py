"""Tests for circe/draw.py uncovered paths."""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy as sp
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

import circe.draw as draw
from circe.utils import add_region_infos


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_genes_df(chromosome="chr1", start=100_000, end=200_000):
    """Minimal genes DataFrame matching the expected schema."""
    return pd.DataFrame({
        "chromosome": [chromosome],
        "start": [start],
        "end": [end],
        "genename": ["GENE1"],
        "strand": ["+"],
    })


def _make_connections_df(chromosome="chr1"):
    """Minimal connections DataFrame matching default column names."""
    return pd.DataFrame({
        "Peak1": [f"{chromosome}_100000_100050"],
        "Peak2": [f"{chromosome}_200000_200050"],
        "score": [0.8],
    })


def _make_atac_with_network():
    """AnnData with region infos, network in varp, and CCAN labels."""
    n_cells, n_regions = 10, 6
    data = np.random.randint(1, 10, size=(n_cells, n_regions))
    var_names = [f"chr1_{i*20000}_{i*20000+50}" for i in range(n_regions)]
    adata = ad.AnnData(pd.DataFrame(data, columns=var_names))
    adata = add_region_infos(adata)

    # Add a sparse network with off-diagonal connections so triu(k=1) is non-empty
    mat = sp.sparse.lil_matrix((n_regions, n_regions))
    mat[0, 2] = 0.5
    mat[1, 3] = 0.7
    mat[2, 4] = 0.6
    adata.varp["atac_network"] = mat.tocsr()

    # Add CCAN column – first 3 regions in module "M1"
    adata.var["CCAN"] = ["M1"] * 3 + ["M2"] * 3
    return adata


# ---------------------------------------------------------------------------
# plot_genes
# ---------------------------------------------------------------------------

class TestPlotGenes:
    def test_missing_required_column_raises(self):
        """Missing a required column raises ValueError (line 61)."""
        genes = _make_genes_df().drop(columns=["strand"])
        with pytest.raises(ValueError, match="strand"):
            draw.plot_genes(genes, "chr1", 0, 300_000)

    def test_no_genes_on_chromosome_raises(self):
        """No genes on the requested chromosome raises ValueError (line 66)."""
        genes = _make_genes_df(chromosome="chr1")
        with pytest.raises(ValueError):
            draw.plot_genes(genes, "chr99", 0, 300_000)

    def test_no_genes_in_region_raises(self):
        """Chromosome matches but genes are outside the region (line 78)."""
        genes = _make_genes_df(chromosome="chr1", start=500_000, end=600_000)
        with pytest.raises(ValueError):
            draw.plot_genes(genes, "chr1", 0, 100_000)

    def test_ax_none_creates_figure(self):
        """When ax=None, plot_genes creates its own figure (line 88)."""
        genes = _make_genes_df()
        # should not raise; a figure is created internally
        draw.plot_genes(genes, "chr1", 0, 300_000, ax=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_connections
# ---------------------------------------------------------------------------

class TestPlotConnections:
    def test_anndata_with_explicit_col_names_warns(self):
        """AnnData input + explicit peak1_col triggers UserWarning (line 225)."""
        adata = _make_atac_with_network()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            draw.plot_connections(
                adata, "chr1", 0, 200_000,
                peak1_col="Peak1",   # should trigger warning
                sep=("_", "_"),
                abs_threshold=0.0,
            )
        assert any("ignored" in str(warning.message).lower() for warning in w)
        plt.close("all")

    def test_dataframe_with_varp_warns(self):
        """DataFrame input + varp parameter triggers UserWarning (line 242)."""
        df = _make_connections_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            draw.plot_connections(
                df, "chr1", 50_000, 300_000,
                varp="atac_network",  # should trigger warning
                sep=("_", "_"),
                abs_threshold=0.0,
            )
        assert any("varp" in str(warning.message).lower() for warning in w)
        plt.close("all")

    def test_invalid_input_type_raises(self):
        """Non-AnnData, non-DataFrame input raises ValueError (line 254)."""
        with pytest.raises(ValueError, match="AnnData"):
            draw.plot_connections("not_valid", "chr1", 0, 300_000)


# ---------------------------------------------------------------------------
# plot_ccan
# ---------------------------------------------------------------------------

class TestPlotCcan:
    def test_missing_ccan_col_raises(self):
        """Missing CCAN column in adata.var raises ValueError (line 565-566)."""
        adata = _make_atac_with_network()
        adata.var.drop(columns=["CCAN"], inplace=True)
        with pytest.raises(ValueError, match="CCAN"):
            draw.plot_ccan(adata, ccan_module="M1")

    def test_unknown_ccan_module_raises(self):
        """Module not in adata.var CCAN column raises ValueError (line 571-572)."""
        adata = _make_atac_with_network()
        with pytest.raises(ValueError, match="No regions"):
            draw.plot_ccan(adata, ccan_module="NONEXISTENT")

    def test_multichr_ccan_raises(self):
        """CCAN module spanning multiple chromosomes raises ValueError (line 580-581)."""
        # Build an adata where module "MULTI" spans chr1 and chr2
        n_cells = 10
        var_names_chr1 = [f"chr1_{i*20000}_{i*20000+50}" for i in range(3)]
        var_names_chr2 = [f"chr2_{i*20000}_{i*20000+50}" for i in range(3)]
        all_names = var_names_chr1 + var_names_chr2
        data = np.random.randint(1, 10, size=(n_cells, len(all_names)))
        adata = ad.AnnData(pd.DataFrame(data, columns=all_names))
        adata = add_region_infos(adata)
        n = len(all_names)
        adata.varp["atac_network"] = sp.sparse.eye(n, format="csr") * 0.5
        adata.var["CCAN"] = ["MULTI"] * n  # all regions in one cross-chr module
        with pytest.raises(ValueError, match="chromosomes"):
            draw.plot_ccan(adata, ccan_module="MULTI")
