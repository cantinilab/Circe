from pybiomart import Server
import pandas as pd


def download_genes(
    server="http://www.ensembl.org",
    species="hsapiens_gene_ensembl"
):
    """
    Download gene coordinates from Ensembl Biomart.

    Parameters
    ----------
    server : str
        URL of the Biomart server.
    species : str
        Species to download gene coordinates for.
        Default is "hsapiens_gene_ensembl".

    Returns
    -------
    genes_df : pandas.DataFrame
        DataFrame with gene coordinates.
    """

    server = Server(host=server)
    dataset = server['ENSEMBL_MART_ENSEMBL'][species]

    # Retrieve gene coordinates
    genes_df = dataset.query(attributes=[
        'chromosome_name',
        'start_position',
        'end_position',
        'strand',
        'external_gene_name'])

    # Convert to Pandas DataFrame
    genes_df = pd.DataFrame(genes_df)
    genes_df.rename(columns={
        "Chromosome/scaffold name": "chromosome",
        "Gene start (bp)": "start",
        "Gene end (bp)": "end",
        "Strand": "strand",
        "Gene name": "genename"
    }, inplace=True)

    genes_df["chromosome"] = "chr" + genes_df["chromosome"]
    genes_df["strand"] = genes_df["strand"].replace({1: "+", -1: "-"})
    return genes_df
