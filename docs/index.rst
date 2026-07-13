mirpy documentation
===================

mirpy is a Python toolkit for immune repertoire analysis. It brings together
parsers, repertoire abstractions, diversity utilities, and exploratory
notebooks in one place.

:doc:`Getting started <getting-started>` ·
:doc:`Notebook gallery <examples>` ·
:doc:`References <references>` ·
`GitHub <https://github.com/antigenomics/mirpy>`_

What mirpy covers
-----------------

- **Large AIRR-seq datasets** — parse repertoires, clonotype tables, and segment annotations into reusable Python objects.
- **Basic repertoire statistics** — diversity summaries, richness estimates, rarefaction, singletons and doubletons.
- **K-mer analysis** — sequence k-mers for repertoire comparison, enrichment, and feature construction.
- **T-cell marker discovery** — search for biomarkers and enriched clonotype patterns across cohorts.
- **Gene usage matrices** — build and analyze V/J usage matrices for samples and cohorts.
- **Resampling workflows** — resample repertoires at matched depths or adjusted usage profiles.
- **Clonotype filtering and clustering** — select, compare by distance, and cluster related sequences.
- **Embeddings** — repertoire and prototype embeddings for downstream ML and visualization.

Quick example
-------------

.. code-block:: python

   import gzip

   from mir.basic.token_tables import filter_token_table, tokenize_clonotypes
   from mir.common.clonotype import Clonotype
   from mir.graph.token_graph import build_token_graph

   # Load CDR3 sequences and build an RS-filtered token graph
   with gzip.open("gilgfvftl_trb_cdr3.txt.gz", "rt") as fh:
       cdr3s = [line.strip() for line in fh if line.strip()]

   clonotypes = [
       Clonotype(junction_aa=seq, locus="TRB", v_call="TRBV", duplicate_count=1)
       for seq in cdr3s
   ]

   table    = tokenize_clonotypes(clonotypes, k=3)
   rs_table = filter_token_table(table, kmer_pattern="RS")
   g_rs     = build_token_graph(clonotypes, rs_table)

   # Largest connected component — the RS-bearing clonotype cluster
   rs_cluster = g_rs.components().giant()
   print(f"RS cluster: {rs_cluster.vcount()} nodes")

Explore next
------------

* :doc:`getting-started` for the shortest path from install to first parsed repertoire
* :doc:`getting-started` for the Copilot agent and companion prompt (``/mirpy-analysis``)
* :doc:`examples` for the full notebook gallery published from the repository
* :doc:`modules` for API documentation generated from the current codebase
* `GitHub repository <https://github.com/antigenomics/mirpy>`_ for source browsing and notebooks

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting-started
   embedding-your-repertoire
   examples
   modules
   references
