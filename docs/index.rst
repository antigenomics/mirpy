mirpy documentation
===================

.. raw:: html

   <div class="mirpy-intro">
     <img class="mirpy-intro__logo" src="_static/mirpy_logo.png" alt="mirpy logo" />
     <div>
       <p class="mirpy-intro__eyebrow">AIRR-seq toolkit</p>
       <p class="mirpy-intro__lead">
         mirpy is a Python toolkit for immune repertoire analysis. It brings together
         parsers, repertoire abstractions, diversity utilities, and exploratory notebooks
         in one place.
       </p>
       <p class="mirpy-intro__links">
         <a href="getting-started.html">Getting started</a>
         <span>&middot;</span>
         <a href="examples.html">Notebook gallery</a>
         <span>&middot;</span>
         <a href="https://github.com/antigenomics/mirpy">GitHub</a>
       </p>
     </div>
   </div>

   <div class="mirpy-card-grid">
     <a class="mirpy-card" href="getting-started.html">
       <h3>Getting Started</h3>
       <p>Install the package, load segment libraries, parse repertoires, and understand the main concepts.</p>
     </a>
     <a class="mirpy-card" href="examples.html">
       <h3>Notebook Gallery</h3>
       <p>Open every analysis notebook from the repository directly in the docs, without maintaining duplicate examples.</p>
     </a>
     <a class="mirpy-card" href="https://github.com/antigenomics/mirpy">
       <h3>Repository</h3>
       <p>Jump to the GitHub repository for source code, issues, notebooks, and the current development state.</p>
     </a>
   </div>

What mirpy covers
=================

.. raw:: html

   <div class="mirpy-feature-grid">
     <div class="mirpy-feature">
       <h3>Large AIRR-seq datasets</h3>
       <p>Parse AIRR-seq repertoires, clonotype tables, and segment annotations into reusable Python objects for downstream analysis.</p>
     </div>
     <div class="mirpy-feature">
       <h3>Basic repertoire statistics</h3>
       <p>Compute diversity summaries, richness estimates, rarefaction, and counts such as singletons and doubletons.</p>
     </div>
     <div class="mirpy-feature">
       <h3>K-mer analysis</h3>
       <p>Work with sequence k-mers for repertoire comparison, enrichment analysis, and exploratory feature construction.</p>
     </div>
     <div class="mirpy-feature">
       <h3>T-cell marker discovery</h3>
       <p>Search for T-cell biomarkers and enriched clonotype patterns across cohorts and phenotype groups.</p>
     </div>
     <div class="mirpy-feature">
       <h3>Gene usage matrices</h3>
       <p>Build and analyze V/J usage matrices for samples, cohorts, and comparative repertoire studies.</p>
     </div>
     <div class="mirpy-feature">
       <h3>Resampling workflows</h3>
       <p>Resample repertoires at matched depths or adjusted segment usage profiles for controlled comparisons.</p>
     </div>
     <div class="mirpy-feature">
       <h3>Clonotype filtering and clustering</h3>
       <p>Select clonotypes, compare them by distance, and cluster related sequences into analysis-ready groups.</p>
     </div>
     <div class="mirpy-feature">
       <h3>Embeddings</h3>
       <p>Generate repertoire and prototype embeddings that can be used in downstream machine learning and visualization pipelines.</p>
     </div>
   </div>

Quick example
=============

.. code-block:: python

   import gzip
   from pathlib import Path

   from mir.basic.token_tables import Rearrangement, filter_token_table, tokenize_rearrangements
   from mir.graph.token_graph import build_token_graph

   # Load CDR3 sequences and build an RS-filtered k-mer graph
   with gzip.open("gilgfvftl_trb_cdr3.txt.gz", "rt") as fh:
       cdr3s = [line.strip() for line in fh if line.strip()]

   rearrangements = [
       Rearrangement(sequence_id=str(i), locus="TRB", v_gene="TRB", junction_aa=seq, duplicate_count=1)
       for i, seq in enumerate(cdr3s)
   ]

   table    = tokenize_rearrangements(rearrangements, k=3)
   rs_table = filter_token_table(table, kmer_pattern="RS")
   g_rs     = build_token_graph(rearrangements, rs_table)

   # Largest connected component — the RS-bearing rearrangement cluster
   rs_cluster = g_rs.components().giant()
   print(f"RS cluster: {rs_cluster.vcount()} nodes")

Explore next
============

* :doc:`getting-started` for the shortest path from install to first parsed repertoire
* :doc:`examples` for the full notebook gallery published from the repository
* :doc:`modules` for API documentation generated from the current codebase
* `GitHub repository <https://github.com/antigenomics/mirpy>`_ for source browsing and notebooks

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting-started
   examples
   modules
