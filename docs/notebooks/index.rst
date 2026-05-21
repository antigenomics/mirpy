Notebook Gallery
================

Rendered walkthroughs for the key mirpy workflows. All notebooks are included
as-is with pre-computed outputs; they are not re-executed during the docs build.

Parsing And Repertoire Basics
-----------------------------

.. toctree::
   :maxdepth: 1

   parsing_example
   sample_repertoire_overview
   gene_usage_correction
   diversity_analysis

**parsing_example** — Parse VDJdb, AIRR, and VDJtools files into repertoire objects.

**sample_repertoire_overview** — Load a multi-locus SRA cohort and inspect coverage statistics.

**gene_usage_correction** — Batch-correct V/J gene usage across donors and derive PCA/UMAP embeddings.

**diversity_analysis** — Reproduce diversity tables, rarefaction, coverage, Hill curves, and MS vs healthy cohort comparisons.

Graph And Sequence Analysis
---------------------------

.. toctree::
   :maxdepth: 1

   gene_similarity
   token_graph
   edit_distance_graph
   vdjdb_junction_graph

**gene_similarity** — Compare germline V gene amino-acid sequences via the GermlineAligner API.

**token_graph** — Build bipartite k-mer/clonotype graphs and filter by sequence motifs.

**edit_distance_graph** — Construct Hamming/Levenshtein edit-distance graphs from junction sequences.

**vdjdb_junction_graph** — Analyse multi-epitope junction Hamming graphs from VDJdb.

Biomarker Detection
-------------------

.. toctree::
   :maxdepth: 1

   alice_analysis
   tcrnet_analysis
   gliph_analysis
   vdjbet_yf

**alice_analysis** — ALICE antigen-expanded clone detection on YF, AS, and MLR datasets.

**tcrnet_analysis** — TCRNET enrichment for CMV+ vs B35+ donors with VDJdb annotation.

**gliph_analysis** — GLIPH-style multi-family token enrichment and clonotype graph clustering.

**vdjbet_yf** — VDJBet disease-associated overlap analysis on YFV vaccine time-series samples.

Aging And Cohort Overlap
------------------------

.. toctree::
   :maxdepth: 1

   aging_analysis

**aging_analysis** — Donor-vs-pool overlap trends across aging, using the shared-worker many-vs-pool overlap path for faster repeated scoring.

Pgen And Selection Analysis
---------------------------

.. toctree::
   :maxdepth: 1

   pgen_analysis

**pgen_analysis** — Benchmark OLGA exact, Monte-Carlo, and hash-enumeration Pgen strategies; analyse the Q-factor (thymic selection correction) on VDJdb clonotypes.

TCREmp Embeddings
-----------------

.. toctree::
   :maxdepth: 1

   tcremp_vdjdb_analysis
   tcremp_vdjdb_analysis_paired
   tcremp_10xdcode_analysis

**tcremp_vdjdb_analysis** — Single-chain TCREmp on VDJdb epitope-labelled clonotypes with UMAP projection.

**tcremp_vdjdb_analysis_paired** — Paired TRA/TRB TCREmp on VDJdb full records with imputed missing chains.

**tcremp_10xdcode_analysis** — 10x CITE-seq + TCREmp embedding with DBSCAN epitope clustering.

Single-Cell Analysis
--------------------

.. toctree::
   :maxdepth: 1

   single_cell_load
   single_cell_pairing_analysis

**single_cell_load** — Load 10x VDJ v1 donors, inspect chain multiplicity, and benchmark against scirpy.

**single_cell_pairing_analysis** — Build and compare pairing graphs across raw, imputed, and cleaned stages.
