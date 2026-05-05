Notebook Gallery
================

All notebooks from the repository are listed below. They are rendered in the
docs site without execution during the build.

Some notebooks are polished walkthroughs, while a smaller set remain as legacy
exploratory analyses that may require local assets or older research context.

Core Workflows
==============

Essential workflows for getting started with mirpy.

.. toctree::
   :maxdepth: 1

   setup_assets
   parsing_airr
   diversity_estimation
   repertoire_resampling
   gene_similarity
   sample_repertoire_overview

**setup_assets** — Download and cache reference data (TCR/BCR gene libraries, VDJDB).

**parsing_airr** — Parse AIRR-compliant TSV files and construct repertoire objects.

**diversity_estimation** — Calculate alpha and beta diversity metrics.

**repertoire_resampling** — Downsample or resample repertoires by gene usage.

**gene_similarity** — Compare gene usage patterns between repertoires.

**sample_repertoire_overview** — Load and inspect a sample cohort structure.

Graph Analysis
==============

Build and analyze similarity and distance graphs from TCR/BCR sequences.

.. toctree::
   :maxdepth: 1

   token_graph
   edit_distance_graph
   cdr3_aln_benchmark
   cdr3_graph

**token_graph** — Build weighted k-mer co-occurrence networks.

**edit_distance_graph** — Construct edit distance graphs and analyze connectedness.

**cdr3_aln_benchmark** — Benchmark CDR3 sequence alignment performance.

**cdr3_graph** — Analyze CDR3 sequences as graph structures.

Distance, Matching, And Embedding
==================================

Advanced sequence matching and dimensionality reduction techniques.

.. toctree::
   :maxdepth: 1

   prototype_embedding
   proto_embedding_new

**prototype_embedding** — Embed repertoires into continuous space via prototype selection.

**proto_embedding_new** — Alternative prototype embedding approach.

Analysis And Modeling
=====================

Advanced analysis and biomarker detection; these notebooks may require larger local datasets.

.. toctree::
   :maxdepth: 1

   vdjbet_yf

**vdjbet_yf** — Disease-associated overlap analysis using VDJBET stratification.

Legacy Exploratory Notebooks
============================

These notebooks are kept for reference, but parts of their workflow predate the
current public API and may need manual adaptation before rerunning.

.. toctree::
   :maxdepth: 1

   biomarkers_inference
   cluster_associations
   grid_search_clf
   kmer_generator

**biomarkers_inference** — ALICE and TCRNET biomarker detection (legacy APIs).

**cluster_associations** — Find associations between clusters and metadata.

**grid_search_clf** — Hyperparameter optimization for classifiers.

**kmer_generator** — K-mer feature extraction and analysis.
