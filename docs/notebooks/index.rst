Notebook Gallery
================

All notebooks from the repository are listed below. They are rendered in the
docs site without execution during the build.

Some notebooks are polished walkthroughs, while a smaller set remain as legacy
exploratory analyses that may require local assets or older research context.

Core Workflows
==============

.. toctree::
   :maxdepth: 1

   setup_assets
   parsing_airr
   diversity_estimation
   repertoire_resampling
   gene_similarity
   sample_repertoire_overview

Graph Analysis
==============

.. toctree::
   :maxdepth: 1

   token_graph
   edit_distance_graph
   cdr3_aln_benchmark
   cdr3_graph

Distance, Matching, And Embedding
==================================

.. toctree::
   :maxdepth: 1

   prototype_embedding
   proto_embedding_new

Analysis And Modeling
=====================

These notebooks are exploratory and may require larger local datasets.

.. toctree::
   :maxdepth: 1

   vdjbet_yf

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
