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
   metaclonotype_examples
   metaclonotype_method_compare

**gene_similarity** — Region-resolved germline V/J similarity for human and mouse: full vs paratope (CDR1+CDR2) vs framework (FR1-3) for V, and CDR3-part vs FR4 for J.

**token_graph** — Build bipartite k-mer/clonotype graphs and filter by sequence motifs.

**edit_distance_graph** — Construct Hamming/Levenshtein edit-distance graphs from junction sequences.

**vdjdb_junction_graph** — Analyse multi-epitope junction Hamming graphs from VDJdb.

**metaclonotype_examples** — Build, attach, summarize, and analyze metaclonotypes for functional diversity workflows.

**metaclonotype_method_compare** — Compare ALICE, TCRnet, TCRdist, edit-distance, TCREmp, and GLIPH metaclonotype clustering methods; benchmark paired-chain combined vs native TCREmp; concordance via ARI.

Biomarker Detection
-------------------

.. toctree::
   :maxdepth: 1

   alice_analysis
   covid19_biomarkers
   covid19_hla_biomarkers
   covid19_pairing_biomarkers
   tcrnet_analysis
   gliph_analysis
   vdjbet_yf

**alice_analysis** — ALICE antigen-expanded clone detection on YF, AS, and MLR datasets.

**covid19_biomarkers** — Whole-cohort COVID vs healthy clonotype association scan with functional filtering, batch correction parity, re-normalization, Fisher + depth-aware (`depth_glm`) modes, and reference concordance diagnostics.

**covid19_hla_biomarkers** — HLA-stratified TCR association analysis: DRB1*16/DQB1*05 sub-cohort Fisher scans, focused TRBV12-3/CASS replication (FDR=0.035), and HLA allele × global biomarker presence screen (3 569 pairs).

**covid19_pairing_biomarkers** — TRA × TRB co-occurrence Fisher tests (156 pairs × 3 strata), co-occurrence heatmaps and bubble chart, and VDJdb SARS-CoV-2 cross-validation (per-chain and paired-chain).

**tcrnet_analysis** — TCRNET enrichment for CMV+ vs B35+ donors with VDJdb annotation.

**gliph_analysis** — GLIPH-style multi-family token enrichment and clonotype graph clustering.

**vdjbet_yf** — VDJBet disease-associated overlap analysis on YFV vaccine time-series samples.

**motif_logos** — PWM construction, standard IC logos, and background-normalised logos for GILGFVFTL and B27 AS CDR3 motifs.

Sequence Logo Analysis
----------------------

.. toctree::
   :maxdepth: 1

   motif_logos

Aging And Cohort Overlap
------------------------

.. toctree::
   :maxdepth: 1

   aging_analysis
   aging_analysis_functional

**aging_analysis** — Donor-vs-pool overlap trends across aging, using the shared-worker many-vs-pool overlap path for faster repeated scoring.

**aging_analysis_functional** — Compare clonotypic and functional diversity, rarefaction, and F overlap for the AIRR benchmark aging cohort. Metaclonotypes built via 1-mismatch Hamming edit graph + Louvain clustering.

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

   tcrdist_analysis
   tcremp_vdjdb_analysis
   tcremp_vdjdb_analysis_paired
   tcremp_10xdcode_analysis
   tcremp_features_compare
   embed_any_chain

**tcrdist_analysis** — Compute TCRdist matrices, score radius-based neighborhoods, and derive metaclonotypes.

**tcremp_vdjdb_analysis** — Single-chain TCREmp on VDJdb epitope-labelled clonotypes with UMAP projection.

**tcremp_vdjdb_analysis_paired** — Paired TRA/TRB TCREmp on VDJdb full records with imputed missing chains.

**tcremp_10xdcode_analysis** — 10x CITE-seq + TCREmp embedding with DBSCAN epitope clustering.

**tcremp_features_compare** — Benchmark TCREmp feature modes (V+J+CDR3 vs CDR1+CDR2+CDR3) on VDJdb via PCA/DBSCAN/UMAP.

**embed_any_chain** — Template: embed any of the seven chains from OLGA mock data with 3000 prototypes; swap in your own repertoire.

Single-Cell Analysis
--------------------

.. toctree::
   :maxdepth: 1

   single_cell_load
   single_cell_pairing_analysis

**single_cell_load** — Load 10x VDJ v1 donors, inspect chain multiplicity, and benchmark against scirpy.

**single_cell_pairing_analysis** — Build and compare pairing graphs across raw, imputed, and cleaned stages.
