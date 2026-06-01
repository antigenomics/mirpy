Getting Started
===============

Installation
------------

``mirpy`` targets Python 3.11+ and includes a compiled extension for distance
calculations, so a working C/C++ toolchain is recommended when installing from
source.

.. code-block:: bash

   pip install mirpy-lib

For source installs from GitHub:

.. code-block:: bash

   git clone https://github.com/antigenomics/mirpy.git
   cd mirpy
   pip install .

For editable development installs:

.. code-block:: bash

   git clone https://github.com/antigenomics/mirpy.git
   cd mirpy
   ./setup.sh
   pip install -e .

To install the documentation toolchain as well:

.. code-block:: bash

   ./setup.sh --docs

Useful Links
------------

* API/module reference: https://antigenomics.github.io/mirpy/modules.html
* Notebook gallery page: https://antigenomics.github.io/mirpy/examples.html
* Notebook source directory: https://github.com/antigenomics/mirpy/tree/main/notebooks
* LLM agent skill guide: https://github.com/antigenomics/mirpy/blob/main/skills/mirpy/SKILL.md

Copilot Agent And Prompt
------------------------

The repository includes a dedicated custom agent and companion prompt for
mirpy-backed notebook analysis workflows.

* Agent file: ``.github/agents/mirpy-analysis.agent.md``
* Prompt file: ``.github/prompts/mirpy-analysis.prompt.md``

In chat, invoke ``/mirpy-analysis`` and provide:

* input data path(s) or instructions to obtain public/control data,
* optional metadata path/schema,
* workflow steps or one or more hypotheses to test.

The agent then creates and executes dedicated notebook(s), and for large data
first runs benchmark chunks to estimate full runtime and peak memory before
requesting confirmation for expensive runs.

Core Concepts
-------------

mirpy works with a small set of core data abstractions.

Clonotype
~~~~~~~~~

The smallest unit is a clonotype. Parsers convert rows from tabular repertoire
formats into ``Clonotype`` objects with AIRR-schema fields: ``junction``,
``junction_aa``, ``v_gene``, ``j_gene``, ``duplicate_count``, and boundary
coordinates.

For matrix-based methods that index germline alleles (for example ``TcrDist``
and ``TCREmp``), ``v_gene`` / ``j_gene`` may be provided with or without an
explicit allele suffix. Missing suffixes are normalized to major allele
``*01`` before distance lookup.

Repertoire
~~~~~~~~~~

A repertoire is a collection of clonotypes from one sample together with
metadata and summary counts such as ``clonotype_count`` and
``duplicate_count``.

Internally, repertoire data may remain as clonotype objects, lazy parser
columns, or a Polars table. Tabular conversion is lazy for list-based
construction and only materializes when needed.

RepertoireDataset
~~~~~~~~~~~~~~~~~

A repertoire dataset is a collection of repertoires loaded together for
multi-sample analysis, comparison, resampling, and cohort-level workflows.

ClonotypeDataset
~~~~~~~~~~~~~~~~

``ClonotypeDataset`` is an additional dataset-level abstraction used when the
analysis is centered on clonotypes themselves rather than on repertoire objects,
for example in matching, biomarker, or graph-oriented workflows.

Typical Workflow
----------------

1. Parse a file with one of the supported repertoire parsers.
2. Wrap clonotypes into a ``LocusRepertoire``, ``SampleRepertoire``, or ``RepertoireDataset``.
3. Compute repertoire-level summaries such as diversity or segment usage.
4. Move to matching, graph, or embedding utilities if deeper analysis is needed.

Diversity Analysis APIs
-----------------------

mirpy provides function-first diversity APIs in ``mir.common.diversity`` for
native summaries, Hill curves, and rarefaction/coverage curves. Repertoire
objects expose convenience methods that delegate to these same functions.

Function-first API
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mir.common.diversity import (
      summarize_clonotypes,
      summarize_loci_clonotypes,
      hill_curve_clonotypes,
      rarefaction_curve_clonotypes,
      summarize_count_groups,
   )

   # Clonotype list -> one summary
   trb_summary = summarize_clonotypes(sample["TRB"].clonotypes)

   # Mapping[locus, clonotypes] -> per-locus summaries
   per_locus = summarize_loci_clonotypes({locus: rep.clonotypes for locus, rep in sample.loci.items()})

   # Curves from raw clonotypes
   trb_hill = hill_curve_clonotypes(sample["TRB"].clonotypes)
   trb_rare = rarefaction_curve_clonotypes(sample["TRB"].clonotypes, m_steps=[100, 500, 1000, 5000])

   # Generic grouped-count API (useful for single-cell/pair-level counts)
   pair_counts = {"TRA_TRB": [2, 1, 1], "TRG_TRD": [1]}
   pair_summary = summarize_count_groups(pair_counts)

Object convenience methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods call the same function-level implementation shown above.

Metric summary (VDJtools-compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   Diversity indices follow `VDJtools <https://pubmed.ncbi.nlm.nih.gov/26606115/>`_
   (Shugay *et al.* 2015, *PLoS Comput. Biol.*, PMID:26606115).

``diversity(...)`` returns:

* ``abundance``: total counts in the selected counting mode
* ``diversity``: number of unique clonotypes (or paired clonotypes)
* ``singletons`` / ``doubletons``
* ``expanded``: clonotypes above 0.1%
* ``hyperexpanded``: clonotypes above 1%
* ``chao1``
* ``gini_simpson``
* ``shannon``

.. code-block:: python

   # Locus-level diversity (default: duplicate_count)
   trb_summary = sample["TRB"].diversity()

   # Optional count mode based on UMI counts
   trb_summary_umi = sample["TRB"].diversity(count_field="umi_count")

   # Sample-level per-locus diversity table
   per_locus = sample.diversity(per_locus=True)

Single-cell defaults and per-locus behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For single-cell paired data, count mode defaults to ``barcode_count``
(unique barcodes per clonotype or pair).

.. code-block:: python

   # Paired-clonotype summaries per locus pair (TRA_TRB, TRG_TRD, ...)
   pair_level = paired_repertoire.diversity()

   # Per-chain-locus summaries (TRA, TRB, ...)
   chain_level = paired_repertoire.diversity_by_locus()

   # SingleCellSample delegates to the paired repertoire
   sc_per_locus = single_cell_sample.diversity(per_locus=True)

Hill diversity profiles
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Locus-level Hill profile
   trb_hill = sample["TRB"].hill_curve()

   # Sample-level per-locus Hill curves
   sample_hill = sample.hill_curve(per_locus=True)

   # Single-cell per-chain-locus Hill curves (barcode_count default)
   sc_hill = single_cell_sample.hill_curve(per_locus=True)

Rarefaction and sample coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rarefaction uses abundance-table computations with NumPy/SciPy kernels and
returns both richness and sample-coverage estimates with confidence bounds.

.. code-block:: python

   # Rarefaction for one locus
   trb_rare = sample["TRB"].rarefaction_curve(m_steps=[100, 500, 1000, 5000])

   # Per-locus rarefaction for a sample
   sample_rare = sample.rarefaction_curve(per_locus=True)

   # Single-cell per-chain-locus rarefaction (barcode_count default)
   sc_rare = single_cell_sample.rarefaction_curve(per_locus=True)

Metaclonotype Workflows
-----------------------

Metaclonotypes are represented as lightweight membership tables instead of
re-wrapping repertoire objects. This keeps existing workflows intact while
adding cluster-level analytics.

Core structures
~~~~~~~~~~~~~~~

* ``MetaClonotypeClustering`` stores single-chain or paired cluster membership.
* ``LocusRepertoire.set_metaclonotypes(...)`` and paired-repertoire attachment
   methods keep clustering alongside existing repertoire objects.

Point-by-point clustering entry points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ALICE and TCRNET enriched seeds + first neighbors:

   * ``mir.biomarkers.metaclonotypes_from_alice``
   * ``mir.biomarkers.metaclonotypes_from_tcrnet``

* tcrtrie/custom search scope clustering (substitutions/indels/total edits):

   * ``mir.common.metaclonotypes_from_search_scope``
   * ``mir.utils.metaclonotype_clustering.metaclonotypes_from_search_scope``

* Continuous-radius / TCRdist-like representative clustering:

   * ``mir.common.metaclonotypes_from_radius_threshold``

* Edit-distance graph connected components / Leiden / Louvain:

   * ``mir.graph.metaclonotypes_from_edit_distance_graph``

* TCREmp DBSCAN/OPTICS/VDBSCAN label arrays:

   * ``mir.embedding.metaclonotypes_from_tcremp_labels``
   * ``mir.embedding.paired_metaclonotypes_from_tcremp_labels``

* Token bigraph / GLIPH clonotype graph communities:

   * ``mir.graph.metaclonotypes_from_token_clonotype_graph``
   * ``mir.graph.build_gliph_metaclonotypes``

Unified clustering interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mir.biomarkers.metaclonotype_cluster`` provides a single config-driven entry
point for all six clustering methods and supports paired-chain analysis.

.. code-block:: python

   from mir.biomarkers.metaclonotype_cluster import (
       MetaclonotypeClusterConfig,
       cluster_metaclonotypes,
       cluster_paired_metaclonotypes,
   )

   cfg = MetaclonotypeClusterConfig(
       method="edit_distance",
       locus="TRB",
       max_distance=1,
       graph_algo="louvain",
       min_cluster_size=2,
   )
   meta = cluster_metaclonotypes(repertoire, cfg)

   # Paired analysis: per-chain clusters combined as "chain1_id.chain2_id"
   paired_cfg = MetaclonotypeClusterConfig(
       method="edit_distance",
       locus_pair="TRA_TRB",
       max_distance=1,
       graph_algo="components",
   )
   paired_meta = cluster_paired_metaclonotypes(paired_locus_repertoire, paired_cfg)

Paired-from-single combining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any single-chain metaclonotype results can be combined into paired-chain
clusters using ``paired_metaclonotypes_from_single_chain``:

.. code-block:: python

   from mir.utils.metaclonotype_clustering import paired_metaclonotypes_from_single_chain

   paired_meta = paired_metaclonotypes_from_single_chain(
       paired_clonotypes,
       meta_chain1=meta_tra,
       meta_chain2=meta_trb,
       cluster_separator=".",
   )

Each combined cluster ID has the form ``"<chain1_cluster>.<chain2_cluster>"``.
Pairs where one or both chains are unassigned are excluded by default; pass
``include_unassigned=True`` to keep them with a placeholder label.

Downstream metaclonotype analytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Functional diversity and rarefaction from aggregated cluster counts:

   * ``functional_diversity``
   * ``functional_hill_curve``
   * ``functional_rarefaction_curve``

* Functional overlap-1 (clusters match if they share at least one clonotype):

   * ``functional_overlap_1``

* Entropy decomposition for pooled-vs-separate clustering:

   * ``pooled_entropy_difference`` computes
      ``H(A pooled with B) - H(A) - H(B)``

* Motif logo inputs from metaclonotypes:

   * ``metaclonotype_junctions`` provides sequence lists for
      ``compute_pwm`` / ``compute_logo`` workflows.

Pairwise Overlap Spaces
-----------------------

``mir.comparative.overlap`` supports four identity spaces (see also
Shugay *et al.* 2018, *PNAS*,
`PMID:30158170 <https://pubmed.ncbi.nlm.nih.gov/30158170/>`_ for a
high-resolution repertoire overlap analysis use case):

* ``"ntvj"``: nucleotide CDR3 + V + J
* ``"nt"``: nucleotide CDR3 only
* ``"aavj"``: amino-acid CDR3 + V + J
* ``"aa"``: amino-acid CDR3 only

Only amino-acid spaces (``"aa"`` and ``"aavj"``) support approximate
matching (Hamming/Levenshtein with ``threshold > 0``).

.. code-block:: python

   from mir.comparative.overlap import pairwise_overlap

   # Exact nucleotide+V+J overlap
   r_ntvj = pairwise_overlap(rep_a, rep_b, overlap_space="ntvj")

   # Approximate amino-acid overlap (1 mismatch)
   r_aa_1mm = pairwise_overlap(
      rep_a,
      rep_b,
      overlap_space="aavj",
      metric="hamming",
      threshold=1,
      n_jobs=-1,
   )

By default, overlap APIs use all physical CPU cores (``n_jobs=-1``). Pass
``n_jobs=1`` for deterministic single-core timing.

The pairwise API reports similarity-centric outputs:

* ``f_similarity``
* ``d_similarity``
* ``f2_similarity``
* ``jaccard``
* ``szymkiewicz_simpson``
* ``morisita_horn``

Use distance-like conversions only where required by downstream methods (for
example UMAP/MDS with precomputed dissimilarities):

* ``f_metric = 1 - f_similarity``
* ``d_metric = 1 / d_similarity`` (for ``d_similarity > 0``)

In amino-acid overlap spaces, non-coding clonotypes are excluded from overlap
matching and reported with bounded warnings.

Single-Cell 10x Paired-Chain Loading
------------------------------------

mirpy includes a dedicated loader for 10x VDJ v1 paired-chain data that keeps
cell barcode linkage separate from paired-clonotype objects.

.. code-block:: python

   from mir.common.single_cell import (
       build_tenx_sample_from_cell_clonotypes,
       load_10x_vdj_v1_sample,
   )
   from mir.common.single_cell_parser import load_10x_vdj_v1_cell_clonotypes
   from mir.common.single_cell_repair import cleanup_cell_clonotypes, impute_missing_chains

   sample = load_10x_vdj_v1_sample(
       consensus_annotations_path="dcode/vdj_v1_hs_aggregated_donor1_consensus_annotations.csv.gz",
       all_contig_annotations_path="dcode/vdj_v1_hs_aggregated_donor1_all_contig_annotations.csv.gz",
       sample_id="sample1",
       check_is_cell=True,
   )

   # Number of barcoded cells with matched consensus linkage
   print(sample.loaded_cell_count)

   # Number of distinct matched clonotypes used for pairing
   print(sample.loaded_clonotype_count)

   # Multiplicity bins by locus pair (n x m chain counts)
   print(sample.chain_multiplicity)

The loader supports locus-pair families ``TRA_TRB``, ``TRG_TRD``,
``IGH_IGK``, and ``IGH_IGL`` and expands multi-chain cells via deterministic
cartesian pairing (e.g., ``2x1`` yields two paired clonotypes).

For parser-first workflows:

.. code-block:: python

   cell_table = load_10x_vdj_v1_cell_clonotypes(
       "..._consensus_annotations.csv.gz",
       "..._all_contig_annotations.csv.gz",
       sample_id="sample1",
       check_is_cell=True,
   )
   imputed = impute_missing_chains(cell_table, reuse_slave_per_master=True)
   cleaned = cleanup_cell_clonotypes(
       imputed,
          # Keep one canonical synthetic slave per master clonotype.
       enforce_consistent_slave_per_master=True,
       consistency_only_on_synthetic_slave=True,
          # Remove oversized master/slave communities from downstream graph analysis.
       max_slave_edges_per_master=10,
   )
   sample = build_tenx_sample_from_cell_clonotypes(cleaned, sample_id="sample1")

Repair controls summary:

* ``reuse_slave_per_master=True`` reuses one synthetic slave clonotype per
   master clonotype during imputation.
* ``enforce_consistent_slave_per_master=True`` keeps master->slave assignments
   consistent across cells after cleanup.
* ``consistency_only_on_synthetic_slave=True`` restricts consistency enforcement
   to synthetic slave chains (set ``False`` to include observed slaves).
* ``max_slave_edges_per_master=10`` removes flagged master/slave families when
   a master clonotype connects to too many distinct slave clonotypes.

Notebook examples:

* ``notebooks/single_cell_load.ipynb``: 10x sample loading and concordance checks.
* ``notebooks/single_cell_pairing_analysis.ipynb``: raw vs imputed vs cleanup pairing graphs
   plus TRA/TRB stage heatmaps.

Single-Cell 10x + CITE-seq Loading
----------------------------------

For dcode 10x workflows with donor-specific ``*_binarized_matrix.csv.gz`` files,
use ``SingleCellSample`` to keep paired repertoires and CITE-seq labels together.

.. code-block:: python

   from mir.common.single_cell import (
      load_10x_vdj_v1_citeseq_sample,
      validate_citeseq_binders_against_vdjdb_10x,
   )

   sample = load_10x_vdj_v1_citeseq_sample(
      consensus_annotations_path="dcode/vdj_v1_hs_aggregated_donor1_consensus_annotations.csv.gz",
      all_contig_annotations_path="dcode/vdj_v1_hs_aggregated_donor1_all_contig_annotations.csv.gz",
      binarized_matrix_path="dcode/vdj_v1_hs_aggregated_donor1_binarized_matrix.csv.gz",
      sample_id="donor1",
   )

   print(sample.paired_repertoire.loaded_cell_count)
   print(sample.cite_seq_matrix.height)
   print(sample.cite_seq_binder_columns.height)

   missing = validate_citeseq_binders_against_vdjdb_10x(
      sample.cite_seq_binder_columns,
      "vdjdb_full.txt.gz",
   )
   print(missing)

Notebook example:

* ``notebooks/tcremp_10xdcode_analysis.ipynb``: donor-wide 10x+CITE-seq sanity checks
   and donor1 TRA/TRB/paired TCREmp diagnostics (polars-only preprocessing), including
   cumulative PCA variance curves, k-nearest-neighbor kneedle/eps plots, UMAP
   projections colored by epitope labels, and per-epitope classification score tables.

Paired TCREmp From VDJdb Full
-----------------------------

mirpy also supports paired-chain prototype embeddings built from VDJdb full
rows. The paired embedding is a simple concatenation of the TRA and TRB
TCREmp vectors in canonical ``TRA_TRB`` order.

.. code-block:: python

   from mir.common.parser import VDJdbFullPairedParser
   from mir.common.single_cell import build_tenx_sample_from_cell_clonotypes
   from mir.common.single_cell_repair import impute_missing_chains
   from mir.embedding.tcremp import PairedTCREmp

   parser = VDJdbFullPairedParser()

   # Strict mode: keep only rows that already contain both alpha and beta chains.
   strict_df, strict_meta = parser.parse_cell_clonotypes_file(
      "vdjdb_full.txt.gz",
      species="HomoSapiens",
      include_incomplete=False,
   )
   strict_sample = build_tenx_sample_from_cell_clonotypes(
      strict_df,
      sample_id="vdjdb_full_human_strict",
      barcode_metadata=strict_meta,
   )

   # Imputation mode: keep incomplete rows, synthesize the missing chain, then pair.
   impute_df, impute_meta = parser.parse_cell_clonotypes_file(
      "vdjdb_full.txt.gz",
      species="HomoSapiens",
      include_incomplete=True,
   )
   imputed_df = impute_missing_chains(impute_df)
   imputed_sample = build_tenx_sample_from_cell_clonotypes(
      imputed_df,
      sample_id="vdjdb_full_human_imputed",
      barcode_metadata=impute_meta,
   )

   model = PairedTCREmp.from_defaults(
      species="human",
      locus_pair="TRA_TRB",
      n_prototypes=500,
   )
   paired_clonotypes = imputed_sample.paired_locus_repertoires["TRA_TRB"].paired_clonotypes
   X = model.embed(paired_clonotypes)

Each synthetic barcode stores VDJdb metadata such as ``mhc.a``, ``mhc.b``,
``mhc.class``, ``antigen.epitope``, ``antigen.gene``, and
``antigen.species`` in ``sample.single_cell_repertoire.barcode_metadata``.

Notebook examples:

* ``notebooks/tcremp_vdjdb_analysis.ipynb``: single-chain TRA/TRB TCRemp analysis on VDJdb slim.
* ``notebooks/tcremp_vdjdb_analysis_paired.ipynb``: paired TRA/TRB embeddings on VDJdb full with strict and imputed workflows,
  including polars-only PCA variance explained, bounded-kneedle DBSCAN diagnostics,
  purity/retention/consistency summaries, and SLL epitope outlier checks across paired,
  TRA-only, and TRB-only embeddings.

In the paired notebook, VDJdb epitope metadata is demonstrated both as direct
``barcode_metadata`` lookups and as a tabular ``metadata_to_polars()`` view for
filtering and joins without pandas.

Benchmarking 10x Loading
------------------------

Use benchmark tests on AIRR benchmark 10x donor files to track speed, memory,
and concordance with scirpy loading behavior.

.. code-block:: bash

   env RUN_BENCHMARK=1 python -m pytest tests/test_single_cell_10x_benchmark.py -s -x
   env RUN_BENCHMARK=1 python -m pytest tests/test_single_cell_repair_benchmark.py -s -x
   env RUN_BENCHMARK=1 python -m pytest tests/test_single_cell_citeseq_benchmark.py -s -x
   env RUN_BENCHMARK=1 python -m pytest tests/test_tcremp_vdjdb_benchmark.py -s -x

The benchmark suite asserts:

* 10x sample objects load with non-empty cell/clonotype/pairing outputs,
* per-donor runtime and RSS deltas remain within bounded limits,
* mirpy vs scirpy TRA/TRB quadrant patterns are concordant on dominant bins,
* mirpy speed and memory are competitive relative to scirpy on the same donor.

For batch-corrected gene usage workflows, use
``compute_batch_corrected_gene_usage`` and consume ``pfinal`` as the corrected
probability output (already normalized per sample/locus).

If you compute only ``scope='vj'`` and need V- or J-marginal views for plots,
reuse ``marginalize_batch_corrected_gene_usage(..., scope='v'|'j')`` from the
same module instead of ad-hoc notebook ``groupby`` code.

Pooling Repertoires Across Samples
----------------------------------

Use ``pool_samples`` to combine clonotypes across samples with explicit
identity rules.

.. code-block:: python

   from mir.common.pool import pool_samples

   # Pool two samples by nucleotide CDR3 + V/J genes.
   pooled = pool_samples([sample_rep_1, sample_rep_2], rule="ntvj", weighted=True)

   # Pool a dataset by amino-acid CDR3 + V/J genes and keep sample ids per pooled clone.
   pooled_ds = pool_samples(dataset, rule="aavj", include_sample_ids=True)

Supported pooling rules:

* ``ntvj``: key ``(junction, v_gene, j_gene)``
* ``nt``: key ``(junction,)``
* ``aavj``: key ``(junction_aa, v_gene, j_gene)``
* ``aa``: key ``(junction_aa,)``

Each pooled clonotype stores:

* ``duplicate_count`` as the sum over grouped clonotypes,
* ``incidence`` in clonotype metadata (number of unique samples containing the key),
* ``occurrences`` in clonotype metadata (number of grouped rows).

Neighborhood Enrichment and Clonotype Similarity
------------------------------------------------

Use ``compute_neighborhood_stats`` to find clonotypes similar to each other
based on edit distance in the CDR3 junction region. This is useful for
TCRnet and ALICE algorithms.

.. code-block:: python

   from mir.graph import compute_neighborhood_stats

   # Count neighbors for each clonotype within edit distance 1
   stats = compute_neighborhood_stats(
       repertoire,
       metric="hamming",
       threshold=1,
       match_v_gene=True,
   )

   # stats["clonotype_id"] = {
   #     "neighbor_count": 15,
   #     "potential_neighbors": 200,
   # }

Supported options:

* ``metric``: ``"hamming"`` or ``"levenshtein"`` for junction_aa comparison
* ``threshold``: Maximum edit distance to consider a clonotype a neighbor
* ``match_v_gene``: If True, only count neighbors with matching V gene
* ``match_j_gene``: If True, only count neighbors with matching J gene

For larger repertoires, neighborhood and edit-distance graph builders run with
multiprocess workers when ``n_jobs > 1`` to leverage true multi-core execution.

You can compute neighborhood stats against an explicit background repertoire:

.. code-block:: python

   # Query against background (adds +1 pseudocount in background mode)
   bg_stats = compute_neighborhood_stats(
      repertoire,
      background=background_repertoire,
      metric="hamming",
      threshold=1,
   )

To attach parent-vs-background neighborhood enrichment metadata in one call:

.. code-block:: python

   from mir.graph import add_neighborhood_enrichment_metadata

   add_neighborhood_enrichment_metadata(
      repertoire,
      background=background_repertoire,
      metric="hamming",
      threshold=1,
      metadata_prefix="neighborhood",
   )

This writes parent/background counts, potentials, densities, and
``neighborhood_enrichment`` for each clonotype.

Control Data Setup
------------------

Background controls are expensive to build/download and are managed explicitly
through ``mir.common.control``.

.. code-block:: python

   from mir.common.control import ControlManager

   mgr = ControlManager()  # default: ~/.cache/mirpy/controls (or MIRPY_CONTROL_DIR)

   # Build synthetic OLGA control (default n=10_000_000)
   mgr.ensure_synthetic_control("human", "TRB", n=1_000_000)

   # Download real control from HuggingFace dataset and convert to pickle
   mgr.ensure_real_control("hsa", "Tbeta")

   # Load normalized ntvj table (duplicate_count, junction, junction_aa, v_gene, j_gene)
   df_control = mgr.load_control_df("synthetic", "human", "TRB")

   # Or build/fetch on demand when a workflow needs a control immediately
   df_real = mgr.ensure_and_load_control_df("real", "human", "TRB")

You can also prebuild controls via CLI:

.. code-block:: bash

   mirpy-control-setup --type synthetic --species human,mouse --loci TRA,TRB --n 1000000

Control setup is concurrency-safe: when multiple workers request the same
control simultaneously, one process builds while others wait on a per-control
lock and then reuse the produced artifact.

Available species aliases: ``human``/``hsa``/``HomoSapiens`` and
``mouse``/``mmu``/``MusMusculus``. Loci aliases include IMGT names and forms
such as ``Talpha``/``Tbeta``.

Bag-of-K-mers Control Profiles
------------------------------

Use ``mir.embedding.bag_of_kmers`` to compute background k-mer statistics for
enrichment workflows.

By default, control k-mer profiles are built in memory (no profile-table write
to cache). This is convenient for one-off analyses.

.. code-block:: python

   from mir.common.control import ControlManager
   from mir.embedding.bag_of_kmers import BagOfKmersParams, build_control_kmer_profile

   mgr = ControlManager()
   params = BagOfKmersParams(use_v=False, k=3, gapped=False, reduced_alphabet=False)

   profile = build_control_kmer_profile(
      mgr,
      control_type="real",
      species="human",
      locus="TRB",
      params=params,
   )

   token_stats = profile.token_stats        # columns: token, n, T, p, idf
   position_stats = profile.position_stats  # columns: token, count, pos, junction_len

To enable profile-table caching for repeated runs, pass ``cache=True``:

.. code-block:: python

   profile_cached = build_control_kmer_profile(
      mgr,
      control_type="real",
      species="human",
      locus="TRB",
      params=params,
      cache=True,
   )

Cached profile writes are lock-protected to avoid race conditions under
concurrent workers.

ALICE-Style Neighborhood Enrichment
-----------------------------------

Use ``mir.biomarkers.alice`` to compute per-clonotype neighborhood enrichment
using OLGA generation probabilities as null model (Pogorelyy *et al.* 2019,
*PLoS Biol.*, `PMID:31194732 <https://pubmed.ncbi.nlm.nih.gov/31194732/>`_).
ALICE estimates how many neighbors each clonotype would accumulate by chance
given its sequence's Pgen.

ALICE is metadata-first: neighbor counts, expected neighbors, fold enrichment,
p-values, and BH-adjusted q-values are written directly into clonotype
metadata. A tabular result is also returned by default.

.. code-block:: python

   from mir.biomarkers.alice import compute_alice, add_alice_metadata

   # Compute enrichment; result.table has pre-computed q_value column.
   result = compute_alice(
      repertoire,
      metric="hamming",
      match_mode="vj",         # one of: none, v, j, vj
      pgen_mode="1mm",         # "exact" (Hamming-0) or "1mm" (Hamming-1)
      pvalue_mode="poisson",   # or "negative-binomial" for overdispersed data
      pseudocount=0.0,         # added to n and N before expected/p-value
      n_jobs=4,
   )
   df = result.table   # columns: sequence_id, locus, junction_aa, ..., q_value

   # Or annotate clonotypes in-place.
   add_alice_metadata(
      repertoire,
      metric="hamming",
      match_mode="vj",
      pgen_mode="1mm",
      pvalue_mode="poisson",
      pseudocount=0.0,
   )

Output columns: ``alice_n``, ``alice_N``, ``alice_pgen_raw``, ``alice_pgen``,
``alice_expected``, ``alice_fold``, ``alice_p_value``, ``alice_q_value``.

OLGA models are cached per ``(species, locus, seed, model_class)`` — repeated
calls within a session reuse the loaded model without re-reading from disk.
Bulk Pgen and per-batch metric stages use multiprocess workers by default to
achieve true CPU parallelism; set ``MIRPY_..._EXECUTOR=thread`` only when
debugging thread-local behavior.

TCRNET-Style Neighborhood Enrichment
------------------------------------

Use ``mir.biomarkers.tcrnet`` to compute per-clonotype neighborhood
enrichment against either user-provided controls or built-in real/synthetic
controls managed by ``ControlManager`` (Shugay *et al.* 2025,
*Briefings Bioinform.*, `PMID:40996146 <https://pubmed.ncbi.nlm.nih.gov/40996146/>`_).

TCRNET is metadata-first: neighbor counts, p-values, and BH-adjusted
q-values are written directly into clonotype metadata. A tabular result is
optional.

.. code-block:: python

   from mir.biomarkers.tcrnet import add_tcrnet_metadata, compute_tcrnet, tcrnet_table

   # User-provided control repertoire, annotate clonotypes in-place.
   annotated = add_tcrnet_metadata(
      target_repertoire,
      control=control_repertoire,
      metric="hamming",
      threshold=1,
      n_jobs=-1,
      match_mode="none",       # one of: none, v, j, vj
      pvalue_mode="binomial",  # or "beta-binomial"
      pseudocount=1.0,         # added to control m and M before density calculation
   )

   # Optional table view from clonotype metadata.
   df = tcrnet_table(annotated)

   # compute_tcrnet can also return a table directly (as_table=True by default).
   result = compute_tcrnet(target_repertoire, control=control_repertoire)
   df2 = result.table

You can also use managed controls directly:

.. code-block:: python

   result = compute_tcrnet(
      target_repertoire,
      control_type="real",   # or "synthetic"
      species="human",
      n_jobs=-1,
      normalize_control_vj_usage=True,
      pvalue_mode="beta-binomial",
   )

You can also add neighborhood stats directly to clonotype metadata:

.. code-block:: python

   from mir.graph import add_neighborhood_metadata

   add_neighborhood_metadata(repertoire, metric="hamming", threshold=1, n_jobs=-1)
   # Adds neighborhood_count and neighborhood_potential to each clonotype's metadata

Benchmark Reference
-------------------

Benchmark coverage includes both synthetic generation and real-control
download/build paths (HuggingFace), with cache-hit timing diagnostics in
``tests/test_control_benchmark.py``.

.. code-block:: bash

   # run only TCRNET benchmarks
   RUN_BENCHMARK=1 pytest -s tests/test_tcrnet_benchmark.py -m benchmark

   # run the slow B35 real-control benchmark only
   RUN_BENCHMARK=1 pytest -s \
      tests/test_tcrnet_benchmark.py::test_tcrnet_benchmark_b35_epl_connected_component_vs_real_control

Benchmark timing details are appended to ``tests/benchmarks.log``.
Timeouts for long benchmarks can be configured via
``MIRPY_BENCH_SLOW_TIMEOUT_S`` and ``MIRPY_BENCH_VERY_SLOW_TIMEOUT_S``.
Defaults are 600 s and 1800 s, respectively.

For a larger ALICE/TCRNET benchmark profile:

.. code-block:: bash

   RUN_BENCHMARK=1 \
   MIRPY_BENCH_FAST_MAX_CLONOTYPES=600 \
   MIRPY_BENCH_FAST_SYNTHETIC_N=200000 \
   MIRPY_BENCH_REAL_MAX_CLONOTYPES=200 \
   MIRPY_BENCH_REAL_CONTROL_LIMIT=100000 \
   MIRPY_BENCH_REAL_SYNTHETIC_N=200000 \
   pytest -s tests/test_alice_tcrnet_benchmark.py

To include the full 1e6 neighborhood scaling benchmark:

.. code-block:: bash

   RUN_BENCHMARK=1 RUN_FULL_BENCHMARK=1 \
   MIRPY_BENCH_WORKERS=1,4,8 \
   pytest -s tests/test_neighborhood_enrichment_scaling_benchmark.py

Next Steps
----------

* Start with :doc:`examples` for the full notebook gallery.
* Browse :doc:`modules` for API documentation.
