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

   import time

   from mir.common.parser import VDJtoolsParser
   from mir.common.repertoire_dataset import RepertoireDataset
   from mir.basic.segment_usage import StandardizedSegmentUsageTable

   t0 = time.time()
   dataset = RepertoireDataset.load(
       parser=VDJtoolsParser(sep=","),
       metadata=metadata,
       threads=32,
       paths=[f"assets/samples/fmba_healthy/{r['run']}.gz" for _, r in metadata.iterrows()],
   )
   print(time.time() - t0)

   folder_to_run_mapping = {}
   for folder in dataset.metadata[["run", "folder"]].folder.unique():
       folder_to_run_mapping[folder] = set(
           dataset.metadata[dataset.metadata.folder == folder].run
       )

   z_score_usage_table_v = StandardizedSegmentUsageTable.load_from_repertoire_dataset(
       repertoire_dataset=dataset,
       gene="TRB",
       segment_type="V",
       group_mapping=folder_to_run_mapping,
       metadata_column_for_group_mapping_name="run",
       standardization_method="log_exp",
   )
   z_score_usage_table_j = StandardizedSegmentUsageTable.load_from_repertoire_dataset(
       repertoire_dataset=dataset,
       gene="TRB",
       segment_type="J",
       group_mapping=folder_to_run_mapping,
       metadata_column_for_group_mapping_name="run",
       standardization_method="log_exp",
   )

   dataset = dataset.resample(
       updated_segment_usage_tables=[z_score_usage_table_v, z_score_usage_table_j],
       threads=32,
   )

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
