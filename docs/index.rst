mirpy documentation
===================

.. raw:: html

   <div class="mirpy-hero">
     <div class="mirpy-hero__content">
       <p class="mirpy-eyebrow">AIRR-seq toolkit</p>
       <h1>Analyze immune repertoires without rebuilding the basics every time.</h1>
       <p class="mirpy-lead">
         mirpy brings together parsing, repertoire abstractions, diversity metrics,
         distance-based analysis, and exploratory notebooks in one codebase.
       </p>
       <div class="mirpy-actions">
         <a class="mirpy-button mirpy-button--primary" href="getting-started.html">Get Started</a>
         <a class="mirpy-button mirpy-button--secondary" href="examples.html">Browse Notebooks</a>
         <a class="mirpy-button mirpy-button--ghost" href="modules.html">API Reference</a>
       </div>
       <div class="mirpy-stats">
         <div class="mirpy-stat"><strong>6</strong><span>core packages</span></div>
         <div class="mirpy-stat"><strong>15+</strong><span>published notebooks</span></div>
         <div class="mirpy-stat"><strong>1</strong><span>docs site from main</span></div>
       </div>
     </div>
     <div class="mirpy-hero__visual">
       <img src="_static/mirpy_logo.png" alt="mirpy logo" />
     </div>
   </div>

   <div class="mirpy-card-grid">
     <a class="mirpy-card" href="getting-started.html">
       <span class="mirpy-card__tag">Start here</span>
       <h3>Getting Started</h3>
       <p>Install the package, load segment libraries, parse repertoires, and understand the main concepts.</p>
     </a>
     <a class="mirpy-card" href="examples.html">
       <span class="mirpy-card__tag">Examples</span>
       <h3>Notebook Gallery</h3>
       <p>Open every analysis notebook from the repository directly in the docs, without maintaining duplicate examples.</p>
     </a>
     <a class="mirpy-card" href="modules.html">
       <span class="mirpy-card__tag">Reference</span>
       <h3>API Modules</h3>
       <p>Browse auto-generated API reference for parsers, repertoires, diversity utilities, matching, and embeddings.</p>
     </a>
   </div>

What mirpy covers
=================

.. raw:: html

   <div class="mirpy-feature-grid">
     <div class="mirpy-feature">
       <h3>Parsers and data structures</h3>
       <p><code>mir.common</code> provides segment libraries, clonotype models, repertoire containers, and format-specific parsers.</p>
     </div>
     <div class="mirpy-feature">
       <h3>Diversity and sampling</h3>
       <p><code>mir.basic</code> covers richness estimates, Hill curves, rarefaction, pgen utilities, and repertoire resampling.</p>
     </div>
     <div class="mirpy-feature">
       <h3>Distance and comparison</h3>
       <p><code>mir.distances</code> and <code>mir.comparative</code> support alignments, overlap analysis, matching, and graph workflows.</p>
     </div>
     <div class="mirpy-feature">
       <h3>Embeddings and notebooks</h3>
       <p><code>mir.embedding</code> plus the notebook gallery make it easier to move from reusable library code to exploratory analysis.</p>
     </div>
   </div>

Quick example
=============

.. code-block:: python

   from mir.common.segments import SegmentLibrary
   from mir.common.parser import VDJtoolsParser
   from mir.common.repertoire import Repertoire

   lib = SegmentLibrary.load_default(
       genes={"TRA", "TRB"},
       organisms={"HomoSapiens"},
   )

   parser = VDJtoolsParser(lib=lib, sep="\t")
   clonotypes = parser.parse("example.tsv")
   repertoire = Repertoire(clonotypes=clonotypes, gene="TRB")

   print(repertoire.number_of_clones)
   print(repertoire.number_of_reads)

Explore next
============

* :doc:`getting-started` for the shortest path from install to first parsed repertoire
* :doc:`examples` for the full notebook gallery published from the repository
* :doc:`modules` for generated API documentation tied to the current codebase

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting-started
   examples
   modules
