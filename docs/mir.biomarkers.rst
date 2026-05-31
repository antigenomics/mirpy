mir.biomarkers package
======================

This package contains repertoire-level biomarker and motif-enrichment tools.

Practical notes
---------------

- ``mir.biomarkers.token_stats`` is the canonical module name for token/k-mer
   enrichment (the previous ``kmer_stats`` name has been retired).
- For GLIPH-style runs on large controls, prefer
   ``extract_gliph_artifacts_batch_from_repertoire(..., build_mappings=False, chunk_size=...)``
   and keep identical ``trim_first``/``trim_last`` settings for sample and control.
- ALICE runs neighborhood search and Pgen in two sequential parallel phases
   (trie search first, then OLGA Pgen) to avoid thread-pool contention.
- ``mir.biomarkers.associations`` provides whole-cohort clonotype/metadata
  association scans for binary or multiclass labels, with optional
  depth-aware GLM mode (``test='depth_glm'``) when sequencing depth should be
  modeled explicitly.

ALICE vs TCRNET
~~~~~~~~~~~~~~~

Both modules detect antigen-driven CDR3 clusters in T-cell or B-cell repertoires,
but differ in how they estimate the background:

**ALICE** (``mir.biomarkers.alice``) uses OLGA generation probability (Pgen) as
the background: ``λ = N × pgen_1mm`` (Pogorelyy et al. *PLoS Biol.* 2019).

- A 10 M synthetic MC pool estimates ``pgen_1mm`` rapidly; sequences with < 2
  pool matches fall back to OLGA analytical 1mm Pgen.
- The paper's original pool size is 100 M; the default ``match_mode="vj"``
  applies V+J gene restriction for both speed and specificity.
- Use ``pgen_mode="mc"`` for production; ``"exact"`` underestimates λ.

**TCRNET** (``mir.biomarkers.tcrnet``) is a **purely MC-control** algorithm:
no OLGA Pgen is computed.  Neighbor density is compared between sample and a
provided control using a binomial (or beta-binomial) test.

- Works with any control: real repertoire (captures V/J and length biases
  automatically) or a synthetic :class:`~mir.basic.pgen.McPgenPool`.
- When using a *synthetic* control, apply ``q_factor ≈ 3–5`` to correct for
  pre-selection bias (OLGA sequences are pre-thymic, at recombination, before
  thymic selection frequencies).  Estimate Q as ``median(pgen_real / pgen_olga)`` using
  :meth:`~mir.basic.pgen.McPgenPool.build_real` on a real control.
- To fully reproduce the original ALICE paper using TCRNET: use a 100 M
  synthetic pool, ``match_mode="vj"``, ``pvalue_mode="beta-binomial"``, and
  ``q_factor=Q``.
- Swap sample and control to detect **neighbor-depleted** sequences (clones
  present in the control but lost in the sample).

Sequence logos and motif selection logos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mir.biomarkers.motif_logo`` implements CDR3 sequence logos with OLGA-derived
background normalisation following Pogorelyy *et al.* (2019 *PLoS Biol.*).

**Scientific purpose.** V-gene and J-gene templates encode highly conserved
residues at the CDR3 ends (e.g. the N-terminal Cys and the J-gene STDTQYF
stretch in TRBV9/TRBJ2-3 CDR3s).  A plain IC logo shows these germline-encoded
letters as the tallest columns, obscuring the antigen-specific motif in the
CDR3 centre.  Subtracting an OLGA-derived background for the **same** V-gene /
J-gene / CDR3-length combination reduces the germline signal to ≈ 0 and reveals
only what is enriched or depleted relative to the recombination expectation.

**Formulas.**

Standard IC logo (Schneider *et al.* 1986)::

    IC[p] = log₂(20) + Σₐ f[p,a] · log₂(f[p,a])
    h_IC[p,a] = f[p,a] · IC[p]

Selection logo (per-residue KL divergence, always use with a VJ-matched
OLGA background)::

    h_sel[p,a] = f[p,a] · log₂(f[p,a] / f_bg[p,a])

h_sel > 0 → enriched (antigen-driven); h_sel < 0 → depleted (drawn inverted);
h_sel ≈ 0 at germline-encoded positions (f ≈ f_bg → VJ signal removed).

**Two background regimes.**

* **Per-VJ-len** (from :func:`~mir.biomarkers.motif_logo.get_vj_background`):
  removes both V-gene and J-gene germline signal.  Use for public TCR motifs
  such as CASSVGL[YF]STDTQYF (TRBV9/TRBJ2-3/len=15) or the GIL RS motif
  (TRBV19/TRBJ2-7/len=13).
* **All-VJ aggregate** (from
  :func:`~mir.biomarkers.motif_logo.aggregate_vj_background`): averages OLGA
  background across all VJ combinations for a given length, weighted by
  background pool size.  Retains V-gene and J-gene contributions; useful as a
  conservative baseline when sequences span multiple VJ genes.

**Key functions.**

- :func:`~mir.biomarkers.motif_logo.compute_pwm` — build a PWM from raw CDR3
  sequences.  Sequences are trimmed to the modal length; a Laplace pseudocount
  avoids zero-frequency cells.
- :func:`~mir.biomarkers.motif_logo.compute_logo` — add IC and selection-logo
  height columns.  Returned ``ic_height`` (bits, ≥ 0) and ``bg_height``
  (log-odds, can be negative for depleted residues).
- :func:`~mir.biomarkers.motif_logo.get_vj_background` — look up the OLGA-derived
  background PWM for a given V-gene / J-gene / CDR3-length.  Always set
  ``species`` and ``gene`` explicitly to avoid mixing TRA/TRB or human/mouse.
- :func:`~mir.biomarkers.motif_logo.aggregate_vj_background` — weighted-average
  OLGA background across all V/J combinations for a given CDR3 length.
- :func:`~mir.biomarkers.motif_logo.build_terminal_anchored_pwm` — build a
  fixed-width PWM that anchors the first *n_term* and last *c_term* residues of
  CDR3 sequences of *any* length.  The left block (labels ``1``, ``2``, …) is
  aligned by the N-terminus (V-gene); the right block (labels ``-c_term``, …,
  ``-1``) is aligned by the C-terminus (J-gene).  Use this to combine CDR3s of
  different lengths into a single publication-ready IC logo.
- :func:`~mir.biomarkers.motif_logo.build_terminal_anchored_logo` — the
  architecturally correct version of the terminal-anchored selection logo.
  Background subtraction (``h_sel = f · log₂(f / f_bg)``) is performed in the
  original *linear* CDR3 coordinate space (per CDR3 length), and positions are
  mapped to terminal display coordinates only afterwards.  Supports both
  ``motif_pwms`` and real/synthetic control backgrounds
  (:func:`~mir.biomarkers.motif_logo.get_vj_background_from_control`).
- :func:`~mir.biomarkers.motif_logo.get_vj_background_from_control` — build a
  VJ/length background PWM from a real or synthetic control repertoire
  DataFrame (e.g. from
  :meth:`~mir.common.control.ControlManager.load_control_df`) without relying
  on the pre-computed ``motif_pwms.txt.gz``.
- :func:`~mir.biomarkers.motif_logo.build_motif_logos_vj` — entry point for
  building selection logos from ALICE / TCRNET hits or connected components.
  Groups sequences by (V, J, length), builds per-VJ-len logos with matched OLGA
  backgrounds, and adds an all-VJ length-aggregated entry (key ``(None, None, len)``).
- :func:`~mir.biomarkers.motif_logo.plot_motif_logos` — two-panel figure (IC
  logo top, selection logo bottom).  V and J gene names appear in the title.
- :func:`~mir.biomarkers.motif_logo.compute_cluster_profiles` — per-position
  IC, entropy (H) and I_norm profiles for all qualifying clusters in
  ``motif_pwms.txt.gz``.

.. note::

   The ``motif_pwms.txt.gz`` ``height.I`` and ``height.I.norm`` columns use the
   VDJdb-motifs normalised scale: ``height.I`` is IC / log₂20 ∈ [0, 1] (not
   bits), and ``height.I.norm = −Σₐ f · ln(f_bg) / ln(20) / 2`` (always ≥ 0).
   These differ from the per-residue log-odds h_sel formula.

See the ``motif_logos`` notebook for worked examples: GILGFVFTL (Influenza A /
HLA-A*02:01 RS motif), HLA-B27 AS CASSVGL[YF]STDTQYF (reproducing Fig 2e of
Pogorelyy *et al.* 2019), aggregate TRA/TRB profile plots with fractional-
position x-axis (pos/len), and a background pool-size benchmark.

Submodules
----------

mir.biomarkers.motif_logo module
--------------------------------

.. automodule:: mir.biomarkers.motif_logo
   :members:
   :undoc-members:
   :show-inheritance:

mir.biomarkers.associations module
----------------------------------

.. automodule:: mir.biomarkers.associations
  :members:
  :undoc-members:
  :show-inheritance:
  :no-index:

mir.biomarkers.token_stats module
---------------------------------

.. automodule:: mir.biomarkers.token_stats
   :members:
   :undoc-members:
   :show-inheritance:

mir.biomarkers.tcrnet module
----------------------------

.. automodule:: mir.biomarkers.tcrnet
   :members:
   :undoc-members:
   :show-inheritance:

mir.biomarkers.alice module
---------------------------

.. automodule:: mir.biomarkers.alice
   :members:
   :undoc-members:
   :show-inheritance:

mir.biomarkers.gliph module
---------------------------

.. automodule:: mir.biomarkers.gliph
   :members:
   :undoc-members:
   :show-inheritance:

mir.biomarkers.metaclonotype_cluster module
-------------------------------------------

.. automodule:: mir.biomarkers.metaclonotype_cluster
   :members:
   :undoc-members:
   :show-inheritance:

