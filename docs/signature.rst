The portable repertoire signature
=================================

One AIRR repertoire in, one **fixed, named, positional** feature vector out — computable by anyone
who ``pip install mirpy-lib``, on their own samples, and directly comparable with yours. That is
the whole design goal: a matrix you can hand a collaborator that drops into PCA, logistic
regression, random forest, boosting or an MLP with no scaler of their own.

Quickstart — one command
------------------------

No Python needed. This is the command to send a collaborator:

.. code-block:: bash

   pip install mirpy-lib

   mir signature --preset classify cohort/*.tsv.gz -o sig.parquet

Files sharing a sample id (the name up to the first dot) are joined into one multi-locus sample, so
a donor sequenced on TRA and TRB is one signature with both loci filled rather than two half-empty
rows. AIRR Rearrangement, native vdjtools, Parquet and the usual third-party exports are
auto-detected.

.. code-block:: bash

   mir signature --preset classify --describe        # the columns, reading no input
   mir signature --preset classify --threads 0 cohort/*.tsv.gz -o sig.parquet   # every core
   mir presets                                       # the named feature sets, ranked
   mir presets classify                              # one in full: what, how, when

Three presets are marked ``recommended``: **compact** (the smallest vector that still describes a
repertoire, usable at *n* = 50), **classify** (general-purpose, the usual random-forest / boosting
input), and **transfer** (for a model that must work on another lab's samples). Unlike
``vdjtools signature``, which serves the ``vsig`` half only, every preset resolves here in full —
mirpy is where the two halves meet.

.. warning::

   **CDR3 vs junction.** The reader prefers AIRR ``junction_aa`` (conserved anchors *included*) and
   falls back to IMGT ``cdr3_aa`` (anchors *excluded*). A file carrying only ``cdr3_aa`` is two
   residues short everywhere, which shifts the length, k-mer and Pgen features. Check your headers
   before you trust a matrix.

Two more things worth knowing: ``--standardize reference`` (the default) is what makes your vector
comparable with anyone else's, and you should **not** PCA-project the result — plain scaling beat
projection at every rank tested.

A folder of AIRR files, and a table you can join
------------------------------------------------

The complete recipe, end to end. Input: a directory of per-sample AIRR TSVs and your own metadata
sheet. Output: one TSV with one row per sample, joinable on ``sample_id``.

.. code-block:: bash

   mir signature --preset classify samples/*.tsv -o sig.tsv

``sample_id`` is the file name up to the first dot, so ``samples/SRR8364167.tsv`` becomes
``SRR8364167``. Name your files after whatever key your metadata already uses and the join needs no
mapping table:

.. code-block:: python

   import polars as pl

   sig  = pl.read_csv("sig.tsv",  separator="\t")
   meta = pl.read_csv("meta.tsv", separator="\t")     # your own sheet
   full = sig.join(meta, left_on="sample_id", right_on="Run", how="left")
   full.write_csv("signature_with_metadata.tsv", separator="\t")

That is the whole pipeline. Everything after it is your analysis.

Run on the 1,764-sample SRA cohort in |airr_benchmark| this takes roughly 0.6 s per sample on eight
cores, and a 20-sample subset produced **615 columns with every ``sample_id`` matching its metadata
row**. See :doc:`notebooks` for the runnable version.

.. |airr_benchmark| raw:: html

   <a href="https://huggingface.co/datasets/isalgo/airr_benchmark">isalgo/airr_benchmark</a>

What ``nan`` means in the output
--------------------------------

A hole is never filled with zero. ``nan`` means *not estimable for this sample*, and there are two
distinct reasons, which you separate with the mask columns ``vsig:mask:<locus>:present`` and
``vsig:mask:<locus>:estimable``:

- **The locus is not in the file.** A TRB-only library has ``nan`` everywhere under ``:IGH:``.
- **The locus is there but too shallow** for that particular estimator.

There is also a third, which is a property of the shipped artifact rather than of your data, and it
is worth knowing before you see it. Measured on 20 samples of the SRA cohort with
``--preset classify``: 28 of 615 columns are ``nan`` for **every** sample, and 20 of those 28 are
the coverage-standardised diversity block -- ``vsig:div:{0D_c,1D_c,2D_c,clonality}`` -- on exactly
the five loci the bundled scale reference has no coverage constant for: IGH, IGK, IGL, TRG and TRD.
The remaining eight are ``vsig:pgen:frac_atypical`` on those same five loci, ``vsig:shm`` on IGH and
``rsig:band:top`` on two loci.

TRA and TRB have **0** such columns. This is not a defect in your samples: the bundled reference was
fitted on targeted TCR libraries, so it carries ``cstar`` for TRA and TRB and for nothing else. The
B-cell and gamma-delta diversity columns come back the moment a reference covering those loci is
selected -- see :ref:`which-scale-reference`.

Raw block values, if you want them
----------------------------------

``--standardize none`` emits the same columns **before** any reference rescaling -- raw counts,
fractions and Hill numbers in their own units rather than standardised against a corpus:

.. code-block:: bash

   mir signature --preset classify --standardize none samples/*.tsv -o raw.tsv

This is the right output when you are building your own within-cohort model and do not need
cross-cohort comparability, and it is also the answer to "where did my IGH diversity go": raw Hill
numbers need no ``cstar``, so they are populated on all seven loci. What you give up is exactly what
standardisation buys -- a column that means the same thing in your matrix and a collaborator's.

The Python API
--------------

.. code-block:: python

   from mir.signature import signature, signature_cohort

   v = signature({"TRB": df})                 # {column: value}, standardised, layout order
   F = signature_cohort(samples)              # one row per sample, positional
   F.write_parquet("cohort.parquet")

Two halves, one contract
------------------------

The signature is the concatenation of two vectors that answer different questions about the same
sample, joined on ``sample_id`` and namespaced so they never collide:

.. list-table::
   :header-rows: 1
   :widths: 12 44 44

   * -
     - ``vsig`` — statistics (:mod:`vdjtools.signature`)
     - ``rsig`` — geometry (:mod:`mir.signature`)
   * - basis
     - the clone-size vector and the germline vocabulary
     - the prototype-sum measure :math:`\Phi(S) = \sum_\sigma w_\sigma z_\sigma`
   * - blocks
     - ``mask qc depth div clon len iso shm pair pgen aa pchem``
     - ``depth div band contrast phiv phij phic``
   * - each column is
     - a defined statistic of the clone-size vector
     - a linear functional, a norm, or a mixture coefficient of :math:`\Phi`

``depth`` and ``div`` appear on **both** sides deliberately. They are different objects with the
same name family — count-native Hill numbers at a frozen coverage level on one side,
embedding-native effective sample size and Rao dispersion on the other — and which one carries a
given phenotype is itself a result.

Why every column is transformed before you see it
-------------------------------------------------

A learner cannot be handed a log-scaled read count, an isotype fraction and a principal component
in one matrix and be expected to weight them sensibly. Each feature therefore carries a
**variance-stabilising transform chosen from its support, not from taste**, and each of those
choices is denominator-aware — the alternative silently lies about shallow samples:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - transform
     - where, and why that one
   * - ``log10`` / ``log1p``
     - counts and norms, whose spread scales with their magnitude
   * - ``logit``
     - a proportion, Haldane–Anscombe corrected, so ``0/3`` and ``0/500`` are different numbers
   * - ``arcsine``
     - Anscombe's variance-stabiliser for a binomial share; defined at exactly zero
   * - ``clr``
     - a composition, over the **whole** composition before any coordinate is selected, shipping
       *k−1* parts because all *k* are linearly dependent and would put a guaranteed zero
       eigenvalue in any PCA

On top of that every column is rescaled against a frozen reference (median and
:math:`1.4826\cdot\mathrm{MAD}`, clipped), which is what makes two people's matrices comparable
rather than each being internally consistent and mutually meaningless.

Geometry is not transformed, and that is not an oversight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two halves of the signature end up with almost opposite transform tables, because they hold
almost opposite kinds of object:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - ``vsig`` — statistics
     - ``rsig`` — geometry
   * - dominant transform
     - ``arcsine`` ×20, ``clr`` ×7, ``logit`` ×7, ``log10`` ×6
     - ``none`` ×116
   * - support
     - :math:`[0,1]` or :math:`[0,\infty)`, bounded, discrete
     - :math:`\mathbb{R}`, signed, continuous
   * - mean–variance coupling
     - yes — binomial or Poisson; variance is a function of the mean
     - no — :math:`\operatorname{Var} \approx \sigma^2 / n_{\text{eff}}`, independent of the value

A coordinate of :math:`\Phi = \sum_\sigma w_\sigma z_\sigma` is a linear functional of a weighted
mean of *fixed* embedding vectors. It is signed and roughly symmetric, there is no boundary to
compress against, and its variance does not depend on its own value — exactly the condition under
which a variance-stabilising transform buys nothing. Applying one would not merely be useless:
``log``, ``logit`` and ``arcsine`` all require a non-negative or :math:`[0,1]` domain, and these
coordinates go negative. What they need instead is location–scale rescaling against the frozen
reference, which is what they get.

The exceptions prove the rule. ``rsig`` transforms exactly where the quantity stops being a
coordinate: the block **norms** (:math:`\lVert\Phi\rVert`, Rao dispersion) are non-negative,
right-skewed magnitudes on :math:`[0,\infty)`, so they take ``log1p``; ``band`` is a genuine closed
composition, so it takes ``clr``.

One block breaks the pattern deliberately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``contrast`` — :math:`\Psi = \mathrm{mass}\cdot(\Phi - \mathrm{naive})` — is flagged
``magnitude=True``: it is divided by **one frozen scalar RMS for the whole block, and is never
centred**. Per-column z-scoring would force every coordinate to unit variance, which makes a sample
sitting near zero — an immune desert, a repertoire that has barely moved from naive — look
identical to a typical one. How far a repertoire is from naive *is* what that block exists to
carry, so rescaling it away would delete precisely the feature. This is the opposite policy from
every other column in the matrix, and it is a property of the block, not of the sample.

How many components should you keep
------------------------------------

Not "enough for 90% of the variance". In a repertoire matrix the leading variance is sequencing
depth, batch and V-gene usage, so a variance-ranked criterion ranks nuisance first.

Measured on the emitted signature matrix — 14,553 samples × 1,369 columns across 182 studies,
robust median/MAD scaling with clipping (``benchmark_signature_dimension.py``):

.. list-table::
   :header-rows: 1
   :widths: 46 14 40

   * - criterion
     - components
     - what it actually measures
   * - 90% cumulative variance
     - 394
     - how much of *this* corpus you reproduce
   * - Horn parallel analysis
     - 241
     - how many exceed a column-permuted null
   * - participation ratio (effective rank)
     - 144
     - how spread the eigenvalue mass is
   * - per-component :math:`|r| \ge 0.95` across a study-disjoint refit
     - **1**
     - which individual axes are identified at all
   * - per-component :math:`|r| \ge 0.90`
     - 1
     -

The gap between 394 and 1 is the finding, not a contradiction. Split-half correlation per component
was 0.949 for PC1, 0.614 for PC2 and 0.15–0.32 from PC3 onward, while eigenvalues 2–12 sit at
73, 56, 51, 48, 44, 42, 38, 34, 32, 31, 30 — near-degenerate. Components of nearly equal eigenvalue
**swap order** between two refits, so a per-component correlation punishes a labelling artifact
rather than a stability failure. That is why the same script also reports the rotation-invariant
subspace overlap :math:`\lVert V_a V_b^\top\rVert_F^2 / k`: the *subspace* can be stable where the
*axes* are not.

So what number do you actually use? Ask the only criterion that knows what the components are
*for*. Mean AUC over the four largest tasks (2,199–8,016 samples, 21–70 studies), study-disjoint
folds, rotation refit inside every fold:

.. list-table::
   :header-rows: 1
   :widths: 14 14 14 14 14 15 15

   * - components
     - 8
     - 16
     - 64
     - 256
     - 512
     - all 1,369
   * - mean AUC
     - 0.575
     - 0.581
     - 0.589
     - **0.591**
     - 0.584
     - 0.555

The curve is flat from 16 to 256 and then falls off a cliff: **the full 1,369-column matrix scores
worse than 16 components**. That is the curse of dimensionality, located. Per task the effect is
large — ``l3_covid`` reads 0.727 at :math:`k = 64` against 0.582 on all columns; ``l1_infection``
0.659 against 0.577.

**The recommendation: 16–64 components.** 16 buys 98% of the achievable AUC at a quarter of the
width; 64 is the plateau; beyond 256 you are paying for noise. Keep the full matrix only when the
learner is regularised for it (L1, gradient boosting) or when you are hunting a rare, sparse signal
that a rotation would average away.

Practical rules that follow:

- **Never interpret an individual PC beyond the first** as though it were a named feature. It is a
  coordinate of the corpus that fitted it.
- **Select rank by out-of-study reproducibility**, not by explained variance — and prefer the
  subspace-overlap criterion to the per-component one, which fails on degeneracy alone.
- **Refit the rotation inside every cross-validation fold.** Fitting it once on the whole task and
  cross-validating only the classifier lets the components see the test studies' covariance.
- **Report a permutation null for anything chosen by looking at the labels.** A maximum over 64
  components reached AUC 0.84 by chance on a 26-vs-7 contrast (p = 0.20).
- For a **rare, discriminative** signal, do not project at all — keep sparse columns and an L1
  model. The SVD optimises for variance, and a motif carried by a handful of donors has none.

This is also why no corpus-fitted rotation ships in the artifact: the ``phiv`` / ``phij`` / ``phic``
bases come from the prototype cloud, which involves zero samples and so has no corpus to be
unstable with respect to.

Holes are never zeros
---------------------

A locus that was not sequenced, a compartment with too few clonotypes to be a compartment, a
statistic the sample is too shallow to estimate — each yields ``nan`` and a ``mask:`` column. A
model that reads "absent" as "zero" reads an unsequenced chain as a biological finding. Most
learners take ``nan`` natively; those that do not should impute *and* keep the mask.

The signature filters for you — do not pre-filter
-------------------------------------------------

:func:`~mir.signature.signature` sanitises before it embeds, and ``sanitise=True`` is the default.
Leave it there. The reason is specific to the geometry: **a stop codon does not raise in the
distance code**. ``*`` is in seqtree's alphabet, so an unfiltered frame used to return a finite,
meaningless distance and contaminate ``Φ`` silently — which is strictly worse than crashing.

Since 3.12.0 :meth:`~mir.embedding.tcremp.TCREmp.embed` refuses it instead:

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - junction
     - default
     - ``allow_nonstandard=True``
   * - ``null``
     - raises
     - raises
   * - outside ``[ACDEFGHIKLMNPQRSTVWY*_]``
     - raises — a **corrupt table**
     - raises
   * - ``_`` (out-of-frame marker)
     - raises — crashes ``gapblock``
     - raises
   * - ``*`` (stop codon)
     - raises
     - embedded

``allow_nonstandard`` covers stop codons and nothing else. A guard against a crash, and a guard
against a damaged file, cannot be switched off; only the guard against a silently-wrong number has
an opt-out, and taking it has to be written down. Measured over 6,047,716 rows of real clinical
AIRR: zero corrupt characters and zero ``_``, so the strict default costs nothing on well-formed
data.

Neither predicate filters on **length** — a two-residue junction and a sixty-residue one both pass.
The question asked is only whether the string is a plain amino-acid string.

**What pre-filtering actually changes.** Exactly one column per locus,
``vsig:qc:<locus>:nonstd_aa_frac``, because ``sanitise`` reports the weight fraction it dropped and
a pre-filtered frame has nothing left to drop. Measured on 1,168 blood samples from a clinical AIRR cohort at
``tier="standard"``: of 688 columns, **7 move** and 681 are bit-identical — including all 528
``rsig`` geometry columns, which do not move because ``rsig`` is handed sanitised frames either
way. Under ``compact``, ``transfer`` or ``classify`` — every ``recommended`` preset — **zero
columns move**, because they drop the ``qc`` block. If you want the column honest on a pre-filtered
corpus, pass ``signature(..., prefiltered=True)`` and it reports a hole rather than a floor.

Tiers
-----

``core`` ⊂ ``standard`` ⊂ ``full``, as exact **index subsets** of one frozen layout — so a narrower
tier is a slice of a wider one and never a differently-computed number.

.. code-block:: python

   from mir.signature import columns, describe

   len(columns("core")), len(columns("standard")), len(columns("full"))
   describe("standard")     # column, sig, block, locus, feature, tier, transform, flags

What is fitted on data, and what is not
---------------------------------------

Two artifacts ship, and the split is the design.

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - the geometry artifact
     - the scale artifact
   * - holds
     - slot rotations, prototype-cloud location and scale, the naive reference
     - per-column location and scale, the measured ``cstar`` and ``pgen_q05``
   * - fitted on
     - **nothing** — bundled resources only
     - a reference corpus draw
   * - re-fit risk
     - none; the rebuild is bit-identical
     - low; a median and a MAD are identified at any *n*

Fitting a **scale** and fitting a **basis** are different statistical problems, and only one of
them is safe at the sample sizes anyone actually has. A rotation over :math:`p = 256` coordinates
per slot is not column-identified at the sample sizes anyone has — measured split-half column
agreement of a fitted junction basis is 0.23 — whereas a per-column median and MAD converge as
:math:`1/\sqrt{n}`. So the rotation is taken from the **prototype cloud** instead: bundled
receptors embedded against bundled receptors, zero samples, nothing to re-fit and nothing of any
corpus in it.

More data does not change that answer, which is the part worth stating plainly. Refitting the
rotation *inside* study-disjoint folds over **14,553 samples across 182 studies**, not one component
of 1,369 reproduces at :math:`|r| \ge 0.95` and no subspace reaches an overlap of 0.80; per-component
split-half agreement is 0.949 for PC1, 0.614 for PC2 and **0.11–0.32 for PC3–PC12**. The cause is
the spectrum, not the sample count — eigenvalues run 182, 73, 56, 51, 48, 44, 42, 38, 34, … so from
PC2 on the components are near-degenerate. A degenerate pair has a determined *plane* and an
undetermined labelling of the two axes inside it, at any :math:`n`. Task performance agrees: AUC is
flat from :math:`k = 16` to 256 and *falls* when all 1,369 columns are used.

.. _which-scale-reference:

Which scale reference your samples need
---------------------------------------

The geometry is one artifact for everybody — it covers all seven loci and no assay enters it. The
**scale** is not: it is fitted on repertoires, and repertoires from a targeted TCR library and from
bulk RNA-seq do not live on the same scale. Reading a sample against the wrong reference is not a
small error.

The reference that ships today was fitted on **seven targeted (amplicon) TCR cohorts, 4,080
samples**, and that has two consequences a user should know before trusting a column:

.. list-table::
   :header-rows: 1
   :widths: 12 16 16 16

   * - locus
     - columns
     - with a fitted scale
     - ``cstar``
   * - TRA
     - 198
     - 197
     - 0.545
   * - TRB
     - 198
     - 197
     - 0.408
   * - TRG / TRD
     - 198 each
     - **2 each**
     - —
   * - IGH / IGK / IGL
     - 209 / 198 / 198
     - **2 each**
     - —

The two scaled columns on the five uncovered loci are only ``mask:present`` and ``mask:estimable``
— the hole indicators. **No real content is standardised outside TRA and TRB**, and because a locus
with no ``cstar`` falls back to a coverage level no finite sample attains, its ``div:`` columns come
back ``nan``. If you are working with B cells today, that is why.

Why not simply pool one reference over everything: the coverage level ``cstar`` differs by **3.2×**
between assays on the same locus — TRB sits at 0.408 in an amplicon corpus and 0.126 in bulk blood
RNA-seq. ``cstar`` is the depth every Hill number is compared at, and a value above what a sample
attains puts it into extrapolation, which is measured to inflate diversity roughly tenfold. Averaging
the two assays would put *both* populations in the wrong regime.

It lands hardest on the six cross-locus columns — ``pair:-:log_IGH_TRB``, ``log_TRG_TRB``,
``log_TRD_TRB``, ``log_TRA_TRB``, ``log_IGK_IGL`` and ``qc:-:n_loci_present``. A ratio between a
locus measured one way and a locus measured another is a statement about assay, not about biology.
For the same reason a reference must be fitted on samples where every locus came from the **same
library**; loci drawn from different samples cannot produce these columns at all.

So a reference is chosen by assay, not by preference:

.. list-table::
   :header-rows: 1
   :widths: 16 22 14 26

   * - your data
     - reference
     - loci
     - notes
   * - targeted / amplicon TCR
     - the shipped amplicon fit
     - TRA, TRB
     - deep; singleton and rare-clone bands are meaningful
   * - bulk **blood** RNA-seq
     - blood reference
     - all 7
     - shallow per locus; γδ is thin and its bands are depth-fragile
   * - bulk **tissue** RNA-seq
     - tissue reference
     - all 7
     - shallower again — a third of tissue TRB samples carry under 10 clonotypes

Pass a reference explicitly with ``--scale`` (or ``load_scale(...)``). A path that does not exist
**raises** rather than returning ``None``: returning ``None`` would conflate "you did not ask for a
reference" with "the one you named is missing", and a typo would hand you an unstandardised matrix
that looks exactly like a standardised one.

Batch is the thing to check first
----------------------------------

A frozen reference removes the *scaling* difference between two cohorts. It does not remove a
batch effect, and nothing in this vector should be read as if it did — sequencing protocol, depth
and sample handling all move real columns. Use the ``depth:`` columns as covariates, and check a
batch label before believing a between-cohort contrast.

How strong is that warning? On a clonal-density read-out, the flagged fraction fell from **19.4% to
0.9%** when the background was drawn from within the same study instead of across studies. Nearly
all of it was batch.

Regenerating these numbers
--------------------------

Every measured table on this page comes from a script in the companion
`2026-mirpy-analysis <https://github.com/antigenomics>`_ repo, so it can be re-measured rather than
believed:

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - script
     - what it re-measures
   * - ``benchmark_signature_dimension.py``
     - variance / Horn / effective rank / split-half rank criteria, both scaling arms
   * - ``benchmark_signature_scale_convergence.py``
     - how many samples a frozen median and MAD need
   * - ``benchmark_signature_rotation.py``
     - prototype-cloud rotation vs a corpus-fitted one
   * - ``benchmark_density_ankspond.py``
     - the within-study vs cross-study background comparison

*Measurements on this page were last taken 2026-08-13.*

Feature presets — pick by intent, not by column
-----------------------------------------------

The signature is over 1,400 columns. Almost nobody wants all of them, and which subset is right
depends on the question — a model that must run on another lab's samples wants different columns
from one scoring samples inside a single study. :mod:`vdjtools.signature.presets` names those
choices, documents each, and **ranks** it:

**recommended**
   Use this unless you have a reason not to.

*specific*
   Correct for a stated purpose and wrong outside it.

``avoid``
   A control, a baseline, or a measured dead end. Named so that choosing it is deliberate.

.. list-table::
   :header-rows: 1
   :widths: 14 14 10 62

   * - preset
     - rank
     - columns
     - what it is
   * - ``compact``
     - **recommended**
     - 152
     - The smallest vector that still describes a repertoire. Start here.
   * - ``transfer``
     - **recommended**
     - 550
     - For models that must work on another lab's samples. Drops the columns whose level moves most between studies.
   * - ``classify``
     - **recommended**
     - 615
     - The general-purpose set. Best measured task performance when train and test come from comparable cohorts.
   * - ``statistics``
     - *specific*
     - 101
     - Classical repertoire statistics only. Needs no embedding, so vdjtools alone suffices.
   * - ``bcell``
     - *specific*
     - 286
     - B-cell receptor work: the immunoglobulin loci with somatic hypermutation and isotype.
   * - ``geometry``
     - *specific*
     - 514
     - Embedding coordinates only — no count statistics at all.
   * - ``full``
     - *specific*
     - 1403
     - Every contract column. For feature selection, not for fitting.
   * - ``nuisance``
     - ``avoid``
     - 73
     - Sequencing protocol only. A control, not a feature set.

Every preset resolves to a column list from the frozen layout alone — block names, loci, tier. No
corpus, no fitted artifact and no private data is involved, so two people selecting the same preset
get the same columns in the same order.

.. code-block:: bash

   mir presets                      # the table above
   mir presets transfer             # one preset in full: features, how, use cases, caveats
   mir signature *.tsv --preset transfer --describe    # the exact columns it selects

.. code-block:: python

   from vdjtools.signature import presets

   presets.get("transfer").rank        # 'recommended'
   cols = presets.columns("compact")   # a concrete, ordered column list
   presets.table()                     # the whole registry as a DataFrame

Where the rankings come from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A benchmark over a public multi-study AIRR corpus — several hundred study groups, tens of thousands
of samples — scored with **study-disjoint folds**: fit on some studies, predict on studies the fit
never saw. Under that split a column that merely encodes sequencing protocol scores at chance, which
is the point. Three findings shaped the presets:

* **A nuisance floor of depth + presence masks + call quality is a surprisingly strong predictor**
  on many contrasts. Any feature set worth using has to beat its own floor, which is why
  ``nuisance`` ships as a named control rather than being hidden.
* **Projection did not help.** Plain robust or ``asinh`` scaling beat PCA at every rank tested, so
  no preset projects by default and ``full`` is documented as a feature-selection tool rather than a
  model input.
* **The two halves have opposite nuisance profiles.** The embedding geometry carries several times
  less study-to-study variance than the count statistics and the most donor-to-donor variance, and
  is nearly unaffected by whether a sample is blood or tissue — but wins fewer supervised tasks
  outright. Hence ``transfer`` and ``geometry`` for robustness, ``classify`` and ``statistics`` for
  raw accuracy.

Anyone with a comparable SRA/AIRR corpus can reproduce this; none of it depends on a private
dataset.

API
---

.. automodule:: mir.signature
   :members: signature, signature_cohort, rsig, columns, describe, load_reference, self_test
   :undoc-members:
   :show-inheritance:

``mir.signature.blocks``
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mir.signature.blocks
   :members:
   :undoc-members:
   :show-inheritance:

``mir.signature.reference``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mir.signature.reference
   :members:
   :undoc-members:
   :show-inheritance:

``mir.signature.scale``
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mir.signature.scale
   :members:
   :undoc-members:
   :show-inheritance:
