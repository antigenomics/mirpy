The portable repertoire signature
=================================

One AIRR repertoire in, one **fixed, named, positional** feature vector out — computable by anyone
who ``pip install mirpy-lib``, on their own samples, and directly comparable with yours. That is
the whole design goal: a matrix you can hand a collaborator that drops into PCA, logistic
regression, random forest, boosting or an MLP with no scaler of their own.

.. code-block:: python

   from mir.signature import signature, signature_cohort

   v = signature({"TRB": df})                 # {column: value}, standardised, layout order
   F = signature_cohort(samples)              # one row per sample, positional
   F.write_parquet("cohort.parquet")

.. code-block:: bash

   mir signature cohort/*.tsv.gz -o sig.parquet --tier standard
   mir signature --describe --tier standard          # the column dictionary, reads no input

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

Holes are never zeros
---------------------

A locus that was not sequenced, a compartment with too few clonotypes to be a compartment, a
statistic the sample is too shallow to estimate — each yields ``nan`` and a ``mask:`` column. A
model that reads "absent" as "zero" reads an unsequenced chain as a biological finding. Most
learners take ``nan`` natively; those that do not should impute *and* keep the mask.

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
per slot is not column-identified at a few thousand samples — measured split-half column agreement
of a fitted junction basis is 0.23 — whereas a per-column median and MAD converge as
:math:`1/\sqrt{n}`. So the rotation is taken from the **prototype cloud** instead: bundled
receptors embedded against bundled receptors, zero samples, nothing to re-fit and nothing of any
corpus in it.

Batch is the thing to check first
----------------------------------

A frozen reference removes the *scaling* difference between two cohorts. It does not remove a
batch effect, and nothing in this vector should be read as if it did — sequencing protocol, depth
and sample handling all move real columns. Use the ``depth:`` columns as covariates, and check a
batch label before believing a between-cohort contrast.

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
