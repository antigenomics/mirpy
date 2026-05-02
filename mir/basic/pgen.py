"""OLGA-based generation probability and sequence generation for V(D)J recombination.

Wraps the OLGA library to provide:

- Pgen computation (exact, amino-acid, and 1-mismatch) with an in-process
  LRU cache to speed up repeated queries across mock-generation iterations.
- Productive CDR3 sequence generation with full VDJ annotation.
- Parallel pool generation via :meth:`OlgaModel.generate_pool` — the
  recommended entry-point for VDJbet analysis.  Each record stores
  ``log2_pgen`` (not log10) so downstream code never needs a unit conversion.
- V/J gene-usage adjustment (importance sampling) via
  :class:`PgenGeneUsageAdjustment`.

Parallel strategy
-----------------
Each worker process rebuilds a fresh :class:`OlgaModel` (OLGA models are not
picklable) and seeds NumPy independently.  Pgen is computed inside the worker
to avoid an IPC round-trip.  For n = 1 000 000 sequences on 8 cores this
reduces wall-clock time from ~10 min (single-process) to ~80 s.
"""

from __future__ import annotations

import math
from multiprocessing import Pool
from typing import Iterable

import numpy as np
import olga.generation_probability as pgen
import olga.load_model as load_model
import olga.sequence_generation as seq_gen

from mir import get_resource_path
from mir.basic.aliases import LOCUS_TO_OLGA_SUFFIX
from mir.basic import mirseq as _mirseq

translate_bidi = _mirseq.translate_bidi


def _mask_positions_fallback(seq: str) -> list[str]:
    """Return all single-position X-masked variants of an amino-acid sequence."""
    return [seq[:i] + "X" + seq[i + 1:] for i in range(len(seq))]


mask_positions = getattr(_mirseq, "mask_positions", _mask_positions_fallback)

# Loci that have a D gene segment in their recombination model.
_D_PRESENT: frozenset[str] = frozenset({"TRB", "TRD", "IGH"})


def _split_n(n: int, k: int) -> list[int]:
    """Split *n* into *k* roughly equal positive chunks."""
    q, r = divmod(n, k)
    return [q + (1 if i < r else 0) for i in range(k) if q + (1 if i < r else 0) > 0]


def _generate_chunk(args: tuple) -> list[str]:
    """Worker: generate CDR3 aa sequences (no Pgen) for generate_sequences_parallel."""
    init_kwargs, n, seed = args
    model = OlgaModel(**init_kwargs, seed=None)
    np.random.seed(seed)
    return [model._sample_cdr3_aa() for _ in range(n)]


def _generate_pool_chunk(args: tuple) -> list[dict]:
    """Worker for :meth:`OlgaModel.generate_pool`.

    Rebuilds a fresh model in the child process (OLGA models are not
    picklable), seeds NumPy, generates *n* sequences, computes Pgen for each,
    and returns annotated dicts with ``log2_pgen``.

    Record keys: ``junction_aa``, ``junction``, ``v_gene``, ``j_gene``,
    ``v_end``, ``j_start``, ``log2_pgen``.
    """
    init_kwargs, n, seed = args
    model = OlgaModel(**init_kwargs, seed=None)
    np.random.seed(seed)
    _gen_one = (
        model._gen_one_vdj_with_meta
        if model.is_d_present
        else model._gen_one_vj_with_meta
    )
    result: list[dict] = []
    for _ in range(n):
        rec = _gen_one()
        p_raw = model.compute_pgen_junction_aa(rec["junction_aa"])
        rec["log2_pgen"] = (
            math.log2(p_raw) if (p_raw is not None and p_raw > 0) else float("-inf")
        )
        result.append(rec)
    return result


class OlgaModel:
    """Interface to an OLGA recombination model for generation-probability
    inference and sequence generation.

    Supports all nine built-in models (human TRA/TRB/TRG/TRD/IGH/IGK/IGL,
    mouse TRA/TRB).  A custom model directory can be provided via *model*.

    By default the numpy RNG is seeded with 42 on construction so that
    subsequent generation calls are deterministic.  Pass ``seed=None`` to
    preserve the caller's random state.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        locus: str = "TRB",
        species: str = "human",
        is_d_present: bool | None = None,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            model: Path to a directory containing an OLGA model.
                When ``None``, the built-in model for *locus*/*species* is used.
            locus: Receptor locus (TRA, TRB, TRG, TRD, IGH, IGK, IGL).
            species: Organism (``"human"`` or ``"mouse"``).
            is_d_present: Override D-gene flag; inferred from *locus* when ``None``.
            seed: Seed passed to ``numpy.random.seed`` after model loading.
                ``None`` leaves the numpy RNG state unchanged.
        """
        locus_u = locus.upper()
        species_l = species.lower()

        if model is None:
            try:
                olga_name = LOCUS_TO_OLGA_SUFFIX[locus_u]
            except KeyError:
                raise ValueError(
                    f"Unsupported locus={locus!r}. Supported: {sorted(LOCUS_TO_OLGA_SUFFIX)}"
                )
            model = get_resource_path(f"olga/default_models/{species_l}_{olga_name}")

        if is_d_present is None:
            is_d_present = locus_u in _D_PRESENT

        self.is_d_present: bool = is_d_present

        # Stored for parallel workers, which must reconstruct the model in a
        # fresh process.  seed is excluded — each worker seeds independently.
        self._init_kwargs: dict = {
            "model": model,
            "locus": locus,
            "species": species,
            "is_d_present": is_d_present,
        }

        params_file_name    = f"{model}/model_params.txt"
        marginals_file_name = f"{model}/model_marginals.txt"
        V_anchor_pos_file   = f"{model}/V_gene_CDR3_anchors.csv"
        J_anchor_pos_file   = f"{model}/J_gene_CDR3_anchors.csv"

        if self.is_d_present:
            genomic_data     = load_model.GenomicDataVDJ()
            generative_model = load_model.GenerativeModelVDJ()
        else:
            genomic_data     = load_model.GenomicDataVJ()
            generative_model = load_model.GenerativeModelVJ()

        genomic_data.load_igor_genomic_data(
            params_file_name, V_anchor_pos_file, J_anchor_pos_file
        )
        generative_model.load_and_process_igor_model(marginals_file_name)

        self.v_names: list[str] = [x[0] for x in genomic_data.__dict__["genV"]]
        self.j_names: list[str] = [x[0] for x in genomic_data.__dict__["genJ"]]

        if self.is_d_present:
            self.pgen_model    = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
        else:
            self.pgen_model    = pgen.GenerationProbabilityVJ(generative_model, genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVJ(generative_model, genomic_data)

        if seed is not None:
            np.random.seed(seed)

        # Cache repeated CDR3aa Pgen queries (common in large cohort analyses).
        self._pgen_aa_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Pgen computation
    # ------------------------------------------------------------------

    def compute_pgen_junction(self, junction: str) -> float:
        """Return the generation probability for a nucleotide junction sequence.

        Args:
            junction: Nucleotide sequence (CDR3, including conserved C and F/W).
        """
        return self.pgen_model.compute_nt_CDR3_pgen(junction)

    def compute_pgen_junction_aa(self, junction_aa: str) -> float:
        """Return the generation probability for an amino-acid junction sequence.

        Args:
            junction_aa: Amino-acid sequence (CDR3).
        """
        cached = self._pgen_aa_cache.get(junction_aa)
        if cached is not None:
            return cached
        val = self.pgen_model.compute_aa_CDR3_pgen(junction_aa)
        # Cap cache growth in long-running notebook sessions.
        if len(self._pgen_aa_cache) < 2_000_000:
            self._pgen_aa_cache[junction_aa] = val
        return val

    def compute_pgen_junction_aa_bulk(self, junction_aas: Iterable[str]) -> list[float]:
        """Compute Pgen for many amino-acid junctions with cache reuse.

        Args:
            junction_aas: Iterable of CDR3 amino-acid sequences.

        Returns:
            List of Pgen values in input order.
        """
        return [self.compute_pgen_junction_aa(s) for s in junction_aas]

    def compute_pgen_junction_aa_1mm(self, junction_aa: str) -> float:
        """Return the cumulative generation probability allowing one amino-acid mismatch.

        Each position is masked to 'X' in turn using :func:`~mir.basic.mirseq.mask_positions`,
        the regex Pgen is summed, and the exact Pgen is subtracted once per
        non-masked position to correct for the overlap.

        Args:
            junction_aa: Amino-acid sequence (CDR3).
        """
        pgen_exact = self.compute_pgen_junction_aa(junction_aa)
        masked_seqs = mask_positions(junction_aa)
        sum_pgen_1mm = sum(map(self.pgen_model.compute_regex_CDR3_template_pgen, masked_seqs))
        return sum_pgen_1mm - pgen_exact * (len(junction_aa) - 1)

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def _sample_cdr3_aa(self) -> str:
        """Draw one productive CDR3 amino-acid sequence from the model."""
        # gen_rnd_prod_CDR3 returns (ntseq, aaseq, v_idx, j_idx); index 1 is aa.
        result = self.seq_gen_model.gen_rnd_prod_CDR3()
        assert result is not None
        return result[1]

    def generate_sequences(self, n: int = 1000, seed: int | None = 42) -> list[str]:
        """Generate *n* productive CDR3 amino-acid sequences.

        Args:
            n: Number of sequences to generate.
            seed: Seed for ``numpy.random`` before sampling.  Pass ``None``
                to continue from the current RNG state (e.g. for chained
                calls that should each return a different batch).

        Returns:
            List of CDR3 amino-acid strings.
        """
        if seed is not None:
            np.random.seed(seed)
        return [self._sample_cdr3_aa() for _ in range(n)]

    def generate_sequences_parallel(
        self,
        n: int = 1000,
        n_jobs: int = 4,
        seed: int = 42,
    ) -> list[str]:
        """Generate *n* productive CDR3 sequences using *n_jobs* worker processes.

        Each worker creates a fresh :class:`OlgaModel` and seeds numpy with
        ``seed + worker_index`` to ensure diverse but reproducible output.
        Falls back to single-process generation when *n_jobs* ≤ 1.

        Args:
            n: Total number of sequences to generate.
            n_jobs: Number of parallel worker processes.
            seed: Base seed; worker *i* uses ``seed + i``.

        Returns:
            List of CDR3 amino-acid strings (length == *n*).
        """
        if n_jobs <= 1:
            return self.generate_sequences(n, seed=seed)

        sizes = _split_n(n, n_jobs)
        args = [
            (self._init_kwargs, size, seed + i)
            for i, size in enumerate(sizes)
        ]
        with Pool(n_jobs) as pool:
            chunks = pool.map(_generate_chunk, args)
        return [seq for chunk in chunks for seq in chunk]

    def generate_pool(
        self,
        n: int = 1_000_000,
        n_jobs: int = 4,
        seed: int = 42,
    ) -> list[dict]:
        """Generate *n* annotated sequences with log₂ Pgen, optionally in parallel.

        This is the primary entry-point for building a
        :class:`~mir.biomarkers.vdjbet.PgenBinPool`.  For large *n* (≥ 100 k)
        parallel execution provides near-linear speedup over the number of
        cores because each worker computes both recombination events and Pgen.

        Args:
            n: Total number of sequences to generate.
            n_jobs: Number of parallel worker processes.  Use 1 for
                reproducible single-process execution (identical output for a
                given *seed*).  Use ≥ 4 for large pools.
            seed: Base RNG seed; worker *i* uses ``seed + i``.

        Returns:
            List of dicts with keys:
            ``junction_aa``, ``junction``, ``v_gene``, ``j_gene``,
            ``v_end``, ``j_start``, ``log2_pgen``.
            Records where Pgen is zero or undefined have
            ``log2_pgen = float("-inf")`` and should be filtered by callers.
        """
        if n_jobs <= 1:
            np.random.seed(seed)
            result: list[dict] = []
            _gen_one = (
                self._gen_one_vdj_with_meta
                if self.is_d_present
                else self._gen_one_vj_with_meta
            )
            for _ in range(n):
                rec = _gen_one()
                p_raw = self.compute_pgen_junction_aa(rec["junction_aa"])
                rec["log2_pgen"] = (
                    math.log2(p_raw)
                    if (p_raw is not None and p_raw > 0)
                    else float("-inf")
                )
                result.append(rec)
            return result

        sizes = _split_n(n, n_jobs)
        args = [(self._init_kwargs, size, seed + i) for i, size in enumerate(sizes)]
        with Pool(n_jobs) as pool:
            chunks = pool.map(_generate_pool_chunk, args)
        return [rec for chunk in chunks for rec in chunk]

    # ------------------------------------------------------------------
    # Annotated generation (junction_aa + V/J gene + optional Pgen)
    # ------------------------------------------------------------------

    def _gen_one_vdj_with_meta(self, conserved_J_residues: str = "FVW") -> dict:
        """Generate one productive VDJ recombination event with full annotation.

        Returns:
            Dict with keys: junction_aa, junction, v_gene, j_gene, v_end, j_start.
        """
        sg = self.seq_gen_model

        while True:
            recomb_events = sg.choose_random_recomb_events()

            V_seq_full = sg.cutV_genomic_CDR3_segs[recomb_events["V"]]
            if len(V_seq_full) <= max(recomb_events["delV"], 0):
                continue

            D_seq_full = sg.cutD_genomic_CDR3_segs[recomb_events["D"]]
            J_seq_full = sg.cutJ_genomic_CDR3_segs[recomb_events["J"]]

            if len(D_seq_full) < (recomb_events["delDl"] + recomb_events["delDr"]):
                continue
            if len(J_seq_full) < recomb_events["delJ"]:
                continue

            V_seq = V_seq_full[: len(V_seq_full) - recomb_events["delV"]]
            D_seq = D_seq_full[recomb_events["delDl"] : len(D_seq_full) - recomb_events["delDr"]]
            J_seq = J_seq_full[recomb_events["delJ"] :]

            if (
                len(V_seq) + len(D_seq) + len(J_seq)
                + recomb_events["insVD"] + recomb_events["insDJ"]
            ) % 3 != 0:
                continue

            insVD_seq = seq_gen.rnd_ins_seq(recomb_events["insVD"], sg.C_Rvd, sg.C_first_nt_bias_insVD)
            insDJ_seq = seq_gen.rnd_ins_seq(recomb_events["insDJ"], sg.C_Rdj, sg.C_first_nt_bias_insDJ)[::-1]

            ntseq = V_seq + insVD_seq + D_seq + insDJ_seq + J_seq
            aaseq = translate_bidi(ntseq)

            if "*" in aaseq:
                continue
            if aaseq[0] != "C" or aaseq[-1] not in conserved_J_residues:
                continue

            v_end   = len(V_seq) // 3
            j_start = len(aaseq) - len(J_seq) // 3 + 1

            return {
                "junction_aa": aaseq,
                "junction":    ntseq,
                "v_gene":      self.v_names[recomb_events["V"]],
                "j_gene":      self.j_names[recomb_events["J"]],
                "v_end":       v_end,
                "j_start":     j_start,
            }

    def _gen_one_vj_with_meta(self, conserved_J_residues: str = "FVW") -> dict:
        """Generate one productive VJ recombination event with full annotation.

        Returns:
            Dict with keys: junction_aa, junction, v_gene, j_gene, v_end, j_start.
        """
        sg = self.seq_gen_model

        while True:
            recomb_events = sg.choose_random_recomb_events()

            V_seq_full = sg.cutV_genomic_CDR3_segs[recomb_events["V"]]
            if len(V_seq_full) <= max(recomb_events["delV"], 0):
                continue

            J_seq_full = sg.cutJ_genomic_CDR3_segs[recomb_events["J"]]
            if len(J_seq_full) < recomb_events["delJ"]:
                continue

            V_seq = V_seq_full[: len(V_seq_full) - recomb_events["delV"]]
            J_seq = J_seq_full[recomb_events["delJ"] :]

            if (len(V_seq) + len(J_seq) + recomb_events["insVJ"]) % 3 != 0:
                continue

            insVJ_seq = seq_gen.rnd_ins_seq(recomb_events["insVJ"], sg.C_Rvj, sg.C_first_nt_bias_insVJ)

            ntseq = V_seq + insVJ_seq + J_seq
            aaseq = translate_bidi(ntseq)

            if "*" in aaseq:
                continue
            if aaseq[0] != "C" or aaseq[-1] not in conserved_J_residues:
                continue

            v_end   = len(V_seq) // 3
            j_start = len(aaseq) - len(J_seq) // 3 + 1

            return {
                "junction_aa": aaseq,
                "junction":    ntseq,
                "v_gene":      self.v_names[recomb_events["V"]],
                "j_gene":      self.j_names[recomb_events["J"]],
                "v_end":       v_end,
                "j_start":     j_start,
            }

    def generate_sequences_with_meta(
        self,
        n: int = 1000,
        pgens: bool = True,
        seed: int | None = 42,
        pgen_adjustment=None,
    ) -> list[dict]:
        """Generate *n* productive sequences with full annotation.

        Each record contains: junction_aa, junction, v_gene, j_gene, v_end,
        j_start, and (when *pgens* is ``True``) pgen_raw and pgen (log10).
        When *pgen_adjustment* is provided, ``pgen`` stores the adjusted
        log₁₀ Pgen (raw × V-J factor); ``pgen_raw`` is always the raw value.

        Args:
            n: Number of sequences to generate.
            pgens: Whether to compute and attach generation probabilities.
            seed: Seed for ``numpy.random`` before sampling.  Pass ``None``
                to continue from the current RNG state.
            pgen_adjustment: Optional :class:`PgenGeneUsageAdjustment`.
                When supplied, ``rec["pgen"]`` is multiplied by the V-J factor
                from the adjustment object.

        Returns:
            List of annotation dicts, one per generated sequence.
        """
        if seed is not None:
            np.random.seed(seed)
        _gen_one = self._gen_one_vdj_with_meta if self.is_d_present else self._gen_one_vj_with_meta
        locus = self._init_kwargs.get("locus", "")
        res = []
        for _ in range(n):
            rec = _gen_one()
            if pgens:
                p_raw = self.compute_pgen_junction_aa(rec["junction_aa"])
                rec["pgen_raw"] = p_raw
                if pgen_adjustment is not None and p_raw is not None and p_raw > 0:
                    p_adj = pgen_adjustment.adjust_pgen(locus, rec["v_gene"], rec["j_gene"], p_raw)
                    rec["pgen"] = math.log10(p_adj) if p_adj > 0 else float("-inf")
                else:
                    rec["pgen"] = math.log10(p_raw) if (p_raw is not None and p_raw > 0) else float("-inf")
            res.append(rec)
        return res

    def compute_usage_cache(
        self,
        n: int = 100_000,
        *,
        seed: int = 42,
    ) -> "GeneUsage":
        """Estimate V-J gene usage by a cold run of *n* OLGA samples.

        Returns a :class:`~mir.basic.gene_usage.GeneUsage` that covers all
        V-J pairs generated at frequency ≥ 1/*n*.  Intended for building a
        :class:`PgenGeneUsageAdjustment` for importance-sampling-based mock
        generation.

        Args:
            n: Number of sequences to sample (higher → more accurate).
            seed: numpy RNG seed.

        Returns:
            GeneUsage with clonotype counts per V-J pair for this locus.
        """
        from mir.basic.gene_usage import GeneUsage

        records = self.generate_sequences_with_meta(n, pgens=False, seed=seed)
        locus = self._init_kwargs.get("locus", "")
        gu = GeneUsage()
        locus_data = gu._data.setdefault(locus, {})
        locus_totals = gu._totals.setdefault(locus, [0, 0])
        for rec in records:
            v = rec["v_gene"].split("*")[0]
            j = rec["j_gene"].split("*")[0]
            entry = locus_data.setdefault((v, j), [0, 0])
            entry[0] += 1
            locus_totals[0] += 1
        return gu


# ---------------------------------------------------------------------------
# Pgen gene-usage adjustment (importance sampling)
# ---------------------------------------------------------------------------

class PgenGeneUsageAdjustment:
    """Adjusts OLGA generation probabilities to match a target V-J gene usage.

    For each generated sequence with genes ``(v, j)``, multiplies its Pgen
    by::

        target_vj_fraction(v, j) / olga_vj_fraction(v, j)

    where both fractions use Laplace smoothing (pseudocount = 1 over observed
    pairs).  This re-weights the OLGA distribution so that Pgen-matched mock
    sequences reflect the target V-J gene usage without requiring explicit V/J
    stratification.

    The OLGA gene usage cache is computed lazily per locus on first access.

    Parameters
    ----------
    target:
        :class:`~mir.basic.gene_usage.GeneUsage` describing the target gene
        usage (e.g. computed from a real sample).
    cache_size:
        Number of OLGA sequences used to estimate the model's native gene
        usage.  Higher values give more accurate factors for rare V-J pairs.
    seed:
        RNG seed for the OLGA cache run.

    Examples
    --------
    >>> from mir.basic.gene_usage import GeneUsage
    >>> gu = GeneUsage.from_repertoire(my_sample)
    >>> adj = PgenGeneUsageAdjustment(gu, cache_size=100_000)
    >>> pool = build_olga_pool("TRB", 50_000, pgen_adjustment=adj)
    """

    def __init__(
        self,
        target: "GeneUsage",
        *,
        cache_size: int = 100_000,
        seed: int = 42,
        count: str = "count_rearrangement",
        pseudocount: float = 1.0,
        reference: "GeneUsage | None" = None,
    ) -> None:
        from mir.basic.gene_usage import GeneUsage as _GeneUsage  # local for TYPE_CHECKING compat

        self._target = target
        self._cache_size = cache_size
        self._seed = seed
        self._count = count
        self._pseudocount = pseudocount
        self._olga_cache: dict[str, _GeneUsage] = {}
        if reference is not None:
            for locus in reference.loci:
                self._olga_cache[locus] = reference
        self._factor_cache: dict[str, dict[tuple[str, str], float]] = {}

    def _get_olga_cache(self, locus: str):
        if locus not in self._olga_cache:
            model = OlgaModel(locus=locus, seed=self._seed)
            self._olga_cache[locus] = model.compute_usage_cache(
                self._cache_size, seed=self._seed
            )
        return self._olga_cache[locus]

    def _get_factor_cache(self, locus: str) -> dict[tuple[str, str], float]:
        if locus not in self._factor_cache:
            olga = self._get_olga_cache(locus)
            factors = self._target.correction_factors(
                olga,
                locus,
                scope="vj",
                count=self._count,
                pseudocount=self._pseudocount,
            )
            self._factor_cache[locus] = {
                key: float(val)
                for key, val in factors.items()
                if isinstance(key, tuple) and len(key) == 2
            }
        return self._factor_cache[locus]

    def factor(self, locus: str, v: str, j: str) -> float:
        """Pgen adjustment factor for (locus, v, j).

        Returns ``target_vj_fraction / olga_vj_fraction`` with Laplace
        pseudocount = 1 over observed pairs.

        Args:
            locus: IMGT locus code (e.g. ``"TRB"``).
            v: V-gene name (allele stripped internally).
            j: J-gene name (allele stripped internally).
        """
        v_base = v.split("*")[0]
        j_base = j.split("*")[0]
        pair = (v_base, j_base)
        return self._get_factor_cache(locus).get(pair, 1.0)

    def adjust_pgen(self, locus: str, v: str, j: str, pgen: float) -> float:
        """Return ``pgen * factor(locus, v, j)``.

        Args:
            locus: IMGT locus code.
            v: V-gene name.
            j: J-gene name.
            pgen: Raw generation probability (linear, not log).
        """
        return pgen * self.factor(locus, v, j)
