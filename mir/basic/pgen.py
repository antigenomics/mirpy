from multiprocessing import Pool
import math

import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen

from mir import get_resource_path
from mir.basic.mirseq import translate_bidi, mask_positions

# Maps (locus, species) → OLGA model directory name fragment.
# Available models: human TRA/TRB/TRG/TRD/IGH/IGK/IGL, mouse TRA/TRB.
_LOCUS_TO_OLGA = {
    "TRA": "T_alpha",
    "TRB": "T_beta",
    "TRG": "T_gamma",
    "TRD": "T_delta",
    "IGH": "B_heavy",
    "IGK": "B_kappa",
    "IGL": "B_lambda",
}

# Loci that have a D gene segment in their recombination model.
_D_PRESENT = frozenset({"TRB", "TRD", "IGH"})


class OlgaModel:
    """Interface to an OLGA recombination model for generation-probability
    inference and sequence generation.

    Supports all nine built-in models (human TRA/TRB/TRG/TRD/IGH/IGK/IGL,
    mouse TRA/TRB).  A custom model directory can be provided via ``model``.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        locus: str = "TRB",
        species: str = "human",
        is_d_present: bool | None = None,
    ):
        """
        :param model: path to a directory containing an OLGA model.
            When ``None``, the built-in model for ``locus``/``species`` is used.
        :param locus: receptor locus (TRA, TRB, TRG, TRD, IGH, IGK, IGL).
        :param species: organism (human or mouse).
        :param is_d_present: override D-gene flag; inferred from ``locus`` when ``None``.
        """
        locus_u = locus.upper()
        species_l = species.lower()

        if model is None:
            try:
                olga_name = _LOCUS_TO_OLGA[locus_u]
            except KeyError:
                raise ValueError(
                    f"Unsupported locus={locus!r}. Supported: {sorted(_LOCUS_TO_OLGA)}"
                )
            model = get_resource_path(f"olga/default_models/{species_l}_{olga_name}")

        if is_d_present is None:
            is_d_present = locus_u in _D_PRESENT

        self.is_d_present = is_d_present

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

        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        generative_model.load_and_process_igor_model(marginals_file_name)

        self.v_names = [x[0] for x in genomic_data.__dict__["genV"]]
        self.j_names = [x[0] for x in genomic_data.__dict__["genJ"]]

        if self.is_d_present:
            self.pgen_model    = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
        else:
            self.pgen_model    = pgen.GenerationProbabilityVJ(generative_model, genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVJ(generative_model, genomic_data)

    def compute_pgen_junction(self, junction: str) -> float:
        """Return the generation probability for a nucleotide junction sequence.

        :param junction: nucleotide sequence (CDR3, including conserved C and F/W).
        """
        return self.pgen_model.compute_nt_CDR3_pgen(junction)

    def compute_pgen_junction_aa(self, junction_aa: str) -> float:
        """Return the generation probability for an amino-acid junction sequence.

        :param junction_aa: amino-acid sequence (CDR3).
        """
        return self.pgen_model.compute_aa_CDR3_pgen(junction_aa)

    def compute_pgen_junction_aa_1mm(self, junction_aa: str) -> float:
        """Return the cumulative generation probability allowing one amino-acid mismatch.

        Each position is masked to 'X' in turn using ``mirseq.mask_positions``,
        the regex Pgen is summed, and the exact Pgen is subtracted once per
        non-masked position to correct for the overlap.

        :param junction_aa: amino-acid sequence (CDR3).
        """
        pgen_exact = self.compute_pgen_junction_aa(junction_aa)
        masked_seqs = mask_positions(junction_aa)
        sum_pgen_1mm = sum(map(self.pgen_model.compute_regex_CDR3_template_pgen, masked_seqs))
        return sum_pgen_1mm - pgen_exact * (len(junction_aa) - 1)

    def generate_sequences(self, n: int = 1000) -> list[str]:
        """Generate *n* productive CDR3 amino-acid sequences.

        :param n: number of sequences to generate.
        :return: list of CDR3 amino-acid strings.
        """
        res = []
        for _ in range(n):
            res.append(self.seq_gen_model.gen_rnd_prod_CDR3()[1])
        return res

    def _gen_one_vdj_with_meta(self, conserved_J_residues: str = "FVW") -> dict:
        """Generate one productive VDJ recombination event with full annotation.

        :return: dict with keys junction_aa, junction, v_gene, j_gene, v_end, j_start.
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

        :return: dict with keys junction_aa, junction, v_gene, j_gene, v_end, j_start.
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

    def generate_sequences_with_meta(self, n: int = 1000, pgens: bool = True) -> list[dict]:
        """Generate *n* productive sequences with full annotation.

        Each record contains: junction_aa, junction, v_gene, j_gene, v_end,
        j_start, and (when ``pgens=True``) pgen_raw and pgen (log10).

        :param n: number of sequences to generate.
        :param pgens: whether to compute and attach generation probabilities.
        """
        res = []
        for _ in range(n):
            rec = self._gen_one_vdj_with_meta() if self.is_d_present else self._gen_one_vj_with_meta()
            if pgens:
                p_raw = self.compute_pgen_junction_aa(rec["junction_aa"])
                rec["pgen_raw"] = p_raw
                rec["pgen"] = math.log10(p_raw) if (p_raw is not None and p_raw > 0) else float("-inf")
            res.append(rec)
        return res

    # TODO: v usage correction
