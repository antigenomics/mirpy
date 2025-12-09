from multiprocessing import Pool
import math

import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen
from olga.utils import nt2aa

from mir import get_resource_path


class OlgaModel:
    """A class to work with OLGA model for
    generation probability inference. You can generate repertoires using this class or identify
    the probability of any clone to be assembled."""

    def __init__(self, model: str = get_resource_path('olga/default_models/human_T_beta'), is_d_present=True):
        """
        A function which creates the model

        :param model: a path to the directory where the OLGA model is stored
        """
        self.is_d_present = is_d_present

        # Define the files for loading in generative model/data
        params_file_name = f'{model}/model_params.txt'
        marginals_file_name = f'{model}/model_marginals.txt'
        V_anchor_pos_file = f'{model}/V_gene_CDR3_anchors.csv'
        J_anchor_pos_file = f'{model}/J_gene_CDR3_anchors.csv'
        # Load data
        if is_d_present:
            genomic_data = load_model.GenomicDataVDJ()
            generative_model = load_model.GenerativeModelVDJ()
        else:
            genomic_data = load_model.GenomicDataVJ()
            generative_model = load_model.GenerativeModelVJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        generative_model.load_and_process_igor_model(marginals_file_name)
        self.v_names = [x[0] for x in genomic_data.__dict__['genV']]
        self.j_names = [x[0] for x in genomic_data.__dict__['genJ']]
        if is_d_present:
            self.pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
        else:
            self.pgen_model = pgen.GenerationProbabilityVJ(generative_model, genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVJ(generative_model, genomic_data)

    def compute_pgen_cdr3nt(self, cdr3nt: str):
        """
        A function to compute the TCR generation probability for the nucleotide sequence

        :param cdr3nt: a nucleotide sequence string
        :return:
        """
        return self.pgen_model.compute_nt_CDR3_pgen(cdr3nt)

    def compute_pgen_cdr3aa(self, cdr3aa: str):
        """
        A function to compute the TCR generation probability for the amino acid sequence

        :param cdr3aa: an amino acid sequence string
        :return:
        """
        return self.pgen_model.compute_aa_CDR3_pgen(cdr3aa)

    def compute_pgen_cdr3aa_1mm(self, cdr3aa: str):
        """
        A function to compute the TCR generation probability for the amino acid sequence allowing to have \
        one amino acid change

        :param cdr3aa: an amino acid sequence string
        :return: the probability to generate cdr3aa allowing for 1 amino acid mismatch
        """

        cdr3_length = len(cdr3aa)
        pgen_exact = self.compute_pgen_cdr3aa(cdr3aa)
        # with Pool(threads) as p:
        probas = map(self.pgen_model.compute_regex_CDR3_template_pgen,
                           [cdr3aa[:i] + 'X' + cdr3aa[i + 1:] for i in range(cdr3_length)])
        sum_pgen_1mm = sum(probas)
        return sum_pgen_1mm - pgen_exact * (cdr3_length - 1)

    def generate_sequences(self, n: int = 1000) -> list[str]:
        """
        generates `n` random CDR3 sequences according to given model
        :param n:
        :return: list of generates CDR3 sequences
        """
        res = []
        for i in range(n):
            res.append(self.seq_gen_model.gen_rnd_prod_CDR3()[1])
        return res

    def _gen_one_vdj_with_meta(self, conserved_J_residues: str = "FVW") -> dict:
        """
        Generate one productive VDJ CDR3 with meta:
        cdr3 (aa), cdr3_nt, v_gene, j_gene, v_end, j_start
        """
        sg = self.seq_gen_model

        while True:
            recomb_events = sg.choose_random_recomb_events()

            V_seq_full = sg.cutV_genomic_CDR3_segs[recomb_events["V"]]
            if len(V_seq_full) <= max(recomb_events["delV"], 0):
                continue

            D_seq_full = sg.cutD_genomic_CDR3_segs[recomb_events["D"]]
            J_seq_full = sg.cutJ_genomic_CDR3_segs[recomb_events["J"]]

            if len(D_seq_full) < (
                recomb_events["delDl"] + recomb_events["delDr"]
            ):
                continue
            if len(J_seq_full) < recomb_events["delJ"]:
                continue

            V_seq = V_seq_full[: len(V_seq_full) - recomb_events["delV"]]
            D_seq = D_seq_full[
                recomb_events["delDl"] : len(D_seq_full) - recomb_events["delDr"]
            ]
            J_seq = J_seq_full[recomb_events["delJ"] :]

            if (
                len(V_seq)
                + len(D_seq)
                + len(J_seq)
                + recomb_events["insVD"]
                + recomb_events["insDJ"]
            ) % 3 != 0:
                continue

            insVD_seq = seq_gen.rnd_ins_seq(
                recomb_events["insVD"], sg.C_Rvd, sg.C_first_nt_bias_insVD
            )
            insDJ_seq = seq_gen.rnd_ins_seq(
                recomb_events["insDJ"], sg.C_Rdj, sg.C_first_nt_bias_insDJ
            )[::-1]

            ntseq = V_seq + insVD_seq + D_seq + insDJ_seq + J_seq
            aaseq = nt2aa(ntseq)

            if "*" in aaseq:
                continue
            if aaseq[0] != "C" or aaseq[-1] not in conserved_J_residues:
                continue

            L_V_nt = len(V_seq)
            L_J_nt = len(J_seq)
            L_aa = len(aaseq)

            V_len_aa = L_V_nt // 3
            J_len_aa = L_J_nt // 3

            v_end = V_len_aa
            j_start = L_aa - J_len_aa + 1

            v_name = self.v_names[recomb_events["V"]]
            j_name = self.j_names[recomb_events["J"]]

            return {
                "cdr3": aaseq,
                "cdr3_nt": ntseq,
                "v_gene": v_name,
                "j_gene": j_name,
                "v_end": v_end,
                "j_start": j_start,
            }

    def _gen_one_vj_with_meta(self, conserved_J_residues: str = "FVW") -> dict:
        """
        Generate one productive VJ CDR3 with meta:
        cdr3 (aa), cdr3_nt, v_gene, j_gene, v_end, j_start
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

            if (
                len(V_seq)
                + len(J_seq)
                + recomb_events["insVJ"]
            ) % 3 != 0:
                continue

            insVJ_seq = seq_gen.rnd_ins_seq(
                recomb_events["insVJ"], sg.C_Rvj, sg.C_first_nt_bias_insVJ
            )

            ntseq = V_seq + insVJ_seq + J_seq
            aaseq = nt2aa(ntseq)

            if "*" in aaseq:
                continue
            if aaseq[0] != "C" or aaseq[-1] not in conserved_J_residues:
                continue

            L_V_nt = len(V_seq)
            L_J_nt = len(J_seq)
            L_aa = len(aaseq)

            V_len_aa = L_V_nt // 3
            J_len_aa = L_J_nt // 3

            v_end = V_len_aa
            j_start = L_aa - J_len_aa + 1

            v_name = self.v_names[recomb_events["V"]]
            j_name = self.j_names[recomb_events["J"]]

            return {
                "cdr3": aaseq,
                "cdr3_nt": ntseq,
                "v_gene": v_name,
                "j_gene": j_name,
                "v_end": v_end,
                "j_start": j_start,
            }

    def generate_sequences_with_meta(self, n: int = 1000) -> list[dict]:
        """
        Generate n sequences with meta:
          - cdr3 (aa)
          - cdr3_nt
          - v_gene, j_gene
          - v_end, j_start (AA, 1-based)
          - pgen_raw, pgen (log10)
        """
        res = []
        for _ in range(n):
            if self.is_d_present:
                rec = self._gen_one_vdj_with_meta()
            else:
                rec = self._gen_one_vj_with_meta()

            p_raw = self.compute_pgen_cdr3aa(rec["cdr3"])
            rec["pgen_raw"] = p_raw
            rec["pgen"] = (
                math.log10(p_raw) if (p_raw is not None and p_raw > 0) else float("-inf")
            )
            res.append(rec)
        return res

    # TODO: v usage correction

