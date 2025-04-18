from multiprocessing import Pool

import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen

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

    # TODO: v usage correction

