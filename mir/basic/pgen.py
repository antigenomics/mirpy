from multiprocessing import Pool

import olga.load_model as load_model
import olga.generation_probability as pgen

from mir import get_resource_path


class OlgaModel:
    """A class to work with OLGA model for
    generation probability inference. You can generate repertoires using this class or identify
    the probability of any clone to be assembled."""

    def __init__(self, model: str = get_resource_path('olga/default_models/human_T_beta')):
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
        genomic_data = load_model.GenomicDataVDJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        # Load model
        generative_model = load_model.GenerativeModelVDJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        # Process model/data for pgen computation by instantiating GenerationProbabilityVDJ
        self.pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)

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

        :param threads: number of threads to perform parallelizing in
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

    # TODO: v usage correction

    # TODO: generate, -> Iterable[Clonotype]

