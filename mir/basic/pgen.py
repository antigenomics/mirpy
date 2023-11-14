import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen


class OlgaModel:
    def __init__(self, model : str = 'default_models/human_T_beta'):
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

    def compute_pgen_cdr3nt(self, cdr3nt : str):
        return self.pgen_model.compute_nt_CDR3_pgen(cdr3nt)
    
    def compute_pgen_cdr3aa(self, cdr3aa : str):
        return self.pgen_model.compute_aa_CDR3_pgen(cdr3aa)
    
    def compute_pgen_cdr3aa_1mm(self, cdr3aa : str):
        l = len(cdr3aa)
        p0 = self.compute_pgen_cdr3aa(cdr3aa)
        for i in range(l):
            s = cdr3aa[:i] + 'X' + cdr3aa[i + 1:]
            p1 = p1 + self.pgen_model.compute_regex_CDR3_template_pgen(s)
        return p1 - p0 * (l - 1)
    
    #TODO: v usage correction

    #TODO: generate, -> Iterable[Clonotype]