from mir.common.clonotype import ClonotypeAA


class ClonotypeDataset:
    def __init__(self, clonotypes: list[ClonotypeAA]):
        for clonotype in clonotypes:
            if not isinstance(clonotype, ClonotypeAA):
                raise Exception('You must have cdr3aa sequence for ClonotypeDataset creation')
        self.clonotypes = {x.cdr3aa: x for x in clonotypes}

    def get_number_of_samples_for_clonotype(self, clonotype_seqaa):
        pass