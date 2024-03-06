from collections import defaultdict

from mir.common.clonotype import ClonotypeAA


class ClonotypeDataset:
    """
    The dataset of clonotypes is stored in this object. To be used to store a set of clonotypes with specific features
    """
    def __init__(self, clonotypes: list[ClonotypeAA]):
        for clonotype in clonotypes:
            if not isinstance(clonotype, ClonotypeAA):
                raise Exception('You must have cdr3aa sequence for ClonotypeDataset creation')
        self.clonotypes = {x.cdr3aa: x for x in clonotypes}
        self.masks = None
        self.masks_to_clonotypes = defaultdict(set)

    def get_number_of_samples_for_clonotype(self, clonotype_seqaa):
        pass

    def get_masked_clonotypes_set(self):
        if self.masks is not None:
            return self.masks
        self.masks = set()
        for c in self.clonotypes.keys():
            for i in range(len(c)):
                self.masks.add(c[:i] + 'X' + c[i + 1:])
                self.masks_to_clonotypes[c[:i] + 'X' + c[i + 1:]].add(c)
        return self.masks

    def get_matching_clonotypes(self, clonotype_of_interest: str):
        found_matches = set()
        if self.masks is None:
            self.get_masked_clonotypes_set()
        for i in range(len(clonotype_of_interest)):
            current_mask = clonotype_of_interest[:i] + 'X' + clonotype_of_interest[i + 1:]
            if current_mask in self.masks:
                found_matches = found_matches.union(self.masks_to_clonotypes[current_mask])
        return found_matches