from multiprocessing import Pool

from mir.common.repertoire import Repertoire
from mir.common.clonotype import ClonotypeAA, PairedChainClone
from mir.distances.aligner import ClonotypeAligner
from mir.embedding.repertoire_embedding import Embedding


class PrototypeEmbedding(Embedding):
    def __init__(self, prototype_repertoire: Repertoire, aligner: ClonotypeAligner = ClonotypeAligner.from_library()):
        super().__init__()
        self.prototype_repertoire = prototype_repertoire
        self.aligner = aligner

    def embed_clonotype(self, clonotype: ClonotypeAA | PairedChainClone):
        embedding = []
        try:
            if isinstance(clonotype, ClonotypeAA):
                for anchor in self.prototype_repertoire:
                    embedding.append(self.aligner.score(anchor, clonotype))
            elif isinstance(clonotype, PairedChainClone):
                for anchor in self.prototype_repertoire:
                    embedding.append(self.aligner.score_paired(anchor, clonotype))
        except Exception as e:
            print(clonotype, e)
        return embedding

    def embed_repertoire(self, repertoire: Repertoire, threads: int = 32, flatten_scores=False):
        with Pool(threads) as p:
            repertoire_embeddings = p.map(self.embed_clonotype, repertoire.clonotypes)
        if flatten_scores:
            repertoire_embeddings = [[item for proto_score in clone_emb for item in proto_score.get_flatten_score()] for
                                     clone_emb in repertoire_embeddings]

        return repertoire_embeddings
