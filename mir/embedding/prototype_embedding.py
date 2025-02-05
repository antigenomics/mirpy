from multiprocessing import Pool

from mir.common.repertoire import Repertoire
from mir.common.clonotype import ClonotypeAA, PairedChainClone
from mir.distances.aligner import ClonotypeAligner
from mir.embedding.repertoire_embedding import Embedding
from tqdm import tqdm

from enum import Enum


class Metrics(Enum):
    SIMILARITY = 'similarity'
    DISSIMILARITY = 'dissimilarity'


class PrototypeEmbedding(Embedding):
    def __init__(self, prototype_repertoire: Repertoire, aligner: ClonotypeAligner = ClonotypeAligner.from_library(),
                 metrics=Metrics.SIMILARITY):
        super().__init__()
        self.prototype_repertoire = prototype_repertoire
        self.embedding_type = metrics
        self.aligner = aligner

    def embed_clonotype(self, clonotype: ClonotypeAA | PairedChainClone):
        embedding = []

        if isinstance(clonotype, ClonotypeAA):
            for anchor in self.prototype_repertoire:
                if self.embedding_type == Metrics.SIMILARITY:
                    embedding.append(self.aligner.score(anchor, clonotype))
                else:
                    embedding.append(self.aligner.score_dist(anchor, clonotype))

        elif isinstance(clonotype, PairedChainClone):
            for anchor in self.prototype_repertoire:
                if self.embedding_type == Metrics.SIMILARITY:
                    embedding.append(self.aligner.score_paired(anchor, clonotype))
                else:
                    embedding.append(self.aligner.score_dist_paired(anchor, clonotype))

        return embedding

    def embed_repertoire(self, repertoire: Repertoire, threads: int = 32, flatten_scores=False):
        with Pool(threads) as p:
            repertoire_embeddings = list(
                tqdm(p.imap(self.embed_clonotype, repertoire.clonotypes), total=repertoire.total))

        if flatten_scores:
            repertoire_embeddings = [[item for proto_score in clone_emb for item in proto_score.get_flatten_score()] for
                                     clone_emb in repertoire_embeddings]

        return repertoire_embeddings
