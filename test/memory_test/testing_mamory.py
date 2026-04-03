import tracemalloc
import psutil
import os
import time
import pandas as pd

import sys
sys.path.append('../../')
from mir.common.segments import SegmentLibrary
from mir.distances.aligner import ClonotypeAligner
from mir.embedding.prototype_embedding import PrototypeEmbedding, Metrics
from mir.common.repertoire import Repertoire
from mir.common.parser import AIRRParser

# ======= Настройки =======
input_path = "/projects/immunestatus/pogorelyy/airr_format/P1_0_F1_with_1.txt"
proto_path = "tcremp_prototypes_olga.tsv"
species = ["HomoSapiens"]
chain = ["TRB"]
locus = "beta"
metric = "dissimilarity"
nproc = 32
llen, hlen = 5, 30
# ==========================

def report_memory(stage):
    process = psutil.Process(os.getpid())
    print(f"[{stage}] RSS: {process.memory_info().rss / 1024**2:.2f} MB")

def validate_cdr3_len(rep, llen, hlen):
    return rep.subsample_by_lambda(lambda x: llen <= len(x.cdr3aa) < hlen)

def load_repertoire(path, lib, locus):
    parser = AIRRParser(lib=lib, locus=locus)
    rep = Repertoire.load(parser=parser, path=path)
    return validate_cdr3_len(rep, llen, hlen)

from pympler import asizeof

def get_obj_size(o, name):
    size_bytes = asizeof.asizeof(o)
    print(f"Total size of {name}: {size_bytes / (1024 ** 2):.2f} MB")

def main():
    tracemalloc.start()

    t0 = time.time()
    print("Loading segment library...")
    lib = SegmentLibrary.load_default(genes=chain, organisms=species)
    report_memory("After segment library load")

    print("Loading repertoires...")
    rep = load_repertoire(input_path, lib, locus)
    proto = load_repertoire(proto_path, lib, locus)
    report_memory("After repertoires loaded")

    print("Creating aligner and embedder...")
    aligner = ClonotypeAligner.from_library(lib=lib)
    embedder = PrototypeEmbedding(proto, aligner=aligner, metrics=Metrics(metric))
    get_obj_size(embedder, 'embedder')
    report_memory("After embedder init")

    print("Running embedding...")
    t1 = time.time()
    embedding = embedder.embed_repertoire(rep, threads=nproc, flatten_scores=True)
    print(f"Embedding took {time.time() - t1:.2f}s")
    get_obj_size(embedder, 'embedder')
    get_obj_size(embedding, 'embedding')
    report_memory("After embedding")

    del embedding
    report_memory("After DataFrame conversion")
    print(f"Total time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
