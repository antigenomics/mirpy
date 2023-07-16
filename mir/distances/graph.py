from multiprocessing import Pool
from itertools import starmap
import igraph as ig
#for fast hamming
#pip install -e git+https://github.com/life4/textdistance.git#egg=textdistance
#pip install "textdistance[Hamming]"
import textdistance


class HammingDistances:
    def __init__(self, seqs, seqs2 = None, indels = False, 
                 nproc = 8, chunk_sz = 4096):
        # https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
        if indels:
            dfun = lambda s1, s2: (s1, s2, textdistance.levenshtein(s1, s2))
        else:
            dfun = lambda s1, s2: (s1, s2, textdistance.hamming(s1, s2))
        if nproc == 1:
            if not seqs2:
                self.distances = starmap(dfun, ((s1, s2) for s1 in seqs for s2 in seqs if s1 > s2))
            else:
                self.distances = starmap(dfun, ((s1, s2) for s1 in seqs for s2 in seqs2))
        else:           
            with Pool(nproc) as pool:
                if not seqs2:
                    self.distances = pool.starmap(dfun, ((s1, s2) for s1 in seqs for s2 in seqs if s1 > s2), 
                                                chunk_sz)
                else:
                    self.distances = pool.starmap(dfun, ((s1, s2) for s1 in seqs for s2 in seqs2), 
                                                chunk_sz)

    def get_edges(self, threshold = 1):
        return ((x[0], x[1]) for x in self.distances if x[2] <= threshold)

# todo: degree
class SequenceGraph:
    def __init__(self, edges):        
        self.graph = ig.Graph.TupleList(edges)
        self.clusters = self.graph.components()
        # todo: optimize
        self.layout = self.graph.layout('graphopt')

    def get_seqs(self):
        return self.graph.vs()['name']

    def get_cluster_ids(self):
        return self.clusters.membership
    
    def get_coords(self):
        return self.layout.coords

        #df_graph = pd.DataFrame(
        #    {'seq': graph.vs()['name'],
        #    'cluster': clusters.membership,
        #    'x': coords[:,0],
        #    'y': coords[:,1]
        #})
        
        # summary
        #df_graph_summary = df_graph.groupby(['cluster']).agg(
        #    cluster_size = ('cluster', 'size'), 
        #    x_mean = ('x', 'mean'), 
        #    y_mean = ('y', 'mean')).reset_index()
        #return pd.merge(df_graph, df_graph_summary)