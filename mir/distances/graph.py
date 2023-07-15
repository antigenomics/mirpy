from multiprocessing import Pool
import numpy as np
import igraph as ig
import textdistance
#pip install -e git+https://github.com/life4/textdistance.git#egg=textdistance
#pip install "textdistance[Hamming]"

class HammingDistances:
    def __init__(self, seqs, seqs2 = None,
                 nproc = 4, chunk_sz = 1024):
        with Pool(nproc) as pool:
            if not seqs2:
                self.distances = pool.starmap(HammingDistances.__wrap_dist, 
                                            ((s1, s2) for s1 in seqs for s2 in seqs2 if s1 > s2), chunk_sz)
            else:
                self.distances = pool.starmap(HammingDistances.__wrap_dist, 
                                            ((s1, s2) for s1 in seqs for s2 in seqs2), chunk_sz)

    @staticmethod
    def __wrap_dist(s1 : str, s2 : str):
        return (s1, s2, textdistance.hamming(s1, s2))

    def get_edges(self, threshold = 1):
        return ((x[0], x[1]) for x in self.distances if x[2] <= threshold)


class SequenceGraph:
    def __init__(self, edges):        
        self.graph = ig.Graph.TupleList(edges)
        self.clusters = self.graph.components()
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