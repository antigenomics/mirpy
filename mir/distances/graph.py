import numpy as np
import igraph as ig
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import textdistance
#pip install -e git+https://github.com/life4/textdistance.git#egg=textdistance
#pip install "textdistance[Hamming]"

class HammingDistances:
    def __init__(self, 
                 seqs : list[str]):
        seqs = np.array(seqs).astype("str")
        # todo: optimize
        dm = squareform(pdist(seqs.reshape(-1, 1), metric = lambda x, y: textdistance.hamming(x[0], y[0])))
        self.__dmf = pd.DataFrame(dm, index=seqs, columns=seqs).stack().reset_index()
        self.__dmf.columns = ['id1', 'id2', 'distance']

    def get_edges(self, threshold = 1) -> tuple[str, str]:
        return self.__dmf[self.__dmf['distance'] <= threshold].apply(tuple, axis=1).tolist()


class SequenceGraph:
    def __init__(self, 
                 edges : list[tuple[str, str, float]]):        
        self.graph = ig.Graph.TupleList(edges, weights=True)
        
    def find_clusters(self):
        self.clusters = self.graph.components()        

    def get_seqs(self):
        return self.graph.vs()['name']
        
    def do_layout(self):
        self.layout = self.graph.layout('graphopt')

    def get_cluster_ids(self):
        return self.clusters.membership
    
    def get_coords(self):
        return np.array(self.layout.coords)

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