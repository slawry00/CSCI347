import unittest
import math
import numpy as np
from Project2_funcs import *
import networkx as nx

small_e = make_edgelist("test_file.csv")
big_e = make_edgelist("proj2_data/musae_git_edges.csv")

small_nx_e = convert_edgelist_to_nx(small_e)
big_nx_e = convert_edgelist_to_nx(big_e)

#small_nx_data = open('test_file.csv', "r")
#small_nx_data.readline()
#small_nx_e = nx.read_edgelist(small_nx_data, delimiter = ',', nodetype=int)
#
#big_nx_data = open('proj2_data/musae_git_edges.csv', "r")
#big_nx_data.readline()
#big_nx_e = nx.read_edgelist(big_nx_data, delimiter = ',', nodetype=int)
#
class TestEdgeList(unittest.TestCase):
    def test_symmetric(self):
        self.assertTrue(1 in small_e[5])
        self.assertTrue(5 in small_e[1])

class TestGetDegree(unittest.TestCase):
    def test_degree_simple(self):
        self.assertEqual(get_degree(small_e, 5), 3)
    def test_degree_more(self):
        self.assertEqual(get_degree(big_e, 831), nx.degree(big_nx_e, 831))

class TestClustCV(unittest.TestCase):
    def test_clustcv_simple(self):
        self.assertEqual(clust_c_v(small_e, 5), 2.0/3)
    def test_clustcv_more(self):
        self.assertEqual(clust_c_v(big_e, 831), nx.clustering(big_nx_e, 831))

class TestClustCG(unittest.TestCase):
    def test_clustcg_simple(self):
        self.assertEqual(clust_c_g(small_e), nx.average_clustering(small_nx_e))
    def test_clustcv_more(self):
        self.assertEqual(clust_c_g(small_e), nx.average_clustering(small_nx_e))

class TestCloseCent(unittest.TestCase):
    def test_closecent_simple(self):
        self.assertEqual(closeness_centrality(small_e, 5),
                         nx.closeness_centrality(small_nx_e, 5)/(len(small_nx_e)-1))
    def test_closecent_more(self):
        self.assertEqual(closeness_centrality(big_e, 831),
                         nx.closeness_centrality(big_nx_e, 831)/(len(big_nx_e)-1))

class TestBetCent(unittest.TestCase):
    def test_bet_cent_simple(self):
        self.assertEqual(betweenness_centrality(small_e, 5),
                         nx.betweenness_centrality(small_nx_e, normalized=False)[5])
    # cant do big graph. Takes too long
    #def test_bet_cent_more(self):
    #    self.assertEqual(betweenness_centrality(big_e, 831),
    #                     nx.betweenness_centrality(big_nx_e, normalized=False)[831])



class TestAvgShortPath(unittest.TestCase):
    def test_avg_short_path_simple(self):
        self.assertEqual(avg_short_path(small_e),
                         nx.average_shortest_path_length(small_nx_e))
    # cant do big graph. Takes too long
    #def test_avg_short_path_more(self):
    #    self.assertEqual(avg_short_path(big_e),
    #                     nx.average_shortest_path_length(big_nx_e))


if __name__ == '__main__':
    unittest.main()
