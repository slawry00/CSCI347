import unittest
import math
import numpy as np
from Project2_funcs import *

small_e = make_edgelist("test_file.csv")
print(small_e)
big_e = make_edgelist("proj2_data/musae_git_edges.csv")

class TestEdgeList(unittest.TestCase):
    def test_symmetric(self):
        self.assertTrue(1 in small_e[5])
        self.assertTrue(5 in small_e[1])

class TestGetDegree(unittest.TestCase):
    def test_degree_simple(self):
        self.assertEqual(get_degree(small_e, 5), 3)
    def test_degree_more(self):
        self.assertEqual(get_degree(big_e, 831), 15)



# Just to view the template, I copied some in here
#class TestMyVar(unittest.TestCase):
#    def test_my_var(self):
#        arr1 = np.array([5,3,10])
#        ans = 13
#        self.assertEqual(my_var(arr1), ans)
#
#class TestMeans(unittest.TestCase):
#    def test_means_basic(self):
#        my_arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
#        ans = np.array([2,5,8])
#        np.testing.assert_array_equal(col_means(my_arr), ans)
#    def test_means_nans(self):
#        my_arr = np.array([[None,None,None],[4,5,6],[None,8,None]])
#        ans = np.array([np.nan,5,8])
#        np.testing.assert_array_equal(col_means(my_arr), ans)
#    def test_one_dim(self):
#        my_arr = np.array([[1,2,3]])
#        ans = np.array([2])
#        #print(col_means(my_arr))
#        np.testing.assert_array_equal(col_means(my_arr), ans)


if __name__ == '__main__':
    unittest.main()
