import unittest
import math
import numpy as np
from Project2_funcs import *

edgelist = make_edgelist("proj2_data/musae_git_edges.csv")


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
