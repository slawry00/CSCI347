import unittest
import math
import numpy as np
from Project1_funcs import *
from sklearn.preprocessing import StandardScaler

class TestMyVar(unittest.TestCase):
    def test_my_var(self):
        arr1 = np.array([5,3,10])
        ans = 13
        self.assertEqual(my_var(arr1), ans)

class TestMeans(unittest.TestCase):
    def test_means_basic(self):
        my_arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
        ans = np.array([2,5,8])
        np.testing.assert_array_equal(col_means(my_arr), ans)
    def test_means_nans(self):
        my_arr = np.array([[None,None,None],[4,5,6],[None,8,None]])
        ans = np.array([np.nan,5,8])
        np.testing.assert_array_equal(col_means(my_arr), ans)
    def test_one_dim(self):
        my_arr = np.array([[1,2,3]])
        ans = np.array([2])
        #print(col_means(my_arr))
        np.testing.assert_array_equal(col_means(my_arr), ans)

class TestCovar(unittest.TestCase):
    def test_covar(self):
        arr1 = np.array([1,2,3])
        arr2 = np.array([4,5,6])
        ans = 1
        np.testing.assert_array_equal(covar(arr1,arr2), ans)
    def test_wrong_size(self):
        arr1 = np.array([1,2])
        arr2 = np.array([4,5,6])
        ans = 1
        np.testing.assert_raises(IndexError, covar, arr1, arr2)
    def test_wrong_dims(self):
        arr1 = np.array([[1,2,3],[1,2,3]])
        arr2 = np.array([4,5,6])
        ans = 1
        np.testing.assert_raises(TypeError, covar, arr1, arr2)

class TestCorr(unittest.TestCase):
    def test_corr1(self):
        arr1 = np.array([1,2,3])
        arr2 = np.array([4,5,6])
        ans = 1
        np.testing.assert_array_equal(corr(arr1,arr2), ans)
    def test_corr2(self):
        arr1 = np.array([5,3,10])
        arr2 = np.array([4,8,12])
        ans = 10.0/(4*math.sqrt(13))
        np.testing.assert_array_equal(corr(arr1,arr2), ans)


class TestCovar_mat(unittest.TestCase):
    def test_covar_mat(self):
        my_arr = np.array([[1,2,3],[3,4,5]])
        ans = np.array([[1,1],[1,1]])
        np.testing.assert_array_equal(covar_mat(my_arr), ans)

    def test_covar_mat_prebuilt(self):
        my_arr = np.array([[9,7,13],[13,24,65]])
        ans = np.cov(my_arr)
        np.testing.assert_array_almost_equal(covar_mat(my_arr), ans)


class TestRangeNorm(unittest.TestCase):
    def test_range_norm_basic(self):
        arr1 = np.array([[5,3,10]])
        ans = np.array([[2.0/7,0,1]])
        np.testing.assert_array_equal(range_norm(arr1), ans)
    def test_range_norm_advanced(self):
        my_arr = np.array([[1,2,3,4],[5,6,7,8]])
        ans = np.array([[0,1.0/3,2.0/3,1],[0,1.0/3,2.0/3,1]])
        np.testing.assert_array_equal(range_norm(my_arr), ans)

class TestStandNorm(unittest.TestCase):
    def test_stand_norm_basic(self):
        stdsc = StandardScaler()
        arr1 = np.array([[5.0,3.0,10.0]])
        ans = np.array([[-1/math.sqrt(13), -3/math.sqrt(13),4/math.sqrt(13)]])
        print(ans)
        np.testing.assert_array_equal(stand_norm(arr1), ans)


if __name__ == '__main__':
    unittest.main()
