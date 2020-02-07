import numpy as np
import math


# one-dimensional numpy array -> float
# computes the mean of an array
def my_mean(vec):
    vec_sum = 0.0
    i = 0
    for row in vec:
        if row is not None and not np.isnan(row):
            vec_sum += row
            i += 1
    if i > 0:
        return vec_sum/i
    else:
        return np.nan

# one-dimensional numpy array + (optional)int -> float
# computes the sample variance of an array
def my_var(vec, div_offset=1):
    if vec.size == 1:
        return 0
    if vec.ndim != 1:
        raise TypeError('The matrix must be a one-dimensional numpy array')
    else:
        vec_mean = my_mean(vec)
        diff_sq_sum = 0.0
        for val in vec:
            diff_sq_sum += (val - vec_mean)**2
        return diff_sq_sum/(vec.size - div_offset)

#  two-dimensional numpy array -> one-dimensional numpy array
# computes the mean of a numerical, multidimensional data set
def col_means(mat):
    #mat = np.transpose(mat)
    ret_arr = []
    for col in mat:
        ret_arr.append(my_mean(col))
    return np.array(ret_arr)


# one-dimensional numpy vector + one-dimensional numpy vector -> double
#compute the sample covariance between two attributes
def covar(vec1, vec2, div_offset=1):
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise TypeError('The vectors must be one-dimensional')
    if vec1.size != vec2.size:
        raise IndexError('The vectors must be the same length')
    if vec1.size == 1:
        return 0
    vec1_mean = my_mean(vec1)
    vec2_mean = my_mean(vec2)
    cov_sum = 0.0
    for i in range(vec1.size):
        cov_sum += (vec1[i] - vec1_mean)*(vec2[i] - vec2_mean)
    return cov_sum/(vec1.size - div_offset)


# one-dimensional numpy vector + one-dimensional numpy vector -> double
# computes the correlation between two attributes
def corr(vec1, vec2):
    cov = covar(vec1, vec2)
    sd1 = math.sqrt(my_var(vec1))
    sd2 = math.sqrt(my_var(vec2))
    return cov/(sd1*sd2)

# two-dimensional numpy array -> two-dimensional numpy array
# normalize the attributes in a two-dimensional numpy array using range normalization
def range_norm(mat):
    if mat.ndim == 1:
        raise TypeError('The matrix must be a two-dimensional numpy array')
    else:
        ret_mat = []
        for col in mat:
            ret_col= []
            col_min = min(col)
            col_max = max(col)
            for val in col:
                norm_val = (val - col_min)/(col_max-col_min)
                ret_col.append(norm_val)
            ret_mat.append(ret_col)
    return ret_mat
            

# two-dimensional numpy array -> two-dimensional numpy array
# normalize the attributes in a two-dimensional numpy array using standard normalization
def stand_norm(mat):
    if mat.ndim == 1:
        raise TypeError('The matrix must be a two-dimensional numpy array')
    else:
        ret_mat = []
        for col in mat:
            ret_col= []
            col_mean = my_mean(col)
            print("col_mean = " + str(col_mean))
            col_std = math.sqrt(my_var(col))
            print("col_std= " + str(col_std))
            for val in col:
                norm_val = (val - col_mean)/(col_std)
                ret_col.append(norm_val)
            ret_mat.append(ret_col)
    return ret_mat

# two-dimensional numpy array -> two-dimensional numpy array
# computes the covariance matrix of a data set.
def covar_mat(mat):
    #mat = np.transpose(mat)
    ret_mat = []
    ret_col = []
    for i, col1 in enumerate(mat):
        ret_col= []
        for j, col2 in enumerate(mat):
            if i == j:
                ret_col.append(my_var(col1))
            else:
                ret_col.append(covar(col1,col2))
        ret_mat.append(ret_col)
    return ret_mat

