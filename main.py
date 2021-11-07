#!/usr/bin/python3

import numpy as np
from time import time

EPS = 1e-3

'''
    Sequential Implementation of https://doi.org/10.1007/978-3-319-11194-0_18
'''


def compute_next(mat, vec):
    sigma = np.diag(vec)
    sigma_inv = np.diag(1/vec)
    return np.matmul(np.matmul(sigma_inv, mat), sigma)


def sum_across_rows(mat):
    n = mat.shape[1]
    v = np.array([np.sum(mat[i]) for i in range(n)])
    return v


def stop(vec):
    return all(map(lambda e: e < EPS, [abs(vec[i] - vec[i-1])
                                       for i in range(1, len(vec))]))


def max_eigen_value_and_vector(mat):
    eigen_val = 0
    eigen_vec = np.ones(mat.shape[0])

    itr = 0
    while True:
        vec = sum_across_rows(mat)
        vec_max = np.max(vec)
        eigen_vec = np.array([j * (vec[i]/vec_max)
                              for i, j in enumerate(eigen_vec)])
        if stop(vec):
            eigen_val = vec[0]
            break

        mat = compute_next(mat, vec)
        itr += 1

    return eigen_val, eigen_vec, itr + 1


if __name__ == '__main__':
    # handwritten test begins
    mat = np.array([[1, 1, 2], [2, 1, 3], [2, 3, 5]])
    val, vec, _ = max_eigen_value_and_vector(mat)

    assert abs(val - 7.5311) < EPS
    assert abs(vec[0] - 0.3941) < EPS
    assert abs(vec[1] - 0.5788) < EPS
    assert abs(vec[2] - 0.9975) < EPS
    # handwritten test ends

    print('Sequential Similarity Transform, for finding maximum eigen value ( with vector )\n')
    for dim in range(5, 11):
        mat = np.random.random((1 << dim, 1 << dim))
        start = time() * 1000
        val, _, itr = max_eigen_value_and_vector(mat)
        end = time() * 1000

        assert val - np.max(np.linalg.eigvals(mat)) < EPS
        print(
            f'{1 << dim:<4} x {1 << dim:>4}\t\t{end - start:>6.2f} ms\t\t{itr:>8} round(s)')
