#!/usr/bin/python3

import numpy as np
from functools import reduce

EPS = 1e-5

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


def main():
    mat = np.array([[1, 1, 2], [2, 1, 3], [2, 3, 5]])

    vec = []
    eigen_vec = np.ones(mat.shape[0])
    while True:
        vec = sum_across_rows(mat)
        eigen_vec = np.array([j * (vec[i]/np.max(vec))
                              for i, j in enumerate(eigen_vec)])
        if stop(vec):
            break

        mat = compute_next(mat, vec)

    return vec[0], eigen_vec


if __name__ == '__main__':
    val, vec = main()
    print(f'eigen value: {val}\neigen vector: {vec}')
