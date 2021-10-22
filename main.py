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

    vecs = []
    while True:
        vec = sum_across_rows(mat)
        vecs.append(vec)  # to be used when computing eigen vector
        if stop(vec):
            break

        mat = compute_next(mat, vec)

    val = vecs[-1][0]
    vec = np.array([reduce(lambda acc, cur: acc * cur, [j[i]/np.max(j)
                                                        for j in vecs], 1.) for i in range(mat.shape[0])])
    return val, vec


if __name__ == '__main__':
    val, vec = main()
    print(f'eigen value: {val}\neigen vector: {vec}')
