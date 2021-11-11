#!/usr/bin/python3

'''
  Before using this python module, make sure you've run
  `make lib` and generated shared object, which is loaded
  here and respective function calls are forwarded to SYCL
  DPC++ implementations. Kernels are run on available
  default accelerator.
'''

import numpy as np
import ctypes

st_lib = ctypes.CDLL('../libsimilarity_transform.so')


def similarity_transform(mat):
    '''
      Applies similarity transform method on provided
      square matrix ( represented as numpy array ) of
      single precision floating point numbers

      Returns (max eigen value, respective eigen vector,
      time spent in milliseconds, iteration count before convergence)

      Assuming max eigen value = λ; eigen vector = v; input matrix = A

      then  Av = λv, must be satisfied !
    '''
    m, n = mat.shape
    assert m == n, "must be square matrix of floating points !"
    assert mat.dtype.num == 11, "dtype of input matrix must be float32 !"

    mat_t = np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=2, flags='CONTIGUOUS')
    vec_t = np.ctypeslib.ndpointer(
        dtype=np.float32, ndim=1, flags='CONTIGUOUS')
    itr_cnt_t = np.ctypeslib.ndpointer(
        dtype=np.uint, ndim=1, flags='CONTIGUOUS')

    st_lib.max_eigen_value.restype = ctypes.c_int64
    st_lib.max_eigen_value.argtypes = [
        mat_t, vec_t, vec_t, ctypes.c_uint, itr_cnt_t]

    eigen_val = np.empty(1, dtype=np.float32)
    eigen_vec = np.empty(n, dtype=np.float32)
    iter_cnt = np.zeros(1, dtype=np.uint)

    ts = st_lib.max_eigen_value(mat, eigen_val, eigen_vec, n, iter_cnt)
    return eigen_val[0], eigen_vec, ts, iter_cnt[0]
