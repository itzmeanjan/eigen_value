#!/usr/bin/python3

'''
  Before using this python module, make sure you've run
  `make lib` and generated shared object, which is loaded
  here and respective function calls are forwarded to SYCL
  DPC++ implementations. Kernels are run on available
  default accelerator.
'''

from typing import Tuple
import numpy as np
import ctypes
from genericpath import exists
from posixpath import abspath


class EigenValue:
    so_path: str = '../libsimilarity_transform.so'
    sycl_q: ctypes.c_void_p = None
    so_lib: ctypes.CDLL = None

    def __init__(self) -> None:
        '''
        Creates an instance of `EigenValue` class, along with backend resource(s)
        like setting up SYCL queue where compute jobs will be submitted, getting
        shared library ready for forwarding future function invocations
        '''
        if not exists(self.so_path):
            raise Exception(
                f'failed to find shared library `{abspath(self.so_path)}`')

        self.so_lib = ctypes.CDLL(self.so_path)

        self.so_lib.make_queue.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.sycl_q = ctypes.c_void_p()
        self.so_lib.make_queue(ctypes.byref(self.sycl_q))

        if self.sycl_q.value == None:
            raise Exception(f'failed to get default SYCL queue')

    def similarity_transform(self, mat: np.ndarray) -> Tuple[np.float32, np.ndarray, int, int]:
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

        self.so_lib.max_eigen_value.restype = ctypes.c_int64
        self.so_lib.max_eigen_value.argtypes = [
            ctypes.c_void_p,
            mat_t, vec_t, vec_t, ctypes.c_uint, itr_cnt_t]

        eigen_val = np.empty(1, dtype=np.float32)
        eigen_vec = np.empty(n, dtype=np.float32)
        iter_cnt = np.zeros(1, dtype=np.uint)

        ts = self.so_lib.max_eigen_value(
            self.sycl_q, mat, eigen_val, eigen_vec, n, iter_cnt)

        return eigen_val[0], eigen_vec, ts, iter_cnt[0]
