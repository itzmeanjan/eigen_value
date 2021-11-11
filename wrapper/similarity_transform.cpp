#include "similarity_transform.hpp"

extern "C" int64_t max_eigen_value(float *mat, float *eigen_val,
                                   float *eigen_vec, uint dim, uint *iter_cnt) {
  sycl::device d{sycl::default_selector{}};
  sycl::queue q{d};

  int64_t ts =
      similarity_transform(q, mat, eigen_val, eigen_vec, dim, 1 << 5, iter_cnt);

  return ts;
}
