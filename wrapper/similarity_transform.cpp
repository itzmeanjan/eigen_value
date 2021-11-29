#include "similarity_transform.hpp"

extern "C" int64_t max_eigen_value(float *mat, float *eigen_val,
                                   float *eigen_vec, uint dim, uint *iter_cnt) {
  sycl::device d{sycl::default_selector{}};
  sycl::queue q{d};
  const size_t max_wg_size =
      d.get_info<sycl::info::device::max_work_group_size>();

  int64_t ts = similarity_transform(
      q, mat, eigen_val, eigen_vec, dim,
      (dim >> 1) > max_wg_size ? max_wg_size : (dim >> 1), iter_cnt);

  return ts;
}
