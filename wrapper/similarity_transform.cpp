#include "similarity_transform.hpp"

extern "C" void
make_queue(void** wq)
{
  sycl::default_selector d_sel{};
  sycl::device d{ d_sel };
  sycl::context c{ d };
  sycl::queue* q = new sycl::queue{ c, d };

  *wq = q;
}

extern "C" int64_t
max_eigen_value(void* wq,
                float* mat,
                float* eigen_val,
                float* eigen_vec,
                uint dim,
                uint* iter_cnt)
{
  sycl::queue* q = reinterpret_cast<sycl::queue*>(wq);

  const size_t max_wg_size =
    q->get_device().get_info<sycl::info::device::max_work_group_size>();

  int64_t ts =
    similarity_transform(*q,
                         mat,
                         eigen_val,
                         eigen_vec,
                         dim,
                         (dim >> 1) > max_wg_size ? max_wg_size : (dim >> 1),
                         iter_cnt);

  return ts;
}
