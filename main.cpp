#include <CL/sycl.hpp>
#include <iostream>
#include <utils.hpp>

using namespace sycl;

const uint N = 4;
const uint B = 4;

void sum_across_rows(queue &q, const float *mat, float *const vec) {
  memset(vec, 0, sizeof(float) * N);

  buffer<float, 2> b_mat{mat, range<2>{N, N}};
  buffer<float, 1> b_vec{vec, range<1>{N}};

  auto evt = q.submit([&](handler &h) {
    accessor<float, 2, access::mode::read, access::target::global_buffer>
        acc_mat{b_mat, h};
    accessor<float, 1, access::mode::write, access::target::global_buffer>
        acc_vec{b_vec, h};

    h.parallel_for<class kernelSumAcrossRows>(
        nd_range<2>{range<2>{N, N}, range<2>{1, B}}, [=](nd_item<2> it) {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          float val = acc_mat[r][c];
          ONEAPI::atomic_ref<float, ONEAPI::memory_order::relaxed,
                             ONEAPI::memory_scope::device,
                             access::address_space::global_space>
              ref(acc_vec[r]);
          ref.fetch_add(val);
        });
  });
  evt.wait();
}

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << std::endl;

  float *mat = (float *)malloc(sizeof(float) * N * N);
  float *vec = (float *)malloc(sizeof(float) * N * 1);

  identity_matrix(q, mat, N, B);
  sum_across_rows(q, mat, vec);

  check(vec, N);
  std::cout << "sum across row works !" << std::endl;

  return 0;
}
