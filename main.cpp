#include <CL/sycl.hpp>
#include <iostream>
#include <utils.hpp>

using namespace sycl;

const uint N = 4;
const uint B = 4;

void compute_eigen_vector(queue &q, const float *vec, const float max,
                          float *const eigen_vec) {
  buffer<float, 1> b_vec{vec, range<1>{N}};
  buffer<float, 1> b_eigen_vec{eigen_vec, range<1>{N}};

  q.submit([&](handler &h) {
    accessor<float, 1, access::mode::read, access::target::global_buffer>
        acc_vec{b_vec, h};
    accessor<float, 1, access::mode::read_write, access::target::global_buffer>
        acc_eigen_vec{b_eigen_vec, h};

    h.parallel_for(nd_range<1>{range<1>{N}, range<1>{B}}, [=](nd_item<1> it) {
      const size_t r = it.get_global_id(0);

      acc_eigen_vec[r] *= (acc_vec[r] / max);
    });
  });
}

void find_max(queue &q, const float *vec, float *max) {
  *max = 0;

  buffer<float, 1> b_vec{vec, range<1>{N}};
  buffer<float, 1> b_max{max, range<1>{1}};

  auto evt = q.submit([&](handler &h) {
    accessor<float, 1, access::mode::read, access::target::global_buffer>
        acc_vec{b_vec, h};
    accessor<float, 1, access::mode::read_write, access::target::global_buffer>
        acc_max{b_max, h};

    h.parallel_for<class kernelMaxInVector>(
        nd_range<1>{range<1>{N}, range<1>{B}}, [=](nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          ONEAPI::atomic_ref<float, ONEAPI::memory_order::relaxed,
                             ONEAPI::memory_scope::device,
                             access::address_space::global_space>
              ref(acc_max[0]);
          ref.fetch_max(acc_vec[r]);
        });
  });
  evt.wait();
}

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

  float max = 0;
  generate_vector(q, vec, N, B);
  find_max(q, vec, &max);
  assert(max == N);
  std::cout << "max from vector works !" << std::endl;

  std::free(mat);
  std::free(vec);

  return 0;
}
