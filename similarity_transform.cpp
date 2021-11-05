#include "similarity_transform.hpp"

void sum_across_rows(sycl::queue &q, const float *mat, float *const vec,
                     const uint count, const uint wg_size) {
  memset(vec, 0, sizeof(float) * count);

  sycl::buffer<float, 2> b_mat{mat, sycl::range<2>{count, count}};
  sycl::buffer<float, 1> b_vec{vec, sycl::range<1>{count}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 2, sycl::access::mode::read,
                   sycl::access::target::global_buffer>
        acc_mat{b_mat, h};
    sycl::accessor<float, 1, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        acc_vec{b_vec, h};

    h.parallel_for<class kernelSumAcrossRows>(
        sycl::nd_range<2>{sycl::range<2>{count, count},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          float val = acc_mat[r][c];
          sycl::ONEAPI::atomic_ref<float, sycl::ONEAPI::memory_order::relaxed,
                                   sycl::ONEAPI::memory_scope::device,
                                   sycl::access::address_space::global_space>
              ref(acc_vec[r]);
          ref.fetch_add(val);
        });
  });
  evt.wait();
}

void find_max(sycl::queue &q, const float *vec, float *max, const uint count,
              const uint wg_size) {
  *max = 0;

  sycl::buffer<float, 1> b_vec{vec, sycl::range<1>{count}};
  sycl::buffer<float, 1> b_max{max, sycl::range<1>{1}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 1, sycl::access::mode::read,
                   sycl::access::target::global_buffer>
        acc_vec{b_vec, h};
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
        acc_max{b_max, h};

    h.parallel_for<class kernelMaxInVector>(
        sycl::nd_range<1>{sycl::range<1>{count}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          sycl::ONEAPI::atomic_ref<float, sycl::ONEAPI::memory_order::relaxed,
                                   sycl::ONEAPI::memory_scope::device,
                                   sycl::access::address_space::global_space>
              ref(acc_max[0]);
          ref.fetch_max(acc_vec[r]);
        });
  });
  evt.wait();
}

void compute_eigen_vector(sycl::queue &q, const float *vec, const float max,
                          float *const eigen_vec, const uint count,
                          const uint wg_size) {
  sycl::buffer<float, 1> b_vec{vec, sycl::range<1>{count}};
  sycl::buffer<float, 1> b_eigen_vec{eigen_vec, sycl::range<1>{count}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 1, sycl::access::mode::read,
                   sycl::access::target::global_buffer>
        acc_vec{b_vec, h};
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
        acc_eigen_vec{b_eigen_vec, h};

    h.parallel_for<class kernelComputeEigenVector>(
        sycl::nd_range<1>{sycl::range<1>{count}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          acc_eigen_vec[r] *= (acc_vec[r] / max);
        });
  });
  evt.wait();
}

void initialise_eigen_vector(sycl::queue &q, float *const vec,
                             const uint count) {
  sycl::buffer<float, 1> b_vec{vec, sycl::range<1>{count}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 1, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        acc_vec{b_vec, h};

    h.fill(acc_vec, 1.f);
  });
  evt.wait();
}

void compute_next_matrix(sycl::queue &q, float *const mat, const float *sum_vec,
                         const uint count, const uint wg_size) {
  sycl::buffer<float, 2> b_mat{mat, sycl::range<2>{count, count}};
  sycl::buffer<float, 1> b_sum_vec{sum_vec, sycl::range<1>{count}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 1, sycl::access::mode::read,
                   sycl::access::target::global_buffer>
        acc_sum_vec{b_sum_vec, h};
    sycl::accessor<float, 2, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
        acc_mat{b_mat, h};

    h.parallel_for<class kernelSimilarityTransform>(
        sycl::nd_range<2>{sycl::range<2>{count, count},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          float v0 = acc_sum_vec[r];
          float v1 = r == c ? v0 : acc_sum_vec[c];

          acc_mat[r][c] *= (1.f / v0) * v1;
        });
  });
  evt.wait();
}

void stop(sycl::queue &q, const float *vec, uint *const ret, const uint count,
          const uint wg_size) {
  *ret = 1; // == 1 denotes should stop !

  sycl::buffer<float, 1> b_vec{vec, sycl::range<1>{count}};
  sycl::buffer<uint, 1> b_ret{ret, sycl::range<1>{1}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 1, sycl::access::mode::read,
                   sycl::access::target::global_buffer>
        a_vec{b_vec, h};
    sycl::accessor<uint, 1, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
        a_ret{b_ret, h};

    h.parallel_for<class kernelStopCriteria>(
        sycl::nd_range<1>{sycl::range<1>{count - 1}, sycl::range<1>{wg_size},
                          sycl::id<1>{1}},
        [=](sycl::nd_item<1> it) {
          sycl::ONEAPI::sub_group sg = it.get_sub_group();
          const size_t r = it.get_global_id(0);

          float diff = sycl::abs(a_vec[r] - a_vec[r - 1]);
          bool res = sycl::ONEAPI::all_of(sg, diff < EPS);

          if (sycl::ONEAPI::leader(sg)) {
            sycl::ONEAPI::atomic_ref<uint, sycl::ONEAPI::memory_order::relaxed,
                                     sycl::ONEAPI::memory_scope::device,
                                     sycl::access::address_space::global_space>
                ref{a_ret[0]};
            ref.fetch_min(res ? 1 : 0);
          }
        });
  });
  evt.wait();
}
