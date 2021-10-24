#include <utils.hpp>

void identity_matrix(sycl::queue &q, float *const mat, const uint dim,
                     const uint wg_size) {
  memset(mat, 0, sizeof(float) * dim * dim);

  sycl::buffer<float, 2> b_mat{mat, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 2, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        acc_mat{b_mat, h};

    h.parallel_for<class kernelIdentityMatrix>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);
          acc_mat[r][r] = 1.f;
        });
  });
  evt.wait();
}

void check(const float *vec, const uint dim) {
  for (uint i = 0; i < dim; i++) {
    assert(vec[i] == 1.f);
  }
}

void generate_vector(sycl::queue &q, float *const vec, const uint dim,
                     const uint wg_size) {
  memset(vec, 0, sizeof(float) * dim);

  sycl::buffer<float, 1> b_vec{vec, sycl::range<1>{dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 1, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        acc_vec{b_vec, h};

    h.parallel_for<class kernelGenerateVector>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          acc_vec[r] = r + 1;
        });
  });
  evt.wait();
}

float check_eigen_vector(const float *vec, const float *eigen_vec,
                         const float max, const uint dim) {
  float max_dev = 0.f;
  for (uint i = 0; i < dim; i++) {
    max_dev = std::max(max_dev, std::abs((vec[i] / max) - eigen_vec[i]));
  }
  return max_dev;
}

void stop_criteria_test__success_data(sycl::queue &q, float *const vec,
                                      const uint dim, const uint wg_size) {
  memset(vec, 0, sizeof(float) * dim);
  const float EPS = 1e-4;

  sycl::buffer<float, 1> b_vec{vec, sycl::range<1>{dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 1, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        a_vec{b_vec, h};

    h.parallel_for<class kernelStopCriteriaTestSuccessData>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          a_vec[r] = (r + 1) * EPS;
        });
  });
  evt.wait();
}

void stop_criteria_test_fail_data(sycl::queue &q, float *const vec,
                                  const uint dim, const uint wg_size) {
  memset(vec, 0, sizeof(float) * dim);
  const float EPS = 1e-4;

  sycl::buffer<float, 1> b_vec{vec, sycl::range<1>{dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 1, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        a_vec{b_vec, h};

    h.parallel_for<class kernelStopCriteriaTestFailData>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          a_vec[r] = r % 3 != 0 ? r * EPS : (r + 2) * EPS;
        });
  });
  evt.wait();
}
