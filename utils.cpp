#include <utils.hpp>

sycl::event identity_matrix(sycl::queue &q, float *const mat, const uint dim,
                            const uint wg_size,
                            std::vector<sycl::event> &evts) {
  auto evt_0 = q.memset(mat, 0, sizeof(float) * dim * dim);
  evts.push_back(evt_0);

  auto evt_1 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelIdentityMatrix>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);
          *(mat + r * dim + r) = 1.f;
        });
  });
  return evt_1;
}

void check(const float *vec, const uint dim) {
  for (uint i = 0; i < dim; i++) {
    assert(vec[i] == 1.f);
  }
}

sycl::event generate_vector(sycl::queue &q, float *const vec, const uint dim,
                            const uint wg_size,
                            std::vector<sycl::event> &evts) {
  auto evt_0 = q.memset(vec, 0, sizeof(float) * dim);
  evts.push_back(evt_0);

  auto evt_1 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelGenerateVector>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          *(vec + r) = r + 1;
        });
  });
  return evt_1;
}

float check_eigen_vector(const float *vec, const float *eigen_vec,
                         const float max, const uint dim) {
  float max_dev = 0.f;
  for (uint i = 0; i < dim; i++) {
    max_dev = std::max(max_dev, std::abs((vec[i] / max) - eigen_vec[i]));
  }
  return max_dev;
}

sycl::event stop_criteria_test_success_data(sycl::queue &q, float *const vec,
                                            const uint dim, const uint wg_size,
                                            std::vector<sycl::event> &evts) {
  const float EPS = 1e-4;
  auto evt_0 = q.memset(vec, 0, sizeof(float) * dim);
  evts.push_back(evt_0);

  auto evt_1 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelStopCriteriaTestSuccessData>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);
          *(vec + r) = (r + 1) * EPS;
        });
  });
  return evt_1;
}

sycl::event stop_criteria_test_fail_data(sycl::queue &q, float *const vec,
                                         const uint dim, const uint wg_size, ,
                                         std::vector<sycl::event> &evts) {
  const float EPS = 1e-4;
  auto evt_0 = q.memset(vec, 0, sizeof(float) * dim);
  evts.push_back(evt_0);

  auto evt_1 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelStopCriteriaTestSuccessData>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);
          if (r == wg_size - 1) {
            *(vec + r) = r + 1;
          } else {
            *(vec + r) = (r + 1) * EPS;
          }
        });
  });
  return evt_1;
}
