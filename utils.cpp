#include "utils.hpp"
#include "similarity_transform.hpp"
#include <random>

sycl::event identity_matrix(sycl::queue &q, float *const mat, const uint dim,
                            const uint wg_size, std::vector<sycl::event> evts) {
  memset(mat, 0, sizeof(float) * dim * dim);
  buffer_2d buf_mat{mat, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    global_2d_writer acc_mat{buf_mat, h};

    h.depends_on(evts);
    h.parallel_for<class kernelIdentityMatrix>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);
          acc_mat[r][r] = 1.f;
        });
  });
  return evt;
}

void check(const float *vec, const uint dim) {
  for (uint i = 0; i < dim; i++) {
    assert(vec[i] == 1.f);
  }
}

sycl::event generate_vector(sycl::queue &q, float *const vec, const uint dim,
                            const uint wg_size, std::vector<sycl::event> evts) {
  memset(vec, 0, sizeof(float) * dim);
  buffer_1d buf_vec{vec, sycl::range<1>{dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    global_1d_writer acc_vec{buf_vec, h};

    h.depends_on(evts);
    h.parallel_for<class kernelGenerateVector>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);
          acc_vec[r] = r + 1;
        });
  });
  return evt;
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
                                            std::vector<sycl::event> evts) {
  const float EPS = 1e-4f;
  memset(vec, 0, sizeof(float) * dim);
  buffer_1d buf_vec{vec, sycl::range<1>{dim}};

  auto evt_1 = q.submit([&](sycl::handler &h) {
    global_1d_writer acc_vec{buf_vec, h};

    h.depends_on(evts);
    h.parallel_for<class kernelStopCriteriaTestSuccessData>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);
          acc_vec[r] = 1.f + EPS;
        });
  });
  return evt_1;
}

sycl::event stop_criteria_test_fail_data(sycl::queue &q, float *const vec,
                                         const uint dim, const uint wg_size,
                                         std::vector<sycl::event> evts) {
  const float EPS = 1e-4f;
  memset(vec, 0, sizeof(float) * dim);
  buffer_1d buf_vec{vec, sycl::range<1>{dim}};

  auto evt_1 = q.submit([&](sycl::handler &h) {
    global_1d_writer acc_vec{buf_vec, h};

    h.depends_on(evts);
    h.parallel_for<class kernelStopCriteriaTestFailData>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);
          acc_vec[r] = (float)(r + 1) * EPS;
        });
  });
  return evt_1;
}

void generate_random_vector(float *const vec, const uint dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.f, 1.f);

  for (uint i = 0; i < dim; i++) {
    *(vec + i) = dis(gen);
  }
}

void generate_hilbert_matrix(sycl::queue &q, float *const mat, const uint dim) {
  buffer_2d buf_mat{mat, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    global_2d_writer acc_mat{buf_mat, h, sycl::no_init};

    h.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, 32}},
        [=](sycl::nd_item<2> it) {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          acc_mat[r][c] = 1.f / (float)(r + c + 1);
        });
  });
  evt.wait();
}
