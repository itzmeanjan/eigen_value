#pragma once
#include <CL/sycl.hpp>

sycl::event identity_matrix(sycl::queue &q, float *const mat, const uint dim,
                            const uint wg_size, std::vector<sycl::event> evts);

void check(const float *vec, const uint dim);

sycl::event generate_vector(sycl::queue &q, float *const vec, const uint dim,
                            const uint wg_size, std::vector<sycl::event> evts);

float check_eigen_vector(const float *vec, const float *eigen_vec,
                         const float max, const uint dim);

sycl::event stop_criteria_test_success_data(sycl::queue &q, float *const vec,
                                            const uint dim, const uint wg_size,
                                            std::vector<sycl::event> evts);

sycl::event stop_criteria_test_fail_data(sycl::queue &q, float *const vec,
                                         const uint dim, const uint wg_size,
                                         std::vector<sycl::event> evts);

void generate_random_positive_matrix(float *const mat, const uint dim);

void generate_hilbert_matrix(sycl::queue &q, float *const mat, const uint dim);
