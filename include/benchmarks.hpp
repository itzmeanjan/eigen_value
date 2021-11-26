#pragma once
#include <similarity_transform.hpp>
#include <utils.hpp>

int64_t benchmark_sum_across_rows_kernel_v0(sycl::queue &q, const uint dim,
                                            const uint wg_size);

int64_t benchmark_sum_across_rows_kernel_v1(sycl::queue &q, const uint dim,
                                            const uint wg_size);

int64_t benchmark_sum_across_rows_kernel_v2(sycl::queue &q, const uint dim,
                                         const uint wg_size);

int64_t benchmark_similarity_transform(sycl::queue &q, const uint dim,
                                       const uint wg_size,
                                       uint *const itr_count);

int64_t benchmark_find_vector_max_v0(sycl::queue &q, const uint dim,
                                     const uint wg_size);

int64_t benchmark_find_vector_max_v1(sycl::queue &q, const uint dim,
                                     const uint wg_size);

int64_t benchmark_find_vector_max_v2(sycl::queue &q, const uint dim,
                                  const uint wg_size);

int64_t benchmark_compute_eigen_vector(sycl::queue &q, const uint dim,
                                       const uint wg_size);

int64_t benchmark_compute_next_matrix(sycl::queue &q, const uint dim,
                                      const uint wg_size);

int64_t benchmark_stop_criteria_tester(sycl::queue &q, const uint dim,
                                       const uint wg_size);
