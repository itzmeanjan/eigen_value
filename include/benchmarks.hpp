#pragma once
#include <similarity_transform.hpp>
#include <utils.hpp>

int64_t benchmark_sum_across_rows_kernel(sycl::queue &q, const uint dim,
                                         const uint wg_size);

int64_t benchmark_similarity_transform(sycl::queue &q, const uint dim,
                                       const uint wg_size,
                                       uint *const itr_count);

int64_t benchmark_find_vector_max(sycl::queue &q, const uint dim,
                                         const uint wg_size);
