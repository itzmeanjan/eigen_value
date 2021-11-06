#pragma once
#include <CL/sycl.hpp>

inline constexpr float EPS = 1e-3;

sycl::event sum_across_rows(sycl::queue &q, const float *mat, float *const vec,
                            const uint count, const uint wg_size,
                            std::vector<sycl::event> evts);

sycl::event find_max(sycl::queue &q, const float *vec, float *const max,
                     const uint count, const uint wg_size,
                     std::vector<sycl::event> evts);

sycl::event compute_eigen_vector(sycl::queue &q, const float *vec,
                                 const float max, float *const eigen_vec,
                                 const uint count, const uint wg_size,
                                 std::vector<sycl::event> evts);

sycl::event initialise_eigen_vector(sycl::queue &q, float *const vec,
                                    const uint count,
                                    std::vector<sycl::event> evts);

sycl::event compute_next_matrix(sycl::queue &q, float *const mat,
                                const float *sum_vec, const uint count,
                                const uint wg_size,
                                std::vector<sycl::event> evts);

// Check for stopping criteria, whether it's good time to
// stop as result has converged to max eigen value which was being
// searched for
sycl::event stop(sycl::queue &q, const float *vec, uint *const ret,
                 const uint count, const uint wg_size,
                 std::vector<sycl::event> evts);
