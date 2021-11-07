#pragma once
#include <CL/sycl.hpp>

inline constexpr float EPS = 1e-3;
inline constexpr uint MAX_ITR = 1000;

typedef std::chrono::_V2::steady_clock::time_point tp;
typedef sycl::buffer<float, 1> buffer_1d;
typedef sycl::buffer<float, 2> buffer_2d;
typedef sycl::accessor<float, 1, sycl::access::mode::read,
                       sycl::access::target::global_buffer>
    global_1d_reader;
typedef sycl::accessor<float, 2, sycl::access::mode::read,
                       sycl::access::target::global_buffer>
    global_2d_reader;
typedef sycl::accessor<float, 1, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
    global_1d_writer;
typedef sycl::accessor<float, 2, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
    global_2d_writer;
typedef sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::global_buffer>
    global_1d_reader_writer;
typedef sycl::accessor<float, 2, sycl::access::mode::read_write,
                       sycl::access::target::global_buffer>
    global_2d_reader_writer;
typedef sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
    local_1d_reader_writer;

int64_t sequential_transform(sycl::queue &q, const float *mat,
                             float *const eigen_val, float *const eigen_vec,
                             const uint dim, const uint wg_size);

sycl::event sum_across_rows(sycl::queue &q, const float *mat, float *const vec,
                            const uint count, const uint wg_size,
                            std::vector<sycl::event> evts);

sycl::event find_max(sycl::queue &q, const float *vec, float *const max,
                     const uint count, const uint wg_size,
                     std::vector<sycl::event> evts);

sycl::event compute_eigen_vector(sycl::queue &q, const float *vec,
                                 const float *max, float *const eigen_vec,
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
