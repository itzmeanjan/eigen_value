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

int64_t similarity_transform(sycl::queue &q, const float *mat,
                             float *const eigen_val, float *const eigen_vec,
                             const uint dim, const uint wg_size,
                             uint *const iter_count);

sycl::event sum_across_rows(sycl::queue &q, buffer_2d mat, buffer_1d vec,
                            const uint dim, const uint wg_size,
                            std::vector<sycl::event> evts);

sycl::event find_max(sycl::queue &q, buffer_1d vec, buffer_1d max,
                     const uint dim, const uint wg_size,
                     std::vector<sycl::event> evts);

sycl::event compute_eigen_vector(sycl::queue &q, buffer_1d vec, buffer_1d max,
                                 buffer_1d eigen_vec, const uint dim,
                                 const uint wg_size,
                                 std::vector<sycl::event> evts);

sycl::event initialise_eigen_vector(sycl::queue &q, buffer_1d vec,
                                    const uint dim,
                                    std::vector<sycl::event> evts);

sycl::event compute_next_matrix(sycl::queue &q, buffer_2d mat, buffer_1d vec,
                                const uint dim, const uint wg_size,
                                std::vector<sycl::event> evts);

sycl::event stop(sycl::queue &q, buffer_1d vec, sycl::buffer<uint, 1> ret,
                 const uint dim, const uint wg_size,
                 std::vector<sycl::event> evts);
