#include <CL/sycl.hpp>

void identity_matrix(sycl::queue &q, float *const mat, const uint dim,
                     const uint wg_size);

void check(const float *vec, const uint dim);

void generate_vector(sycl::queue &q, float *const vec, const uint dim,
                     const uint wg_size);

float check_eigen_vector(const float *vec, const float *eigen_vec,
                         const float max, const uint dim);

void stop_criteria_test_data(sycl::queue &q, float *const vec, const uint dim,
                             const uint wg_size);
