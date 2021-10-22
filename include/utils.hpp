#include <CL/sycl.hpp>

void identity_matrix(sycl::queue &q, float *const mat, const uint dim,
                     const uint wg_size);
