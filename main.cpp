#include "similarity_transform.hpp"
#include "utils.hpp"
#include <iomanip>
#include <iostream>

using namespace sycl;

const uint N = 1 << 10;
const uint B = 1 << 5;

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;
  std::cout << "Parallel Algorithm using Similarity Transform for finding max "
               "eigen value (with vector)\n"
            << std::endl;

  for (uint dim = B; dim <= N; dim <<= 1) {
    float *mat = (float *)malloc(sizeof(float) * dim * dim);
    float *eigen_val = (float *)malloc(sizeof(float) * 1);
    float *eigen_vec = (float *)malloc(sizeof(float) * dim * 1);

    generate_random_positive_matrix(mat, dim);
    int64_t tm = sequential_transform(q, mat, eigen_val, eigen_vec, dim, B);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << tm << " ms" << std::endl;

    std::free(mat);
    std::free(eigen_val);
    std::free(eigen_vec);
  }

  return 0;
}
