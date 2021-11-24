#include "similarity_transform.hpp"
#include "utils.hpp"
#include <iomanip>
#include <iostream>

using namespace sycl;

int main() {
  device d{default_selector{}};
  queue q{d};

  const size_t max_wg_size =
      d.get_info<info::device::max_work_group_size>() >> 1;

  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;
  std::cout << "Parallel Algorithm using Similarity Transform for finding max "
               "eigen value (with vector)\n"
            << std::endl;

  for (uint i = 7; i <= 13; i++) {
    const uint dim = 1ul << i;

    float *mat = (float *)malloc(sizeof(float) * dim * dim);
    float *eigen_val = (float *)malloc(sizeof(float) * 1);
    float *eigen_vec = (float *)malloc(sizeof(float) * dim * 1);

    generate_hilbert_matrix(q, mat, dim);
    uint itr_count = 0;
    int64_t tm = similarity_transform(q, mat, eigen_val, eigen_vec, dim,
                                      dim <= max_wg_size ? dim : max_wg_size,
                                      &itr_count);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << tm << " ms"
              << "\t\t\t" << std::setw(6) << std::right << itr_count
              << " round(s)" << std::endl;

    std::free(mat);
    std::free(eigen_val);
    std::free(eigen_vec);
  }

  return 0;
}
