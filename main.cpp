#include "similarity_transform.hpp"
#include "utils.hpp"
#include <iostream>

using namespace sycl;

const uint N = 3;
const uint B = 3;

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;

  float *mat = (float *)malloc(sizeof(float) * N * N);
  float *eigen_val = (float *)malloc(sizeof(float) * 1);
  float *eigen_vec = (float *)malloc(sizeof(float) * N * 1);

  generate_random_positive_matrix(mat, N);
  sequential_transform(q, mat, eigen_val, eigen_vec, N, B);

  std::free(mat);
  std::free(eigen_val);
  std::free(eigen_vec);

  return 0;
}
