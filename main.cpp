#include "similarity_transform.hpp"
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

  *(mat + 0 * N + 0) = 1;
  *(mat + 0 * N + 1) = 1;
  *(mat + 0 * N + 2) = 2;

  *(mat + 1 * N + 0) = 2;
  *(mat + 1 * N + 1) = 1;
  *(mat + 1 * N + 2) = 3;

  *(mat + 2 * N + 0) = 2;
  *(mat + 2 * N + 1) = 3;
  *(mat + 2 * N + 2) = 5;

  sequential_transform(q, mat, eigen_val, eigen_vec, N, B);

  for (uint i = 0; i < N; i++) {
    std::cout << *(eigen_vec + i) << std::endl;
  }
  std::cout << "\neigen value " << *eigen_val << std::endl;

  std::free(mat);
  std::free(eigen_val);
  std::free(eigen_vec);

  return 0;
}
