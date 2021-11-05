#include "similarity_transform.hpp"
#include "utils.hpp"
#include <iostream>

using namespace sycl;

const uint N = 1 << 10;
const uint B = 1 << 7;

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << std::endl;

  float *mat = (float *)malloc(sizeof(float) * N * N);
  float *vec = (float *)malloc(sizeof(float) * N * 1);
  float *eigen_vec = (float *)malloc(sizeof(float) * N * 1);

  identity_matrix(q, mat, N, B);
  sum_across_rows(q, mat, vec, N, B);

  check(vec, N);
  std::cout << "sum across row works !" << std::endl;

  float max = 0;
  generate_vector(q, vec, N, B);
  find_max(q, vec, &max, N, B);
  assert(max == N);
  std::cout << "max from vector works !" << std::endl;

  initialise_eigen_vector(q, eigen_vec, N);
  compute_eigen_vector(q, vec, max, eigen_vec, N, B);
  float max_dev = check_eigen_vector(vec, eigen_vec, max, N);
  std::cout << "maximum deviation in computing eigen vector " << max_dev
            << std::endl;

  uint ret = 0;
  stop_criteria_test_success_data(q, vec, N, B);
  stop(q, vec, &ret, N, B);
  std::cout << "stopping criteria test result [success]: " << ret << std::endl;

  stop_criteria_test_fail_data(q, vec, N, B);
  stop(q, vec, &ret, N, B);
  std::cout << "stopping criteria test result [fail]: " << ret << std::endl;

  std::free(mat);
  std::free(vec);
  std::free(eigen_vec);

  return 0;
}
