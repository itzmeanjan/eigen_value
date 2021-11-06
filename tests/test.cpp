#include "similarity_transform.hpp"
#include "utils.hpp"
#include <iostream>

using namespace sycl;

const uint N = 1 << 10;
const uint B = 1 << 7;

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;

  float *mat = (float *)malloc_shared(sizeof(float) * N * N, q);
  float *vec = (float *)malloc_shared(sizeof(float) * N * 1, q);
  float *eigen_vec = (float *)malloc_shared(sizeof(float) * N * 1, q);

  identity_matrix(q, mat, N, B, {}).wait();
  sum_across_rows(q, mat, vec, N, B, {}).wait();

  check(vec, N);
  std::cout << "sum across row works !" << std::endl;

  float *max = (float *)malloc_shared(sizeof(float) * 1, q);
  generate_vector(q, vec, N, B, {}).wait();
  find_max(q, vec, max, N, B, {}).wait();

  assert(*max == N);
  std::cout << "max from vector works !" << std::endl;

  initialise_eigen_vector(q, eigen_vec, N, {}).wait();
  compute_eigen_vector(q, vec, max, eigen_vec, N, B, {}).wait();

  float max_dev = check_eigen_vector(vec, eigen_vec, *max, N);
  std::cout << "maximum deviation in computing eigen vector " << max_dev
            << std::endl;

  uint *ret = (uint *)malloc_shared(sizeof(uint) * 1, q);
  stop_criteria_test_success_data(q, vec, N, B, {}).wait();

  stop(q, vec, ret, N, B, {}).wait();
  std::cout << "stopping criteria test result [success]: " << *ret << std::endl;

  stop_criteria_test_fail_data(q, vec, N, B, {}).wait();

  stop(q, vec, ret, N, B, {}).wait();
  std::cout << "stopping criteria test result [fail]: " << *ret << std::endl;

  sycl::free(mat, q);
  sycl::free(vec, q);
  sycl::free(eigen_vec, q);

  return 0;
}
