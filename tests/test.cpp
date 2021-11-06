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

  mat = (float *)malloc(sizeof(float) * 3 * 3);
  eigen_vec = (float *)malloc(sizeof(float) * 3 * 1);
  float *eigen_val = (float *)malloc(sizeof(float) * 1);

  *(mat + 0 * 3 + 0) = 1;
  *(mat + 0 * 3 + 1) = 1;
  *(mat + 0 * 3 + 2) = 2;

  *(mat + 1 * 3 + 0) = 2;
  *(mat + 1 * 3 + 1) = 1;
  *(mat + 1 * 3 + 2) = 3;

  *(mat + 2 * 3 + 0) = 2;
  *(mat + 2 * 3 + 1) = 3;
  *(mat + 2 * 3 + 2) = 5;

  sequential_transform(q, mat, eigen_val, eigen_vec, 3, 3);

  assert(abs(*eigen_val - 7.53114) < EPS);
  assert(abs(*(eigen_vec + 0) - 0.394074) < EPS);
  assert(abs(*(eigen_vec + 1) - 0.578844) < EPS);
  assert(abs(*(eigen_vec + 2) - 0.997451) < EPS);
  std::cout << "sequential transform worked !" << std::endl;

  std::free(mat);
  std::free(eigen_val);
  std::free(eigen_vec);

  return 0;
}
