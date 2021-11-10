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

  float *mat = (float *)malloc(sizeof(float) * N * N);
  float *vec = (float *)malloc(sizeof(float) * N * 1);
  float *eigen_vec = (float *)malloc(sizeof(float) * N * 1);

  identity_matrix(q, mat, N, B, {}).wait();
  {
    buffer_2d buf_mat{mat, range<2>{N, N}};
    buffer_1d buf_vec{vec, range<1>{N}};

    sum_across_rows(q, buf_mat, buf_vec, N, B, {}).wait();
  }
  check(vec, N);
  std::cout << "sum across row works !" << std::endl;

  float *max = (float *)malloc(sizeof(float) * 1);
  generate_vector(q, vec, N, B, {}).wait();
  {
    buffer_1d buf_vec{vec, sycl::range<1>{N}};
    buffer_1d buf_max{max, sycl::range<1>{1}};

    find_max(q, buf_vec, buf_max, N, B, {}).wait();
  }
  assert(*max == N);
  std::cout << "max from vector works !" << std::endl;

  {
    buffer_1d buf_vec{vec, sycl::range<1>{N}};
    buffer_1d buf_eigen_vec{eigen_vec, sycl::range<1>{N}};
    buffer_1d buf_max{max, sycl::range<1>{1}};

    initialise_eigen_vector(q, buf_eigen_vec, N, {}).wait();
    compute_eigen_vector(q, buf_vec, buf_max, buf_eigen_vec, N, B, {}).wait();
  }

  float max_dev = check_eigen_vector(vec, eigen_vec, *max, N);
  std::cout << "maximum deviation in computing eigen vector " << max_dev
            << std::endl;

  uint *ret = (uint *)malloc(sizeof(uint) * 1);
  stop_criteria_test_success_data(q, vec, N, B, {}).wait();
  {
    buffer_1d buf_vec{vec, sycl::range<1>{N}};
    buffer<uint, 1> buf_ret{ret, range<1>{1}};

    stop(q, buf_vec, buf_ret, N, B, {}).wait();
  }
  std::cout << "stopping criteria test result [success]: " << *ret << std::endl;

  stop_criteria_test_fail_data(q, vec, N, B, {}).wait();
  {
    buffer_1d buf_vec{vec, sycl::range<1>{N}};
    buffer<uint, 1> buf_ret{ret, range<1>{1}};

    stop(q, buf_vec, buf_ret, N, B, {}).wait();
  }
  std::cout << "stopping criteria test result [fail]: " << *ret << std::endl;

  std::free(mat);
  std::free(vec);
  std::free(eigen_vec);

  mat = (float *)malloc(sizeof(float) * 3 * 3);
  eigen_vec = (float *)malloc(sizeof(float) * 3 * 1);
  float *eigen_val = (float *)malloc(sizeof(float) * 1);
  uint iter_count = 0;

  *(mat + 0 * 3 + 0) = 1;
  *(mat + 0 * 3 + 1) = 1;
  *(mat + 0 * 3 + 2) = 2;

  *(mat + 1 * 3 + 0) = 2;
  *(mat + 1 * 3 + 1) = 1;
  *(mat + 1 * 3 + 2) = 3;

  *(mat + 2 * 3 + 0) = 2;
  *(mat + 2 * 3 + 1) = 3;
  *(mat + 2 * 3 + 2) = 5;

  similarity_transform(q, mat, eigen_val, eigen_vec, 3, 3, &iter_count);

  assert(abs(*eigen_val - 7.53114) < EPS);
  assert(abs(*(eigen_vec + 0) - 0.394074) < EPS);
  assert(abs(*(eigen_vec + 1) - 0.578844) < EPS);
  assert(abs(*(eigen_vec + 2) - 0.997451) < EPS);
  std::cout << "sequential transform worked !\t\t[ " << iter_count << " iterations ]"
            << std::endl;

  std::free(mat);
  std::free(eigen_val);
  std::free(eigen_vec);

  return 0;
}
