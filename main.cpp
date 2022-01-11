#include <benchmarks.hpp>
#include <iomanip>
#include <iostream>

using namespace sycl;

int
main()
{
  device d{ default_selector{} };
  context c{ d };
  queue q{ c, d };

  const size_t max_wg_size =
    d.get_info<info::device::max_work_group_size>() >> 1;

  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;
  std::cout << "Parallel Similarity Transform for finding max "
               "eigen value (with vector)\n"
            << std::endl;

  for (uint i = 7; i <= 13; i++) {
    const uint dim = 1ul << i;

    uint itr_count = 0;
    int64_t tm = benchmark_similarity_transform(
      q, dim, dim <= max_wg_size ? dim : max_wg_size, &itr_count);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << tm << " ms"
              << "\t\t\t" << std::setw(6) << std::right << itr_count
              << " round(s)" << std::endl;
  }

  std::cout << "\n[kernel] Sum Across Rows of Matrix (v0)\n" << std::endl;

  for (uint i = 7; i <= 13; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_sum_across_rows_kernel_v0(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Sum Across Rows of Matrix (v1)\n" << std::endl;

  for (uint i = 7; i <= 13; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_sum_across_rows_kernel_v1(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Sum Across Rows of Matrix (v2)\n" << std::endl;

  for (uint i = 7; i <= 13; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_sum_across_rows_kernel_v2(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Max Value in Vector (v0)\n" << std::endl;

  for (uint i = 16; i <= 25; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_find_vector_max_v0(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(9) << std::right << dim << "\t\t\t" << std::setw(10)
              << std::right << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Max Value in Vector (v1)\n" << std::endl;

  for (uint i = 16; i <= 25; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_find_vector_max_v1(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(9) << std::right << dim << "\t\t\t" << std::setw(10)
              << std::right << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Max Value in Vector (v2)\n" << std::endl;

  for (uint i = 16; i <= 25; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_find_vector_max_v2(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(9) << std::right << dim << "\t\t\t" << std::setw(10)
              << std::right << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Eigen Vector Computation (v0)\n" << std::endl;

  for (uint i = 16; i <= 25; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_compute_eigen_vector_v0(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(9) << std::right << dim << "\t\t\t" << std::setw(10)
              << std::right << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Eigen Vector Computation (v1)\n" << std::endl;

  for (uint i = 16; i <= 25; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_compute_eigen_vector_v1(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(9) << std::right << dim << "\t\t\t" << std::setw(10)
              << std::right << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Next Matrix Computation\n" << std::endl;

  for (uint i = 7; i <= 13; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_compute_next_matrix(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(5) << std::left << dim << "x" << std::setw(5)
              << std::right << dim << "\t\t\t" << std::setw(10) << std::right
              << (double)tm * 1e-3 << " ms" << std::endl;
  }

  std::cout << "\n[kernel] Stop Criteria Checker\n" << std::endl;

  for (uint i = 16; i <= 25; i++) {
    const uint dim = 1ul << i;

    int64_t tm = benchmark_stop_criteria_tester(
      q, dim, dim <= max_wg_size ? dim : max_wg_size);

    std::cout << std::setw(9) << std::right << dim << "\t\t\t" << std::setw(10)
              << std::right << (double)tm * 1e-3 << " ms" << std::endl;
  }

  return 0;
}
