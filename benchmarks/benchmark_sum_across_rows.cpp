#include <benchmarks.hpp>

int64_t benchmark_sum_across_rows_kernel(sycl::queue &q, const uint dim,
                                         const uint wg_size) {
  float *mat = (float *)malloc(sizeof(float) * dim * dim);
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  int64_t tm = 0;

  generate_random_positive_matrix(mat, dim);

  {
    buffer_2d buf_mat{mat, sycl::range<2>{dim, dim}};
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};

    tp start = std::chrono::steady_clock::now();
    sum_across_rows(q, buf_mat, buf_vec, dim, wg_size, {}).wait();
    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(mat);
  std::free(vec);

  return tm;
}
