#include <benchmarks.hpp>

int64_t benchmark_similarity_transform(sycl::queue &q, const uint dim,
                                       const uint wg_size,
                                       uint *const itr_count) {
  float *mat = (float *)malloc(sizeof(float) * dim * dim);
  float *eigen_val = (float *)malloc(sizeof(float) * 1);
  float *eigen_vec = (float *)malloc(sizeof(float) * dim * 1);

  generate_hilbert_matrix(q, mat, dim);
  int64_t tm = similarity_transform(q, mat, eigen_val, eigen_vec, dim, wg_size,
                                    itr_count);

  std::free(mat);
  std::free(eigen_val);
  std::free(eigen_vec);

  return tm;
}

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

int64_t benchmark_find_vector_max(sycl::queue &q, const uint dim,
                                  const uint wg_size) {
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  float *max = (float *)malloc(sizeof(float) * 1);
  int64_t tm = 0;

  generate_vector(q, vec, dim, wg_size, {}).wait();

  {
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};
    buffer_1d buf_max{max, sycl::range<1>{1}};

    tp start = std::chrono::steady_clock::now();
    find_max(q, buf_vec, buf_max, dim, wg_size, {}).wait();
    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(vec);
  std::free(max);

  return tm;
}
