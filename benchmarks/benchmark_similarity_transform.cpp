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
