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

int64_t benchmark_sum_across_rows_kernel_v0(sycl::queue &q, const uint dim,
                                            const uint wg_size) {
  float *mat = (float *)malloc(sizeof(float) * dim * dim);
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  int64_t tm = 0;

  generate_random_vector(mat, dim * dim);
  memset(vec, 0, sizeof(float) * dim);
  {
    buffer_2d buf_mat{mat, sycl::range<2>{dim, dim}};
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};

    tp start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
      global_2d_reader acc_mat{buf_mat, h};
      global_1d_reader_writer acc_vec{buf_vec, h};

      h.parallel_for<class kernelSumAcrossRowsv0>(
          sycl::nd_range<2>{sycl::range<2>{dim, dim},
                            sycl::range<2>{1, wg_size}},
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
            const size_t r = it.get_global_id(0);
            const size_t c = it.get_global_id(1);

            sycl::ext::oneapi::atomic_ref<
                float, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::device,
                sycl::access::address_space::global_space>
                ref(acc_vec[r]);
            ref.fetch_add(acc_mat[r][c]);
          });
    });
    q.wait();

    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(mat);
  std::free(vec);

  return tm;
}

int64_t benchmark_sum_across_rows_kernel_v1(sycl::queue &q, const uint dim,
                                            const uint wg_size) {
  float *mat = (float *)malloc(sizeof(float) * dim * dim);
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  int64_t tm = 0;

  generate_random_vector(mat, dim * dim);
  memset(vec, 0, sizeof(float) * dim);
  {
    buffer_2d buf_mat{mat, sycl::range<2>{dim, dim}};
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};

    tp start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
      global_2d_reader acc_mat{buf_mat, h};
      global_1d_reader_writer acc_vec{buf_vec, h};

      h.parallel_for<class kernelSumAcrossRowsv1>(
          sycl::nd_range<2>{sycl::range<2>{dim, dim},
                            sycl::range<2>{1, wg_size}},
          [=](sycl::nd_item<2> it) {
            sycl::sub_group sg = it.get_sub_group();

            const size_t r = it.get_global_id(0);
            const size_t c = it.get_global_id(1);

            float sg_sum =
                sycl::reduce_over_group(sg, acc_mat[r][c], sycl::plus<float>());

            if (sycl::ext::oneapi::leader(sg)) {
              sycl::ext::oneapi::atomic_ref<
                  float, sycl::ext::oneapi::memory_order::relaxed,
                  sycl::ext::oneapi::memory_scope::device,
                  sycl::access::address_space::global_space>
                  ref(acc_vec[r]);
              ref.fetch_add(sg_sum);
            }
          });
    });
    q.wait();

    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(mat);
  std::free(vec);

  return tm;
}

int64_t benchmark_sum_across_rows_kernel_v2(sycl::queue &q, const uint dim,
                                            const uint wg_size) {
  float *mat = (float *)malloc(sizeof(float) * dim * dim);
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  int64_t tm = 0;

  generate_random_vector(mat, dim * dim);
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

int64_t benchmark_find_vector_max_v0(sycl::queue &q, const uint dim,
                                     const uint wg_size) {
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  float *max = (float *)malloc(sizeof(float) * 1);
  int64_t tm = 0;

  generate_random_vector(vec, dim);
  {
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};
    buffer_1d buf_max{max, sycl::range<1>{1}};

    tp start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
      global_1d_reader acc_vec{buf_vec, h};
      global_1d_reader_writer acc_max{buf_max, h};

      h.parallel_for<class kernelMaxInVectorV0>(
          sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
          [=](sycl::nd_item<1> it) {
            const size_t r = it.get_global_id(0);

            sycl::ext::oneapi::atomic_ref<
                float, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::device,
                sycl::access::address_space::global_space>
                ref(acc_max[0]);
            ref.fetch_max(acc_vec[r]);
          });
    });
    q.wait();

    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(vec);
  std::free(max);

  return tm;
}

int64_t benchmark_find_vector_max_v1(sycl::queue &q, const uint dim,
                                     const uint wg_size) {
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  float *max = (float *)malloc(sizeof(float) * 1);
  int64_t tm = 0;

  generate_random_vector(vec, dim);
  {
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};
    buffer_1d buf_max{max, sycl::range<1>{1}};

    tp start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
      global_1d_reader acc_vec{buf_vec, h};
      global_1d_reader_writer acc_max{buf_max, h};

      h.parallel_for<class kernelMaxInVectorV1>(
          sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
          [=](sycl::nd_item<1> it) {
            sycl::sub_group sg = it.get_sub_group();

            const size_t r = it.get_global_id(0);

            float sg_max =
                sycl::reduce_over_group(sg, acc_vec[r], sycl::maximum<float>());

            if (sycl::ext::oneapi::leader(sg)) {
              sycl::ext::oneapi::atomic_ref<
                  float, sycl::ext::oneapi::memory_order::relaxed,
                  sycl::ext::oneapi::memory_scope::device,
                  sycl::access::address_space::global_space>
                  ref(acc_max[0]);
              ref.fetch_max(sg_max);
            }
          });
    });
    q.wait();

    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(vec);
  std::free(max);

  return tm;
}

int64_t benchmark_find_vector_max_v2(sycl::queue &q, const uint dim,
                                     const uint wg_size) {
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  float *max = (float *)malloc(sizeof(float) * 1);
  int64_t tm = 0;

  generate_random_vector(vec, dim);
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

int64_t benchmark_compute_eigen_vector_v0(sycl::queue &q, const uint dim,
                                          const uint wg_size) {
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  float *eigen_vec = (float *)malloc(sizeof(float) * dim * 1);
  float *max = (float *)malloc(sizeof(float) * 1);
  int64_t tm = 0;

  generate_random_vector(vec, dim);
  {
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};
    buffer_1d buf_eigen_vec{eigen_vec, sycl::range<1>{dim}};
    buffer_1d buf_max{max, sycl::range<1>{1}};

    find_max(q, buf_vec, buf_max, dim, wg_size, {}).wait();
    initialise_eigen_vector(q, buf_eigen_vec, dim, {}).wait();

    tp start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &h) {
      global_1d_reader_writer acc_eigen_vec{buf_eigen_vec, h};
      global_1d_reader acc_vec{buf_vec, h};
      global_1d_reader acc_max{buf_max, h};

      h.parallel_for<class kernelComputeEigenVectorV0>(
          sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}}, [=
      ](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
            const size_t r = it.get_global_id(0);
            acc_eigen_vec[r] *= (acc_vec[r] / acc_max[0]);
          });
    });
    q.wait();
    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(vec);
  std::free(eigen_vec);
  std::free(max);

  return tm;
}

int64_t benchmark_compute_eigen_vector_v1(sycl::queue &q, const uint dim,
                                          const uint wg_size) {
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  float *eigen_vec = (float *)malloc(sizeof(float) * dim * 1);
  float *max = (float *)malloc(sizeof(float) * 1);
  int64_t tm = 0;

  generate_random_vector(vec, dim);
  {
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};
    buffer_1d buf_eigen_vec{eigen_vec, sycl::range<1>{dim}};
    buffer_1d buf_max{max, sycl::range<1>{1}};

    find_max(q, buf_vec, buf_max, dim, wg_size, {}).wait();
    initialise_eigen_vector(q, buf_eigen_vec, dim, {}).wait();

    tp start = std::chrono::steady_clock::now();
    compute_eigen_vector(q, buf_vec, buf_max, buf_eigen_vec, dim, wg_size, {})
        .wait();
    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(vec);
  std::free(eigen_vec);
  std::free(max);

  return tm;
}

int64_t benchmark_compute_next_matrix(sycl::queue &q, const uint dim,
                                      const uint wg_size) {
  float *mat = (float *)malloc(sizeof(float) * dim * dim);
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  float *eigen_vec = (float *)malloc(sizeof(float) * dim * 1);
  float *max = (float *)malloc(sizeof(float) * 1);
  int64_t tm = 0;

  generate_random_vector(mat, dim * dim);
  {
    buffer_2d buf_mat{mat, sycl::range<2>{dim, dim}};
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};
    buffer_1d buf_eigen_vec{eigen_vec, sycl::range<1>{dim}};
    buffer_1d buf_max{max, sycl::range<1>{1}};

    initialise_eigen_vector(q, buf_eigen_vec, dim, {}).wait();
    sum_across_rows(q, buf_mat, buf_vec, dim, wg_size, {}).wait();
    find_max(q, buf_vec, buf_max, dim, wg_size, {}).wait();
    compute_eigen_vector(q, buf_vec, buf_max, buf_eigen_vec, dim, wg_size, {})
        .wait();

    tp start = std::chrono::steady_clock::now();
    compute_next_matrix(q, buf_mat, buf_vec, dim, wg_size, {}).wait();
    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(mat);
  std::free(vec);
  std::free(eigen_vec);
  std::free(max);

  return tm;
}

int64_t benchmark_stop_criteria_tester(sycl::queue &q, const uint dim,
                                       const uint wg_size) {
  float *vec = (float *)malloc(sizeof(float) * dim * 1);
  uint *ret = (uint *)malloc(sizeof(uint) * 1);
  int64_t tm = 0;

  generate_random_vector(vec, dim);
  {
    buffer_1d buf_vec{vec, sycl::range<1>{dim}};
    sycl::buffer<uint, 1> buf_ret{ret, sycl::range<1>{1}};

    tp start = std::chrono::steady_clock::now();
    stop(q, buf_vec, buf_ret, dim, wg_size, {}).wait();
    tp end = std::chrono::steady_clock::now();

    tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count();
  }

  std::free(vec);
  std::free(ret);

  return tm;
}
