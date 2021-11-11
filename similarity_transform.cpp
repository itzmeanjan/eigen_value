#include "similarity_transform.hpp"
#include <chrono>
#include <limits>

int64_t similarity_transform(sycl::queue &q, const float *mat,
                             float *const eigen_val, float *const eigen_vec,
                             const uint dim, const uint wg_size,
                             uint *const iter_count) {
  float *mat_ = (float *)malloc(sizeof(float) * dim * dim);
  float *sum_vec = (float *)malloc(sizeof(float) * dim);
  float *max_elm = (float *)malloc(sizeof(float) * 1);
  uint *ret = (uint *)malloc(sizeof(uint) * 1);

  memcpy(mat_, mat, sizeof(float) * dim * dim);
  int64_t ts = 0;

  // just to automatically destroy buffers
  // putting in different scope, so that following
  // std::free doesn't segfault !
  {
    buffer_2d b_mat{mat_, sycl::range<2>{dim, dim}};
    buffer_1d b_eigen_vec{eigen_vec, sycl::range<1>{dim}};
    buffer_1d b_eigen_val{eigen_val, sycl::range<1>{1}};

    buffer_1d b_sum_vec{sum_vec, sycl::range<1>{dim}};
    buffer_1d b_max_elm{max_elm, sycl::range<1>{1}};
    sycl::buffer<uint, 1> b_ret{ret, sycl::range<1>{1}};

    initialise_eigen_vector(q, b_eigen_vec, dim, {});

    tp start = std::chrono::steady_clock::now();

    uint i = 0;
    for (; i < MAX_ITR; i++) {
      sum_across_rows(q, b_mat, b_sum_vec, dim, wg_size, {});
      find_max(q, b_sum_vec, b_max_elm, dim, wg_size, {});
      compute_eigen_vector(q, b_sum_vec, b_max_elm, b_eigen_vec, dim, wg_size,
                           {});
      stop(q, b_sum_vec, b_ret, dim, wg_size, {});
      {
        sycl::host_accessor<uint, 1, sycl::access_mode::read> h_ret{b_ret};
        if (h_ret[0] == 1) {
          break;
        }
      }

      compute_next_matrix(q, b_mat, b_sum_vec, dim, wg_size, {});
    }
    *iter_count = i;

    q.wait();
    tp end = std::chrono::steady_clock::now();
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count();

    q.submit([&](sycl::handler &h) {
      global_1d_reader acc_sum_vec{b_sum_vec, h, sycl::range<1>{1}};
      global_1d_writer acc_eigen_val{b_eigen_val, h};

      h.copy(acc_sum_vec, acc_eigen_val);
    });
    q.wait();
  }

  std::free(mat_);
  std::free(sum_vec);
  std::free(max_elm);
  std::free(ret);

  return ts;
}

sycl::event sum_across_rows(sycl::queue &q, buffer_2d mat, buffer_1d vec,
                            const uint dim, const uint wg_size,
                            std::vector<sycl::event> evts) {
  q.submit([&](sycl::handler &h) {
    global_1d_writer acc_vec{vec, h, sycl::no_init};

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.fill(acc_vec, 0.f);
  });

  auto evt = q.submit([&](sycl::handler &h) {
    global_2d_reader acc_mat{mat, h};
    global_1d_reader_writer acc_vec{vec, h};

    h.parallel_for<class kernelSumAcrossRowsOthers>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
          sycl::sub_group sg = it.get_sub_group();
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          float loc_sum =
              sycl::reduce_over_group(sg, acc_mat[r][c], sycl::plus<float>());

          if (sycl::ext::oneapi::leader(sg)) {
            sycl::ext::oneapi::atomic_ref<
                float, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::device,
                sycl::access::address_space::global_space>
                ref(acc_vec[r]);
            ref.fetch_add(loc_sum);
          }
        });
  });

  return evt;
}

sycl::event find_max(sycl::queue &q, buffer_1d vec, buffer_1d max,
                     const uint dim, const uint wg_size,
                     std::vector<sycl::event> evts) {
  q.submit([&](sycl::handler &h) {
    global_1d_writer acc_max{max, h, sycl::no_init};

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.fill(acc_max, 0.f);
  });

  auto evt = q.submit([&](sycl::handler &h) {
    global_1d_reader acc_vec{vec, h};
    global_1d_reader_writer acc_max{max, h};

    h.parallel_for<class kernelMaxInVector>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}}, [=
    ](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
          sycl::sub_group sg = it.get_sub_group();
          const size_t r = it.get_global_id(0);

          float loc_max =
              sycl::reduce_over_group(sg, acc_vec[r], sycl::maximum<float>());

          if (sycl::ext::oneapi::leader(sg)) {
            sycl::ext::oneapi::atomic_ref<
                float, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::device,
                sycl::access::address_space::global_space>
                ref(acc_max[0]);
            ref.fetch_max(loc_max);
          }
        });
  });

  return evt;
}

sycl::event compute_eigen_vector(sycl::queue &q, buffer_1d vec, buffer_1d max,
                                 buffer_1d eigen_vec, const uint dim,
                                 const uint wg_size,
                                 std::vector<sycl::event> evts) {
  auto evt = q.submit([&](sycl::handler &h) {
    global_1d_reader_writer acc_eigen_vec{eigen_vec, h};
    global_1d_reader acc_vec{vec, h};
    global_1d_reader acc_max{max, h};

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.parallel_for<class kernelComputeEigenVector>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}}, [=
    ](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
          sycl::ext::oneapi::sub_group sg = it.get_sub_group();

          const size_t r = it.get_global_id(0);
          acc_eigen_vec[r] *=
              (acc_vec[r] / sycl::group_broadcast(sg, acc_max[0]));
        });
  });

  return evt;
}

sycl::event initialise_eigen_vector(sycl::queue &q, buffer_1d vec,
                                    const uint dim,
                                    std::vector<sycl::event> evts) {
  auto evt = q.submit([&](sycl::handler &h) {
    global_1d_writer acc_vec{vec, h, sycl::no_init};

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.fill(acc_vec, 1.f);
  });

  return evt;
}

sycl::event compute_next_matrix(sycl::queue &q, buffer_2d mat, buffer_1d vec,
                                const uint dim, const uint wg_size,
                                std::vector<sycl::event> evts) {
  auto evt = q.submit([&](sycl::handler &h) {
    global_2d_reader_writer acc_mat{mat, h};
    global_1d_reader acc_vec{vec, h};
    local_1d_reader_writer acc_loc_row_ds{sycl::range<1>{1}, h};
    local_1d_reader_writer acc_loc_col_ds{sycl::range<1>{wg_size}, h};

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.parallel_for<class kernelSimilarityTransform>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);
          const size_t ll_id = it.get_local_linear_id();
          const size_t gl_id = it.get_global_linear_id();
          sycl::sub_group sg = it.get_sub_group();

          if (sycl::ext::oneapi::leader(sg)) {
            acc_loc_row_ds[0] = acc_vec[r];
          }

          acc_loc_col_ds[ll_id] = acc_vec[gl_id % dim];

          it.barrier(sycl::access::fence_space::local_space);

          acc_mat[r][c] *= (1.f / acc_loc_row_ds[0]) * acc_loc_col_ds[ll_id];
        });
  });

  return evt;
}

sycl::event stop(sycl::queue &q, buffer_1d vec, sycl::buffer<uint, 1> ret,
                 const uint dim, const uint wg_size,
                 std::vector<sycl::event> evts) {
  using global_flag_reader_writer =
      sycl::accessor<uint, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer>;

  q.submit([&](sycl::handler &h) {
    global_flag_reader_writer acc_ret{ret, h, sycl::no_init};

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.fill(acc_ret, 1U);
  });

  auto evt = q.submit([&](sycl::handler &h) {
    global_1d_reader acc_vec{vec, h};
    global_flag_reader_writer acc_ret{ret, h};
    local_1d_reader_writer acc_loc_ds{sycl::range<1>{wg_size}, h};

    h.parallel_for<class kernelStopCriteria>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}}, [=
    ](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
          sycl::sub_group sg = it.get_sub_group();
          const size_t g_id = it.get_global_id(0);
          const size_t l_id = it.get_local_id(0);

          acc_loc_ds[l_id] = acc_vec[g_id];

          it.barrier(sycl::access::fence_space::local_space);

          float diff =
              sycl::abs((l_id == wg_size - 1 ? acc_vec[(g_id + 1) % dim]
                                             : acc_loc_ds[l_id + 1]) -
                        acc_loc_ds[l_id]);
          bool res = sycl::all_of_group(sg, diff < EPS);

          if (sycl::ext::oneapi::leader(sg)) {
            sycl::ext::oneapi::atomic_ref<
                uint, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::device,
                sycl::access::address_space::global_space>
                ref{acc_ret[0]};
            ref.fetch_min(res ? 1 : 0);
          }
        });
  });

  return evt;
}
