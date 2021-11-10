#include "similarity_transform.hpp"
#include <chrono>

int64_t sequential_transform(sycl::queue &q, const float *mat,
                             float *const eigen_val, float *const eigen_vec,
                             const uint dim, const uint wg_size,
                             uint *const iter_count) {
  float *mat_ = (float *)malloc(sizeof(float) * dim * dim);
  float *sum_vec = (float *)malloc(sizeof(float) * dim);
  float *max_elm = (float *)malloc(sizeof(float) * 1);
  uint *ret = (uint *)malloc(sizeof(uint) * 1);

  memcpy(mat_, mat, sizeof(float) * dim * dim);

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

  q.submit([&](sycl::handler &h) {
    global_1d_reader acc_sum_vec{b_sum_vec, h, sycl::range<1>{1}};
    global_1d_writer acc_eigen_val{b_eigen_val, h};

    h.copy(acc_sum_vec, acc_eigen_val);
  });
  q.wait();

  std::free(mat_);
  std::free(sum_vec);
  std::free(max_elm);
  std::free(ret);

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
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

    if (q.get_device().is_gpu()) {
      local_1d_reader_writer lds(sycl::range<1>{1}, h);

      h.parallel_for<class kernelSumAcrossRowsGPU>(
          sycl::nd_range<2>{sycl::range<2>{dim, dim},
                            sycl::range<2>{1, wg_size}},
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
            const size_t r = it.get_global_id(0);
            const size_t c = it.get_global_id(1);

            float val = acc_mat[r][c];

            const size_t loc_id = it.get_local_linear_id();
            if (loc_id == 0) {
              lds[0] = 0;
            }

            it.barrier(sycl::access::fence_space::local_space);

            sycl::ext::oneapi::atomic_ref<
                float, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::work_group,
                sycl::access::address_space::local_space>
                ref(lds[0]);
            ref.fetch_add(val);

            it.barrier(sycl::access::fence_space::local_space);

            if (loc_id == 0) {
              sycl::ext::oneapi::atomic_ref<
                  float, sycl::ext::oneapi::memory_order::relaxed,
                  sycl::ext::oneapi::memory_scope::device,
                  sycl::access::address_space::global_space>
                  ref(acc_vec[r]);
              ref.fetch_add(lds[0]);
            }
          });
    } else {
      h.parallel_for<class kernelSumAcrossRowsOthers>(
          sycl::nd_range<2>{sycl::range<2>{dim, dim},
                            sycl::range<2>{1, wg_size}},
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
            const size_t r = it.get_global_id(0);
            const size_t c = it.get_global_id(1);

            float val = acc_mat[r][c];
            sycl::ext::oneapi::atomic_ref<
                float, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::device,
                sycl::access::address_space::global_space>
                ref(acc_vec[r]);
            ref.fetch_add(val);
          });
    }
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
          const size_t r = it.get_global_id(0);

          float val = acc_vec[r];

          sycl::ext::oneapi::atomic_ref<
              float, sycl::ext::oneapi::memory_order::relaxed,
              sycl::ext::oneapi::memory_scope::device,
              sycl::access::address_space::global_space>
              ref(acc_max[0]);
          ref.fetch_max(val);
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
              (acc_vec[r] / sg.broadcast(acc_max[0], it.get_local_id(0)));
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

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.parallel_for<class kernelSimilarityTransform>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          float v0 = acc_vec[r];
          float v1 = r == c ? v0 : acc_vec[c];

          acc_mat[r][c] *= (1.f / v0) * v1;
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

    h.parallel_for<class kernelStopCriteria>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}}, [=
    ](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
          sycl::ext::oneapi::sub_group sg = it.get_sub_group();
          const size_t r = it.get_global_id(0);
          if (r == 0) {
            return;
          }

          float diff = sycl::abs(acc_vec[r] - acc_vec[r - 1]);
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
