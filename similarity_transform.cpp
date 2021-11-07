#include "similarity_transform.hpp"
#include <chrono>

using tp = std::chrono::_V2::steady_clock::time_point;

int64_t sequential_transform(sycl::queue &q, const float *mat,
                             float *const eigen_val, float *const eigen_vec,
                             const uint dim, const uint wg_size) {
  float *_mat = (float *)sycl::malloc_device(sizeof(float) * dim * dim, q);
  float *tmp_vec = (float *)sycl::malloc_device(sizeof(float) * dim, q);
  float *_eigen_vec = (float *)sycl::malloc_device(sizeof(float) * dim, q);
  float *max_elm = (float *)sycl::malloc_shared(sizeof(float) * 1, q);
  uint *ret = (uint *)sycl::malloc_shared(sizeof(uint) * 1, q);

  auto evt_0 = q.memcpy(_mat, mat, sizeof(float) * dim * dim);
  auto evt_1 = initialise_eigen_vector(q, _eigen_vec, dim, {});

  tp start = std::chrono::steady_clock::now();

  sycl::event evt;
  for (uint i = 0; i < MAX_ITR; i++) {
    sycl::event evt_2;
    if (i == 0) {
      evt_2 = sum_across_rows(q, _mat, tmp_vec, dim, wg_size, {evt_0});
    } else {
      evt_2 = sum_across_rows(q, _mat, tmp_vec, dim, wg_size, {evt});
    }

    auto evt_3 = find_max(q, tmp_vec, max_elm, dim, wg_size, {evt_2});

    sycl::event evt_4;
    if (i == 0) {
      evt_4 = compute_eigen_vector(q, tmp_vec, max_elm, _eigen_vec, dim,
                                   wg_size, {evt_1, evt_3});
    } else {
      evt_4 = compute_eigen_vector(q, tmp_vec, max_elm, _eigen_vec, dim,
                                   wg_size, {evt_3});
    }

    auto evt_5 = stop(q, tmp_vec, ret, dim, wg_size, {evt_3});
    evt_5.wait();

    if (*ret == 1) {
      evt = evt_4;
      break;
    }

    evt = compute_next_matrix(q, _mat, tmp_vec, dim, wg_size, {});
  }

  tp end = std::chrono::steady_clock::now();

  q.memcpy(eigen_val, tmp_vec, sizeof(float) * 1);
  evt.wait();

  evt = q.memcpy(eigen_vec, _eigen_vec, sizeof(float) * dim);
  sycl::free(tmp_vec, q);
  sycl::free(_mat, q);
  sycl::free(max_elm, q);
  sycl::free(ret, q);
  evt.wait();

  sycl::free(_eigen_vec, q);
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

sycl::event sum_across_rows(sycl::queue &q, const float *mat, float *const vec,
                            const uint count, const uint wg_size,
                            std::vector<sycl::event> evts) {
  auto evt_0 = q.memset(vec, 0, sizeof(float) * count);
  evts.push_back(evt_0);

  auto evt_1 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);

    if (q.get_device().is_gpu()) {
      sycl::accessor<float, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          lds(sycl::range<1>{1}, h);

      h.parallel_for<class kernelSumAcrossRowsGPU>(
          sycl::nd_range<2>{sycl::range<2>{count, count},
                            sycl::range<2>{1, wg_size}},
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
            const size_t r = it.get_global_id(0);
            const size_t c = it.get_global_id(1);

            float val = *(mat + r * count + c);

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
                  ref(*(vec + r));
              ref.fetch_add(lds[0]);
            }
          });
    } else {
      h.parallel_for<class kernelSumAcrossRowsOthers>(
          sycl::nd_range<2>{sycl::range<2>{count, count},
                            sycl::range<2>{1, wg_size}},
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
            const size_t r = it.get_global_id(0);
            const size_t c = it.get_global_id(1);

            float val = *(mat + r * count + c);
            sycl::ext::oneapi::atomic_ref<
                float, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::device,
                sycl::access::address_space::global_space>
                ref(*(vec + r));
            ref.fetch_add(val);
          });
    }
  });

  return evt_1;
}

sycl::event find_max(sycl::queue &q, const float *vec, float *const max,
                     const uint count, const uint wg_size,
                     std::vector<sycl::event> evts) {
  auto evt_0 = q.memset(max, 0, sizeof(float));
  evts.push_back(evt_0);

  auto evt_1 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelMaxInVector>(
        sycl::nd_range<1>{sycl::range<1>{count}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          sycl::ext::oneapi::atomic_ref<
              float, sycl::ext::oneapi::memory_order::relaxed,
              sycl::ext::oneapi::memory_scope::device,
              sycl::access::address_space::global_space>
              ref(*max);
          ref.fetch_max(*(vec + r));
        });
  });

  return evt_1;
}

sycl::event compute_eigen_vector(sycl::queue &q, const float *vec,
                                 const float *max, float *const eigen_vec,
                                 const uint count, const uint wg_size,
                                 std::vector<sycl::event> evts) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelComputeEigenVector>(
        sycl::nd_range<1>{sycl::range<1>{count}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const size_t r = it.get_global_id(0);

          *(eigen_vec + r) *= (*(vec + r) / *max);
        });
  });

  return evt;
}

sycl::event initialise_eigen_vector(sycl::queue &q, float *const vec,
                                    const uint count,
                                    std::vector<sycl::event> evts) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.fill(vec, 1.f, count);
  });

  return evt;
}

sycl::event compute_next_matrix(sycl::queue &q, float *const mat,
                                const float *sum_vec, const uint count,
                                const uint wg_size,
                                std::vector<sycl::event> evts) {
  auto evt = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelSimilarityTransform>(
        sycl::nd_range<2>{sycl::range<2>{count, count},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          float v0 = *(sum_vec + r);
          float v1 = r == c ? v0 : *(sum_vec + c);

          *(mat + r * count + c) *= (1.f / v0) * v1;
        });
  });

  return evt;
}

sycl::event stop(sycl::queue &q, const float *vec, uint *const ret,
                 const uint count, const uint wg_size,
                 std::vector<sycl::event> evts) {
  auto evt_0 = q.single_task([=]() {
    // == 1, denotes should stop !
    *ret = 1;
  });
  evts.push_back(evt_0);

  auto evt_1 = q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelStopCriteria>(
        sycl::nd_range<1>{sycl::range<1>{count}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          sycl::ext::oneapi::sub_group sg = it.get_sub_group();
          const size_t r = it.get_global_id(0);
          if (r == 0) {
            return;
          }

          float diff = sycl::abs(*(vec + r) - *(vec + (r - 1)));
          bool res = sycl::all_of_group(sg, diff < EPS);

          if (sycl::ext::oneapi::leader(sg)) {
            sycl::ext::oneapi::atomic_ref<
                uint, sycl::ext::oneapi::memory_order::relaxed,
                sycl::ext::oneapi::memory_scope::device,
                sycl::access::address_space::global_space>
                ref{*ret};
            ref.fetch_min(res ? 1 : 0);
          }
        });
  });

  return evt_1;
}
