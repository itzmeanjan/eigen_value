#include "similarity_transform.hpp"
#include <chrono>
#include <limits>

int64_t
similarity_transform(sycl::queue& q,
                     const float* mat,
                     float* const eigen_val,
                     float* const eigen_vec,
                     const uint dim,
                     const uint wg_size,
                     uint* const iter_count)
{
  float* mat_ = (float*)malloc(sizeof(float) * dim * dim);
  float* sum_vec = (float*)malloc(sizeof(float) * dim);
  float* max_elm = (float*)malloc(sizeof(float) * 1);
  uint* ret = (uint*)malloc(sizeof(uint) * 1);

  memcpy(mat_, mat, sizeof(float) * dim * dim);
  int64_t ts = 0;

  // just to automatically destroy buffers
  // putting in different scope, so that following
  // std::free doesn't segfault !
  {
    buffer_2d b_mat{ mat_, sycl::range<2>{ dim, dim } };
    buffer_1d b_eigen_vec{ eigen_vec, sycl::range<1>{ dim } };
    buffer_1d b_eigen_val{ eigen_val, sycl::range<1>{ 1 } };

    buffer_1d b_sum_vec{ sum_vec, sycl::range<1>{ dim } };
    buffer_1d b_max_elm{ max_elm, sycl::range<1>{ 1 } };
    sycl::buffer<uint, 1> b_ret{ ret, sycl::range<1>{ 1 } };

    initialise_eigen_vector(q, b_eigen_vec, dim, {});

    tp start = std::chrono::steady_clock::now();

    uint i = 0;
    for (; i < MAX_ITR; i++) {
      sum_across_rows(q, b_mat, b_sum_vec, dim, wg_size, {});
      find_max(q, b_sum_vec, b_max_elm, dim, wg_size, {});
      compute_eigen_vector(
        q, b_sum_vec, b_max_elm, b_eigen_vec, dim, wg_size, {});
      stop(q, b_sum_vec, b_ret, dim, wg_size, {});
      {
        sycl::host_accessor<uint, 1, sycl::access_mode::read> h_ret{ b_ret };
        if (h_ret[0] == 1) {
          break;
        }
      }

      compute_next_matrix(q, b_mat, b_sum_vec, dim, wg_size, {});
    }
    *iter_count = i;

    tp end = std::chrono::steady_clock::now();
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
           .count();

    q.submit([&](sycl::handler& h) {
      global_1d_reader acc_sum_vec{ b_sum_vec, h, sycl::range<1>{ 1 } };
      global_1d_writer acc_eigen_val{ b_eigen_val, h };

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

sycl::event
sum_across_rows(sycl::queue& q,
                buffer_2d mat,
                buffer_1d vec,
                const uint dim,
                const uint wg_size,
                std::vector<sycl::event> evts)
{
  q.submit([&](sycl::handler& h) {
    global_1d_writer acc_vec{ vec, h, sycl::no_init };

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.fill(acc_vec, 0.f);
  });

  auto evt = q.submit([&](sycl::handler& h) {
    global_2d_reader acc_mat{ mat, h };
    global_1d_reader_writer acc_vec{ vec, h };
    local_1d_reader_writer lds{ sycl::range<1>{ 1 }, h };

    h.parallel_for<class kernelSumAcrossAllRows>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
        sycl::group<2> grp = it.get_group();
        sycl::sub_group sg = it.get_sub_group();

        // let work group leader reset local memory
        if (sycl::ext::oneapi::leader(grp)) {
          lds[0] = 0.f;
        }

        // make sure everyone in work group has arrived here
        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        const size_t r = it.get_global_id(0);
        const size_t c = it.get_global_id(1);

        // compute sum of all subgroup elements, using reduction functionality
        float loc_sum =
          sycl::reduce_over_group(sg, acc_mat[r][c], sycl::plus<float>());

        // let subgroup leader atomically add subgroup-local-sum to local
        // memory
        if (sycl::ext::oneapi::leader(sg)) {
          sycl::ext::oneapi::atomic_ref<
            float,
            sycl::ext::oneapi::memory_order::relaxed,
            sycl::ext::oneapi::memory_scope::work_group,
            sycl::access::address_space::local_space>
            ref(lds[0]);
          ref.fetch_add(loc_sum);
        }

        // wait for all in work group to reach here
        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        // only leader of work group atomically adds workgroup-local-sum
        // to destination memory location in global memory
        if (sycl::ext::oneapi::leader(grp)) {
          sycl::ext::oneapi::atomic_ref<
            float,
            sycl::ext::oneapi::memory_order::relaxed,
            sycl::ext::oneapi::memory_scope::device,
            sycl::access::address_space::global_space>
            ref(acc_vec[r]);
          ref.fetch_add(lds[0]);
        }
      });
  });

  return evt;
}

sycl::event
find_max(sycl::queue& q,
         buffer_1d vec,
         buffer_1d max,
         const uint dim,
         const uint wg_size,
         std::vector<sycl::event> evts)
{
  q.submit([&](sycl::handler& h) {
    global_1d_writer acc_max{ max, h, sycl::no_init };

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.fill(acc_max, 0.f);
  });

  auto evt = q.submit([&](sycl::handler& h) {
    global_1d_reader acc_vec{ vec, h };
    global_1d_reader_writer acc_max{ max, h };
    local_1d_reader_writer lds{ sycl::range<1>{ 1 }, h };

    h.parallel_for<class kernelMaxInVector>(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
        sycl::group<1> grp = it.get_group();
        sycl::sub_group sg = it.get_sub_group();

        // get work group leader to reset local memory allocated
        if (sycl::ext::oneapi::leader(grp)) {
          lds[0] = 0.f;
        }

        // wait for all in work group to reach here
        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        const size_t r = it.get_global_id(0);

        // use reduction function to reduce to maximum value held by all
        // work-items present in current subgroup
        float loc_max =
          sycl::reduce_over_group(sg, acc_vec[r], sycl::maximum<float>());

        // subgroup leader atomically updates local memory
        if (sycl::ext::oneapi::leader(sg)) {
          sycl::ext::oneapi::atomic_ref<
            float,
            sycl::ext::oneapi::memory_order::relaxed,
            sycl::ext::oneapi::memory_scope::work_group,
            sycl::access::address_space::local_space>
            ref(lds[0]);
          ref.fetch_max(loc_max);
        }

        // wait for all in work group to reach here
        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        // only work group leader writes maximum value computed in this work
        // group to designated location in global memory
        if (sycl::ext::oneapi::leader(grp)) {
          sycl::ext::oneapi::atomic_ref<
            float,
            sycl::ext::oneapi::memory_order::relaxed,
            sycl::ext::oneapi::memory_scope::device,
            sycl::access::address_space::global_space>
            ref(acc_max[0]);
          ref.fetch_max(lds[0]);
        }
      });
  });

  return evt;
}

sycl::event
compute_eigen_vector(sycl::queue& q,
                     buffer_1d vec,
                     buffer_1d max,
                     buffer_1d eigen_vec,
                     const uint dim,
                     const uint wg_size,
                     std::vector<sycl::event> evts)
{
  auto evt = q.submit([&](sycl::handler& h) {
    global_1d_reader_writer acc_eigen_vec{ eigen_vec, h };
    global_1d_reader acc_vec{ vec, h };
    global_1d_reader acc_max{ max, h };

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.parallel_for<class kernelComputeEigenVector>(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
        sycl::ext::oneapi::sub_group sg = it.get_sub_group();
        const size_t r = it.get_global_id(0);

        float max_val = 0.f;
        if (sg.leader()) {
          max_val = acc_max[0];
        }
        sg.barrier();

        max_val = sycl::group_broadcast(sg, max_val);
        acc_eigen_vec[r] *= (acc_vec[r] / max_val);
      });
  });

  return evt;
}

sycl::event
initialise_eigen_vector(sycl::queue& q,
                        buffer_1d vec,
                        const uint dim,
                        std::vector<sycl::event> evts)
{
  auto evt = q.submit([&](sycl::handler& h) {
    global_1d_writer acc_vec{ vec, h, sycl::no_init };

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.fill(acc_vec, 1.f);
  });

  return evt;
}

sycl::event
compute_next_matrix(sycl::queue& q,
                    buffer_2d mat,
                    buffer_1d vec,
                    const uint dim,
                    const uint wg_size,
                    std::vector<sycl::event> evts)
{
  auto evt = q.submit([&](sycl::handler& h) {
    global_2d_reader_writer acc_mat{ mat, h };
    global_1d_reader acc_vec{ vec, h };
    local_1d_reader_writer acc_loc_row_ds{ sycl::range<1>{ 1 }, h };
    local_1d_reader_writer acc_loc_col_ds{ sycl::range<1>{ wg_size }, h };

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.parallel_for<class kernelSimilarityTransform>(
      sycl::nd_range<2>{ sycl::range<2>{ dim, dim },
                         sycl::range<2>{ 1, wg_size } },
      [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(32)]] {
        const size_t r = it.get_global_id(0);
        const size_t c = it.get_global_id(1);

        const size_t ll_id = it.get_local_linear_id();
        const size_t gl_id = it.get_global_linear_id();

        sycl::group<2> grp = it.get_group();
        sycl::sub_group sg = it.get_sub_group();

        if (sycl::ext::oneapi::leader(grp)) {
          acc_loc_row_ds[0] = acc_vec[r];
        }
        acc_loc_col_ds[ll_id] = acc_vec[gl_id % dim];

        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        acc_mat[r][c] *= (1.f / sycl::group_broadcast(sg, acc_loc_row_ds[0])) *
                         acc_loc_col_ds[ll_id];
      });
  });

  return evt;
}

sycl::event
stop(sycl::queue& q,
     buffer_1d vec,
     sycl::buffer<uint, 1> ret,
     const uint dim,
     const uint wg_size,
     std::vector<sycl::event> evts)
{
  using global_flag_reader_writer =
    sycl::accessor<uint,
                   1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>;
  using local_flag_reader_writer =
    sycl::accessor<uint,
                   1,
                   sycl::access::mode::read_write,
                   sycl::access::target::local>;

  q.submit([&](sycl::handler& h) {
    global_flag_reader_writer acc_ret{ ret, h, sycl::no_init };

    if (!evts.empty()) {
      h.depends_on(evts);
    }

    h.fill(acc_ret, 1U);
  });

  auto evt = q.submit([&](sycl::handler& h) {
    global_1d_reader acc_vec{ vec, h };
    global_flag_reader_writer acc_ret{ ret, h };
    local_flag_reader_writer lds{ sycl::range<1>{ 1 }, h };

    h.parallel_for<class kernelStopCriteria>(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
        sycl::group<1> grp = it.get_group();
        sycl::sub_group sg = it.get_sub_group();

        // first let work group leader set local memory
        // to initial required state
        if (sycl::ext::oneapi::leader(grp)) {
          lds[0] = 1U;
        }

        // wait for all work items in work group to reach here
        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        const size_t g_id = it.get_global_id(0);

        // read value at index `i`, if work item's index is `i`
        // i.e. just read own value
        float self = acc_vec[g_id];
        // use subgroup shuffling to obtain value
        // at index `i + 1`
        //
        // this is tricky !
        //
        // assume {0, 1, 2, 3} are values kept in 4 work items,
        // a leftwards shuffling, results into {1, 2, 3, 0}
        // reception sequence, for work items {0, 1, 2, 3}
        //
        // but this is only for subgroups, we've to think about
        // work groups
        //
        // say in our work group we've {0, 1, 2, 3, 4, 5, 6, 7}
        // and we're explicitly using subgroup size of 4
        // then after below shuffling work group level reception
        // sequence must look like {1, 2, 3, 0, 5, 6, 7, 4}
        //
        // I don't want that, I want to have it received this way
        // {1, 2, 3, 4, 5, 6, 7, 0}
        //
        // as next is not going to be what it's supposed to be
        // at subgroup boundary, thus I keep below conditional block
        // which exactly solves that problem by performing
        // an expensive global memory read
        //
        // I want to reduce global memory read/ write as much as possible
        // but this is it as of now !
        float next = sg.shuffle_down(self, 1);

        if (sg.get_local_id()[0] == (sg.get_local_range()[0] - 1)) {
          next = acc_vec[(g_id + 1) % dim];
        }

        float diff = sycl::abs(self - next);
        // check whether all good in subgroup level
        bool res = sycl::all_of_group(sg, diff < EPS);

        // only let subgroup leader update status and put it
        // in local memory
        // use atomic op, because I don't know yet whether
        // there's only one subgroup in work group or not
        //
        // anyway local memory is shared among work group participants
        // so all subgroups have access to it
        //
        // if not handled carefully might result into data race
        if (sycl::ext::oneapi::leader(sg)) {
          sycl::ext::oneapi::atomic_ref<
            uint,
            sycl::ext::oneapi::memory_order::relaxed,
            sycl::ext::oneapi::memory_scope::work_group,
            sycl::access::address_space::local_space>
            ref{ lds[0] };
          ref.fetch_min(res ? 1 : 0);
        }

        // wait for all work items in work group to reach this point
        sycl::group_barrier(grp, sycl::memory_scope::work_group);

        // finally only work group leader writes work group result
        // back to global memory, from local memory
        if (sycl::ext::oneapi::leader(grp)) {
          sycl::ext::oneapi::atomic_ref<
            uint,
            sycl::ext::oneapi::memory_order::relaxed,
            sycl::ext::oneapi::memory_scope::device,
            sycl::access::address_space::global_space>
            ref{ acc_ret[0] };
          ref.fetch_min(lds[0] ? 1 : 0);
        }
      });
  });

  return evt;
}
