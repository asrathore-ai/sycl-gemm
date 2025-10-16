#ifndef SyclGEMM_SplitK_HPP
#define SyclGEMM_SplitK_HPP

#include<sycl/sycl.hpp>

namespace SyclGEMM{

template<typename T, int TileDim, int SplitFactor>
struct SplitKGEMM{

    const T* A;
    const T* B;
    T* C;
    const int M;
    const int N;
    const int K;
    const float alpha;
    const float beta;

    sycl::local_accessor<T, 2> tile_A;
    sycl::local_accessor<T, 2> tile_B;

    SplitKGEMM(T* a, T* b, T* c, int m, int n, int k, float alpha, float beta,
    sycl::local_accessor<T, 2> tile_a, sycl::local_accessor<T, 2> tile_b)
    : A(a), B(b), C(c), M(m), N(n), K(k), alpha(alpha), beta(beta),
      tile_A(tile_a), tile_B(tile_b)  
    {}

    void operator()(sycl::nd_item<3> work_item) const {
        const int global_row = work_item.get_global_id(0);
        const int global_col = work_item.get_global_id(1);
        
        const int local_row = work_item.get_local_id(0);
        const int local_col = work_item.get_local_id(1);

        const int group_row = work_item.get_group(0);
        const int group_col = work_item.get_group(1);
        const int group_k_split = work_item.get_group(2);

        const int k_chunk_size = (K + SplitFactor - 1) / SplitFactor;
        const int k_start = group_k_split * k_chunk_size;
        const int k_end = sycl::min((group_k_split + 1) * k_chunk_size, K);
        
        T accumulator = 0;

        for (int tile_k = k_start; tile_k < k_end; tile_k += TileDim) {
            const int a_row_idx = group_row * TileDim + local_row;
            const int a_col_idx = tile_k + local_col;
            const int b_row_idx = tile_k + local_row;
            const int b_col_idx = group_col * TileDim + local_col;

            if (a_row_idx < M && a_col_idx < K) {
                tile_A[local_row][local_col] = A[a_row_idx * K + a_col_idx];
            } else {
                tile_A[local_row][local_col] = 0;
            }

            if (b_row_idx < K && b_col_idx < N) {
                tile_B[local_row][local_col] = B[b_row_idx * N + b_col_idx];
            } else {
                tile_B[local_row][local_col] = 0;
            }

            work_item.barrier(sycl::access::fence_space::local_space);

            for (int k_inner = 0; k_inner < TileDim; ++k_inner) {
                accumulator += tile_A[local_row][k_inner] * tile_B[k_inner][local_col];
            }

            work_item.barrier(sycl::access::fence_space::local_space);

        }

        if (global_row < M && global_col < N) {
            auto atomic_c = sycl::atomic_ref<T, 
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                (C[global_row * N + global_col]);

            atomic_c.fetch_add(alpha * accumulator);
        }
    }
};

} //SyclGEMM

#endif // SyclGEMM_SplitK_HPP