#ifndef SyclGEMM_Tiled_HPP
#define SyclGEMM_Tiled_HPP

#include<sycl/sycl.hpp>

namespace SyclGEMM{

template<typename T, int TileDim>
struct TiledGEMM{

    const T* A;
    const T* B;
    T* C;
    const int M;
    const int N;
    const int K;

    sycl::local_accessor<T, 2> tile_A;
    sycl::local_accessor<T, 2> tile_B;

    TiledGEMM(T* a, T* b, T* c, int m, int n, int k,
    sycl::local_accessor<T, 2> tile_a, sycl::local_accessor<T, 2> tile_b)
    : A(a), B(b), C(c), M(m), N(n), K(k),
      tile_A(tile_a), tile_B(tile_b)  
    {}

    void operator()(sycl::nd_item<2> work_item) const {
        const int global_row = work_item.get_global_id(0);
        const int global_col = work_item.get_global_id(1);
        
        const int local_row = work_item.get_local_id(0);
        const int local_col = work_item.get_local_id(1);

        const int group_row = work_item.get_group(0);
        const int group_col = work_item.get_group(1);

        T accumulator = 0;

        for (int tile_k = 0; tile_k < K; tile_k += TileDim) {
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
            C[global_row * N + global_col] = accumulator +  C[global_row * N + global_col];
        }
    }
};

} //SyclGEMM

#endif // SyclGEMM_Tiled_HPP