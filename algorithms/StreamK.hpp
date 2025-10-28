#ifndef SyclGEMM_StreamK_HPP
#define SyclGEMM_StreamK_HPP

#include<sycl/sycl.hpp>

namespace SyclGEMM{

template<typename T>
struct StreamKGEMM{

    const T* A;
    const T* B;
    T* C;
    const int M;
    const int N;
    const int K;

    StreamKGEMM(T* a, T* b, T* c, int m, int n, int k)
    : A(a), B(b), C(c), M(m), N(n), K(k)
    {}

    void operator()(sycl::nd_item<1> work_item) const {
        const size_t work_id = work_item.get_global_id(0);
        const size_t total_work_items = M * N * K;

        if (work_id >= total_work_items) {
            return;
        }

        const size_t k = work_id % K;
        const size_t j = (work_id / K) % N;
        const size_t i = work_id / (K * N);
        
        T value = A[i * K + k] * B[k * N + j];
        auto atomic_c = sycl::atomic_ref<T,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(C[i * N + j]);
        atomic_c.fetch_add(value);
    }
};

} //SyclGEMM

#endif // SyclGEMM_StreamK_HPP