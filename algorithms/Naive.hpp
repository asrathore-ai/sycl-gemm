#ifndef SyclGEMM_Naive_HPP
#define SyclGEMM_Naive_HPP

#include<sycl/sycl.hpp>

namespace SyclGEMM{

template<typename T>
struct NaiveGemm{

    const T* A;
    const T* B;
    T* C;
    const int M;
    const int N;
    const int K;
    const float alpha;
    const float beta;

    NaiveGemm(const T* a, const T* b, T* c, int m, int n, int k, float alpha, float beta)
    : A(a), B(b), C(c), M(m), N(n), K(k), alpha(alpha), beta(beta)
    {}

    void operator()(sycl::nd_item<2> work_item) const {
        int i = work_item.get_global_id(0);
        int j = work_item.get_global_id(1);
        
        if (i >= M || j >= N) {
            return;
        }
        
        T ab = 0;
        for(int k = 0; k < K; k++){
            ab += A[i*K + k]*B[k*N + j];
        }
        C[i*N + j] = alpha*ab + beta*C[i*N + j];
    }
};

} //SyclGEMM

#endif // SyclGEMM_Naive_HPP