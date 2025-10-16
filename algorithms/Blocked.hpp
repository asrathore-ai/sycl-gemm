#ifndef SyclGEMM_Blocked_HPP
#define SyclGEMM_Blocked_HPP

#include<sycl/sycl.hpp>

namespace SyclGEMM{

template<typename T>
struct BlockedGemm{

    const T* A;
    const T* B;
    T* C;
    const int M;
    const int N;
    const int K;
    const float alpha;
    const float beta;

    BlockedGemm(T* a, T* b, T* c, int m, int n, int k, float alpha, float beta)
    : A(a), B(b), C(c), M(m), N(n), K(k), alpha(alpha), beta(beta)
    {}

    void operator()(sycl::nd_item<2> work_item) const {
        
        
    }
};

} //SyclGEMM

#endif // SyclGEMM_Blocked_HPP