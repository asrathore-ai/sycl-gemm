#ifndef SyclGEMM_Reference_HPP
#define SyclGEMM_Reference_HPP

namespace SyclGEMM{

    template<typename T>
    void reference_gemm(const T *const A, const T *const B, T* C, int M, int N, int K){
        // A:M*K
        // B:K*N
        // C:M*N
        // C = AB + C;

        for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
                T ab = 0;
                for(int k = 0; k < K; k++){
                    ab += A[i*K + k]*B[k*N + j];
                }
                C[i*N + j] = ab + C[i*N + j];
            }
        }
    }

} //SyclGEMM

#endif //SyclGEMM_Reference_HPP