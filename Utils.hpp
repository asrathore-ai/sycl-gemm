#ifndef SyclGEMM_Utils_HPP
#define SyclGEMM_Utils_HPP

#include<sycl/sycl.hpp>
#include<random>
#include<iostream>
#include <sstream>

template<typename T, int lo, int hi>
inline float get_random() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(lo, hi);
    return static_cast<T>(dis(e));
}


namespace SyclGEMM{

    // Create a uniform random matrix filled with numbers in range lo to hi
    template<typename T, int lo, int hi>
    struct RandomMatrix{
        sycl::queue& Q;
        const int size;
        T* host_ptr;
        T* device_ptr;

        RandomMatrix(sycl::queue& q, int sz)
        : Q(q), size(sz), host_ptr(new T[size]),
          device_ptr( sycl::malloc_device<T>(size, Q) )
        {
            for(int i = 0; i < size; ++i) host_ptr[i] = get_random<T, lo, hi>();
            Q.memcpy(device_ptr, host_ptr, size * sizeof(T)).wait();
        }

        void assert_host_device_equality(T tolerance=static_cast<T>(0.0001f)){
            T* cpu_copy = new T[size];
            Q.memcpy(cpu_copy, device_ptr, size * sizeof(T)).wait();

            int mismatch_counts = 0;
            T max_diff = static_cast<T>(-1.0f);
            for(int i = 0; i < size; i++){
                T abs_diff = std::abs( cpu_copy[i] - host_ptr[i]);
                if (abs_diff > tolerance){
                    mismatch_counts++;
                    max_diff = std::max( max_diff, abs_diff);
                }
            }

            if(mismatch_counts > 0) {
                std::cout << "Mismatches = " << mismatch_counts << " / " << size << '\n';
                std::cout << "Max difference = " << max_diff << std::endl;
            } else {
                std::cout << " Test Passed !" << std::endl;
            }

            delete[] cpu_copy;
        }

        ~RandomMatrix(){
            delete[] host_ptr;
            sycl::free(device_ptr, Q);
        }

        RandomMatrix(const RandomMatrix&) = delete;
        RandomMatrix& operator=(const RandomMatrix&) = delete;
        RandomMatrix(RandomMatrix&&) = delete;
        RandomMatrix& operator=(RandomMatrix&&) = delete;
    };

    template<typename T>
    struct Context{
        sycl::queue& Q;
        const int M;
        const int N;
        const int K;
        const float alpha;
        const float beta;

        RandomMatrix<T, 50, 100> A;
        RandomMatrix<T, 50, 100> B;
        RandomMatrix<T, 50, 100> C;

        Context(sycl::queue& q, int m, int n, int k)
        : Q(q), 
          M(m), N(n), K(k), 
          alpha(get_random<float, 0, 1>()), beta(get_random<float, 0, 1>()),
          A(Q, M*K),
          B(Q, K*N),
          C(Q, M*N)
        {}
    };

} // SyclGEMM

namespace SyclGEMM{

    std::ostream& operator<<(std::ostream& os, const sycl::range<1>& obj){
        os << "[" << obj[0] << "]";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const sycl::range<2>& obj){
        os << "[" << obj[0] << "," << obj[1] <<  "]";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const sycl::range<3>& obj){
        os << "[" << obj[0] << "," << obj[1] << "," << obj[2] << "]";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const sycl::info::local_mem_type& obj){
        switch(obj){
            case sycl::info::local_mem_type::none :
                os << "None";
                break;

            case sycl::info::local_mem_type::local :
                os << "Local";
                break;

            case sycl::info::local_mem_type::global :
                os << "Global";    
                break;

            default:
                os << "Unknown Local Memory Type";
        }
        return os;
    }

    struct PropertyDisplay{
        sycl::device _d;
        std::stringstream _ss;

        PropertyDisplay(sycl::device&& device) : 
            _d(device)
        {}

        template<typename P>
        void query_property(const std::string& query_tag){
            _ss << query_tag << " : " << _d.get_info<P>() << "\n"; 
        }

        std::string to_str() {
            return _ss.str();
        }
    };

    void print_device_info(sycl::queue& q){
        PropertyDisplay P(q.get_device());

        P.query_property<sycl::info::device::name>("Running On Device");
        P.query_property<sycl::info::device::max_work_group_size>("Max Work Group Size");
        P.query_property<sycl::info::device::max_work_item_sizes<1>>("Max Work Item Size (1D)");
        P.query_property<sycl::info::device::max_work_item_sizes<2>>("Max Work Item Size (2D)");
        P.query_property<sycl::info::device::max_work_item_sizes<3>>("Max Work Item Size (3D)");
        P.query_property<sycl::info::device::max_mem_alloc_size>("Max Allocatable memory size (bytes)");
        P.query_property<sycl::info::device::local_mem_type>("Local Memory Type");
        P.query_property<sycl::info::device::local_mem_size>("Local memory size (bytes)");

        std::cout << P.to_str() << std::endl;
    }

} // SyclGEMM

#endif //SyclGEMM_Utils_HPP