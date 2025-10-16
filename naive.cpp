#include "Utils.hpp"
#include "algorithms/Reference.hpp"
#include "algorithms/Naive.hpp"

constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;
constexpr int wgp_size_m = 32;
constexpr int wgp_size_n = 32;

int main(){
    sycl::queue q{sycl::gpu_selector_v};
    SyclGEMM::print_device_info(q);

    using opT = float;

    SyclGEMM::Context<opT> ctx(q, M, N, K);

    size_t num_wgps_m = (M + wgp_size_m - 1) / wgp_size_m;
    size_t num_wgps_n = (N + wgp_size_n - 1) / wgp_size_n;

    sycl::range<2> global_range(num_wgps_m* wgp_size_m, num_wgps_n* wgp_size_n);
    sycl::range<2> local_range(wgp_size_m, wgp_size_n);
    sycl::nd_range<2> nd_range(global_range, local_range);

    q.submit(
    [&](sycl::handler& syclHandler)
    {
        SyclGEMM::NaiveGemm<opT> 
        kernel(
            ctx.A.device_ptr,
            ctx.B.device_ptr,
            ctx.C.device_ptr,
            ctx.M,
            ctx.N,
            ctx.K,
            ctx.alpha,
            ctx.beta
        );
        syclHandler.parallel_for(nd_range, kernel);
    }).wait();

    SyclGEMM::reference_gemm(
        ctx.A.host_ptr,
        ctx.B.host_ptr,
        ctx.C.host_ptr,
        ctx.M,
        ctx.N,
        ctx.K,
        ctx.alpha,
        ctx.beta
    );

    ctx.C.assert_host_device_equality();
}