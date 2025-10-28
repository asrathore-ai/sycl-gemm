#include "Utils.hpp"
#include "algorithms/StreamK.hpp"
#include "algorithms/Reference.hpp"

constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 4096;

int main(){
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
    SyclGEMM::print_device_info(q);

    using opT = int;

    SyclGEMM::Context<opT> ctx(q, M, N, K);

    int numMACs = M*N*K;
    int wgpSize = 128;
    int numWgps = (numMACs + wgpSize - 1) / wgpSize;
    sycl::range<1> global_range(numWgps * wgpSize);
    sycl::range<1> local_range(wgpSize);
    sycl::nd_range<1> nd_range(global_range, local_range);

    auto event = q.submit(
    [&](sycl::handler& syclHandler)
    {
        
        SyclGEMM::StreamKGEMM<opT> 
        kernel(
            ctx.A.device_ptr,
            ctx.B.device_ptr,
            ctx.C.device_ptr,
            ctx.M,
            ctx.N,
            ctx.K
        );
        syclHandler.parallel_for(nd_range, kernel);
    });
    event.wait();

    SyclGEMM::reference_gemm(
        ctx.A.host_ptr,
        ctx.B.host_ptr,
        ctx.C.host_ptr,
        ctx.M,
        ctx.N,
        ctx.K
    );

    ctx.C.assert_host_device_equality();

    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    double kernel_time_ns = (end - start);
    double kernel_time_ms = kernel_time_ns * 1e-6;

    std::cout << "Stream-K Kernel execution time: " << kernel_time_ms << " ms" << std::endl;
}