#include "Utils.hpp"
#include "algorithms/Reference.hpp"
#include "algorithms/Tiled.hpp"

constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;
constexpr int tile_dim = 32;
constexpr int wgp_size_m = tile_dim;
constexpr int wgp_size_n = tile_dim;
constexpr int tile_size = wgp_size_m*wgp_size_n;

int main(){
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
    SyclGEMM::print_device_info(q);

    using opT = float;

    SyclGEMM::Context<opT> ctx(q, M, N, K);

    size_t num_wgps_m = (M + wgp_size_m - 1) / wgp_size_m;
    size_t num_wgps_n = (N + wgp_size_n - 1) / wgp_size_n;

    sycl::range<2> global_range(num_wgps_m* wgp_size_m, num_wgps_n* wgp_size_n);
    sycl::range<2> local_range(wgp_size_m, wgp_size_n);
    sycl::nd_range<2> nd_range(global_range, local_range);

    auto event = q.submit(
    [&](sycl::handler& syclHandler)
    {

        sycl::local_accessor<opT, 2> tile_A(local_range, syclHandler);
        sycl::local_accessor<opT, 2> tile_B(local_range, syclHandler);

        SyclGEMM::TiledGEMM<opT, tile_dim> 
        kernel(
            ctx.A.device_ptr,
            ctx.B.device_ptr,
            ctx.C.device_ptr,
            ctx.M,
            ctx.N,
            ctx.K,
            ctx.alpha,
            ctx.beta,
            tile_A,
            tile_B
        );
        syclHandler.parallel_for(nd_range, kernel);
    });
    event.wait();

    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    double kernel_time_ns = (end - start);
    double kernel_time_ms = kernel_time_ns * 1e-6;

    std::cout << "Tiled Kernel execution time: " << kernel_time_ms << " ms" << std::endl;
}