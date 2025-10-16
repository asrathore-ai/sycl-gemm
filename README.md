# GEMM ANalysis

ALgorithms implemented:
* Naive GEMM
* Tiled GEMM

## How to verify

```
using opT = int; # Verify using int to check alagorithm correctness.

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
```
