# ReLU HPC Benchmark Explanation

## What these numbers mean

For `N=33,554,432`:
- `single(ms): 19.683` -> `13.64 GB/s`
- `openmp(ms): 3.165` -> `84.80 GB/s` (`6.22x` speedup vs single)
- `cuda_kernel(ms): 0.205` -> very high device-only throughput
- `cuda_plus_D2H(ms): 17.444` -> much slower than kernel-only because host transfer is included

ReLU is a low-compute operation (one compare/select per element), so it is mostly limited by memory bandwidth.

## Why OpenMP is much faster than single-thread on HPC

- A single CPU core uses only part of available memory bandwidth.
- With 8 threads, more memory channels/controllers are utilized at once.
- Throughput rises from `~13.6 GB/s` to `~84.8 GB/s`, which explains the large speedup.
- `6.22x` instead of `8x` is expected due to OpenMP overhead and partial bandwidth saturation.

## Why CUDA kernel-only is extremely fast

- `cuda_kernel(ms)` measures only GPU kernel execution on data already in device memory.
- GPUs have much higher memory bandwidth than CPUs, so ReLU runs very quickly on-device.
- This is an upper-bound style metric for pure compute/device-memory performance.

## Why CUDA + D2H is much slower than kernel-only

- `cuda_plus_D2H(ms)` includes copying the output back to host memory each iteration.
- For simple kernels like ReLU, transfer overhead can be comparable to or larger than kernel time.
- This is expected and does not indicate a bad kernel.

## Fair comparison guidance

When presenting results, report both:
- **Kernel-only** (`cuda_kernel`) to show GPU compute capability.
- **End-to-end** (`cuda_plus_D2H`) to show practical wall-clock cost when host output is required.