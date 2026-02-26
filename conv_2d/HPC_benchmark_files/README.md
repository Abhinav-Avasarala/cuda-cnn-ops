# HPC Conv2D Benchmark (CPU single-thread vs OpenMP vs CUDA)

This folder benchmarks the three conv2d implementations in `conv_2d/`:
- **Single-thread CPU**: `../single_thread_conv2d.cpp` (`solve_with_cpu`)
- **OpenMP CPU**: `../omp_conv2d.c` (`solve_with_cpu_omp`)
- **CUDA**: `../conv_2d_naive.cu` (`solve_with_cuda`, device pointers)

## 1) Get an interactive GPU compute node (LSF)

Example (adjust `-n` cores, `-W` minutes, queue, and GPU options to your cluster rules):

```bash
bsub -Is -n 8 -W 30 -q gpu -gpu "num=1:mode=shared:mps=no" tcsh
```

Notes:
- `-n` controls how many CPU cores you get for OpenMP.
- If you prefer `bash` instead of `tcsh`, replace the last token with `bash`.

## 2) Load toolchains

Module names vary; on the compute node do something like:

```bash
module load gcc
module load cuda
```

Quick sanity checks:

```bash
which gcc g++ nvcc
nvcc --version
nvidia-smi
```

Note: `nvidia-smi` reports the **driver** CUDA capability (e.g. \"CUDA Version: 12.6\"). Your build uses the **toolkit** selected by `nvcc --version` / your loaded `cuda` module, and that toolkit determines which GCC versions are accepted by nvcc.

## 3) Build

```bash
cd conv_2d/HPC_benchmark_files
make clean && make
```

### If nvcc complains about your GCC version

CUDA 12.0 supports host GCC up to 12.x. If your default `gcc` is newer (e.g. 13/14), do **one** of these:

- **Preferred**: load a GCC 12 module, then rebuild:

```bash
module avail gcc
module load gcc/12
make clean && make
```

- **Or**: point `nvcc` at a GCC 12 binary explicitly:

```bash
make clean && make NVCC_CCBIN=g++-12
```

- **Last resort (not recommended)**: force nvcc to accept the newer compiler:

```bash
make clean && make ALLOW_UNSUPPORTED=1
```

If your GPU architecture differs, override `CUDA_GENCODE`. Examples:

```bash
# P100
make clean && make CUDA_GENCODE="-gencode arch=compute_60,code=sm_60"

# V100
make clean && make CUDA_GENCODE="-gencode arch=compute_70,code=sm_70"

# A100
make clean && make CUDA_GENCODE="-gencode arch=compute_80,code=sm_80"
```

## 4) Run the benchmark

```bash
./bin/bench_all_conv2d --in=1024 --k=3 --iters=20 --warmup=5 --threads=8
```

Arguments:
- `--in=N`: square input size \(N x N\)
- `--k=K`: square kernel size \(K x K\)
- `--iters=N`: timed iterations (best-of)
- `--warmup=N`: warmup iterations
- `--threads=N`: OpenMP threads (0 = OpenMP runtime default)

Output includes:
- `single(ms)` (CPU single-thread)
- `openmp(ms)` and speedup vs single
- `cuda_kernel(ms)` and speedup vs single
- `cuda_plus_D2H(ms)` (kernel + device→host copy of output; inputs are copied once)

## How the C + C++ functions are linked

`bench_all_conv2d.cu` is compiled as C++ (via `nvcc`). The OpenMP function is implemented in a **C** file (`../omp_conv2d.c`), so the benchmark declares it as:

```c++
extern "C" void solve_with_cpu_omp(...);
```

and the Makefile compiles `omp_conv2d.c` with `gcc` (not `g++`). This keeps a C ABI symbol name that the C++ linker can resolve.

