# HPC ReLU Benchmark (CPU single-thread vs OpenMP vs CUDA)

This folder benchmarks the three ReLU implementations in `ReLU/`:
- **Single-thread CPU**: `../relu_single_thread_cpu.cpp` (`cpu_single_thread_relu_kernel`)
- **OpenMP CPU**: `../omp_relu.c` (`omp_relu_kernel`)
- **CUDA**: `../relu.cu` (`solve`, device pointers)

---

## Exact steps from NCSU HPC login node

Assume you are already in your repo root (`cuda-cnn-ops`) on the login node.

### 1) Submit as a batch job (recommended)

```bash
cd ReLU/HPC_benchmark_files
bsub < submit_relu_bench.lsf
bjobs
```

When job finishes:

```bash
ls -lt relu_bench.*.out relu_bench.*.err | head
```

Open output:

```bash
less relu_bench.<jobid>.out
```

### 2) Interactive GPU job (optional)

```bash
bsub -Is -q gpu -gpu "num=1" -n 8 -W 30 bash
module load gcc
module load cuda
cd ~/path/to/cuda-cnn-ops/ReLU/HPC_benchmark_files
make clean && make
./bin/bench_all_relu --n=33554432 --iters=30 --warmup=5 --threads=8
```

---

## Build and run manually (inside compute node)

```bash
cd ReLU/HPC_benchmark_files
make clean && make
./bin/bench_all_relu --n=33554432 --iters=30 --warmup=5 --threads=8
```

Arguments:
- `--n=N`: 1D vector length for ReLU
- `--iters=N`: timed iterations (best-of)
- `--warmup=N`: warmup iterations
- `--threads=N`: OpenMP threads (`0` = OpenMP runtime default)

Output includes:
- `single(ms)` and effective GB/s
- `openmp(ms)` speedup vs single, and GB/s
- `cuda_kernel(ms)` speedup vs single, and GB/s
- `cuda_plus_D2H(ms)` (kernel + device->host output copy), and GB/s

---

## If nvcc complains about GCC version

CUDA 12.0 typically supports host GCC up to 12.x. If default `gcc` is newer:

- Preferred:

```bash
module avail gcc
module load gcc/12
make clean && make
```

- Or:

```bash
make clean && make NVCC_CCBIN=g++-12
```

- Last resort:

```bash
make clean && make ALLOW_UNSUPPORTED=1
```

If your GPU architecture differs, override `CUDA_GENCODE`, for example:

```bash
# P100
make clean && make CUDA_GENCODE="-gencode arch=compute_60,code=sm_60"

# V100
make clean && make CUDA_GENCODE="-gencode arch=compute_70,code=sm_70"

# A100
make clean && make CUDA_GENCODE="-gencode arch=compute_80,code=sm_80"
```

---

## Why `extern "C"` appears in benchmark declarations

`bench_all_relu.cu` is compiled as C++. `omp_relu.c` is compiled as C, so `extern "C"` is used in the benchmark declarations to avoid C++ name mangling and keep linker symbols compatible.
