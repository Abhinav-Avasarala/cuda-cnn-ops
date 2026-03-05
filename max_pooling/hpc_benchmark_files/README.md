# HPC Max Pooling Benchmark (CPU single-thread vs OpenMP vs CUDA)

This folder benchmarks the three max pooling implementations in `max_pooling/`:
- **Single-thread CPU**: `../max_pooling_single_thread_cpu.cpp` (`solve_with_cpu`)
- **OpenMP CPU**: `../omp_max_pooling.c` (`solve_with_cpu_omp`)
- **CUDA**: `../max_pooling.cu` (`solve`, device pointers)

The benchmark driver is `bench_all_max_pooling.cu`, and it uses the same test cases as the Docker benchmark.

---

## Exact steps from NCSU HPC login node

Assume you are already in your repo root (`cuda-cnn-ops`) on the login node.

### 1) Submit as a batch job (recommended)

```bash
cd max_pooling/hpc_benchmark_files
bsub < submit_maxpool_bench.lsf
bjobs
```

When job finishes:

```bash
ls -lt maxpool_bench.*.out maxpool_bench.*.err | head
```

Open output:

```bash
less maxpool_bench.<jobid>.out
```

### 2) Interactive GPU job (optional)

```bash
bsub -Is -q gpu -gpu "num=1" -n 8 -W 30 bash
module load gcc
module load cuda
cd ~/path/to/cuda-cnn-ops/max_pooling/hpc_benchmark_files
make clean && make bench
./bin/bench_all_max_pooling 8 20 5
```

---

## Build and run manually (inside compute node)

```bash
cd max_pooling/hpc_benchmark_files
make clean && make bench
./bin/bench_all_max_pooling 8 20 5
```

Arguments:
- `threads` (arg1): OpenMP threads (`0` = OpenMP runtime default)
- `iters` (arg2): timed iterations (best-of)
- `warmup` (arg3): warmup iterations

Output includes:
- OpenMP thread count
- CUDA device name
- Timing policy (`best-of iters`, with warmup)
- Per-case timings:
  - `single(ms)`
  - `omp(ms)`
  - `omp_speedup` (`single / omp`)
  - `cuda_kernel(ms)`
  - `cuda_speedup` (`single / cuda_kernel`)
  - `cuda_plus_D2H(ms)` (kernel + device->host output copy)

---

## If nvcc complains about GCC version

CUDA 12.0 typically supports host GCC up to 12.x. If default `gcc` is newer:

- Preferred:

```bash
module avail gcc
module load gcc/12
make clean && make bench
```

- Or:

```bash
make clean && make bench NVCC_CCBIN=g++-12
```

- Last resort:

```bash
make clean && make bench ALLOW_UNSUPPORTED=1
```

If your GPU architecture differs, override `CUDA_GENCODE`, for example:

```bash
# P100
make clean && make bench CUDA_GENCODE="-gencode arch=compute_60,code=sm_60"

# V100
make clean && make bench CUDA_GENCODE="-gencode arch=compute_70,code=sm_70"

# A100
make clean && make bench CUDA_GENCODE="-gencode arch=compute_80,code=sm_80"
```

---

## Optional Slurm usage

If your cluster uses Slurm instead of LSF:

```bash
sbatch slurm_bench.sh
```

Override threads:

```bash
sbatch --cpus-per-task=16 --export=ALL,OMP_NUM_THREADS=16 slurm_bench.sh
```

---

## Why `extern "C"` appears in benchmark declarations

`bench_all_max_pooling.cu` is compiled as C++. `omp_max_pooling.c` is compiled as C, so `extern "C"` is used in benchmark declarations to avoid C++ name mangling and keep linker symbols compatible.
