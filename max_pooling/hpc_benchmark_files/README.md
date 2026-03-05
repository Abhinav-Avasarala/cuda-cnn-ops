# HPC Max Pooling Benchmark (CPU single-thread vs OpenMP)

This folder benchmarks the two CPU max pooling implementations in `max_pooling/`:
- **Single-thread CPU**: `../max_pooling_single_thread_cpu.cpp` (`solve_with_cpu`)
- **OpenMP CPU**: `../omp_max_pooling.c` (`solve_with_cpu_omp`)

The benchmark driver is `bench_cpu_max_pooling.cpp`, and it uses the same test cases as the Docker benchmark.

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

### 2) Interactive CPU job (optional)

```bash
bsub -Is -n 8 -W 30 bash
module load gcc
cd ~/path/to/cuda-cnn-ops/max_pooling/hpc_benchmark_files
make clean && make bench
./bin/bench_cpu_max_pooling 8 20 5
```

---

## Build and run manually (inside compute node)

```bash
cd max_pooling/hpc_benchmark_files
make clean && make bench
./bin/bench_cpu_max_pooling 8 20 5
```

Arguments:
- `threads` (arg1): OpenMP threads (`0` = OpenMP runtime default)
- `iters` (arg2): timed iterations (best-of)
- `warmup` (arg3): warmup iterations

Output includes:
- OpenMP thread count
- Timing policy (`best-of iters`, with warmup)
- Per-case timings:
  - `single(ms)`
  - `omp(ms)`
  - `speedup` (`single / omp`)

---

## If OpenMP build fails

If your default compiler does not support OpenMP:

- Preferred:

```bash
module avail gcc
module load gcc/12
make clean && make bench
```

- Or explicitly set compilers:

```bash
make clean && make bench CC=gcc CXX=g++
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

`bench_cpu_max_pooling.cpp` is compiled as C++. `omp_max_pooling.c` is compiled as C, so `extern "C"` is used in benchmark declarations to avoid C++ name mangling and keep linker symbols compatible.
