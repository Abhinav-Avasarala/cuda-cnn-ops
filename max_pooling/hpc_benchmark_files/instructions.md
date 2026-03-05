## HPC benchmark files for CPU max pooling

This folder provides a cluster-friendly benchmark setup for comparing:
- single-thread CPU max pooling (`solve_with_cpu`)
- OpenMP CPU max pooling (`solve_with_cpu_omp`)

The benchmark workload matches the Docker benchmark cases.

### Files
- `bench_cpu_max_pooling.cpp`: benchmark driver
- `Makefile`: builds benchmark binary with OpenMP
- `run_hpc_bench.sh`: local/manual run script
- `slurm_bench.sh`: example Slurm submission script

## Quick start

From repo root:

```bash
chmod +x max_pooling/hpc_benchmark_files/run_hpc_bench.sh
chmod +x max_pooling/hpc_benchmark_files/slurm_bench.sh
```

### Run directly on a login/compute node

```bash
bash max_pooling/hpc_benchmark_files/run_hpc_bench.sh
```

Custom settings:

```bash
bash max_pooling/hpc_benchmark_files/run_hpc_bench.sh 8 20 5
```

Args are:
1. `threads` (OpenMP threads, `0` = runtime default)
2. `iters` (timed runs)
3. `warmup` (warmup runs)

### Run via Slurm

```bash
sbatch max_pooling/hpc_benchmark_files/slurm_bench.sh
```

To override threads at submission:

```bash
sbatch --cpus-per-task=16 --export=ALL,OMP_NUM_THREADS=16 max_pooling/hpc_benchmark_files/slurm_bench.sh
```

## Output

You will see:
- OpenMP thread count
- timing policy (`best-of iters`)
- per-case timings and speedup:
  - `NCHW=... K=... S=... P=... | single=... ms omp=... ms speedup=...`
