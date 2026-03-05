## Exactly how to benchmark

### Recommended (Docker, works well on macOS)
From repo root (`cuda-cnn-ops`):

```bash
bash ReLU/docker_benchmark_files/docker_bench.sh
```

With custom settings:
```bash
bash ReLU/docker_benchmark_files/docker_bench.sh 8 20 5
```

Args are:
1. `threads` (OpenMP threads, `0` = default)
2. `iters` (timed runs)
3. `warmup` (warmup runs)

So `8 20 5` means 8 threads, best-of 20, warmup 5.

### Local (if your compiler supports OpenMP)
```bash
bash ReLU/docker_benchmark_files/run_cpu_bench.sh
```

or
```bash
bash ReLU/docker_benchmark_files/run_cpu_bench.sh 8 20 5
```

## What output you’ll see
- OpenMP thread count
- Timing policy
- Per-size results like:
  - `N=... | single=... ms omp=... ms speedup=...`
