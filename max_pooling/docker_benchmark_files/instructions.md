## Exactly how to benchmark max pooling (CPU single-thread vs OpenMP)

### Recommended (Docker, works well on macOS)
From repo root (`cuda-cnn-ops`):

```bash
bash max_pooling/docker_benchmark_files/docker_bench.sh
```

With custom settings:
```bash
bash max_pooling/docker_benchmark_files/docker_bench.sh 8 20 5
```

Args are:
1. `threads` (OpenMP threads, `0` = default)
2. `iters` (timed runs)
3. `warmup` (warmup runs)

So `8 20 5` means 8 threads, best-of 20, warmup 5.

### Local (if your compiler supports OpenMP)
```bash
bash max_pooling/docker_benchmark_files/run_cpu_bench.sh
```

or
```bash
bash max_pooling/docker_benchmark_files/run_cpu_bench.sh 8 20 5
```

## What output you will see
- OpenMP thread count
- Timing policy
- Per-case results like:
  - `NCHW=... K=... S=... P=... | single=... ms omp=... ms speedup=...`
