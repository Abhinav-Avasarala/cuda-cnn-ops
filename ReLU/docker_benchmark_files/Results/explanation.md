# ReLU Docker Benchmark Explanation

## Test sizes and memory traffic

Each ReLU pass touches about `8N` bytes (`4N` read from input + `4N` written to output, float32):

- `N=1,048,576` -> ~8 MB
- `N=4,194,304` -> ~32 MB
- `N=16,777,216` -> ~128 MB
- `N=33,554,432` -> ~256 MB

## Why OpenMP speedup is weak (or below 1x) in Docker on macOS

- ReLU is memory-bandwidth bound, not compute bound.
- Single-thread code is already efficient (vectorized), so baseline performance is strong.
- OpenMP adds overhead (thread launch/scheduling/synchronization), especially visible on smaller `N`.
- More threads can hit shared memory-bandwidth limits, so extra threads stop helping and may hurt.

## Why macOS Docker makes this effect stronger

- Docker Desktop runs Linux workloads inside a VM.
- Thread scheduling and CPU resource mapping are less direct than native Linux.
- Apple silicon has heterogeneous cores (P-cores and E-cores), which can reduce OpenMP scaling consistency.
- Result: OpenMP may underperform relative to native HPC Linux runs.

## Takeaway

These results are expected for a memory-bound kernel under Docker/macOS virtualization.  
For stronger OpenMP gains, run on a native Linux HPC node and sweep thread counts (`1, 2, 4, 8`) to find the best point.