For each ReLU run, roughly 8N bytes are touched (input read + output write, float32):

N=1,048,576 → ~8 MB traffic
N=4,194,304 → ~32 MB
N=16,777,216 → ~128 MB
N=33,554,432 → ~256 MB
Why speedup is bad (or <1):

ReLU is memory-bandwidth bound, not compute bound (very little math per element).
Single-thread version already gets strong SIMD/vectorized performance.
OpenMP adds overhead (thread scheduling/sync/parallel region startup), which hurts smaller cases.
At higher threads (like 8), threads can compete for memory bandwidth, so runtime can get worse.
In Docker on macOS, CPU scheduling/limits and P-core/E-core behavior can add more variability.