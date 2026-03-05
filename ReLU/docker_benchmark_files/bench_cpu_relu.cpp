#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <omp.h>

// From `../relu_single_thread_cpu.cpp`.
extern "C" void cpu_single_thread_relu_kernel(const float* input, float* output, int N);

// From `../omp_relu.c` (C file).
extern "C" void omp_relu_kernel(const float* input, float* output, int N);

static void fill_random(std::vector<float>& v, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : v) x = dist(rng);
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
  return m;
}

template <class Fn>
static double best_ms(Fn&& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; i++) fn();
  double best = 1e100;
  for (int i = 0; i < iters; i++) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

static void bench_one(int n, int warmup, int iters, std::mt19937& rng) {
  std::vector<float> input((size_t)n);
  std::vector<float> out_single((size_t)n, 0.0f);
  std::vector<float> out_omp((size_t)n, 0.0f);
  fill_random(input, rng);

  auto run_single = [&]() { cpu_single_thread_relu_kernel(input.data(), out_single.data(), n); };
  auto run_omp = [&]() { omp_relu_kernel(input.data(), out_omp.data(), n); };

  // Correctness cross-check.
  run_single();
  run_omp();
  const double diff = max_abs_diff(out_single, out_omp);
  if (diff > 1e-7) {
    std::cerr << "FAIL diff=" << diff << " for N=" << n << "\n";
    std::exit(1);
  }

  const double single_ms = best_ms(run_single, warmup, iters);
  const double omp_ms = best_ms(run_omp, warmup, iters);

  std::cout << "N=" << n
            << " | single=" << std::fixed << std::setprecision(3) << single_ms << " ms"
            << " omp=" << std::fixed << std::setprecision(3) << omp_ms << " ms"
            << " speedup=" << std::fixed << std::setprecision(2) << (single_ms / omp_ms)
            << "\n";
}

int main(int argc, char** argv) {
  // args:
  //   arg1: threads (optional; default = OpenMP runtime default)
  //   arg2: iters   (optional; default 20)
  //   arg3: warmup  (optional; default 5)
  const int threads = (argc >= 2) ? std::atoi(argv[1]) : 0;
  const int iters = (argc >= 3) ? std::atoi(argv[2]) : 20;
  const int warmup = (argc >= 4) ? std::atoi(argv[3]) : 5;

  if (threads > 0) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
  }

  std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
  std::cout << "Timing: best-of " << iters << " (warmup " << warmup << ")\n";

  std::mt19937 rng(42);
  bench_one(1 << 20, warmup, iters, rng);
  bench_one(1 << 22, warmup, iters, rng);
  bench_one(1 << 24, warmup, iters, rng);
  bench_one(1 << 25, warmup, iters, rng);
  return 0;
}
