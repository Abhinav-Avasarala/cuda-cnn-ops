#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <omp.h>

// From `../single_thread_conv2d.cpp` (C++ file). Normal C++ declaration.
void solve_with_cpu(const float* input, const float* kernel, float* output,
                    int input_rows, int input_cols,
                    int kernel_rows, int kernel_cols);

// From `../omp_conv2d.c` (C file). IMPORTANT: `extern "C"` prevents C++ name mangling,
// so the linker can find the symbol emitted by the C compiler.
extern "C" void solve_with_cpu_omp(const float* input, const float* kernel, float* output,
                                  int input_rows, int input_cols,
                                  int kernel_rows, int kernel_cols);

static int out_rows(int in_r, int k_r) { return in_r - k_r + 1; }
static int out_cols(int in_c, int k_c) { return in_c - k_c + 1; }

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

static void bench_one(int in_r, int in_c, int k_r, int k_c, int warmup, int iters, std::mt19937& rng) {
  const int o_r = out_rows(in_r, k_r);
  const int o_c = out_cols(in_c, k_c);
  if (o_r <= 0 || o_c <= 0) {
    std::cerr << "Skipping invalid case in=" << in_r << "x" << in_c << " k=" << k_r << "x" << k_c << "\n";
    return;
  }

  std::vector<float> input(in_r * in_c);
  std::vector<float> kernel(k_r * k_c);
  std::vector<float> out_cpu(o_r * o_c, 0.0f);
  std::vector<float> out_omp(o_r * o_c, 0.0f);
  fill_random(input, rng);
  fill_random(kernel, rng);

  auto run_cpu = [&]() { solve_with_cpu(input.data(), kernel.data(), out_cpu.data(), in_r, in_c, k_r, k_c); };
  auto run_omp = [&]() { solve_with_cpu_omp(input.data(), kernel.data(), out_omp.data(), in_r, in_c, k_r, k_c); };

  // Correctness cross-check (same inputs => outputs should match).
  run_cpu();
  run_omp();
  const double diff = max_abs_diff(out_cpu, out_omp);
  if (diff > 1e-4) {
    std::cerr << "FAIL diff=" << diff << " for in=" << in_r << "x" << in_c << " k=" << k_r << "x" << k_c << "\n";
    std::exit(1);
  }

  const double cpu_ms = best_ms(run_cpu, warmup, iters);
  const double omp_ms = best_ms(run_omp, warmup, iters);

  std::cout << "in=" << in_r << "x" << in_c
            << " k=" << k_r << "x" << k_c
            << " out=" << o_r << "x" << o_c
            << " | single=" << std::fixed << std::setprecision(3) << cpu_ms << " ms"
            << " omp=" << std::fixed << std::setprecision(3) << omp_ms << " ms"
            << " speedup=" << std::fixed << std::setprecision(2) << (cpu_ms / omp_ms)
            << "\n";
}

int main(int argc, char** argv) {
  // Minimal knobs:
  //   arg1: threads (optional; default = OpenMP runtime default)
  //   arg2: iters (optional; default 10)
  //   arg3: warmup (optional; default 3)
  const int threads = (argc >= 2) ? std::atoi(argv[1]) : 0;
  const int iters = (argc >= 3) ? std::atoi(argv[2]) : 10;
  const int warmup = (argc >= 4) ? std::atoi(argv[3]) : 3;

  if (threads > 0) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
  }

  std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
  std::cout << "Timing: best-of " << iters << " (warmup " << warmup << ")\n";

  std::mt19937 rng(42);

  // Sample cases (edit freely).
  bench_one(256, 256, 3, 3, warmup, iters, rng);
  bench_one(512, 512, 3, 3, warmup, iters, rng);
  bench_one(1024, 1024, 3, 3, warmup, iters, rng);
  bench_one(512, 512, 7, 7, warmup, iters, rng);

  return 0;
}

