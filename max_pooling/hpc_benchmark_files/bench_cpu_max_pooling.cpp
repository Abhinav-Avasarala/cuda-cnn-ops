#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <omp.h>

extern "C" void solve_with_cpu(const float* input, float* output,
                               int N, int C, int H, int W,
                               int kernel_size, int stride, int padding);

extern "C" void solve_with_cpu_omp(const float* input, float* output,
                                   int N, int C, int H, int W,
                                   int kernel_size, int stride, int padding);

static int out_dim(int in, int k, int stride, int pad) {
  return (in + 2 * pad - k) / stride + 1;
}

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

static void bench_one(int N, int C, int H, int W, int K, int S, int P,
                      int warmup, int iters, std::mt19937& rng) {
  const int H_out = out_dim(H, K, S, P);
  const int W_out = out_dim(W, K, S, P);
  if (H_out <= 0 || W_out <= 0) {
    std::cerr << "Skipping invalid case NCHW=" << N << "x" << C << "x" << H << "x" << W
              << " K=" << K << " S=" << S << " P=" << P << "\n";
    return;
  }

  std::vector<float> input((size_t)N * C * H * W);
  std::vector<float> out_single((size_t)N * C * H_out * W_out, 0.0f);
  std::vector<float> out_omp((size_t)N * C * H_out * W_out, 0.0f);
  fill_random(input, rng);

  auto run_single = [&]() { solve_with_cpu(input.data(), out_single.data(), N, C, H, W, K, S, P); };
  auto run_omp = [&]() { solve_with_cpu_omp(input.data(), out_omp.data(), N, C, H, W, K, S, P); };

  run_single();
  run_omp();
  const double diff = max_abs_diff(out_single, out_omp);
  if (diff > 1e-6) {
    std::cerr << "FAIL diff=" << diff << " for NCHW=" << N << "x" << C << "x" << H << "x" << W
              << " K=" << K << " S=" << S << " P=" << P << "\n";
    std::exit(1);
  }

  const double single_ms = best_ms(run_single, warmup, iters);
  const double omp_ms = best_ms(run_omp, warmup, iters);

  std::cout << "NCHW=" << N << "x" << C << "x" << H << "x" << W
            << " K=" << K << " S=" << S << " P=" << P
            << " out=" << H_out << "x" << W_out
            << " | single=" << std::fixed << std::setprecision(3) << single_ms << " ms"
            << " omp=" << std::fixed << std::setprecision(3) << omp_ms << " ms"
            << " speedup=" << std::fixed << std::setprecision(2) << (single_ms / omp_ms)
            << "\n";
}

int main(int argc, char** argv) {
  // args:
  //   arg1: threads (optional; default = OpenMP runtime default)
  //   arg2: iters   (optional; default 10)
  //   arg3: warmup  (optional; default 3)
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
  bench_one(8, 64, 112, 112, 2, 2, 0, warmup, iters, rng);
  bench_one(8, 64, 224, 224, 2, 2, 0, warmup, iters, rng);
  bench_one(16, 128, 112, 112, 2, 2, 0, warmup, iters, rng);
  bench_one(8, 128, 224, 224, 3, 2, 1, warmup, iters, rng);
  return 0;
}
