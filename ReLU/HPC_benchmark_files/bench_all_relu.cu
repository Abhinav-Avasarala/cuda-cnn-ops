#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>
#include <cuda_runtime.h>

// CPU single-thread implementation.
extern "C" void cpu_single_thread_relu_kernel(const float* input, float* output, int N);

// CPU OpenMP implementation in C.
extern "C" void omp_relu_kernel(const float* input, float* output, int N);

// CUDA implementation in `../relu.cu`. Pointers are device pointers.
extern "C" void solve(const float* input, float* output, int N);

static void die(const std::string& msg) {
  std::cerr << "ERROR: " << msg << "\n";
  std::exit(1);
}

static void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA error (" << what << "): " << cudaGetErrorString(e) << "\n";
    std::exit(1);
  }
}

static bool starts_with(const std::string& s, const std::string& pfx) {
  return s.size() >= pfx.size() && s.compare(0, pfx.size(), pfx) == 0;
}

static int parse_int_flag(int argc, char** argv, const std::string& name, int def) {
  const std::string pfx = name + "=";
  for (int i = 1; i < argc; i++) {
    const std::string a(argv[i]);
    if (starts_with(a, pfx)) return std::atoi(a.c_str() + pfx.size());
  }
  return def;
}

static void usage_and_exit() {
  std::cerr
      << "Usage:\n"
      << "  ./bench_all_relu [--n=33554432] [--iters=30] [--warmup=5] [--threads=8]\n";
  std::exit(2);
}

static void fill_random(std::vector<float>& v, unsigned int seed) {
  std::srand(seed);
  for (float& x : v) {
    const float r = (float)std::rand() / (float)RAND_MAX;  // [0,1]
    x = 2.0f * r - 1.0f;                                    // [-1,1]
  }
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

static double gb_per_s(int n, double ms) {
  const double bytes = 2.0 * (double)n * sizeof(float); // read input + write output
  const double sec = ms / 1000.0;
  return (bytes / sec) / 1e9;
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--help") usage_and_exit();
  }

  const int n = parse_int_flag(argc, argv, "--n", 1 << 25);
  const int iters = parse_int_flag(argc, argv, "--iters", 30);
  const int warmup = parse_int_flag(argc, argv, "--warmup", 5);
  const int threads = parse_int_flag(argc, argv, "--threads", 0); // 0 => runtime default

  if (n <= 0) die("--n must be positive");
  if (iters <= 0 || warmup < 0) die("--iters must be > 0 and --warmup must be >= 0");

  if (threads > 0) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
  }

  std::cout << "N: " << n << "\n";
  std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
  std::cout << "Timing: best-of " << iters << " (warmup " << warmup << ")\n";

  std::vector<float> h_input((size_t)n);
  std::vector<float> out_cpu((size_t)n, 0.0f);
  std::vector<float> out_omp((size_t)n, 0.0f);
  std::vector<float> out_gpu((size_t)n, 0.0f);
  fill_random(h_input, 42);

  auto run_cpu = [&]() { cpu_single_thread_relu_kernel(h_input.data(), out_cpu.data(), n); };
  auto run_omp = [&]() { omp_relu_kernel(h_input.data(), out_omp.data(), n); };

  float* d_input = nullptr;
  float* d_output = nullptr;
  cuda_check(cudaMalloc(&d_input, sizeof(float) * h_input.size()), "cudaMalloc d_input");
  cuda_check(cudaMalloc(&d_output, sizeof(float) * out_gpu.size()), "cudaMalloc d_output");
  cuda_check(cudaMemcpy(d_input, h_input.data(), sizeof(float) * h_input.size(), cudaMemcpyHostToDevice), "H2D input");

  // Kernel-only timing; `solve` synchronizes internally.
  auto run_cuda = [&]() { solve(d_input, d_output, n); };

  // Correctness checks against single-thread CPU.
  run_cpu();
  run_omp();
  run_cuda();
  cuda_check(cudaMemcpy(out_gpu.data(), d_output, sizeof(float) * out_gpu.size(), cudaMemcpyDeviceToHost), "D2H output");

  const double diff_omp = max_abs_diff(out_cpu, out_omp);
  const double diff_gpu = max_abs_diff(out_cpu, out_gpu);
  if (diff_omp > 1e-7) die("OpenMP differs from single-thread (max_abs_diff=" + std::to_string(diff_omp) + ")");
  if (diff_gpu > 1e-7) die("CUDA differs from single-thread (max_abs_diff=" + std::to_string(diff_gpu) + ")");

  const double cpu_ms = best_ms(run_cpu, warmup, iters);
  const double omp_ms = best_ms(run_omp, warmup, iters);
  const double gpu_ms = best_ms(run_cuda, warmup, iters);

  // End-to-end GPU: kernel + D2H.
  auto run_cuda_plus_d2h = [&]() {
    solve(d_input, d_output, n);
    cuda_check(cudaMemcpy(out_gpu.data(), d_output, sizeof(float) * out_gpu.size(), cudaMemcpyDeviceToHost), "D2H output");
  };
  const double gpu_total_ms = best_ms(run_cuda_plus_d2h, warmup, iters);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "single(ms): " << cpu_ms
            << "  GB/s: " << gb_per_s(n, cpu_ms) << "\n";
  std::cout << "openmp(ms): " << omp_ms
            << "  speedup_vs_single: " << std::setprecision(2) << (cpu_ms / omp_ms)
            << std::setprecision(3) << "  GB/s: " << gb_per_s(n, omp_ms) << "\n";
  std::cout << "cuda_kernel(ms): " << gpu_ms
            << "  speedup_vs_single: " << std::setprecision(2) << (cpu_ms / gpu_ms)
            << std::setprecision(3) << "  GB/s: " << gb_per_s(n, gpu_ms) << "\n";
  std::cout << "cuda_plus_D2H(ms): " << gpu_total_ms
            << "  GB/s: " << gb_per_s(n, gpu_total_ms) << "\n";

  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}
