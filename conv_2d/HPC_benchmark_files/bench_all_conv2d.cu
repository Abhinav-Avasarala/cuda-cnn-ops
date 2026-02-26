#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>
#include <cuda_runtime.h>

// C++ single-thread implementation (compiled as C++).
void solve_with_cpu(const float* input, const float* kernel, float* output,
                    int input_rows, int input_cols,
                    int kernel_rows, int kernel_cols);

// C OpenMP implementation (compiled as C). `extern "C"` prevents C++ name mangling.
extern "C" void solve_with_cpu_omp(const float* input, const float* kernel, float* output,
                                  int input_rows, int input_cols,
                                  int kernel_rows, int kernel_cols);

// CUDA implementation in `conv_2d_naive.cu`. IMPORTANT: pointers are device pointers.
void solve_with_cuda(const float* input, const float* kernel, float* output,
                     int input_rows, int input_cols, int kernel_rows, int kernel_cols);

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
      << "  ./bench_all_conv2d [--in=1024] [--k=3] [--iters=20] [--warmup=5] [--threads=8]\n"
      << "  (square input and square kernel; valid convolution)\n";
  std::exit(2);
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--help") usage_and_exit();
  }

  const int in = parse_int_flag(argc, argv, "--in", 1024);
  const int k = parse_int_flag(argc, argv, "--k", 3);
  const int iters = parse_int_flag(argc, argv, "--iters", 20);
  const int warmup = parse_int_flag(argc, argv, "--warmup", 5);
  const int threads = parse_int_flag(argc, argv, "--threads", 0); // 0 => runtime default

  if (in <= 0 || k <= 0) die("in and k must be positive");
  const int o_r = out_rows(in, k);
  const int o_c = out_cols(in, k);
  if (o_r <= 0 || o_c <= 0) die("kernel larger than input (valid convolution output <= 0)");

  if (threads > 0) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
  }

  std::cout << "Input: " << in << "x" << in << "  Kernel: " << k << "x" << k
            << "  Output: " << o_r << "x" << o_c << "\n";
  std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
  std::cout << "Timing: best-of " << iters << " (warmup " << warmup << ")\n";

  std::mt19937 rng(42);
  std::vector<float> h_input(in * in);
  std::vector<float> h_kernel(k * k);
  std::vector<float> out_cpu(o_r * o_c, 0.0f);
  std::vector<float> out_omp(o_r * o_c, 0.0f);
  std::vector<float> out_gpu(o_r * o_c, 0.0f);
  fill_random(h_input, rng);
  fill_random(h_kernel, rng);

  auto run_cpu = [&]() { solve_with_cpu(h_input.data(), h_kernel.data(), out_cpu.data(), in, in, k, k); };
  auto run_omp = [&]() { solve_with_cpu_omp(h_input.data(), h_kernel.data(), out_omp.data(), in, in, k, k); };

  // GPU: allocate/copy once; time kernel-only via `solve_with_cuda` (which synchronizes internally).
  float* d_input = nullptr;
  float* d_kernel = nullptr;
  float* d_output = nullptr;
  cuda_check(cudaMalloc(&d_input, sizeof(float) * h_input.size()), "cudaMalloc d_input");
  cuda_check(cudaMalloc(&d_kernel, sizeof(float) * h_kernel.size()), "cudaMalloc d_kernel");
  cuda_check(cudaMalloc(&d_output, sizeof(float) * out_gpu.size()), "cudaMalloc d_output");
  cuda_check(cudaMemcpy(d_input, h_input.data(), sizeof(float) * h_input.size(), cudaMemcpyHostToDevice), "H2D input");
  cuda_check(cudaMemcpy(d_kernel, h_kernel.data(), sizeof(float) * h_kernel.size(), cudaMemcpyHostToDevice), "H2D kernel");

  auto run_cuda = [&]() { solve_with_cuda(d_input, d_kernel, d_output, in, in, k, k); };

  // Correctness (single-thread baseline).
  run_cpu();
  run_omp();
  run_cuda();
  cuda_check(cudaMemcpy(out_gpu.data(), d_output, sizeof(float) * out_gpu.size(), cudaMemcpyDeviceToHost), "D2H output");

  const double diff_omp = max_abs_diff(out_cpu, out_omp);
  const double diff_gpu = max_abs_diff(out_cpu, out_gpu);
  if (diff_omp > 1e-4) die("OpenMP differs from single-thread (max_abs_diff=" + std::to_string(diff_omp) + ")");
  if (diff_gpu > 1e-3) die("CUDA differs from single-thread (max_abs_diff=" + std::to_string(diff_gpu) + ")");

  const double cpu_ms = best_ms(run_cpu, warmup, iters);
  const double omp_ms = best_ms(run_omp, warmup, iters);

  // GPU timing (kernel-only as wrapped by `solve_with_cuda`).
  const double gpu_ms = best_ms(run_cuda, warmup, iters);

  // Also report an end-to-end GPU time including D2H (input copies excluded because they are one-time here).
  auto run_cuda_plus_d2h = [&]() {
    solve_with_cuda(d_input, d_kernel, d_output, in, in, k, k);
    cuda_check(cudaMemcpy(out_gpu.data(), d_output, sizeof(float) * out_gpu.size(), cudaMemcpyDeviceToHost), "D2H output");
  };
  const double gpu_total_ms = best_ms(run_cuda_plus_d2h, warmup, iters);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "single(ms): " << cpu_ms << "\n";
  std::cout << "openmp(ms): " << omp_ms << "  speedup_vs_single: " << std::setprecision(2) << (cpu_ms / omp_ms) << "\n";
  std::cout << std::setprecision(3) << "cuda_kernel(ms): " << gpu_ms
            << "  speedup_vs_single: " << std::setprecision(2) << (cpu_ms / gpu_ms) << "\n";
  std::cout << std::setprecision(3) << "cuda_plus_D2H(ms): " << gpu_total_ms << "\n";

  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
  return 0;
}

