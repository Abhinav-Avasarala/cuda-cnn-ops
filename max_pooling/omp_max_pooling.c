#include <float.h>  // for FLT_MAX
#include <omp.h>

// CPU OpenMP max pooling.
// input/output layout: NCHW flattened as ((n * C + c) * H + h) * W + w.
void solve_with_cpu_omp(const float* input, float* output,
                        int N, int C, int H, int W,
                        int kernel_size, int stride, int padding) {
    if (input == 0 || output == 0) return;
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    if (kernel_size <= 0 || stride <= 0 || padding < 0) return;

    const int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    if (H_out <= 0 || W_out <= 0) return;

    #pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    const int h_start = h_out * stride - padding;
                    const int w_start = w_out * stride - padding;

                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        const int h_in = h_start + kh;
                        if (h_in < 0 || h_in >= H) continue;

                        for (int kw = 0; kw < kernel_size; ++kw) {
                            const int w_in = w_start + kw;
                            if (w_in < 0 || w_in >= W) continue;

                            const int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                            const float v = input[in_idx];
                            if (v > max_val) max_val = v;
                        }
                    }

                    const int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}
