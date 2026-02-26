#include <omp.h>

void solve_with_cpu_omp(const float* input, const float* kernel, float* output,
                        int input_rows, int input_cols,
                        int kernel_rows, int kernel_cols) {

    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < out_rows; row++) {
        for (int col = 0; col < out_cols; col++) {

            float result = 0.0f;  // private per-iteration

            for (int m = 0; m < kernel_rows; m++) {
                int in_base = (row + m) * input_cols;      // small reuse
                int k_base  = m * kernel_cols;

                for (int n = 0; n < kernel_cols; n++) {
                    float in_val = input[in_base + (col + n)];
                    float k_val  = kernel[k_base + n];
                    result += in_val * k_val;
                }
            }

            output[row * out_cols + col] = result; // unique index per (row,col)
        }
    }
}