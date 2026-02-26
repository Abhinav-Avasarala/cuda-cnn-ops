#include <cmath>

void solve_with_cpu(const float* input, const float* kernel, float* output,
                    int input_rows, int input_cols,
                    int kernel_rows, int kernel_cols) {

    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;

    for (int row = 0; row < out_rows; row++) {
        for (int col = 0; col < out_cols; col++) {

            float result = 0.0f;

            for (int m = 0; m < kernel_rows; m++) {
                for (int n = 0; n < kernel_cols; n++) {

                    int in_r = row + m;
                    int in_c = col + n;

                    float in_val = input[in_r * input_cols + in_c];
                    float k_val  = kernel[m * kernel_cols + n];

                    result += in_val * k_val;
                }
            }

            output[row * out_cols + col] = result;
        }
    }
}