#include <cuda_runtime.h>

static inline void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) {
        // Keep it simple: propagate error via device reset + abort.
        // (Callers in benchmarks will see a clear failure instead of silent wrong results.)
        printf("CUDA error: %s\n", cudaGetErrorString(e));
        fflush(stdout);
        abort();
    }
}

__global__ void conv_2d_kernel(const float* input, const float* kernel, float* output,
                               int input_rows, int input_cols,
                               int kernel_rows, int kernel_cols,
                               int out_rows, int out_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // output row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output col

    if (row >= out_rows || col >= out_cols) return;

    float result = 0.0f;

    // valid convolution: kernel fully overlaps input
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

// input, kernel, output are device pointers
void solve_with_cuda(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;

    dim3 block(16, 16);
    dim3 grid((out_cols + block.x - 1) / block.x,
              (out_rows + block.y - 1) / block.y);

    conv_2d_kernel<<<grid, block>>>(input, kernel, output,
                                    input_rows, input_cols,
                                    kernel_rows, kernel_cols,
                                    out_rows, out_cols);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
}