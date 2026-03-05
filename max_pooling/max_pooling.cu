#include <cuda_runtime.h>
#include <float.h>   // for FLT_MAX

__global__ void max_pool_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_size, int stride, int padding,
    int H_out, int W_out
) {
    // 1) output coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    // 2) map blockIdx.z -> (n, c)
    int nc = blockIdx.z;      // 0 .. N*C-1
    int n  = nc / C;
    int c  = nc % C;

    // 3) bounds check
    if (h_out >= H_out || w_out >= W_out) {
        return;
    }
    
    // 4) pooling window start in input coordinates (account for padding)
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // 5) compute max
    float max_val = -FLT_MAX;

    for (int kh = 0; kh < kernel_size; kh++) {
        int h_in = h_start + kh;
        if (h_in < 0 || h_in >= H) continue;

        for (int kw = 0; kw < kernel_size; kw++) {
            int w_in = w_start + kw;
            if (w_in < 0 || w_in >= W) continue;

            // input index: ((n*C + c)*H + h_in)*W + w_in
            int in_idx = ((n * C + c) * H + h_in) * W + w_in;
            float v = input[in_idx];
            if (v > max_val)  {
                max_val = v;
            }
        }
    }

    // 6) write output
    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = max_val;
}

extern "C" void solve(const float* input, float* output,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {

    int h_out = (H + 2 * padding - kernel_size) / stride + 1;
    int w_out = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (w_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (h_out + threadsPerBlock.y - 1) / threadsPerBlock.y,
        N * C
    );

    max_pool_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output,
        N, C, H, W,
        kernel_size, stride, padding,
        h_out, w_out
    );

    cudaDeviceSynchronize();
}