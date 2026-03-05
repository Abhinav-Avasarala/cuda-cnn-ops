#include <cstddef>

// input and output are host pointers (CPU memory).
extern "C" void cpu_single_thread_relu_kernel(const float* input, float* output, int N) {
    if (input == nullptr || output == nullptr || N <= 0) {
        return;
    }

    for (int i = 0; i < N; ++i) {
        const float x = input[i];
        output[i] = (x > 0.0f) ? x : 0.0f;
    }
}
