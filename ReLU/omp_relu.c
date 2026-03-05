#include <omp.h>

void omp_relu_kernel(const float* input, float* output, int N) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const float x = input[i];
        output[i] = (x > 0.0f) ? x : 0.0f;
    }
}