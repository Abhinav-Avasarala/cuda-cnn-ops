
Let’s turn your resume bullets into an **actual project plan**.

---

# What your resume claims you did (translated to tasks)

Your bullets imply you implemented:

1. **2D convolution kernel in CUDA**
2. **ReLU kernel**
3. **Max pooling kernel**
4. **Correctness tests vs PyTorch**
5. **Optimization using shared memory**
6. **Nsight profiling**
7. **Comparison with cuDNN**

We’ll implement these in stages.

---

# Stage 1 — Setup (30 minutes)

Install GPU environment.

You need:

* CUDA Toolkit
* PyTorch
* Nsight Compute
* C++

Directory:

```
cuda-cnn-from-scratch/
 ├── conv.cu
 ├── relu.cu
 ├── maxpool.cu
 ├── test.py
 ├── baseline_conv.cu
 ├── optimized_conv.cu
```

Compile with:

```bash
nvcc conv.cu -o conv
```

---

# Stage 2 — Write a **simple convolution kernel**

Start with **naive convolution**.

Formula:

[
output[n][c][h][w] =
\sum_{k=0}^{K-1}
\sum_{r=0}^{R-1}
\sum_{s=0}^{S-1}
input[n][k][h+r][w+s] * weight[c][k][r][s]
]

CUDA version (simple):

```cpp
__global__ void conv2d(
    float* input,
    float* weight,
    float* output,
    int H, int W,
    int K
){
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    for(int r=0; r<K; r++){
        for(int s=0; s<K; s++){
            sum += input[(h+r)*W + (w+s)] *
                   weight[r*K + s];
        }
    }

    output[h*W + w] = sum;
}
```

Launch kernel:

```cpp
dim3 block(16,16);
dim3 grid(W/16, H/16);

conv2d<<<grid,block>>>(d_input,d_weight,d_output,H,W,K);
```

That alone already gives you:

> Implemented 2D convolution kernel in CUDA

---

# Stage 3 — Implement **ReLU kernel**

Very easy.

```cpp
__global__ void relu(float* x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        x[i] = max(0.0f, x[i]);
    }
}
```

Launch:

```cpp
relu<<<n/256 + 1, 256>>>(d_tensor, n);
```

---

# Stage 4 — Implement **max pooling**

Example: 2×2 pool

```cpp
__global__ void maxpool(
    float* input,
    float* output,
    int H,
    int W
){

    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    int h_in = h*2;
    int w_in = w*2;

    float m = input[h_in*W + w_in];

    m = max(m, input[h_in*W + w_in+1]);
    m = max(m, input[(h_in+1)*W + w_in]);
    m = max(m, input[(h_in+1)*W + w_in+1]);

    output[h*(W/2) + w] = m;
}
```

Now you have:

* conv
* relu
* maxpool

Which is **basically a CNN layer**.

---

# Stage 5 — Verify correctness vs PyTorch

Write Python script.

```python
import torch
import numpy as np

x = torch.randn(1,1,32,32)
w = torch.randn(1,1,3,3)

y = torch.nn.functional.conv2d(x,w)

np.save("input.npy", x.numpy())
np.save("weight.npy", w.numpy())
np.save("output_pytorch.npy", y.numpy())
```

Then compare with CUDA output.

Compute error:

```
max absolute error
```

Now your resume bullet:

> validated correctness against PyTorch on 100+ randomized inputs

Just run loop:

```
for i in range(100):
    random input
    compare outputs
```

---

# Stage 6 — Optimize convolution (important)

Your resume mentions:

> shared memory tiling

This is the **real GPU optimization**.

Naive convolution repeatedly loads global memory.

Instead:

1. Load tile of input into **shared memory**
2. All threads reuse it

Example concept:

```
Global memory
      ↓
Shared memory tile
      ↓
Threads compute convolution
```

Example:

```cpp
__shared__ float tile[BLOCK+K][BLOCK+K];
```

Threads cooperate to load:

```cpp
tile[ty][tx] = input[global_index];
__syncthreads();
```

Then compute convolution **using shared memory instead of global memory**.

This improves:

* memory reuse
* cache locality

Result: **2–4× speedup**.

---

# Stage 7 — Profile with Nsight Compute

Run:

```
ncu ./conv
```

Look at metrics:

* memory throughput
* occupancy
* warp efficiency

You can then say:

> Profiled kernels with Nsight Compute to identify memory bandwidth bottlenecks.

---

# Stage 8 — Compare with cuDNN

Just benchmark:

PyTorch conv:

```python
torch.nn.functional.conv2d(x,w)
```

vs your kernel.

Measure time:

```python
import time
```

You’ll see:

* your kernel slower
* cuDNN heavily optimized

This gives you:

> Benchmarked against cuDNN and analyzed performance gaps.

---

# Final result (what you actually build)

A repo like:

```
cuda-cnn-from-scratch
 ├── naive_conv.cu
 ├── optimized_conv.cu
 ├── relu.cu
 ├── maxpool.cu
 ├── test_vs_pytorch.py
 ├── benchmark.py
 └── README.md
```

This is actually a **very strong ML systems project**.

---

# Your resume bullet becomes **real**

Then your bullets are **legit**:

• Implemented CNN operators (2D convolution, ReLU, max-pool) in CUDA and validated correctness against PyTorch across 100+ randomized inputs
• Optimized convolution kernel using shared-memory tiling and improved memory coalescing, achieving **2× speedup**
• Profiled kernels with **Nsight Compute** and analyzed performance vs **cuDNN**

---

# Important (career advice)

For **NVIDIA / ML systems / GPU infra roles**, this project is **extremely good**.

Much stronger than typical student projects.

Because it shows:

* GPU architecture knowledge
* CUDA
* performance optimization
* ML systems understanding

---

If you want, I can also show you the **cleanest way to implement the entire CNN pipeline (conv → relu → maxpool) in CUDA in ~150 lines**, which is how most ML systems engineers prototype this.
