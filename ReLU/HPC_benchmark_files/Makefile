NVCC ?= nvcc
CXX  ?= g++
CC   ?= gcc

OPT ?= -O3

CPPFLAGS ?=
CXXFLAGS ?= $(OPT) -std=c++17
CFLAGS   ?= $(OPT)

OPENMP_CFLAGS   ?= -fopenmp
OPENMP_CXXFLAGS ?= -fopenmp

# nvcc uses a host compiler for C++ compilation. CUDA 12.0 may reject GCC > 12.
# Use a GCC 12 toolchain when needed:
#   make NVCC_CCBIN=g++-12
NVCC_CCBIN ?= $(CXX)

# If your site only provides newer GCC, you can force nvcc to proceed (not recommended):
#   make ALLOW_UNSUPPORTED=1
ALLOW_UNSUPPORTED ?= 0
ifeq ($(ALLOW_UNSUPPORTED),1)
  NVCCFLAGS += -allow-unsupported-compiler
endif

# Default gencodes cover common NCSU-era GPUs:
# - P100: sm_60
# - V100: sm_70
# - A100: sm_80
# Override if needed, e.g.:
#   make CUDA_GENCODE="-gencode arch=compute_80,code=sm_80"
CUDA_GENCODE ?= \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_80,code=sm_80

NVCCFLAGS ?= $(OPT) -std=c++17 $(CUDA_GENCODE)

ROOT := ..
BUILD_DIR := build
BIN_DIR := bin
OUT := $(BIN_DIR)/bench_all_relu

OBJS := \
	$(BUILD_DIR)/bench_all_relu.o \
	$(BUILD_DIR)/relu_cuda.o \
	$(BUILD_DIR)/relu_single_thread_cpu.o \
	$(BUILD_DIR)/omp_relu.o

.PHONY: all clean
all: $(OUT)

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

$(OUT): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(NVCC_CCBIN) $^ -Xcompiler "$(OPENMP_CXXFLAGS)" -o $@ -lgomp

$(BUILD_DIR)/bench_all_relu.o: bench_all_relu.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(NVCC_CCBIN) -Xcompiler "$(OPENMP_CXXFLAGS)" -c $< -o $@

$(BUILD_DIR)/relu_cuda.o: $(ROOT)/relu.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(NVCC_CCBIN) -c $< -o $@

$(BUILD_DIR)/relu_single_thread_cpu.o: $(ROOT)/relu_single_thread_cpu.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/omp_relu.o: $(ROOT)/omp_relu.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
