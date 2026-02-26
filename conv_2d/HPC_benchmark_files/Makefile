NVCC ?= nvcc
CXX  ?= g++
CC   ?= gcc

OPT ?= -O3

CPPFLAGS ?=
CXXFLAGS ?= $(OPT) -std=c++17
CFLAGS   ?= $(OPT)

OPENMP_CFLAGS   ?= -fopenmp
OPENMP_CXXFLAGS ?= -fopenmp

# Default targets should run on V100 (sm_70) and A100 (sm_80).
# Override if needed, e.g.: make CUDA_GENCODE="-gencode arch=compute_90,code=sm_90"
CUDA_GENCODE ?= -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80

NVCCFLAGS ?= $(OPT) -std=c++17 $(CUDA_GENCODE)

ROOT := ..
BUILD_DIR := build
BIN_DIR := bin
OUT := $(BIN_DIR)/bench_all_conv2d

OBJS := \
	$(BUILD_DIR)/bench_all_conv2d.o \
	$(BUILD_DIR)/conv_2d_naive.o \
	$(BUILD_DIR)/single_thread_conv2d.o \
	$(BUILD_DIR)/omp_conv2d.o

.PHONY: all clean
all: $(OUT)

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

$(OUT): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -Xcompiler "$(OPENMP_CXXFLAGS)" -o $@ -lgomp

$(BUILD_DIR)/bench_all_conv2d.o: bench_all_conv2d.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(OPENMP_CXXFLAGS)" -c $< -o $@

$(BUILD_DIR)/conv_2d_naive.o: $(ROOT)/conv_2d_naive.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/single_thread_conv2d.o: $(ROOT)/single_thread_conv2d.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/omp_conv2d.o: $(ROOT)/omp_conv2d.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

