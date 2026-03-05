NVCC ?= nvcc
CXX  ?= g++
CC   ?= gcc

OPT ?= -O3

CPPFLAGS ?=
CXXFLAGS ?= $(OPT) -std=c++17 -march=native -ffast-math
CFLAGS   ?= $(OPT) -march=native -ffast-math

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

# Default gencodes cover common NCSU-era GPUs.
CUDA_GENCODE ?= \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_80,code=sm_80

NVCCFLAGS ?= $(OPT) -std=c++17 $(CUDA_GENCODE)

ROOT := ..
BUILD_DIR := build
BIN_DIR := bin
BENCH_BIN := $(BIN_DIR)/bench_all_max_pooling

OBJS := \
	$(BUILD_DIR)/bench_all_max_pooling.o \
	$(BUILD_DIR)/max_pooling_cuda.o \
	$(BUILD_DIR)/max_pooling_single_thread_cpu.o \
	$(BUILD_DIR)/omp_max_pooling.o

.PHONY: all bench clean

all: bench
bench: $(BENCH_BIN)

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

$(BENCH_BIN): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(NVCC_CCBIN) $^ -Xcompiler "$(OPENMP_CXXFLAGS)" -o $@ -lgomp

$(BUILD_DIR)/bench_all_max_pooling.o: bench_all_max_pooling.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(NVCC_CCBIN) -Xcompiler "$(OPENMP_CXXFLAGS)" -c $< -o $@

$(BUILD_DIR)/max_pooling_cuda.o: $(ROOT)/max_pooling.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(NVCC_CCBIN) -c $< -o $@

$(BUILD_DIR)/max_pooling_single_thread_cpu.o: $(ROOT)/max_pooling_single_thread_cpu.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/omp_max_pooling.o: $(ROOT)/omp_max_pooling.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
