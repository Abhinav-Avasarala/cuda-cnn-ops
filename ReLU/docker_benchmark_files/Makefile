CXX ?= g++
CC  ?= gcc

CXXFLAGS ?= -O3 -std=c++17 -march=native -ffast-math
CFLAGS   ?= -O3 -march=native -ffast-math

OPENMP_CFLAGS   ?= -fopenmp
OPENMP_CXXFLAGS ?= -fopenmp

BIN_DIR := bin
BENCH_BIN := $(BIN_DIR)/bench_cpu_relu

.PHONY: all bench clean

all: bench

bench: $(BENCH_BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/bench_cpu_relu: $(BIN_DIR)/bench_cpu_relu.o $(BIN_DIR)/relu_single_thread_cpu.o $(BIN_DIR)/omp_relu.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_CXXFLAGS) $^ -o $@

$(BIN_DIR)/bench_cpu_relu.o: bench_cpu_relu.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_CXXFLAGS) -c $< -o $@

$(BIN_DIR)/relu_single_thread_cpu.o: ../relu_single_thread_cpu.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/omp_relu.o: ../omp_relu.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_DIR)
