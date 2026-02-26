CXX ?= g++
CC  ?= gcc

CXXFLAGS ?= -O3 -std=c++17 -march=native -ffast-math
CFLAGS   ?= -O3 -march=native -ffast-math

OPENMP_CFLAGS   ?= -fopenmp
OPENMP_CXXFLAGS ?= -fopenmp

BIN_DIR := bin
BENCH_BIN := $(BIN_DIR)/bench_cpu_conv2d

.PHONY: all bench clean

all: bench

bench: $(BENCH_BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/bench_cpu_conv2d: $(BIN_DIR)/bench_cpu_conv2d.o $(BIN_DIR)/single_thread_conv2d.o $(BIN_DIR)/omp_conv2d.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_CXXFLAGS) $^ -o $@

$(BIN_DIR)/bench_cpu_conv2d.o: bench_cpu_conv2d.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_CXXFLAGS) -c $< -o $@

$(BIN_DIR)/single_thread_conv2d.o: ../single_thread_conv2d.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/omp_conv2d.o: ../omp_conv2d.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_DIR)

