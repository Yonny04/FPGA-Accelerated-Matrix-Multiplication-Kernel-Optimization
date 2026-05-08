# FPGA-Accelerated Matrix Multiplication Kernel Optimization

This repository contains a high-performance Matrix Multiplication kernel optimized for FPGA architectures (Zynq-7000/UltraScale+). The project focuses on leveraging hardware parallelism, memory tiling, and AXI-stream interfaces to achieve significant speedups over traditional CPU-based implementations.

## 🚀 Overview

The core objective of this project is to optimize the $C = A \times B$ operation by minimizing memory latency and maximizing throughput using:
* **Loop Unrolling & Pipelining:** Increasing instruction-level parallelism.
* **Memory Tiling (Blocking):** Optimizing cache/BRAM hits to reduce global memory access.
* **Resource Mapping:** Efficient use of DSP48 slices for MAC (Multiply-Accumulate) operations.

---

## 📂 Repository Structure

```text
# FPGA-Accelerated Matrix Multiplication Kernel Optimization

This repository is a collection of high-performance computing (HPC) optimizations targeting various architectures, including FPGAs, GPUs (CUDA), and Multi-Core CPUs. The primary focus is on accelerating the Matrix Multiplication kernel through hardware parallelism and memory hierarchy optimizations.

---

## 📂 Repository Structure

The project is organized into specific optimization domains:

```text
├── CPU-GEMM-Optimization-SIMD/
│   └── src/                      # CPU-based General Matrix Multiply (GEMM)
│       ├── Makefile              # Build script for SIMD optimizations
│       ├── mm.cpp                # Logic utilizing AVX/SSE SIMD instructions
│       └── my_timer.h            # Timing utility for profiling
│
├── CUDA-Image-Blur-Optimization/
│   └── G30/                      # GPU-accelerated image processing
│       ├── imgBlur.cu            # CUDA kernel for parallel image blurring
│       ├── Makefile              # GPU build configuration
│       ├── libwb/                # WebGPU-style logging/testing library
│       ├── input.ppm             # Sample input asset
│       ├── golden_output.ppm     # Reference output for verification
│       └── my_timer.h            # CUDA-compatible timing utility
│
├── Multi-Core-CPU-Optimization/
│   └── G30/                      # Multi-threaded CPU optimizations
│       ├── mm.cpp                # Matrix Multiply using OpenMP/Pthreads
│       ├── Makefile              # Multi-core build instructions
│       └── my_timer.h            # Performance profiling header
│
└── Vitis-HLS-Matrix-Multiplication.zip 
                                  # Compressed Vitis HLS workspace for FPGA
