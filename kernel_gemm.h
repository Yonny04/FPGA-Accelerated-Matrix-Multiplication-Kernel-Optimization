#ifndef KERNEL_GEMM_H
#define KERNEL_GEMM_H

#define NI 1024
#define NJ 1024
#define NK 1024
#define TS 64

#define LARGE_BUS 512
#include "mars_wide_bus.h"

void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta);
#endif
