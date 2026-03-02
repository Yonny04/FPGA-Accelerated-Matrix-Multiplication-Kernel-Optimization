#ifndef KERNEL_GEMM_H
#define KERNEL_GEMM_H

#define NI 4096
#define NJ 4096
#define NK 4096
#define TS 64

void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta);
#endif
