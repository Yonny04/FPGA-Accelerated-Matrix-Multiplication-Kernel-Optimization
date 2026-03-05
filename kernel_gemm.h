#ifndef KERNEL_GEMM_H
#define KERNEL_GEMM_H

#define NI 1024
#define NJ 1024
#define NK 1024
#define TS 64

#include <ap_int.h>          
#define LARGE_BUS 512       
typedef ap_uint<LARGE_BUS> MARS_WIDE_BUS_TYPE;

void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta);
#endif
