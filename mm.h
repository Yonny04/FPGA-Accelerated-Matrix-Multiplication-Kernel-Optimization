#ifndef KERNEL_GEMM_H
#define KERNEL_GEMM_H

#ifndef NI
#define NI 1024
#endif
#ifndef NJ
#define NJ 1024
#endif
#ifndef NK
#define NK 1024
#endif
#ifndef TS
#define TS 64
#endif


#include <ap_int.h>
#define LARGE_BUS 512
typedef ap_uint<LARGE_BUS> MARS_WIDE_BUS_TYPE;

#ifdef __cplusplus
extern "C" {
#endif
void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta);
#ifdef __cplusplus
}
#endif

#endif
