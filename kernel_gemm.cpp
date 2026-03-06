#include <stdio.h>
#include <stdlib.h>
#include "kernel_gemm.h"

#include <stdio.h>
#include <stdlib.h>


static float loc_A[TS][TS];
static float loc_B[TS][TS];
static float loc_C[TS][TS];


static void load_A(float A[NI*NK], int i_idx, int k_idx, float alpha){
#pragma HLS INLINE
Load_Loop_A_I:
    for (int i = 0; i < TS; i++) {
    	#pragma HLS PIPELINE II = 1
		#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		// float offset = (((i + i_idx) * NK + k_idx) / (LARGE_BUS / 32)) * sizeof(float);
		size_t offsetA = ((size_t)(i + i_idx) * NK + k_idx) * sizeof(float);
        	memcpy_wide_bus_read_float(
            	&loc_A[i][0],
            	(class ap_uint<LARGE_BUS> *)A,
            	offsetA,
            	TS * sizeof(float)
        	);
    	}

Scale_Loop_A_I:
    for (int j = 0; j < TS; j++) {
    	//#pragma HLS PIPELINE II=1
        for (int k = 0; k < TS; k++) {
            loc_A[j][k] *= alpha;
        }
    }
}

static void load_B(float B[NK*NJ], int k_idx, int j_idx){
#pragma HLS INLINE
Load_Loop_C_K:	
	for (int i = 0; i < TS; i++) {
		#pragma HLS PIPELINE II = 1
		// #pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		size_t offsetB = ((size_t)(i + k_idx) * NJ + j_idx) * sizeof(float);
		memcpy_wide_bus_read_float(
			&loc_B[i][0],
			(class ap_uint<LARGE_BUS> *)B,
			offsetB,
			TS * sizeof(float)
		);
	}
}

static void load_C(float C[NI*NJ], int i_idx, int j_idx, float beta) {
#pragma HLS INLINE
Load_Loop_C_i:
	for (int i = 0; i < TS; i++) {
		#pragma HLS PIPELINE II=1
		size_t offset = ((size_t)(i + i_idx) * NJ + j_idx) * sizeof(float);
		memcpy_wide_bus_read_float(
			&loc_C[i][0],
			(class ap_uint<LARGE_BUS> *)C,
			offset,
			TS * sizeof(float)
		);
	}
Scale_Loop_C:
    for (int i = 0; i < TS; i++) {
		#pragma HLS PIPELINE II = 1
        for (int j = 0; j < TS; j++) {
            loc_C[i][j] *= beta;
        }
    }
}


void store_C(float C[NI*NJ], float loc_C[TS][TS], int i_idx, int j_idx){
#pragma HLS INLINE 
Store_Loop_i:	
	for (int i = 0; i < TS; i++) {
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		size_t offset = ((size_t)(i + i_idx) * NJ + j_idx) * sizeof(float);
		memcpy_wide_bus_write_float(
			(class ap_uint<LARGE_BUS> *)C,
			&loc_C[i][0],
			offset,
			TS * sizeof(float)
		);
	}
}

void compute_gemm() {
#pragma HLS INLINE off
// We process the K dimension as the outer compute loop
Compute_Loop_k: 
	for(int k = 0; k < TS; k++) {
		#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
        for(int i = 0; i < TS; i++) {
			#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
			#pragma HLS PIPELINE II=1
			for(int j = 0; j < TS; j++) {
				#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
				#pragma HLS UNROLL factor=64
                // loc_C stores the partial sums across K iterations
                loc_C[i][j] += loc_A[i][k] * loc_B[k][j];
            }
        }
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta){
int i, j, k;
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512

#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable=loc_B cyclic factor=64 dim=2
#pragma HLS ARRAY_PARTITION variable=loc_C cyclic factor=64 dim=2
#pragma HLS ARRAY_PARTITION variable=loc_A cyclic factor=64 dim=2

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  
Main_Loop_i: 
	for (i = 0; i < NI; i+=TS) {
		#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
		for (j = 0; j < NJ; j+=TS) {
			#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
			load_C(C, i, j, beta);
			for (k = 0; k < NK; k+=TS) { 			
				#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
				load_A(A, i, k, alpha);
				load_B(B, k, j);
				compute_gemm();
			}
			store_C(C, loc_C, i, j);
		}
  	}
}