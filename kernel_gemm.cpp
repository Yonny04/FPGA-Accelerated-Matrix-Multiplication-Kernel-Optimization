#include <stdio.h>
#include <stdlib.h>
#include "kernel_gemm.h"

#include <stdio.h>
#include <stdlib.h>


static float loc_A[TS][TS];
static float loc_B[TS][TS];
static float loc_C[TS][TS];

static void load_A(float A[NI*NK], int i_idx, int k_idx, float alpha){
#pragma HLS INLINE off
Load_Loop_A_I:
    for (int i = 0; i < TS; i++) {
    #pragma HLS PIPELINE II = 1
        memcpy_wide_bus_read_float(
            &loc_A[i][0],
            (class ap_uint<LARGE_BUS> *)(A + ((i + i_idx) * NK + k_idx) / (LARGE_BUS / 32)),
            0 * sizeof(float),
            TS * sizeof(float)
        );
    }

Scale_Loop_A_I:
    for (int i = 0; i < TS; i++) {
    #pragma HLS PIPELINE II = 1
Scale_Loop_A_K:
        for (int k = 0; k < TS; k++) {
            loc_A[i][k] *= alpha;
        }
    }
}

static void load_B(float B[NK*NJ], int k_idx, int j_idx){
#pragma HLS INLINE  
Load_Loop_C_K:	
	for (int k = 0; k < TS; k++){
		#pragma HLS PIPELINE II = 1
Load_Loop_B_J:	for (int j = 0; j < TS; j++){
			loc_B[k][j] = B[(k + k_idx) * NJ + (j+j_idx)];
		}
    	}
}

static void load_C (float C[NI*NJ], float loc_C[TS][TS], int i_idx, int j_idx, float beta){
#pragma HLS INLINE
Load_Loop_C_i:
	for (int i = 0; i < TS; i++){
      		#pragma HLS PIPELINE II = 1
Load_loop_C_J:
		for (int j = 0; j < TS; j++){
        		loc_C[i][j] = C[(i + i_idx) * NJ + (j+j_idx)] * beta;
	   	}	
	}
}


void store_C(float C[NI*NJ], float loc_C[TS][TS], int i_idx, int j_idx){
#pragma HLS INLINE
Store_Loop_i: for (int i = 0; i < TS; i++){
        #pragma HLS PIPELINE II = 1
	Store_Loop_J: for(int j = 0; j < TS; j++){
            		// qCorrect indexing for a 1D flattened array
			C[(i + i_idx) * NJ + (j + j_idx)] = loc_C[i][j];
			}
    		}
}

void compute_gemm(float alpha, int k_idx) {
    #pragma HLS INLINE
    
    // We process the K dimension as the outer compute loop
Compute_Loop_k: for(int k = 0; k < TS; k++) {
	Compute_Loop_i: for(int i = 0; i < TS; i++) {
	#pragma HLS PIPELINE II=1
		Compute_Loop_j: for(int j = 0; j < TS; j++) {
					#pragma HLS UNROLL
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
#pragma HLS ARRAY_PARTITION variable=loc_B cyclic factor=64 dim=2
#pragma HLS ARRAY_PARTITION variable=loc_C cyclic factor=64 dim=2
#pragma HLS ARRAY_PARTITION variable=loc_A complete dim=2

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  

Main_Loop_i: 
	for (i = 0; i < NI; i+=TS) {
Main_Loop_J: 	for (j = 0; j < NJ; j+=TS) {
			load_C(C, loc_C, i, j, beta);
Main_Loop_K:		for (k = 0; k < NK; k+=TS) { 	   
				load_A(A, i, k, alpha);
				load_B(B, k, j);
				compute_gemm(alpha, k);
			}
			store_C(C, loc_C, i, j);
		}
  	}
}
