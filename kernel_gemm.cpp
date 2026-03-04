#include <stdio.h>
#include <stdlib.h>
#include "kernel_gemm.h"

#include <stdio.h>
#include <stdlib.h>


static float loc_A[TS][TS];
static float loc_B[TS][TS];
static float loc_C[TS][TS];


static void load_A (float A[NI*NK], int i_idx, int k_idx, float alpha){
    //maybe pipeline here too or inlining 
    //look into inlining 
    #pragma HLS INLINE
Load_Loop_A_i:
	for (int i = 0; i < TS; i++){
	#pragma HLS PIPELINE II = 1
Load_Loop_K:
		for (int k = 0; k < TS; k++){
        		loc_A[i][k] = A[(i + i_idx) * NK + (k + k_idx)] * alpha;
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

static void load_C(float C[NI*NJ], int i_idx, int j_idx, float beta) {
#pragma HLS INLINE
Load_Loop_C_i:
  for (int i = 0; i < TS; i++) {
#pragma HLS PIPELINE II=1
Load_Loop_C_j:
    for (int j = 0; j < TS; j++) {
      loc_C[i][j] = beta * C[(i + i_idx) * NJ + (j + j_idx)];
    }
  }
}


void store_C(float C[NI*NJ], float loc_C[TS][TS], int i_idx, int j_idx){
#pragma HLS INLINE
Store_Loop_i:	for (int i = 0; i < TS; i++){
        	#pragma HLS PIPELINE II = 1
	Store_Loop_J:for(int j = 0; j < TS; j++){
            		// Correct indexing for a 1D flattened array
			C[(i + i_idx) * NJ + (j + j_idx)] = loc_C[i][j];
			}
    		}
}

void compute_gemm() {
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
	#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512
	#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512
	#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512

	#pragma HLS INTERFACE s_axilite port=A bundle=control
	#pragma HLS INTERFACE s_axilite port=B bundle=control
	#pragma HLS INTERFACE s_axilite port=C bundle=control
	#pragma HLS INTERFACE s_axilite port=alpha bundle=control
	#pragma HLS INTERFACE s_axilite port=beta bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

int i, j, k;
#pragma HLS ARRAY_PARTITION variable=loc_B cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=loc_C cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=loc_A complete dim=2

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  

Main_Loop_i: 
	for (i = 0; i < NI; i+=TS) {
				#pragma HLS PIPELINE II = 1
Main_Loop_J: 	for (j = 0; j < NJ; j+=TS) {
			load_C(C, i, j, beta);
Main_Loop_K:		for (k = 0; k < NK; k+=TS) { 
				//load_tiles(A, B, C, loc_A, loc_B, loc_C, i, j, k);	   
				load_A(A, i, k, alpha);
				load_B(B, k, j);
				compute_gemm();
			}
			store_C(C, loc_C, i, j);
		}
  	}
}
