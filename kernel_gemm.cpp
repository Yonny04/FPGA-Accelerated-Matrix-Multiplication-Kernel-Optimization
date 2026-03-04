#include <stdio.h>
#include <stdlib.h>
#include "kernel_gemm.h"

#include <stdio.h>
#include <stdlib.h>


static float loc_A[3][TS][TS];
static float loc_B[3][TS][TS];
static float loc_C[3][TS][TS];


static void load_A (float A[NI*NK], int i_idx, int k_idx, float alpha, int pp){
//maybe pipeline here too or inlining 
//look into inlining 
#pragma HLS INLINE
Load_Loop_A_i:
	for (int i = 0; i < TS; i++){
		#pragma HLS PIPELINE II = 1
		for (int k = 0; k < TS; k++){
        		loc_A[pp][i][k] = A[(i + i_idx) * NK + (k + k_idx)] * alpha;
      		}
    	}
}

static void load_B(float B[NK*NJ], int k_idx, int j_idx, int pp){
#pragma HLS INLINE  
Load_Loop_C_K:	
	for (int k = 0; k < TS; k++){
		#pragma HLS PIPELINE II = 1
		for (int j = 0; j < TS; j++){
			loc_B[pp][k][j] = B[(k + k_idx) * NJ + (j+j_idx)];
		}
    	}
}

static void load_C(float C[NI*NJ], int i_idx, int j_idx, float beta, int pp, int first_K) {
#pragma HLS INLINE
Load_Loop_C_i:
	for (int i = 0; i < TS; i++) {
		#pragma HLS PIPELINE II=1
		for (int j = 0; j < TS; j++) {
			if(first_K == 1){
				loc_C[pp][i][j] = beta * C[(i + i_idx) * NJ + (j + j_idx)];
			}
			else{
				loc_C[pp][i][j] = C[(i + i_idx) * NJ + (j + j_idx)];			
			}
		}
	}
}


void store_C(float C[NI*NJ], int i_idx, int j_idx, int pp){
#pragma HLS INLINE
Store_Loop_i:	
	for (int i = 0; i < TS; i++){
		#pragma HLS PIPELINE II = 1
		for(int j = 0; j < TS; j++){
			// Correct indexing for a 1D flattened array
			C[(i + i_idx) * NJ + (j + j_idx)] = loc_C[pp][i][j];
		}
	}
}

void compute_gemm(int pp) {
#pragma HLS INLINE
Compute_Loop_k: 
	for(int k = 0; k < TS; k++) {
		for(int i = 0; i < TS; i++) {
			#pragma HLS PIPELINE II=1
			for(int j = 0; j < TS; j++) {
                		#pragma HLS UNROLL
				// loc_C stores the partial sums across K iterations
				loc_C[pp][i][j] += loc_A[pp][i][k] * loc_B[pp][k][j];
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

#pragma HLS ARRAY_PARTITION variable=loc_B complete dim=3
#pragma HLS ARRAY_PARTITION variable=loc_C complete dim=3
#pragma HLS ARRAY_PARTITION variable=loc_A complete dim=3

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  
// location of tiles i, j, k
int ti[3], tj[3], tk[3];

// firstK for beta Scale and lastK for storing into C
int first_K[3];

const int NTI = NI / TS;
const int NTJ = NJ / TS;
const int NTK = NK / TS;
const int TileSize = NTI * NTJ * NTK; // Total size of the tiles in matrix

Main_Loop_i: 
	// TileSize + 2 bc first two cycle is load and compute
	for (int i = 0; i < TileSize + 2; i++){

		int pp = i % 3; // pingpong rotation index
		int cur_ti = (i / (NTJ * NTK)) * TS; // Store i location 
		int cur_tk = ((i / NTJ) % NTK) * TS; // Store k location
		int cur_tj = (i % NTJ) * TS; // Store j location

		if (i < TileSize){
		    ti[pp] = cur_ti; // location of i tile
		    tj[pp] = cur_tj;
		    tk[pp] = cur_tk;
		    first_K[pp] = (cur_tk == 0);
		}

		switch(pp){
			case 0:
				if (i >= 2) store_C(C, ti[1], tj[1], 1);
				if (i >= 1 && i < TileSize + 1) compute_gemm(2);
				if (i < TileSize){ 
					load_A(A, ti[0], tk[0], alpha, 0);
					load_B(B, tk[0], tj[0], 0);
					load_C(C, ti[0], tj[0], beta, 0, first_K[0]);
				}
				break;

			case 1:
				if (i >= 2) store_C(C, ti[2], tj[2], 2);
				if (i >= 1 && i < TileSize + 1) compute_gemm(0);
				if(i < TileSize){
					load_A(A, ti[1], tk[1], alpha, 1);
					load_B(B, tk[1], tj[1], 1);
					load_C(C, ti[1], tj[1], beta, 1, first_K[1]);

				}
				break;

			case 2:
				if (i >= 2) store_C(C, ti[0], tj[0], 0);
				if (i >= 1 && i < TileSize + 1) compute_gemm(1);
				if(i < TileSize){
					load_A(A, ti[2], tk[2], alpha, 2);
					load_B(B, tk[2], tj[2], 2);
					load_C(C, ti[2], tj[2], beta, 2, first_K[2]);
				}
				break;
		}			
/*
Main_Loop_i: 
	for (i = 0; i < NI; i+=TS) {
				#pragma HLS PIPELINE II = 1
		for (j = 0; j < NJ; j+=TS) {
			load_C(C, i, j, beta);
			for (k = 0; k < NK; k+=TS) { 
				//load_tiles(A, B, C, loc_A, loc_B, loc_C, i, j, k);	   
				load_A(A, i, k, alpha);
				load_B(B, k, j);
				compute_gemm();
			}
			store_C(C, loc_C, i, j);
		}
  	}
*/
	}
}
