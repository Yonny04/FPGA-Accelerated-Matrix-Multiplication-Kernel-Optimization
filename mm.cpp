#include <stdio.h>
#include <stdlib.h>
#include "mm.h"
#include "mars_wide_bus.h"



static float loc_A[2][TS][TS];
static float loc_B[2][TS][TS];
static float loc_C[TS][TS];

static void load_A (float A[NI*NK], float pp_buf[TS][TS], int i_idx, int k_idx, float alpha){
    //maybe pipeline here too or inlining 
    //look into inlining 
    #pragma HLS INLINE

	//make a buffer to hold the row of the tile of A that we are loading
	//1D so it's contiguous
	float buf_A[TS];
	#pragma HLS ARRAY_PARTITION variable = buf_A complete

	//make a pointer to MARS_WIDE_BUS_TYPE to read from the wide bus
	//A_wide is a pointer to the wide bus version of A
	MARS_WIDE_BUS_TYPE *A_wide = (MARS_WIDE_BUS_TYPE *)A;

	float a = alpha;
Load_Loop_A_i:
	for (int i = 0; i < TS; i++){

	#pragma HLS PIPELINE II = 1

	//matrix row we are currently loading into the buffer
	int matrix_row = i + i_idx;
	//the beginning index of the tile 
	size_t init_idx = (size_t)matrix_row * NK + (size_t)k_idx;
	//converting to bytes
	size_t offset_byte = init_idx * sizeof(float);

	//copying the row of tile A into the buffer
	//read the row in the buffer as wide bus values
	//the offset is the starting index of the tile row in bytes
	//TS * sizeof(float) = size of tyle times 4 bytes per float = overall size of tile in bytes
	memcpy_wide_bus_read_float(buf_A, A_wide, offset_byte, TS * sizeof(float));

	#pragma HLS UNROLL
	for (int k = 0; k < TS; k++){
    	pp_buf[i][k] = buf_A[k] * a;
      	}
    }
}

static void load_B(float B[NK*NJ], float pp_buf[TS][TS], int k_idx, int j_idx){
#pragma HLS INLINE 

float buf_B[TS];
#pragma HLS ARRAY_PARTITION variable = buf_B complete

MARS_WIDE_BUS_TYPE *B_wide = (MARS_WIDE_BUS_TYPE *)B;

Load_Loop_C_K:	
	for (int k = 0; k < TS; k++){
		
		#pragma HLS PIPELINE II = 1
		int matrix_row = k + k_idx;
		size_t init_idx = (size_t)matrix_row * NJ + (size_t)j_idx;
		size_t offset_byte = init_idx * sizeof(float);
		memcpy_wide_bus_read_float(buf_B, B_wide, offset_byte, TS * sizeof(float));

Load_Loop_B_J:	
			#pragma HLS UNROLL
			for (int j = 0; j < TS; j++){
				pp_buf[k][j] = buf_B[j];
		}
    	}
}

static void load_C(float C[NI*NJ], int i_idx, int j_idx, float beta) {
#pragma HLS INLINE

float buf_C[TS];
#pragma HLS ARRAY_PARTITION variable = buf_C complete

MARS_WIDE_BUS_TYPE *C_wide = (MARS_WIDE_BUS_TYPE *)C;

Load_Loop_C_i:
  for (int i = 0; i < TS; i++) {

	#pragma HLS PIPELINE II=1	
	int matrix_row = i + i_idx;
	size_t init_idx = (size_t)matrix_row * NJ + (size_t)j_idx;
	size_t offset_byte = init_idx * sizeof(float);
	memcpy_wide_bus_read_float(buf_C, C_wide, offset_byte, TS * sizeof(float));

Load_Loop_C_j:
	#pragma HLS UNROLL
    for (int j = 0; j < TS; j++) {
      loc_C[i][j] = beta * buf_C[j];
    }
  }
}

void store_C(float C[NI*NJ], float loc_C[TS][TS], int i_idx, int j_idx){
#pragma HLS INLINE

float buf_C[TS];
#pragma HLS ARRAY_PARTITION variable = buf_C complete

MARS_WIDE_BUS_TYPE *C_wide = (MARS_WIDE_BUS_TYPE *)C;

Store_Loop_i:	
#pragma HLS PIPELINE II=1			
for (int i = 0; i < TS; i++){
	Store_Loop_J:
				#pragma HLS UNROLL
				for(int j = 0; j < TS; j++){
				buf_C[j] = loc_C[i][j];
			}
		int matrix_row = i + i_idx;
		size_t init_idx = (size_t)matrix_row * NJ + (size_t)j_idx;
		size_t offset_byte = init_idx * sizeof(float);
		memcpy_wide_bus_write_float(C_wide, buf_C, offset_byte, TS * sizeof(float));

    	}
}

void compute_gemm(float A_ppbuf[TS][TS], float B_ppbuf[TS][TS]) {
    #pragma HLS INLINE
    // We process the K dimension as the outer compute loop
    Compute_Loop_k: for(int k = 0; k < TS; k++) {
        Compute_Loop_i: for(int i = 0; i < TS; i++) {
            #pragma HLS PIPELINE II=1
            Compute_Loop_j: for(int j = 0; j < TS; j++) {
                #pragma HLS UNROLL factor=8
                // loc_C stores the partial sums across K iterations
                loc_C[i][j] += A_ppbuf[i][k] * B_ppbuf[k][j];
            }
        }
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
extern "C" void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta){
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

#pragma HLS ARRAY_PARTITION variable=loc_B complete dim=3
#pragma HLS ARRAY_PARTITION variable=loc_C cyclic factor=64 dim=2
#pragma HLS ARRAY_PARTITION variable=loc_A complete dim=3

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  

Main_Loop_i: 
	for (i = 0; i < NI; i+=TS) {
				
Main_Loop_J: 	
			#pragma HLS PIPELINE II = 1
			for (j = 0; j < NJ; j+=TS) {
			load_C(C, i, j, beta);
			
			int pp = 0;
			load_A(A, loc_A[pp], i, 0, alpha);
			load_B(B, loc_B[pp], 0, j);

			for (int k = 0; k < NK; k += TS) {
			compute_gemm(loc_A[pp], loc_B[pp]);

			int next_k = k + TS;
			if (next_k < NK) {
				int next_pp = 1 - pp;
				load_A(A, loc_A[next_pp], i, next_k, alpha);
				load_B(B, loc_B[next_pp], next_k, j);
				pp = next_pp;
			}
			}
			store_C(C, loc_C, i, j);
		}
  	}
}
