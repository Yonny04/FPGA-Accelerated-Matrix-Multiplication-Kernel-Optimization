#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"

#define numThread 20
#define NI 4096
#define NJ 4096
#define NK 4096

#define Tile_I 64
#define Tile_J 64
#define Tile_K 64


/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

printf("sum of C array = %f\n", sum);
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gemm(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int ii, jj, kk, i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma omp parallel for collapse(2) num_threads(numThread)
for(i=0; i < NI; i++){
   for(j=0; j < NJ; j++){
	C[i*NJ+j] *= beta;
	   }
}

#pragma omp parallel for private(i, j, k, ii, jj, kk) num_threads(numThread)
for(i=0; i<NI; i+=Tile_I){
//	for(j=0; j<NJ; j+= Tile_J){
	//      for(k=0; k < NK; k += Tile_K){

	for(k=0;k < NK; k += Tile_K){
		for(j = 0; j < NJ; j += Tile_J){

			for(ii = i; ii < i + Tile_I ; ii++){
			for(kk = k; kk < k+Tile_K; kk++){
					float a = alpha * A[ii*NK+kk];
#pragma omp simd
   for(jj=j; jj < j+ Tile_J; jj++){
      C[ii*NJ+jj] += a * B[kk*NJ+jj];

}
}
}
}
}

}
}


int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  /* Run kernel. */
  kernel_gemm (C, A, B, 1.5, 2.5);

  /* Stop and print timer. */
  toc(&timer, "kernel execution");
  
  /* Print results. */
  print_array_sum (C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}
