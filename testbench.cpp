#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include "kernel_gemm.h"

/* Array initialization. */ 
static void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ]) { 
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

//CPU reference
static void kernel_gemm_reference(float*C, float*A, float*B, float alpha, float beta){ 
    int i, j, k; 
    for (int i = 0; i < NI; i++){ 
        for (int j = 0; j < NJ; j++){ 
            C[i*NJ + j] *= beta; 
        } 
        for (int j = 0; j < NJ; j++){ 
            float sum = 0.0f; 
            for (int k = 0; k < NK; k++){
                sum += A[i*NK + k] * B[k*NJ + j]; 
            } 
            C[i*NJ + j] += alpha * sum; 
        } 
    } 
}
//for comparing the HLS and CPU sums
static float checksum_C(const float* C_arr) {
    float s = 0; 
    for (int idx = 0; idx < NI*NJ; idx++) {
        s += C_arr[idx]; 
    } 
    return s; 
}

int main(int argc, char** argv){ 
/* Variable declaration/allocation. */ 
float *A = (float *)malloc(NI*NK*sizeof(float)); 
float *B = (float *)malloc(NK*NJ*sizeof(float)); 
float *C = (float *)malloc(NI*NJ*sizeof(float)); 
//CPU refernce 
float *C_reference = (float *)malloc(NI*NJ*sizeof(float));

/* Initialize array(s). */ 
init_array (C, A, B);
/*C transfer*/ 
//so that both the HLS and the CPU reference use the same C 
int c_idx = 0; 
for(c_idx; c_idx < NI*NJ; c_idx++){ 
    C_reference[c_idx] = C[c_idx]; 
}

/* Run kernel. */ 
kernel_gemm (C, A, B, 1.5, 2.5);
/*CPU refernce*/ 
float sum =0;
int i, j;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

kernel_gemm_reference(C_reference, A, B, 1.5, 2.5); 


int errors = 0;
const float tolerance = 1e-2f;  

/* What do you think about instead of getting the entire sum, round individual sum calculation to remove the tiny errors
for (int idx = 0; idx < NI*NJ; idx++) {
	long hls_val = (long)(C[i*NJ + j] + 0.5f);
        long cpu_val = (long)(C_reference[i*NJ + j] + 0.5f);
        
        if (hls_val - cpu_val != 0) {
            errors++;
        }
}   
*/
for (int idx = 0; idx < NI*NJ; idx++) {
    float difference = fabsf(C[idx] - C_reference[idx]);
    if (difference > tolerance) {
        errors++;
    }
}
if (errors == 0) {
    printf("PASS (tolerance=%f)\n", tolerance);
} else {
    printf("FAIL: %d mismatches (tolerance=%f)\n", errors, tolerance);
}

//get and compare the HLS and CPU refernce values 
float HLS_sum = checksum_C(C); 
float CPU_sum = checksum_C(C_reference); 
float comparison = fabsf(HLS_sum - CPU_sum);
  
printf("sum of CPU_sum array = %f\n", CPU_sum);
printf("sum of HLS_sum array = %f\n", HLS_sum);
printf("sum of C array = %f\n", sum);
printf("sum of comparison = %f\n", comparison);

int pass = (errors == 0);  

free(A); 
free(B); 
free(C); 
free(C_reference);

if (pass){
    return 0;
}
else
    return 1;

}
