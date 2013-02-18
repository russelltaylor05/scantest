#include <sys/stat.h>
#include <sys/mman.h> 
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>

#define N 100

/* 
 * Handles CUDA errors, taking from provided sample code on clupo site
 */
/*
static void HandleError( cudaError_t err, const char * file, int line)
{
  if(err !=cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
*/


__global__ void MMKernel(int *deviceInput)
{
  printf("hi");
  printf("%d, \n", threadIdx.x);
}


__global__ void test()
{
  printf("hi");
  printf("%d, \n", threadIdx.x);
}


int main (int argc, const char * argv[])
{ 
  int inputArray[N];  
  int outArray[N];

  int *deviceInput, *deviceOutput;
  int i = 0;
  int size;
  
  for (i = 0; i < N; i++) {
    inputArray[i] = i;
  }
  
  outArray[0] = 0;
  for(i = 1; i < N; i++) {
    outArray[i] = outArray[i-1] + inputArray[i];
  }
  
  for (i = 0; i < N; i++) {
    //printf("%d, ", outArray[i]);
  }

  /* Malloc and Copy space on GPU */
  size = N * sizeof(int);
  cudaMalloc(&deviceInput, size);
  cudaMemcpy(deviceInput, inputArray, size, cudaMemcpyHostToDevice);

  size = N * sizeof(int);
  cudaMalloc(&deviceOutput, size);

  /*
  dim3 dimGrid(1,1);
  dim3 dimBlock(10,10);
  MMKernel<<<dimGrid,dimBlock>>>(deviceInput);
  */
  test<<<1,1>>>();

  cudaMemcpy(outArray, deviceOutput, size, cudaMemcpyDeviceToHost);    
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  /*
  size = Brow * Bcol * sizeof(TYPEUSE);
  cudaMalloc(&B_d, size);
  cudaMemcpy(B_d, Bmatrix, size, cudaMemcpyHostToDevice);

  size = Arow * Bcol * sizeof(TYPEUSE);
  cudaMalloc(&C_d, size);
  
  blockRow = (Arow+31) / 32;
  blockCol = (Bcol+31) / 32;
    
  dim3 dimGrid(blockCol,blockRow);
  dim3 dimBlock(32,32);
  MMKernel<<<dimGrid,dimBlock>>>(deviceInput);

  cudaMemcpy(Cmatrix,C_d,size, cudaMemcpyDeviceToHost);

  output_matrix(Cfile, Cmatrix, Arow, Bcol);
  
  //print_matrix(Cmatrix, Arow, Bcol);  
  */
  

  
  printf("\n");
  return 0;
}
