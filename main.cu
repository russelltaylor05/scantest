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

#define N 64

static void HandleError( cudaError_t err, const char * file, int line)
{
  if(err !=cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



__global__ void scanKernel(int *deviceInput, int *deviceOutput, int n)
{

  __shared__ int temp[128];
  int index = threadIdx.y * 8 + threadIdx.x;
  int pout = 0;
  int pin = 1;
  
  temp[pout * n + index] = (index > 0) ? deviceInput[index - 1] : 0;
  __syncthreads();
  
  for (int offset = 1; offset < n; offset *= 2) {

      pout = 1 - pout;
      pin  = 1 - pout;
      if(index == 1) 
        printf("%d, ", pout);

      if (index >= offset) {
        temp[pout * n + index] += temp[ pin * n + index - offset];
      } else  {
        temp[pout * n + index] = temp[ pin * n + index];        
      }
      __syncthreads();  
  }
    
  deviceOutput[index] = temp[pout * n + index];
  
}


__global__ void simpleSumReduction(int *deviceInput, int *deviceOutput, int n)
{
  int index = threadIdx.x;
  int stride;
  int i;
  deviceOutput[index] = 0;
  
  for (i = N/2;  i > 0; i >>= 1) {
    __syncthreads();   
    if(index < i) {
      if(i > 31) {
        deviceOutput[index] = deviceInput[index] + deviceInput[index + i];
      } else  {
        deviceOutput[index] = deviceOutput[index] + deviceOutput[index + i];
      }      
    }
  } 
}



int main (int argc, const char * argv[])
{ 
  int inputArray[N];  
  int outArray[N];

  int *deviceInput, *deviceOutput;
  int i = 0;
  int size;
  int sum = 0;
  
  /* Initialize Input */
  for (i = 0; i < N; i++) {
    inputArray[i] = i;
  }
  printf("INPUT array\n");
  for (i = 0; i < N; i++) { 
    printf("%d, ", inputArray[i]);
    if(!((i+1) % 8) && i != 0)
      printf("\n");
  }
  printf("\n");

  
  /* CPU Scan */
  outArray[0] = 0;
  for(i = 1; i < N; i++) {
    outArray[i] = outArray[i-1] + inputArray[i];
  }  
  //printf("CPU Output\n");
  //for (i = 0; i < N; i++) { printf("%d, ", outArray[i]);  }
  //printf("\n\n\n");

  
  /* CPU Sum */
  for(i = 1; i < N; i++) {
    sum += inputArray[i];
  }
  printf("CPU sum: %d\n\n", sum);

  



  /* clear output */
  for (i = 0; i < N; i++) {
    outArray[i] = 0;
  }


  /* Malloc and Copy space on GPU */
  size = N * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&deviceInput, size));
  HANDLE_ERROR(cudaMemcpy(deviceInput, inputArray, size, cudaMemcpyHostToDevice));

  size = N * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&deviceOutput, size));

  /*
  dim3 dimGrid(1,1);
  dim3 dimBlock(8,8);
  scanKernel<<<dimGrid,dimBlock>>>(deviceInput, deviceOutput, N);
  */
  simpleSumReduction<<<1,N>>>(deviceInput, deviceOutput, N);

  HANDLE_ERROR(cudaMemcpy(outArray, deviceOutput, size, cudaMemcpyDeviceToHost)); 
  HANDLE_ERROR(cudaFree(deviceInput));
  HANDLE_ERROR(cudaFree(deviceOutput));
  
  
  
  //printf("Results: %d\n", outArray[0]);

  /* Print Array */
  printf("GPU Output\n");
  for (i = 0; i < N; i++) { 
    printf("%d, ", outArray[i]);
    if(!((i+1) % 8) && i != 0)
      printf("\n");
  }
  printf("\n\n");

  

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
