#ifndef CUDA_KERNEL_PRIR
#define CUDA_KERNEL_PRIR
#include <cuda.h>
#include <cuda_runtime_api.h>



__global__ void calculate() {


}
// helper functions and utilities to work with CUDA
void cudaWraper (dim3 grid, dim3 block) {
    calculate<<<grid, block>>>();

}
#endif
