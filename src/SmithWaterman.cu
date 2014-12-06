#include "SmithWaterman.ch"

#include <cstdio>


#define CUERR do{ cudaError_t err;    .\
if ((err = cudaGetLastError()) != cudaSuccess) {    \
            int device;\
        cudaGetDevice(&device);\
    printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(err), __FILE__, __LINE__); }}while(0);

__global__ void searchCUDA() {
    printf("%d", threadIdx.x);
}

SmithWaterman::SmithWaterman(Params &params) {

}

SmithWaterman::~SmithWaterman() {

}

void SmithWaterman::search(dim3 grid, dim3 block) {
    searchCUDA<<<grid, block>>>();
}


