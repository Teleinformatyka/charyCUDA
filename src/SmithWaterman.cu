#include "SmithWaterman.h"

#include <cstdio>


#define CUERR do{ cudaError_t err;    .\
if ((err = cudaGetLastError()) != cudaSuccess) {    \
            int device;\
        cudaGetDevice(&device);\
    printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(err), __FILE__, __LINE__); }}while(0);

__global__ void searchCUDA() {
    printf("%d", threadIdx.x);
    

}

SmithWaterman::SmithWaterman() {

}

SmithWaterman::~SmithWaterman() {

}

void SmithWaterman::search(Params &params) {
    

    dim3 grid( 3 );
    dim3 block( 3, 32 );
    searchCUDA<<<grid, block>>>();
    uint *g_H;
}


