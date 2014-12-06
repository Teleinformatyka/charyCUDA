#include <iostream>
#include <cstdlib>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Params.h"
#include "SmithWaterman.ch"


void cudaWraper(dim3, dim3);

int main(int argc, char **argv) {
    Params params;
    if (!params.parse(argc, argv)) {
        return -1;
    }

    auto test = 1;
    dim3 grid( 3 );
    dim3 block( 3, 32 );
    SmithWaterman alg{params};
    alg.search(grid, block);
    cudaWraper(grid, block);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr) {
        std::cerr<<"kernel launch failed with error "<<cudaGetErrorString(cudaerr);
        return -1;
    }
    return 0;
}
