#include <iostream>
#include <cstdlib>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Params.h"
#include "SmithWaterman.h"



int main(int argc, char **argv) {
    Params params;
    if (!params.parse(argc, argv)) {
        return -1;
    }

    SmithWaterman alg{params};
    alg.search();
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr) {
        std::cerr<<"kernel launch failed with error "<<cudaGetErrorString(cudaerr);
        return -1;
    }
    alg.print();
    return 0;
}
