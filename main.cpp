#include <iostream>
#include <cstdlib>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>


void cudaWraper(dim3, dim3);

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr<<"Za malo argumentow"<<std::endl;
        return 1;
    }
    auto test = 1;
    dim3 grid( 512 );
    dim3 block( 1024, 1024 );
    cudaWraper(grid, block);
    return 0;
}
