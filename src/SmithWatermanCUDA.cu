
#include <cstdio>


#define CUERR do{ cudaError_t err;    .\
if ((err = cudaGetLastError()) != cudaSuccess) {    \
            int device;\
        cudaGetDevice(&device);\
    printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(err), __FILE__, __LINE__); }}while(0);

__global__ void runCUDA() {
    printf("%d", threadIdx.x);
    

}
void searchCUDA() {

    runCUDA<<<20, 30>>>();
    

}

