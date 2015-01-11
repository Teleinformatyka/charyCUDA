#include <cstdio>
#include <algorithm>    // std::max



#include "smith_waterman_params.h"



#define CUERR do{ cudaError_t err;    .\
if ((err = cudaGetLastError()) != cudaSuccess) {    \
            int device;\
        cudaGetDevice(&device);\
    printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(err), __FILE__, __LINE__); }}while(0);



__device__ void get_value(long long match, long long deletion, long long insertion, long long value, unsigned  char &direction){
  /* direction = 0; */

  if(value == 0){ return; }
  /*  */
  /* if     (value == match)    { direction = 1; } */
  /* else if(value == deletion) { direction = 2; } */
  /* else if(value == insertion){ direction = 3; } */


}


__global__ void runCUDA(CUDA& params) {
    long long match, deletion, insertion, value;

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int x = params.iteration - id;
    int y = id * params.cells_per_thread + 1;
    int end_y = y + params.cells_per_thread;

    unsigned char direction = 0;
    bool first=true;

    while(y < end_y && y <= params.rows_count && x >= 0 && x < params.columns_count){
        match = (first == true ? params.column.before_prev : params.column.prev)[y-1] + (params.sequence_1[x] == params.sequence_2[y-1] ? params.match : params.mismatch);
    /*     deletion = params.column.prev[y] + params.gap_penalty; */
    /*     insertion = (first == true ? params.column.prev : params.column.current)[y-1] + params.gap_penalty; */
    /*  */
        /* get_value(match, deletion, insertion, value, direction); */
    /*  */
    /*  */
    /*     params.column.current[y] = value; */
    /*     params.directions[y-1] = direction; */
    /*  */
    /*     y++; */
        /* first = false; */
    }


}
void searchCUDA(CUDA_params &params) {
  cudaMemset( params.cuda.directions, 0, params.directions_size );
  cudaMemcpy( params.cuda.column.before_prev, params.cuda.column.prev, params.cuda.column.size, cudaMemcpyDeviceToDevice );
  cudaMemcpy( params.cuda.column.prev, params.cuda.column.current, params.cuda.column.size, cudaMemcpyDeviceToDevice );
  cudaMemset( params.cuda.column.current, 0, params.cuda.column.size );
 runCUDA<<<params.cuda.blocks_count, params.cuda.threads_per_block>>>(params.cuda);
 cudaError_t err;   
    if ((err = cudaGetLastError()) != cudaSuccess) {    
            int device;
        cudaGetDevice(&device);
    printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(err), __FILE__, __LINE__); 
    return ;
    }


  cudaMemcpy( params.result.directions, params.cuda.directions, params.directions_size, cudaMemcpyDeviceToHost );
  cudaMemcpy( params.result.column, params.cuda.column.current, params.cuda.column.size, cudaMemcpyDeviceToHost );

}


void initCUDA(CUDA_params &params) {
    params.cuda.column.size = (params.sequence_2->size+1) * sizeof(long); // first row is 0
    params.cuda.columns_count = params.sequence_1->size;
    params.cuda.rows_count = params.sequence_2->size;

    cudaMalloc( (void**)&params.cuda.sequence_1,     params.sequence_1->size );
    cudaMalloc( (void**)&params.cuda.sequence_2,     params.sequence_2->size );
    cudaMalloc( (void**)&params.cuda.column.current, params.cuda.column.size );
    cudaMalloc( (void**)&params.cuda.column.prev,    params.cuda.column.size );
    cudaMalloc( (void**)&params.cuda.column.before_prev, params.cuda.column.size );
    cudaMalloc( (void**)&params.cuda.directions,     params.directions_size ); 

    cudaMemcpy( params.cuda.sequence_1, params.sequence_1->data, params.sequence_1->size, cudaMemcpyHostToDevice );
    cudaMemcpy( params.cuda.sequence_2, params.sequence_2->data, params.sequence_2->size, cudaMemcpyHostToDevice );

    cudaMemset( params.cuda.column.current, 0, params.cuda.column.size );
    cudaMemset( params.cuda.column.before_prev, 0, params.cuda.column.size );

}


void deinitCUDA(CUDA &cuda) {
    cudaFree(cuda.sequence_1);
    cudaFree(cuda.sequence_2);
    cudaFree(cuda.column.current);
    cudaFree(cuda.column.prev);
    cudaFree(cuda.column.before_prev);
    cudaFree(cuda.directions);

}
