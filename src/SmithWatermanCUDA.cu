#include <cstdio>



#include "smith_waterman_params.h"



#define CUERR do{ cudaError_t err;    .\
if ((err = cudaGetLastError()) != cudaSuccess) {    \
            int device;\
        cudaGetDevice(&device);\
    printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(err), __FILE__, __LINE__); }}while(0);


__constant__ int g_match;
__constant__ int g_mismatch;
__constant__ int g_gap_penalty;
__constant__ int g_rows_count;





__device__ inline  void get_value(long long match, long long deletion, long long insertion, long long &value, char &direction){
    value = max((long long)0, max(match, max(deletion, insertion)));
  direction = 0;


  if(value == 0){ return; }

  if     (value == match)    { direction = 1; }
  else if(value == deletion) { direction = 2; }
  else if(value == insertion){ direction = 3; }


}


__global__ void runCUDA(CUDA params, Column column) {
    long long match, deletion, insertion, value;

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int x = params.iteration - id;
    int y = id * params.cells_per_thread + 1;
    int end_y = y + params.cells_per_thread;
    
    int tmp_y = y - 1;

    char direction = 0;
    if (x < 0 ||  x >= params.columns_count) {
        return;
    }

    __shared__ extern long columns_before_prev[];
    __shared__ extern long columns_prev[];
    __shared__ extern long columns_current[];
    printf(" %d ", );
    printf(" x = %d | ", threadIdx.x);
    match = column.before_prev[tmp_y] + (params.sequence_1[x] == params.sequence_2[tmp_y] ? g_match : g_mismatch);
    deletion = column.prev[y] + g_gap_penalty;
    insertion = column.prev[tmp_y] + g_gap_penalty;

    get_value(match, deletion, insertion, value, direction);

    column.current[y] = value;
    params.directions[tmp_y] = direction;


    y++;

    while(y < end_y && y <= g_rows_count ){
        tmp_y = y - 1;

        match = column.prev[tmp_y] + (params.sequence_1[x] == params.sequence_2[tmp_y] ? g_match : g_mismatch);
        deletion = column.prev[y] + g_gap_penalty;
        insertion = column.current[tmp_y] + g_gap_penalty;

        get_value(match, deletion, insertion, value, direction);

        column.current[y] = value;
        params.directions[tmp_y] = direction;

        y++;
    }


}
void searchCUDA(CUDA_params &params) {
    cudaMemset( params.cuda.directions, 0, params.directions_size );
    cudaMemcpy( params.cuda.column.before_prev, params.cuda.column.prev, params.cuda.column.size, cudaMemcpyDeviceToDevice );
    cudaMemcpy( params.cuda.column.prev, params.cuda.column.current, params.cuda.column.size, cudaMemcpyDeviceToDevice );
    cudaMemset( params.cuda.column.current, 0, params.cuda.column.size );
    runCUDA<<<params.cuda.blocks_count, params.cuda.threads_per_block, params.sequence_2->size + 1>>>(params.cuda, params.cuda.column);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudaError_t err;   
        if ((err = cudaGetLastError()) != cudaSuccess) {    
               int device;
              cudaGetDevice(&device);
          printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(err), __FILE__, __LINE__); 
          return ;
        }
#endif

  cudaMemcpy( params.result.directions, params.cuda.directions, params.directions_size, cudaMemcpyDeviceToHost );
  cudaMemcpy( params.result.column, params.cuda.column.current, params.cuda.column.size, cudaMemcpyDeviceToHost );
    printf("-------------------------")    ;

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
    
    cudaMemcpyToSymbol(g_mismatch, &params.cuda.mismatch, sizeof(params.cuda.mismatch));
    cudaMemcpyToSymbol(g_match, &params.cuda.match, sizeof(params.cuda.match));
    cudaMemcpyToSymbol(g_gap_penalty, &params.cuda.gap_penalty, sizeof(params.cuda.gap_penalty));
    cudaMemcpyToSymbol(g_rows_count, &params.cuda.rows_count, sizeof(params.cuda.rows_count));

#ifdef DEBUG
    cudaError_t err;   
    if ((err = cudaGetLastError()) != cudaSuccess) {    
           int device;
          cudaGetDevice(&device);
      printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(err), __FILE__, __LINE__); 
      return ;
    }
#endif


}


void deinitCUDA(CUDA &cuda) {
    cudaFree(cuda.sequence_1);
    cudaFree(cuda.sequence_2);
    cudaFree(cuda.column.current);
    cudaFree(cuda.column.prev);
    cudaFree(cuda.column.before_prev);
    cudaFree(cuda.directions);

}
