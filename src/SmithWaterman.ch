#ifndef SMITH_WATERMAN
#define SMITH_WATERMAN

#include <cuda.h>
#include <cuda_runtime_api.h>


#include "Params.h"


class SmithWaterman {
public:
       __host__ __device__  SmithWaterman (Params &params);
    virtual ~SmithWaterman ();
    virtual void search(dim3 grid, dim3 block);
private:

};
#endif
