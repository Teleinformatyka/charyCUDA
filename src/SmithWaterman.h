#ifndef SMITH_WATERMAN
#define SMITH_WATERMAN

#include <cuda.h>
#include <cuda_runtime_api.h>


#include "Params.h"


class SmithWaterman {
public:
       __host__ __device__  SmithWaterman ();
    virtual ~SmithWaterman ();
    virtual void search(Params &params);
private:

};
#endif
