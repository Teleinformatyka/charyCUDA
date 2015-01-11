#include "SmithWaterman.h"

#include <cstdio>

void searchCUDA();

SmithWaterman::SmithWaterman() {

}

SmithWaterman::~SmithWaterman() {

}

void SmithWaterman::search(Params &params) {


    dim3 grid( 3 );
    dim3 block( 3, 32 );
    searchCUDA();
    uint *g_H;
}


