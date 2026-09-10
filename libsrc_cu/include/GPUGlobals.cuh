/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-07-26     Modified: 2024-07-26

******************************************************************/
/*****************************************************************

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
SuOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

******************************************************************/

#ifndef __GPUGlobals_included
#define __GPUGlobals_included

#include "Globals.h"

// /* HIP specific */
// #include <hip/hip_runtime.h>
// #include <hipblas/hipblas.h>
// #include "hiprand.h"
// #include <hiprand_kernel.h>

/* CUDA specific */
#include <curand_kernel.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#define CUDA_CHECK_ERROR(call) {                                      \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                  << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// // HIP Error checking macro
// #define HIP_CHECK_ERROR(error)                       \
//     if(error != hipSuccess)                          \
//         {                                            \
//             fprintf(stderr,                          \
//                     "Hip error: '%s'(%d) at %s:%d\n",   \
//                     hipGetErrorString(error),           \
//                     error,                              \
//                     __FILE__,                           \
//                     __LINE__);                          \
//             exit(EXIT_FAILURE);                         \
//         }


namespace GPUGlobals {
    extern int blockSize, numBlocks, numBlocksi, numBlocksj, numBlocksji,
           blockSize1, numBlocks_hj, numBlocks_nj, numBlocks_hjdhi, numBlocks_dhi, numBlocks_njdni,
           numBlocks_hjhi;

    extern bool cublasinitiated;
    extern cublasHandle_t cubhandle;

    extern bool curandinitiated;
    extern curandGenerator_t gen_cu;
    extern curandStatus_t istat;

    const char *CUBLAS_CHECK_ERROR(cublasStatus_t error);

    void cublas_setup();
    void cublas_destroy();

    void curand_setup();
    void cugsetseed(unsigned long long seed);

}

#endif // __GPUGlobals_included
