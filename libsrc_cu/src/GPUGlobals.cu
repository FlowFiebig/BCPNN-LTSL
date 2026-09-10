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

#include "Globals.h"
#include "GPUGlobals.cuh"

using namespace std;
using namespace Globals;
using namespace GPUGlobals;

bool GPUGlobals::cublasinitiated = false;
cublasHandle_t GPUGlobals::cubhandle;
bool GPUGlobals::curandinitiated = false;
curandGenerator_t GPUGlobals::gen_cu;
curandStatus_t GPUGlobals::istat;

int GPUGlobals::blockSize = 128, GPUGlobals::blockSize1 = 1,
    GPUGlobals::numBlocks = 1, GPUGlobals::numBlocksi = 1, GPUGlobals::numBlocksj = 1,
    GPUGlobals::numBlocksji = 1, GPUGlobals::numBlocks_hj = 1, GPUGlobals::numBlocks_nj = 1,
    GPUGlobals::numBlocks_dhi = 1, GPUGlobals::numBlocks_hjdhi = 1, GPUGlobals::numBlocks_njdni = 1;

const char *GPUGlobals::CUBLAS_CHECK_ERROR(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

// const char *GPUGlobals::HIPBLAS_CHECK_ERROR(hipblasStatus_t error) {
//     switch (error) {
//         case HIPBLAS_STATUS_SUCCESS:
//             return "HIPBLAS_STATUS_SUCCESS";
//         case HIPBLAS_STATUS_NOT_INITIALIZED:
//             return "HIPBLAS_STATUS_NOT_INITIALIZED";
//         case HIPBLAS_STATUS_ALLOC_FAILED:
//             return "HIPBLAS_STATUS_ALLOC_FAILED";
//         case HIPBLAS_STATUS_INVALID_VALUE:
//             return "HIPBLAS_STATUS_INVALID_VALUE";
//         case HIPBLAS_STATUS_ARCH_MISMATCH:
//             return "HIPBLAS_STATUS_ARCH_MISMATCH";
//         case HIPBLAS_STATUS_MAPPING_ERROR:
//             return "HIPBLAS_STATUS_MAPPING_ERROR";
//         case HIPBLAS_STATUS_EXECUTION_FAILED:
//             return "HIPBLAS_STATUS_EXECUTION_FAILED";
//         case HIPBLAS_STATUS_INTERNAL_ERROR:
//             return "HIPBLAS_STATUS_INTERNAL_ERROR";
//     }
//     return "<unknown>";
// }


void GPUGlobals::cublas_setup() {
    if (cublasinitiated)
        return;
    CUBLAS_CHECK_ERROR(cublasCreate(&cubhandle));
    cublasinitiated = true;
}

void GPUGlobals::cublas_destroy() {
    warning("GPUGlobals::cublas_destroy", "Not yet implemented");
}

void GPUGlobals::curand_setup() {
    if (curandinitiated)
        return;
    /* Create pseudo-random number generator */
    istat = curandCreateGenerator(&gen_cu, CURAND_RNG_PSEUDO_DEFAULT);
    if (istat != CURAND_STATUS_SUCCESS)
        error("curand_setup", "CURAND initialization failed", 2);
    /* Set seed */
    unsigned long long seed = ggetseed(); // 1234ULL;
    // unsigned long long seed = static_cast<unsigned long long>(time(NULL));
    istat = curandSetPseudoRandomGeneratorSeed(gen_cu, seed);
    if (istat != CURAND_STATUS_SUCCESS)
        error("curand_setup", "CURAND setseed failed", 3);
    curandinitiated = true;
}

void GPUGlobals::cugsetseed(unsigned long long seed) {
    if (not curandinitiated)
        // error("cugsetseed", "CURAND setseed failed", 3);
        curand_setup();
    seed = static_cast<unsigned long long>(time(NULL));
    istat = curandSetPseudoRandomGeneratorSeed(gen_cu, seed);
    if (istat != CURAND_STATUS_SUCCESS)
        error("curand_setup", "CURAND setseed failed", 4);
}
