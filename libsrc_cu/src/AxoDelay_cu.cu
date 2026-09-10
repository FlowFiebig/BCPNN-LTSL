/*****************************************************************

  Created: 2024-07-28  Modified: 2024-07-28

  Authors: Anders Lansner, Naresh Ravchandran

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
// #include <stdio.h>
// #include <iostream>
// #include <cuda_runtime.h>

#include "Globals.h"
#include "GPUGlobals.cuh"
#include "AxoDelay.cuh"

using namespace std;
using namespace Globals;
using namespace GPUGlobals;

__global__
void updaxodelbuf_kernel(float *axodelbuf, int denNi, int maxidelay, const float *srcact,
                         int now) {
    const uint ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= denNi) return;
    axodelbuf[now * denNi + ni] = srcact[ni]; // Could be done by a simple memcpy on device
}


__device__
int d_getdslot(int maxidelay, int idelay, int now) {
    return (maxidelay + now - idelay) % maxidelay;
}


__global__
void upddelsrcact_kernel(float *axodelbuf, int *axodeltap, float *delsrcact, int denNi,
                         int maxidelay, int now) {
    const uint ni = blockIdx.x * blockDim.x + threadIdx.x;
    delsrcact[ni] = 0;
    int dslot = d_getdslot(maxidelay, axodeltap[ni], now);
    delsrcact[ni] = axodelbuf[dslot*denNi + ni];
}


void upddelsrcact(float *axodelbuf, int *axodeltap, float *delsrcact, int denNi, int maxidelay, int now) {
    upddelsrcact_kernel<<<numBlocks, blockSize>>>(axodelbuf, axodeltap, delsrcact, denNi,
                                                  now, maxidelay);
}


void updaxodelbuf_cpy(float *axodelbuf, int denNi, int maxidelay, const float *srcact, int now) {
    cudaMemcpy(axodelbuf + now*denNi, srcact, denNi, cudaMemcpyDeviceToDevice);
}


void updaxodelbuf(float *axodelbuf, int denNi, int maxidelay, const float *srcact, int now) {
    updaxodelbuf_kernel<<<numBlocks, blockSize>>>(axodelbuf, denNi, maxidelay, srcact, now);
    // updaxodelbuf_cpy(axodelbuf, denNi, maxidelay, srcact, now);
    now = (now + 1) % maxidelay;
}


void 
        upddelsrcact_cu(axodelbuf_cu, axodeltap, delsrcact_cu + step*denNi, denNi, now, maxidelay);
        // printf("%3d ", step);
        // for (int ni = 0; ni < denNi; ni++) printf("%2.0f ", srcact_cc[step*denNi + ni]);
        // printf(" :: ");
        // for (int ni = 0; ni < denNi; ni++) printf("%2.0f ", delsrcact_cc[step*denNi + ni]);
        // printf("\n");
        updaxodelbuf(axodelbuf_cu, denNi, maxidelay, srcact_cu + step*denNi, now);
        now = (now + 1) % maxidelay;
