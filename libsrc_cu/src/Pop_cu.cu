/*****************************************************************

  Created: 2024-07-21  Modified: 2024-08-21

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

#include "Globals.h"
#include "GPUGlobals.cuh"
#include "Pop.cuh"

using namespace std;
using namespace Globals;
using namespace GPUGlobals;

__global__
void updsup_kernel(int N, float *lgi, float *bwsup, float *sup, float *supinf, float *act,
                   float *ada, float *sada, uint *pnoise,
                   float taumdt, float igain, float bwgain, float adgain, float tauadt,
                   float sadgain, float tausadt, float nampl, float nfreq) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (N <= n)
        return;
    supinf[n] = igain * lgi[n] + bwgain * bwsup[n];
    supinf[n] += nampl * (pnoise[n] - nfreq);
    if (adgain != 0) {
        ada[n] += (adgain * act[n] - ada[n]) * tauadt;
        supinf[n] -= ada[n];
    }
    if (sadgain != 0) {
        sada[n] += (sadgain * act[n] - sada[n]) * tausadt;
        supinf[n] -= sada[n];
    }
    sup[n] += (supinf[n] - sup[n]) * taumdt;
}

void updsup_cu(int N, float *lgi, float *bwsup, float *sup, float *supinf, float *act,
               float *ada, float *sada, uint *pnoise,
               float taumdt, float igain, float bwgain, float adgain, float tauadt,
               float sadgain, float tausadt, float nampl, float nfreq) {
    if (nampl > 0) {
        if (not curandinitiated)
            curand_setup();
        /* Generate N floats on device */
        istat = curandGeneratePoisson(gen_cu, pnoise, N, nfreq);
        if (istat != CURAND_STATUS_SUCCESS)
            error("curand_generate", "CURAND generate failed (" + to_string(istat) + ")", 4);
    }
    blockSize = 128;
    numBlocks = (N + blockSize - 1) / blockSize;
    updsup_kernel <<<numBlocks, blockSize>>> (N, lgi, bwsup, sup, supinf, act, ada, sada, pnoise, taumdt, igain, bwgain, adgain, tauadt, sadgain, tausadt, nampl, nfreq);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__global__
void fullnorm_kernel(int H, int M, float *sup, float *act, float again,
                     float *hfmax, float *hfsum, float lowest) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H)
        return;
    // Find max of sup per hypercol
    hfmax[h] = sup[h * M];
    for (int n = h * M + 1; n < (h + 1) * M; n++)
        hfmax[h] = fmax(hfmax[h], sup[n]);
    // Find sum of exp(sup) per hypercol
    hfsum[h] = 0;
    for (int n = h * M; n < (h + 1) * M; n++) {
        act[n] = expf(again * (sup[n] - hfmax[h]));
        hfsum[h] += act[n];
    }
    // Compute normalized exp(sup) per hypercol
    for (int n = h * M; n < (h + 1) * M; n++)
        act[n] /= hfsum[h];
}

__global__
void halfnorm_kernel(int H, int M, float *sup, float *act, float again,
                     float *hfmax, float *hfsum, float lowest) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H)
        return;
    hfmax[h] = sup[h * M];
    for (int n = h * M + 1; n < (h + 1) * M; n++)
        if (sup[n] > hfmax[h])
            hfmax[h] = sup[n];
    if (hfmax[h] > 0)
        for (int n = h * M; n < (h + 1) * M; n++)
            act[n] = exp(again * (sup[n] - hfmax[h]));
    else
        for (int n = h * M; n < (h + 1) * M; n++)
            act[n] = exp(again * sup[n]);
    hfsum[h] = 0;
    for (int n = h * M; n < (h + 1) * M; n++)
        hfsum[h] += act[n];
    if (hfsum[h] > 1)
        for (int n = h * M; n < (h + 1) * M; n++)
            act[n] /= hfsum[h];
}

void normact_cu(int H, int M, float *sup, float *act, int normfn, float again,
                float *hfmax, float *hfsum, float LOWESTFLT) {
    blockSize = 128;
    numBlocks = H;
    switch (normfn) {
        case FULLNORM:
            fullnorm_kernel <<<numBlocks, blockSize>>>(H, M, sup, act, again, hfmax, hfsum, LOWESTFLT);
            break;
        case HALFNORM:
            halfnorm_kernel <<<numBlocks, blockSize>>>(H, M, sup, act, again, hfmax, hfsum, LOWESTFLT);
            break;
        default:
            error("::normact_cu", "No such normfn", 6);
    }
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}


__global__
void expact_kernel(int N, float *sup, float *act) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (N <= n)
        return;
    act[n] = exp(sup[n]);
}


__global__
void wtaact_kernel(int H, int M, float *sup, float *act, float *hfmax, int *nimax) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (H <= h)
        return;
    hfmax[h] = sup[h * M];
    nimax[h] = h * M;
    for (int n = h * M + 1; n < (h + 1) * M; n++) {
        if (sup[n] > hfmax[h]) {
            hfmax[h] = sup[n];
            nimax[h] = n;
        }
    }
    act[nimax[h]] = 1;
}

/// THIS IS A MODIFIER TO ACT, NOT COMPUTING ACT FROM SUP
__global__
void spkact_kernel(int H, int M, float timestep, float *sup, float *act, float *spkthres, float maxfq) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (H <= h)
        return;
    // Scale to maxfq
    for (int n = h * M; n < (h + 1)*M; n++)
        act[n] = act[n] * timestep * maxfq ;
    // Generate spike
    for (int n = h * M; n < (h + 1)*M; n++) {
        act[n] = (float)(spkthres[n] < act[n]);
    }
}

__global__
void stcwta_kernel(float* sup, int *himax, float *cumsum, float again, float* act, float *spkthres,
                   float maxfq, float timestep, int H, int M) {

    /* draw one spike as winner-takes-all from firing rate */

    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    
    float maxsup = sup[M*h], sumexpsup = 0;  
    for (int m=0; m<M; m++) maxsup = fmax(maxsup, again * sup[M*h+m]); 
    for (int m=0; m<M; m++) act[M*h+m] = expf(again * sup[M*h+m] - maxsup);
    for (int m=0; m<M; m++) sumexpsup += act[M*h+m];  
    for (int m=0; m<M; m++) act[M*h+m] /= sumexpsup;
    for (int m=0; m<M; m++) act[M*h+m] = act[M*h+m] * timestep * maxfq ; // scale to maxfq
    for (int m=0; m<M; m++) {
        if (m==0) cumsum[M*h+m] = act[M*h+m];
        else cumsum[M*h+m] = act[M*h+m] + cumsum[M*h+m-1];
    }
    for (int m=0; m<M; m++) {
        if (m==0 and spkthres[M*h+m] <= cumsum[M*h+m]) himax[h] = m;
        if (m!=0 and spkthres[M*h+m] <= cumsum[M*h+m] and spkthres[M*h+m] > cumsum[M*h+m-1]) himax[h] = m;
    }
    for (int m=0; m<M; m++) act[M*h+m] = 1.*(m==himax[h]);
}

void updact_cu(int H, int M, float *sup, float *act, int normfn, int actfn,
               float again, float maxfq, float timestep, float LOWESTFLT,
               float *spkthres, float *hfmax, float *hfsum, int *himax, int *nimax,
               float *cumsum) {
    int N = H * M;
    if (actfn == SPK or actfn == STCWTA) {
        if (not curandinitiated)
            curand_setup();
        /* Generate N floats on device */
        istat = curandGenerateUniform(gen_cu, spkthres, N);
        if (istat != CURAND_STATUS_SUCCESS)
            error("curand_setup", "CURAND generate failed", 7);
    }
    blockSize = 128;
    numBlocks = (N + blockSize - 1) / blockSize;

    switch(actfn) {
        case EXP:
            numBlocks = (N + blockSize - 1) / blockSize;
            expact_kernel <<<numBlocks, blockSize>>>(N, sup, act);
            break;
        case WTA:
            blockSize = 1;
            numBlocks = (N + blockSize - 1) / blockSize;
            CUDA_CHECK_ERROR(cudaMemset(act, 0, N * sizeof(float)));
            wtaact_kernel <<<numBlocks, blockSize>>>(H, M, sup, act, hfmax, nimax);
            break;
        case SOFTMAX:
            normact_cu(H, M, sup, act, normfn, again, hfmax, hfsum, LOWESTFLT);
            break;
        case SPK:
            normact_cu(H, M, sup, act, normfn, again, hfmax, hfsum, LOWESTFLT);
            spkact_kernel <<<numBlocks, blockSize>>>(H, M, timestep, sup, act, spkthres, maxfq);
            break;
        case STCWTA:
            if (cumsum == nullptr)
                cudaMallocManaged(&cumsum, N * sizeof(float));
            blockSize = 128;
            numBlocks = (H + blockSize - 1) / blockSize;
            stcwta_kernel<<<numBlocks, blockSize>>>(sup, himax, cumsum, again, act, spkthres, maxfq,
                                                    Globals::timestep, H, M);
            break;
        default:
            error("updact_cu", "No such actfn", 8);
    }
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    // if (axodelbuf != nullptr)
    //     axodelbuf->updstate();
    // for (size_t a = 0; a < axos.size(); a++)
    //     axos[a]->updstate();
}

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
