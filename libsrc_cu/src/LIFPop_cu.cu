/*****************************************************************

  Created: 2024-08-12  Modified: 2024-08-12

  Authors: Anders Lansner, Naresh Ravchandran

  Copyright (c) 2024 Anders Lansner, Naresh Ravichandran

******************************************************************/

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#include "../include/Globals.h"
#include "../include/GPUGlobals.cuh"
#include "../include/LIFPop.cuh"

using namespace std;
using namespace Globals;
using namespace GPUGlobals;

__global__
void updada_kernel_ALIF(int N, float *ada, float *act, float adgain, float tauadt) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    ada[n] += (adgain * act[n] - ada[n]) * tauadt;
}

__global__
void updada_kernel_ADEX(int N, float *ada, float *sup, float adgain, float tauadt, float EL) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    ada[n] += (adgain * (sup[n] - EL) - ada[n]) * tauadt;
}

__global__
void updsada_kernel_ALIF(int N, float *sada, float *act, float sadgain, float tausadt) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    sada[n] += (sadgain * act[n] - sada[n]) * tausadt;
}

__global__
void updsada_kernel_ADEX(int N, float *sada, float *sup, float sadgain, float tausadt, float EL) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    sada[n] += (sadgain * (sup[n] - EL) - sada[n]) * tausadt;
}

void updada_cu(int N, Globals::LIF_T liftype, float *ada, float *sup, float *act, float adgain, float tauadt,
               float EL) {
    blockSize = 128;
    numBlocks = (N + blockSize - 1) / blockSize;
    switch (liftype) {
    case ADEXS:
    case ALIF:
        updada_kernel_ALIF<<<numBlocks, blockSize>>>(N, ada, act, adgain, tauadt);
        break;
    case ADEX:
        updada_kernel_ADEX<<<numBlocks, blockSize>>>(N, ada, sup, adgain, tauadt, EL);
        break;
    default:
        error("updada_cu", "No such LIF type");
    }
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

void updsada_cu(int N, Globals::LIF_T liftype, float *sada, float *sup, float *act, float sadgain, float tausadt,
                float EL) {
    blockSize = 128;
    numBlocks = (N + blockSize - 1) / blockSize;
    switch (liftype) {
    case ADEXS:
    case ALIF:
        updsada_kernel_ALIF<<<numBlocks, blockSize>>>(N, sada, act, sadgain, tausadt);
        break;
    case ADEX:
        updsada_kernel_ADEX<<<numBlocks, blockSize>>>(N, sada, sup, sadgain, tausadt, EL);
        break;
    default:
        error("updsada_cu", "No such LIF type");
    }
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}


__global__
void updsup_kernelADEX(int N, LIF_T liftype, float *input, float *supinf, float *sup, float *bwsup, float *ada,
                       float *sada, uint *pnoise, float taum, float taumdt, float igain, float bwgain, float nampl,
                       float nfreq, float DT, float VT, float C, float gL, float EL, float timestep, float *tmp) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    if (liftype == ADEX or liftype == ADEXS) {
        *tmp = DT * exp((sup[n] - VT)/DT);
        if (*tmp > 1e-10)
            *tmp = C / timestep * 0.1;
    }
}

__global__
void updsup_kernel(int N, LIF_T liftype, float *input, float *supinf, float *sup, float *bwsup, float *ada,
                   float *sada, uint *pnoise, float taum, float taumdt, float igain, float bwgain, float nampl,
                   float nfreq, float DT, float VT, float C, float gL, float EL, float timestep, float tmp) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    supinf[n] = bwgain * bwsup[n];
    if (ada != nullptr)
        supinf[n] += ada[n];
    if (sada != nullptr)
        supinf[n] += sada[n];
    supinf[n] += (nampl * (pnoise[n] - nfreq)) / gL;
    supinf[n] += (input[n] / gL + EL + tmp);
    if (supinf[n] > 0.020)
        supinf[n] = 0.020;  // Na reversal potential
    else if (supinf[n] < -0.080)
        supinf[n] = -0.080; // Cl reversal potential
    sup[n] += (supinf[n] - sup[n]) * taumdt;
    
}

void updsup_cu(int N, Globals::LIF_T liftype, float *input, float *supinf, float *sup, float *bwsup, float *ada,
               float *sada, uint *pnoise, float taum, float taumdt, float igain, float bwgain,
               float nampl, float nfreq, float DT, float VT, float C, float gL, float EL, float timestep) {
    if (nampl > 0) {
        /* Generate N floats on device */
        if (not curandinitiated)
            curand_setup();
        istat = curandGeneratePoisson(gen_cu, pnoise, N, nfreq);
        if (istat != CURAND_STATUS_SUCCESS)
            error("curand_generate", "CURAND generate failed (" + to_string(istat) + ")", 4);
    }
    blockSize = 128;
    numBlocks = (N + blockSize - 1) / blockSize;
    float tmp = 0;
    if (liftype == ADEX or liftype == ADEXS) 
        updsup_kernelADEX<<<numBlocks, blockSize>>>(N, liftype, input, supinf, sup, bwsup,
                                                    ada, sada, pnoise, taum, taumdt, igain,
                                                    bwgain, nampl, nfreq, DT, VT,
                                                    C, gL, EL, timestep, &tmp);
    updsup_kernel<<<numBlocks, blockSize>>>(N, liftype, input, supinf, sup, bwsup,
                                            ada, sada, pnoise, taum, taumdt, igain,
                                            bwgain, nampl, nfreq, DT, VT,
                                            C, gL, EL, timestep, tmp);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__global__
void updact_kernel(int N, float *sup, float *act, int *spkstep, int ireft, int ispkwid, float VR, float VT) {
                   int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    if (spkstep[n]>0) {
        spkstep[n]--;
        if (spkstep[n]==0) {
            sup[n] = VR;
            act[n] = 0;
            spkstep[n] = -ireft;
        }
    } else if (spkstep[n]<0)
        spkstep[n]++;
    else if (act[n] == 0 and VT<=sup[n]) {
            sup[n] += 0.050; // Top of spike
            if (sup[n] > 0.02)
                sup[n] = 0.02;  // Na reversal potential
            act[n] = 1;
            spkstep[n] = ispkwid;
    // else if (VT<=sup[n]) {
        // sup[n] = 0.010; // Top of spike
        // act[n] = 1;
        // spkstep[n] = ispkwid;
    }    
}

void updact_cu(int N, float *sup, float *act, int *spkstep, int ireft, int ispkwid, float VR, float VT) {
    blockSize = 128;
    numBlocks = (N + blockSize - 1) / blockSize;
    updact_kernel <<<numBlocks, blockSize>>>(N, sup, act, spkstep, ireft, ispkwid, VR, VT);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}
