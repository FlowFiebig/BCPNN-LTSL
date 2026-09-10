/*****************************************************************

  Created: 2026-02-24

  Author: Anders Lansner

  Copyright (c) 2025 Anders Lansner

******************************************************************/

#ifndef __TDPop_cu_included
#define __TDPop_cu_included

#include "Globals.h"
#include "GPUGlobals.cuh"

__global__
void tdpop_ltsl_setinput_kernel(
    const float* __restrict__ input, // [M] (managed OK)
    float* __restrict__ s,           // [H*M], stage-major s[k*M + m]
    const float* __restrict__ a,     // [H]
    const float* __restrict__ b,     // [H]
    float* __restrict__ lgi,         // [H*M], lgi[h*M + m]
    int M, int H,
    float eps);

void setinput_cu(float *input, float *s, float *a, float *b, float *lgi, int M, int H, float eps );


#endif // __TDPop_cu_included
