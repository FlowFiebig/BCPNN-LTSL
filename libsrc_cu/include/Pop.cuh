/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-07-08     Modified: 2024-08-05

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

#ifndef __Pop_cu_included
#define __Pop_cu_included

#include "Globals.h"
#include "GPUGlobals.cuh"

// __global__ void updsup_kernel(int N, float *lgi, float *bwsup, float *sup, float *supinf, float *act,
//                               float *ada, float *sada, uint *pnoise,
//                               float taumdt, float bwgain, float adgain, float tauadt,
//                               float sadgain, float tausadt, float nampl, float nfreq);

void updsup_cu(int N, float *lgi, float *bwsup, float *sup, float *supinf, float *act,
               float *ada, float *sada, uint *pnoise,
               float taumdt, float igain, float bwgain, float adgain, float tauadt,
               float sadgain, float tausadt, float nampl, float nfreq);

// __global__ void fullnorm_kernel(int H, int M, float *sup, float *act, float again);

// __global__ void halfnorm_kernel(int H, int M, float *sup, float *act, float again, float lowest);

void normact_cu(int H, int M, float *sup, float *act, int normfn, float again,
                float *hmax, float *hsum, float LOWESTFLT);

// __global__ void expact_kernel(int N, float *sup, float *act);

// __global__ void wtaact_kernel(int H, int M, float *sup, float *act, float *hmax, uint *maxn);

// __global__ void spkact_kernel(int H, int M, float timestep, float *sup, float *act, float *spkthres, float maxfq,
//                               float *hmax, uint *maxn);

void updact_cu(int H, int M, float *sup, float *act, int normfn, int actfn,
               float again, float maxfq, float timestep, float LOWESTFLT,
               float *spkthres, float *hfmax, float *hfsum, int *himax, int *nimax,
               float *cumsum);

void setinput_cu(float *input, float *s, float *a, float *b, float *lgi, int M, int H, float eps );


#endif // __Pop_cu_included
