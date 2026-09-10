/*****************************************************************

  CUDA kernel for TDPop::setinput — mirrors the CPU version in TDPop.cpp.

******************************************************************/

#include "TDPop.cuh"

__global__
void tdpop_ltsl_setinput_kernel(
    const float* __restrict__ input,
    float* __restrict__ s,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ lgi,
    int M, int H,
    float eps)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    // stage 0
    float s0 = s[0*M + m];
    s0 = a[0]*s0 + b[0]*input[m];
    s[0*M + m] = s0;
    float prev_y = s0;

    // stages 1..H-1
    for (int k = 1; k < H; ++k) {
        float sk = s[k*M + m];
        sk = a[k]*sk + b[k]*prev_y;
        s[k*M + m] = sk;
        prev_y = sk;
        int h = (H - 1 - k);
        lgi[h*M + m] = logf(sk + eps);
    }
    lgi[(H-1)*M + m] = logf(input[m] + eps);
}

void setinput_cu(float *input, float *s, float *a, float *b, float *lgi, int M, int H, float eps)
{
    int threads = 256;
    int blocks  = (M + threads - 1) / threads;
    tdpop_ltsl_setinput_kernel<<<blocks, threads>>>(input, s, a, b, lgi, M, H, eps);
    cudaDeviceSynchronize();
}
