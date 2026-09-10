/*****************************************************************

  Created: 2026-01-31

  Authors: Anders Lansner

******************************************************************/

#include <cmath>
#include <cstdlib>

#include "Globals.h"
#include "GPUGlobals.cuh"
#include "Prj1.h"
// #include "Prj1.cuh"

// upddenact_cuda.cu
#include <cuda_runtime.h>
#include <cstdio>

// Case 1: Iji == nullptr  -> replicate axoact[0..denNi-1] into each nj row.
__global__
void denact_replicate_kernel(float* __restrict__ denact,
                             const float* __restrict__ axoact,
                             int denNi, int Hj)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HN = denNi * Hj;
    if (idx < HN) {
        int dni = idx - (idx / denNi) * denNi;  // idx % denNi (fast)
        denact[idx] = axoact[dni];
    }
}

// Case 2: Iji != nullptr  -> gather.
__global__
void denact_gather_kernel(float* __restrict__ denact,
                          const float* __restrict__ axoact,
                          const int* __restrict__ Iji,
                          int denNi, int Hj)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HN = denNi * Hj;
    if (idx < HN) {
        int ni = Iji[idx];
        denact[idx] = axoact[ni];
    }
}

// Host wrapper.
// - If you have an axo object with axo->updstate(), call it before launching denact kernels.
// - stream is optional (0 is fine).
void Prj1::upddenact_cu(float* denact, float* axoact, const int* Iji, int axoNi, int denNi, int Hj,
                        cudaStream_t stream) {
    const int threads = 256; // good default for sm_75
    const int HN = denNi * Hj;
    const int blocks = (HN + threads - 1) / threads;

    if (Iji == nullptr) {
        denact_replicate_kernel<<<blocks, threads, 0, stream>>>(denact, axoact, denNi, Hj);
    } else {
        denact_gather_kernel<<<blocks, threads, 0, stream>>>(denact, axoact, Iji, denNi, Hj);
    }

    CUDA_CHECK_ERROR(cudaGetLastError());
    // If you need completion here:
    // CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__
void updzitrc_kernel(float* __restrict__ Zi, const float* __restrict__ denact, int HN, float fgain,
                     float eps, float tauzidt)
{
    int hidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hidx < HN) {
        float z  = Zi[hidx];
        float da = denact[hidx];
        z += (fgain * da * (1.0f - eps) + eps - z) * tauzidt;
        Zi[hidx] = z;
    }
}

void Prj1::updzitrc_cu(float* Zi, const float* denact, int Hj, int denNi, float fgain, float eps, float tauzidt,
                       cudaStream_t stream)
{
    const int HN = Hj * denNi;
    if (HN <= 0) return;

    // Prefetch UM to GPU to avoid page faults during kernel
    int dev = 0;
    CUDA_CHECK_ERROR(cudaGetDevice(&dev));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(Zi,     (size_t)HN * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)denact, (size_t)HN * sizeof(float), dev, stream));

    const int threads = 256;
    const int blocks  = (HN + threads - 1) / threads;

    updzitrc_kernel<<<blocks, threads, 0, stream>>>(Zi, denact, HN, fgain, eps, tauzidt);
    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
}

#ifdef OLDCODE
__global__
void upd_zj_pj_kernel(float* __restrict__ Zj, float* __restrict__ Pj, const float* __restrict__ trgact,
                      int Nj, float fgain, float eps, float tauzjdt, float prntaupdt) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj < Nj) {
        float z = Zj[nj];
        z += (fgain * trgact[nj] * (1.0f - eps) + eps - z) * tauzjdt;
        Zj[nj] = z;

        float p = Pj[nj];
        p += (z - p) * prntaupdt;
        Pj[nj] = p;
    }
}

__global__
void upd_pi_kernel(const float* __restrict__ Zi, float* __restrict__ Pi, int HN,
                   float prntaupdt) {
    int hidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hidx >= HN) return;

    float p = Pi[hidx];
    p += (Zi[hidx] - p) * prntaupdt;
    Pi[hidx] = p;
}

__global__
void upd_pji_kernel(const float* __restrict__ Zi, const float* __restrict__ Zj, float* __restrict__ Pji,
                    int Mj, int denNi, int Nj, float prntaupdt) {
    int nidx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Nj * denNi;
    if (idx < N) {
        int nj = idx / denNi;
        int nj = nidx / denNi, dni = nidx - nj * denNi, hj = nj / Mj,
            hidx = hj * denNi + dni;
        float p = Pji[idx];
        p += (Zi[hidx] * Zj[nj] - p) * prntaupdt;
        Pji[idx] = p;
    }
}

#endif // OLDCODE

__global__
void upd_Zj_Pj_kernel(float* __restrict__ Zj, float* __restrict__ Pj,
                      const float* __restrict__ trgact, int Nj, float fgain, float eps,
                      float tauzjdt, float prntaupdt) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj >= Nj) return;

    float zj = Zj[nj];
    zj += (fgain * trgact[nj] * (1.0f - eps) + eps - zj) * tauzjdt;
    Zj[nj] = zj;

    float pj = Pj[nj];
    pj += (zj - pj) * prntaupdt;
    Pj[nj] = pj;
}

__global__
void upd_Pi_kernel(float* __restrict__ Pi, const float* __restrict__ Zi, int HN,
                   float prntaupdt) {
    int hidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hidx >= HN) return;

    float pi = Pi[hidx];
    pi += (Zi[hidx] - pi) * prntaupdt;
    Pi[hidx] = pi;
}

__global__
void upd_Pji_kernel(float* __restrict__ Pji, const float* __restrict__ Zi,  // indexed by hidx
    const float* __restrict__ Zj,  // indexed by nj
    int Mj, int Nj, int denNi, float prntaupdt) {

    int nidx = blockIdx.x * blockDim.x + threadIdx.x;
    int NN = Nj * denNi;
    if (nidx >= NN) return;

    int nj  = nidx / denNi;
    int dni = nidx - nj * denNi;

    int hj   = nj / Mj;
    int hidx = hj * denNi + dni;

    float p = Pji[nidx];
    float target = Zi[hidx] * Zj[nj];
    p += (target - p) * prntaupdt;
    Pji[nidx] = p;
}

void Prj1::updtraces_cu(const float* trgact, float prn, bool frozen, int Hj, int Mj, int Nj, int denNi, float fgain, float eps,
                        float tauzjdt, float taupdt, const float* Zi, float* Zj, float* Pi, float* Pj, float* Pji,
                        cudaStream_t stream) {
    if (frozen) return;
    float prntaupdt = prn * taupdt;
    int N = Nj * denNi, HN = Hj * denNi;
    if (Nj <= 0 || denNi <= 0) return;

    int dev = 0;
    CUDA_CHECK_ERROR(cudaGetDevice(&dev));

    // Prefetch UM to GPU to avoid page faults
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)trgact, (size_t)Nj * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Zi,     (size_t)HN  * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(Zj,            (size_t)Nj * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(Pj,            (size_t)Nj * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(Pi,            (size_t)HN  * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(Pji,           (size_t)N  * sizeof(float), dev, stream));

    const int threads = 256;

    {
        int blocks = (Nj + threads - 1) / threads;
        upd_Zj_Pj_kernel<<<blocks, threads, 0, stream>>>(
            Zj, Pj, trgact, Nj, fgain, eps, tauzjdt, prntaupdt
        );
        CUDA_CHECK_ERROR(cudaGetLastError());
    }

    {
        int blocks = (HN + threads - 1) / threads;
        upd_Pi_kernel<<<blocks, threads, 0, stream>>>(
            Pi, Zi, HN, prntaupdt);
        CUDA_CHECK_ERROR(cudaGetLastError());
    }
    {
        int blocks = (N + threads - 1) / threads;
        upd_Pji_kernel<<<blocks, threads, 0, stream>>>(
            Pji, Zi, Zj, Mj, Nj, denNi, prntaupdt);
        CUDA_CHECK_ERROR(cudaGetLastError());
    }

    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

    return;

}


// Kernel: one thread per synapse (nidx). Also computes Bj once per nj using the
// first synapse thread (dni==0) for that nj.
__global__
void updbw_kernel(float* __restrict__ Bj, float* __restrict__ Wji, const float* __restrict__ Pj,
                  const float* __restrict__ Pi, const float* __restrict__ Pji, const int* __restrict__ Cji,
                  const int* __restrict__ Iji, int Hj, int Mj, int Nj, int denNi, float bgain, float wgain, float ewgain,
                  float iwgain, float nactNi_over_axoNi, int recurrent, int selfcisHDOFF) {
    int nidx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Nj * denNi;
    if (nidx >= N) return;

    int nj = nidx / denNi, dni = nidx - nj * denNi, hj = nj / Mj,
        hidx = hj * denNi + dni;

    // Bj[nj] computed once per nj (use dni==0 thread)
    if (nidx == nj * denNi) {
        float pj0 = Pj[nj];
        Bj[nj] = bgain * logf(pj0) * nactNi_over_axoNi;
    }

    float pj  = Pj[nj];
    float pi  = Pi[hidx];
    float pji = Pji[nidx];

    // (Cji==ACTIVE and not(all gains 0)) * log(pji/(pi*pj))
    const int gains_all_zero = (wgain == 0.0f && ewgain == 0.0f && iwgain == 0.0f);
    float w = 0.0f;
    if (Cji[hidx] == ACTIVE && !gains_all_zero) {
        w = logf(pji / (pi * pj));
    }

    // w *= wgain + (w>0)*ewgain + (w<0)*iwgain;
    w *= (wgain + (w > 0.0f) * ewgain + (w < 0.0f) * iwgain);

    // w *= not (Iji[nidx] == nj and recurrent and selfc == HDOFF);
    // i.e. if that condition holds => multiply by 0
    if (recurrent && selfcisHDOFF && Iji != nullptr) {
        if (Iji[hidx] == nj) w = 0.0f;
    }

    Wji[nidx] = w;
}

void Prj1::updbw_cu(float* Bj, float* Wji, const float* Pj, const float* Pi, const float* Pji, const int* Cji,
                    const int* Iji, int Hj, int Mj, int Nj, int denNi, float bgain, float wgain, float ewgain, float iwgain,
                    float nactNi, float axoNi, bool recurrent, int selfcisHDOFF, cudaStream_t stream) {
    int N = Nj * denNi, HN = Hj * denNi;
    if (N <= 0) return;

    float nactNi_over_axoNi = (axoNi != 0.0f) ? (nactNi / axoNi) : 0.0f;

    // Prefetch UM to GPU (recommended)
    int dev = 0;
    CUDA_CHECK_ERROR(cudaGetDevice(&dev));

    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(Bj,  (size_t)Nj * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(Wji, (size_t)N  * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Pj,  (size_t)Nj * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Pi,  (size_t)HN  * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Pji, (size_t)N  * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Cji, (size_t)HN  * sizeof(int),   dev, stream));

    if (recurrent && Iji) {
        CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Iji, (size_t)HN * sizeof(int), dev, stream));
    }

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    updbw_kernel<<<blocks, threads, 0, stream>>>(Bj, Wji, Pj, Pi, Pji, Cji, Iji, Hj, Mj, Nj, denNi, bgain, wgain,
                                                 ewgain, iwgain, nactNi_over_axoNi, recurrent, selfcisHDOFF);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
}

// Kernel 1: compute bwsupinf[nj] = Bj[nj] + sum(Zi*Wji over dni)
__global__
void bwsupinf_kernel(float* __restrict__ bwsupinf, const float* __restrict__ Bj, const float* __restrict__ Zi,
                                const float* __restrict__ Wji, int Hj, int Mj, int denNi) {
    // One block per nj
    int nj = blockIdx.x, hj = nj / Mj;

    // Parallel reduction over dni inside block
    float sum = 0.0f;
    for (int dni = threadIdx.x; dni < denNi; dni += blockDim.x) {
        int nidx = nj * denNi + dni, hidx = hj * denNi + dni;
        sum += Zi[hidx] * Wji[nidx];
    }

    // Reduce within block (shared memory)
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Standard power-of-two reduction (blockDim.x should be power of 2)
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        bwsupinf[nj] = Bj[nj] + sdata[0];
    }
}

// Kernel 2: bwsup[nj] += (bwsupinf[nj] - bwsup[nj]) * tauzidt
__global__
void bwsup_update_kernel(float* __restrict__ bwsup, const float* __restrict__ bwsupinf, int Nj, float tauzidt) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj < Nj) {
        float b = bwsup[nj];
        b += (bwsupinf[nj] - b) * tauzidt;
        bwsup[nj] = b;
    }
}

void Prj1::updbwsup_cu(float* bwsupinf, float* bwsup, const float* Bj, const float* Zi, const float* Wji,
                       int Hj, int Mj, int Nj, int denNi, float tauzidt, cudaStream_t stream) {
    if (Nj <= 0 || denNi <= 0) return;

    int dev = 0;
    CUDA_CHECK_ERROR(cudaGetDevice(&dev));

    // Prefetch UM to GPU (recommended)
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(bwsupinf, (size_t)Nj * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(bwsup,    (size_t)Nj * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Bj, (size_t)Nj * sizeof(float), dev, stream));

    size_t N = (size_t)Nj * (size_t)denNi, HM = (size_t)Hj * (size_t)denNi;
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Zi,  HM * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Wji, N * sizeof(float), dev, stream));

    // --- Kernel 1: compute bwsupinf ---
    // Choose a power-of-two block size. 256 is a good default for sm_75.
    int threads1 = 256;
    // If denNi is very small, reduce threads to avoid wasted shared memory
    if (denNi < threads1) {
        // round up to next power of two <= 256
        int t = 1;
        while (t * 2 <= denNi && t * 2 <= 256) t *= 2;
        threads1 = t;
    }
    int blocks1 = Nj; // one block per nj
    size_t shmem = (size_t)threads1 * sizeof(float);

    bwsupinf_kernel<<<blocks1, threads1, shmem, stream>>>(bwsupinf, Bj, Zi, Wji, Hj, Mj, denNi);
    CUDA_CHECK_ERROR(cudaGetLastError());

    // --- Kernel 2: update bwsup ---
    int threads2 = 256;
    int blocks2  = (Nj + threads2 - 1) / threads2;

    bwsup_update_kernel<<<blocks2, threads2, 0, stream>>>(bwsup, bwsupinf, Nj, tauzidt);
    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
}


// Kernel: trgpopbwsup[nj] += bwsup[nj]
__global__
void contribute_kernel(float* __restrict__ trgpopbwsup, const float* __restrict__ bwsup, int Nj) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj < Nj) {
        trgpopbwsup[nj] += bwsup[nj];
    }
}

// Unified Memory–friendly wrapper
void Prj1::contribute_cu(float* trgpopbwsup, const float* bwsup, int Nj, cudaStream_t stream) {
    if (Nj <= 0) return;

    int dev = 0;
    CUDA_CHECK_ERROR(cudaGetDevice(&dev));

    // Prefetch UM to GPU (recommended)
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(trgpopbwsup,
                                          (size_t)Nj * sizeof(float),
                                          dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)bwsup,
                                          (size_t)Nj * sizeof(float),
                                          dev, stream));

    const int threads = 256;
    const int blocks  = (Nj + threads - 1) / threads;

    contribute_kernel<<<blocks, threads, 0, stream>>>(trgpopbwsup, bwsup, Nj);
    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
}

// ---- device helpers ----

__device__ __forceinline__
double mi_term_d(double p, double q) {
    // mirrors: if (p > 0) p*log(p/q) else 0
    // If q<=0 we return 0 to avoid NaN/inf explosions (optional safety).
    if (p > 0.0 && q > 0.0) return p * log(p / q);
    return 0.0;
}

__device__ __forceinline__
double mutual_information_binary_d(double pi, double pj, double p11) {
    double p10 = pi - p11;
    double p01 = pj - p11;
    double p00 = 1.0 - pi - pj + p11;

    double MI = 0.0;
    MI += mi_term_d(p11, pi * pj);
    MI += mi_term_d(p10, pi * (1.0 - pj));
    MI += mi_term_d(p01, (1.0 - pi) * pj);
    MI += mi_term_d(p00, (1.0 - pi) * (1.0 - pj));
    return MI;
}

// ---- kernel ----
// One thread per nidx = nj*denNi + dni
__global__
void updMIsc_kernel(float* __restrict__ MIji, const float* __restrict__ Pi, const float* __restrict__ Pj,
                    const float* __restrict__ Pji,
                    const int* __restrict__ Iji, // may be nullptr if not recurrent/selfc
                    int Mj, int Nj, int denNi, int recurrent, int selfcisHDOFF)
{
    int nidx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Nj * denNi;
    if (nidx >= N) return;

    int nj = nidx / denNi, dni = nidx - nj * denNi, hj = nj / Mj,
        hidx = hj * denNi + dni;

    // Self-connection suppression: if recurrent && selfc==HDOFF && Iji[nidx]==nj => MI=0
    if (recurrent && selfcisHDOFF && Iji != nullptr) {
        if (Iji[hidx] == nj) {
            MIji[hidx] = 0.0f;
            return;
        }
    }

    double pi  = (double)Pi[nidx];
    double pj  = (double)Pj[nj];
    double p11 = (double)Pji[nidx];

    double MI = mutual_information_binary_d(pi, pj, p11);
    MIji[hidx] = (float)MI;
}

// ---- host wrapper (UM-friendly) ----
void Prj1::updMIsc_cu(float* MIji, const float* Pi, const float* Pj, const float* Pji,
                const int* Iji,   // can be nullptr if not used
                int Hj, int Mj, int Nj, int denNi, bool recurrent, int selfcisHDOFF,
                cudaStream_t stream) {
    int N = Nj * denNi;
    if (N <= 0) return;

    int HN = Hj * denNi;

    int dev = 0;
    CUDA_CHECK_ERROR(cudaGetDevice(&dev));

    // Prefetch UM to GPU (recommended)
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync(MIji, (size_t)HN  * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Pi,  (size_t)HN  * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Pji, (size_t)N  * sizeof(float), dev, stream));
    CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Pj,  (size_t)Nj * sizeof(float), dev, stream));
    if (recurrent && selfcisHDOFF && Iji) {
        CUDA_CHECK_ERROR(cudaMemPrefetchAsync((void*)Iji, (size_t)HN * sizeof(int), dev, stream));
    }

    // memset MIji to 0 (like your code)
    CUDA_CHECK_ERROR(cudaMemsetAsync(MIji, 0, (size_t)HN * sizeof(float), stream));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    updMIsc_kernel<<<blocks, threads, 0, stream>>>(MIji, Pi, Pj, Pji, Iji, Mj, Nj, denNi, recurrent, selfcisHDOFF);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
}

__global__
void nrmMIsc_kernel(const int* __restrict__ Iji, const float* __restrict__ MIji, const int* __restrict__ Ifanout,
                    float* __restrict__ nMIji, int HjdenNi)
{
    int hidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hidx >= HjdenNi) return;

    int ni = Iji[hidx];
    float denom = 1.0f + (float)Ifanout[ni];
    nMIji[hidx] = MIji[hidx] / denom;
}

void Prj1::nrmMIsc_cu(float* MIji, const int* Iji,  // can be nullptr if not used
                      int * Ifanout, float* nMIji, int HjdenNi, cudaStream_t stream) {
    int threads = 256;
    int blocks  = (HjdenNi + threads - 1) / threads;
    nrmMIsc_kernel<<<blocks, threads, 0, stream>>>(Iji, MIji, Ifanout, nMIji, HjdenNi);
}

__global__
void swapconns_kernel(int* __restrict__ Cji, const float* __restrict__ nMIji, bool* __restrict__ spmask,
                      int Hj, int denNi, float nswap, float swaprthr, unsigned long long seed,
                      int* __restrict__ nswapped /* global counter*/ ) {
    int hj = blockIdx.x * blockDim.x + threadIdx.x;
    if (hj >= Hj) return;

    // RNG state per nj-thread (only used for tie-break jitter)
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)hj, 0ULL, &rng);

    int hidxbase = hj * denNi;

    for (int swapid = 0; swapid < nswap; ++swapid) {

        int silmax_dni = -1, actmin_dni = -1;
        float silmax_score = -1.0e7f, actmin_score =  1.0e7f;

        // scan all dni for this hj
        for (int dni = 0; dni < denNi; ++dni) {
            int hidx = hidxbase + dni;

            if (!spmask[hidx]) continue;

            int c = Cji[hidx];
            float s = nMIji[hidx];

            if (c == ACTIVE) {
                // small randomness to avoid index-order bias
                float jitter = 1.0e-6f * curand_uniform(&rng); // (0,1]
                float v = s + jitter;

                if (v < actmin_score) {
                    actmin_score = v;
                    actmin_dni = dni;
                }
            } else if (c == SILENT) {
                if (s > silmax_score) {
                    silmax_score = s;
                    silmax_dni = dni;
                }
            }
        }

        // stop if no candidates or not beneficial
        if (silmax_dni == -1 || actmin_dni == -1) break;
        if (silmax_score < swaprthr * actmin_score) break;

        Cji[hidxbase + actmin_dni] = SILENT;
        Cji[hidxbase + silmax_dni] = ACTIVE;
        atomicAdd(nswapped, 1);

        // prevent picking same unit again
        spmask[hidxbase + actmin_dni] = false;
    }
}


void Prj1::swapconns_cu(bool *spmask, int *nswapped, cudaStream_t stream) {
    int threads = 256;
    int blocks  = (Hj + threads - 1) / threads;

    unsigned long long seed = 123456789ULL; // pick any seed you like

    swapconns_kernel<<<blocks, threads, 0, stream>>>(Cji, nMIji, spmask, Hj, denNi, nswap,
                                                     swaprthr, (unsigned long long)seed, nswapped);

    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream)); // if you need results immediately

}

__device__ __forceinline__
bool ISABSENT_dev(const int* __restrict__ Iji, int hidxbase, int denNi, int ni) {
    for (int dni = 0; dni < denNi; ++dni) {
        if (Iji[hidxbase + dni] == ni) {
            return false;
        }
    }
    return true;
}

__global__
void replconns_kernel(const int* __restrict__ Cji, const float* __restrict__ nMIji, bool* __restrict__ spmask,
                           int* __restrict__ Iji, int Hj, int denNi, int nrepl, int nsilNi, float replkthr,  int axoNi,
                           unsigned long long seed, int* __restrict__ nrepled) {
    int hj = blockIdx.x * blockDim.x + threadIdx.x;
    if (hj >= Hj) return;

    // printf("0: nrepled = %d\n", nrepled);

    // RNG per hj-thread (used like gnextint)
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)hj, 0ULL, &rng);

    int hidxbase = hj * denNi;

    for (int replid = 0; replid < nrepl; ++replid) {

        int silmin_dni = -1;
        float silscore_min =  1.0e7f;
        float silscore_max = -1.0e7f;  // kept for parity (not used later)
        float silscore_mean = 0.0f;
        float silscore_std  = 0.0f;

        // mean/min/max over SILENT and spmask==true
        for (int dni = 0; dni < denNi; ++dni) {
            int hidx = hidxbase + dni;

            if (spmask[hidx] && Cji[hidx] == SILENT) {
                float s = nMIji[hidx];
                silscore_mean += s;

                if (s < silscore_min) {
                    silscore_min = s;
                    silmin_dni = dni;
                }
                if (s > silscore_max) silscore_max = s;
            }
        }

        // match CPU: mean /= nsilNi
        if (nsilNi > 0) silscore_mean /= (float)nsilNi;
        if (silmin_dni == -1) continue;
        if (silscore_mean < 0.0f) silscore_mean = 0.0f;

        // std over all SILENT
        for (int dni = 0; dni < denNi; ++dni) {
            int hidx = hidxbase + dni;
            if (Cji[hidx] == SILENT) {
                float d = nMIji[hidx] - silscore_mean;
                silscore_std += d * d;
            }
        }

        if (nsilNi > 1) silscore_std = sqrtf(silscore_std / (float)(nsilNi - 1));
        else            silscore_std = 0.0f;

        // printf("silscore_min = %f silscore_mean = %f silscore_mean - replkthr * silscore_std = %f\n",
        //        silscore_min, silscore_mean, silscore_mean - replkthr * silscore_std);

        // stopping condition
        if (silscore_min > silscore_mean - replkthr * silscore_std) break;

        // pick random ni that is absent (up to 100 tries)
        int ni = 0;
        // int ntry = 0;
        // bool looping = false; // Not used now, but kept for possible future use

        do {
            ni = (int)(curand(&rng) % (unsigned)axoNi);
            // if (++ntry == 100) {
            //     looping = true;
            //     break;
            // }
        } while (!ISABSENT_dev(Iji, hidxbase, denNi, ni));

        // CPU assigns even if looping==true (kept identical)
        Iji[hidxbase + silmin_dni] = ni;

        atomicAdd(nrepled, 1);

        spmask[hidxbase + silmin_dni] = false;
    }

    // printf("nrepled = %d\n", nrepled);

}

void Prj1::replconns_cu(bool *spmask, int *nrepled, cudaStream_t stream) {
    // Presupposes that swapconns is run before replconns and that it cudaMemsetAsync(spmask)
    //    cudaMemsetAsync(spmask, 1, n * sizeof(bool), stream);

    int threads = 256;
    int blocks  = (Hj + threads - 1) / threads;
    unsigned long long seed = 123456789ULL;

    // NOTE: Swithed denNi and repl place in call
    replconns_kernel<<<blocks, threads, 0, stream>>>(Cji, nMIji, spmask, Iji, Hj, denNi, nrepl, nsilNi,
                                                          replkthr, axoNi, seed, nrepled);

    cudaGetLastError();            // or your CUDA_CHECK wrapper
    cudaStreamSynchronize(stream); // if you need result now

}
