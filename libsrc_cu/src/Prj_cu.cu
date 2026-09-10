/*****************************************************************

  Created: 2024-07-08  Modified: 2024-08-22

  Authors: Anders Lansner, Naresh Ravichandran

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
#include "Prj.h"
#include "Prj.cuh"

using namespace Globals;
using namespace GPUGlobals;

// ------------------------------------------------------------
// Static device helpers for swapconns
// ------------------------------------------------------------
static int* d_did_swap = nullptr;
static int* d_hi_old   = nullptr;
static int* d_hi_new   = nullptr;
static int* hi_edge_offsets;   // [axoHi + 1]
static int* hi_edge_list;      // [Hj * denHi]  // stores hidx = hj*denHi + dhi
static uint32_t swap_seed = 0x12345678u;
static bool swap_helpers_initialized = false;

static void init_swap_helpers_once()
{
    if (swap_helpers_initialized) return;

    CUDA_CHECK_ERROR(cudaMalloc(&d_did_swap, sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_hi_old,   sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_hi_new,   sizeof(int)));

    CUDA_CHECK_ERROR(cudaMemset(d_did_swap, 0, sizeof(int)));

    swap_helpers_initialized = true;
}


__global__
void upddenact_kernel(float *axoact, int *Hihjhi, int *Chjhi, int Hj, int denHi, int Mi, float *denact) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= Hj * denHi)
        return;
    int denNi = denHi * Mi;
    int hj = h / denHi;
    int dhi = h % denHi;
    int hi = Hihjhi[hj * denHi + dhi];
    for (int mi = 0; mi < Mi; mi++)
        denact[hj * denNi + dhi * Mi + mi] = axoact[hi * Mi + mi];
}

void upddenact_cu(float *axoact, int *Hihjhi, int *Chjhi, int Hj, int denHi, int Mi, float *denact) {
    blockSize = 128;
    numBlocks_hjdhi = (Hj * denHi + blockSize - 1) / blockSize;
    upddenact_kernel <<< numBlocks_hjdhi, blockSize>>>(axoact, Hihjhi, Chjhi, Hj, denHi, Mi, denact);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__global__
void updzitrc_kernel(float *denact, int Hj, int denNi, float fgain, float eps, float tauzidt, float *Zi) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Hj * denNi)
        return;
    int hj = n / denNi;
    int dni = n % denNi;
    int k = hj * denNi + dni;
    Zi[k] += (fgain * denact[k] * (1 - eps) + eps - Zi[k]) * tauzidt;
}


void updzitrc_cu(float *denact, int Hj, int denNi, float fgain, float eps, float tauzidt, float *Zi) {
    int numBlock = (Hj * denNi + blockSize - 1) / blockSize;
    updzitrc_kernel <<< numBlock, blockSize>>>(denact, Hj, denNi, fgain, eps, tauzidt, Zi);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}


__global__
void updzpjtrc_kernel(float *trgact, int Nj, float fgain, float eps, float tauzjdt, float prntaupdt,
                      float *Zj, float *Pj) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj >= Nj)
        return;
    Zj[nj] += (fgain * trgact[nj] * (1 - eps) + eps - Zj[nj]) * tauzjdt;
    Pj[nj] += (Zj[nj] - Pj[nj]) * prntaupdt;
}


__global__
void updpitrc_kernel(int Hj, int denNi, float eps, float prntaupdt,
                     float *Zi, float *Pi) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Hj * denNi)
        return;
    int hj = n / denNi;
    int ni = n % denNi;
    int k = hj * denNi + ni;
    Pi[k] += (Zi[k] - Pi[k]) * prntaupdt;
}


__global__
void updpjitrc_kernel(float *Zj, float *Zi,
                      int Nj, int Mj, int denNi,
                      float prntaupdt,
                      float *Pji) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denNi * Nj)
        return;
    int nj = n / denNi;
    int ni = n % denNi;
    int hj = nj / Mj;
    Pji[nj * denNi + ni] += (Zi[hj * denNi + ni] * Zj[nj] - Pji[nj * denNi + ni]) * prntaupdt;
}


void updtraces_cu(float *denact, float *trgact, float prn, bool frozen,
                  int Hj, int Nj, int Mj, int denNi,
                  float fgain, float eps, float tauzidt, float tauzjdt, float taupdt,
                  float *Zj, float *Zi, float *Pj, float *Pi, float *Pji) {
    float prntaupdt = prn * taupdt;
    blockSize = 128;
    numBlocksj = (Nj + blockSize - 1) / blockSize;
    numBlocksi = (Hj * denNi + blockSize - 1) / blockSize;
    numBlocksji = (Nj * denNi + blockSize - 1) / blockSize;
    // updating of Zi is done by updzitrc_cu
    if (not frozen) {
        updzpjtrc_kernel <<< numBlocksj, blockSize>>>(trgact, Nj, fgain, eps, tauzjdt, prntaupdt, Zj, Pj);
        updpitrc_kernel <<< numBlocksi, blockSize>>>(Hj, denNi, eps, prntaupdt, Zi, Pi);
        updpjitrc_kernel <<< numBlocksji, blockSize>>>(Zj, Zi, Nj, Mj, denNi, prntaupdt, Pji);
    }
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__device__
float debiased_pj(float pj, float eps, bool bdebias) {
    if (!bdebias)
        return pj;
    float p_hat = (pj - eps) / (1.0f - eps);
    return fmaxf(p_hat, eps * eps);
}

__global__
void BCPupdbwC_kernel(int Nj, int Mj, int denHi, int denNi, int Mi,
                     int *Chjhi,
                     float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
                     float eps, float bgain, float wgain, float ewgain, float iwgain,
                     bool bdebias) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denNi * Nj)
        return;
    int nj = n / denNi;
    int hj = nj/Mj;
    int dni = n % denNi;
    int dhi = dni/Mi;
    int k = hj * denNi + dni;
    if (dni==0) 
        Bj[nj] = bgain * logf(debiased_pj(Pj[nj], eps, bdebias));
    float wji;
    wji = (Chjhi[hj*denHi + dhi] == ACTIVE) * logf(Pji[nj * denNi + dni] / (Pi[k] * Pj[nj]));
    wji *= wgain + (wji > 0) * ewgain + (wji < 0) * iwgain;
    Wji[nj * denNi + dni] = wji;
}

__global__
void BCPupdbw_kernel(int Nj, int Mj, int denHi, int denNi, int Mi,
                     float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
                     float eps, float bgain, float wgain, float ewgain, float iwgain,
                     bool bdebias) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denNi * Nj)
        return;
    int nj = n / denNi;
    int hj = nj/Mj;
    int dni = n % denNi;
    int k = hj * denNi + dni;
    if (dni==0) 
        Bj[nj] = bgain * logf(debiased_pj(Pj[nj], eps, bdebias));
    float wji;
    wji = logf(Pji[nj * denNi + dni] / (Pi[k] * Pj[nj]));
    wji *= wgain + (wji > 0) * ewgain + (wji < 0) * iwgain;
    Wji[nj * denNi + dni] = wji;
}

__global__
void WILLupdbw_kernel(int Nj, int denNi,
                      float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
                      float eps, float bgain, float wgain) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denNi * Nj)
        return;
    int nj = n / denNi;
    int dni = n % denNi;
    Bj[nj] = 0;
    Wji[nj * denNi + dni] = wgain * Pji[nj * denNi + dni] > wgain * eps * eps;
}


__global__
void HEBBupdbw_kernel(int Nj, int denNi,
                      float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
                      float eps, float bgain, float wgain) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denNi * Nj)
        return;
    int nj = n / denNi;
    int dni = n % denNi;
    Bj[nj] = 0;
    Wji[nj * denNi + dni] = wgain * Pji[nj * denNi + dni];
}


__global__
void COVupdbw_kernel(int Nj, int Mj, int denNi,
                     float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
                     float eps, float bgain, float wgain) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denNi * Nj)
        return;
    int nj = n / denNi;
    int hj = nj/Mj;
    int dni = n % denNi;
    int k = hj*denNi + dni;
    Bj[nj] = 0;
    Wji[nj * denNi + dni] = wgain * (Pji[nj * denNi + dni] - Pi[k] * Pj[nj]);
}


void updbw_cu(int lrule, int Nj, int Mj, int denHi, int denNi, int Mi,
              int *Chjhi,
              float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
              float eps, float bgain, float wgain, float ewgain, float iwgain,
              bool bdebias) {
    blockSize = 128;
    numBlocksji = (Nj * denNi + blockSize - 1) / blockSize;
    switch(lrule) {
        case BCP:
            if (Chjhi == nullptr)
                BCPupdbw_kernel<<< numBlocksji, blockSize>>>(Nj, Mj, denHi, denNi, Mi,
                                                             Pj, Pi, Pji, Bj, Wji,
                                                             eps, bgain, wgain, ewgain, iwgain,
                                                             bdebias);
            else
                BCPupdbwC_kernel<<< numBlocksji, blockSize>>>(Nj, Mj, denHi, denNi, Mi,
                                                              Chjhi,
                                                              Pj, Pi, Pji, Bj, Wji,
                                                              eps, bgain, wgain, ewgain, iwgain,
                                                              bdebias);
            break;
        case WILL:
            WILLupdbw_kernel <<< numBlocksji, blockSize>>>(Nj, denNi,
                    Pj, Pi, Pji, Bj, Wji,
                    eps, bgain, wgain);
            break;
        case HEBB:
            HEBBupdbw_kernel <<< numBlocksji, blockSize>>>(Nj, denNi,
                    Pj, Pi, Pji, Bj, Wji,
                    eps, bgain, wgain);
            break;
        case COV:
            COVupdbw_kernel <<< numBlocksji, blockSize>>>(Nj, Mj, denNi,
                    Pj, Pi, Pji, Bj, Wji,
                    eps, bgain, wgain);
            break;
    }
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}


__global__
void updbwsup_kernel(float *Wji, int Nj, int denNi, float tauzidt,
                     float *bwsupinf, float *bwsup) {
    int nj = blockIdx.x * blockDim.x + threadIdx.x;
    if (nj >= Nj)
        return;
    bwsup[nj] += (bwsupinf[nj] - bwsup[nj]) * tauzidt;
}

void updbwsup_cu(float *Zi, float *Bj, float *Wji, int Hj, int Mj, int denNi, float tauzidt,
                 float *bwsupinf, float *bwsup) {
    if (not cublasinitiated)
        cublas_setup();
    int Nj = Hj * Mj;
    
    float alpha = 1, beta = 0;
    for (int hj = 0; hj < Hj; hj++) {
        CUBLAS_CHECK_ERROR(cublasSgemv(cubhandle, CUBLAS_OP_T, denNi, Mj, &alpha, &Wji[hj * Mj * denNi],
                                       denNi,
                                       &Zi[hj * denNi],
                                       1, &beta, &bwsupinf[hj * Mj], 1));
        // nbrav
        CUDA_CHECK_ERROR(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }
    blockSize = 256;
    numBlocksj = (Nj + blockSize - 1) / blockSize;
    updbwsup_kernel <<< numBlocksj, blockSize>>>(Wji, Nj, denNi, tauzidt, bwsupinf, bwsup);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();

}

__global__
void contribute_kernel(float *bwsup, float *trgpopbwsup, int Nj) {
    const uint nj = threadIdx.x + blockIdx.x * blockDim.x;
    if (nj >= Nj)
        return;
    // atomicAdd(&(trgpopbwsup[nj]), bwsup[nj]); // nbrav: fails on CUDA/Dardel
    trgpopbwsup[nj] += bwsup[nj]; // nbrav
}

void contribute_cu(float *bwsup, float *trgpopbwsup, int Nj) {
    numBlocks_nj = (Nj + blockSize - 1) / blockSize;
    contribute_kernel <<< numBlocks_nj, blockSize>>>(bwsup, trgpopbwsup, Nj);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}


__global__
void updMIsc_kernel1(float *Pj, float *Pi, float *Pji, float eps, int Nj, int Mj, int denHi, int denNi,
                    int Mi, float *MIhjhi) {
    const uint njdni = threadIdx.x + blockIdx.x * blockDim.x;
    if (njdni >= Nj * denNi)
        return;
    int nj = njdni / denNi;
    int ni = njdni % denNi;
    int hj = nj / Mj;
    int hi = ni / Mi;
    float pj = Pj[nj];
    float pi = Pi[hj * denNi + ni];
    float pji = Pji[nj * denNi + ni];
    atomicAdd(&(MIhjhi[hj * denHi + hi]), pji * log(pji / (pi * pj)));
}

__global__
void updMIsc_kernel2(float *Pj, float *Pi, float *Pji, float eps, int Hj, int Mj, int denHi, int Mi, float *MIhjhi) {
    const uint hjdhi = threadIdx.x + blockIdx.x * blockDim.x;
    if (hjdhi >= Hj * denHi)
        return;
    int denNi = denHi * Mi;
    int hj = hjdhi / denHi;
    int hi = hjdhi % denHi;
    for (int ni=hi*Mi; ni<(hi+1)*Mi; ni++) {
        for (int nj=hj*Mj; nj<(hj+1)*Mj; nj++) {
            float pj = Pj[nj];
            float pi = Pi[hj * denNi + ni];
            float pji = Pji[nj * denNi + ni];
            MIhjhi[hj * denHi + hi] += pji * log(pji / (pi * pj));
        }
    }
}

void updMIsc_cu(float *Pj, float *Pi, float *Pji, float eps, int *Hihjhi, int *Hifanout, int Hj, int Nj, int Mj,
                int denHi, int denNi, int Mi, float *MIhjhi) {
    // numBlocks_njdni = (Nj * denNi + blockSize - 1) / blockSize;
    // updMIsc_kernel1 <<< numBlocks_njdni, blockSize >>> (Pj, Pi, Pji, eps, Nj, Mj, denHi, denNi, Mi, MIhjhi);
    int numBlocks_hjdhi = (Hj * denHi + blockSize - 1) / blockSize;
    updMIsc_kernel2 <<< numBlocks_hjdhi, blockSize >>> (Pj, Pi, Pji, eps, Hj, Mj, denHi, Mi, MIhjhi);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

__global__
void nrmMIsc_kernel(int *Hihjhi, int *Hifanout, int Hj, int denHi, float *MIhjhi, float *nMIhjhi) {
    const uint hjdhi = threadIdx.x + blockIdx.x * blockDim.x;
    if (hjdhi >= Hj * denHi)
        return;
    int hj = hjdhi / denHi;
    int dhi = hjdhi % denHi;
    int hi = Hihjhi[hj * denHi + dhi];
    nMIhjhi[hj * denHi + dhi] = MIhjhi[hj * denHi + dhi] / (1 + Hifanout[hi]);
}

void nrmMIsc_cu(int *Hihjhi, int *Hifanout, 
                int Hj, int Nj, int denHi, int denNi, 
                float *MIhjhi, float *nMIhjhi) {

    numBlocks_hjdhi = (Hj * denHi + blockSize - 1) / blockSize;
    nrmMIsc_kernel <<< numBlocks_hjdhi, blockSize >>> (Hihjhi, Hifanout, Hj, denHi, MIhjhi, nMIhjhi);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();    
}

__global__
void ISABSENT_kernel(int *Hihjhi, int hj, int hi, int Hj, int denHi, int *dhi) {
    const uint n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= denHi)
        return;
    if (Hihjhi[hj * denHi + n] == hi)
        *dhi = 0;
    *dhi = 1;
}


__device__ __forceinline__
void warp_min(float& val, int& idx)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        float v = __shfl_down_sync(0xffffffff, val, offset);
        int   i = __shfl_down_sync(0xffffffff, idx, offset);
        if (v < val) { val = v; idx = i; }
    }
}

__device__ __forceinline__
void warp_max(float& val, int& idx)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        float v = __shfl_down_sync(0xffffffff, val, offset);
        int   i = __shfl_down_sync(0xffffffff, idx, offset);
        if (v > val) { val = v; idx = i; }
    }
}

__device__ __forceinline__ float hash01(uint32_t x)
{
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (x & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__global__ void nrmMIsc_local_kernel(
    const float* __restrict__ MIhjhi,
    float*       __restrict__ nMIhjhi,
    const int*   __restrict__ hi_edge_offsets,
    const int*   __restrict__ hi_edge_list,
    const int*   __restrict__ Hifanout,
    int hi
){
    int start = hi_edge_offsets[hi];
    int end   = hi_edge_offsets[hi + 1];

    int p = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= end) return;

    int hidx = hi_edge_list[p];
    int fo = Hifanout[hi];
    nMIhjhi[hidx] = MIhjhi[hidx] / (1.0f + (float)fo);
}

__global__ void updhifanout_kernel(
    const int* __restrict__ Chjhi,
    const int* __restrict__ Hihjhi,
    int* __restrict__ Hifanout,
    int Hj, int denHi
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int E = Hj * denHi;
    if (idx >= E) return;

    if (Chjhi[idx] == ACTIVE) {
        int hi = Hihjhi[idx];
        atomicAdd(&Hifanout[hi], 1);
    }
}

static void updhifanout_cu(int* Chjhi, int* Hihjhi, int* Hifanout, int Hj, int denHi, int axoHi)
{
    CUDA_CHECK_ERROR(cudaMemset(Hifanout, 0, axoHi * sizeof(int)));

    int threads = 256;
    int E = Hj * denHi;
    int blocks = (E + threads - 1) / threads;

    updhifanout_kernel<<<blocks, threads>>>(Chjhi, Hihjhi, Hifanout, Hj, denHi);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

static void build_hi_edge_lists_once(int* d_Hihjhi, int Hj, int denHi, int axoHi)
{
    // Allocate device buffers once
    if (hi_edge_offsets && hi_edge_list) return;

    const int E = Hj * denHi;

    // Pull Hihjhi to host
    std::vector<int> h_Hihjhi(E);
    CUDA_CHECK_ERROR(cudaMemcpy(h_Hihjhi.data(), d_Hihjhi, E * sizeof(int), cudaMemcpyDeviceToHost));

    // Count edges per hi
    std::vector<int> counts(axoHi, 0);
    for (int e = 0; e < E; ++e) {
        int hi = h_Hihjhi[e];
        if ((unsigned)hi < (unsigned)axoHi) counts[hi]++;
    }

    // Prefix sum -> offsets
    std::vector<int> offsets(axoHi + 1, 0);
    for (int hi = 0; hi < axoHi; ++hi) offsets[hi + 1] = offsets[hi] + counts[hi];

    // Fill edge list (stable)
    std::vector<int> cursor = offsets;
    std::vector<int> edges(E);
    for (int e = 0; e < E; ++e) {
        int hi = h_Hihjhi[e];
        int p = cursor[hi]++;
        edges[p] = e; // hidx = hj*denHi + dhi
    }

    // Allocate + upload to device
    CUDA_CHECK_ERROR(cudaMalloc(&hi_edge_offsets, (axoHi + 1) * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&hi_edge_list,   E * sizeof(int)));

    CUDA_CHECK_ERROR(cudaMemcpy(hi_edge_offsets, offsets.data(), (axoHi + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(hi_edge_list,    edges.data(),    E * sizeof(int),          cudaMemcpyHostToDevice));
}

void Prj::renorm_hi(int hi)
{
    int start, end;
    cudaMemcpy(&start, hi_edge_offsets + hi,     sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&end,   hi_edge_offsets + hi + 1, sizeof(int), cudaMemcpyDeviceToHost);

    int n = end - start;
    if (n <= 0) return;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    nrmMIsc_local_kernel<<<blocks, threads>>>(
        MIhjhi,
        nMIhjhi,
        hi_edge_offsets,
        hi_edge_list,
        Hifanout,
        hi
    );
}

__global__ void pick_and_swap_hj_kernel(
    int*     __restrict__ Chjhi,     // <-- FIXED
    bool*    __restrict__ spmask,
    const int* __restrict__ Hihjhi,
    int*     __restrict__ Hifanout,
    const float* __restrict__ nMIhjhi,
    int hj,
    int denHi,
    int swapid,
    float swaprthr,
    uint32_t seed,
    int* __restrict__ did_swap,
    int* __restrict__ hi_old_out,
    int* __restrict__ hi_new_out
){
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float actmin = 1.0e30f;
    int   actidx = -1;
    float silmax = -1.0e30f;
    int   silidx = -1;

    for (int dhi = tid; dhi < denHi; dhi += blockDim.x) {
        int hidx = hj * denHi + dhi;
        int st   = Chjhi[hidx];
        float sc = nMIhjhi[hidx];

        if (st == ACTIVE) {
            uint32_t h = seed ^ (hj*0x9E3779B9u) ^ (swapid*0x85EBCA6Bu) ^ dhi;
            float v = sc + 1.0e-6f * ((h & 0x00FFFFFFu) * (1.0f / 16777216.0f));
            if (v < actmin) { actmin = v; actidx = dhi; }
        }
        else if (st == SILENT && spmask[hidx]) {
            if (sc > silmax) { silmax = sc; silidx = dhi; }
        }
    }

    // warp reductions
    for (int o = 16; o > 0; o >>= 1) {
        float v = __shfl_down_sync(0xffffffff, actmin, o);
        int   i = __shfl_down_sync(0xffffffff, actidx, o);
        if (v < actmin) { actmin = v; actidx = i; }

        v = __shfl_down_sync(0xffffffff, silmax, o);
        i = __shfl_down_sync(0xffffffff, silidx, o);
        if (v > silmax) { silmax = v; silidx = i; }
    }

    __shared__ float s_actmin[8], s_silmax[8];
    __shared__ int   s_actidx[8], s_silidx[8];

    if (lane == 0) {
        s_actmin[warp] = actmin;
        s_actidx[warp] = actidx;
        s_silmax[warp] = silmax;
        s_silidx[warp] = silidx;
    }
    __syncthreads();

    if (warp == 0 && lane < (blockDim.x >> 5)) {
        actmin = s_actmin[lane];
        actidx = s_actidx[lane];
        silmax = s_silmax[lane];
        silidx = s_silidx[lane];

        for (int o = 4; o > 0; o >>= 1) {
            float v = __shfl_down_sync(0xffffffff, actmin, o);
            int   i = __shfl_down_sync(0xffffffff, actidx, o);
            if (v < actmin) { actmin = v; actidx = i; }

            v = __shfl_down_sync(0xffffffff, silmax, o);
            i = __shfl_down_sync(0xffffffff, silidx, o);
            if (v > silmax) { silmax = v; silidx = i; }
        }

        if (lane == 0) {
            if (actidx < 0 || silidx < 0 || silmax < swaprthr * actmin) {
                *did_swap = 0;
                return;
            }

            int aidx = hj * denHi + actidx;
            int sidx = hj * denHi + silidx;

            int hi_old = Hihjhi[aidx];
            int hi_new = Hihjhi[sidx];

            Chjhi[aidx] = SILENT;
            Chjhi[sidx] = ACTIVE;

            spmask[aidx] = false;
            spmask[sidx] = false;

            atomicSub(&Hifanout[hi_old], 1);
            atomicAdd(&Hifanout[hi_new], 1);

            *hi_old_out = hi_old;
            *hi_new_out = hi_new;
            *did_swap   = 1;                
        }
    }
}

int Prj::swapconns_cu()
{
    if (frozen) return 0;

    // struct timeval swapconns_time;
    // gettimeofday(&swapconns_time, 0);

    updMIsc();
    updhifanout();
    nrmMIsc();

    init_swap_helpers_once();

    // 0) Build MI and normalized MI (like CPU swapconns does)
    CUDA_CHECK_ERROR(cudaMemset(MIhjhi, 0, Hj * denHi * sizeof(float)));
    updMIsc_cu(Pj, Pi, Pji, eps, Hihjhi, Hifanout, Hj, Nj, Mj, denHi, denNi, Mi, MIhjhi);

    updhifanout_cu(Chjhi, Hihjhi, Hifanout, Hj, denHi, axoHi);

    nrmMIsc_cu(Hihjhi, Hifanout, Hj, Nj, denHi, denNi, MIhjhi, nMIhjhi);

    // 1) Build hi-edge adjacency lists once (needed for renorm_hi)
    build_hi_edge_lists_once(Hihjhi, Hj, denHi, axoHi);

    int total_swaps = 0;

    for (int hj = 0; hj < Hj; ++hj)
    {
        cudaMemset(spmask + hj*denHi, 1, denHi*sizeof(bool));

        for (int swapid = 0; swapid < nswap; ++swapid)
        {
            cudaMemset(d_did_swap, 0, sizeof(int));

            pick_and_swap_hj_kernel<<<1,256>>>(
                Chjhi, spmask, Hihjhi, Hifanout,
                nMIhjhi,
                hj, denHi,
                swapid,
                swaprthr,
                swap_seed,
                d_did_swap,
                d_hi_old,
                d_hi_new
            );

            int did;
            cudaMemcpy(&did, d_did_swap, sizeof(int), cudaMemcpyDeviceToHost);
            if (!did) break;

            int hi_old, hi_new;
            cudaMemcpy(&hi_old, d_hi_old, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&hi_new, d_hi_new, sizeof(int), cudaMemcpyDeviceToHost);

            renorm_hi(hi_old);
            renorm_hi(hi_new);

            total_swaps++;
            swap_seed = swap_seed * 1664525u + 1013904223u;
        }
    }

    needsupdbw = true;

    // float timelapsed = getDiffTime(swapconns_time) / 1000;
    // printf("swapconns_time (GPU) = %.3f sec\n", timelapsed);

    return total_swaps;
}
