/*****************************************************************

  Created: 2026-02-19

  Authors: Anders Lansner

  Copyright (c) 2026 Anders Lansner

******************************************************************/

#include <algorithm>
#include <string>
#include <cctype>

#include "Globals.h"
#include "Pop.h"
#include "TDPop.h"
#include "TDPop.cuh"

#define CUDATDSETINPUT

using namespace std;
using namespace Globals;

static inline void CUDA_CHECK(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}

TDPop::TDPop(int K, int My, std::string name, float fs_, float taumin_, float taumax_, int K_mode_)
: Pop(K, My), fs(fs_), tau0(taumin_), taumax(taumax_), c(0.0f), K_mode(K_mode_)
{
    if (fs <= 0.0f)  error("TDPop::TDPop", "fs <= 0");
    if (H <= 0)      error("TDPop::TDPop", "H <= 0");

    // Managed alloc
    CUDA_CHECK(cudaMallocManaged(&a, H * sizeof(float)), "cudaMallocManaged(a)");
    CUDA_CHECK(cudaMallocManaged(&b, H * sizeof(float)), "cudaMallocManaged(b)");
    CUDA_CHECK(cudaMallocManaged(&s, (size_t)H * M * sizeof(float)), "cudaMallocManaged(s)");

    CUDA_CHECK(cudaMallocManaged(&d_input, M * sizeof(float)), "cudaMallocManaged(d_input)");

    const float dt = 1.0f / fs;
    if (K_mode == 0) {
        // Logarithmic (geometric) spacing: tau_k = tau0 * c^k
        c = powf(taumax / tau0, 1.0f / (H - 1.0f));
        float tau_k = tau0;
        for (int k = 0; k < H; ++k) {
            a[k] = expf(-dt / tau_k);
            b[k] = 1.0f - a[k];
            tau_k *= c;
        }
    } else {
        // Linear spacing: tau_k = tau0 + k * dtau
        float dtau = (taumax - tau0) / (H - 1.0f);
        float tau_k = tau0;
        for (int k = 0; k < H; ++k) {
            a[k] = expf(-dt / tau_k);
            b[k] = 1.0f - a[k];
            tau_k += dtau;
        }
    }

    // Init state
    for (int i = 0; i < H * M; ++i) s[i] = 0.0f;

    // Make sure managed writes are visible before first kernel use
    CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize after init");

    setactfn("SOFTMAX");
}

TDPop::~TDPop() {
    if (s) cudaFree(s);
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    s = a = b = nullptr;
}

ulong TDPop::nallocbyte() {
    ulong nbyte = Pop::nallocbyte();
    nbyte += (ulong)H * sizeof(float);           // a
    nbyte += (ulong)H * sizeof(float);           // b
    nbyte += (ulong)H * (ulong)M * sizeof(float);// s
    return nbyte;
}

void TDPop::reset(bool dum) {
    // reset ladder state
    for (int i = 0; i < H * M; ++i) s[i] = 0.0f;
    Pop::reset(false);
    CUDA_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize reset");
}

#ifndef CUDATDSETINPUT
// Optional CPU version (for reference / debugging)
void TDPop::setinput(float* input) {
    for (int m = 0; m < M; ++m) {
        // stage0
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
            lgi[h*M + m] = log(sk + EPS);
        }
        // override ys[0]=input[m]
        lgi[(H-1)*M + m] = log(input[m] + EPS);
    }
}

#else

// Optional CPU version (for reference / debugging)
void TDPop::setinput(float* input) {
    // copy CPU input into managed buffer
    memcpy(d_input, input, M * sizeof(float));
    // ensure GPU sees updated values
    setinput_cu(d_input, s, a, b, lgi, M, H, EPS);
}

#endif // CUDATDSETINPUT

// TDPop::TDPop(int K, int My, std::string name, float fs, float tau0, float c) :
//     Pop(K, My) {
//     // K <--> Ht, My <--> M
//     if (not c > 1.0)
//         error("TDPops::TDPops","c not > 1");
    
//     for (int my = 0; my < My; my++) {
//         ltsls.push_back(new LTSL(fs, tau0, c, K));
//         ltsls.back()->reset();
//     }
//     setactfn("SOFTMAX");
// }

// TDPop::~TDPop() {
//     for (auto* p : ltsls) delete p;
//     ltsls.clear();
//     ys = nullptr; // (optional) ys is non-owning anyway
// }

// ulong TDPop::nallocbyte() {
//     ulong nbyte = Pop::nallocbyte();
//     for (int m = 0; m < M; m++)
//         nbyte += ltsls[m]->nallocbyte();
//     return nbyte;
// }

// void TDPop::reset(bool dum) {
//     for (int m = 0; m < M; m++)
//         ltsls[m]->reset();
//     Pop::reset(false);
// }

// #ifndef CUDATDSETINPUT

// void TDPop::setinput(float *input) {
//     // For each LTSL-line serving an y-bin
//     // 1) Update the state (ys) of entire LTSL-line
//     // 2) Deiver the input to the [m, K = 0] tip of the LTSL-line
//     // 3) Deliver the ys[m] input to lgi of the Pop
//     for (int m = 0; m < M; m++) {
//         ys = ltsls[m]->step(input[m]);
//         ys[0] = input[m];
//         for (int h = 0; h < H; h++)
//             lgi[h*M + m] = log(ys[H-h-1] + EPS);
//     }
// }

// #else

// void TDPop::setinput(float *input) {
//     int threads = 256;
//     int blocks  = (M + threads - 1) / threads;
//     tdpop_ltsl_setinput_kernel<<<blocks, threads>>>(input, s, a, b, lgi, M, H, eps);
//     cudaDeviceSynchronize();
// }

// #endif //
