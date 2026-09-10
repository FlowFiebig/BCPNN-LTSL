/*****************************************************************

  Created: 2026-01-14

  Authors: Anders Lansner

  Copyright (c) 2026 Anders Lansner

******************************************************************/

#include "AxoDelay.h"
#include "Prjbase.h"
#include "Prj1.h"

#define CUDADENACT
#define CUDAUPDZITRC
#define CUDAUPDTRACES
#define CUDAUPDBW
#define CUDAUPDBWSUP
#define CUDAUPDMISC
#define CUDANRMMISC
// #define CUDASWAPCONNS
// #define CUDAREPLCONNS
// #define FAST_CPU_SWAPCONNS

using namespace std;
using namespace Globals;

void Prj1::initialize() {
    if (nactNi < 0)
        error("Prj1::initialize", "Illegal: nactNi < 0");
    if (nsilNi < 0)
        error("Prj1::initialize", "Illegal: nsilNi < 0 nsilNi =" + to_string(nsilNi));
    axo = nullptr;
    recurrent = srcpop == trgpop;
    // 'selfc' apply only to recurrent Prj
    selfc = HDON;
    if (libverbosity > 1)
        printf("name = %s Mi = %d, denNi = %d\n", name.c_str(), Mi, denNi);
    if (axoNi < nactNi + nsilNi)
        error("Prj1::initialize", "Illegal axoNi < nactNi + nsilNi");
    tauzidt = 1;
    tauzjdt = 1;
    tauedt = 0;
    taupdt = 1;
    eps = EPS;
    needsupdbw = true;
    wgain = 1;
    bgain = 1;
    ewgain = 0;
    iwgain = 0;
    fgain = 1;
    Vrev = 0.020; // Used for conductance based synapses onto LIF neurons
    nswap = 0;
    nrepl = -1;
    swaprthr = 1.1;
    replkthr = 1;
    knrepl = 1;
    gnswapped = 0;
    gnrepled = 0;
    frozen = false;
    nilist = nullptr;
    nilistoffs = nullptr;
    allocmem();
    initconns(nactNi, nsilNi);
    xfieldi_NjNi = nullptr;
    xfieldf_NjNi = nullptr;
    xfieldi_HjNi = nullptr;
    xfieldf_HjNi = nullptr;
}


Prj1::Prj1(Pop *srcpop, Pop *trgpop, int nactNi, int nsilNi, std::string name) :
    Prjbase(srcpop, trgpop, nactNi, nsilNi, name) {
    axoNi = srcpop->N; 
    denNi = nactNi + nsilNi;
    this->nactNi = nactNi;
    this->nsilNi = nsilNi;

    gsetseed(4711 + 19*id); // Why?
    Nj = trgpop->N;
    Hj = trgpop->H;
    denHi = srcpop->H;
    Mi = srcpop->M;
    Mj = trgpop->M;
    if (axoNi < nactNi + nsilNi) {
        printf("axoNi = %d nactNi = %d nsilNi = %d\n", axoNi, nactNi, nsilNi);
        error("Prj1::Prj1", "Illegal axoNi < nactNi + nsilNi (" + name + ")");
    }
    srcpopact = srcpop->act;
    trgpopact = trgpop->act;
    trgpopbwsup = trgpop->bwsup;
    initialize();
}


Prj1::Prj1(Pop *srcpop, Pop *trgpop, std::string name) :
    Prjbase(srcpop, trgpop, srcpop->N, 0, name) {
    axoNi = srcpop->N; 
    denNi = axoNi;
    nactNi = nactQi; // Prjbase sets nactQi, nsilQi
    nsilNi = nsilQi;

    gsetseed(4711 + 19*id);
    Nj = trgpop->N;
    Hj = trgpop->H;
    denHi = srcpop->H;
    Mi = srcpop->M; // remove Mi?
    Mj = trgpop->M; // remove Mj?
    srcpopact = srcpop->act;
    trgpopact = trgpop->act;
    trgpopbwsup = trgpop->bwsup;
    initialize();
}


Prj1::~Prj1() {
    cudaFree(Zi);
    Zi = nullptr;
    cudaFree(Zj);
    Zj = nullptr;
    cudaFree(Ei);
    Ei = nullptr;
    cudaFree(Ej);
    Ej = nullptr;
    cudaFree(Eji);
    Eji = nullptr;
    cudaFree(Pi);
    Pi = nullptr;
    cudaFree(Pj);
    Pj = nullptr;
    cudaFree(Pji);
    Pji = nullptr;
    cudaFree(Bj);
    Bj = nullptr;
    cudaFree(Wji);
    Wji = nullptr;
    cudaFree(denact);
    denact = nullptr;
    cudaFree(axoact); 
    axoact = nullptr;
    cudaFree(bwsupinf);
    bwsupinf = nullptr;
    cudaFree(bwsup);
    bwsup = nullptr;
    cudaFree(MIji);
    MIji = nullptr;
    cudaFree(nMIji);
    nMIji = nullptr;
    cudaFree(Cji);
    Cji = nullptr;
    cudaFree(Iji);
    Iji = nullptr;
    cudaFree(Ifanout);
    Ifanout = nullptr;
    cudaFree(xfieldi_NjNi);
    xfieldi_NjNi = nullptr;
    cudaFree(xfieldi_HjNi);
    xfieldi_HjNi = nullptr;
    cudaFree(xfieldf_NjNi);
    xfieldf_NjNi = nullptr;
    cudaFree(xfieldf_HjNi);
    xfieldf_HjNi = nullptr;
    delete axo;
}


void Prj1::allocmem() {
    CUDA_CHECK_ERROR(cudaMallocManaged(&Zi, Hj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Pi, Hj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Zj, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Pj, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Bj, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Pji, Nj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Wji, Nj * denNi * sizeof(float)));
    Ei = nullptr;
    Ej = nullptr;
    Eji = nullptr;
    if (tauedt > 0) {
        CUDA_CHECK_ERROR(cudaMallocManaged(&Ei, Hj * denNi * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&Ej, Hj * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&Eji, Nj * denNi * sizeof(float)));
    }
    CUDA_CHECK_ERROR(cudaMallocManaged(&Cji, Hj * denNi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Iji, Hj * denNi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&MIji, Hj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&spmask, Hj * denNi * sizeof(bool)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Ifanout, axoNi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&nMIji, Hj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&nswapped, sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&nrepled, sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&denact, Hj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&axoact, axoNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&bwsupinf, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&bwsup, Nj * sizeof(float)));
    reinitialize();
}


void Prj1::reinitialize() {
    // Initializes all state fields
    if (frozen) return;
    fill1f(Zi, Hj * denNi, eps);
    fill1f(Pi, Hj * denNi, eps);
    fill1f(Zj, Nj, eps);
    fill1f(Pj, Nj, eps);
    fill1f(Pji, Nj * denNi, eps * eps);
    if (tauedt > 0) {
        fill1f(Ei, Hj * denNi, eps);
        fill1f(Ej, Nj, eps);
        fill1f(Eji, Nj * denNi, eps * eps);
    }
    fill1f(Bj, Nj, 0);
    fill1f(Wji, Nj * denNi, 0);
    fill1f(denact, Hj * denNi, 0);
    fill1f(axoact, axoNi, 0);
    fill1f(bwsupinf, Nj, 0);
    fill1f(bwsup, Nj, 0);
    updbw(true);
}


void Prj1::reset() {
    // Resets current state, keeps trained memory
    fill1f(Zi, Hj * denNi, eps);
    fill1f(Zj, Nj, eps);
    if (tauedt > 0 and not frozen) {
        fill1f(Ei, Hj * denNi, eps);
        fill1f(Ej, Nj, eps);
        fill1f(Eji, Nj * denNi, eps * eps);
    }
    fill1f(denact, Hj * denNi, 0);
    fill1f(axoact, axoNi, 0);
    fill1f(bwsupinf, Nj, 0);
    fill1f(bwsup, Nj, 0);
}


ulong Prj1::nallocbyte() {
    ulong nbyte = 0;
    nbyte += 2 * Hj * denNi; // Zi, Pi
    nbyte += 3 * Nj; // Zj. Pj, Bj
    nbyte += 2 * Nj * denNi; // Pji, Wji
    if (tauedt > 0)
        nbyte += Nj + Hj * denNi + Nj * denNi; // Ej, Ei, Eji
    nbyte += Hj * denNi; // denact
    nbyte += axoNi; // axoact
    nbyte += Nj; // bwsupinf
    nbyte += Nj; // bwsup
    nbyte += axoNi; // Ifanout
    // Structural plasticity!
    nbyte += 5 * Hj * denNi; // Cji, Iji, MIji, nMIji, spmask
    nbyte += axoNi; // Ifanout
    nbyte *= 4;
    return nbyte;
}


bool Prj1::ISABSENT(int hj, int ni) {
    int dni, dnifound = false;
    for (dni = 0; dni < denNi; dni++) {
        if (Iji[hj * denNi + dni] == ni) {
            dnifound = true;
            break;
        }
    }
    return not dnifound;
}


void Prj1::initIji(int nactNi, int nsilNi) {
    if (frozen) return;
    if (nactNi + nsilNi > axoNi) {
        printf("nactNi = %d nsilNi = %d selfc == HDOFF = %d axoNi = %d\n",
               nactNi, nsilNi, selfc == HDOFF, axoNi);
        error("Prj1::initIji", "nactNi + nsilNi > axoNi");
    }
    CUDA_CHECK_ERROR(cudaMemset(Cji, ABSENT, Hj * denNi * sizeof(int)));

    // Create index vector later to be shuffled
    std::vector<int> shuffled(axoNi);
    // Loop over each trg hypercol
    for (int hj = 0; hj < Hj; hj++) {
        int ni = 0;
        for (int axoni = 0; axoni < axoNi; axoni++)
            shuffled[axoni] = axoni;
        // Shuffle index for selecting connections
        shuffle(begin(shuffled), end(shuffled), RndGen::grndgen->generator);
        /// Loop and assign SILENT conns, if HDOFF avoid hj == hi
        int dni = 0, axoni = 0, hidx;
        for (int nsilhi = 0; nsilhi < nsilNi; nsilhi++) {
            while (true) {
                ni = shuffled[axoni];
                if (0<=ni) break;
                axoni++;
            }
            hidx = hj * denNi + dni;
            Cji[hidx] = SILENT;
            Iji[hidx] = ni;
            shuffled[axoni] = -1;
            axoni++; dni++;
        }
        for (int nactni = 0; nactni < nactNi; nactni++) {
            while (true) {
                ni = shuffled[axoni];
                if (0<=ni) break;
                axoni++;
            }
            hidx = hj * denNi + dni;
            Cji[hidx] = ACTIVE;
            Iji[hidx] = ni;
            shuffled[axoni] = -1;
            axoni++; dni++;
        }
    }
    updIfanout();
    updbw(true);

    // Create lists for swapconns()
    CUDA_CHECK_ERROR(cudaMallocManaged(&nilist, Hj * denNi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&nilistoffs, (axoNi + 1) * sizeof(int)));
    int nipos = 0;
    for (int ni = 0; ni < axoNi; ni++) {
        nilistoffs[ni] = nipos;
        // printf("*** %d\n", nilistoffs[ni]);
        for (int hidx = 0; hidx < Hj * denNi; hidx++) {
            if (Iji[hidx] == ni) {
                // printf("ni = %4d hidx = %5d nipos = %4d\n", ni, hidx, nipos);
                nilist[nipos++] = hidx;
            }
        }
    }

    nilistoffs[axoNi] = nipos;

}


void Prj1::initconns(int nactNi, int nsilNi) {
    initIji(nactNi, nsilNi);
}


void Prj1::updIfanout() {
    if (not MIji) return;
    CUDA_CHECK_ERROR(cudaMemset(Ifanout, 0, axoNi * sizeof(int)));
    for (int hj = 0; hj < Hj; hj++) {
        for(int dni = 0, ni; dni < denNi; dni++) {
            ni = Iji[hj * denNi + dni];
            Ifanout[ni] += Cji[hj * denNi + dni] == ACTIVE;
        }
    }
    cudaDeviceSynchronize();

}


#ifndef CUDADENACT

void Prj1::upddenact() {
    if (axo == nullptr) {
        CUDA_CHECK_ERROR(cudaMemcpy(axoact, srcpopact, axoNi * sizeof(float), cudaMemcpyDeviceToDevice));
    } else
        axo->updstate();
    for (int hj = 0; hj < Hj; hj++) {
        if (Iji == nullptr) {
            CUDA_CHECK_ERROR(cudaMemcpy(&denact[hj * denNi], axoact, denNi * sizeof(float),
                                        cudaMemcpyDeviceToDevice));
        } else {
            for (int dni = 0; dni < denNi; dni++) {
                int ni = Iji[hj * denNi + dni];
                denact[hj * denNi + dni] = axoact[ni];
            }
        }
    }
}

#else

void Prj1::upddenact() {
    if (axo == nullptr) {
        CUDA_CHECK_ERROR(cudaMemcpy(axoact, srcpopact, axoNi * sizeof(float), cudaMemcpyDeviceToDevice));
    } else {
        axo->updstate();
    }
    if (Iji == nullptr) {
        for (int hj = 0; hj < Hj; hj++)
            CUDA_CHECK_ERROR(cudaMemcpy(&denact[hj * denNi], axoact, denNi * sizeof(float),
                                        cudaMemcpyDeviceToDevice));
    } else
        upddenact_cu(denact, axoact, Iji, axoNi, denNi, Hj, 0);
}

#endif // CUDADENACT

#ifndef CUDAUPDZITRC

void Prj1::updzitrc() {
    for (int hj = 0; hj < Hj; hj++) {
        for (int dni = 0; dni < denNi; dni++) {
            int hidx = hj * denNi + dni;
            Zi[hidx] += (fgain * denact[hidx] * (1 - eps) + eps - Zi[hidx]) * tauzidt;
        }
    }
}

#else

void Prj1::updzitrc() {
    updzitrc_cu(Zi, denact, Hj, denNi, fgain, eps, tauzidt, 0);
}

#endif // CUDAUPDZITRC

#ifndef CUDAUPDTRACES

void Prj1::updtraces(float *trgact, float prn) {

    if (not frozen) {

        float prntaupdt = prn * taupdt;

        for (int nj = 0; nj < Nj; nj++) {
            Zj[nj] += (fgain * trgact[nj] * (1 - eps) + eps - Zj[nj]) * tauzjdt;
            Pj[nj] += (Zj[nj] - Pj[nj]) * prntaupdt;
            if (Pj[nj] == 0) error("Prj1::updtraces", "Illegal: Pj == 0");
        }
        for (int hj = 0; hj < Hj; hj++) {
            for (int dni = 0; dni < denNi; dni++) {
                int hidx = hj * denNi + dni;
                Pi[hidx] += (Zi[hidx] - Pi[hidx]) * prntaupdt;
                if (Pi[hidx] == 0) error("Prj1::updtraces", "Illegal: Pi == 0");
            }
        }
        for (int nj = 0, hj; nj < Nj; nj++) {
            hj = nj/Mj;
            for (int dni = 0; dni < denNi; dni++) {
                int nidx = nj * denNi + dni, hidx = hj * denNi + dni;
                Pji[nidx] += (Zi[hidx] * Zj[nj] - Pji[nidx]) * prntaupdt;
            }
        }
        if (prn > 0)
            needsupdbw = true;
    }
}

#else

void Prj1::updtraces(float *trgact, float prn) {
    updtraces_cu(trgact, prn, frozen, Hj, Mj, Nj, denNi, fgain, eps, tauzjdt, taupdt, Zi, Zj,
                 Pi, Pj, Pji, 0);
    if (prn > 0)
        needsupdbw = true;
}

#endif // CUDAUPDTRACES

void Prj1::updtraces(float prn) {
    updtraces(trgpopact, prn);
}


#ifndef CUDAUPDBW

void Prj1::updbw(bool force) {
    if (frozen) return;
    float pi, pj, pji;
    if (not needsupdbw and not force)
        return;
    for (int nj = 0, hj; nj < Nj; nj++) {
        hj = nj/Mj;
        pj = Pj[nj]; 
        if (pj == 0) error("Prj1::updbw", "pj == 0");  // Numeric cancellation
        Bj[nj] = bgain * log(pj) * nactNi/axoNi; // Co-localized weight + bias in spine

        for (int dni = 0; dni < denNi; dni++) {
            int nidx = nj * denNi + dni, hidx = hj * denNi + dni;            
            pi = Pi[hidx];
            if (pi == 0) error("Prj1::updbw", "pi == 0"); // Numeric cancellation
            pji = Pji[nidx];
            if (pji == 0) error("Prj1::updbw", "pji == 0"); // Numeric cancellation

            float wji = (Cji[hidx] == ACTIVE and not (wgain==0 && ewgain==0 && iwgain==0)) * log(pji/(pi*pj));
            wji *= wgain + (wji > 0) * ewgain + (wji < 0) * iwgain;
            wji *= not (Iji[hidx] == hj and recurrent and selfc == HDOFF);

            Wji[nidx] = wji;

            if (isnan(Wji[nidx])) {
                printf("pj = %f pi = %f pji = %f\n", pj, pi, pji);
                error("Pji1::updbw", "Wji is NaN");
            }
            if (isinf(Wji[nidx])) {
                printf("pi = %f pj = %f pji = %f\n", pi, pj, pji);
                error("Pji1::updbw", "Wji is inf");
            }
        }
    }
    needsupdbw = false;

}

#else

void Prj1::updbw(bool force) {
    if (frozen or not needsupdbw and not force)
        return;
    updbw_cu(Bj, Wji, Pj, Pi, Pji, Cji, Iji, Hj, Mj, Nj, denNi, bgain, wgain, ewgain, iwgain, nactNi, axoNi,
             recurrent, selfc == HDOFF, 0);
    needsupdbw = false;
}

#endif // CUDAUPDBW

void Prj1::putsrcact(float *srcact) {
    for (int i = 0; i < axoNi; i++)
        this->axoact[i] = srcact[i];
}


#ifndef CUDAUPDBWSUP

void Prj1::updbwsup() {
    CUDA_CHECK_ERROR(cudaMemcpy(bwsupinf, Bj, Nj * sizeof(float), cudaMemcpyDeviceToDevice));
    for (int nj = 0, hj; nj < Nj; nj++) {
        hj = nj/Mj;
        for (int dni = 0; dni < denNi; dni++) {
            int nidx = nj * denNi + dni, hidx = hj * denNi + dni;
            bwsupinf[nj] += Zi[hidx] * Wji[nidx];
        }
    }
    for (int nj = 0; nj < Nj; nj++)
        bwsup[nj] += (bwsupinf[nj] - bwsup[nj]) * tauzidt;
}

#else

void Prj1::updbwsup() {
    CUDA_CHECK_ERROR(cudaMemcpy(bwsupinf, Bj, Nj * sizeof(float), cudaMemcpyDeviceToDevice));
    updbwsup_cu(bwsupinf, bwsup, Bj, Zi, Wji, Hj, Mj, Nj, denNi, tauzidt, 0);
}

#endif // CUDAUPDBWSUP


#ifndef CUDACONTRIBUTE

void Prj1::contribute() {
    for (int nj = 0; nj < Nj; nj++)
        trgpopbwsup[nj] += bwsup[nj];
}

#else

void Prj1::contribute() {
    contribute_cu(trgpopbwsup, bwsup, Nj, 0);
}

#endif // CUDACONTRIBUTE

#ifndef CUDAUPDMISC

double mi_term(double p, double q)
{
    if (p > 0.0)
        return p * std::log(p / q);
    else
        return 0.0;
}

double mutual_information_binary(double pi, double pj, double p11)
{
    double p10 = pi - p11;
    double p01 = pj - p11;
    double p00 = 1.0 - pi - pj + p11;

    double MI = 0.0;

    MI += mi_term(p11, pi * pj);
    MI += mi_term(p10, pi * (1.0 - pj));
    MI += mi_term(p01, (1.0 - pi) * pj);
    MI += mi_term(p00, (1.0 - pi) * (1.0 - pj));

    return MI;
}

double log_odds_ratio(double pi, double pj, double pji) {
    return pji * log(pji * (1 - pi - pj + pji) / ((pi - pji) * (pj - pji)));
}

void Prj1::updMIsc() {
    if (frozen) return;
    if (MIji == nullptr)
        error("Prj1::updMisc", "updMIsc disabled");
    CUDA_CHECK_ERROR(cudaMemset(MIji, 0, Hj * denNi * sizeof(float)));
    for (int nj = 0, hj; nj < Nj; nj++) {
        hj = nj/Mj;
        float pj = Pj[nj];
        for (int dni = 0; dni < denNi; dni++) {
            int nidx = nj * denNi + dni, hidx = hj * denNi + dni;
            float pi = Pi[nidx];
            float pji = Pji[nidx];
            float wji, scr;

            // scr += mutual_information_binary(pi, pj, pji);
            // scr = log_odds_ratio(pi, pj, pji);
            wji = log(pji/(pi*pj)); scr = wji;
            // wji = log(pji/(pi*pj)); scr = wji>0 * wji;
            // wji = log(pji/(pi*pj)); scr = fabs(wji);

            if (isnan(scr) or isinf(scr)) continue;
            MIji[hidx] += scr;

        }
        if (recurrent and selfc == HDOFF) {
            for (int dni = 0; dni < denNi; dni++) {
                if (Iji[hj * denNi + dni] == hj)
                    MIji[hj * denNi + dni] = 0;
            }
        }
    }
}

#else

void Prj1::updMIsc() {
    if (frozen) return;
    if (MIji == nullptr)
        error("Prj1::updMIsc", "updMIsc disabled");
    CUDA_CHECK_ERROR(cudaMemset(MIji, 0, Hj * denNi * sizeof(float)));
    updMIsc_cu(MIji, Pi, Pj, Pji, Iji, Hj, Mj, Nj, denNi, recurrent, selfc == HDOFF, 0);
    cudaDeviceSynchronize();
}

#endif // CUDAUPDMISC

#ifndef CUDANRMMISC
void Prj1::nrmMIsc() {
    if (frozen) return;
    if (nMIji == nullptr)
        error("Prj1::updconnscore", "updconnscore disabled");
    CUDA_CHECK_ERROR(cudaMemset(nMIji, 0, Hj * denNi * sizeof(float)));
    for (int hj = 0; hj < Hj; hj++) {
        for (int dni = 0, ni; dni < denNi; dni++) {
            int hidx = hj * denNi + dni;
            ni = Iji[hidx];
            nMIji[hidx] = MIji[hidx] / (1 + Ifanout[ni]);
        }
    }
}

#else

void Prj1::nrmMIsc() {
    if (frozen) return;
    if (nMIji == nullptr)
        error("Prj1::updconnscore", "updconnscore disabled");
    CUDA_CHECK_ERROR(cudaMemset(nMIji, 0, Hj * denNi * sizeof(float)));
    nrmMIsc_cu(MIji, Iji, Ifanout, nMIji, Hj * denNi, 0);
    cudaDeviceSynchronize();
}

#endif // CUDANRMMISC


float Prj1::miscsum(string filename) {
    if (frozen) return 0;
    float silsum = 0, actsum = 0;
    for (int hj = 0; hj < Hj; hj++) {
        for (int dni = 0; dni < denNi;  dni++) {
            int hidx = hj * denNi + dni;
            silsum += (Cji[hidx] == SILENT) * nMIji[hidx];
            actsum += (Cji[hidx] == ACTIVE) * nMIji[hidx];
        }
    }
    silsum /= nsilNi;
    actsum /= nactNi;
    if (filename == "") {
        if (libverbosity > 0) {
            if (nsilNi > 0)
                printf("actsum = %6.3f silsum = %6.3f\n", actsum, silsum);
            else
                printf("actsum = %6.3f\n", actsum);
        }
    } else {
        FILE *outf = fopen(filename.c_str(), "a");
        fprintf(outf, "%6.3f %6.3f\n", actsum, silsum);
        fclose(outf);
    }
    return actsum;
}


void Prj1::miscsumx(float& actmiscsum, float& silmiscsum, string filename) {
    if (frozen) return;
    float amiscsum = 0, smiscsum = 0;
    for (int hj = 0; hj < Hj; hj++) {
        for (int dni = 0; dni < denNi;  dni++) {
            int hidx = hj * denNi + dni;
            amiscsum += (Cji[hidx] == ACTIVE) * nMIji[hidx]/nactNi;
            smiscsum += (Cji[hidx] == SILENT) * nMIji[hidx]/nsilNi;
        }
    }
    actmiscsum += amiscsum;
    silmiscsum += smiscsum;
    if (filename == "") {
        if (libverbosity > 0) {
            if (nsilNi > 0)
                printf("amiscsum = %6.3f silmiscsum = %6.3f", amiscsum, smiscsum);
            else
                printf("amiscsum = %6.3f\n", amiscsum);
        }
    } else {
        FILE *outf = fopen(filename.c_str(), "a");
        fprintf(outf, "%6.3f %6.3f\n", amiscsum, smiscsum);
        fclose(outf);
    }
}


#ifndef CUDASWAPCONNS

void Prj1::swapconns() {
    if (frozen) return;
    struct timeval swapconns_time;
    gettimeofday(&swapconns_time, 0);

    updMIsc();
    updIfanout();
    nrmMIsc();

    CUDA_CHECK_ERROR(cudaMemset(spmask, 1, Hj * denNi * sizeof(bool)));

    *nswapped = 0;

    for (int hj = 0; hj < Hj; hj++) {

        for (int swapid = 0; swapid < nswap; swapid++) {

            int silmax_dni = -1, actmin_dni = -1;
            float silmax_score = -1e7, actmin_score = 1e7;
        
            for (int dni = 0; dni < denNi; dni++) {
                int hidx = hj * denNi + dni;
                if (Cji[hidx]==ACTIVE and spmask[hidx]) {
                    if (nMIji[hidx] < actmin_score) {
                        actmin_dni = dni;
                        // Add jitter to avoid packing index order effects
                        actmin_score = nMIji[hidx] + 1e-6 * gnextfloat();
                    }
                } else if (Cji[hidx]==SILENT and spmask[hidx]) {
                    if (nMIji[hidx] > silmax_score) {
                        silmax_dni = dni;
                        silmax_score = nMIji[hidx];
                    }
                }
            }

            if (silmax_dni==-1 or actmin_dni==-1 or silmax_score < swaprthr * actmin_score) break;

            Cji[hj * denNi + actmin_dni] = SILENT;
            Cji[hj * denNi + silmax_dni] = ACTIVE;
            (*nswapped)++;

#ifdef FAST_CU_SWAPCONNS
            // Targeted updating Ifanout and nMIhjhi
            int ni = Iji[hj * denNi + actmin_dni];
            int nextIfanout_ni = Ifanout[ni] - 1;
            float factor = (1.0f + (float)Ifanout[ni]) / (1.0f + (float)nextIfanout_ni);
            for (int i = nilistoffs[ni]; i < nilistoffs[ni+1]; i++)
                nMIji[nilist[i]] *= factor;
            Ifanout[ni] = nextIfanout_ni;

            ni = Iji[hj * denNi + silmax_dni];
            nextIfanout_ni = Ifanout[ni] + 1;
            float factor = (1.0f + (float)Ifanout[ni]) / (1.0f + (float)nextIfanout_ni);
            for (int i = nilistoffs[ni]; i < nilistoffs[ni+1]; i++)
                nMIji[nilist[i]] *= factor;
            Ifanout[ni] = nextIfanout_ni;
#else
            updIfanout();
            nrmMIsc();
#endif // FAST_CU_SWAPCONNS

            spmask[hj * denNi + actmin_dni] = false;
            spmask[hj * denNi + silmax_dni] = false;

        }
    }

    gnswapped += *nswapped;

    needsupdbw = true;

    // float timelapsed = getDiffTime(swapconns_time) / 1000;
    // printf("swapconns_time (CPU) = %.3f sec\n", timelapsed);

}

#else

void Prj1::swapconns() {
    struct timeval swapconns_time;
    gettimeofday(&swapconns_time, 0);
    if (frozen) return;

    updMIsc();
    nrmMIsc();
    updIfanout();

    CUDA_CHECK_ERROR(cudaMemsetAsync(nswapped, 0, sizeof(*nswapped)));
    size_t n = (size_t)Hj * (size_t)denNi;
    CUDA_CHECK_ERROR(cudaMemsetAsync(spmask, 1, n));

    swapconns_cu(spmask, nswapped, 0);

    gnswapped += *nswapped;

    needsupdbw = true;

    float timelapsed = getDiffTime(swapconns_time) / 1000;
    printf("swapconns_time (GPU) = %.3f sec\n", timelapsed);

}

#endif // CUDASWAPCONNS

#ifndef CUDAREPLCONNS

void Prj1::replconns() {
    if (frozen) return;
    struct timeval replconns_time;
    gettimeofday(&replconns_time, 0);

    if (nsilNi <= 0 or nactNi + nsilNi == axoNi)
        return;

    // some entries in spmask set to false by previous call to swapconns
    if (not spmask)
        error("Prj1::replconns", "Illegal: spmask missing");

    updMIsc();

    *nrepled = 0;

    for (int hj = 0; hj < Hj; hj++) {
       
        for (int replid=0; replid<nrepl; replid++) {

            int silmin_dni = -1;
            float silscore_min = 1e7, silscore_max = -1e7, silscore_mean = 0, silscore_std = 0;
        
            for (int dni = 0; dni < denNi; dni++) {
                int hidx = hj*denNi + dni;
                if (Cji[hidx]==SILENT and spmask[hidx]) {
                    silscore_mean += nMIji[hidx];
                    if (nMIji[hidx] < silscore_min) {
                        silmin_dni = dni;
                        silscore_min = nMIji[hidx];
                    }
                    if (nMIji[hidx] > silscore_max)
                        silscore_max = nMIji[hidx];
                }
            }
            silscore_mean /= nsilNi;
            if (silmin_dni==-1)
                continue;
            if (silscore_mean < 0) silscore_mean = 0;
            for (int dni = 0; dni < denNi; dni++) {
                int hidx = hj*denNi + dni;
                if (Cji[hidx]==SILENT) {
                    silscore_std += (nMIji[hidx] - silscore_mean) * (nMIji[hidx] - silscore_mean);
                }
            }
            if (nsilNi > 1)
                silscore_std = sqrt(silscore_std/(nsilNi-1));
            else
                silscore_std = 0;
            bool looping = false;

            // printf("hj = %2d silscore_min = %.2e silscore_mean = %.2e silscore_std = %.2e silscore_thr = %.2e\n",
            //        hj, silscore_min, silscore_mean, silscore_std, silscore_mean - replkthr * silscore_std);
            // printf("hj = %d silscore_min = %.1e silscore_max = %.1e silscore_mean = %.1e silscore_std = %.1e\n",
            //        hj, silscore_min, silscore_max, silscore_mean, silscore_std);

            if (silscore_min > silscore_mean - replkthr * silscore_std) break;

            int ni, ntry = 0;
            do {
                ni = gnextint() % axoNi;
                if (++ntry == 100) {
                    looping = true;
                    break;
                }
            } while (not ISABSENT(hj, ni));
            // if (looping) printf("replconns looping! %d ", hj);

            Iji[hj * denNi + silmin_dni] = ni;

            (*nrepled)++;
            spmask[hj * denNi + silmin_dni] = false; // Prevents picking same silmax_dni

        }
    }

    gnrepled += *nrepled;

    needsupdbw = true;

    // float timelapsed = getDiffTime(replconns_time) / 1000;
    // printf("replconns_time = %.3f sec\n", timelapsed);

}

#else

void Prj1::replconns() {
    // struct timeval replconns_time;
    // gettimeofday(&replconns_time, 0);

    CUDA_CHECK_ERROR(cudaMemsetAsync(nrepled, 0, sizeof(*nrepled)));

    // size_t n = (size_t)Nj * (size_t)denNi;
    // Presupposes that swapconns is run before replconns and that it cudaMemsetAsync(spmask)
    // CUDA_CHECK_ERROR(cudaMemsetAsync(spmask, 1, n));

    replconns_cu(spmask, nrepled, 0);

    gnrepled += *nrepled;

    needsupdbw = true;

    // float timelapsed = getDiffTime(replconns_time) / 1000;
    // printf("replconns_time = %.3f sec\n", timelapsed);
}

#endif // CUDAREPLCONNS


int Prj1::getnelem(int field) {
    int nelem;
    switch (field) {
        case CJI:
        case IJI:
            nelem = Hj * denNi;
            break;
        case IFANOUT:
            nelem = axoNi;
            break;
        case AXOACT:
            nelem = axoNi;
            break;
        case DENACT:
            nelem = Hj * denNi;
            break;
        case NMIJI:
        case MIJI:
            nelem = Hj * denNi;
            break;
        case BWSUPINF:
        case BWSUP:
            nelem = Nj;
            break;
        case ZJ:
        case EJ:
        case PJ:
        case BJ:
            nelem = Nj;
            break;
        case ZI:
        case EI:
        case PI:
            nelem = Hj * denNi;
            break;
        case EJI:
        case PJI:
        case PJPI:
        case WJI:
            nelem = Nj * denNi;
            break;
        case DENACTX:
        case ZIX:
        case EIX:
        case PIX:
        case CJIX:
        case MIJIX:
        case NMIJIX:
            nelem = Hj * axoNi;
            break;
        case PJIX:
        case WJIX:
            nelem = Nj * axoNi;
            break;
        default:
            error("Prj1::getnelem", "No such field: '" + fieldtostring(field) + "'");
    }
    return nelem;
}


int *Prj1::expandfieldi(int field) {
    if (xfieldi_HjNi == nullptr)
        CUDA_CHECK_ERROR(cudaMallocManaged(&xfieldi_HjNi, Hj * axoNi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset(xfieldi_HjNi, 0, Hj * axoNi * sizeof(int)));
    for (int hj = 0; hj < Hj; hj++) {
        for (int dni = 0; dni < denNi; dni++) {
            if (field ==  CJIX)
                xfieldi_HjNi[hj*axoNi + Iji[hj*denNi + dni]] = Cji[hj*denNi + dni];
            else
                error("Prj1::expandfieldi", "Illegal field");
        }
    }
    return xfieldi_HjNi;
}


float *Prj1::expandfieldf(int field) {
    if (field == WJIX or field == PJIX) {
        if (xfieldf_NjNi == nullptr)
            CUDA_CHECK_ERROR(cudaMallocManaged(&xfieldf_NjNi, Nj * axoNi * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(xfieldf_NjNi, 0, Nj * axoNi * sizeof(float)));
        for (int nj = 0, hj; nj < Nj; nj++) {
            hj = nj/Mj;
            for (int dni = 0; dni < denNi; dni++) {
                int ni = Iji[hj*denNi + dni];
                if (field == WJIX)
                    xfieldf_NjNi[nj*axoNi + ni] = Wji[nj*denNi + dni];
                else if (field == PJIX)
                    xfieldf_NjNi[nj*axoNi + ni] = Pji[nj*denNi + dni];
                else
                    error("Prj1::expandfieldf", "Illegal field " + fieldtostring(field));
            }
        }
 
       return xfieldf_NjNi;
 
   } else if (field == DENACTX or field == ZIX or field == EIX or field == PIX or
              field == MIJIX or field == NMIJIX) {
        if (xfieldf_HjNi == nullptr)
            CUDA_CHECK_ERROR(cudaMallocManaged(&xfieldf_HjNi, Hj * axoNi * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(xfieldf_HjNi, 0, Hj * axoNi * sizeof(float)));
        for (int hj = 0; hj < Hj; hj++) {
            for (int dni = 0; dni < denNi; dni++) {
                int ni = Iji[hj*denNi + dni];
                if (field == DENACTX)
                    xfieldf_HjNi[hj*axoNi + ni] = denact[hj*denNi + dni];
                else if (field == ZIX)
                    xfieldf_HjNi[hj*axoNi + ni] = Zi[hj*denNi + dni];
                else if (field == EIX)
                    xfieldf_HjNi[hj*axoNi + ni] = Ei[hj*denNi + dni];
                else if (field == PIX)
                    xfieldf_HjNi[hj*axoNi + ni] = Pi[hj*denNi + dni];
                else if (field == MIJIX)
                    xfieldf_HjNi[hj*axoNi + ni] = MIji[hj*denNi + dni];
                else if (field == NMIJIX)
                    xfieldf_HjNi[hj*axoNi + ni] = nMIji[hj*denNi + dni];
                else
                    error("Prj1::expandfieldf", "Illegal field " + fieldtostring(field));
            }
        }

        return xfieldf_HjNi;

    }

    return nullptr;

}

float *Prj1::expandfieldf1(int field) {  // Remove from Prjbase.h
    return nullptr;
} 

int *Prj1::getfieldi(int field) {
    switch(field) {
        case CJI:
            return Cji;
        case IJI:
            return Iji;
        case IFANOUT:
            return Ifanout;
            break;
        default:
            error("getfieldi", "No such int field: '" + fieldtostring(field) + "'");
    }
    return nullptr;
}

float *Prj1::getfieldf(int field) {
    switch(field) {
        case BWSUPINF:
            return bwsupinf;
            break;
        case BWSUP:
            return bwsup;
            break;
        case AXOACT:
            return axoact;
            break;
        case DENACT: 
            return denact;
            break;
        case ZI:
            return Zi;
            break;
        case PI:
            return Pi;
            break;
        case EI:
            if (tauedt == 0)
                error("Prj1::getfieldf", "E* state not enabled");
            return getfieldf(field);
            break;
        case ZJ:
            return Zj;
            break;
        case EJ:
            if (tauedt == 0)
                error("Prj1::getfieldf", "E* state not enabled");
            return Ej;
            break;
        case PJ:
            return Pj;
            break;
        case PJI:
            return Pji;
            break;
        case EJI:
            if (tauedt == 0)
                error("Prj1::getfieldf", "E* state not enabled");
            return Eji;
            break;
        case BJ:
            return Bj;
            break;
        case WJI:
            return Wji;
            break;
        case MIJI:
            return MIji;
            break;
        default:
            error("getfieldf", "No such float field: '" + fieldtostring(field) + "'");
    }
    return nullptr;
}

void Prj1::prnfield(int field, string filename) {
    FILE *outfp;
    if (filename == "")
        outfp = stderr;
    else
        outfp = fopen(filename.c_str(), "wb");
    int *fielddatai;
    float *fielddataf;
    switch (field) {
        case CJI:
        case IJI:
        case IFANOUT:
            fielddatai = getfieldi(field);
            if (not fielddatai)
                error("Prj1::prnfield", "Field " + fieldtostring(field) + " is invalid");
            fwrite(fielddatai, sizeof(float), getnelem(field), outfp);
            break;
        case DENACTX:
        case ZIX:
        case EIX:
        case PIX:
        case MIJIX:
        case NMIJIX:
        case PJIX:
        case WJIX:
            fielddataf = expandfieldf(field);
            fwrite(fielddataf, sizeof(float), getnelem(field), outfp);
            break;
        case CJIX:
            fwrite(expandfieldi(field), sizeof(int), getnelem(field), outfp);
            break;
        case AXOACT:
        case DENACT:
        case MIJI:
        case BWSUPINF:
        case BWSUP:
        case ZI:
        case EI:
        case PI:
        case ZJ:
        case EJ:
        case PJ:
        case BJ:
        case EJI:
        case PJI:
        case PJPI:
        case WJI:
            fielddataf = getfieldf(field);
            if (not fielddataf)
                error("Prj1::prnfield", "Field " + fieldtostring(field) + " is invalid");
            fwrite(fielddataf, sizeof(int), getnelem(field), outfp);
            break;
        default:
            error("Prj1::prnfield", "No such field: '" + fieldtostring(field) + "'");
    }
    fclose(outfp);
}

void Prj1::prnfield(string field, string filename) {
    prnfield(stringtofield(field), filename);
}
