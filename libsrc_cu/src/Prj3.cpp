/*****************************************************************

  Created: 2026-01-14

  Authors: Anders Lansner

  Copyright (c) 2026 Anders Lansner

******************************************************************/

#include "AxoDelay.h"
#include "Prjbase.h"
#include "Prj3.h"

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
    nswap = 0;
    gnswapped = 0;
    gnrepled = 0;
    frozen = false;
    allocmem();
    initconns(nactNi, nsilNi);
    xfieldi_NjNi = nullptr;
    xfieldf_NjNi = nullptr;
}


Prj1::Prj1(Pop *srcpop, Pop *trgpop, int nactNi, int nsilNi, std::string name) :
    Prjbase(srcpop, trgpop, nactNi, nsilNi, name) {
    axoNi = srcpop->N; 
    denNi = nactNi + nsilNi;
    nactQi = nactNi; nsilQi = nsilNi;
    this->nactNi = nactNi; // Prjbase has nactQi, nsilQi
    this->nsilNi = nsilNi;
    Nj = trgpop->N;
    denHi = srcpop->H;
    Mi = srcpop->M; // remove Mi?
    Mj = trgpop->M; // remove Mj?
    trgact2 = nullptr;
    if (axoNi < nactNi + nsilNi)
        error("Prj1::Prj1", "Illegal axoNi < nactNi + nsilNi");
    srcpopact = srcpop->act;
    trgpopact = trgpop->act;
    trgpopbwsup = trgpop->bwsup;
    initialize();
}


Prj1::Prj1(Pop *srcpop, Pop *trgpop, std::string name) :
    Prjbase(srcpop, trgpop, 0, 0, name) {
    axoNi = srcpop->N; 
    denNi = nactNi + nsilNi;
    this->nactNi = nactQi; // Prjbase has nactQi, nsilQi
    this->nsilNi = nsilQi;
    Nj = trgpop->N;
    denHi = srcpop->H;
    trgact2 = nullptr;
    this->nactNi = axoNi;
    this->nsilNi = 0;
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
    cudaFree(trgact2);
    trgact2 = nullptr;
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
    cudaFree(xfieldf_NjNi);
    xfieldf_NjNi = nullptr;
    delete axo;
}


void Prj1::allocmem() {
    CUDA_CHECK_ERROR(cudaMallocManaged(&Zi, Nj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Pi, Nj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Zj, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Pj, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Bj, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Pji, Nj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Wji, Nj * denNi * sizeof(float)));
    Ei = nullptr;
    Ej = nullptr;
    Eji = nullptr;
    if (tauedt > 0) {
        CUDA_CHECK_ERROR(cudaMallocManaged(&Ei, Nj * denNi * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&Ej, Nj * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&Eji, Nj * denNi * sizeof(float)));
    }
    CUDA_CHECK_ERROR(cudaMallocManaged(&Cji, Nj * denNi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Iji, Nj * denNi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&MIji, Nj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Ifanout, axoNi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&nMIji, Nj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&nswapped, sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&nrepled, sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&denact, Nj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&axoact, axoNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&bwsupinf, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&bwsup, Nj * sizeof(float)));
    reinitialize();
}


void Prj1::reinitialize() {
    // Initializes all state fields
    if (frozen) return;
    fill1f(Zi, Nj * denNi, eps);
    fill1f(Pi, Nj * denNi, eps);
    fill1f(Zj, Nj, eps);
    fill1f(Pj, Nj, eps);
    fill1f(Pji, Nj * denNi, eps * eps);
    if (tauedt > 0) {
        fill1f(Ei, Nj * denNi, eps);
        fill1f(Ej, Nj, eps);
        fill1f(Eji, Nj * denNi, eps * eps);
    }
    fill1f(Bj, Nj, 0);
    fill1f(Wji, Nj * denNi, 0);
    fill1f(denact, Nj * denNi, 0);
    fill1f(axoact, axoNi, 0);
    fill1f(bwsupinf, Nj, 0);
    fill1f(bwsup, Nj, 0);
    updbw(true);
}


void Prj1::reset() {
    // Resets current state, keeps trained memory
    fill1f(Zi, Nj * denNi, eps);
    fill1f(Zj, Nj, eps);
    if (tauedt > 0 and not frozen) {
        fill1f(Ei, Nj * denNi, eps);
        fill1f(Ej, Nj, eps);
        fill1f(Eji, Nj * denNi, eps * eps);
    }
    fill1f(denact, Nj * denNi, 0);
    fill1f(axoact, axoNi, 0);
    fill1f(bwsupinf, Nj, 0);
    fill1f(bwsup, Nj, 0);
}


ulong Prj1::nallocbyte() {
    ulong nbyte = 0;
    nbyte += 2 * Nj * denNi; // Zi, Pi
    nbyte += 3 * Nj; // Zj. Pj, Bj
    nbyte += 2 * Nj * denNi; // Pji, Wji
    if (tauedt > 0)
        nbyte += Nj + 2 * Nj * denNi;
    nbyte += Nj * denNi; // denact
    nbyte += axoNi; // axoact
    nbyte += Nj; // bwsupinf
    nbyte += Nj; // bwsup
    nbyte += axoNi; // Ifanout
    // Structural plasticity!
    nbyte += 4 * Nj * denNi; // Cji, Iji, MIji, nMIji
    nbyte += axoNi; // Ifanout
    nbyte *= 4;
    return nbyte;
}


bool Prj1::ISABSENT(int nj, int ni) {
    int dni, dnifound = false;
    for (dni = 0; dni < denNi; dni++) {
        if (Iji[nj * denNi + dni] == ni) {
            dnifound = true;
            break;
        }
    }
    return not dnifound;
}


// Not used
void Prj1::fixupselfc(vector<int> shuffled) {
    // selfc in { HDON, HDOFF }
    for (int nj = 0; nj < Nj; nj++) {
        for (int dni = 0; dni < denNi; dni++) {
            switch (selfc) {
            // case HDONL: // Should be subclass Prj11
            //     // ACTIVE only allowed on diagonal
            //     if (nactNi != 1)
            //         error("Prj1::fixupselfc","Illegal: nactNi != 1");
            //     // Reset Cji an dHiji and recreate on diagonal
            //     Cji[nj*denNi + dni] = ACTIVE;
            //     Iji[nj*denNi + dni] = nj;
            //     if (Iji)
            //         error("Prj1::fixupselfc","Illegal Iji != nullptr when HDON");
            //     break;
            case HDOFF:
                if (Iji[nj*denNi + dni] == nj and Cji[nj*denNi + dni] != ABSENT) {
                    printf("nj = %d ni = %d state = %d\n", nj, Iji[nj*denNi + dni], Cji[nj*denNi + dni]);
                    error("Prj1::fixupselfc","Illegal nj == Iji[] when HDOFF");
                }
                break;
            default:
                /* No modifications apply */
                break;
            }
        }
    }
}


bool Prj1::ondiag(int nj, int ni) {
    return recurrent and nj == ni;
}


void Prj1::initIji(int nactNi, int nsilNi) {
    if (frozen) return;
    if (nactNi + nsilNi > axoNi) {
        printf("nactNi = %d nsilNi = %d selfc == HDOFF = %d axoNi = %d\n",
               nactNi, nsilNi, selfc == HDOFF, axoNi);
        error("Prj1::initIji", "nactNi + nsilNi > axoHi");
    }
    CUDA_CHECK_ERROR(cudaMemset(Cji, ABSENT, Nj * denNi * sizeof(int)));
    // Create index vector later to be shuffled
    std::vector<int> shuffled(axoNi);
    // Loop over each trg hypercol
    for (int nj = 0; nj < Nj; nj++) {
        int ni = 0;
        for (int axoni = 0; axoni < axoNi; axoni++)
            shuffled[axoni] = axoni;
        // Shuffle index for selecting connections
        shuffle(begin(shuffled), end(shuffled), RndGen::grndgen->generator);
        /// Loop and assign SILENT conns, if HDOFF avoid nj == hi
        int dni = 0, axoni = 0, nidx;
        for (int nsilhi = 0; nsilhi < nsilNi; nsilhi++) {
            while (true) {
                ni = shuffled[axoni];
                if (0<=ni) break;
                axoni++;
            }
            nidx = nj * denNi + dni;
            Cji[nidx] = SILENT;
            Iji[nidx] = ni;
            shuffled[axoni] = -1;
            axoni++; dni++;
        }
        for (int nactni = 0; nactni < nactNi; nactni++) {
            while (true) {
                ni = shuffled[axoni];
                if (0<=ni) break;
                axoni++;
            }
            nidx = nj * denNi + dni;
            Cji[nidx] = ACTIVE;
            Iji[nidx] = ni;
            shuffled[axoni] = -1;
            axoni++; dni++;
        }
    }

    updIfanout();
    updbw(true);

}


void Prj1::initconns(int nactNi, int nsilNi) {
    initIji(nactNi, nsilNi);
}


void Prj1::setreplkthr(float replkthr) {
    this->replkthr = replkthr;
}


void Prj1::updIfanout() {
    if (not MIji) return;
    CUDA_CHECK_ERROR(cudaMemset(Ifanout, 0, axoNi * sizeof(int)));
    for (int nj = 0; nj < Nj; nj++) {
        for(int dni = 0, ni; dni < denNi; dni++) {
            ni = Iji[nj * denNi + dni];
            Ifanout[ni] += Cji[nj * denNi + dni] == ACTIVE;
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
    for (int nj = 0; nj < Nj; nj++) {
        if (Iji == nullptr) {
            CUDA_CHECK_ERROR(cudaMemcpy(&denact[nj * denNi], axoact, denNi * sizeof(float),
                                        cudaMemcpyDeviceToDevice));
        } else {
            for (int dni = 0; dni < denNi; dni++) {
                int ni = Iji[nj * denNi + dni];
                denact[nj * denNi + dni] = axoact[ni];
            }
        }
    }
}

#else

void Prj1::upddenact() {
    error("Prj1::upddenact()", "Under construction");
    if (axo == nullptr) {
        CUDA_CHECK_ERROR(cudaMemcpy(axoact, srcpopact, axoNi * sizeof(float), cudaMemcpyDeviceToDevice));
    } else {
        axo->updstate();
    }
}

#endif // CUDADENACT

#ifndef CUDAUPDZITRC

void Prj1::updzitrc() {
    for (int nj = 0; nj < Nj; nj++) {
        for (int dni = 0; dni < denNi; dni++) {
            int k = nj * denNi + dni;
            Zi[k] += (fgain * denact[k] * (1 - eps) + eps - Zi[k]) * tauzidt;
        }
    }
}

#else

void Prj1::updzitrc() {
    updzitrc_cu(denact, Nj, denNi, fgain, eps, tauzidt, Zi); // Hj --> Nj
}

#endif // CUDAUPDZITRC

#ifndef CUDAUPDTRACES

void Prj1::updtraces(float *denact, float *trgact, float prn) {

    // updating of Zi is done by updzitrc

    if (not frozen) {

        float prntaupdt = prn * taupdt;

        for (int nj = 0; nj < Nj; nj++) {
            Zj[nj] += (fgain * trgact[nj] * (1 - eps) + eps - Zj[nj]) * tauzjdt;
            Pj[nj] += (Zj[nj] - Pj[nj]) * prntaupdt;
        }

        for (int nj = 0; nj < Nj; nj++) {
            for (int dni = 0; dni < denNi; dni++) {
                int k = nj * denNi + dni;
                Pi[k] += (Zi[k] - Pi[k]) * prntaupdt;
            }
        }
        for (int nj = 0; nj < Nj; nj++) {
            for (int dni = 0; dni < denNi; dni++) {
                int k = nj * denNi + dni;
                Pji[k] += (Zi[k] * Zj[nj] - Pji[k]) * prntaupdt;
            }
        }
        if (prn > 0)
            needsupdbw = true;
    }
}

#else

void Prj1::updtraces(float *denact, float *trgact, float prn) {
    updtraces_cu(denact, trgpopact, prn, frozen,
                 Nj, Nj, Mj, denNi,                       // Hj --> Nj, remove Mj?
                 fgain, eps, tauzidt, tauzjdt, taupdt,
                 Zj, Zi, Pj, Pi, Pji);
    if (prn > 0)
        needsupdbw = true;
}

#endif // CUDAUPDTRACES

void Prj1::updtraces(float prn) {
    updtraces(denact, trgpopact, prn);
}


#ifndef CUDAUPDBW

void Prj1::updbw(bool force) {
    if (frozen) return;
    float pi, pj, pji;
    if (not needsupdbw and not force)
        return;
    for (int nj = 0; nj < Nj; nj++) {
        pj = Pj[nj];
        Bj[nj] = bgain * log(pj);
        for (int dni = 0; dni < denNi; dni++) {
            pi = Pi[nj * denNi + dni];
            pji = Pji[nj * denNi + dni];
            float wji = (Cji[nj*denNi + dni] == ACTIVE) * log(pji / (pi * pj));
            wji *= wgain + (wji > 0) * ewgain + (wji < 0) * iwgain;
            Wji[nj * denNi + dni] = wji;
        }
        if (recurrent and selfc == HDOFF) {
            for (int dni = 0; dni < denNi; dni++) {
                if (Iji[nj*denNi + dni] == nj)
                    Wji[nj * denNi + dni] = 0;
            }
        }
    }
    needsupdbw = false;
}

#else

void Prj1::updbw(bool force) {
    if (frozen or not needsupdbw and not force)
        return;
    // remove Mi, Mj? Iji?
    updbw_cu(BCP, Nj, Mj, denHi, denNi, Mi, Cji, Pj, Pi, Pji, Bj, Wji, eps, bgain, wgain, ewgain, iwgain);
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
    for (int nj = 0; nj < Nj; nj++)
        for (int dni = 0; dni < denNi; dni++)
            bwsupinf[nj] += Zi[nj * denNi + dni] * Wji[nj * denNi + dni];
    for (int nj = 0; nj < Nj; nj++)
        bwsup[nj] += (bwsupinf[nj] - bwsup[nj]) * tauzidt;
}

#else

void Prj1::updbwsup() {
    CUDA_CHECK_ERROR(cudaMemcpy(bwsupinf, Bj, Nj * sizeof(float), cudaMemcpyDeviceToDevice));
    updbwsup_cu(Zi, Bj, Wji, Nj, Mj, denNi, tauzidt, bwsupinf, bwsup);  // Hj --> Nj, remove Mj
}

#endif // CUDAUPDBWSUP


#ifndef CUDACONTRIBUTE

void Prj1::contribute() {
    for (int nj = 0; nj < Nj; nj++)
        trgpopbwsup[nj] += bwsup[nj];
}

#else

void Prj1::contribute() {
    contribute_cu(bwsup, trgpopbwsup, Nj);
}

#endif // CUDACONTRIBUTE


#include <cmath>

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

#ifndef CUDAUPDMISC

void Prj1::updMIsc() {
    if (frozen) return;
    if (MIji == nullptr)
        error("Prj1::updMisc", "updMIsc disabled");
    CUDA_CHECK_ERROR(cudaMemset(MIji, 0, Nj * denNi * sizeof(float)));
    for (int nj = 0; nj < Nj; nj++) {
        float pj = Pj[nj];
        for (int dni = 0; dni < denNi; dni++) {
            int nidx = nj * denNi + dni;
            float pi = Pi[nidx];
            float pji = Pji[nidx];
            // if (pi == pji)
            //     MIji[nidx] += pji * log(pji * (1 - pj) / (pj - pji));
            // else if (pj == pji)
            //     MIji[nidx] += pji * log(pji * (1 - pi) / (pi - pji));
            // else if ((pi == pji) and (pj == pji))
            //     MIji[nidx] += pji * log(pji * (1 - pi));
            // else
            //     MIji[nidx] += pji * log(pji * (1 - pi - pj + pji) / ((pi - pji) * (pj - pji)));

            //     MIji[nidx] = pi * pj + pi * (1 - pj) + (1 - pi) * pj + (1 - pi) * (1 - pj);

            MIji[nidx] = mutual_information_binary(pi, pj, pji);

            if (isinf(MIji[nidx])) {
                fprintf(stderr, "%f %f %f %f %f %f/n", pji, pi, pj, (1 - pi - pj + pji), (pi - pji), (pj - pji));
                exit(9);
            }
            
        }
        if (recurrent and selfc == HDOFF) {
            for (int dni = 0; dni < denNi; dni++) {
                if (Iji[nj * denNi + dni] == nj)
                    MIji[nj * denNi + dni] = 0;
            }
        }
    }
}

#else

void Prj1::updMIsc() {
    if (frozen) return;
    if (MIji == nullptr)
        error("Prj1::updMIsc", "updMIsc disabled");
    CUDA_CHECK_ERROR(cudaMemset(MIji, 0, Nj * denHi * sizeof(float)));
    updMIsc_cu(Pj, Pi, Pji, eps, Cji, Ifanout, 0, Nj, Mj, denHi, denNi, Mi, MIji); // Hj --> 0, remove Mi? Iji?
    cudaDeviceSynchronize();
}

#endif // CUDAUPDMISC

#ifndef CUDANRMMISC

void Prj1::nrmMIsc() {
    if (frozen) return;
    if (nMIji == nullptr)
        error("Prj1::updconnscore", "updconnscore disabled");
    CUDA_CHECK_ERROR(cudaMemset(nMIji, 0, Nj * denNi * sizeof(float)));
    for (int nj = 0; nj < Nj; nj++) {
        for (int dni = 0, ni; dni < denNi; dni++) {
            ni = Iji[nj * denNi + dni];
            nMIji[nj * denNi + dni] = MIji[nj * denNi + dni] / (1 + Ifanout[ni]);
        }
    }
}

#else

void Prj1::nrmMIsc() {
    if (frozen) return;
    if (nMIji == nullptr)
        error("Prj1::updconnscore", "updconnscore disabled");
    CUDA_CHECK_ERROR(cudaMemset(nMIji, 0, Nj * denNi * sizeof(float)));

    nrmMIsc_cu(Iji, Ifanout, Nj, denNi, denHi, MIji, nMIji); // Hj -> 0
    cudaDeviceSynchronize();
}

#endif // CUDANRMMISC


float Prj1::miscsum(string filename) {
    if (frozen) return 0;
    float silsum = 0, actsum = 0;
    for (int nj = 0; nj < Nj; nj++) {
        for (int dni = 0; dni < denNi;  dni++) {
            int nidx = nj * denNi + dni;
            silsum += (Cji[nidx] == SILENT) * nMIji[nidx];
            actsum += (Cji[nidx] == ACTIVE) * nMIji[nidx];
        }
    }
    silsum /= nsilNi;
    actsum /= nactNi;
    if (filename == "") {
        if (libverbosity > 0) {
            if (nsilNi > 0)
                printf("actsum = %6.3f silsum = %6.3f ratio = %6.3f\n", actsum, silsum, actsum / silsum);
            else
                printf("actsum = %6.3f\n", actsum);
        }
    } else {
        FILE *outf = fopen(filename.c_str(), "a");
        fprintf(outf, "%6.3f %6.3f %6.3f\n", actsum, silsum, actsum / silsum);
        fclose(outf);
    }
    return actsum;
}


void Prj1::miscsumx(float& actmiscsum, float& silmiscsum, string filename) {
    if (frozen) return;
    float amiscsum = 0, smiscsum = 0;
    for (int nj = 0; nj < Nj; nj++) {
        for (int dni = 0; dni < denNi;  dni++) {
            int nidx = nj * denNi + dni;
            if (nactNi>0) amiscsum += (Cji[nidx] == ACTIVE) * nMIji[nidx]/nactNi;
            if (nsilNi>0) smiscsum += (Cji[nidx] == SILENT) * nMIji[nidx]/nsilNi;
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

    updMIsc();

    *nswapped = 0;

    for (int nj = 0; nj < Nj; nj++) {

        updIfanout();

        nrmMIsc();

        for (int swapid = 0; swapid < nswap; swapid++) {

            int silmax_dni = -1, actmin_dni = -1;
            float silmax_score = -1e7, actmin_score = 1e7;
        
            for (int dni = 0; dni < denNi; dni++) {
                int nidx = nj * denNi + dni;
                if (Cji[nidx]==ACTIVE) {
                    if (nMIji[nidx] < actmin_score) {
                        actmin_dni = dni;
                        actmin_score = nMIji[nidx] + 1e-6 * gnextfloat(); // To avoid packing index order effects
                    }
                } else if (Cji[nidx]==SILENT) {
                    if (nMIji[nidx] > silmax_score) {
                        silmax_dni = dni;
                        silmax_score = nMIji[nidx];
                    }
                }
            }

            if (silmax_dni==-1 or actmin_dni==-1 or silmax_score < actmin_score) break;

            Cji[nj * denNi + actmin_dni] = SILENT;
            Cji[nj * denNi + silmax_dni] = ACTIVE;
            (*nswapped)++;
            nMIji[silmax_dni] -= 1e7; // Prevents picking same silmax_dni
        }
    }

    gnswapped += *nswapped;

    needsupdbw = true;

}

#else

    // *nswapped = 0;
    // swapconns_cu(Cji, MIji, Iji, Hj, Mj, denHi, Mi,
    //                LOWESTFLT, MAXFLT, eps, swaprthr,
    //                Hifanout, nswapped, &gnswapped,
    //                Pj, Pi, Pji, Bj, Wji);

#endif // CUDASWAPCONNS

#ifndef CUDAREPLCONNS

void Prj1::replconns() {
    if (frozen) return;

    if (nsilNi <= 0 or nactNi + nsilNi == axoNi)
        return;

    updMIsc();

    float nrepleff = 0;
    *nrepled = 0;

    nrepleff = nrepl;
    if (nrepl < 0) {
        nrepleff = knrepl * (*nswapped)/Nj;
        printf("nrepleff = %f\n", nrepleff);
    }

    for (int nj = 0; nj < Nj; nj++) {
       
        for (int replid=0; replid<nrepleff; replid++) {

            int silmin_dni = -1;
            float silscore_min = 1e7, silscore_mean = 0, silscore_std = 0;
        
            for (int dni = 0; dni < denNi; dni++) {
                int nidx = nj*denNi + dni;
                if (Cji[nidx]==SILENT) {
                    silscore_mean += nMIji[nidx];
                    if (nMIji[nidx] < silscore_min) {
                        silmin_dni = dni;
                        silscore_min = nMIji[nidx];
                    }
                }
            }
            if (silmin_dni==-1)
                continue;
            if (silscore_mean < 0) silscore_mean = 0;
            for (int dni = 0; dni < denNi; dni++) {
                int nidx = nj*denNi + dni;
                if (Cji[nidx]==SILENT) {
                    silscore_std += (nMIji[nidx] - silscore_mean) * (nMIji[nidx] - silscore_mean);
                }
            }
            if (nsilNi > 1)
                silscore_std = sqrt(silscore_std/(nsilNi-1));
            else
                silscore_std = 0;
            bool looping = false;

            // printf("nj = %2d silscore_min = %.2e silscore_mean = %.2e silscore_std = %.2e silscore_thr = %.2e\n",
            //        nj, silscore_min, silscore_mean, silscore_std, silscore_mean - replkthr * silscore_std);
            if (silscore_min > silscore_mean - replkthr * silscore_std) break;

            int ni, ntry = 0;
            do {
                ni = gnextint() % axoNi;
                if (++ntry == 100) {
                    looping = true;
                    break;
                }
            } while (not ISABSENT(nj, ni));
            // if (looping) printf("replconns looping! %d ", nj);

            Iji[nj * denNi + silmin_dni] = ni;

            (*nrepled)++;
            nMIji[silmin_dni] += 1e7; // Prevents picking same silmin_dni
        }
    }

    gnrepled += *nrepled;

    needsupdbw = true;

}

#else

    // *nrepled = 0;
    // replconns_cu(Cji, MIji, Iji, rnduints,
    //                   eps,
    //                   Nj, Mj, denNi, axoNi, Mi,
    //                   MAXFLT, replrthr, Ifanout, nrepled, &gnrepled,
    //                   Pj, Pi, Pji, Bj, Wji);

#endif // CUDAREPLCONNS


int Prj1::getnelem(int field) {
    int nelem;
    switch (field) {
        case CJI:
        case IJI:
            nelem = Nj * denNi;
            break;
        case IFANOUT:
            nelem = axoNi;
            break;
        case AXOACT:
            nelem = axoNi;
            break;
        case DENACT:
            nelem = Nj * denNi;
            break;
        case MIJI:
            nelem = Nj * denNi;
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
        case PJIX:
        case WJIX:
        case CJIX:
        case MIJIX:
            nelem = Nj * axoNi;
            break;
        default:
            error("Prj1::getnelem", "No such field: '" + fieldtostring(field) + "'");
    }
    return nelem;
}


int *Prj1::expandfieldi(int field) {
    if (xfieldi_NjNi == nullptr)
        cudaMallocManaged(&xfieldi_NjNi, Nj * axoNi * sizeof(int));
    CUDA_CHECK_ERROR(cudaMemset(xfieldi_NjNi, 0, Nj * axoNi * sizeof(int)));
    for (int nj = 0; nj < Nj; nj++) {
        for (int dni = 0; dni < denNi; dni++) {
            if (field ==  CJIX)
                xfieldi_NjNi[nj*axoNi + Iji[nj*denNi + dni]] = Cji[nj*denNi + dni];
            else
                error("Prj1::expandfieldi", "Illegal field");
        }
    }
    return xfieldi_NjNi;
}


float *Prj1::expandfieldf(int field) {
    if (xfieldf_NjNi == nullptr)
        cudaMallocManaged(&xfieldf_NjNi, Nj * axoNi * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(xfieldf_NjNi, 0, Nj * axoNi * sizeof(float)));
    for (int nj = 0; nj < Nj; nj++) {
        for (int dni = 0; dni < denNi; dni++) {
            int ni = Iji[nj*denNi + dni];
            if (field == DENACTX)
                xfieldf_NjNi[nj*axoNi + ni] = denact[nj*denNi + dni];
            else if (field == ZIX)
                xfieldf_NjNi[nj*axoNi + ni] = Zi[nj*denNi + dni];
            else if (field == EIX)
                xfieldf_NjNi[nj*axoNi + ni] = Ei[nj*denNi + dni];
            else if (field == PIX)
                xfieldf_NjNi[nj*axoNi + ni] = Pi[nj*denNi + dni];
            else if (field == WJIX)
                xfieldf_NjNi[nj*axoNi + ni] = Wji[nj*denNi + dni];
            else if (field == PJIX)
                xfieldf_NjNi[nj*axoNi + ni] = Pji[nj*denNi + dni];
            else if (field == MIJIX)
                xfieldf_NjNi[nj*axoNi + ni] = MIji[nj*denNi + dni];
            else
                error("Prj1::expandfieldf", "Illegal field " + fieldtostring(field));
        }
    }
    return xfieldf_NjNi;
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
        case PJIX:
        case WJIX:
            fielddataf = expandfieldf(field);
            fwrite(fielddataf, sizeof(int), getnelem(field), outfp);
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

