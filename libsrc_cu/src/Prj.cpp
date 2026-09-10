/*****************************************************************

  Created: 2024-07-10

  Authors: Anders Lansner, Naresh Ravchandran

******************************************************************/

#include "Prjbase.h"
#include "Pop.h"
#include "Prj.h"
#include "AxoDelay.h"
#include "Prj.cuh"

#define CUDADENACT
#define CUDAUPDZITRC
#define CUDAUPDTRACES
#define CUDAUPDBW
#define CUDAUPDBWSUP
#define CUDAUPDMISC
#define CUDANRMMISC
// #define CUDASWAPCONNS
// #define CUDAREPLCONNS
#define FAST_CPU_SWAPCONNS

using namespace std;
using namespace Globals;

void Prj::initialize() {
    selfc = HDON;
    lrule = BCP;
    axoNi = axoHi * Mi;
    denHi = nactHi + nsilHi;
    denNi = denHi * Mi;
    if (libverbosity > 1)
        printf("name = %s denHi = %d Mi = %d, denNi = %d\n", name.c_str(), denHi, Mi, denNi);
    if (axoHi < nactHi + nsilHi)
        error("Prj::initialize", "Illegal axoHi < nactHi + nsilHi");
    Nj = Hj * Mj;

    fprintf(stderr, "Hj = %d nactHi = %d nsilHi = %d denHi = %d Mi = %d denNi = %d\n",
            Hj, nactHi, nsilHi, denHi, Mi, denNi);

    tauzidt = 1;
    tauzjdt = 1;
    tauedt = 0;
    taupdt = 1;
    eps = EPS;
    bdebias = false;
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
    knrepl = 1;
    nswap = 0;
    gnswapped = 0;
    gnrepled = 0;
    frozen = false;
    hilist = nullptr;
    hilistoffs = nullptr;
    allocmem();
    initconns(nactHi, nsilHi);
    xfieldi_HjHi = nullptr;
    xfieldf_HjHi = nullptr;
    xfieldf_NjNi = nullptr;
    xfieldf_HjNi = nullptr;
}


Prj::Prj(Pop *srcpop, Pop *trgpop, int nactHi, int nsilHi, std::string name, bool doinit) :
    Prjbase(srcpop, trgpop, nactHi, nsilHi, name) {
    axoHi = srcpop->H; 
    denHi = nactHi + nsilHi;
    this->nactHi = nactQi;
    this->nsilHi = nsilQi;

    Mi = srcpop->M;
    srcN = srcpop->N; 
    axoNi = srcN;
    Mj = trgpop->M;
    Hj = trgpop->H;
    if ((doinit and recurrent and axoHi - 1 < nactHi + nsilHi) or
        (doinit and axoHi < nactHi + nsilHi)) {
        error("Prj::Prj", "Illegal axoHi < nactHi + nsilHi (" + name + ")");
    }
    if (axoHi < nactHi + nsilHi) {
        printf("axoHi = %d nactHi = %d nsilHi = %d\n", axoHi, nactHi, nsilHi);
        error("Prj::Prj", "Illegal axoHi < nactHi + nsilHi (" + name + ")");
    }
    srcpopact = srcpop->act;
    trgpopact = trgpop->act;
    trgpopbwsup = trgpop->bwsup;
    if (doinit) {
        initialize();
        if (libverbosity > 1)
            printf("name = %s axoHi = %d Mi = %d nactHi = %d nsilHi = %d, denHi = %d\n",
                   name.c_str(), axoHi, Mi, nactHi, nsilHi, denHi);
    }
}


Prj::Prj(Pop *srcpop, Pop *trgpop, std::string name, bool doinit) :
    Prjbase(srcpop, trgpop, srcpop->H, 0, name) {
    axoHi = srcpop->H; 
    denHi = axoHi;
    nactHi = nactQi;
    nsilHi = nsilQi;

    Mi = srcpop->M;
    srcN = srcpop->N; 
    axoNi = srcN;
    Mj = trgpop->M;
    Hj = trgpop->H;
    srcpopact = srcpop->act;
    trgpopact = trgpop->act;
    trgpopbwsup = trgpop->bwsup;
    if (doinit)
        initialize();
}


Prj::~Prj() {
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
    cudaFree(MIhjhi);
    MIhjhi = nullptr;
    cudaFree(nMIhjhi);
    nMIhjhi = nullptr;
    cudaFree(Chjhi);
    Chjhi = nullptr;
    cudaFree(Hihjhi);
    Hihjhi = nullptr;
    cudaFree(Hifanout);
    Hifanout = nullptr;
    cudaFree(rnduints);
    rnduints = nullptr;
    cudaFree(xfieldi_HjHi);
    xfieldi_HjHi = nullptr;
    cudaFree(xfieldf_HjHi);
    xfieldf_HjHi = nullptr;
    cudaFree(xfieldf_NjNi);
    xfieldf_NjNi = nullptr;
    cudaFree(xfieldf_HjNi);
    xfieldf_HjNi = nullptr;
    delete axo;
}


void Prj::allocmem() {
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
        CUDA_CHECK_ERROR(cudaMallocManaged(&Ej, Nj * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&Eji, Nj * denNi * sizeof(float)));
    }
    CUDA_CHECK_ERROR(cudaMallocManaged(&Chjhi, Hj * denHi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&Hihjhi, Hj * denHi * sizeof(int)));
    MIhjhi = nullptr;
    nMIhjhi = nullptr;
    if (nactHi < axoHi) {
        CUDA_CHECK_ERROR(cudaMallocManaged(&MIhjhi, Hj * denHi * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&nMIhjhi, Hj * denHi * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&Hifanout, axoHi * sizeof(int)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&rnduints, Hj * axoHi * sizeof(uint)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&spmask, Hj * denHi * sizeof(bool)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&nswapped, sizeof(int)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&nrepled, sizeof(int)));
    }
    CUDA_CHECK_ERROR(cudaMallocManaged(&denact, Hj * denNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&axoact, axoNi * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&bwsupinf, Nj * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&bwsup, Nj * sizeof(float)));
    reinitialize();
}


void Prj::reinitialize() {
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


void Prj::reset() {
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


ulong Prj::nallocbyte() {
    ulong nbyte = 0;
    nbyte += 2 * Hj * denNi; // Zi, Pi
    nbyte += 3 * Nj; // Zj. Pj, Bj
    nbyte += 2 * Nj * denNi; // Pji, Wji
    if (tauedt > 0)
        nbyte += Hj * denNi + Nj + Nj * denNi;
    nbyte += Hj * denNi; // denact
    nbyte += axoNi; // axoact
    nbyte += Nj; // bwsupinf
    nbyte += Nj; // bwsup
    nbyte += axoHi; // Hifanout
    // Structural plasticity!
    nbyte += 5 * Hj * denHi; // Chjhi, Hihjhi, MIhjhi, nMIhjhi, spmask
    nbyte += 2 * axoHi; // Hifanout, rnduints
    nbyte *= 4;
    return nbyte;
}


bool Prj::ISABSENT(int hj, int hi) {
    int dhi, dhifound = false;
    for (dhi = 0; dhi < denHi; dhi++) {
        if (Hihjhi[hj * denHi + dhi] == hi) {
            dhifound = true;
            break;
        }
    }
    return not dhifound;
}


void Prj::setWjix(float *Wjix) {
    // NOTE: Wji must be a full Wji, i.e. have shape Nj*axoNi
    for (int nj = 0, hj; nj < Nj; nj++) {
        hj = nj / Mj;
        for (int dni = 0, dhi, hi, ni; dni < denNi; dni++) {
            dhi = dni / Mi;
            hi = Hihjhi[hj * denHi + dhi];
            ni = hi * Mi + dni % Mi;
            Wji[nj * denNi + dni] = Wjix[nj * axoNi + ni];
        }
    }
}


void Prj::fixupselfc(vector<int> shuffled) {
    // selfc in { HDON, HDOFF, HDONL, NOFF }
    for (int hj = 0; hj < Hj; hj++) {
        for (int dhi = 0; dhi < denHi; dhi++) {
            switch (selfc) {
            case HDONL:
                // ACTIVE only allowed on diagonal
                if (nactHi != 1)
                    error("Prj::fixupselfc","Illegal: nactHi != 1");
                // Reset Chjhi an dHihjhi and recreate on diagonal
                Chjhi[hj*denHi + dhi] = ACTIVE;
                Hihjhi[hj*denHi + dhi] = hj;
                break;
            case HDOFF:
                // ACTIVE not allowed on diagonal
                if (Chjhi[hj*denHi + dhi] == ACTIVE and hj == Hihjhi[hj*denHi + dhi]) {
                    printf("hj = %d hi = %d\n",hj, Hihjhi[hj*denHi + dhi]);
                    error("Prj::fixupselfc","Illegal: HDOFF and hj == hi");
                }
                break;
            case HDON:
            default:
                /* No modifications apply */
                break;
            }
        }
    }
}


bool Prj::onhdiag(int hj, int hihjhi) {
    return recurrent and hj == hihjhi;
}


void Prj::initconns(int nactHi, int nsilHi) {
    if (frozen) return;
    if (nactHi + nsilHi + (selfc == HDOFF) > axoHi) {
        printf("nactHi = %d nsilHi = %d selfc == HDOFF = %d axoHi = %d\n",
               nactHi, nsilHi, selfc == HDOFF, axoHi);
        error("initconns", "nactHi + nsilHi + (selfc == HDOFF) > axoHi", 2);
    }
    CUDA_CHECK_ERROR(cudaMemset(Chjhi, 0, Hj * denHi * sizeof(int)));
    for (int n = 0; n < Hj * denHi; n++)
        Chjhi[n] = ABSENT;
    // Create index vector later to be shuffled
    std::vector<int> shuffled(axoHi);
    // Loop over each trg hypercol
    for (int hj = 0; hj < Hj; hj++) {
        int hi = 0;
        for (size_t axohi = 0; axohi < axoHi; axohi++)
            shuffled[axohi] = axohi;
        // Shuffle index for selecting connections
        shuffle(begin(shuffled), end(shuffled), RndGen::grndgen->generator);
        /// Loop and assign SILENT conns, if HDOFF avoid hj == hi
        int dhi = 0, axohi = 0, hidx;
        for (int nsilhi = 0; nsilhi < nsilHi; nsilhi++) {
            do {
                hi = shuffled[axohi%axoHi];
                if (hi < 0 or (selfc == HDOFF and onhdiag(hj, hi))) axohi++;
            } while (hi < 0 or (selfc == HDOFF and onhdiag(hj, hi)));
            hidx = hj * denHi + dhi%denHi;
            Chjhi[hidx] = SILENT;
            Hihjhi[hidx] = hi;
            shuffled[axohi%axoHi] = -1;
            axohi++; dhi++;
        }
        /// Loop and assign ACTIVE conns, if HDOFF avoid hj == hi
        for (int nacthi = 0; nacthi < nactHi; nacthi++) {
            do {
                hi = shuffled[axohi%axoHi];
                if (hi < 0 or (selfc == HDOFF and onhdiag(hj, hi))) axohi++;
                // printf("hj = %d hi = %d nactHi = %d onhdiag = %d axohi = %d\n",
                //        hj, hi, nactHi, onhdiag(hj, hi), axohi);
                if (axoHi < axohi) {
                    error("Prj::initconns","axoHi < axohi");
                }
            } while (hi < 0 or (selfc == HDOFF and onhdiag(hj, hi)));
            hidx = hj * denHi + dhi%denHi;
            // printf("hidx = %d nacthi = %d axohi = %d\n", hidx, nacthi, axohi);
            Chjhi[hidx] = ACTIVE;
            Hihjhi[hidx] = hi;
            shuffled[axohi%axoHi] = -1;
            axohi++; dhi++;
        }
    }

    if (Hj == axoHi) fixupselfc(shuffled);

    updhifanout();
    updbw(true);

    // Create lists for swapconns()
    CUDA_CHECK_ERROR(cudaMallocManaged(&hilist, Hj * denHi * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&hilistoffs, (axoHi + 1) * sizeof(int)));
    int hipos = 0;
    for (int hi = 0; hi < axoHi; hi++) {
        hilistoffs[hi] = hipos;
        for (int hidx = 0; hidx < Hj * denHi; hidx++) {
            if (Hihjhi[hidx] == hi)
                hilist[hipos++] = hidx;
        }
    }
    hilistoffs[axoHi] = hipos;
}


#ifndef CUDADENACT

void Prj::upddenact() {
    if (axo == nullptr) {
        CUDA_CHECK_ERROR(cudaMemcpy(axoact, srcpop->act, axoNi * sizeof(float), cudaMemcpyDeviceToDevice));
    } else
        axo->updstate();
    int denNi = denHi * Mi;
    for (int hj = 0; hj < Hj; hj++) {
        if (Hihjhi == nullptr) {
            CUDA_CHECK_ERROR(cudaMemcpy(&denact[hj * denNi], axoact, denNi * sizeof(float),
                                        cudaMemcpyDeviceToDevice));
        } else {
            for (int dhi = 0; dhi < denHi; dhi++) {
                int hi = Hihjhi[hj * denHi + dhi];
                for (int mi = 0; mi < Mi; mi++)
                    denact[hj * denNi + dhi * Mi + mi] = axoact[hi * Mi + mi];
            }
        }
    }
}

#else

void Prj::upddenact() {
    if (axo == nullptr) {
        CUDA_CHECK_ERROR(cudaMemcpy(axoact, srcpopact, axoNi * sizeof(float), cudaMemcpyDeviceToDevice));
    } else {
        axo->updstate();
    }
    if (Hihjhi == nullptr) {
        for (int hj = 0; hj < Hj; hj++)
            CUDA_CHECK_ERROR(cudaMemcpy(&denact[hj * denNi], axoact, denNi * sizeof(float), cudaMemcpyDeviceToDevice));
    } else {
        upddenact_cu(axoact, Hihjhi, Chjhi, Hj, denHi, Mi, denact);
    }
}

#endif // CUDADENACT

#ifndef CUDAUPDZITRC

void Prj::updzitrc() {
    for (int hj = 0; hj < Hj; hj++) {
        for (int dni = 0; dni < denNi; dni++) {
            int k = hj * denNi + dni;
            Zi[k] += (fgain * denact[k] * (1 - eps) + eps - Zi[k]) * tauzidt;
        }
    }
}

#else

void Prj::updzitrc() {
    updzitrc_cu(denact, Hj, denNi, fgain, eps, tauzidt, Zi);
}

#endif // CUDAUPDZITRC

#ifndef CUDAUPDTRACES

void Prj::updtraces(float *denact, float *trgact, float prn) {

    if (not frozen) {

        float prntaupdt = prn * taupdt;

        for (int nj = 0; nj < Nj; nj++) {
            Zj[nj] += (fgain * trgact[nj] * (1 - eps) + eps - Zj[nj]) * tauzjdt;
            if (prntaupdt > 0)
                Pj[nj] += (Zj[nj] - Pj[nj]) * prntaupdt;
        }

        for (int hj = 0; hj < Hj; hj++) {
            for (int dni = 0; dni < denNi; dni++) {
                int hidx = hj * denNi + dni;
                if (prntaupdt > 0) {
                    Pi[hidx] += (Zi[hidx] - Pi[hidx]) * prntaupdt;
                    for (int nj = hj * Mj; nj < (hj + 1)*Mj; nj++)
                        Pji[nj * denNi + dni] += (Zi[hidx] * Zj[nj] - Pji[nj * denNi + dni]) * prntaupdt;
                }
            }
        }
        if (prn > 0)
            needsupdbw = true;
    }
}

#else

void Prj::updtraces(float *denact, float *trgact, float prn) {
    updtraces_cu(denact, trgpopact, prn, frozen,
                 Hj, Nj, Mj, denNi,
                 fgain, eps, tauzidt, tauzjdt, taupdt,
                 Zj, Zi, Pj, Pi, Pji);
    if (prn > 0)
        needsupdbw = true;
}

#endif // CUDAUPDTRACES

void Prj::updtraces(float prn) {
    updtraces(denact, trgpopact, prn);
}


#ifndef CUDAUPDBW

void Prj::updbw(bool force) {
    if (frozen) return;
    float pi, pj, pji;
    if (not needsupdbw and not force)
        return;
    switch(lrule) {
        case BCP:
            for (int nj = 0, hj; nj < Nj; nj++) {
                pj = Pj[nj];
                hj = nj / Mj;
                if (bdebias) {
                    float p_hat = (pj - eps) / (1.f - eps);
                    if (p_hat < eps * eps)
                        p_hat = eps * eps;
                    Bj[nj] = bgain * log(p_hat);
                } else {
                    Bj[nj] = bgain * log(pj);
                }
                for (int dni = 0, dhi; dni < denNi; dni++) {
                    dhi = dni / Mi;
                    pi = Pi[hj * denNi + dni];
                    pji = Pji[nj * denNi + dni];
                    float wji = (Chjhi[hj*denHi + dhi] == ACTIVE) * log(pji / (pi * pj));
                    wji *= wgain + (wji > 0) * ewgain + (wji < 0) * iwgain;
                    Wji[nj * denNi + dni] = wji;
                }
            }
            break;
        case WILL:
            for (int nj = 0, hj; nj < Nj; nj++) {
                hj = nj / Mj;
                for (int dni = 0, dhi; dni < denNi; dni++) {
                    dhi = dni / Mi;
                    Wji[nj * denNi + dni] =
                        (Chjhi[hj * denHi + dhi] == ACTIVE) * wgain * Pji[nj * denNi + dni] > wgain * eps * eps;
                }
            }
            break;
        case HEBB:
            for (int nj = 0, hj; nj < Nj; nj++) {
                hj = nj / Mj;
                for (int dni = 0, dhi; dni < denNi; dni++) {
                    dhi = dni / Mi;
                    Wji[nj * denNi + dni] =
                        (Chjhi[hj * denHi + dhi] == ACTIVE) * wgain * Pji[nj * denNi + dni];
                }
            }
            break;
        case COV:
            for (int nj = 0, hj; nj < Nj; nj++) {
                hj = nj / Mj;
                for (int dni = 0, dhi; dni < denNi; dni++) {
                    dhi = dni / Mi;
                    pi = Pi[hj * denNi + dni];
                    Wji[nj * denNi + dni] =
                        (Chjhi[hj * denHi + dhi] == ACTIVE) * wgain * (Pji[nj * denNi + dni] - pi * Pj[nj]);
                }
            }
        default:
            error("updbw", "No such lrule", 1);
            break;
    }
    needsupdbw = false;
}

#else

void Prj::updbw(bool force) {
    if (frozen or not needsupdbw and not force)
        return;
    updbw_cu(lrule, Nj, Mj, denHi, denNi, Mi, Chjhi, Pj, Pi, Pji, Bj, Wji, eps, bgain, wgain, ewgain, iwgain, bdebias);
    needsupdbw = false;
}

#endif // CUDAUPDBW


#ifndef CUDAUPDBWSUP

void Prj::updbwsup() {
    CUDA_CHECK_ERROR(cudaMemcpy(bwsupinf, Bj, Nj * sizeof(float), cudaMemcpyDeviceToDevice));
    for (int hj = 0; hj < Hj; hj++) {
        for (int nj = hj * Mj; nj < (hj + 1)*Mj; nj++) {
            for (int dni = 0; dni < denNi; dni++)
                bwsupinf[nj] += Zi[hj * denNi + dni] * Wji[nj * denNi + dni];
        }
    }
    for (int nj = 0; nj < Nj; nj++)
        bwsup[nj] += (bwsupinf[nj] - bwsup[nj]) * tauzidt;
}

#else

void Prj::updbwsup() {
    CUDA_CHECK_ERROR(cudaMemcpy(bwsupinf, Bj, Nj * sizeof(float), cudaMemcpyDeviceToDevice));
    updbwsup_cu(Zi, Bj, Wji, Hj, Mj, denNi, tauzidt, bwsupinf, bwsup);
}

#endif // CUDAUPDBWSUP


#ifndef CUDACONTRIBUTE

void Prj::contribute() {
    for (int nj = 0; nj < Nj; nj++)
        trgpopbwsup[nj] += bwsup[nj];
}

#else

void Prj::contribute() {
    contribute_cu(bwsup, trgpopbwsup, Nj);
}

#endif // CUDACONTRIBUTE


float Prj::miscsum(string filename) {
    if (frozen) return 0;
    float silsum = 0, actsum = 0;
    for (int hj = 0; hj < Hj; hj++) {
        for (int dhi = 0; dhi < denHi;  dhi++) {
            int hidx = hj * denHi + dhi;
            silsum += (Chjhi[hidx] == SILENT) * MIhjhi[hidx];
            actsum += (Chjhi[hidx] == ACTIVE) * MIhjhi[hidx];
        }
    }
    silsum /= nsilHi;
    actsum /= nactHi;
    if (filename == "") {
        if (libverbosity > 0) {
            if (nsilHi > 0)
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


void Prj::miscsumx(float& actmiscsum, float& silmiscsum, string filename) {
    if (frozen) return;
    float amiscsum = 0, smiscsum = 0;
    for (int hj = 0; hj < Hj; hj++) {
        for (int dhi = 0; dhi < denHi;  dhi++) {
            int hidx = hj * denHi + dhi;
            if (nactHi>0) amiscsum += (Chjhi[hidx] == ACTIVE) * MIhjhi[hidx]/nactHi;
            if (nsilHi>0) smiscsum += (Chjhi[hidx] == SILENT) * MIhjhi[hidx]/nsilHi;
        }
    }
    actmiscsum += amiscsum;
    silmiscsum += smiscsum;
    if (filename == "") {
        if (libverbosity > 0) {
            if (nsilHi > 0)
                printf("amiscsum = %6.3f silmiscsum = %6.3f ratio = %6.3f\n",
                       amiscsum, smiscsum, amiscsum / smiscsum);
            else
                printf("amiscsum = %6.3f\n", amiscsum);
        }
    } else {
        FILE *outf = fopen(filename.c_str(), "a");
        fprintf(outf, "%6.3f %6.3f %6.3f\n", amiscsum, smiscsum, amiscsum / smiscsum);
        fclose(outf);
    }
}


#ifndef CUDAUPDMISC

void Prj::updMIsc() {
    if (frozen) return;
    if (MIhjhi == nullptr)
        error("Prj::updMisc", "updMIsc disabled");
    CUDA_CHECK_ERROR(cudaMemset(MIhjhi, 0, Hj * denHi * sizeof(float)));
    for (int nj = 0; nj < Nj; nj++) {
        int hj = nj / Mj;
        float pj = Pj[nj];
        for (int dni = 0; dni < denNi; dni++) {
            int hi = dni / Mi;
            float pi = Pi[hj * denNi + dni];
            float pji = Pji[nj * denNi + dni];
            MIhjhi[hj * denHi + hi] += pji * log(pji / (pi * pj));
        }
    }
}

#else

void Prj::updMIsc() {
    if (frozen) return;
    if (MIhjhi == nullptr)
        error("Prj::updMIsc", "updMIsc disabled");
    CUDA_CHECK_ERROR(cudaMemset(MIhjhi, 0, Hj * denHi * sizeof(float)));
    updMIsc_cu(Pj, Pi, Pji, eps, Hihjhi, Hifanout, Hj, Nj, Mj, denHi, denNi, Mi, MIhjhi);
    cudaDeviceSynchronize();

    // for (int hj = 0; hj < Hj; hj++) {
    //     for (int dni = 0; dni < denNi; dni++) {
    //         int hi = dni / Mi;
    //         printf("%2d %2d %.2f\n", hj, hi, MIhjhi[hj * denHi + hi]);
    //     }
    //     printf("\n");
    // }
}

#endif // CUDAUPDMISC

#ifndef CUDANRMMISC

void Prj::nrmMIsc() {
    if (frozen) return;
    if (nMIhjhi == nullptr)
        error("Prj::updconnscore", "updconnscore disabled");
    CUDA_CHECK_ERROR(cudaMemset(nMIhjhi, 0, Hj * denHi * sizeof(float)));
    for (int hj = 0; hj < Hj; hj++) {
        for (int dhi = 0; dhi < denHi; dhi++) {
            int hi = Hihjhi[hj * denHi + dhi];
            nMIhjhi[hj * denHi + dhi] = MIhjhi[hj * denHi + dhi] / (1 + Hifanout[hi]);
        }
    }
}

#else

void Prj::nrmMIsc() {
    if (frozen) return;
    if (nMIhjhi == nullptr)
        error("Prj::updconnscore", "updconnscore disabled");
    CUDA_CHECK_ERROR(cudaMemset(nMIhjhi, 0, Hj * denHi * sizeof(float)));

    nrmMIsc_cu(Hihjhi, Hifanout, Hj, Nj, denHi, denNi, MIhjhi, nMIhjhi);
    cudaDeviceSynchronize();
}

#endif // CUDANRMMISC


void Prj::updhifanout() {
    if (not MIhjhi) return;
    CUDA_CHECK_ERROR(cudaMemset(Hifanout, 0, axoHi * sizeof(int)));
    for (int hj = 0; hj < Hj; hj++) {
        for(int dhi = 0, hi; dhi < denHi; dhi++) {
            hi = Hihjhi[hj * denHi + dhi];
            Hifanout[hi] += Chjhi[hj * denHi + dhi] == ACTIVE;
        }
    }
    cudaDeviceSynchronize();
}


#ifndef CUDASWAPCONNS

void Prj::swapconns() {
    if (frozen) return;

    struct timeval swapconns_time;
    gettimeofday(&swapconns_time, 0);

    updMIsc();
    updhifanout();
    nrmMIsc();

    CUDA_CHECK_ERROR(cudaMemset(spmask, 1, Hj * denHi * sizeof(bool)));

    *nswapped = 0;

    for (int hj = 0; hj < Hj; hj++) {

        for (int swapid = 0; swapid < nswap; swapid++) {

            int silmax_dhi = -1, actmin_dhi = -1;
            float silmax_score = -1e7, actmin_score = 1e7;
        
            for (int dhi = 0; dhi < denHi; dhi++) {
                int hidx = hj * denHi + dhi;
                if (Chjhi[hidx]==ACTIVE and spmask[hidx]) {
                    if (nMIhjhi[hidx] < actmin_score) {
                        actmin_dhi = dhi;
                        // Add jitter to avoid packing index order effects
                        actmin_score = nMIhjhi[hidx] + 1e-6 * gnextfloat();
                    }
                }
                if (Chjhi[hidx]==SILENT and spmask[hidx]) {
                    if (nMIhjhi[hidx] > silmax_score) {
                        silmax_dhi = dhi;
                        silmax_score = nMIhjhi[hidx];
                    }
                }
            }
            if (silmax_dhi==-1 or actmin_dhi==-1 or silmax_score < swaprthr * actmin_score) break;
            Chjhi[hj * denHi + actmin_dhi] = SILENT;
            Chjhi[hj * denHi + silmax_dhi] = ACTIVE;
            (*nswapped)++;

#ifdef FAST_CPU_SWAPCONNS
            // Targeted updating Ifanout and nMIhjhi
            int hi = Hihjhi[hj * denHi + actmin_dhi];
            int nextHifanout_hi = Hifanout[hi] - 1;
            float factor = (1.0f + (float)Hifanout[hi]) / (1.0f + (float)nextHifanout_hi);
            for (int i = hilistoffs[hi]; i < hilistoffs[hi+1]; i++)
                nMIhjhi[hilist[i]] *= factor;
            Hifanout[hi] = nextHifanout_hi;

            hi = Hihjhi[hj * denHi + silmax_dhi];
            nextHifanout_hi = Hifanout[hi] + 1;
            factor = (1.0f + (float)Hifanout[hi]) / (1.0f + (float)nextHifanout_hi);
            for (int i = hilistoffs[hi]; i < hilistoffs[hi+1]; i++)
                nMIhjhi[hilist[i]] *= factor;
            Hifanout[hi] = nextHifanout_hi;

            spmask[hj * denHi + actmin_dhi] = false;
            spmask[hj * denHi + silmax_dhi] = false;
#else
            updhifanout();
            nrmMIsc();

#endif // FAST_CPU_SWAPCONNS

        }
    }

    gnswapped += *nswapped;

    needsupdbw = true;

    // float timelapsed = getDiffTime(swapconns_time) / 1000;
    // printf("swapconns_time (CPU) = %.3f sec\n", timelapsed);
}

#else

void Prj::swapconns() {

    *nswapped = swapconns_cu();

    gnswapped += *nswapped;

}

#endif // CUDASWAPCONNS

#ifndef CUDAREPLCONNS

int Prj::nhdiagoff() {
    return (selfc == HDOFF);
}

void Prj::replconns() {
    if (frozen) return;

    struct timeval replconns_time;
    gettimeofday(&replconns_time, 0);

    if (nsilHi <= 0 or nactHi + nsilHi + nhdiagoff() == axoHi)
        return;

    // some entries in spmask set to false by previous call to swapconns
    if (not spmask)
        error("Prj::replconns", "Illegal: spmask missing");

    updMIsc();

    float nrepleff = 0;
    *nrepled = 0;

    nrepleff = nrepl;
    if (nrepl < 0) {
        nrepleff = knrepl * (*nswapped)/Hj;
        // printf("nrepleff = %f\n", nrepleff);
    }

    for (int hj = 0; hj < Hj; hj++) {
        float actmin_score = 1e7;
        int actmin_dhi = -1;
        
        for (int dhi = 0; dhi < denHi; dhi++) {
            int hidx = hj*denHi + dhi;
            if (Chjhi[hidx]==ACTIVE) {
                if (nMIhjhi[hidx] < actmin_score) {
                    actmin_dhi = dhi;
                    actmin_score = nMIhjhi[hidx];
                }
            }
        }

        for (int replid=0; replid<nrepleff; replid++) {

            int silmin_dhi = -1;
            float silmin_score = 1e7;
        
            for (int dhi = 0; dhi < denHi; dhi++) {
                int hidx = hj*denHi + dhi;
                if (Chjhi[hidx]==SILENT and spmask[hidx]) {
                    if (nMIhjhi[hidx] < silmin_score and
                        (selfc == HDON or not onhdiag(hj, Hihjhi[hj * denHi + dhi]))) {
                        silmin_dhi = dhi;
                        silmin_score = nMIhjhi[hidx];
                    }
                }
            }
            
            bool looping = false;
            if (silmin_dhi == -1 or silmin_score > actmin_score) break;
            int hi, ntry = 0;
            float rfpos[2], pos[2];
            do {
                hi = gnextint() % axoHi;
                if (++ntry == 100) {
                    looping = true;
                    break;
                }
            } while (not ISABSENT(hj, hi) or (selfc == HDOFF and onhdiag(hj, hi)));
            if (not looping) {
                Hihjhi[hj * denHi + silmin_dhi] = hi;
                (*nrepled)++;
                spmask[hj * denHi + silmin_dhi] = false;
            } else 
                warning("Prj::replconns", "replconns looping! (" + to_string(Hj) + ")");

        }
    }

    gnrepled += *nrepled;

    needsupdbw = true;

    // float timelapsed = getDiffTime(replconns_time) / 1000;
    // printf("replconns_time (CPU) = %.3f sec\n", timelapsed);
}

#else

    // *nrepled = 0;
    // replconns_cu(Chjhi, MIhjhi, Hihjhi, rnduints,
    //                   eps,
    //                   Hj, Mj, denHi, axoHi, Mi,
    //                   MAXFLT, replrthr, Hifanout, nrepled, &gnrepled,
    //                   Pj, Pi, Pji, Bj, Wji);

#endif // CUDAREPLCONNS


int Prj::getnelem(int field) {
    int nelem;
    switch (field) {
        case CHJHI:
        case HIHJHI:
            nelem = Hj * denHi;
            break;
        case HIFANOUT:
            nelem = axoHi;
            break;
        case AXOACT:
            nelem = axoNi;
            break;
        case DENACT:
            nelem = Hj * denNi;
            break;
        case MIHJHI:
            nelem = Hj * denHi;
            break;
        case CHJHIX:
        case MIHJHIX:
            nelem = Hj * axoHi;
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
            nelem = Hj * axoNi;  // Fields are expanded
            break;
        case EJI:
        case PJI:
        case PJPI:
        case WJI:
            nelem = Nj * denNi;
            break;
        case PJIX:
        case WJIX:
            nelem = Nj * axoNi;
            break;
        default:
            error("Prj::getnelem", "No such field: '" + fieldtostring(field) + "'");
    }
    return nelem;
}


int *Prj::expandfieldi(int field) {
    if (xfieldi_HjHi == nullptr)
        cudaMallocManaged(&xfieldi_HjHi, Hj * axoHi * sizeof(int));
    CUDA_CHECK_ERROR(cudaMemset(xfieldi_HjHi, 0, Hj * axoHi * sizeof(int)));
    for (int hj = 0; hj < Hj; hj++)
        for (int dhi = 0; dhi < denHi; dhi++) {
            if (field ==  CHJHIX) {
                if (Chjhi[hj*denHi + dhi] != ABSENT)
                    xfieldi_HjHi[hj*axoHi + Hihjhi[hj*denHi + dhi]] = Chjhi[hj*denHi + dhi];
            } else
                error("Prj::expandfieldi", "Illegal field");
        }
    return xfieldi_HjHi;
}


float *Prj::expandfieldf(int field) {
    if (field == MIHJHIX) {
        if (MIhjhi == nullptr)
            error("Prj::expandfieldf", "MIhjhi == nullptr");
        if (xfieldf_HjHi == nullptr)
            cudaMallocManaged(&xfieldf_HjHi, Hj * axoHi * sizeof(float));
        CUDA_CHECK_ERROR(cudaMemset(xfieldf_HjHi, 0, Hj * axoHi * sizeof(float)));
        for (int hj = 0; hj < Hj; hj++) {
            for (int dhi = 0; dhi < denHi; dhi++)
                xfieldf_HjHi[hj*axoHi + Hihjhi[hj*denHi + dhi]] = MIhjhi[hj*denHi + dhi];
        }
        return xfieldf_HjHi;
    } else {
        if (xfieldf_NjNi == nullptr)
            cudaMallocManaged(&xfieldf_NjNi, Nj * axoNi * sizeof(float));
        CUDA_CHECK_ERROR(cudaMemset(xfieldf_NjNi, 0, Nj * axoNi * sizeof(float)));
        for (int hj = 0; hj < Hj; hj++) {
            for (int dhi = 0, hi; dhi < denHi; dhi++) {
                if (Chjhi == nullptr or Chjhi[hj*denHi + dhi] != 0) {
                    hi = Hihjhi[hj*denHi + dhi];
                    for (int mj = 0, nj; mj < Mj; mj++) {
                        nj = hj*Mj + mj;
                        for (int mi = 0,ni, dni; mi < Mi; mi++) {
                            ni = hi*Mi + mi;
                            dni = dhi*Mi + mi;
                            if (field == WJIX) {
                                xfieldf_NjNi[nj*axoNi + ni] = Wji[nj*denNi + dni];
                            } else if (field == PJIX) {
                                xfieldf_NjNi[nj*axoNi + ni] = Pji[nj*denNi + dni];
                            } else
                                error("Prj::expandfieldf", "Illegal field");
                        }
                    }
                }
            }
        }
        return xfieldf_NjNi;
    }
    error("Prj::expandfieldf", "Cannot expand field");
    return 0;
}


float *Prj::expandfieldf1(int field) {
    if (xfieldf_HjNi == nullptr)
        cudaMallocManaged(&xfieldf_HjNi, Hj * axoNi * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(xfieldf_HjNi, 0, Hj * axoNi * sizeof(float)));
    for (int hj = 0; hj < Hj; hj++) {
        for (int dhi = 0; dhi < denHi; dhi++) {
            int hi = Hihjhi[hj * denHi + dhi];
            switch (field) {
            case DENACT:
                for (int mi = 0; mi < Mi; mi++)
                    xfieldf_HjNi[hj * axoNi + hi * Mi + mi] = denact[hj * denNi + dhi * Mi + mi];
                break;
            case ZI:
                for (int mi = 0; mi < Mi; mi++)
                    xfieldf_HjNi[hj * axoNi + hi * Mi + mi] = Zi[hj * denNi + dhi * Mi + mi];
                break;
            case EI:
                for (int mi = 0; mi < Mi; mi++)
                    xfieldf_HjNi[hj * axoNi + hi * Mi + mi] = Ei[hj * denNi + dhi * Mi + mi];
                break;
            case PI:
                for (int mi = 0; mi < Mi; mi++) {
                    xfieldf_HjNi[hj * axoNi + hi * Mi + mi] = Pi[hj * denNi + dhi * Mi + mi];
                }
                break;
            default:
                error("expandfieldf1", "No such float field: '" + fieldtostring(field) + "'");
            }
        }
    }
    return xfieldf_HjNi;
}


int *Prj::getfieldi(int field) {
    switch(field) {
        case CHJHI:
            return Chjhi;
        case HIHJHI:
            return Hihjhi;
        case HIFANOUT:
            return Hifanout;
            break;
        default:
            error("getfieldi", "No such int field: '" + fieldtostring(field) + "'");
    }
    return nullptr;
}


float *Prj::getfieldf(int field) {
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
        case ZI:
        case PI:
            return expandfieldf1(field);
            break;
        case EI:
            if (tauedt == 0)
                error("Prj::getfieldf", "E* state not enabled");
            return expandfieldf1(field);
            break;
        case ZJ:
            return Zj;
            break;
        case EJ:
            if (tauedt == 0)
                error("Prj::getfieldf", "E* state not enabled");
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
                error("Prj::getfieldf", "E* state not enabled");
            return Eji;
            break;
        case BJ:
            return Bj;
            break;
        case WJI:
            return Wji;
            break;
        case MIHJHI:
            return MIhjhi;
            break;
        default:
            error("getfieldf", "No such float field: '" + fieldtostring(field) + "'");
    }
    return nullptr;
}

void Prj::prnfield(int field, string filename) {
    FILE *outfp;
    if (filename == "")
        outfp = stderr;
    else
        outfp = fopen(filename.c_str(), "wb");
    int *fielddatai;
    float *fielddataf;
    switch (field) {
        case CHJHI:
        case HIHJHI:
        case HIFANOUT:
            fielddatai = getfieldi(field);
            if (not fielddatai)
                error("Prj::prnfield", "Field " + fieldtostring(field) + " is invalid");
            fwrite(fielddatai, sizeof(float), getnelem(field), outfp);
            break;
        case ZI:
        case EI:
        case PI:
            fwrite(expandfieldf1(field), sizeof(float), getnelem(field), outfp);
            break;
        case AXOACT:
        case DENACT:
        case MIHJHI:
        case BWSUPINF:
        case BWSUP:
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
                error("Prj::prnfield", "Field " + fieldtostring(field) + " is invalid");
            fwrite(fielddataf, sizeof(int), getnelem(field), outfp);
            break;
        case CHJHIX:
            fwrite(expandfieldi(field), sizeof(int), getnelem(field), outfp);
            break;
        case MIHJHIX:
            fwrite(expandfieldf(field), sizeof(int), getnelem(field), outfp);
            break;
        case PJIX:
        case WJIX:
            fielddataf = expandfieldf(field);
            fwrite(fielddataf, sizeof(int), getnelem(field), outfp);
            break;
        default:
            error("Prj::prnfield", "No such field: '" + fieldtostring(field) + "'");
    }
    fclose(outfp);
}

void Prj::prnfield(string field, string filename) {
    prnfield(stringtofield(field), filename);
}

