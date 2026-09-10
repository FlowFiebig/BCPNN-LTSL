/*****************************************************************

  Created: 2023-09-08  Modified: 2024-07-13

  Authors: Anders Lansner, Naresh Ravichandran

  Copyright (c) 2023 Anders Lansner, Naresh Ravichandran

******************************************************************/

#include "LIFPop.h"
#include "LIFPop.cuh"

using namespace std;
using namespace Globals;

///// PLEASE NOTE: Cuda version runs slower than serial code, for some reason. /////
#define CUDAUPDADA
#define CUDAUPDSUP
#define CUDAUPDACT

//************** Class variables and methods **************

LIFPop::LIFPop(int H, int M, Globals::LIF_T liftype, string name) : Pop(H, M, name) {
    this->liftype = liftype;
    setCgL(1e-9, 1e-6);
    setEL(-76e-3);
    setDT(1e-3);
    setVR(-60e-3);
    setVT(-50e-3);
    ispkwid = 1;
    setreft(5e-3);
    setspkwid(1e-3);
    setnfreq(2000);
    setnampl(0);
    cudaMallocManaged(&spkstep, N * sizeof(int));
    CUDA_CHECK_ERROR(cudaMemset(spkstep, 0, N * sizeof(int)));
    cudaMallocManaged(&inp, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(inp, 0, N * sizeof(float)));
    reset();
}


LIFPop::~LIFPop() {
    cudaFree(spkstep);
    spkstep = nullptr;
    cudaFree(inp);
    inp = nullptr;
}


ulong LIFPop::nallocbyte() {
    ulong nbyte = Pop::nallocbyte() / 4;
    return nbyte * 4;
}


void LIFPop::setCgL(float C, float gL, bool verbose) {
    this->C = C;
    this->gL = gL;
    taum = C / gL;
    taumdt = timestep / taum;
    if (verbose)
        printf("taum = %f taumdt = %f\n", taum, taumdt);
}


void LIFPop::settaum(float taum) {
    error("LIFPop::settaum","Not applicable for LIFPop, use setCgL()!");
}

void LIFPop::setEL(float EL) {
    this->EL = EL;
}


void LIFPop::setDT(float DT) {
    this->DT = DT;
}


void LIFPop::setVR(float VR) {
    this->VR = VR;
}


void LIFPop::setVT(float VT) {
    this->VT = VT;
}


void LIFPop::setadgain(float adgain) {
    this->adgain = -adgain /gL * taum * timestep;
}


void LIFPop::setsadgain(float sadgain) {
    this->sadgain = -sadgain/gL * taum * timestep;
}


void LIFPop::setreft(float reft) {
    if (reft < timestep)
        error("LIFPop::setreft","Illegal: refractory time < timestep");
    ireft = int(reft/timestep);
    if (ireft < ispkwid)
        error("LIFPop::setreft","Illegal: refractory time < spike width");
}


void LIFPop::setspkwid(float spkwid) {
    if (spkwid < timestep)
        error("LIFPop::setspkwid","Illegal: spike width < timestep");
    ispkwid = int(spkwid/timestep);
    if (ireft < ispkwid)
        error("LIFPop::setspkwid","Illegal: refractory time < spike width");
}


void LIFPop::setnfreq(float nfreq) {
    this->nfreq = nfreq;
    gsetpoissonmean(nfreq);
}

void LIFPop::setinput(float input) {
    for (int n = 0; n < N; n++)
        this->inp[n] = igain * input;
}

void LIFPop::setinput(float *input) {
    if (input == nullptr)
        for (int n = 0; n < N; n++)
            this->inp [n] = 0;
    else
        for (int n = 0; n < N; n++) {
            this->inp[n] = igain * input[n];
        }
}


void LIFPop::setclamp(float *input, float clampval) {
    // sets high input where input>0 and 0 input for the rest
    for (int n = 0; n < N; n++)
        if (input[n] > 0.5)
            this->inp[n] = clampval;
        else
            this->inp[n] = -clampval;
}


void LIFPop::reset(bool resetaxodelbuf) {
    Pop::reset(resetaxodelbuf);
    for (int n = 0; n < N; n++) {
        inp[n] = 0;
        supinf[n] = EL;
        sup[n] = supinf[n];
        spkstep[n] = 0;
    }
}


#ifndef CUDAUPDADA

void LIFPop::updada() {
    switch (liftype) {
        case ADEXS:
        case ALIF:
            if (ada != nullptr)
                for (int n = 0; n < N; n++)
                    ada[n] += (adgain * act[n] - ada[n]) * tauadt;
            if (sada != nullptr)
                for (int n = 0; n < N; n++)
                    sada[n] += (sadgain * act[n] - sada[n]) * tausadt;
            break;
        case ADEX:
            if (ada != nullptr)
                for (int n = 0; n < N; n++)
                    ada[n] += (adgain * (sup[n] - EL) - ada[n]) * tauadt;
            if (sada != nullptr)
                for (int n = 0; n < N; n++)
                    sada[n] += (sadgain * (sup[n] - EL) - sada[n]) * tausadt;
            break;
    }
}

#else

void LIFPop::updada() {
    if (ada != nullptr)
        updada_cu(N, liftype, ada, sup, act, adgain, tauadt, EL);
}

#endif // CUDAUPDADA

#ifndef CUDAUPDSUP

void LIFPop::updsup() {
    updada();
    float tmp = 0, pnoise;
    for (int n = 0; n < N; n++) {
        switch (liftype) {
            case ADEX:
            case ADEXS:
                tmp = DT * exp((sup[n] - VT) / DT);
                if (tmp > 1e-10)
                    tmp = C / timestep * 0.1;
            case ALIF:
                supinf[n] = bwgain * bwsup[n];
                if (ada != nullptr)
                    supinf[n] += ada[n];
                if (sada != nullptr)
                    supinf[n] += sada[n];
                pnoise = (nampl * (gnextpoisson() - nfreq)) / gL;
                supinf[n] += pnoise;
                supinf[n] += inp[n] / gL + EL + tmp;
                // This lumps together excitatory and inhibitory conductances so not entirely correct!
                if (supinf[n] > 0.020)
                    supinf[n] = 0.020;  // Na reversal potential, synapses are current based
                else if (supinf[n] < -0.080)
                    supinf[n] = -0.080; // Cl reversal potential, synapses are current based
                sup[n] += (supinf[n] - sup[n]) * taumdt;
                break;
        }
    }
}

#else

void LIFPop::updsup() {
    updada();
    updsup_cu(N, liftype, inp, supinf, sup, bwsup, ada, sada, pnoise, taum, taumdt, igain,
              bwgain, nampl, nfreq, DT, VT, C, gL, EL, timestep);
}

#endif // CUDAUPDSUP

#ifndef CUDAUPDACT

void LIFPop::updact() {
    for (int n = 0; n < N; n++) {
        if (spkstep[n] > 0) {
            spkstep[n]--;
            if (act[n] > 0 and spkstep[n] == 0) {
                sup[n] = VR;
                act[n] = 0;
                spkstep[n] = -ireft;
            }
        } else if (spkstep[n] < 0) {
            spkstep[n]++;
        } else if (act[n] == 0 and VT <= sup[n]) {
            sup[n] += 0.050; // Top of spike
            if (sup[n] > 0.02)
                sup[n] = 0.02;  // Na reversal potential
            act[n] = 1;
            spkstep[n] = ispkwid;
        }
    }
}

#else

void LIFPop::updact() {

    updact_cu(N, sup, act, spkstep, ireft, ispkwid, VR, VT);

}

#endif // CUDAUPDACT


int LIFPop::getnelem(int field) {
    if (field == INP)
        return N;
    else
        return Pop::getnelem(field);
}


float *LIFPop::getfieldf(int field) {
    switch(field) {
        case INP:
            return inp;
            break;
    }
    return Pop::getfieldf(field);
}
