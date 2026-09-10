/*****************************************************************

  Created: 2023-09-08  Modified: 2024-07-13

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

#include "AxoDelay.h"
#include "Pop.h"
#include "Pop.cuh"

// #define CUDAUPDSUP
#define CUDANORMACT
// #define CUDAUPDACT

using namespace std;
using namespace Globals;

//********** Static variables and methods **********

vector<Pop *> Pop::pops = vector<Pop *>(0);

bool Pop::popnamexists(string name) {
    for (size_t p = 0; p < pops.size(); p++)
        if (pops[p]->name == name)
            return true;
    return false;
}


ulong Pop::nallocbyteall() {
    ulong nbyteall = 0;
    for (size_t p = 0; p < pops.size(); p++)
        nbyteall += pops[p]->nallocbyte();
    return nbyteall;
}


int Pop::npop() {
    return pops.size();
}


void Pop::clear() {
    pops.clear();
}


void Pop::resetall(bool resetaxodelbuf) {
    for (size_t p = 0; p < pops.size(); p++)
        pops[p]->reset(resetaxodelbuf);
}


void Pop::resetbwsupall() {
    for (size_t p = 0; p < pops.size(); p++)
        pops[p]->resetbwsup();
}


void Pop::updsupall() {
    for (size_t p = 0; p < pops.size(); p++)
        pops[p]->updsup();
}


void Pop::updactall() {
    for (size_t p = 0; p < pops.size(); p++)
        pops[p]->updact();
}

//************** Class variables and methods **************

Pop::Pop(int H, int M, string name) {
    if (name != "" and popnamexists(name))
        error("Pop::Pop", "Pop-name '" + name + "' taken");

    this->name = name;
    this->H = H;
    this->M = M;
    N = H * M;

    liftype = NOLIF;
   
    cudaMallocManaged((void**)&lgi, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(lgi, 0, N * sizeof(float)));
    cudaMallocManaged(&bwsup, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(bwsup, 0, N * sizeof(float)));
    cudaMallocManaged(&supinf, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(supinf, 0, N * sizeof(float)));
    cudaMallocManaged(&sup, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(sup, 0, N * sizeof(float)));
    cudaMallocManaged(&act, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(act, 0, N * sizeof(float)));
    cudaMallocManaged(&pnoise, N * sizeof(uint));
    CUDA_CHECK_ERROR(cudaMemset(pnoise, 0, N * sizeof(uint)));
    cudaMallocManaged(&spkthres, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(spkthres, 0, N * sizeof(float)));

    axodelbuf = nullptr;
    ada = nullptr;
    sada = nullptr;
    hfmax = nullptr;
    hfsum = nullptr;
    himax = nullptr;
    nimax = nullptr;
    cumsum = nullptr;

    setactfn("WTA");
    setnormfn("FULLNORM");
    settaum(timestep);
    setagain(1);
    setigain(1);
    setbwgain(1);
    setnampl(0);
    setnfreq(1);
    settaua(0);
    setadgain(0);
    settausa(0);
    setsadgain(0);
    setmaxfq(1 / timestep);
    fgain = 1;
    popdata = nullptr;
    id = pops.size();
    pops.push_back(this);
}


Pop::~Pop() {
    cudaFree(lgi);
    lgi = nullptr;
    cudaFree(bwsup);
    bwsup = nullptr;
    cudaFree(supinf);
    supinf = nullptr;
    cudaFree(sup);
    sup = nullptr;
    cudaFree(act);
    act = nullptr;
    cudaFree(ada);
    ada = nullptr;
    cudaFree(sada);
    sada = nullptr;
    cudaFree(popdata);
    popdata = nullptr;
    cudaFree(pnoise);
    pnoise = nullptr;
    cudaFree(spkthres);
    spkthres = nullptr;
    cudaFree(hfmax);
    hfmax = nullptr;
    cudaFree(hfsum);
    hfsum = nullptr;
    cudaFree(himax);
    himax = nullptr;
    cudaFree(nimax);
    nimax = nullptr;
    cudaFree(cumsum);
    cumsum = nullptr;
    delete axodelbuf;
}


ulong Pop::nallocbyte() {
    ulong nbyte = 6 * N; // lgi, bwsup, supinf, sup, act, spkthres
    if (ada != nullptr)
        nbyte += N; // ada
    if (sada != nullptr)
        nbyte += N; // sada
    if (axodelbuf != nullptr)
        nbyte += axodelbuf->nallocbyte(); // Axodelbuf
    if (hfmax != nullptr)
        nbyte += H;
    if (hfsum != nullptr)
        nbyte += H;
    if (himax != nullptr)
        nbyte += H;
    if (nimax != nullptr)
        nbyte += N;
    return nbyte * 4;
}


void Pop::setnormfn(string normfn) {
    if (normfn == "FULLNORM")
        this->normfn = FULLNORM;
    else if (normfn == "HALFNORM")
        this->normfn = HALFNORM;
    else
        error("Pop::setnormfn", "No such normfn " + normfn);
}


void Pop::setactfn(string actfn) {
    if (actfn == "EXP")
        this->actfn = EXP;
    else if (actfn == "WTA")
        this->actfn = WTA;
    else if (actfn == "SOFTMAX")
        this->actfn = SOFTMAX;
    else if (actfn == "SPK")
        this->actfn = SPK;
    else if (actfn == "STCWTA")
        this->actfn = STCWTA;
    else
        error("Pop::setactfn", "No such actfn " + actfn);
}


void Pop::settaum(float taum) {
    if (taum == 0) {
        this->taumdt = 1;
        return;
    }
    if (taum < timestep)
        error("Pop::settaum", "Illegal taum<taumdt");
    this->taumdt = timestep / taum;
}


void Pop::setagain(float again) {
    if (again < 0)
        error("Pop::setagain", "Illegal again<0");
    this->again = again;
}


void Pop::setigain(float igain) {
    if (igain < 0)
        error("Pop::setigain", "Illegal igain<0");
    this->igain = igain;
}


void Pop::setbwgain(float bwgain) {
    if (bwgain < 0)
        error("Pop::setbwgain", "Illegal bwgain<0");
    this->bwgain = bwgain;
}


void Pop::setnampl(float nampl) {
    if (nampl < 0)
        error("Pop::setnampl", "Illegal nampl<0");
    this->nampl = nampl;
}


void Pop::setnfreq(float nfreq) {
    if (nfreq < 0)
        error("Pop::setnfreq", "Illegal nfreq<0");
    this->nfreq = nfreq * timestep; // Intensity needs to be per timestep not second
}


void Pop::settaua(float taua) {
    if (taua == 0) {
        if (ada != nullptr) {
            delete [] ada;
            ada = nullptr;
        }
        this->tauadt = 0;
        return;
    }
    if (taua < timestep)
        error("Pop::settaua", "Illegal taua<tauadt");
    this->tauadt = timestep / taua;
    cudaMallocManaged(&ada, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(ada, 0, N * sizeof(float)));
}


void Pop::setadgain(float adgain) {
    if (adgain < 0)
        error("Pop::setadgain", "Illegal adgain<0");
    this->adgain = adgain;
}


void Pop::settausa(float tausa) {
    if (tausa == 0) {
        if (sada != nullptr) {
            delete [] sada;
            sada = nullptr;
        }
        this->tausadt = 0;
        return;
    }
    if (tausa < timestep)
        error("Pop::settausa", "Illegal taua<tauadt");
    this->tausadt = timestep / tausa;
    cudaMallocManaged(&sada, N * sizeof(float));
    CUDA_CHECK_ERROR(cudaMemset(sada, 0, N * sizeof(float)));
}


void Pop::setsadgain(float sadgain) {
    if (sadgain < 0)
        error("Pop::setsadgain", "Illegal sadgain<0");
    this->sadgain = sadgain;
}


void Pop::setmaxfq(float maxfq) {
    if (maxfq < 0)
        error("Pop::setmaxfq", "Illegal maxfq<0");
    if (maxfq > 1 / timestep + 0.5)
        error("Pop::setmaxfq", "Illegal maxfq>1/timestep");
    fgain = 1 / (timestep * maxfq);
    this->maxfq = maxfq;
}


void Pop::reset(bool resetaxodelbuf) {
    CUDA_CHECK_ERROR(cudaMemset(lgi, 0, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(bwsup, 0, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(supinf, 0, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(sup, 0, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(act, 0, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(pnoise, 0, N * sizeof(uint)));
    if (ada != nullptr)
        CUDA_CHECK_ERROR(cudaMemset(ada, 0, N * sizeof(float)));
    if (sada != nullptr)
        CUDA_CHECK_ERROR(cudaMemset(sada, 0, N * sizeof(float)));
    if (resetaxodelbuf and axodelbuf != nullptr)
        axodelbuf->reset();
}


void Pop::setinput(float input) {
    for (int n = 0; n < N; n++)
        lgi[n] = log(input + EPS);
}


void Pop::setinput(float *input) {
    if (input == nullptr)
        for (int n = 0; n < N; n++)
            lgi[n] = 0;
    else
        for (int n = 0; n < N; n++)
            lgi[n] = log(input[n] + EPS);
}


void Pop::setlgi(float *lgi) {
    if (lgi == nullptr)
        for (int n = 0; n < N; n++)
            this->lgi[n] = 0;
    else
        for (int n = 0; n < N; n++)
            this->lgi[n] = lgi[n];
}


void Pop::setclamp(float *input, float clampval) {
    // sets high lgi where input>0 and 0 input for the rest
    for (int n = 0; n < N; n++)
        if (input[n] > 0.5)
            lgi[n] = clampval;
        else
            lgi[n] = -clampval;
}


void Pop::unsetclamp() {
    setinput();
}


void Pop::resetbwsup() {
    CUDA_CHECK_ERROR(cudaMemset(bwsup, 0, N * sizeof(float)));
}

#ifndef CUDAUPDSUP

void Pop::updsup() {
    for (int n = 0; n < N; n++)
        supinf[n] = igain * lgi[n] + bwgain * bwsup[n];
    if (ada != nullptr)
        for (int n = 0; n < N; n++) {
            ada[n] += (adgain * act[n] - ada[n]) * tauadt;
            supinf[n] -= ada[n];
        }
    if (sada != nullptr)
        for (int n = 0; n < N; n++) {
            sada[n] += (sadgain * act[n] - sada[n]) * tausadt;
            supinf[n] -= sada[n];
        }
    if (nampl > 0) {
        gsetpoissonmean(nfreq);
        for (int n = 0; n < N; n++)
            supinf[n] += nampl * (gnextpoisson() - nfreq);
    }
    for (int n = 0; n < N; n++)
        sup[n] += (supinf[n] - sup[n]) * taumdt;
}

#else

void Pop::updsup() {
    updsup_cu(N, lgi, bwsup, sup, supinf, act, ada, sada, pnoise,
              taumdt, igain, bwgain, adgain, tauadt, sadgain, tausadt,
              nampl, nfreq);
}

#endif // CUDAUPDSUP

#ifndef CUDANORMACT

void Pop::normact() {
    switch (normfn) {
        case FULLNORM:
            float supmax, actsum;
            for (int h = 0, nmax; h < H; h++) {
                nmax = argmax(sup, h * M, M);
                supmax = sup[nmax];
                actsum = 0;
                for (int n = h * M; n < (h + 1) * M; n++) {
                    act[n] = exp(again * (sup[n] - supmax));
                    actsum += act[n];
                }
                if (actsum > 0) {
                    for (int n = h * M; n < (h + 1) * M; n++)
                        act[n] /= actsum;
                }
            }
            break;
        case HALFNORM:
            float sumexpsup;
            for (int h = 0; h < H; h++) {
                supmax = -1e12;
                sumexpsup = 0;
                for (int n = h * M; n < (h + 1) * M; n++)
                    supmax = max(supmax, again * sup[n]);
                if (supmax > 0)
                    for (int n = h * M; n < (h + 1) * M; n++)
                        act[n] = exp(again * sup[n] - supmax);
                else
                    for (int n = h * M; n < (h + 1) * M; n++)
                        act[n] = exp(again * sup[n]);
                for (int n = h * M; n < (h + 1) * M; n++)
                    sumexpsup += act[n];
                if (sumexpsup > 1)
                    for (int n = h * M; n < (h + 1) * M; n++)
                        act[n] /= sumexpsup;
            }
            break;
    }
}

#else

void Pop::normact() {
    if (hfmax == nullptr)
        cudaMallocManaged(&hfmax, H * sizeof(int));
    if (hfsum == nullptr)
        cudaMallocManaged(&hfsum, H * sizeof(int));
    normact_cu(H, M, sup, act, normfn, again, hfmax, hfsum, LOWESTFLT);
}

#endif // CUDANORMACT

#ifndef CUDAUPDACT

void Pop::updact() {
    int nmax;
    switch(actfn) {
        case EXP:
            for (int n = 0; n < N; n++)
                act[n] = exp(sup[n]);
            break;
        case WTA:
            CUDA_CHECK_ERROR(cudaMemset(act, 0, N * sizeof(float)));
            for (int h = 0; h < H; h++) {
                nmax = argmax(sup, h * M, M);
                act[nmax] = 1;
            }
            break;
        case STCWTA:
            /* draw one spike as winner-takes-all from firing rate */
            if (himax == nullptr)
                cudaMallocManaged(&himax, H * sizeof(int));
            if (cumsum == nullptr)
                cudaMallocManaged(&cumsum, N * sizeof(float));
            for (int h = 0; h < H; h++) {
                float maxsup = sup[M*h], sumexpsup = 0;  
                for (int m=0; m<M; m++) maxsup = max(maxsup, again * sup[M*h+m]); 
                for (int m=0; m<M; m++) act[M*h+m] = exp(again * sup[M*h+m] - maxsup);
                for (int m=0; m<M; m++) sumexpsup += act[M*h+m];  
                for (int m=0; m<M; m++) act[M*h+m] /= sumexpsup;
                for (int m=0; m<M; m++) act[M*h+m] = act[M*h+m] * timestep * maxfq ; // scale to maxfq

                for (int m=0; m<M; m++) {
                    if (m==0) cumsum[M*h+m] = act[M*h+m];
                    else cumsum[M*h+m] = act[M*h+m] + cumsum[M*h+m-1];
                }
                float randn = gnextfloat();
                for (int m=0; m<M; m++) {
                    if (m==0 and randn <= cumsum[M*h+m]) himax[h] = m;
                    if (m!=0 and randn <= cumsum[M*h+m] and randn > cumsum[M*h+m-1]) himax[h] = m;
                }
                for (int m=0; m<M; m++) act[M*h+m] = 1.*(m==himax[h]);
            }
            break;
        case SOFTMAX:
            normact();
            break;
        case SPK:
            normact();
            // Scale to maxfq
            for (int n = 0; n < N; n++)
                act[n] = act[n] * timestep * maxfq ;
            // Generate spike
            for (int n = 0; n < N; n++)
                act[n] = gnextfloat() < act[n];
            break;
    }
    if (axodelbuf != nullptr)
        axodelbuf->updstate();
    for (size_t a = 0; a < axos.size(); a++)
        axos[a]->updstate();
}

#else

void Pop::updact() {
    if (hfmax == nullptr)
        cudaMallocManaged(&hfmax, H * sizeof(float));
    if (hfsum == nullptr)
        cudaMallocManaged(&hfsum, H * sizeof(float));
    if (himax == nullptr)
        cudaMallocManaged(&himax, H * sizeof(int));
    if (nimax == nullptr)
        cudaMallocManaged(&nimax, N * sizeof(int));
    if (cumsum == nullptr)
        cudaMallocManaged(&cumsum, N * sizeof(int));
    updact_cu(H, M, sup, act, normfn, actfn, again, maxfq, timestep, LOWESTFLT,
              spkthres, hfmax, hfsum, himax, nimax, cumsum);
}

#endif // CUDAUPDACT

int Pop::getnelem(int field) {
    if (field == LGI or field == BWSUP or field == BWSUPINF or
            field == SUP or field == ACT or field == ADA or field == SADA)
        return N;
    if (field == ENERGY)
        return 1;
    return 0;
}

float *Pop::getfieldf(int field) {
    switch(field) {
        case LGI:
            return lgi;
            break;
        case BWSUP:
            return bwsup;
            break;
        case BWSUPINF:
            return supinf;
            break;
        case SUP:
            return sup;
            break;
        case ACT:
            return act;
            break;
        case ADA:
            if (ada == nullptr)
                error("Pop::getfieldf", "Illegal: ada==nullptr");
            return ada;
            break;
        case SADA:
            if (sada == nullptr)
                error("Pop::getfieldf", "Illegal: sada==nullptr");
            return sada;
            break;
        case ENERGY:
            if (popdata == nullptr)
                cudaMallocManaged(&popdata, N * sizeof(float));
            for (int n = 0; n < N; n++)
                popdata[n] -= act[n] * sup[n];
            return popdata;
        default:
            error("Pop::getfieldf", "Illegal: field = " + fieldtostring(field));
    }
    return nullptr;
}


void Pop::prnfieldf(int field, string filename) {
    FILE  *outfp = fopen(filename.c_str(), "wb");
    fwrite(getfieldf(field), sizeof(float), getnelem(field), outfp);
    fclose(outfp);
}


void Pop::prnfield(int field, string filename) {
    prnfieldf(field, filename);
}


void Pop::prnfield(string field, string filename) {
    prnfieldf(stringtofield(field), filename);
}
