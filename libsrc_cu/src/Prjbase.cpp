/*****************************************************************

  Created: 2026-01-14

  Authors: Anders Lansner

  Copyright (c) 2026 Anders Lansner

******************************************************************/

#include "Pop.h"
#include "Prjbase.h"
#include "Prj.h"
#include "AxoDelay.h"

using namespace std;
using namespace Globals;

std::vector<Prjbase*> Prjbase::baseprjs;

Prjbase::Prjbase(Pop *srcpop, Pop *trgpop, int nactQi, int nsilQi, std::string name) {
    if (name != "" and prjnamexists(name))
        error("Prjbase::Prjbase", "Prj-name '" + name + "' taken");
    this->name = name;
    if (nactQi <= 0)
        error("Prjbase::Prjbase", "Illegal: nactQi <= 0");
    if (nsilQi < 0)
        error("Prjbase::Prjbase", "Illegal: nsilQi < 0 nsilQi =" + to_string(nsilQi));
    this->nactQi = nactQi;
    this->nsilQi = nsilQi;

    this->srcpop = srcpop;    // Use with care, will not be available in MPI version
    this->trgpop = trgpop;

    id = (int)baseprjs.size();
    recurrent = srcpop != nullptr && trgpop != nullptr && srcpop == trgpop;
    needsupdbw = false;
    frozen = false;
    bdebias = false;
    axo = nullptr;

    baseprjs.push_back(this);
}

bool Prjbase::prjnamexists(string name) {
    for (size_t p = 0; p < Prjbase::baseprjs.size(); p++)
        if (baseprjs[p]->name == name)
            return true;
    return false;
}


void Prjbase::clear() {
    Prjbase::baseprjs.clear();
}


ulong Prjbase::nallocbyteall() {
    ulong nbyteall = 0;
    for (size_t p = 0; p < baseprjs.size(); p++)
        nbyteall += baseprjs[p]->nallocbyte();
    return nbyteall;
}


void Prjbase::resetall() {
    for (size_t p = 0; p < baseprjs.size(); p++)
        baseprjs[p]->reset();
}


void Prjbase::upddenactall() {
    for (size_t p = 0; p < baseprjs.size(); p++)
        baseprjs[p]->upddenact();
}


void Prjbase::updtracesall(vector<float> prns) {
    if (prns.size()==0) {
        for (size_t p = 0; p < baseprjs.size(); p++)
            baseprjs[p]->updtraces();
        return;
    }
    if (prns.size() != baseprjs.size())
        error("Prjbase::updtracesall", "prns.size -- baseprjs.size mismatch");
    for (size_t p = 0; p < baseprjs.size(); p++)
        baseprjs[p]->updtraces(prns[p]);
}


void Prjbase::updbwall(bool force) {
    for (size_t p = 0; p < baseprjs.size(); p++)
        baseprjs[p]->updbw(force);
}


void Prjbase::updbwsupall() {
    for (size_t p = 0; p < baseprjs.size(); p++)
        baseprjs[p]->updbwsup();
}


void Prjbase::contributeall() {
    for (size_t p = 0; p < baseprjs.size(); p++)
        baseprjs[p]->contribute();
}


void Prjbase::updconnsall() {
    for (size_t p = 0; p < baseprjs.size(); p++) {
        baseprjs[p]->swapconns();
        baseprjs[p]->replconns();
        baseprjs[p]->updbw();
    }
}


void Prjbase::putsrcact(float *srcact) {
    for (int i = 0; i < axoNi; i++)
        this->axoact[i] = srcact[i];
}


void Prjbase::setlrule(string lrule) {
    if (lrule == "BCP")
        this->lrule = BCP;
    else if (lrule == "WILL")
        this->lrule = WILL;
    else if (lrule == "HEBB")
        this->lrule = HEBB;
    else if (lrule == "COV")
        this->lrule = COV;
    else
        error("Prjbase::setlrule", "Illegal lrule: " + lrule );
    updbw(true);
}


void Prjbase::seteps(float eps) {
    if (eps < 0)
        error("Prjbase::seteps", "Illegal eps<0");
    this->eps = eps;
    reinitialize();
}


void Prjbase::setbdebias(bool bdebias) {
    this->bdebias = bdebias;
    updbw(true);
}


void Prjbase::settauzi(float tauzi) {
    if (tauzi == 0) {
        this->tauzidt = 1;
        return;
    }
    if (tauzi < timestep)
        error("Prjbase::settauzi", "Illegal tauzi<tauzidt");
    this->tauzidt = timestep / tauzi;
}


void Prjbase::settauzj(float tauzj) {
    if (tauzj == 0) {
        this->tauzjdt = 1;
        return;
    }
    if (tauzj < timestep)
        error("Prjbase::settauzj", "Illegal tauzj<timestep");
    this->tauzjdt = timestep / tauzj;
}


void Prjbase::settaue(float taue) {
    if (taue == 0) {
        this->tauedt = 0;
        return;
    }
    if (taue < timestep)
        error("Prjbase::settaue", "Illegal taue<timestep");
    this->tauedt = timestep / taue;
    if (tauedt > 0) {
        CUDA_CHECK_ERROR(cudaMallocManaged(&Ei, Hj * denNi * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&Ej, Nj * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMallocManaged(&Eji, Nj * denNi * sizeof(float)));
    }
    fill1f(Eji, Nj * denNi, eps * eps);
    fill1f(Ej, Nj, eps);
    fill1f(Ei, Hj * denNi, eps);
}


void Prjbase::settaup(float taup) {
    if (taup == 0) {
        this->taupdt = 1;
        return;
    }
    if (taup < timestep)
        error("Prjbase::settaup", "Illegal taup<timestep");
    this->taupdt = timestep / taup;
}


void Prjbase::setVrev(float Vrev) {
    this->Vrev = Vrev;
}


void Prjbase::setselfc(string selfc) {
    if (simstep > 0)
        error("Prj", "Illegal: 'selfc' cannot be set when simstep > 0");
    if (selfc == "HDON")
        this->selfc = HDON;
    else if (selfc == "HDOFF")
        this->selfc = HDOFF;
    else
        error("Prjbase::setselfc", "Illegal selfc (" + selfc + ")");
    initconns(nactQi, nsilQi);
}


void Prjbase::pscrambleps(float kN) {
    if (frozen) return;
    float pi = 1.0/Mi, pj = 1.0/Mj, N = kN/(pi * pj);
    gsetpoissonmean(N*pj);
    for (int nj = 0; nj < Nj; nj++)
        Pj[nj] = gnextpoisson()/N + eps;
    gsetpoissonmean(N*pi);
    for (int hj = 0; hj < Hj; hj++)
        for (int dni = 0; dni < denNi; dni++)
            Pi[hj*denNi + dni] = gnextpoisson()/N + eps;
    gsetpoissonmean(N*pj*pi);
    for (int nj = 0; nj < Nj; nj++)
        for (int dni = 0; dni < denNi; dni++)
            Pji[nj*denNi + dni] = gnextpoisson()/N + eps*eps;
    updbw(true);
}


void Prjbase::setrandPji() {
    if (frozen) return;
    for (int nj = 0; nj < Nj; nj++) {
        Pj[nj] = eps * gnextfloat();
        for (int dni = 0; dni < denNi; dni++)
            Pji[nj * denNi + dni] = eps * eps * gnextfloat();
    }
    for (int hj = 0; hj < Hj; hj++)
        for (int dni = 0; dni < denNi; dni++)
            Pi[hj * denNi + dni] = eps * gnextfloat();
    updbw(true);
}


void Prjbase::setBj(float *Bj) {
    for (int nj = 0; nj < Nj; nj++)
        this->Bj[nj] = Bj[nj];
}


void Prjbase::setWji(float *Wji) {
    // NOTE: Wji must have shape Nj*denNi not fullWji
    for  (int nj = 0; nj < Nj; nj++)
        for (int dni = 0; dni < denNi; dni++)
            this->Wji[nj * denNi + dni] = Wji[nj * denNi + dni];
}


void Prjbase::setbgain(float bgain) {
    if (trgpop->liftype != NOLIF and bgain != 0)
        error("Prjbase::setbgain","Illegal: trgpop is LIF type");
    this->bgain = bgain;
    updbw(true);
}

void Prjbase::setwgain(float wgain) {
    this->wgain = wgain;
    updbw(true);
}


void Prjbase::setwgainx(float ewgain, float iwgain) {
    this->ewgain = ewgain;
    if (isnan(iwgain))
        this->iwgain = ewgain;
    else
        this->iwgain = iwgain;
    updbw(true);
}


void Prjbase::setbwgain(float bwgain) {
    setbgain(bwgain);
    setwgain(bwgain);
}


void Prjbase::setnswap(float nswap) {
    if (nswap < 0)
        error("Prjbase::setnswap", "Illegal, nswap<0");
    this->nswap = nswap;
}


void Prjbase::setnrepl(float nrepl) {
    if (nrepl < 0)
        error("Prjbase::setnrepl", "Illegal, nrepl<0");
    this->nrepl = nrepl;
}


void Prjbase::setreplkthr(float replkthr) {
    this->replkthr = replkthr;
}


void Prjbase::setswaprthr(float swaprthr) {
    this->swaprthr = swaprthr;
}


void Prjbase::setknrepl(float knrepl) {
    // Only valid if nrepl < 0
    this->knrepl = knrepl;
}


void Prjbase::setdelays(float delay, float spread) {
    if (axo == nullptr)
        axo = new Axo(srcpop, this);
    axo->setdelays(delay, spread);
}


void Prjbase::setdelays(vector<vector<float> > delaymat) {
    if (axo == nullptr)
        axo = new Axo(srcpop, this);
    axo->setdelays(delaymat);
}


